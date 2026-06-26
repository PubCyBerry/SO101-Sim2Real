"""Isaac Lab HDF5 dataset → LeRobot v3 변환 (host-only fallback).

**Fallback Converter**: in-container LeRobot v3 recorder 의존성이 없을 때 사용.
HDF5 → LeRobot v3 형식 변환. AppLauncher 부팅이 필요하므로 host 에서만 실행:

  uv run --group isaac python scripts/convert/isaaclab2lerobotv3.py \\
    --task_name SimToReal-SO101-PickCube-v0 \\
    --repo_id my-dataset \\
    --hdf5_root ./datasets

**주의**: 이 스크립트는 end-to-end 테스트되지 않은 best-effort fallback 이다.
in-container recorder(scripts/datagen/record_state_machine.py) 를 우선한다.

Dependencies:
  lerobot==0.4.2
  numpy==1.26.0
  torch>=2.7
  isaaclab>=2.3.2
  (pyproject.toml 의 isaac 그룹 참조)
"""

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

# 로컬 helper 모듈 (leisaac vendor)
from _lerobot_features import (
    LeRobotDatasetCfg,
    build_feature_from_env,
)

# 로컬 utility
from sim_to_real.utils.env_utils import get_task_type

# Parse arguments
parser = argparse.ArgumentParser(description="Isaac Lab HDF5 → LeRobot Dataset v3 변환 (host fallback).")
parser.add_argument("--task_name", type=str, default=None, help="환경 ID (e.g., SimToReal-SO101-PickCube-v0).")
parser.add_argument(
    "--task_type",
    type=str,
    default=None,
    help="teleop device 타입 (keyboard/gamepad/so101leader). 미설정 시 task_name 에서 자동 추론.",
)
parser.add_argument(
    "--repo_id",
    type=str,
    default="so101_sim_pick_cube",
    help="LeRobot Hub 저장소 ID.",
)
parser.add_argument(
    "--fps",
    type=int,
    default=30,
    help="프레임/초.",
)
parser.add_argument(
    "--hdf5_root",
    type=str,
    default="./datasets",
    help="HDF5 파일 루트 디렉터리.",
)
parser.add_argument(
    "--hdf5_files",
    type=str,
    default=None,
    help="HDF5 파일 목록(쉼표 구분). 미설정 시 hdf5_root/dataset.hdf5 사용.",
)
parser.add_argument(
    "--task_description",
    type=str,
    default=None,
    help="작업 설명. 미설정 시 환경 기본값.",
)
parser.add_argument(
    "--push_to_hub",
    action="store_true",
    help="변환 후 Hugging Face Hub 에 업로드.",
)

# Append AppLauncher 인자
AppLauncher.add_app_launcher_args(parser)

# Parse
args_cli = parser.parse_args()

# AppLauncher 기본값
default_args = {
    "headless": True,
    "enable_cameras": True,
}
app_launcher_args = vars(args_cli)
app_launcher_args.update(default_args)

# Launch Isaac Sim
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app


# AppLauncher 부팅 이후의 import (ABI 호환성)
import gymnasium as gym
import torch
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab_tasks.utils import parse_env_cfg


def split_episode(episode: EpisodeData, num_frames: int) -> list[EpisodeData]:
    """에피소드를 frame 단위로 분할."""

    def slice_at_index(data, idx: int):
        """nested 구조에서 idx 번째 프레임 추출."""
        if isinstance(data, dict):
            return {k: slice_at_index(v, idx) for k, v in data.items()}
        if isinstance(data, torch.Tensor):
            safe_idx = idx if idx < data.shape[0] else 0
            return [data[safe_idx]]
        return data

    full_data = episode.data
    sub_episodes: list[EpisodeData] = []
    for idx in range(num_frames):
        sub_episode = EpisodeData()
        sub_episode.data = slice_at_index(full_data, idx)
        sub_episodes.append(sub_episode)

    return sub_episodes


def add_episode(
    dataset: LeRobotDataset,
    episode: EpisodeData,
    env: ManagerBasedRLEnv | DirectRLEnv,
    dataset_cfg: LeRobotDatasetCfg,
    task: str,
) -> bool:
    """성공 에피소드를 dataset 에 추가.

    Args:
        dataset: LeRobotDataset 객체.
        episode: 변환할 EpisodeData.
        env: 환경(frame 메타 생성용).
        dataset_cfg: LeRobotDatasetCfg.
        task: 작업 설명 문자열.

    Returns:
        True if added, False if skipped (프레임 수 < 10).
    """
    all_data = episode.data
    num_frames = all_data["actions"].shape[0]
    if num_frames < 10:
        print(f"Episode {episode.env_id} has {num_frames} frames (< 10), skipped.")
        return False

    episode_list = split_episode(episode, num_frames)

    # 첫 5 프레임 스킵 (안정화)
    for frame_index in tqdm(range(5, num_frames), desc="Processing each frame"):
        frame = env.cfg.build_lerobot_frame(episode_list[frame_index], dataset_cfg)
        if task is not None:
            frame["task"] = task
        dataset.add_frame(frame=frame)

    return True


def convert_isaaclab_to_lerobot():
    """IsaacLab HDF5 → LeRobot v3 변환 메인 로직."""

    # 환경 설정 및 load
    env_cfg = parse_env_cfg(args_cli.task_name, device=args_cli.device, num_envs=1)
    task_type = get_task_type(args_cli.task_name, args_cli.task_type)
    env_cfg.use_teleop_device(task_type)

    env: ManagerBasedRLEnv | DirectRLEnv = gym.make(args_cli.task_name, cfg=env_cfg).unwrapped

    # LeRobot dataset 설정
    dataset_cfg = LeRobotDatasetCfg(
        repo_id=args_cli.repo_id,
        fps=args_cli.fps,
        robot_type=env_cfg.robot_name,
    )
    dataset_cfg.features = build_feature_from_env(env, dataset_cfg)

    # LeRobot dataset 생성
    dataset = LeRobotDataset.create(
        repo_id=dataset_cfg.repo_id,
        fps=dataset_cfg.fps,
        robot_type=dataset_cfg.robot_type,
        features=dataset_cfg.features,
    )

    # HDF5 파일 목록 결정
    if args_cli.hdf5_files is None:
        hdf5_files_list = [os.path.join(args_cli.hdf5_root, "dataset.hdf5")]
    else:
        hdf5_files_list = [
            os.path.join(args_cli.hdf5_root, f.strip()) if not os.path.isabs(f.strip()) else f.strip()
            for f in args_cli.hdf5_files.split(",")
        ]

    # HDF5 → LeRobot 변환
    now_episode_index = 0
    for hdf5_id, hdf5_file in enumerate(hdf5_files_list):
        print(f"[{hdf5_id + 1}/{len(hdf5_files_list)}] Processing: {hdf5_file}")

        dataset_file_handler = HDF5DatasetFileHandler()
        dataset_file_handler.open(hdf5_file)

        episode_names = dataset_file_handler.get_episode_names()
        print(f"Found {len(episode_names)} episodes: {episode_names}")

        for episode_name in tqdm(episode_names, desc="Converting episodes"):
            episode = dataset_file_handler.load_episode(episode_name, device=args_cli.device)

            if not episode.success:
                print(f"Episode {episode_name} not successful, skipped.")
                continue

            valid = add_episode(dataset, episode, env, dataset_cfg, args_cli.task_description)
            if valid:
                now_episode_index += 1
                dataset.save_episode()
                print(f"Episode {now_episode_index} saved successfully.")
            else:
                dataset.clear_episode_buffer()

        dataset_file_handler.close()

    # Finalize dataset
    dataset.finalize()

    if args_cli.push_to_hub:
        print("Pushing dataset to Hugging Face Hub...")
        dataset.push_to_hub()

    print(f"✓ Conversion complete: {now_episode_index} episodes saved to {args_cli.repo_id}")
    env.close()


if __name__ == "__main__":
    convert_isaaclab_to_lerobot()
