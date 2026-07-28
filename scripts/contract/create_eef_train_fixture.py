#!/usr/bin/env python3
"""운영 ``lerobot-train`` 경로 검증용 최소 absolute EEF LeRobot v3 fixture 생성.

실제 학습 데이터 대체물이 아니라 다음 통합 계약만 반복 검증하기 위한 도구다.

- LeRobot v3 reader가 읽을 수 있는 완전한 dataset metadata/stats
- canonical ``xyz + Rot6D(rows) + absolute gripper`` 10D schema
- production 검증을 통과하는 horizon-aware relative action stats
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from so101_contract.eef_action_contract import (  # noqa: E402
    CANONICAL_ACTION_NAMES,
    ActionRepresentationConfig,
)
from so101_contract.eef_deployment_contract import sha256_file  # noqa: E402
from so101_contract.eef_kinematics import EEF_KINEMATICS_VERSION  # noqa: E402
from so101_contract.eef_relative_action import (  # noqa: E402
    matrix_to_rot6d_rows,
    relative_actions_to_absolute,
)
from so101_contract.eef_relative_stats import (  # noqa: E402
    RelativeActionSamplingConfig,
    calculate_relative_action_stats,
    write_relative_action_stats_profile,
)


def _rotation_z(angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.asarray(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _state(episode: int, frame: int) -> np.ndarray:
    angle = -0.25 + 0.015 * frame + 0.03 * episode
    pose = np.concatenate(
        [
            np.asarray(
                [
                    0.18 + 0.001 * frame,
                    0.22 + 0.01 * episode,
                    0.16 + 0.002 * np.sin(frame * 0.2),
                ],
                dtype=np.float32,
            ),
            matrix_to_rot6d_rows(_rotation_z(angle)),
        ]
    )
    gripper = np.asarray([20.0 + 60.0 * ((frame % 16) / 15.0)], dtype=np.float32)
    return np.concatenate([pose, gripper]).astype(np.float32)


def _absolute_action(state: np.ndarray, frame: int) -> np.ndarray:
    relative = np.zeros((1, 1, 10), dtype=np.float32)
    relative[0, 0, :3] = np.asarray(
        [0.002, 0.001 * np.sin(frame * 0.3), 0.001],
        dtype=np.float32,
    )
    relative[0, 0, 3:9] = matrix_to_rot6d_rows(_rotation_z(0.01))
    relative[0, 0, 9] = 80.0 if frame % 20 < 10 else 20.0
    return relative_actions_to_absolute(state[None], relative)[0, 0].astype(np.float32)


def create_fixture(
    output_dir: Path,
    *,
    repo_id: str,
    episodes: int,
    frames_per_episode: int,
    horizon: int,
    with_images: bool,
    image_size: int,
) -> dict[str, object]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if frames_per_episode < horizon:
        raise ValueError(
            f"frames_per_episode must be >= horizon: {frames_per_episode} < {horizon}"
        )
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"fixture output must be absent or empty: {output_dir}")

    feature = {
        "dtype": "float32",
        "shape": (10,),
        "names": list(CANONICAL_ACTION_NAMES),
    }
    features = {
        "observation.state": dict(feature),
        "action": dict(feature),
    }
    camera_keys = ("top", "wrist", "front") if with_images else ()
    if not with_images:
        # ACT는 state 외에 image 또는 environment_state 입력을 하나 요구한다.
        features["observation.environment_state"] = {
            "dtype": "float32",
            # LeRobot v3 writer는 길이 1 vector를 scalar로 squeeze하므로
            # 실제 batch shape까지 안정적으로 유지되는 2D fixture를 사용한다.
            "shape": (2,),
            "names": ["fixture_phase", "fixture_episode"],
        }
    for camera_key in camera_keys:
        features[f"observation.images.{camera_key}"] = {
            "dtype": "image",
            "shape": (image_size, image_size, 3),
            "names": ["height", "width", "channel"],
        }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        features=features,
        root=output_dir,
        robot_type="so101_follower",
        use_videos=False,
        image_writer_threads=2 if with_images else 0,
    )
    for episode in range(episodes):
        for frame in range(frames_per_episode):
            state = _state(episode, frame)
            dataset_frame = {
                "observation.state": state,
                "action": _absolute_action(state, frame),
                "task": "EEF-relative CLI integration fixture",
            }
            if not with_images:
                dataset_frame["observation.environment_state"] = np.asarray(
                    [
                        frame / max(frames_per_episode - 1, 1),
                        episode / max(episodes - 1, 1),
                    ],
                    dtype=np.float32,
                )
            for camera_index, camera_key in enumerate(camera_keys):
                image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
                image[..., camera_index] = np.uint8((frame * 11 + episode * 37) % 256)
                image[:, :, (camera_index + 1) % 3] = np.arange(
                    image_size,
                    dtype=np.uint8,
                )[:, None]
                dataset_frame[f"observation.images.{camera_key}"] = image
            dataset.add_frame(dataset_frame)
        dataset.save_episode()
    dataset.finalize()

    info_path = output_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    urdf_path = ROOT / "assets" / "robots" / "urdf" / "so_arm101.urdf"
    robot_yaml_path = ROOT / "assets" / "robots" / "so101.yml"
    info["so101_eef_conversion"] = {
        "base_frame": "base_link",
        "eef_frame": "tcp_grasp",
        "eef_kinematics_version": EEF_KINEMATICS_VERSION,
        "rotation_representation": "rot6d",
        "rotation_format": "xyz+rot6d_rows",
        "gripper_format": "canonical_policy_feature_[0,100]",
        "keep_joints": False,
        "urdf_sha256": sha256_file(urdf_path),
        "robot_yaml_sha256": sha256_file(robot_yaml_path),
    }
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    modality = {
        name: {
            "eef_9d": {"start": 0, "end": 9},
            "gripper_position": {"start": 9, "end": 10},
        }
        for name in ("state", "action")
    }
    (output_dir / "meta" / "modality.json").write_text(
        json.dumps(modality, indent=2) + "\n",
        encoding="utf-8",
    )

    sampling = RelativeActionSamplingConfig(action_delta_indices=tuple(range(horizon)))
    stats_result = calculate_relative_action_stats(
        output_dir,
        sampling,
        config=ActionRepresentationConfig(mode="eef_relative"),
        scratch_dir=ROOT / "scratch",
    )
    stats_path, _ = write_relative_action_stats_profile(output_dir, stats_result)
    return {
        "dataset_root": str(output_dir.resolve()),
        "repo_id": repo_id,
        "episodes": episodes,
        "frames_per_episode": frames_per_episode,
        "horizon": horizon,
        "camera_keys": list(camera_keys),
        "relative_stats": str(stats_path.resolve()),
        "relative_stats_profile_id": stats_result.profile_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/so101_eef_cli_fixture")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--frames-per-episode", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--with-images", action="store_true")
    parser.add_argument("--image-size", type=int, default=32)
    args = parser.parse_args()
    result = create_fixture(
        args.output_dir,
        repo_id=args.repo_id,
        episodes=args.episodes,
        frames_per_episode=args.frames_per_episode,
        horizon=args.horizon,
        with_images=args.with_images,
        image_size=args.image_size,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
