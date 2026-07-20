"""SkillGen source HDF5를 초기 scene state부터 action replay해 물리 성공률을 검증한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="SimToReal-SO101-PickCube-SkillGen-DRBase-v0")
parser.add_argument("--dataset-file", required=True)
parser.add_argument("--summary-file", required=True)
parser.add_argument("--min-success-rate", type=float, default=0.9)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab_tasks.utils import parse_env_cfg

import sim_to_real  # noqa: F401
from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import remove_pick_cube_cameras


def _strip_visual_events(env_cfg) -> None:
    for event_name in (
        "randomize_lights",
        "randomize_camera_focal",
        "randomize_robot_color",
        "randomize_cube1_material",
    ):
        if hasattr(env_cfg.events, event_name):
            delattr(env_cfg.events, event_name)


def main() -> None:
    dataset_path = Path(args_cli.dataset_file).resolve()
    summary_path = Path(args_cli.summary_file).resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if not 0.0 <= args_cli.min_success_rate <= 1.0:
        raise ValueError("min-success-rate must be in [0, 1]")

    handler = HDF5DatasetFileHandler()
    handler.open(str(dataset_path))
    episode_names = list(handler.get_episode_names())
    if not episode_names:
        raise ValueError(f"no episodes in {dataset_path}")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    success_term = env_cfg.terminations.success
    env_cfg.recorders = {}
    env_cfg.terminations = {}
    remove_pick_cube_cameras(env_cfg)
    _strip_visual_events(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise TypeError(f"expected ManagerBasedRLMimicEnv, got {type(env)}")
    env.reset()

    results = []
    env_id = torch.tensor([0], dtype=torch.int64, device=env.device)
    try:
        # reset_to가 Isaac Lab buffer를 in-place 갱신하므로 inference_mode 대신 no_grad를 쓴다.
        with torch.no_grad():
            for index, episode_name in enumerate(episode_names):
                episode = handler.load_episode(episode_name, env.device)
                env.reset_to(
                    episode.get_initial_state(),
                    env_ids=env_id,
                    seed=int(episode.seed) if episode.seed is not None else None,
                    is_relative=True,
                )
                frame_count = 0
                while True:
                    action = episode.get_next_action()
                    if action is None:
                        break
                    env.step(action.unsqueeze(0))
                    frame_count += 1
                success = bool(success_term.func(env, **success_term.params)[0].item())
                results.append(
                    {
                        "episode_index": index,
                        "episode_name": episode_name,
                        "seed": int(episode.seed) if episode.seed is not None else None,
                        "frames": frame_count,
                        "success": success,
                    }
                )
                print(f"[replay] {index + 1}/{len(episode_names)} success={success}")
    finally:
        env.close()

    successes = sum(int(item["success"]) for item in results)
    success_rate = successes / len(results)
    summary = {
        "status": "complete" if success_rate >= args_cli.min_success_rate else "failed",
        "dataset_file": str(dataset_path),
        "task_id": args_cli.task,
        "episodes": len(results),
        "successes": successes,
        "success_rate": success_rate,
        "minimum_required": args_cli.min_success_rate,
        "results": results,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if success_rate < args_cli.min_success_rate:
        raise RuntimeError(
            f"source replay success rate {success_rate:.1%} < required {args_cli.min_success_rate:.1%}"
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
