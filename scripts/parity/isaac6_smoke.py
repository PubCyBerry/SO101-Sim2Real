#!/usr/bin/env python
"""Isaac Sim 6 / Isaac Lab 3 새 경로의 비대화형 import·env·camera smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher


def _emit(payload: dict) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(text + "\n", encoding="utf-8")
    print(text, file=sys.__stdout__, flush=True)


parser = argparse.ArgumentParser()
parser.add_argument("--stage", choices=("import", "environment", "camera"), default="import")
parser.add_argument("--task", default="SimToReal-SO101-PickCube-Isaac6Parity-v0")
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=5)
parser.add_argument(
    "--report",
    type=Path,
    default=Path("outputs/parity/isaac6_smoke.json"),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# AppLauncher가 실제 소비하는 값만 전달한다. 커스텀 smoke 인자를 Kit에 넘기지 않는다.
launcher = AppLauncher(
    {
        "visualizer": args.visualizer or "none",
        "device": args.device,
        "enable_cameras": args.stage == "camera",
        "livestream": 0,
    }
)
simulation_app = launcher.app


def _torch_value(value):
    """Lab 3 ProxyArray는 명시적으로 torch view를 선택한다."""
    return value.torch if hasattr(value, "torch") else value


def main() -> int:
    env = None
    try:
        import gymnasium as gym
        import torch

        import isaaclab
        import isaaclab_physx
        import sim_to_real  # noqa: F401
        import sim_to_real.isaac6  # noqa: F401

        result = {
            "stage": args.stage,
            "isaaclab_version": isaaclab.__version__,
            "isaaclab_source": isaaclab.ISAACLAB_EXT_DIR,
            "isaaclab_physx_source": isaaclab_physx.ISAACLAB_PHYSX_EXT_DIR,
            "task": args.task,
        }
        if args.stage == "import":
            result["status"] = "passed"
            _emit(result)
            return 0

        from isaaclab_tasks.utils import parse_env_cfg

        env_cfg = parse_env_cfg(
            args.task,
            device=args.device,
            num_envs=args.num_envs,
        )
        env_cfg.episode_length_s = max(1.0, args.steps / 30.0 + 1.0)
        if args.stage == "camera" and "Isaac6Parity" not in args.task:
            if "PickCube" in args.task:
                from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import add_pick_cube_cameras

                add_pick_cube_cameras(env_cfg.scene)
            else:
                from sim_to_real.tasks.pick_pen.pick_pen_env_cfg import add_pick_pen_cameras

                add_pick_pen_cameras(env_cfg.scene)

        env = gym.make(args.task, cfg=env_cfg)
        env.reset()
        action = torch.zeros((args.num_envs, 6), device=args.device)
        for _ in range(args.steps):
            env.step(action)

        robot = env.unwrapped.scene["robot"]
        joint_pos = _torch_value(robot.data.joint_pos)
        result["joint_pos_shape"] = list(joint_pos.shape)
        result["joint_pos_dtype"] = str(joint_pos.dtype)

        if args.stage == "camera":
            cameras = {}
            for contract_name, scene_name in (
                ("top", "top_camera"),
                ("wrist", "wrist_camera"),
                ("front", "front_camera"),
            ):
                rgb = _torch_value(env.unwrapped.scene[scene_name].data.output["rgb"])
                cameras[contract_name] = {
                    "shape": list(rgb.shape),
                    "dtype": str(rgb.dtype),
                }
            result["cameras"] = cameras
            result["camera_contract_ok"] = all(
                item["shape"] == [args.num_envs, 480, 640, 3]
                and item["dtype"] == "torch.uint8"
                for item in cameras.values()
            )
        result["status"] = "passed"
        _emit(result)
        return 0
    except Exception as exc:
        _emit(
            {
                "stage": args.stage,
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return 1
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
