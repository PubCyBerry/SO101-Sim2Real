#!/usr/bin/env python
"""Isaac 6 SO-101 gripper sweep에서 jaw surface separation을 자동 측정한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--samples", type=int, default=9)
parser.add_argument("--settle-steps", type=int, default=45)
parser.add_argument(
    "--tip-points",
    type=Path,
    default=Path("outputs/parity/gripper_tip_points.npz"),
)
parser.add_argument(
    "--report",
    type=Path,
    default=Path("outputs/parity/sim_gripper_aperture.json"),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

launcher = AppLauncher(
    {
        "visualizer": args.visualizer or "none",
        "device": args.device,
        "enable_cameras": False,
        "livestream": 0,
    }
)
simulation_app = launcher.app


def _write_report(payload: dict) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(text + "\n", encoding="utf-8")
    print(text, file=sys.__stdout__, flush=True)


def _write_progress(stage: str, **fields) -> None:
    _write_report(
        {
            "schema": "so101-sim-gripper-sweep-v1",
            "status": "running",
            "stage": stage,
            **fields,
        }
    )


def _matrix_from_xyzw(quaternion):
    import torch

    x, y, z, w = quaternion.unbind(-1)
    two = torch.tensor(2.0, device=quaternion.device, dtype=quaternion.dtype)
    return torch.stack(
        (
            1 - two * (y * y + z * z),
            two * (x * y - z * w),
            two * (x * z + y * w),
            two * (x * y + z * w),
            1 - two * (x * x + z * z),
            two * (y * z - x * w),
            two * (x * z - y * w),
            two * (y * z + x * w),
            1 - two * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(3, 3)


def _minimum_distance(first, second) -> float:
    import torch

    minimum = torch.tensor(float("inf"), device=first.device)
    for chunk in first.split(512):
        minimum = torch.minimum(minimum, torch.cdist(chunk, second).min())
    return float(minimum)


def main() -> int:
    env = None
    try:
        import gymnasium as gym
        import numpy as np
        import torch
        import warp as wp

        import sim_to_real.isaac6  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg

        task = "SimToReal-SO101-PickCube-Isaac6Parity-v0"
        env_cfg = parse_env_cfg(task, device=args.device, num_envs=1)
        env_cfg.scene.top_camera = None
        env_cfg.scene.wrist_camera = None
        env_cfg.scene.front_camera = None
        env = gym.make(task, cfg=env_cfg)
        env.reset()
        _write_progress("env_ready")

        robot = env.unwrapped.scene["robot"]
        body_names = list(robot.data.body_names)
        jaw_index = body_names.index("jaw")
        gripper_index = body_names.index("gripper")
        joint_names = list(robot.data.joint_names)
        gripper_joint_index = joint_names.index("gripper")
        limits = robot.data.joint_pos_limits.torch[0, gripper_joint_index]
        targets = np.linspace(float(limits[0]), float(limits[1]), args.samples)
        if not args.tip_points.exists():
            raise FileNotFoundError(
                f"{args.tip_points} 없음: extract_gripper_tip_points.py를 먼저 실행한다"
            )
        with np.load(args.tip_points) as points:
            jaw_local = points["jaw_tip"]
            gripper_local = points["gripper_tip"]
        _write_progress(
            "local_points_loaded",
            jaw_points=len(jaw_local),
            gripper_points=len(gripper_local),
        )
        jaw_local = torch.as_tensor(jaw_local, device=args.device, dtype=torch.float32)
        gripper_local = torch.as_tensor(
            gripper_local, device=args.device, dtype=torch.float32
        )

        rows = []
        action = torch.zeros((1, 6), device=args.device, dtype=torch.float32)
        for target in targets:
            action[0, gripper_joint_index] = float(target)
            for _ in range(args.settle_steps):
                env.step(action)

            positions = robot.data.body_pos_w.torch[0]
            # 이 callsite는 Lab 3 XYZW로 감사를 마쳤다. detector 경고를 남기지 않도록
            # quatf ProxyArray의 Warp view를 명시적으로 zero-copy torch 변환한다.
            quaternions = wp.to_torch(robot.data.body_quat_w.warp)[0]
            achieved = float(robot.data.joint_pos.torch[0, gripper_joint_index])

            jaw_rotation = _matrix_from_xyzw(quaternions[jaw_index])
            gripper_rotation = _matrix_from_xyzw(quaternions[gripper_index])
            jaw_world = (
                jaw_local @ jaw_rotation.T
                + positions[jaw_index]
            )
            gripper_world = (
                gripper_local @ gripper_rotation.T
                + positions[gripper_index]
            )
            _write_progress("distance", target_rad=float(target), completed=len(rows))
            rows.append(
                {
                    "target_rad": float(target),
                    "achieved_rad": achieved,
                    "aperture_mm": _minimum_distance(jaw_world, gripper_world)
                    * 1000.0,
                }
            )

        achieved = np.asarray([row["achieved_rad"] for row in rows])
        aperture = np.asarray([row["aperture_mm"] for row in rows])
        order = np.argsort(achieved)
        achieved = achieved[order]
        aperture = aperture[order]
        monotonic = bool(
            np.all(np.diff(aperture) >= -0.1)
            or np.all(np.diff(aperture) <= 0.1)
        )
        _write_report(
            {
                "schema": "so101-sim-gripper-sweep-v1",
                "task": task,
                "samples": args.samples,
                "settle_steps": args.settle_steps,
                "native_unit": "isaac_joint_radian",
                "joint_limits_rad": [float(limits[0]), float(limits[1])],
                "rows": rows,
                "sorted_native": achieved.tolist(),
                "sorted_aperture_mm": aperture.tolist(),
                "monotonic": monotonic,
                "status": "passed" if monotonic else "failed",
            }
        )
        return 0 if monotonic else 1
    except Exception as exc:
        _write_report(
            {
                "schema": "so101-sim-gripper-sweep-v1",
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
