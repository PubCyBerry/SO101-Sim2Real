"""SO-101 SkillGen Mimic API/frame/round-trip 계약을 1-env runtime에서 검증한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="SimToReal-SO101-PickCube-SkillGen-DRBase-v0")
parser.add_argument("--report-file", required=True)
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--position-tolerance-m", type=float, default=0.005)
parser.add_argument("--rotation-tolerance-rad", type=float, default=0.05)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.utils import math as math_utils
from isaaclab_tasks.utils import parse_env_cfg

import sim_to_real  # noqa: F401
from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import remove_pick_cube_cameras


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.recorders = {}
    env_cfg.terminations = {}
    remove_pick_cube_cameras(env_cfg)
    for event_name in (
        "randomize_lights",
        "randomize_camera_focal",
        "randomize_robot_color",
        "randomize_cube1_material",
    ):
        if hasattr(env_cfg.events, event_name):
            delattr(env_cfg.events, event_name)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    checks: dict[str, bool] = {}
    max_position_error = 0.0
    max_rotation_error = 0.0
    object_finite_steps = 0
    try:
        env.reset(seed=54000)
        checks["manager_based_rl_mimic_env"] = isinstance(env, ManagerBasedRLMimicEnv)
        if not checks["manager_based_rl_mimic_env"]:
            raise TypeError(type(env))

        gripper_term = env.action_manager.get_term("gripper_action")
        checks["single_gripper_joint"] = list(gripper_term._joint_names) == ["gripper"]
        checks["attached_object_subtask_0"] = (
            env.get_expected_attached_object("so101", 0, env_cfg) is None
        )
        checks["attached_object_subtask_1"] = (
            env.get_expected_attached_object("so101", 1, env_cfg) == "Cube1"
        )
        subtask_contract = [
            {
                "object_ref": cfg.object_ref,
                "term_signal": cfg.subtask_term_signal,
            }
            for cfg in env.cfg.subtask_configs["so101"]
        ]
        checks["subtask_config_matches_hdf5_contract"] = subtask_contract == [
            {"object_ref": "Cube1", "term_signal": "grasp_cube"},
            {"object_ref": "Bowl", "term_signal": "place_cube"},
        ]

        with torch.no_grad():
            for _ in range(args_cli.steps):
                pose = env.get_robot_eef_pose("so101")
                gripper = torch.ones((1,), dtype=torch.float32, device=env.device)
                action = env.target_eef_pose_to_action(
                    {"so101": pose[0]}, {"so101": gripper}, env_id=0
                )
                reconstructed = env.action_to_target_eef_pose(action)["so101"]
                source_pos, source_rot = math_utils.unmake_pose(pose[0])
                target_pos, target_rot = math_utils.unmake_pose(reconstructed)
                position_error = torch.linalg.vector_norm(source_pos - target_pos)
                relative_rot = source_rot.transpose(-1, -2) @ target_rot
                trace = torch.trace(relative_rot)
                rotation_error = torch.acos(torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0))
                max_position_error = max(max_position_error, float(position_error.item()))
                max_rotation_error = max(max_rotation_error, float(rotation_error.item()))

                gripper_roundtrip = env.actions_to_gripper_actions(action)["so101"]
                if not torch.allclose(gripper_roundtrip, gripper):
                    raise RuntimeError("gripper action round-trip mismatch")
                object_poses = env.get_object_poses()
                if set(object_poses) != {"Bowl", "Cube1"}:
                    raise RuntimeError(f"object pose keys mismatch: {set(object_poses)}")
                if not all(torch.isfinite(value).all() for value in object_poses.values()):
                    raise RuntimeError("object pose contains NaN/Inf")
                object_finite_steps += 1
                env.step(action.unsqueeze(0))

        checks["eef_position_roundtrip"] = max_position_error <= args_cli.position_tolerance_m
        checks["eef_rotation_roundtrip"] = max_rotation_error <= args_cli.rotation_tolerance_rad
        checks["object_pose_finite_all_steps"] = object_finite_steps == args_cli.steps
        checks["all_required_apis_callable"] = True
    finally:
        env.close()

    report = {
        "status": "complete" if all(checks.values()) else "failed",
        "task_id": args_cli.task,
        "env_class": type(env).__name__,
        "steps": args_cli.steps,
        "checks": checks,
        "max_eef_position_roundtrip_error_m": max_position_error,
        "position_tolerance_m": args_cli.position_tolerance_m,
        "max_eef_rotation_roundtrip_error_rad": max_rotation_error,
        "rotation_tolerance_rad": args_cli.rotation_tolerance_rad,
        "object_pose_finite_steps": object_finite_steps,
        "subtask_contract": subtask_contract,
    }
    report_path = Path(args_cli.report_file).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "complete":
        raise RuntimeError("SkillGen environment contract validation failed")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
