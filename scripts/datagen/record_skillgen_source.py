"""SO-101 SkillGen source와 자동 annotation HDF5를 함께 생성한다.

Source와 annotated 파일은 서로 다른 경로에 보존한다. 두 파일 모두 native
Isaac Lab HDF5이며, annotated 파일만 cumulative start/term signal을 추가한다.
실패 trial은 어느 파일에도 commit하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from pathlib import Path

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Record automatic SO-101 SkillGen source demonstrations.")
parser.add_argument(
    "--task",
    default="SimToReal-SO101-PickCube-SkillGen-DRBase-v0",
    help="SkillGen 전용 Gym task ID",
)
parser.add_argument("--num-demos", type=int, default=10, help="commit할 성공 source episode 수")
parser.add_argument("--max-attempts", type=int, default=100, help="성공률을 숨기지 않는 전체 attempt 상한")
parser.add_argument("--seed", type=int, default=31000, help="첫 DR seed; attempt마다 1씩 증가")
parser.add_argument("--fps", type=int, default=30, help="policy/data 기록 FPS")
parser.add_argument("--settle-steps", type=int, default=30, help="reset 직후 기록하지 않는 물리 안정화 step")
parser.add_argument("--max-frames", type=int, default=750, help="trial timeout frame")
parser.add_argument(
    "--grasp-offset-world",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    default=(-0.03, -0.01, 0.07),
    help="Cube1 중심 기준 gripper-frame grasp offset[m], world frame",
)
parser.add_argument(
    "--source-file", default="datasets/skillgen/source/source.hdf5", help="annotation 전 source HDF5"
)
parser.add_argument(
    "--annotated-file",
    default="datasets/skillgen/annotated/annotated.hdf5",
    help="SM boundary가 들어간 SkillGen input HDF5",
)
parser.add_argument(
    "--summary-file",
    default="outputs/skillgen/reports/source_generation.json",
    help="attempt/seed/layout/boundary 요약 JSON",
)
parser.add_argument(
    "--evidence-dir",
    default="outputs/skillgen/evidence/source_boundaries",
    help="boundary image 출력 디렉터리",
)
parser.add_argument("--evidence-episodes", type=int, default=3, help="모든 boundary image를 남길 성공 episode 수")
parser.add_argument("--overwrite", action="store_true", help="기존 source/annotated 파일을 명시적으로 덮어씀")
parser.add_argument("--print-phase-trace", action="store_true", help="phase transition physics snapshot 출력")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab_tasks.utils import parse_env_cfg

import sim_to_real  # noqa: F401  Gym task registration
from sim_to_real.data.lerobot_units import CAMERA_SCENE_NAMES, read_camera_rgb_u8
from sim_to_real.datagen.state_machine.skillgen_pick_cube import SkillGenPickCubeStateMachine
from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import remove_pick_cube_cameras


def _env_value(value, env_id: int = 0):
    """scene/observation tree에서 단일 env 값을 선택한다."""
    if isinstance(value, dict):
        return {key: _env_value(item, env_id) for key, item in value.items()}
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"EpisodeData only supports tensor/dict values, got {type(value)}")
    return value if value.ndim == 0 else value[env_id]


def _add(episodes: tuple[EpisodeData, ...], key: str, value) -> None:
    for episode in episodes:
        episode.add(key, value)


def _joint_vectors(env) -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.scene["robot"]
    joint_order = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]
    joint_ids, resolved = robot.find_joints(joint_order, preserve_order=True)
    if resolved != joint_order:
        raise RuntimeError(f"canonical joint resolve mismatch: {resolved} != {joint_order}")

    arm_term = env.action_manager.get_term("arm_action")
    gripper_term = env.action_manager.get_term("gripper_action")
    target = torch.cat([arm_term.applied_joint_targets[0], gripper_term.applied_joint_targets[0]], dim=-1)
    state = robot.data.joint_pos[0, joint_ids]
    return state, target


def _hold_current_pose_open(env) -> torch.Tensor:
    pose = env.get_robot_eef_pose("so101")
    pos, rot = math_utils.unmake_pose(pose)
    quat = math_utils.quat_from_matrix(rot)
    return torch.cat([pos, quat, torch.ones((1, 1), device=env.device)], dim=-1)


def _capture_transition_images(env, previous, current, output) -> None:
    for group in ("start", "term"):
        for name, signal in current[group].items():
            key = f"{name}_{group}"
            now = bool(signal[0, 0].item())
            before = bool(previous[group][name][0, 0].item())
            if now and not before:
                output[key] = {
                    camera: read_camera_rgb_u8(env, scene_name)
                    for camera, scene_name in CAMERA_SCENE_NAMES.items()
                }


def _save_evidence(evidence_dir: Path, episode_index: int, frames) -> list[str]:
    import imageio.v3 as iio

    saved = []
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for boundary, camera_images in frames.items():
        for camera, image in camera_images.items():
            path = evidence_dir / f"demo_{episode_index:04d}_{boundary}_{camera}.png"
            iio.imwrite(path, image)
            saved.append(str(path.resolve()))
    return saved


def _prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists; pass --overwrite explicitly: {path}")
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)


def _physics_snapshot(env, frame: int, phase: str) -> dict:
    eef = env.get_robot_eef_pose("so101")[0, :3, 3]
    cube = env.get_object_poses()["Cube1"][0, :3, 3]
    bowl = env.get_object_poses()["Bowl"][0, :3, 3]
    frame_sensor = env.scene["ee_frame"]
    finger_positions = frame_sensor.data.target_pos_w[0]
    jaw_force = torch.linalg.vector_norm(env.scene["contact_jaw"].data.force_matrix_w[0], dim=-1).max()
    fixed_force = torch.linalg.vector_norm(
        env.scene["contact_gripper"].data.force_matrix_w[0], dim=-1
    ).max()
    state, target = _joint_vectors(env)
    subtask = env.obs_buf["subtask_terms"]
    return {
        "frame": frame,
        "phase": phase,
        "eef_xyz": eef.detach().cpu().tolist(),
        "cube_xyz": cube.detach().cpu().tolist(),
        "bowl_xyz": bowl.detach().cpu().tolist(),
        "gripper_frame_world_xyz": finger_positions[0].detach().cpu().tolist(),
        "jaw_frame_world_xyz": finger_positions[1].detach().cpu().tolist(),
        "cube_world_xyz": env.scene["Cube1"].data.root_pos_w[0].detach().cpu().tolist(),
        "jaw_cube_force_n": float(jaw_force.item()),
        "fixed_cube_force_n": float(fixed_force.item()),
        "joint_state_rad": state.detach().cpu().tolist(),
        "joint_target_rad": target.detach().cpu().tolist(),
        "cube_grasped": bool(subtask["cube_grasped"][0].reshape(-1)[0].item()),
        "cube_in_bowl": bool(subtask["place_cube1"][0].reshape(-1)[0].item()),
    }


def main() -> None:
    if args_cli.num_demos < 1 or args_cli.max_attempts < args_cli.num_demos:
        raise ValueError("num-demos must be >=1 and max-attempts must be >= num-demos")
    if args_cli.evidence_episodes > 0 and not args_cli.enable_cameras:
        raise ValueError("boundary evidence requires --enable_cameras (or set --evidence-episodes 0)")

    source_path = Path(args_cli.source_file).resolve()
    annotated_path = Path(args_cli.annotated_file).resolve()
    if source_path == annotated_path:
        raise ValueError("source-file and annotated-file must be different")
    _prepare_output(source_path, args_cli.overwrite)
    _prepare_output(annotated_path, args_cli.overwrite)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.seed = args_cli.seed
    if not args_cli.enable_cameras:
        remove_pick_cube_cameras(env_cfg)
        # DRBase에는 Replicator 기반 visual/material event도 포함된다. 카메라 없는
        # physics probe에서는 SDGPipeline을 만들 이유가 없고 headless Kit에서 graph
        # 생성 실패를 유발하므로 layout/physics DR만 남긴다.
        for event_name in (
            "randomize_lights",
            "randomize_camera_focal",
            "randomize_robot_color",
            "randomize_cube1_material",
        ):
            if hasattr(env_cfg.events, event_name):
                delattr(env_cfg.events, event_name)
    # Trial 성공/실패는 이 스크립트가 실제 term과 timeout으로 판정한다. 자동 reset은
    # initial-state/action 정합을 깨므로 manager termination을 기록 중에는 끈다.
    success_term = env_cfg.terminations.success
    cube_lost_term = env_cfg.terminations.cube_lost
    env_cfg.terminations = None
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise TypeError(f"expected ManagerBasedRLMimicEnv, got {type(env)}")
    if env.action_manager.total_action_dim != 8:
        raise RuntimeError(f"source action must be 8D, got {env.action_manager.total_action_dim}")
    if abs(float(env.step_dt) - 1.0 / args_cli.fps) > 1e-7:
        raise RuntimeError(f"env step_dt={env.step_dt} does not match requested {args_cli.fps} FPS")

    source_handler = HDF5DatasetFileHandler()
    annotated_handler = HDF5DatasetFileHandler()
    source_handler.create(str(source_path), env_name=args_cli.task)
    annotated_handler.create(str(annotated_path), env_name=args_cli.task)
    common_args = {
        "task": args_cli.task,
        "source_action": "absolute_robot_base_eef_xyz_qwxyz_gripper_binary",
        "fps": args_cli.fps,
        "annotation": "automatic_skillgen_state_machine_phase_and_physics",
    }
    source_handler.add_env_args(common_args)
    annotated_handler.add_env_args({**common_args, "annotated": True})

    attempts = 0
    successes = 0
    failures = {"planning": 0, "ik_tracking": 0, "grasp": 0, "place": 0, "timeout": 0}
    trial_records = []
    evidence_files = []
    started_at = time.time()

    try:
        while simulation_app.is_running() and successes < args_cli.num_demos and attempts < args_cli.max_attempts:
            trial_seed = args_cli.seed + attempts
            attempts += 1
            env.reset(seed=trial_seed)

            # Isaac Lab asset data buffers are reset in-place between trials. inference_mode에서
            # 생성된 buffer는 다음 reset에서 in-place update가 금지되므로 no_grad를 사용한다.
            with torch.no_grad():
                for _ in range(args_cli.settle_steps):
                    env.step(_hold_current_pose_open(env))

                sm = SkillGenPickCubeStateMachine(tuple(args_cli.grasp_offset_world))
                sm.reset(env)
                source_episode = EpisodeData()
                annotated_episode = EpisodeData()
                source_episode.seed = trial_seed
                annotated_episode.seed = trial_seed
                initial_state = _env_value(env.scene.get_state(is_relative=True))
                _add((source_episode, annotated_episode), "initial_state", initial_state)

                cube_start = env.get_object_poses()["Cube1"][0, :3, 3].detach().cpu().tolist()
                bowl_start = env.get_object_poses()["Bowl"][0, :3, 3].detach().cpu().tolist()
                previous_signals = sm.boundary_signals()
                evidence_frames = {}
                phase_trace = [_physics_snapshot(env, 0, sm.phase.name)]

                for frame in range(args_cli.max_frames):
                    if sm.is_done or sm.is_failed:
                        break

                    action = sm.get_action(env)
                    quat_norm = torch.linalg.vector_norm(action[:, 3:7], dim=-1)
                    if not torch.allclose(quat_norm, torch.ones_like(quat_norm), atol=1e-5):
                        raise RuntimeError(f"non-unit source quaternion at attempt {attempts}, frame {frame}")

                    current_signals = sm.boundary_signals()
                    if successes < args_cli.evidence_episodes:
                        _capture_transition_images(env, previous_signals, current_signals, evidence_frames)
                    previous_signals = current_signals

                    object_poses = env.get_object_poses(env_ids=[0])
                    eef_poses = {"so101": env.get_robot_eef_pose("so101", env_ids=[0])}
                    target_poses = env.action_to_target_eef_pose(action)
                    policy_obs = _env_value(env.obs_buf["policy"])
                    scene_state = _env_value(env.scene.get_state(is_relative=True))

                    _add((source_episode, annotated_episode), "actions", action[0])
                    _add((source_episode, annotated_episode), "obs/policy", policy_obs)
                    _add((source_episode, annotated_episode), "states", scene_state)
                    _add((source_episode, annotated_episode), "obs/datagen_info/object_pose", _env_value(object_poses))
                    _add((source_episode, annotated_episode), "obs/datagen_info/eef_pose", _env_value(eef_poses))
                    _add(
                        (source_episode, annotated_episode),
                        "obs/datagen_info/target_eef_pose",
                        _env_value(target_poses),
                    )
                    source_episode.add("timestamp", torch.tensor(frame / args_cli.fps, dtype=torch.float64))
                    annotated_episode.add("timestamp", torch.tensor(frame / args_cli.fps, dtype=torch.float64))
                    source_episode.add("frame_index", torch.tensor(frame, dtype=torch.int64))
                    annotated_episode.add("frame_index", torch.tensor(frame, dtype=torch.int64))

                    annotated_episode.add(
                        "obs/datagen_info/subtask_start_signals", _env_value(current_signals["start"])
                    )
                    annotated_episode.add(
                        "obs/datagen_info/subtask_term_signals", _env_value(current_signals["term"])
                    )

                    phase_before = sm.phase
                    env.step(action)
                    state, target = _joint_vectors(env)
                    _add((source_episode, annotated_episode), "joint_states", state)
                    _add((source_episode, annotated_episode), "joint_targets", target)
                    sm.after_step(env)
                    if sm.phase != phase_before:
                        snapshot = _physics_snapshot(env, frame + 1, sm.phase.name)
                        phase_trace.append(snapshot)
                        if args_cli.print_phase_trace:
                            print("[phase] " + json.dumps(snapshot, separators=(",", ":")))

                    if bool(cube_lost_term.func(env, **cube_lost_term.params)[0].item()):
                        sm._fail("place")
                        break
                else:
                    sm._fail("timeout")

                # 마지막 transition은 다음 action frame이 없을 수 있으므로 capture만 보강한다.
                current_signals = sm.boundary_signals()
                if successes < args_cli.evidence_episodes:
                    _capture_transition_images(env, previous_signals, current_signals, evidence_frames)

                physics_success = bool(success_term.func(env, **success_term.params)[0].item())
                valid_boundaries = all(
                    key in sm.boundary_indices
                    for key in (
                        "grasp_cube_start",
                        "grasp_cube_term",
                        "place_cube_start",
                        "place_cube_term",
                    )
                )
                if valid_boundaries:
                    b = sm.boundary_indices
                    valid_boundaries = (
                        0 <= b["grasp_cube_start"]
                        < b["grasp_cube_term"]
                        < b["place_cube_start"]
                        < b["place_cube_term"]
                        < len(source_episode.data["actions"])
                    )

                success = sm.is_done and physics_success and valid_boundaries
                record = {
                    "attempt": attempts,
                    "seed": trial_seed,
                    "success": success,
                    "failure_category": None if success else (sm.failure_category or "place"),
                    "cube_pose": cube_start,
                    "bowl_pose": bowl_start,
                    "frames": len(source_episode.data["actions"]),
                    "boundaries": dict(sm.boundary_indices),
                    "phase_trace": phase_trace,
                }

                if success:
                    source_episode.success = True
                    annotated_episode.success = True
                    source_episode.pre_export()
                    annotated_episode.pre_export()
                    source_handler.write_episode(source_episode)
                    annotated_handler.write_episode(annotated_episode)
                    source_handler.flush()
                    annotated_handler.flush()
                    if successes < args_cli.evidence_episodes:
                        evidence_files.extend(_save_evidence(Path(args_cli.evidence_dir), successes, evidence_frames))
                    successes += 1
                    print(f"[source] success {successes}/{args_cli.num_demos} attempt={attempts} seed={trial_seed}")
                else:
                    category = record["failure_category"]
                    if category not in failures:
                        category = "ik_tracking"
                        record["failure_category"] = category
                    failures[category] += 1
                    print(f"[source] failed attempt={attempts} seed={trial_seed} category={category}")
                trial_records.append(record)
    finally:
        source_handler.close()
        annotated_handler.close()
        env.close()

    summary = {
        "status": "complete" if successes >= args_cli.num_demos else "failed",
        "task_id": args_cli.task,
        "source_file": str(source_path),
        "annotated_file": str(annotated_path),
        "attempts": attempts,
        "successes": successes,
        "success_rate": successes / attempts if attempts else 0.0,
        "failure_categories": failures,
        "fps": args_cli.fps,
        "grasp_offset_world": list(args_cli.grasp_offset_world),
        "seed_start": args_cli.seed,
        "elapsed_seconds": time.time() - started_at,
        "evidence_files": evidence_files,
        "trials": trial_records,
    }
    summary_path = Path(args_cli.summary_file).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("status", "attempts", "successes", "success_rate")}, indent=2))
    if successes < args_cli.num_demos:
        print(f"[ERROR] recorded {successes}/{args_cli.num_demos} successes before attempt limit")
    simulation_app.close()


if __name__ == "__main__":
    main()
