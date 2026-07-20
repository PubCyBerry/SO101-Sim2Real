"""최종 LeRobot 6D action을 초기 HDF5 scene state에서 직접 물리 replay한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="SimToReal-SO101-PickCube-SkillGen-DR-v0")
parser.add_argument("--generated-hdf5", required=True)
parser.add_argument("--lerobot-root", required=True)
parser.add_argument("--summary-file", required=True)
parser.add_argument("--video-dir", required=True)
parser.add_argument("--sample-count", type=int, default=30)
parser.add_argument("--selection-seed", type=int, default=93017)
parser.add_argument("--episode-indices", help="테스트용 comma-separated explicit indices")
parser.add_argument("--hold-steps", type=int, default=15)
parser.add_argument("--min-success-rate", type=float, default=0.9)
parser.add_argument("--video-codec", default="libx264")
parser.add_argument("--video-quality", type=int, default=7)
parser.add_argument("--no-videos", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if not args_cli.no_videos and not args_cli.enable_cameras:
    parser.error("video evidence requires --enable_cameras (or pass --no-videos)")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import h5py
import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq
import torch

from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab_tasks.utils import parse_env_cfg

import sim_to_real  # noqa: F401
from sim_to_real.data.lerobot_units import CAMERA_SCENE_NAMES, from_lerobot_units, read_camera_rgb_u8
from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import remove_pick_cube_cameras
from so101_contract.feature_codec import FPS, SO101_JOINT_ORDER


def _demo_names(data: h5py.Group) -> list[str]:
    return sorted(
        (name for name in data if name.startswith("demo_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )


def _initial_state(group: h5py.Group, device: str) -> dict:
    def load(node: h5py.Group) -> dict:
        result = {}
        for key, value in node.items():
            result[key] = load(value) if isinstance(value, h5py.Group) else torch.as_tensor(value[:], device=device)
        return result

    return load(group["initial_state"])


def _layout_table(path: Path) -> list[dict]:
    layouts: list[dict] = []
    with h5py.File(path, "r") as source:
        data = source["data"]
        for episode_index, name in enumerate(_demo_names(data)):
            demo = data[name]
            cube = np.asarray(demo["initial_state/rigid_object/Cube1/root_pose"][0, :3], dtype=float)
            bowl = np.asarray(demo["initial_state/rigid_object/Bowl/root_pose"][0, :3], dtype=float)
            layouts.append(
                {
                    "episode": episode_index,
                    "cube_xyz": cube.tolist(),
                    "bowl_xyz": bowl.tolist(),
                    "cube_bowl_xy_distance": float(np.linalg.norm(cube[:2] - bowl[:2])),
                }
            )
    return layouts


def _stratified_selection(layouts: list[dict], count: int, seed: int) -> list[dict]:
    if count > len(layouts):
        raise ValueError(f"sample-count {count} exceeds episodes {len(layouts)}")
    xy = np.asarray([item["cube_xyz"][:2] for item in layouts])
    center = np.median(xy, axis=0)
    radial = np.linalg.norm(xy - center, axis=1)
    separation = np.asarray([item["cube_bowl_xy_distance"] for item in layouts])
    rng = np.random.default_rng(seed)
    quotas = {
        "region_center": count // 4,
        "dr_boundary": count // 4,
        "bowl_near": count // 4,
        "fixed_seed_sample": count - 3 * (count // 4),
    }
    candidates = {
        "region_center": np.argsort(radial),
        "dr_boundary": np.argsort(-radial),
        "bowl_near": np.argsort(separation),
        "fixed_seed_sample": rng.permutation(len(layouts)),
    }
    selected: list[dict] = []
    used: set[int] = set()
    for stratum, quota in quotas.items():
        for index in candidates[stratum]:
            episode = int(index)
            if episode in used:
                continue
            used.add(episode)
            selected.append({"episode": episode, "stratum": stratum})
            if sum(item["stratum"] == stratum for item in selected) == quota:
                break
    if len(selected) != count:
        raise RuntimeError(f"could not select {count} unique episodes, got {len(selected)}")
    return selected


def _read_lerobot_actions(root: Path, selected: set[int]) -> tuple[dict[int, np.ndarray], int, int]:
    files = sorted((root / "data").glob("**/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet under {root / 'data'}")
    actions: dict[int, list] = {episode: [] for episode in selected}
    total_frames = 0
    max_episode = -1
    for path in files:
        table = pq.read_table(path, columns=["action", "episode_index", "frame_index"])
        action_values = table["action"].to_pylist()
        episode_values = table["episode_index"].to_pylist()
        frame_values = table["frame_index"].to_pylist()
        total_frames += len(episode_values)
        if episode_values:
            max_episode = max(max_episode, max(episode_values))
        for action, episode, frame in zip(action_values, episode_values, frame_values, strict=True):
            episode = int(episode)
            if episode in actions:
                if int(frame) != len(actions[episode]):
                    raise ValueError(f"episode {episode}: non-contiguous frame_index {frame}")
                actions[episode].append(action)
    result = {episode: np.asarray(values, dtype=np.float32) for episode, values in actions.items()}
    for episode, values in result.items():
        if values.ndim != 2 or values.shape[1] != 6:
            raise ValueError(f"episode {episode}: action shape {values.shape}")
    return result, max_episode + 1, total_frames


def _video_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(
        path,
        fps=FPS,
        codec=args_cli.video_codec,
        quality=args_cli.video_quality,
        macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )


def _capture_composite(env) -> np.ndarray:
    images = [read_camera_rgb_u8(env, CAMERA_SCENE_NAMES[name]) for name in ("top", "wrist", "front")]
    return np.ascontiguousarray(np.concatenate(images, axis=1))


def main() -> None:
    generated_path = Path(args_cli.generated_hdf5).resolve()
    lerobot_root = Path(args_cli.lerobot_root).resolve()
    summary_path = Path(args_cli.summary_file).resolve()
    video_dir = Path(args_cli.video_dir).resolve()
    if not generated_path.is_file() or not lerobot_root.is_dir():
        raise FileNotFoundError(f"missing input: {generated_path} or {lerobot_root}")
    if not 0.0 <= args_cli.min_success_rate <= 1.0:
        raise ValueError("min-success-rate must be in [0,1]")

    layouts = _layout_table(generated_path)
    if args_cli.episode_indices:
        indices = [int(value) for value in args_cli.episode_indices.split(",")]
        selected = [{"episode": episode, "stratum": "explicit"} for episode in indices]
    else:
        selected = _stratified_selection(layouts, args_cli.sample_count, args_cli.selection_seed)
    selected_set = {item["episode"] for item in selected}
    actions_by_episode, lerobot_episode_count, lerobot_frame_count = _read_lerobot_actions(
        lerobot_root, selected_set
    )
    if lerobot_episode_count != len(layouts):
        raise ValueError(
            f"HDF5/LeRobot episode count mismatch: {len(layouts)} != {lerobot_episode_count}"
        )

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    success_term = env_cfg.terminations.success
    env_cfg.recorders = {}
    env_cfg.terminations = {}
    if args_cli.no_videos:
        remove_pick_cube_cameras(env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise TypeError(f"expected ManagerBasedRLMimicEnv, got {type(env)}")
    env.reset()
    robot = env.scene["robot"]
    joint_ids, joint_names = robot.find_joints(list(SO101_JOINT_ORDER), preserve_order=True)
    if joint_names != list(SO101_JOINT_ORDER):
        raise RuntimeError(f"joint order mismatch: {joint_names}")
    env_id = torch.tensor([0], dtype=torch.int64, device=env.device)
    results: list[dict] = []
    layout_by_episode = {item["episode"]: item for item in layouts}

    try:
        with h5py.File(generated_path, "r") as source, torch.no_grad():
            data = source["data"]
            names = _demo_names(data)
            for audit_index, item in enumerate(selected):
                episode_index = item["episode"]
                demo = data[names[episode_index]]
                initial_state = _initial_state(demo, env.device)
                env.reset_to(initial_state, env_ids=env_id, seed=None, is_relative=True)
                action_rad = from_lerobot_units(actions_by_episode[episode_index])
                video_path = video_dir / f"episode_{episode_index:04d}_{item['stratum']}.mp4"
                writer = None if args_cli.no_videos else _video_writer(video_path)
                max_cube_z = float("-inf")
                stable_success_steps = 0
                ever_success = False
                try:
                    for step_index in range(action_rad.shape[0] + args_cli.hold_steps):
                        target_np = action_rad[min(step_index, action_rad.shape[0] - 1)]
                        target = torch.as_tensor(target_np, dtype=torch.float32, device=env.device).unsqueeze(0)
                        robot.set_joint_position_target(target, joint_ids=joint_ids)
                        env.scene.write_data_to_sim()
                        for substep in range(env.cfg.decimation):
                            env.sim.step(render=(substep == env.cfg.decimation - 1))
                            env.scene.update(env.cfg.sim.dt)
                        success_now = bool(success_term.func(env, **success_term.params)[0].item())
                        ever_success = ever_success or success_now
                        stable_success_steps = stable_success_steps + 1 if success_now else 0
                        cube_z = float(env.scene["Cube1"].data.root_pos_w[0, 2].item())
                        max_cube_z = max(max_cube_z, cube_z)
                        if writer is not None:
                            writer.append_data(_capture_composite(env))
                finally:
                    if writer is not None:
                        writer.close()

                final_success = bool(success_term.func(env, **success_term.params)[0].item())
                cube_pos = (
                    env.scene["Cube1"].data.root_pos_w[0] - env.scene.env_origins[0]
                ).detach().cpu().numpy()
                bowl_pos = (
                    env.scene["Bowl"].data.root_pos_w[0] - env.scene.env_origins[0]
                ).detach().cpu().numpy()
                initial_cube_z = float(layout_by_episode[episode_index]["cube_xyz"][2])
                if final_success:
                    failure_category = None
                elif cube_pos[2] < initial_cube_z - 0.10:
                    failure_category = "cube_lost"
                elif max_cube_z - initial_cube_z < 0.04:
                    failure_category = "grasp"
                elif np.linalg.norm(cube_pos[:2] - bowl_pos[:2]) >= 0.05:
                    failure_category = "place_outside_bowl"
                else:
                    failure_category = "unstable_or_height"
                result = {
                    **item,
                    "frames": int(action_rad.shape[0]),
                    "hold_steps": args_cli.hold_steps,
                    "success": final_success,
                    "ever_success": ever_success,
                    "stable_success_steps_at_end": stable_success_steps,
                    "failure_category": failure_category,
                    "final_cube_xyz": cube_pos.tolist(),
                    "final_bowl_xyz": bowl_pos.tolist(),
                    "max_cube_lift_m": max_cube_z - initial_cube_z,
                    "video": None if args_cli.no_videos else str(video_path),
                    "layout": layout_by_episode[episode_index],
                }
                results.append(result)
                print(
                    f"[replay] {audit_index + 1}/{len(selected)} episode={episode_index} "
                    f"stratum={item['stratum']} success={final_success}",
                    flush=True,
                )
    finally:
        env.close()

    successes = sum(int(item["success"]) for item in results)
    success_rate = successes / len(results)
    failure_categories: dict[str, int] = {}
    for item in results:
        if item["failure_category"]:
            key = item["failure_category"]
            failure_categories[key] = failure_categories.get(key, 0) + 1
    report = {
        "status": "complete" if success_rate >= args_cli.min_success_rate else "failed",
        "task_id": args_cli.task,
        "generated_hdf5": str(generated_path),
        "lerobot_root": str(lerobot_root),
        "action_source": "final LeRobot v3 action",
        "action_conversion": "canonical PolicyFeature v1 -> sim joint radians",
        "selection_seed": args_cli.selection_seed,
        "selection_method": "region center / DR radial boundary / Bowl proximity / fixed seeded sample",
        "sample_count": len(results),
        "successes": successes,
        "failures": len(results) - successes,
        "success_rate": success_rate,
        "minimum_required": args_cli.min_success_rate,
        "hdf5_episode_count": len(layouts),
        "lerobot_episode_count": lerobot_episode_count,
        "lerobot_frame_count": lerobot_frame_count,
        "failure_categories": failure_categories,
        "successful_videos": [item["video"] for item in results if item["success"] and item["video"]],
        "failure_videos": [item["video"] for item in results if not item["success"] and item["video"]],
        "results": results,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    if success_rate < args_cli.min_success_rate:
        raise RuntimeError(
            f"final LeRobot replay rate {success_rate:.1%} < required {args_cli.min_success_rate:.1%}"
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
