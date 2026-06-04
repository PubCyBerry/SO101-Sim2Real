"""PickCube rule-based state machine expert.

RL 전에 cube_task 씬이 물리적으로 pick-and-place 가능한지 증명하기 위한
scripted controller다. 큐브/그리퍼를 순간이동하지 않고, leisaac 과 같은
jaw-offset grasp point를 목표 위치에 맞춘 뒤 joint-position action으로 실행한다.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
from pathlib import Path
import shutil
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from isaaclab.app import AppLauncher


if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)


parser = argparse.ArgumentParser(description="PickCube rule-based state machine expert")
parser.add_argument("--task", default="SimToReal-SO101-PickCube-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--active_objects", type=int, default=1, choices=[1, 2, 3, 4])
parser.add_argument("--object_radius_scale", type=float, default=0.0)
parser.add_argument("--container_angle_scale", type=float, default=0.0)
parser.add_argument("--container_radius_scale", type=float, default=1.0)
parser.add_argument("--settle_steps", type=int, default=60)
parser.add_argument("--approach_steps", type=int, default=180)
parser.add_argument("--descend_steps", type=int, default=140)
parser.add_argument("--close_steps", type=int, default=80)
parser.add_argument("--lift_steps", type=int, default=180)
parser.add_argument("--transport_steps", type=int, default=240)
parser.add_argument("--place_steps", type=int, default=160)
parser.add_argument("--open_steps", type=int, default=80)
parser.add_argument("--final_settle_steps", type=int, default=120)
parser.add_argument("--command_settle_steps", type=int, default=200)
parser.add_argument("--max_grasp_attempts", type=int, default=3)
parser.add_argument("--approach_height", type=float, default=0.14)
parser.add_argument("--lift_height", type=float, default=0.18)
parser.add_argument("--transport_height", type=float, default=0.18)
parser.add_argument("--place_height", type=float, default=0.065)
parser.add_argument("--grasp_z_offset", type=float, default=0.002)
parser.add_argument("--target_tolerance", type=float, default=0.018)
parser.add_argument("--ik_damping", type=float, default=0.05)
parser.add_argument("--ik_gain", type=float, default=0.85)
parser.add_argument("--max_joint_delta", type=float, default=0.075)
parser.add_argument(
    "--max_arm_step_delta",
    type=float,
    default=0.01,
    help="Max per-step arm joint command change in radians (0.01 ~= 0.3 rad/s at 30 Hz)",
)
parser.add_argument(
    "--max_gripper_step_delta",
    type=float,
    default=0.005,
    help="Max per-step gripper joint command change in radians (0.005 ~= 0.15 rad/s at 30 Hz)",
)
parser.add_argument("--fk_samples", type=int, default=5000)
parser.add_argument("--continuity_weight", type=float, default=0.015)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--gripper_open", type=float, default=1.0)
parser.add_argument("--gripper_closed", type=float, default=0.0)
parser.add_argument("--output_json", type=Path, default=Path("outputs/pick_cube_state_machine.json"))
parser.add_argument("--dataset_dir", type=Path, default=None, help="Optional LeRobot v3 episode output directory")
parser.add_argument("--expert_dataset_pt", type=Path, default=None, help="Optional raw rl_state/action expert dataset (.pt)")
parser.add_argument("--record_seconds", type=float, default=30.0, help="Seconds to record when --dataset_dir is set")
parser.add_argument("--overwrite_dataset", action="store_true", help="Replace --dataset_dir if it already exists")
parser.add_argument("--no_videos", action="store_true", help="Skip camera videos in the LeRobot dataset")
parser.add_argument("--warmup_steps", type=int, default=5, help="Render warmup steps before recording starts")
parser.add_argument("--gui", action="store_true", help="Open the Isaac GUI instead of forced headless mode")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = not args.gui
if args.livestream < 0:
    args.livestream = 0
args.enable_cameras = args.dataset_dir is not None and not args.no_videos

if args.num_envs != 1:
    raise ValueError("pick_cube_state_machine.py currently supports --num_envs=1 only.")

launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402

if args.dataset_dir is not None and not args.no_videos:
    import imageio.v2 as imageio  # noqa: E402

import sim_to_real  # noqa: E402,F401
from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import (  # noqa: E402
    BOWL_HEIGHT_RANGE,
    BOWL_NAME,
    BOWL_SUCCESS_RADIUS,
    add_pick_cube_cameras,
    CUBE_NAMES,
    apply_curriculum,
)
from sim_to_real.tasks.pick_pen import mdp as task_mdp  # noqa: E402


ARM_DOF = 5
DESK_TOP_Z = 0.76
JAW_GRASP_OFFSET = (-0.021, -0.070, 0.020)
FPS = 30
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
IMAGE_CHANNELS = 3
CAMERA_KEYS = ("top", "wrist", "front")
CAMERA_SCENE_NAMES = {
    "top": "top_camera",
    "wrist": "wrist_camera",
    "front": "front_camera",
}
CUBE_TASK_NAME = "pick up the cube and place it in the bowl"
JOINT_FEATURE_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]
GRIPPER_LEROBOT_SCALE = 31.75


@dataclass
class ImageStats:
    count: int = 0
    channel_min: np.ndarray = field(default_factory=lambda: np.full(3, 1.0, dtype=np.float64))
    channel_max: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    channel_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    channel_sumsq: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def update(self, image_u8: np.ndarray) -> None:
        values = image_u8.astype(np.float64) / 255.0
        flat = values.reshape(-1, 3)
        self.count += flat.shape[0]
        self.channel_min = np.minimum(self.channel_min, flat.min(axis=0))
        self.channel_max = np.maximum(self.channel_max, flat.max(axis=0))
        self.channel_sum += flat.sum(axis=0)
        self.channel_sumsq += np.square(flat).sum(axis=0)

    def to_json(self) -> dict[str, Any]:
        if self.count <= 0:
            mean = np.full(3, 0.5, dtype=np.float64)
            std = np.zeros(3, dtype=np.float64)
            min_v = np.zeros(3, dtype=np.float64)
            max_v = np.ones(3, dtype=np.float64)
        else:
            mean = self.channel_sum / self.count
            var = np.maximum(self.channel_sumsq / self.count - np.square(mean), 0.0)
            std = np.sqrt(var)
            min_v = self.channel_min
            max_v = self.channel_max

        def nested(values: np.ndarray) -> list[list[list[float]]]:
            return [[[float(v)]] for v in values.tolist()]

        return {
            "min": nested(min_v),
            "max": nested(max_v),
            "mean": nested(mean),
            "std": nested(std),
            "count": [int(self.count)],
            "q01": nested(min_v),
            "q10": nested(mean),
            "q50": nested(mean),
            "q90": nested(mean),
            "q99": nested(max_v),
        }


class LeRobotV3EpisodeRecorder:
    """Single-episode LeRobot v3 writer for scripted state-machine traces."""

    def __init__(self, root: Path, *, seconds: float, overwrite: bool, videos: bool) -> None:
        self.root = root.resolve()
        self.max_frames = max(1, int(round(seconds * FPS)))
        self.videos = videos
        self.rows: list[dict[str, Any]] = []
        self.image_stats = {cam: ImageStats() for cam in CAMERA_KEYS}
        self.writers: dict[str, Any] = {}
        self._prepare_output_dir(overwrite)
        if self.videos:
            self._open_video_writers()

    @property
    def frame_count(self) -> int:
        return len(self.rows)

    @property
    def done(self) -> bool:
        return self.frame_count >= self.max_frames

    def record(self, env, action: torch.Tensor) -> None:
        if self.done:
            return
        frame_idx = self.frame_count
        self.rows.append(
            {
                "action": _action_to_record(action).tolist(),
                "observation.state": _read_joint_state(env).tolist(),
                "timestamp": frame_idx / FPS,
                "frame_index": frame_idx,
                "episode_index": 0,
                "index": frame_idx,
                "task_index": 0,
            }
        )
        if self.videos:
            for cam, image in _capture_images(env).items():
                self.writers[cam].append_data(image)
                self.image_stats[cam].update(image)

    def finalize(self, *, task_name: str, run_result: dict[str, Any]) -> dict[str, Any]:
        self.close()
        if not self.rows:
            raise RuntimeError("No frames recorded for LeRobot dataset")
        self._write_data_parquet()
        self._write_tasks(task_name)
        self._write_episodes(task_name)
        self._write_info(task_name)
        self._write_stats()
        meta = {
            "task_id": "TA.CUBE.STATE_MACHINE.DATASET",
            "status": "passed",
            "output_dir": str(self.root),
            "frames": self.frame_count,
            "seconds": self.frame_count / FPS,
            "fps": FPS,
            "videos": self.videos,
            "state_machine_status": run_result.get("status"),
            "placed_and_released": run_result.get("placed_and_released"),
        }
        (self.root / "meta" / "state_machine_result.json").write_text(
            json.dumps({"dataset": meta, "run": run_result}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return meta

    def close(self) -> None:
        for writer in self.writers.values():
            try:
                writer.close()
            except Exception:
                pass
        self.writers = {}

    def _prepare_output_dir(self, overwrite: bool) -> None:
        if self.root.exists():
            if not overwrite:
                raise FileExistsError(f"dataset_dir already exists: {self.root}")
            unsafe_targets = {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve()}
            if self.root in unsafe_targets or len(self.root.parts) < 4:
                raise ValueError(f"Refusing to remove unsafe dataset_dir: {self.root}")
            shutil.rmtree(self.root)

        (self.root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        (self.root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
        if self.videos:
            for cam in CAMERA_KEYS:
                (self.root / "videos" / f"observation.images.{cam}" / "chunk-000").mkdir(parents=True, exist_ok=True)

    def _open_video_writers(self) -> None:
        for cam in CAMERA_KEYS:
            path = self.root / "videos" / f"observation.images.{cam}" / "chunk-000" / "file-000.mp4"
            self.writers[cam] = imageio.get_writer(
                path,
                fps=FPS,
                codec="libx264",
                quality=8,
                macro_block_size=1,
                ffmpeg_params=["-pix_fmt", "yuv420p"],
            )

    def _write_data_parquet(self) -> None:
        fsl6 = pa.list_(pa.float32(), 6)
        table = pa.table(
            {
                "action": pa.array([r["action"] for r in self.rows], type=fsl6),
                "observation.state": pa.array([r["observation.state"] for r in self.rows], type=fsl6),
                "timestamp": pa.array([r["timestamp"] for r in self.rows], type=pa.float32()),
                "frame_index": pa.array([r["frame_index"] for r in self.rows], type=pa.int64()),
                "episode_index": pa.array([r["episode_index"] for r in self.rows], type=pa.int64()),
                "index": pa.array([r["index"] for r in self.rows], type=pa.int64()),
                "task_index": pa.array([r["task_index"] for r in self.rows], type=pa.int64()),
            }
        )
        pq.write_table(table, self.root / "data" / "chunk-000" / "file-000.parquet")

    def _write_tasks(self, task_name: str) -> None:
        table = pa.table({"task_index": [0], "__index_level_0__": [task_name]})
        pq.write_table(table, self.root / "meta" / "tasks.parquet")

    def _write_episodes(self, task_name: str) -> None:
        length = self.frame_count
        meta: dict[str, Any] = {
            "episode_index": 0,
            "tasks": [task_name],
            "length": length,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": 0,
            "dataset_to_index": length,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        if self.videos:
            for cam in CAMERA_KEYS:
                meta[f"videos/observation.images.{cam}/chunk_index"] = 0
                meta[f"videos/observation.images.{cam}/file_index"] = 0
                meta[f"videos/observation.images.{cam}/from_timestamp"] = 0.0
                meta[f"videos/observation.images.{cam}/to_timestamp"] = length / FPS

        arrays: dict[str, Any] = {
            "episode_index": pa.array([meta["episode_index"]], type=pa.int64()),
            "tasks": pa.array([meta["tasks"]], type=pa.list_(pa.string())),
            "length": pa.array([meta["length"]], type=pa.int64()),
            "data/chunk_index": pa.array([meta["data/chunk_index"]], type=pa.int64()),
            "data/file_index": pa.array([meta["data/file_index"]], type=pa.int64()),
            "dataset_from_index": pa.array([meta["dataset_from_index"]], type=pa.int64()),
            "dataset_to_index": pa.array([meta["dataset_to_index"]], type=pa.int64()),
            "meta/episodes/chunk_index": pa.array([meta["meta/episodes/chunk_index"]], type=pa.int64()),
            "meta/episodes/file_index": pa.array([meta["meta/episodes/file_index"]], type=pa.int64()),
        }
        if self.videos:
            for cam in CAMERA_KEYS:
                for name in ("chunk_index", "file_index"):
                    key = f"videos/observation.images.{cam}/{name}"
                    arrays[key] = pa.array([meta[key]], type=pa.int64())
                for name in ("from_timestamp", "to_timestamp"):
                    key = f"videos/observation.images.{cam}/{name}"
                    arrays[key] = pa.array([meta[key]], type=pa.float64())
        pq.write_table(pa.table(arrays), self.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    def _write_info(self, task_name: str) -> None:
        video_info = {
            "video.height": IMAGE_HEIGHT,
            "video.width": IMAGE_WIDTH,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": FPS,
            "video.channels": IMAGE_CHANNELS,
            "has_audio": False,
        }
        features: dict[str, Any] = {
            "action": {"dtype": "float32", "names": JOINT_FEATURE_NAMES, "shape": [6]},
            "observation.state": {"dtype": "float32", "names": JOINT_FEATURE_NAMES, "shape": [6]},
        }
        if self.videos:
            for cam in CAMERA_KEYS:
                features[f"observation.images.{cam}"] = {
                    "dtype": "video",
                    "shape": [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS],
                    "names": ["height", "width", "channels"],
                    "info": video_info,
                }
        features.update(
            {
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
            }
        )
        info = {
            "codebase_version": "v3.0",
            "robot_type": "so_follower",
            "total_episodes": 1,
            "total_frames": self.frame_count,
            "total_tasks": 1,
            "chunks_size": 1000,
            "data_files_size_in_mb": 100,
            "video_files_size_in_mb": 200,
            "fps": FPS,
            "splits": {"train": "0:1"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": features,
            "task": task_name,
        }
        (self.root / "meta" / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_stats(self) -> None:
        stats = {
            "action": _numeric_stats([r["action"] for r in self.rows]),
            "observation.state": _numeric_stats([r["observation.state"] for r in self.rows]),
            "timestamp": _numeric_stats([r["timestamp"] for r in self.rows]),
            "frame_index": _numeric_stats([r["frame_index"] for r in self.rows]),
            "episode_index": _numeric_stats([r["episode_index"] for r in self.rows]),
            "index": _numeric_stats([r["index"] for r in self.rows]),
            "task_index": _numeric_stats([r["task_index"] for r in self.rows]),
        }
        if self.videos:
            for cam in CAMERA_KEYS:
                stats[f"observation.images.{cam}"] = self.image_stats[cam].to_json()
        (self.root / "meta" / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


class ExpertTrajectoryRecorder:
    """Step-pre expert pairs for BC warm-start.

    LeRobot dataset rows intentionally store post-step observations. BC needs
    state_t -> action_t, so this recorder is called immediately before
    env.step(action).
    """

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.obs: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.phases: list[str] = []

    def record(self, env, action: torch.Tensor, phase: str) -> None:
        obs = task_mdp.rl_state(
            env.unwrapped,
            pen_names=CUBE_NAMES,
            cup_name=BOWL_NAME,
        )
        self.obs.append(obs[0].detach().cpu().to(torch.float32))
        self.actions.append(action[0, :6].detach().cpu().to(torch.float32))
        self.phases.append(phase)

    def finalize(self, *, run_result: dict[str, Any]) -> dict[str, Any]:
        if self.obs:
            obs = torch.stack(self.obs, dim=0)
            actions = torch.stack(self.actions, dim=0)
        else:
            obs = torch.empty((0, 37), dtype=torch.float32)
            actions = torch.empty((0, 6), dtype=torch.float32)

        meta = {
            "task_id": "TA.CUBE.STATE_MACHINE.EXPERT",
            "status": "passed" if run_result.get("placed_and_released") else "failed",
            "task": run_result.get("task"),
            "frames": int(obs.shape[0]),
            "active_objects": run_result.get("active_objects"),
            "object_radius_scale": run_result.get("object_radius_scale"),
            "container_angle_scale": run_result.get("container_angle_scale"),
            "container_radius_scale": run_result.get("container_radius_scale"),
            "placed_and_released": run_result.get("placed_and_released"),
            "final_inside": run_result.get("final_inside"),
            "controller": run_result.get("controller"),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "obs": obs,
                "actions": actions,
                "phases": self.phases,
                "meta": meta,
            },
            self.path,
        )
        return {
            "task_id": "TA.CUBE.STATE_MACHINE.EXPERT",
            "status": meta["status"],
            "path": str(self.path),
            "frames": int(obs.shape[0]),
            "obs_shape": list(obs.shape),
            "action_shape": list(actions.shape),
        }


def _round_list(values: torch.Tensor, digits: int = 5) -> list[float]:
    return [round(float(v), digits) for v in values.detach().cpu().flatten().tolist()]


def _to_lerobot_units(values_rad: np.ndarray) -> np.ndarray:
    """Convert Isaac joint radians to the real LeRobot SO-101 convention."""

    values = np.asarray(values_rad, dtype=np.float32).copy()
    values[:ARM_DOF] = values[:ARM_DOF] * (180.0 / math.pi)
    values[5] = values[5] * GRIPPER_LEROBOT_SCALE
    return values.astype(np.float32)


def _read_joint_state(env) -> np.ndarray:
    robot = env.unwrapped.scene["robot"]
    return _to_lerobot_units(robot.data.joint_pos[0, :6].detach().cpu().numpy())


def _action_to_record(action_tensor: torch.Tensor) -> np.ndarray:
    action = action_tensor[0, :6].detach().cpu().numpy()
    return _to_lerobot_units(action)


def _capture_images(env) -> dict[str, np.ndarray]:
    images: dict[str, np.ndarray] = {}
    for key in CAMERA_KEYS:
        cam = env.unwrapped.scene[CAMERA_SCENE_NAMES[key]]
        rgb = cam.data.output["rgb"][0].detach().cpu().numpy()
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        if rgb.dtype != np.uint8:
            if np.issubdtype(rgb.dtype, np.floating):
                rgb = np.clip(rgb, 0.0, 1.0) * 255.0
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        image = np.ascontiguousarray(rgb)
        expected_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
        if image.shape != expected_shape:
            raise ValueError(f"{key} image shape {image.shape}, expected {expected_shape}")
        images[key] = image
    return images


def _numeric_stats(array_like: list[Any] | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(array_like)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[0] == 0:
        width = arr.shape[1] if arr.ndim > 1 else 1
        zeros = [0.0 for _ in range(width)]
        return {k: zeros for k in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")} | {
            "count": [0]
        }
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [int(arr.shape[0])],
        "q01": np.quantile(arr, 0.01, axis=0).tolist(),
        "q10": np.quantile(arr, 0.10, axis=0).tolist(),
        "q50": np.quantile(arr, 0.50, axis=0).tolist(),
        "q90": np.quantile(arr, 0.90, axis=0).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).tolist(),
    }


def _body_pos(robot, body_name: str) -> torch.Tensor:
    body_id = robot.data.body_names.index(body_name)
    return robot.data.body_pos_w[:, body_id, :]


def _body_quat(robot, body_name: str) -> torch.Tensor:
    body_id = robot.data.body_names.index(body_name)
    return robot.data.body_quat_w[:, body_id, :]


def _quat_apply_wxyz(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply wxyz quaternion to a vector."""

    q_xyz = quat[:, 1:4]
    q_w = quat[:, 0:1]
    t = 2.0 * torch.cross(q_xyz, vec, dim=-1)
    return vec + q_w * t + torch.cross(q_xyz, t, dim=-1)


def _jacobian_row(robot, body_name: str) -> torch.Tensor:
    """Return body Jacobian row, shape (1, 6, ARM_DOF).

    PhysX omits the fixed root body from get_jacobians(), so body index N maps
    to Jacobian row N-1 for this fixed-base articulation.
    """

    body_id = robot.data.body_names.index(body_name)
    row = body_id - 1
    if row < 0:
        raise ValueError(f"Body {body_name!r} has no movable-body Jacobian row")
    jac = robot.root_physx_view.get_jacobians()
    return jac[:, row, :6, :ARM_DOF]


def _grasp_point_pos(robot) -> torch.Tensor:
    offset = torch.tensor(JAW_GRASP_OFFSET, device=robot.data.joint_pos.device, dtype=torch.float32).reshape(1, 3)
    offset_w = _quat_apply_wxyz(_body_quat(robot, "jaw"), offset)
    return _body_pos(robot, "jaw") + offset_w


def _diagnostic_pose(env) -> dict[str, Any]:
    scene = env.unwrapped.scene
    robot = scene["robot"]
    gripper = _body_pos(robot, "gripper")[0]
    jaw = _body_pos(robot, "jaw")[0]
    return {
        "gripper_w": _round_list(gripper),
        "jaw_w": _round_list(jaw),
        "gripper_jaw_midpoint_w": _round_list(0.5 * (gripper + jaw)),
        "cube_w": {
            name: _round_list(scene[name].data.root_pos_w[0])
            for name in CUBE_NAMES[: args.active_objects]
        },
        "bowl_w": _round_list(scene[BOWL_NAME].data.root_pos_w[0]),
    }


def _grasp_point_jacobian(robot) -> torch.Tensor:
    body_jac = _jacobian_row(robot, "jaw")
    linear = body_jac[:, :3, :]
    angular = body_jac[:, 3:6, :]
    offset = torch.tensor(JAW_GRASP_OFFSET, device=robot.data.joint_pos.device, dtype=torch.float32).reshape(1, 3)
    offset_w = _quat_apply_wxyz(_body_quat(robot, "jaw"), offset)
    point_terms = []
    for j in range(ARM_DOF):
        point_terms.append(linear[:, :, j] + torch.cross(angular[:, :, j], offset_w, dim=-1))
    return torch.stack(point_terms, dim=-1)


def _arm_limits(robot, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    limits = robot.data.soft_joint_pos_limits[0, :ARM_DOF].to(device=device)
    lower = limits[:, 0]
    upper = limits[:, 1]
    # 일부 USD가 넓은 soft limit을 제공하지 못하는 경우를 위한 보수적 fallback.
    lower = torch.where(torch.isfinite(lower), lower, torch.full_like(lower, -3.14))
    upper = torch.where(torch.isfinite(upper), upper, torch.full_like(upper, 3.14))
    return lower, upper


def _joint_position_action(
    robot,
    command: torch.Tensor,
    arm_target: torch.Tensor,
    gripper_target: float,
    device: str,
) -> torch.Tensor:
    """Build a slew-limited joint-position action.

    FK waypoints can be far apart. Sending the final target directly makes the
    implicit PD drive snap toward it, so the commanded target itself is moved at
    a bounded per-step rate.
    """

    arm_target = arm_target[:ARM_DOF].to(device=device, dtype=torch.float32)
    lower, upper = _arm_limits(robot, device)

    arm_delta = torch.clamp(
        arm_target - command[:ARM_DOF],
        -abs(args.max_arm_step_delta),
        abs(args.max_arm_step_delta),
    )
    gripper_delta = torch.clamp(
        torch.tensor(float(gripper_target), device=device, dtype=torch.float32) - command[5],
        -abs(args.max_gripper_step_delta),
        abs(args.max_gripper_step_delta),
    )

    command[:ARM_DOF] = torch.minimum(torch.maximum(command[:ARM_DOF] + arm_delta, lower), upper)
    command[5] = command[5] + gripper_delta
    return command.reshape(1, 6).clone()


def _slew_limited_step_count(command: torch.Tensor, q_goal: torch.Tensor, gripper_target: float) -> int:
    arm_delta = float(torch.max(torch.abs(q_goal[:ARM_DOF] - command[:ARM_DOF])).item())
    gripper_delta = abs(float(gripper_target) - float(command[5].item()))
    arm_steps = math.ceil(arm_delta / max(abs(args.max_arm_step_delta), 1e-6))
    gripper_steps = math.ceil(gripper_delta / max(abs(args.max_gripper_step_delta), 1e-6))
    return max(arm_steps, gripper_steps) + max(0, args.command_settle_steps)


def _ik_action(
    robot,
    target_grasp_point_w: torch.Tensor,
    gripper_target: float,
    device: str,
    *,
    damping: float,
    gain: float,
    max_joint_delta: float,
) -> tuple[torch.Tensor, float]:
    grasp_point = _grasp_point_pos(robot)
    error = target_grasp_point_w - grasp_point
    jac = _grasp_point_jacobian(robot)  # (1, 3, 5)
    j = jac[0]
    err = error[0]

    eye = torch.eye(3, device=device, dtype=j.dtype)
    lhs = j @ j.transpose(0, 1) + (damping * damping) * eye
    rhs = torch.linalg.solve(lhs, err.unsqueeze(-1)).squeeze(-1)
    dq = j.transpose(0, 1) @ rhs
    dq = torch.clamp(gain * dq, -max_joint_delta, max_joint_delta)

    lower, upper = _arm_limits(robot, device)
    q = robot.data.joint_pos[0, :ARM_DOF]
    q_target = torch.minimum(torch.maximum(q + dq, lower), upper)

    action = torch.zeros((1, 6), device=device, dtype=torch.float32)
    action[0, :ARM_DOF] = q_target
    action[0, 5] = float(gripper_target)
    return action, float(torch.linalg.norm(error[0]).item())


def _fk_solve_joint_target(
    env,
    target_grasp_point_w: torch.Tensor,
    gripper_target: float,
    device: str,
    *,
    samples: int,
    continuity_weight: float,
    seed_offset: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Random-FK waypoint solver.

    현재 scene에서 robot joint state만 잠깐 써서 FK 후보를 평가한 뒤 원래 상태로
    되돌린다. cube/bowl pose는 쓰지 않으며, 실제 rollout은 반환된 joint target을
    action으로 추종한다.
    """

    scene = env.unwrapped.scene
    robot = scene["robot"]
    env_ids = torch.tensor([0], device=device, dtype=torch.long)
    saved_q = robot.data.joint_pos[:, :6].clone()
    saved_v = robot.data.joint_vel[:, :6].clone()
    current_arm = saved_q[0, :ARM_DOF].clone()
    lower, upper = _arm_limits(robot, device)
    target = target_grasp_point_w.to(device=device, dtype=torch.float32).reshape(3)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed + seed_offset))

    best_score = float("inf")
    best_dist = float("inf")
    best_q = current_arm.clone()
    best_grasp_point = _grasp_point_pos(robot)[0].clone()
    zero_vel = torch.zeros((1, 6), device=device)

    def evaluate(q_arm: torch.Tensor) -> None:
        nonlocal best_score, best_dist, best_q, best_grasp_point
        q = torch.zeros((1, 6), device=device)
        q[0, :ARM_DOF] = q_arm
        q[0, 5] = float(gripper_target)
        robot.write_joint_state_to_sim(q, zero_vel, env_ids=env_ids)
        scene.update(0.0)
        grasp_point = _grasp_point_pos(robot)[0].clone()
        dist = float(torch.linalg.norm(grasp_point - target).item())
        continuity = float(torch.linalg.norm(q_arm - current_arm).item())
        score = dist + continuity_weight * continuity
        if score < best_score:
            best_score = score
            best_dist = dist
            best_q = q_arm.clone()
            best_grasp_point = grasp_point.clone()

    # 현재 근처 후보와 전역 후보를 섞는다. 전역 후보가 테이블 근처 자세를 찾고,
    # 근처 후보가 불필요한 큰 관절 점프를 줄인다.
    evaluate(current_arm)
    local_count = min(max(samples // 3, 1), samples)
    global_count = max(samples - local_count, 1)
    for _ in range(local_count):
        noise = torch.randn((ARM_DOF,), generator=gen, device=device) * 0.35
        evaluate(torch.minimum(torch.maximum(current_arm + noise, lower), upper))
    for _ in range(global_count):
        evaluate(lower + (upper - lower) * torch.rand((ARM_DOF,), generator=gen, device=device))

    robot.write_joint_state_to_sim(saved_q, saved_v, env_ids=env_ids)
    scene.update(0.0)
    return best_q, {
        "planned_error_m": round(best_dist, 5),
        "planned_score": round(best_score, 5),
        "planned_grasp_point_w": _round_list(best_grasp_point),
        "planned_joint_target": _round_list(best_q),
    }


def _cube_inside_bowl(env, cube_name: str, radius: float) -> bool:
    inside = task_mdp.pen_inside_cup(
        env.unwrapped,
        object_cfg=SceneEntityCfg(cube_name),
        cup_cfg=SceneEntityCfg(BOWL_NAME),
        radius=radius,
        height_range=BOWL_HEIGHT_RANGE,
    )
    return bool(inside[0].item())


def _cube_lifted(env, cube_name: str, min_lift: float = 0.08) -> bool:
    cube = env.unwrapped.scene[cube_name]
    return bool((cube.data.root_pos_w[0, 2] > DESK_TOP_Z + min_lift).item())


def _placed_and_released(env, cube_names: list[str], radius: float) -> bool:
    robot = env.unwrapped.scene["robot"]
    gripper_open = bool((robot.data.joint_pos[0, 5] > 0.60).item())
    if not gripper_open:
        return False
    return all(_cube_inside_bowl(env, name, radius) for name in cube_names)


def _phase(
    env,
    device: str,
    name: str,
    target_fn: Callable[[], torch.Tensor],
    gripper_target: float,
    steps: int,
    trace: list[dict[str, Any]],
    command: torch.Tensor,
    recorder: LeRobotV3EpisodeRecorder | None,
    expert_recorder: ExpertTrajectoryRecorder | None = None,
    *,
    tolerance: float,
) -> dict[str, Any]:
    robot = env.unwrapped.scene["robot"]
    min_error = float("inf")
    final_error = float("inf")
    reached_step: int | None = None
    done_seen = False
    target = target_fn().to(device=device, dtype=torch.float32).reshape(1, 3)
    q_goal, plan = _fk_solve_joint_target(
        env,
        target,
        gripper_target,
        device,
        samples=args.fk_samples,
        continuity_weight=args.continuity_weight,
        seed_offset=len(trace) * 997,
    )

    requested_steps = int(steps)
    actual_steps = max(1, requested_steps, _slew_limited_step_count(command, q_goal, gripper_target))
    for step in range(actual_steps):
        action = _joint_position_action(robot, command, q_goal, gripper_target, device)
        err = float(torch.linalg.norm(_grasp_point_pos(robot)[0] - target[0]).item())
        if expert_recorder is not None:
            expert_recorder.record(env, action, name)
        step_out = env.step(action)
        if len(step_out) == 5:
            _obs, _rew, terminated, truncated, _infos = step_out
            dones = terminated | truncated
        else:
            _obs, _rew, dones, _infos = step_out
        if recorder is not None:
            recorder.record(env, action)
        min_error = min(min_error, err)
        final_error = err
        if bool(dones[0].item()):
            done_seen = True
        if err <= tolerance and reached_step is None:
            reached_step = step + 1

    stat = {
        "phase": name,
        "steps": int(actual_steps),
        "requested_steps": requested_steps,
        "reached_step": reached_step,
        "min_error_m": round(min_error, 5),
        "final_error_m": round(final_error, 5),
        "done_seen": done_seen,
        "target_grasp_point_w": _round_list(target[0]),
        "grasp_point_w": _round_list(_grasp_point_pos(robot)[0]),
        "joint_pos": _round_list(robot.data.joint_pos[0, :6]),
        **_diagnostic_pose(env),
        **plan,
    }
    trace.append(stat)
    return stat


def _hold_joint_target(
    env,
    target: torch.Tensor,
    gripper_target: float,
    steps: int,
    command: torch.Tensor,
    recorder: LeRobotV3EpisodeRecorder | None,
    expert_recorder: ExpertTrajectoryRecorder | None = None,
    phase: str = "hold",
) -> None:
    robot = env.unwrapped.scene["robot"]
    device = str(target.device)
    for _ in range(max(1, steps)):
        action = _joint_position_action(robot, command, target[:ARM_DOF], gripper_target, device)
        if expert_recorder is not None:
            expert_recorder.record(env, action, phase)
        env.step(action)
        if recorder is not None:
            recorder.record(env, action)


def _target_from_cube(env, cube_name: str, dz: float) -> Callable[[], torch.Tensor]:
    def target() -> torch.Tensor:
        cube = env.unwrapped.scene[cube_name]
        pos = cube.data.root_pos_w[0].clone()
        pos[2] += dz
        return pos

    return target


def _target_from_bowl(env, dz: float) -> Callable[[], torch.Tensor]:
    def target() -> torch.Tensor:
        bowl = env.unwrapped.scene[BOWL_NAME]
        pos = bowl.data.root_pos_w[0].clone()
        pos[2] = DESK_TOP_Z + dz
        return pos

    return target


def _run_state_machine(
    env,
    device: str,
    active_names: list[str],
    recorder: LeRobotV3EpisodeRecorder | None = None,
    expert_recorder: ExpertTrajectoryRecorder | None = None,
) -> dict[str, Any]:
    scene = env.unwrapped.scene
    robot = scene["robot"]
    trace: list[dict[str, Any]] = []
    bowl_radius = BOWL_SUCCESS_RADIUS * max(0.1, args.container_radius_scale)
    command = robot.data.joint_pos[0, :6].clone()

    for _ in range(args.settle_steps):
        zero_action = _joint_position_action(robot, command, torch.zeros(ARM_DOF, device=device), args.gripper_open, device)
        if expert_recorder is not None:
            expert_recorder.record(env, zero_action, "settle")
        env.step(zero_action)
        if recorder is not None:
            recorder.record(env, zero_action)

    for cube_name in active_names:
        cube_start = scene[cube_name].data.root_pos_w[0].clone()
        phase_prefix = cube_name.lower()
        grasped = False
        for attempt in range(1, max(1, args.max_grasp_attempts) + 1):
            attempt_prefix = f"{phase_prefix}.attempt{attempt}"
            _phase(
                env,
                device,
                f"{attempt_prefix}.approach",
                _target_from_cube(env, cube_name, args.approach_height),
                args.gripper_open,
                args.approach_steps,
                trace,
                command,
                recorder,
                expert_recorder,
                tolerance=args.target_tolerance,
            )
            _phase(
                env,
                device,
                f"{attempt_prefix}.descend",
                _target_from_cube(env, cube_name, args.grasp_z_offset),
                args.gripper_open,
                args.descend_steps,
                trace,
                command,
                recorder,
                expert_recorder,
                tolerance=args.target_tolerance,
            )
            grasp_target = _target_from_cube(env, cube_name, args.grasp_z_offset)().clone()
            _phase(
                env,
                device,
                f"{attempt_prefix}.close",
                lambda target=grasp_target: target,
                args.gripper_closed,
                args.close_steps,
                trace,
                command,
                recorder,
                expert_recorder,
                tolerance=args.target_tolerance,
            )
            _phase(
                env,
                device,
                f"{attempt_prefix}.lift",
                lambda target=grasp_target: target + torch.tensor(
                    [0.0, 0.0, args.lift_height], device=device, dtype=torch.float32
                ),
                args.gripper_closed,
                args.lift_steps,
                trace,
                command,
                recorder,
                expert_recorder,
                tolerance=args.target_tolerance,
            )
            grasped = _cube_lifted(env, cube_name)
            trace.append({
                "phase": f"{attempt_prefix}.lift_check",
                "grasped": grasped,
                "cube_w": _round_list(scene[cube_name].data.root_pos_w[0]),
                "joint_pos": _round_list(robot.data.joint_pos[0, :6]),
                **_diagnostic_pose(env),
            })
            if grasped:
                break

            retry_hold = robot.data.joint_pos[0, :ARM_DOF].clone()
            retry_open_steps = max(args.open_steps, args.command_settle_steps // 2)
            _hold_joint_target(
                env,
                retry_hold,
                args.gripper_open,
                retry_open_steps,
                command,
                recorder,
                expert_recorder,
                phase=f"{attempt_prefix}.retry_open",
            )
            trace.append({
                "phase": f"{attempt_prefix}.retry_open",
                "steps": retry_open_steps,
                "grasp_point_w": _round_list(_grasp_point_pos(robot)[0]),
                "joint_pos": _round_list(robot.data.joint_pos[0, :6]),
                **_diagnostic_pose(env),
            })

        if not grasped:
            cube_end = scene[cube_name].data.root_pos_w[0].clone()
            trace.append({
                "phase": f"{phase_prefix}.result",
                "cube_start_w": _round_list(cube_start),
                "cube_end_w": _round_list(cube_end),
                "inside_bowl": _cube_inside_bowl(env, cube_name, bowl_radius),
                "grasped": False,
            })
            continue

        _phase(
            env,
            device,
            f"{phase_prefix}.transport",
            _target_from_bowl(env, args.transport_height),
            args.gripper_closed,
            args.transport_steps,
            trace,
            command,
            recorder,
            expert_recorder,
            tolerance=args.target_tolerance,
        )
        _phase(
            env,
            device,
            f"{phase_prefix}.place",
            _target_from_bowl(env, args.place_height),
            args.gripper_closed,
            args.place_steps,
            trace,
            command,
            recorder,
            expert_recorder,
            tolerance=args.target_tolerance,
        )
        # 열 때는 마지막 joint target을 유지한다. 위치 IK가 그릇 안 큐브를 다시
        # 추적하며 건드리지 않도록, release 동안 관절 목표를 고정한다.
        joint_hold = robot.data.joint_pos[0, :ARM_DOF].clone()
        _hold_joint_target(
            env,
            joint_hold,
            args.gripper_open,
            args.open_steps,
            command,
            recorder,
            expert_recorder,
            phase=f"{phase_prefix}.open",
        )
        trace.append({
            "phase": f"{phase_prefix}.open",
            "steps": args.open_steps,
            "grasp_point_w": _round_list(_grasp_point_pos(robot)[0]),
            "joint_pos": _round_list(robot.data.joint_pos[0, :6]),
        })
        _hold_joint_target(
            env,
            joint_hold,
            args.gripper_open,
            args.final_settle_steps,
            command,
            recorder,
            expert_recorder,
            phase=f"{phase_prefix}.final_settle",
        )

        cube_end = scene[cube_name].data.root_pos_w[0].clone()
        trace.append({
            "phase": f"{phase_prefix}.result",
            "cube_start_w": _round_list(cube_start),
            "cube_end_w": _round_list(cube_end),
            "inside_bowl": _cube_inside_bowl(env, cube_name, bowl_radius),
        })

    final_inside = {
        name: _cube_inside_bowl(env, name, bowl_radius)
        for name in active_names
    }
    return {
        "trace": trace,
        "final_inside": final_inside,
        "placed_and_released": _placed_and_released(env, active_names, bowl_radius),
        "final_gripper": round(float(robot.data.joint_pos[0, 5].item()), 5),
        "final_grasp_point_w": _round_list(_grasp_point_pos(robot)[0]),
        "final_joint_pos": _round_list(robot.data.joint_pos[0, :6]),
        "bowl_w": _round_list(scene[BOWL_NAME].data.root_pos_w[0]),
        "cube_w": {
            name: _round_list(scene[name].data.root_pos_w[0])
            for name in active_names
        },
    }


def main() -> None:
    env = None
    recorder: LeRobotV3EpisodeRecorder | None = None
    expert_recorder: ExpertTrajectoryRecorder | None = None
    try:
        device: str = args.device
        env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
        apply_curriculum(
            env_cfg,
            active_objects=args.active_objects,
            object_radius_scale=args.object_radius_scale,
            container_angle_scale=args.container_angle_scale,
            container_radius_scale=args.container_radius_scale,
        )
        # State-machine 검증 중에는 중간 성공 termination으로 자동 reset되지 않게
        # 끄고, 마지막에 gripper release까지 포함해 직접 판정한다.
        env_cfg.terminations.success = None
        total_steps = (
            args.settle_steps
            + args.active_objects
            * (
                args.approach_steps
                + args.descend_steps
                + args.close_steps
                + args.lift_steps
                + args.transport_steps
                + args.place_steps
                + args.open_steps
                + args.final_settle_steps
            )
            + 120
        )
        env_cfg.episode_length_s = max(
            env_cfg.episode_length_s,
            total_steps * env_cfg.sim.dt * env_cfg.decimation + 5.0,
            args.record_seconds + 30.0,
            180.0,
        )
        if args.dataset_dir is not None and not args.no_videos:
            add_pick_cube_cameras(env_cfg.scene)

        env = gym.make(args.task, cfg=env_cfg)
        env.reset()
        if args.dataset_dir is not None:
            recorder = LeRobotV3EpisodeRecorder(
                args.dataset_dir,
                seconds=args.record_seconds,
                overwrite=args.overwrite_dataset,
                videos=not args.no_videos,
            )
        if args.expert_dataset_pt is not None:
            expert_recorder = ExpertTrajectoryRecorder(args.expert_dataset_pt)
        zero_action = torch.zeros((1, 6), device=device)
        zero_action[0, 5] = args.gripper_open
        for _ in range(max(0, args.warmup_steps)):
            env.step(zero_action)
        active_names = CUBE_NAMES[: args.active_objects]
        result = _run_state_machine(env, device, active_names, recorder, expert_recorder)
        passed = bool(result["placed_and_released"])
        payload = {
            "task_id": "TA.CUBE.STATE_MACHINE",
            "task": args.task,
            "status": "passed" if passed else "failed",
            "active_objects": args.active_objects,
            "object_radius_scale": args.object_radius_scale,
            "container_angle_scale": args.container_angle_scale,
            "container_radius_scale": args.container_radius_scale,
            "controller": {
                "type": "random_fk_waypoint_joint_position",
                "end_effector": "jaw + quat(jaw) * (-0.021, -0.070, 0.020)",
                "fk_samples": args.fk_samples,
                "continuity_weight": args.continuity_weight,
                "max_arm_step_delta_rad": args.max_arm_step_delta,
                "max_gripper_step_delta_rad": args.max_gripper_step_delta,
                "command_settle_steps": args.command_settle_steps,
                "max_grasp_attempts": args.max_grasp_attempts,
                "gripper_open": args.gripper_open,
                "gripper_closed": args.gripper_closed,
            },
            **result,
        }
        if recorder is not None:
            payload["dataset"] = recorder.finalize(task_name=CUBE_TASK_NAME, run_result=payload)
            recorder = None
        if expert_recorder is not None:
            payload["expert_dataset"] = expert_recorder.finalize(run_result=payload)
            expert_recorder = None
        env.close()
        env = None
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        if not passed:
            sys.exit(1)
    except Exception as exc:
        payload = {
            "task_id": "TA.CUBE.STATE_MACHINE",
            "task": args.task,
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        sys.exit(1)
    finally:
        if recorder is not None:
            recorder.close()
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        simulation_app.close()


if __name__ == "__main__":
    main()
