"""TC.1 RSL-RL expert rollout -> LeRobot v3 dataset recorder.

This script records successful SimToReal-SO101 PickCube/PickPen episodes only.
It writes the LeRobot v3 data contract used by the real SO-101 datasets:
6-dim joint action/state, 30 FPS timestamps, and two h264 camera videos.

Example:
    uv run --group isaac --locked python scripts/sim/rollout_to_lerobot.py \
        --checkpoint /DISK1/so101-sim2real/outputs/pick_cube_rl/model_200.pt \
        --output_dir /DISK1/so101-sim2real/outputs/tc1_rollout_10ep \
        --episodes 10 --overwrite
"""

from __future__ import annotations

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import json
import shutil
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


# 단위/카메라 변환 상수·헬퍼는 lerobot_units 공용 모듈 (같은 디렉터리, AppLauncher 무의존).
from lerobot_units import (  # noqa: E402
    CAMERA_KEYS,
    CAMERA_SCENE_NAMES,
    FPS,
    GRIPPER_LEROBOT_SCALE,  # noqa: F401  (하위 호환 재노출)
    IMAGE_CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    JOINT_FEATURE_NAMES,
    read_camera_rgb_u8,
    to_lerobot_units as _to_lerobot_units,
)

TASK_ID = "TC.1"
PEN_TASK_NAME = "pick up the pen and place it in the holder"
CUBE_TASK_NAME = "pick up the cube and place it in the bowl"


parser = argparse.ArgumentParser(description="TC.1 rollout-to-LeRobot-v3 recorder")
parser.add_argument("--task", default="SimToReal-SO101-PickCube-v0")
parser.add_argument("--checkpoint", required=True, help="RSL-RL OnPolicyRunner checkpoint")
parser.add_argument("--output_dir", required=True, help="LeRobot dataset output directory")
parser.add_argument("--episodes", type=int, default=10, help="Number of successful episodes to keep")
parser.add_argument("--max_attempts", type=int, default=None, help="Stop after this many attempted episodes")
parser.add_argument("--max_episode_steps", type=int, default=900)
parser.add_argument("--num_envs", type=int, default=1, help="TC.1 recorder currently supports only 1 env")
parser.add_argument("--rl_device", default=None, help="RL device; defaults to --device")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--obs_group", default="rl_policy")
parser.add_argument("--critic_obs_group", default=None)
parser.add_argument("--clip_actions", type=float, default=1.0)
parser.add_argument("--init_noise_std", type=float, default=0.2)
parser.add_argument("--num_learning_epochs", type=int, default=20)
parser.add_argument("--num_mini_batches", type=int, default=4)
parser.add_argument("--overwrite", action="store_true", help="Replace output_dir if it already exists")
parser.add_argument("--no_videos", action="store_true", help="Skip mp4 writing, but keep video metadata contract")
parser.add_argument("--deterministic", action="store_true", help="Use deterministic act_inference instead of stochastic act")
parser.add_argument("--warmup_steps", type=int, default=5, help="Camera/render warm-up steps before recording")

# Curriculum defaults.
parser.add_argument("--active_objects", "--active_pens", dest="active_objects",
                    type=int, default=4, choices=[1, 2, 3, 4])
parser.add_argument("--object_radius_scale", "--pen_radius_scale", dest="object_radius_scale",
                    type=float, default=1.0)
parser.add_argument("--container_angle_scale", "--cup_angle_scale", dest="container_angle_scale",
                    type=float, default=1.0)
parser.add_argument("--container_radius_scale", "--cup_radius_scale", dest="container_radius_scale",
                    type=float, default=1.0)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = not args.no_videos

launcher = AppLauncher(args)
simulation_app = launcher.app

# Isaac Sim must be running before these imports.
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402

if not args.no_videos:
    import imageio.v2 as imageio  # noqa: E402

import sim_to_real  # noqa: E402  # registers SimToReal-SO101-PickCube/PickPen-v0

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import (  # noqa: E402
    add_pick_cube_cameras,
    apply_curriculum as apply_cube_curriculum,
)
from sim_to_real.tasks.pick_pen.pick_pen_env_cfg import (  # noqa: E402
    add_pick_pen_cameras,
    apply_curriculum as apply_pen_curriculum,
)


@dataclass
class FrameRecord:
    action: np.ndarray
    state: np.ndarray
    images: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class EpisodeRecord:
    frames: list[FrameRecord] = field(default_factory=list)

    def clear(self) -> None:
        self.frames.clear()

    @property
    def length(self) -> int:
        return len(self.frames)


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
            # Streaming quantiles are not worth the memory here; keep sane approximations.
            "q01": nested(min_v),
            "q10": nested(mean),
            "q50": nested(mean),
            "q90": nested(mean),
            "q99": nested(max_v),
        }


def _build_train_cfg(cli_args: argparse.Namespace) -> dict[str, Any]:
    """Recreate the actor/critic network shape used by train.py/eval_success.py."""
    rl_device = cli_args.rl_device if cli_args.rl_device is not None else cli_args.device
    obs_group = cli_args.obs_group
    critic_group = cli_args.critic_obs_group if cli_args.critic_obs_group is not None else obs_group
    return {
        "seed": cli_args.seed,
        "device": rl_device,
        "num_steps_per_env": 24,
        "max_iterations": 1,
        "save_interval": 1,
        "experiment_name": "rollout_tmp",
        "run_name": "",
        "resume": False,
        "load_run": ".*",
        "load_checkpoint": "model_.*.pt",
        "logger": "tensorboard",
        "obs_groups": {"policy": [obs_group], "critic": [critic_group]},
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": cli_args.init_noise_std,
            "actor_hidden_dims": [128, 128],
            "critic_hidden_dims": [128, 128],
            "activation": "elu",
            "actor_obs_normalization": False,
            "critic_obs_normalization": False,
        },
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": cli_args.num_learning_epochs,
            "num_mini_batches": cli_args.num_mini_batches,
            "learning_rate": 3e-4,
            "schedule": "fixed",
            "gamma": 0.99,
            "lam": 0.95,
            "entropy_coef": 0.005,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
        },
    }


def _task_name(task: str) -> str:
    return CUBE_TASK_NAME if "PickCube" in task else PEN_TASK_NAME


def _apply_task_curriculum(env_cfg, cli_args: argparse.Namespace) -> None:
    """task 이름에 맞는 curriculum을 적용한다."""

    params = {
        "active_objects": cli_args.active_objects,
        "object_radius_scale": cli_args.object_radius_scale,
        "container_angle_scale": cli_args.container_angle_scale,
        "container_radius_scale": cli_args.container_radius_scale,
    }
    if cli_args.task and "PickCube" in cli_args.task:
        apply_cube_curriculum(env_cfg, **params)
    else:
        apply_pen_curriculum(
            env_cfg,
            active_pens=params["active_objects"],
            pen_radius_scale=params["object_radius_scale"],
            cup_angle_scale=params["container_angle_scale"],
            cup_radius_scale=params["container_radius_scale"],
        )


def _add_task_cameras(env_cfg, task: str) -> None:
    if "PickCube" in task:
        add_pick_cube_cameras(env_cfg.scene)
    else:
        add_pick_pen_cameras(env_cfg.scene)


def _prepare_output_dir(path: Path, overwrite: bool) -> None:
    resolved = path.resolve()
    if resolved.exists():
        if not overwrite:
            raise FileExistsError(f"output_dir already exists: {resolved}")
        unsafe_targets = {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve()}
        if resolved in unsafe_targets or len(resolved.parts) < 4:
            raise ValueError(f"Refusing to remove unsafe output_dir: {resolved}")
        shutil.rmtree(resolved)

    (resolved / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (resolved / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    for cam in CAMERA_KEYS:
        (resolved / "videos" / f"observation.images.{cam}" / "chunk-000").mkdir(parents=True, exist_ok=True)


def _read_joint_state(raw_env) -> np.ndarray:
    robot = raw_env.unwrapped.scene["robot"]
    return _to_lerobot_units(robot.data.joint_pos[0].detach().cpu().numpy())


def _action_to_record(action_tensor: torch.Tensor) -> np.ndarray:
    action = action_tensor[0].detach().cpu().numpy()
    return _to_lerobot_units(action)


def _capture_images(raw_env) -> dict[str, np.ndarray]:
    return {key: read_camera_rgb_u8(raw_env, CAMERA_SCENE_NAMES[key]) for key in CAMERA_KEYS}


def _open_video_writers(root: Path) -> dict[str, Any]:
    writers: dict[str, Any] = {}
    for cam in CAMERA_KEYS:
        path = root / "videos" / f"observation.images.{cam}" / "chunk-000" / "file-000.mp4"
        writers[cam] = imageio.get_writer(
            path,
            fps=FPS,
            codec="libx264",
            quality=8,
            macro_block_size=1,
            ffmpeg_params=["-pix_fmt", "yuv420p"],
        )
    return writers


def _close_video_writers(writers: dict[str, Any]) -> None:
    for writer in writers.values():
        try:
            writer.close()
        except Exception:
            pass


def _numeric_stats(array_like: list[Any] | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(array_like)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[0] == 0:
        width = arr.shape[1] if arr.ndim > 1 else 1
        zeros = [0.0 for _ in range(width)]
        return {k: zeros for k in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")} | {"count": [0]}
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


def _write_data_parquet(root: Path, rows: list[dict[str, Any]]) -> None:
    fsl6 = pa.list_(pa.float32(), 6)
    table = pa.table(
        {
            "action": pa.array([r["action"] for r in rows], type=fsl6),
            "observation.state": pa.array([r["observation.state"] for r in rows], type=fsl6),
            "timestamp": pa.array([r["timestamp"] for r in rows], type=pa.float32()),
            "frame_index": pa.array([r["frame_index"] for r in rows], type=pa.int64()),
            "episode_index": pa.array([r["episode_index"] for r in rows], type=pa.int64()),
            "index": pa.array([r["index"] for r in rows], type=pa.int64()),
            "task_index": pa.array([r["task_index"] for r in rows], type=pa.int64()),
        }
    )
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")


def _write_tasks(root: Path, task_name: str) -> None:
    table = pa.table({"task_index": [0], "__index_level_0__": [task_name]})
    pq.write_table(table, root / "meta" / "tasks.parquet")


def _write_episodes(root: Path, episodes_meta: list[dict[str, Any]]) -> None:
    table = pa.table(
        {
            "episode_index": pa.array([e["episode_index"] for e in episodes_meta], type=pa.int64()),
            "tasks": pa.array([e["tasks"] for e in episodes_meta], type=pa.list_(pa.string())),
            "length": pa.array([e["length"] for e in episodes_meta], type=pa.int64()),
            "data/chunk_index": pa.array([0 for _ in episodes_meta], type=pa.int64()),
            "data/file_index": pa.array([0 for _ in episodes_meta], type=pa.int64()),
            "dataset_from_index": pa.array([e["dataset_from_index"] for e in episodes_meta], type=pa.int64()),
            "dataset_to_index": pa.array([e["dataset_to_index"] for e in episodes_meta], type=pa.int64()),
            **{
                f"videos/observation.images.{cam}/{name}": pa.array(
                    [e[f"videos/observation.images.{cam}/{name}"] for e in episodes_meta],
                    type=pa.float64() if "timestamp" in name else pa.int64(),
                )
                for cam in CAMERA_KEYS
                for name in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
            },
            "meta/episodes/chunk_index": pa.array([0 for _ in episodes_meta], type=pa.int64()),
            "meta/episodes/file_index": pa.array([0 for _ in episodes_meta], type=pa.int64()),
        }
    )
    pq.write_table(table, root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")


def _write_info(root: Path, total_episodes: int, total_frames: int) -> None:
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
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "fps": FPS,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    with (root / "meta" / "info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def _write_stats(root: Path, rows: list[dict[str, Any]], image_stats: dict[str, ImageStats]) -> None:
    stats = {
        "action": _numeric_stats([r["action"] for r in rows]),
        "observation.state": _numeric_stats([r["observation.state"] for r in rows]),
        "timestamp": _numeric_stats([r["timestamp"] for r in rows]),
        "frame_index": _numeric_stats([r["frame_index"] for r in rows]),
        "episode_index": _numeric_stats([r["episode_index"] for r in rows]),
        "index": _numeric_stats([r["index"] for r in rows]),
        "task_index": _numeric_stats([r["task_index"] for r in rows]),
    }
    for cam in CAMERA_KEYS:
        stats[f"observation.images.{cam}"] = image_stats[cam].to_json()
    with (root / "meta" / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def _reset_wrapper_env(env) -> Any:
    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        return reset_out[0]
    if reset_out is None:
        return env.get_observations()
    return reset_out


def _append_success_episode(
    episode: EpisodeRecord,
    episode_index: int,
    task_name: str,
    rows: list[dict[str, Any]],
    episodes_meta: list[dict[str, Any]],
    writers: dict[str, Any],
    image_stats: dict[str, ImageStats],
) -> None:
    start_index = len(rows)
    length = episode.length
    for frame_idx, frame in enumerate(episode.frames):
        rows.append(
            {
                "action": frame.action.tolist(),
                "observation.state": frame.state.tolist(),
                "timestamp": frame_idx / FPS,
                "frame_index": frame_idx,
                "episode_index": episode_index,
                "index": start_index + frame_idx,
                "task_index": 0,
            }
        )

    video_from = start_index / FPS
    video_to = (start_index + length) / FPS
    meta: dict[str, Any] = {
        "episode_index": episode_index,
        "tasks": [task_name],
        "length": length,
        "dataset_from_index": start_index,
        "dataset_to_index": start_index + length,
    }
    for cam in CAMERA_KEYS:
        meta[f"videos/observation.images.{cam}/chunk_index"] = 0
        meta[f"videos/observation.images.{cam}/file_index"] = 0
        meta[f"videos/observation.images.{cam}/from_timestamp"] = video_from
        meta[f"videos/observation.images.{cam}/to_timestamp"] = video_to

    for frame in episode.frames:
        for cam, image in frame.images.items():
            if cam in writers:
                writers[cam].append_data(image)
            image_stats[cam].update(image)

    episodes_meta.append(meta)


def main() -> None:
    root = Path(args.output_dir)
    env = None
    writers: dict[str, Any] = {}
    try:
        if args.num_envs != 1:
            raise ValueError("TC.1 recorder supports --num_envs 1 only; TC.2 will extend camera/env parallelism")
        if args.episodes <= 0:
            raise ValueError("--episodes must be positive")

        _prepare_output_dir(root, args.overwrite)
        root = root.resolve()
        if not args.no_videos:
            writers = _open_video_writers(root)

        device: str = args.device
        rl_device: str = args.rl_device if args.rl_device is not None else device
        task_name = _task_name(args.task)
        env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
        if hasattr(env_cfg, "seed"):
            env_cfg.seed = args.seed
        _apply_task_curriculum(env_cfg, args)
        policy_dt = env_cfg.sim.dt * env_cfg.decimation
        env_cfg.episode_length_s = args.max_episode_steps * policy_dt
        if not args.no_videos:
            _add_task_cameras(env_cfg, args.task)

        raw_env = gym.make(args.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(raw_env, clip_actions=args.clip_actions)

        torch.manual_seed(args.seed)
        if hasattr(env, "seed"):
            env.seed(args.seed)

        train_cfg = _build_train_cfg(args)
        runner = OnPolicyRunner(env, train_cfg, log_dir="", device=rl_device)
        try:
            runner.load(args.checkpoint, load_optimizer=False, map_location=rl_device)
        except TypeError:
            runner.load(args.checkpoint)

        if args.deterministic:
            policy = runner.get_inference_policy(device=rl_device)
        else:
            runner.eval_mode()
            runner.alg.policy.to(rl_device)
            policy = runner.alg.policy.act

        obs_dict = _reset_wrapper_env(env)
        zero_action = torch.zeros(args.num_envs, 6, device=device)
        for _ in range(max(0, args.warmup_steps)):
            obs_dict, _, _, _ = env.step(zero_action)
        obs_dict = _reset_wrapper_env(env)

        rows: list[dict[str, Any]] = []
        episodes_meta: list[dict[str, Any]] = []
        image_stats = {cam: ImageStats() for cam in CAMERA_KEYS}
        current = EpisodeRecord()
        attempts = 1
        successes = 0
        failures = 0
        max_attempts = args.max_attempts if args.max_attempts is not None else max(args.episodes * 20, args.episodes)

        while successes < args.episodes and attempts <= max_attempts:
            with torch.no_grad():
                actions = policy(obs_dict)

            frame = FrameRecord(
                action=_action_to_record(actions),
                state=_read_joint_state(raw_env),
                images={} if args.no_videos else _capture_images(raw_env),
            )
            current.frames.append(frame)

            obs_dict, _rewards, dones, infos = env.step(actions)
            if bool(dones[0].item()):
                time_outs = infos.get("time_outs", torch.zeros_like(dones))
                success = bool((dones.bool() & ~time_outs.bool())[0].item())
                if success and current.length > 0:
                    _append_success_episode(
                        current,
                        successes,
                        task_name,
                        rows,
                        episodes_meta,
                        writers,
                        image_stats,
                    )
                    successes += 1
                    print(
                        json.dumps(
                            {
                                "task_id": TASK_ID,
                                "event": "episode_recorded",
                                "successes": successes,
                                "attempts": attempts,
                                "frames": current.length,
                            }
                        ),
                        flush=True,
                    )
                else:
                    failures += 1
                current.clear()
                attempts += 1

        if successes < args.episodes:
            raise RuntimeError(
                f"Only recorded {successes}/{args.episodes} successful episodes "
                f"after {attempts - 1} attempts"
            )

        _close_video_writers(writers)
        writers = {}
        _write_data_parquet(root, rows)
        _write_tasks(root, task_name)
        _write_episodes(root, episodes_meta)
        _write_info(root, total_episodes=successes, total_frames=len(rows))
        _write_stats(root, rows, image_stats)

        result = {
            "task_id": TASK_ID,
            "status": "passed",
            "task": args.task,
            "checkpoint": args.checkpoint,
            "output_dir": str(root),
            "episodes": successes,
            "attempts": attempts - 1,
            "failures": failures,
            "total_frames": len(rows),
            "fps": FPS,
            "videos": not args.no_videos,
            "stochastic": not args.deterministic,
            "curriculum": {
                "active_objects": args.active_objects,
                "object_radius_scale": args.object_radius_scale,
                "container_angle_scale": args.container_angle_scale,
                "container_radius_scale": args.container_radius_scale,
            },
        }
        print(json.dumps(result), flush=True)

    except Exception as exc:
        tb = traceback.format_exc()
        print(
            json.dumps(
                {
                    "task_id": TASK_ID,
                    "status": "failed",
                    "error": str(exc),
                    "traceback": tb,
                }
            ),
            flush=True,
        )
        sys.exit(1)

    finally:
        _close_video_writers(writers)
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        simulation_app.close()


if __name__ == "__main__":
    main()
