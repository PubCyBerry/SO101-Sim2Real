"""Local GUI teleoperation for SimToReal-SO101 Pick&Place tasks (PickPen / PickCube).

This entry point intentionally keeps the old leisaac CLI shape, but the runtime
is pure Isaac Lab + LeRobot. It supports the SO-101 leader arm on Windows COM
ports, HDF5 action/state recording, and camera snapshot capture with the `c`
key while the Isaac GUI is open.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
import multiprocessing
import os
from pathlib import Path
import signal
import time
from typing import Callable
import weakref

from isaaclab.app import AppLauncher


if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)


CAMERA_NAMES = ("top_camera", "wrist_camera")
CAMERA_PRIM_PATHS = {
    "top_camera": "/World/envs/env_0/TopCamera",
    "wrist_camera": "/World/envs/env_0/Robot/gripper/WristCamera",
}
KEY_BINDINGS = {
    "Q": (0, 1.0, "shoulder_pan +"),
    "A": (0, -1.0, "shoulder_pan -"),
    "W": (1, 1.0, "shoulder_lift +"),
    "S": (1, -1.0, "shoulder_lift -"),
    "E": (2, 1.0, "elbow_flex +"),
    "D": (2, -1.0, "elbow_flex -"),
    "U": (3, 1.0, "wrist_flex +"),
    "J": (3, -1.0, "wrist_flex -"),
    "T": (4, 1.0, "wrist_roll +"),
    "G": (4, -1.0, "wrist_roll -"),
    "Y": (5, 1.0, "gripper open +"),
    "H": (5, -1.0, "gripper close -"),
}


def _parse_floats(text: str, count: int, name: str) -> tuple[float, ...]:
    values = [float(v) for v in text.replace(",", " ").split()]
    if len(values) != count:
        raise argparse.ArgumentTypeError(f"{name} expects {count} floats, got {len(values)}: {text!r}")
    return tuple(values)


def _vec3(text: str) -> tuple[float, float, float]:
    return _parse_floats(text, 3, "vec3")  # type: ignore[return-value]


def _vec6(text: str) -> tuple[float, float, float, float, float, float]:
    return _parse_floats(text, 6, "vec6")  # type: ignore[return-value]


def _quat(text: str) -> tuple[float, float, float, float]:
    return _parse_floats(text, 4, "quat")  # type: ignore[return-value]


parser = argparse.ArgumentParser(description="SO-101 pick-pen GUI teleoperation")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    choices=[
        "keyboard",
        "gamepad",
        "so101leader",
        "bi-so101leader",
        "lekiwi-keyboard",
        "lekiwi-gamepad",
        "lekiwi-leader",
    ],
    help="Device for interacting with environment",
)
parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="Port for so101leader")
parser.add_argument("--remote_endpoint", type=str, default=None, help="Reserved for old remote so101leader path")
parser.add_argument("--left_arm_port", type=str, default="/dev/ttyACM0")
parser.add_argument("--right_arm_port", type=str, default="/dev/ttyACM1")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed for the environment.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Keyboard sensitivity factor.")
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument("--record", action="store_true", help="Enable lightweight HDF5 action/state recording")
parser.add_argument("--dataset_file", type=str, default="./datasets/dataset.hdf5", help="HDF5 recording path")
parser.add_argument("--resume", action="store_true", help="Append to an existing dataset file")
parser.add_argument("--num_demos", type=int, default=0, help="Number of demonstrations to record. 0 = infinite.")
parser.add_argument("--max_steps", type=int, default=0, help="Maximum GUI loop iterations. 0 = infinite.")
parser.add_argument("--recalibrate", action="store_true", help="Allow interactive SO-101 leader calibration")
parser.add_argument("--quality", action="store_true", help="Enable quality render mode.")
parser.add_argument(
    "--tune_cameras",
    action="store_true",
    help="카메라 보정 모드: top/wrist viewport 분할 docking + 실시간 튜너 위젯을 띄운다. "
    "미지정 시 메인 viewport 만 렌더해 실시간 제어 성능을 확보한다(카메라 sensor 는 30fps 유지).",
)
parser.add_argument("--use_lerobot_recorder", action="store_true", help="Accepted for CLI compatibility; ignored.")
parser.add_argument("--lerobot_dataset_repo_id", type=str, default=None, help="Accepted for CLI compatibility; ignored.")
parser.add_argument("--lerobot_dataset_fps", type=int, default=30, help="Recording metadata FPS.")
parser.add_argument("--capture_dir", type=Path, default=Path("outputs/captured_images"))
parser.add_argument("--capture_on_start", action="store_true", help="Capture camera images immediately after reset.")
parser.add_argument("--leader_id", default="so101_teleop", help="LeRobot calibration id for SO-101 leader")
parser.add_argument(
    "--leader_gripper_divisor",
    type=float,
    default=100.0,
    help="(legacy, 미사용) 그리퍼는 이제 follower joint-limit affine 매핑(-10°~100°)을 쓴다.",
)
parser.add_argument("--leader_joint_signs", type=_vec6, default=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
parser.add_argument("--leader_joint_offsets", type=_vec6, default=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
parser.add_argument("--leader_smoothing", type=float, default=0.0)
parser.add_argument(
    "--max_arm_speed",
    type=float,
    default=0.20,
    help="Max commanded arm target speed in rad/s of sim time. <=0 disables controller-side limiting.",
)
parser.add_argument(
    "--max_gripper_speed",
    type=float,
    default=0.20,
    help="Max commanded gripper target speed in rad/s of sim time. <=0 disables controller-side limiting.",
)
parser.add_argument("--joint_step", type=float, default=0.035, help="Keyboard arm joint step in radians")
parser.add_argument("--gripper_step", type=float, default=0.05, help="Keyboard gripper joint step")

# Camera overrides. Defaults live in pick_pen_env_cfg.py.
parser.add_argument("--top_pos", type=_vec3, default=None, help="x,y,z world position")
parser.add_argument("--top_target", type=_vec3, default=None, help="x,y,z world look-at target")
parser.add_argument("--top_focal", type=float, default=None, help="top focal length in mm")
parser.add_argument("--wrist_pos", type=_vec3, default=None, help="x,y,z gripper-local position")
parser.add_argument("--wrist_rot", type=_quat, default=None, help="w,x,y,z gripper-local quaternion")
parser.add_argument("--wrist_focal", type=float, default=None, help="wrist focal length in mm")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs != 1:
    raise ValueError("This local GUI teleop script currently supports --num_envs=1 only.")
if args_cli.teleop_device not in {"keyboard", "so101leader"}:
    raise NotImplementedError(
        f"{args_cli.teleop_device!r} was a leisaac device path and has not been ported yet. "
        "Use --teleop_device=so101leader or keyboard for the registered task."
    )
if args_cli.remote_endpoint:
    raise NotImplementedError("--remote_endpoint is not available in the pure Isaac Lab local teleop path.")
if args_cli.enable_cameras and not args_cli.experience:
    # Windows Isaac Sim 5.1 camera rendering is more stable with this experience.
    args_cli.experience = "isaaclab.python.rendering.kit"
simulation_app = AppLauncher(vars(args_cli)).app

import carb  # noqa: E402
import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import omni.appwindow  # noqa: E402
from PIL import Image  # noqa: E402
import torch  # noqa: E402

import sim_to_real  # noqa: E402,F401  # registers the gym env
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from sim_to_real.tasks.pick_pen.pick_pen_env_cfg import (  # noqa: E402
    SO101_JOINT_ORDER,
    add_pick_pen_cameras,
)

# Task별 카메라 주입 함수 import (PickCube는 향후 추가됨)
try:
    from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import add_pick_cube_cameras  # noqa: E402
except ImportError:
    add_pick_cube_cameras = None  # PickCube 모듈이 아직 없을 수 있음


class RateLimiter:
    def __init__(self, hz: float) -> None:
        self.period = 1.0 / max(hz, 1e-6)
        self.next_time = time.perf_counter() + self.period
        self.render_period = min(0.0166, self.period)

    def sleep(self, env) -> None:
        while time.perf_counter() < self.next_time:
            time.sleep(self.render_period)
            env.sim.render()
        self.next_time = max(self.next_time + self.period, time.perf_counter())


class GuiKeyboard:
    def __init__(self) -> None:
        self.started = False
        self._callbacks: dict[str, Callable[[], None]] = {}
        self._key_listeners: list[Callable[[str], None]] = []
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

    def close(self) -> None:
        if getattr(self, "_keyboard_sub", None) is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def add_callback(self, key: str, func: Callable[[], None]) -> None:
        self._callbacks[key.upper()] = func

    def add_key_listener(self, func: Callable[[str], None]) -> None:
        self._key_listeners.append(func)

    def _on_keyboard_event(self, event, *_args) -> None:
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return
        key = event.input.name.upper()
        if key == "B":
            self.started = True
            print("[teleop] started")
            return
        elif key in self._callbacks:
            self._callbacks[key]()
            return
        for listener in self._key_listeners:
            listener(key)

    def display_controls(self, teleop_device: str) -> None:
        print(f"\nTeleoperation Controls for {teleop_device}")
        print("  B: start control")
        print("  R: reset simulation and mark current recording as failure")
        print("  N: reset simulation and mark current recording as success")
        print("  c: save top/wrist camera PNGs + camera metadata JSON")
        print("  Ctrl+C or close Isaac window: quit")
        if teleop_device == "keyboard":
            print("  Keyboard joint controls: q/a, w/s, e/d, u/j, t/g, y/h")
        else:
            print("  SO-101 Leader: move the physical leader arm on the configured serial port")


# LeRobot SO101 leader 의 그리퍼는 use_degrees 와 무관하게 항상 RANGE_0_100
# (0=열림 .. 100=닫힘) 로 읽힌다(lerobot SOLeader 소스 확인). 팔 관절은
# use_degrees=True 라 degree 로 읽혀 radians() 로 바로 변환한다.
_LEADER_GRIPPER_RANGE: tuple[float, float] = (0.0, 100.0)
# Follower USD(assets/robots/so101_follower.usd) gripper revolute joint limit(deg).
# leisaac SO101_FOLLOWER_USD_JOINT_LIMITS["gripper"] 와 동일. leader 0..100 을 이
# 범위로 affine 매핑해야 leader 를 끝까지 닫았을 때 follower 도 100°(=1.745 rad)
# 까지 닫힌다.
_FOLLOWER_GRIPPER_JOINT_DEG: tuple[float, float] = (-10.0, 100.0)


def _joint_limits(device: str) -> torch.Tensor:
    limits = [
        (-math.pi, math.pi),
        (-math.pi, math.pi),
        (-math.pi, math.pi),
        (-math.pi, math.pi),
        (-math.pi, math.pi),
        (math.radians(_FOLLOWER_GRIPPER_JOINT_DEG[0]), math.radians(_FOLLOWER_GRIPPER_JOINT_DEG[1])),
    ]
    return torch.tensor(limits, dtype=torch.float32, device=device)


def _policy_dt(env) -> float:
    return float(env.cfg.sim.dt * env.cfg.decimation)


def _slew_limited_targets(
    current: torch.Tensor,
    desired: torch.Tensor,
    limits: torch.Tensor,
    *,
    policy_dt: float,
) -> torch.Tensor:
    delta = desired - current
    if args_cli.max_arm_speed > 0.0:
        arm_step = float(args_cli.max_arm_speed) * policy_dt
        delta[:5] = torch.clamp(delta[:5], -arm_step, arm_step)
    if args_cli.max_gripper_speed > 0.0:
        gripper_step = float(args_cli.max_gripper_speed) * policy_dt
        delta[5] = torch.clamp(delta[5], -gripper_step, gripper_step)
    next_targets = current + delta
    return torch.maximum(torch.minimum(next_targets, limits[:, 1]), limits[:, 0])


class KeyboardJointController:
    def __init__(self, env, keyboard: GuiKeyboard, limits: torch.Tensor) -> None:
        self.env = env
        self.keyboard = keyboard
        self.limits = limits
        self.policy_dt = _policy_dt(env)
        self.desired_targets = env.scene["robot"].data.joint_pos[0, :6].clone().to(env.device)
        self.targets = self.desired_targets.clone()
        keyboard.add_key_listener(self._on_key)

    def _on_key(self, key: str) -> None:
        if not self.keyboard.started or key not in KEY_BINDINGS:
            return
        joint_id, direction, label = KEY_BINDINGS[key]
        delta = args_cli.gripper_step if joint_id == 5 else args_cli.joint_step
        self.desired_targets[joint_id] += direction * delta * args_cli.sensitivity
        self.desired_targets = torch.maximum(torch.minimum(self.desired_targets, self.limits[:, 1]), self.limits[:, 0])
        print(f"[key] {label}: {self.desired_targets[joint_id].item():+.4f}")

    def reset(self) -> None:
        self.desired_targets = self.env.scene["robot"].data.joint_pos[0, :6].clone().to(self.env.device)
        self.targets = self.desired_targets.clone()

    def close(self) -> None:
        pass

    def advance(self) -> torch.Tensor | None:
        if not self.keyboard.started:
            return None
        self.targets = _slew_limited_targets(
            self.targets,
            self.desired_targets,
            self.limits,
            policy_dt=self.policy_dt,
        )
        return self.targets.unsqueeze(0)


class SO101LeaderJointController:
    def __init__(self, env, keyboard: GuiKeyboard, limits: torch.Tensor) -> None:
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

        self.env = env
        self.keyboard = keyboard
        self.limits = limits
        self.signs = torch.tensor(args_cli.leader_joint_signs, dtype=torch.float32, device=env.device)
        self.offsets = torch.tensor(args_cli.leader_joint_offsets, dtype=torch.float32, device=env.device)
        self.targets = env.scene["robot"].data.joint_pos[0, :6].clone().to(env.device)
        self.filtered_targets = self.targets.clone()
        self.policy_dt = _policy_dt(env)
        cfg = SO101LeaderConfig(port=args_cli.port, id=args_cli.leader_id, use_degrees=True)
        self.teleop = SO101Leader(cfg)
        print(f"[leader] connecting SO-101 leader on {args_cli.port} (id={args_cli.leader_id})")
        self.teleop.connect(calibrate=args_cli.recalibrate)
        if not self.teleop.is_calibrated:
            raise RuntimeError(
                "SO-101 leader calibration is missing or mismatched. Re-run with --recalibrate, "
                "or calibrate the leader with LeRobot first."
            )
        print("[leader] connected")

    def reset(self) -> None:
        self.targets = self.env.scene["robot"].data.joint_pos[0, :6].clone().to(self.env.device)
        self.filtered_targets = self.targets.clone()

    def close(self) -> None:
        if self.teleop.is_connected:
            self.teleop.disconnect()

    def _read_leader_targets(self) -> torch.Tensor:
        raw = self.teleop.get_action()
        values = []
        for joint_id, joint_name in enumerate(SO101_JOINT_ORDER):
            value = float(raw[f"{joint_name}.pos"])
            if joint_id < 5:
                value = math.radians(value)
            else:
                # leader gripper(0..100) → follower USD gripper joint(-10°..100°) → rad.
                # leisaac convert_action_from_so101_leader 와 동일한 affine 매핑.
                # (이전 value/leader_gripper_divisor 는 [0,1] rad(≈57°)에 캡돼
                #  leader 를 끝까지 닫아도 그리퍼가 57°까지만 닫혔다.)
                lo_m, hi_m = _LEADER_GRIPPER_RANGE
                lo_j, hi_j = _FOLLOWER_GRIPPER_JOINT_DEG
                frac = (value - lo_m) / (hi_m - lo_m)
                value = math.radians(frac * (hi_j - lo_j) + lo_j)
            values.append(value)
        targets = torch.tensor(values, dtype=torch.float32, device=self.env.device)
        targets = targets * self.signs + self.offsets
        return torch.maximum(torch.minimum(targets, self.limits[:, 1]), self.limits[:, 0])

    def advance(self) -> torch.Tensor | None:
        if not self.keyboard.started:
            return None
        leader_targets = self._read_leader_targets()
        smoothing = min(max(float(args_cli.leader_smoothing), 0.0), 0.99)
        if smoothing > 0.0:
            self.filtered_targets = smoothing * self.filtered_targets + (1.0 - smoothing) * leader_targets
        else:
            self.filtered_targets = leader_targets
        self.targets = _slew_limited_targets(
            self.targets,
            self.filtered_targets,
            self.limits,
            policy_dt=self.policy_dt,
        )
        return self.targets.unsqueeze(0)


@dataclass
class FrameRecord:
    action: np.ndarray
    state: np.ndarray
    timestamp: float


@dataclass
class Hdf5TeleopRecorder:
    path: Path
    fps: int
    frames: list[FrameRecord] = field(default_factory=list)
    exported_successful_episode_count: int = 0
    episode_index: int = 0

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            with h5py.File(self.path, "a") as h5:
                existing = [int(k.split("_")[-1]) for k in h5.keys() if k.startswith("demo_")]
                self.episode_index = max(existing, default=-1) + 1
                self.exported_successful_episode_count = sum(
                    1 for k in h5.keys() if k.startswith("demo_") and bool(h5[k].attrs.get("success", False))
                )
            if not args_cli.resume:
                print(f"[record] appending to existing dataset {self.path}; pass a new path for a fresh file")
        else:
            with h5py.File(self.path, "w") as h5:
                h5.attrs["format"] = "sim_to_real_hdf5_teleop_v1"
                h5.attrs["fps"] = int(self.fps)
                h5.attrs["joint_names"] = json.dumps(SO101_JOINT_ORDER)

    def record_step(self, env, action: torch.Tensor) -> None:
        robot = env.scene["robot"]
        state = robot.data.joint_pos[0, :6].detach().cpu().numpy().astype(np.float32)
        action_np = action[0].detach().cpu().numpy().astype(np.float32)
        timestamp = len(self.frames) / float(self.fps)
        self.frames.append(FrameRecord(action=action_np, state=state, timestamp=timestamp))

    def finalize_episode(self, success: bool) -> None:
        if not self.frames:
            return
        with h5py.File(self.path, "a") as h5:
            group = h5.create_group(f"demo_{self.episode_index:06d}")
            group.create_dataset(
                "action",
                data=np.stack([f.action for f in self.frames]),
                compression="lzf",
            )
            group.create_dataset(
                "observation.state",
                data=np.stack([f.state for f in self.frames]),
                compression="lzf",
            )
            group.create_dataset(
                "timestamp",
                data=np.asarray([f.timestamp for f in self.frames], dtype=np.float32),
                compression="lzf",
            )
            group.attrs["success"] = bool(success)
            group.attrs["num_frames"] = len(self.frames)
            group.attrs["task"] = args_cli.task or ""
            group.attrs["joint_names"] = json.dumps(SO101_JOINT_ORDER)
        if success:
            self.exported_successful_episode_count += 1
        print(
            f"[record] wrote demo_{self.episode_index:06d} "
            f"frames={len(self.frames)} success={success} path={self.path}"
        )
        self.episode_index += 1
        self.frames.clear()

    def close(self) -> None:
        self.finalize_episode(success=False)


def _rgb_to_u8(rgb: torch.Tensor) -> np.ndarray:
    arr = rgb[0].detach().cpu().numpy()
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _quat_wxyz_to_euler_xyz_degrees(quat: list[float]) -> list[float]:
    w, x, y, z = quat
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return [math.degrees(v) for v in (roll, pitch, yaw)]


def _euler_xyz_deg_to_quat_wxyz(roll_deg: float, pitch_deg: float, yaw_deg: float) -> list[float]:
    """_quat_wxyz_to_euler_xyz_degrees 의 역변환 (Tait-Bryan XYZ, intrinsic)."""
    r = math.radians(roll_deg) * 0.5
    p = math.radians(pitch_deg) * 0.5
    y = math.radians(yaw_deg) * 0.5
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    yq = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return [w, x, yq, z]


def _camera_metadata(env) -> dict:
    metadata: dict[str, dict] = {}
    for name in CAMERA_NAMES:
        try:
            cam = env.scene[name]
        except KeyError:
            continue
        item: dict[str, object] = {"prim_path": CAMERA_PRIM_PATHS.get(name, "")}
        try:
            item["cfg_prim_path"] = cam.cfg.prim_path
        except Exception:
            pass
        try:
            item["local_pos"] = list(cam.cfg.offset.pos)
            item["local_rot_wxyz"] = list(cam.cfg.offset.rot)
        except Exception:
            pass
        try:
            item["pos_w"] = cam.data.pos_w[0].detach().cpu().tolist()
        except Exception:
            pass
        try:
            if hasattr(cam.data, "quat_w_world"):
                quat = cam.data.quat_w_world[0].detach().cpu().tolist()
            else:
                quat = cam.data.quat_w_ros[0].detach().cpu().tolist()
            item["rot_w_wxyz"] = quat
            item["rot_euler_xyz_deg"] = _quat_wxyz_to_euler_xyz_degrees(quat)
        except Exception:
            pass
        try:
            intr = cam.data.intrinsic_matrices[0].detach().cpu().tolist()
            item["intrinsics_3x3"] = intr
            width = float(getattr(cam.cfg, "width", 640))
            height = float(getattr(cam.cfg, "height", 480))
            fx = float(intr[0][0])
            fy = float(intr[1][1])
            item["fov_horizontal_deg"] = math.degrees(2.0 * math.atan(width / (2.0 * fx)))
            item["fov_vertical_deg"] = math.degrees(2.0 * math.atan(height / (2.0 * fy)))
        except Exception:
            pass
        try:
            item["focal_length_mm"] = float(cam.cfg.spawn.focal_length)
        except Exception:
            pass
        metadata[name] = item
    return metadata


def capture_camera_views(env, capture_dir: Path) -> None:
    capture_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    saved: dict[str, str] = {}
    env.sim.render()
    for name in CAMERA_NAMES:
        try:
            cam = env.scene[name]
        except KeyError:
            continue
        # 키 입력 없이 step 이 멈춰 있는 동안에도 capture 가 최신 프레임을 받도록 강제 갱신.
        try:
            cam.update(dt=0.0, force_recompute=True)
        except Exception:
            pass
        if "rgb" not in cam.data.output:
            continue
        image = _rgb_to_u8(cam.data.output["rgb"])
        path = capture_dir / f"{stamp}_{name}.png"
        Image.fromarray(image).save(path)
        saved[name] = str(path)
    meta = {
        "timestamp": stamp,
        "saved": saved,
        "cameras": _camera_metadata(env),
        "cli_overrides": {
            "top_pos": args_cli.top_pos,
            "top_target": args_cli.top_target,
            "top_focal": args_cli.top_focal,
            "wrist_pos": args_cli.wrist_pos,
            "wrist_rot": args_cli.wrist_rot,
            "wrist_focal": args_cli.wrist_focal,
        },
    }
    meta_path = capture_dir / f"{stamp}_camera_metadata.json"
    latest_path = capture_dir / "latest_camera_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[capture] saved {len(saved)} camera images + metadata to {capture_dir.resolve()}")


def create_camera_viewports() -> list[object]:
    """2개 센서 카메라 viewport 를 메인 Perspective 뷰포트와 함께 수직 분할로 docking.

    레이아웃::

        ┌─────────────┬─────────────┐
        │ Perspective │ Top Cam     │
        │             ├─────────────┤
        │             │ Wrist Cam   │
        └─────────────┴─────────────┘
    """

    if args_cli.headless or not args_cli.enable_cameras:
        return []
    try:
        import omni.kit.app
        import omni.ui as ui
        from pxr import Sdf

        ext_manager = omni.kit.app.get_app().get_extension_manager()
        for ext_name in ("omni.kit.viewport.window", "omni.kit.viewport.utility"):
            is_enabled = False
            try:
                is_enabled = bool(ext_manager.is_extension_enabled(ext_name))
            except Exception:
                pass
            if not is_enabled:
                ext_manager.set_extension_enabled_immediate(ext_name, True)
        from omni.kit.viewport.utility import create_viewport_window
    except Exception as exc:
        print(f"[viewport] camera viewport windows unavailable: {exc}")
        return []

    # 카메라 viewport window 2개 생성. 위치/크기는 아래 docking 으로 덮어쓰므로
    # floating 좌표를 지정하지 않는다.
    created: dict[str, object] = {}
    windows: list[object] = []
    for title, camera_name in (
        ("Top Camera", "top_camera"),
        ("Wrist Camera", "wrist_camera"),
    ):
        prim_path = CAMERA_PRIM_PATHS[camera_name]
        try:
            window = create_viewport_window(
                name=f"SO101 {title}",
                camera_path=Sdf.Path(prim_path),
            )
            created[camera_name] = window
            windows.append(window)
            print(f"[viewport] opened {title}: {prim_path}")
        except Exception as exc:
            print(f"[viewport] failed to open {title} ({prim_path}): {exc}")

    # 메인 Perspective viewport + 카메라 2개를 수직 분할로 docking.
    #   top   → 메인 오른쪽 절반  (좌:메인,   우:top)
    #   wrist → top 아래 절반     (우상:top, 우하:wrist)
    try:
        app = omni.kit.app.get_app()
        for _ in range(3):
            app.update()  # 새 window 들이 dock space 에 mount 될 시간을 준다
        main_vp = ui.Workspace.get_window("Viewport")
        top = created.get("top_camera")
        wrist = created.get("wrist_camera")
        if main_vp is not None and top is not None:
            top.dock_in(main_vp, ui.DockPosition.RIGHT, 0.5)
        if top is not None and wrist is not None:
            wrist.dock_in(top, ui.DockPosition.BOTTOM, 0.5)
        for _ in range(3):
            app.update()
        print("[viewport] docked cameras (L=Perspective, RT=Top, RB=Wrist)")
    except Exception as exc:
        print(f"[viewport] docking failed (windows remain floating): {exc}")

    return windows


def create_camera_tuner(env) -> object | None:
    """omni.ui 패널로 top/wrist 카메라의 위치·회전(deg)·focal 을 실시간 조정.

    슬라이더를 움직이면 해당 카메라 USD prim 의 local transform(translate/orient)과
    focalLength 가 즉시 갱신되어 viewport 에 바로 반영된다. wrist 는 부모
    링크(gripper) 기준 local, top 은 env(거의 world) 기준.

    'Print cfg values' 버튼은 현재 값을 pick_cube_env_cfg.py 의
    _TOP_*/_WRIST_CAM_* 형식(pos + world-convention wxyz quat + focal)
    으로 콘솔에 출력한다. 회전은 prim(opengl) → Isaac Lab world convention 으로
    변환해 출력하므로 그대로 cfg 상수에 붙여넣을 수 있다.
    """
    if args_cli.headless or not args_cli.enable_cameras:
        return None
    try:
        import omni.ui as ui
        import omni.usd
        from pxr import Gf, Usd, UsdGeom
    except Exception as exc:
        print(f"[tuner] camera tuner unavailable: {exc}")
        return None

    # prim(opengl) → Isaac Lab world convention quaternion 변환 (가능하면).
    try:
        import torch as _torch
        from isaaclab.utils.math import convert_camera_frame_orientation_convention

        def _to_world_quat(wxyz: list[float]) -> list[float]:
            q = _torch.tensor([list(wxyz)], dtype=_torch.float32)
            out = convert_camera_frame_orientation_convention(q, origin="opengl", target="world")
            return [round(float(v), 4) for v in out[0].tolist()]
    except Exception:
        def _to_world_quat(wxyz: list[float]) -> list[float]:
            return [round(float(v), 4) for v in wxyz]

    stage = omni.usd.get_context().get_stage()

    def _read_prim(prim_path: str):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return None
        m = UsdGeom.Xformable(prim).GetLocalTransformation(Usd.TimeCode.Default())
        t = m.ExtractTranslation()
        q = m.ExtractRotationQuat()
        im = q.GetImaginary()
        euler = _quat_wxyz_to_euler_xyz_degrees([q.GetReal(), im[0], im[1], im[2]])
        focal = 14.0
        try:
            focal = float(UsdGeom.Camera(prim).GetFocalLengthAttr().Get())
        except Exception:
            pass
        return [t[0], t[1], t[2]], euler, focal

    def _apply(prim_path: str, pos, euler, focal: float) -> None:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
        w, x, y, z = _euler_xyz_deg_to_quat_wxyz(euler[0], euler[1], euler[2])
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(float(w), Gf.Vec3d(float(x), float(y), float(z)))
        )
        try:
            UsdGeom.Camera(prim).GetFocalLengthAttr().Set(float(focal))
        except Exception:
            pass

    specs = [
        ("Top Camera", "top_camera", CAMERA_PRIM_PATHS["top_camera"], "world"),
        ("Wrist Camera", "wrist_camera", CAMERA_PRIM_PATHS["wrist_camera"], "gripper-local"),
    ]
    state: dict[str, dict] = {}
    window = ui.Window("SO101 Camera Tuner", width=430, height=640)

    def _make_change_fn(key: str):
        def _fn(*_a) -> None:
            s = state[key]
            pos = [s["px"].get_value_as_float(), s["py"].get_value_as_float(), s["pz"].get_value_as_float()]
            euler = [s["rx"].get_value_as_float(), s["ry"].get_value_as_float(), s["rz"].get_value_as_float()]
            _apply(s["prim"], pos, euler, s["fc"].get_value_as_float())
        return _fn

    with window.frame:
        with ui.ScrollingFrame():
            with ui.VStack(spacing=6, height=0):
                for title, scene_name, prim_path, frame_label in specs:
                    read = _read_prim(prim_path)
                    if read is None:
                        ui.Label(f"{title}: prim not found ({prim_path})")
                        continue
                    pos0, eul0, focal0 = read
                    px = ui.SimpleFloatModel(pos0[0])
                    py = ui.SimpleFloatModel(pos0[1])
                    pz = ui.SimpleFloatModel(pos0[2])
                    rx = ui.SimpleFloatModel(eul0[0])
                    ry = ui.SimpleFloatModel(eul0[1])
                    rz = ui.SimpleFloatModel(eul0[2])
                    fc = ui.SimpleFloatModel(focal0)
                    state[scene_name] = {
                        "prim": prim_path, "title": title, "frame": frame_label,
                        "px": px, "py": py, "pz": pz, "rx": rx, "ry": ry, "rz": rz, "fc": fc,
                    }
                    change = _make_change_fn(scene_name)
                    for mdl in (px, py, pz, rx, ry, rz, fc):
                        mdl.add_value_changed_fn(change)
                    with ui.CollapsableFrame(f"{title}  [{frame_label}]"):
                        with ui.VStack(spacing=3, height=0):
                            for lbl, mdl, lo, hi, st in (
                                ("Pos X (m)", px, -3.0, 3.0, 0.005),
                                ("Pos Y (m)", py, -3.0, 3.0, 0.005),
                                ("Pos Z (m)", pz, -1.0, 3.0, 0.005),
                                ("Rot X (deg)", rx, -180.0, 180.0, 0.5),
                                ("Rot Y (deg)", ry, -180.0, 180.0, 0.5),
                                ("Rot Z (deg)", rz, -180.0, 180.0, 0.5),
                                ("Focal (mm)", fc, 4.0, 80.0, 0.5),
                            ):
                                with ui.HStack(height=22):
                                    ui.Label(lbl, width=90)
                                    ui.FloatDrag(model=mdl, min=lo, max=hi, step=st)

                def _print_cfg(*_a) -> None:
                    print("\n[tuner] ===== camera cfg values (paste into pick_cube_env_cfg.py) =====")
                    for title, scene_name, prim_path, frame_label in specs:
                        s = state.get(scene_name)
                        if s is None:
                            continue
                        pos = (s["px"].get_value_as_float(), s["py"].get_value_as_float(), s["pz"].get_value_as_float())
                        euler = (s["rx"].get_value_as_float(), s["ry"].get_value_as_float(), s["rz"].get_value_as_float())
                        world_q = _to_world_quat(_euler_xyz_deg_to_quat_wxyz(*euler))
                        print(f"  # {title} [{frame_label}]")
                        print(f"  pos          = ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
                        print(f"  rot_xyz_deg  = ({euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f})  # 위젯 슬라이더 Rot X/Y/Z (prim frame)")
                        print(f"  rot_quat     = ({world_q[0]:.4f}, {world_q[1]:.4f}, {world_q[2]:.4f}, {world_q[3]:.4f})  # wxyz, world-conv (cfg 상수용)")
                        print(f"  focal        = {s['fc'].get_value_as_float():.2f}")
                    print("[tuner] ===============================================================\n")

                ui.Button("Print cfg values to console", height=30, clicked_fn=_print_cfg)

    print("[tuner] camera tuner panel opened — adjust sliders for live update, then 'Print cfg values'")
    return window


def _set_initial_view() -> None:
    """메인 Perspective viewport 카메라를 책상 작업공간이 보이는 초기 구도로 맞춘다.

    로봇 base 가 화면 오른쪽, 그릇/큐브가 가운데~왼쪽에 오는 비스듬한 부감 뷰.
    viewport 는 인터랙티브이므로 사용자가 마우스로 자유롭게 다시 돌릴 수 있다.
    """
    if args_cli.headless:
        return
    try:
        from isaacsim.core.utils.viewports import set_camera_view

        set_camera_view(eye=[1.60, -1.20, 1.20], target=[2.18, -0.30, 0.79])
        print("[view] initial viewport camera set (eye=[1.60,-1.20,1.20] target=[2.18,-0.30,0.79])")
    except Exception as exc:
        print(f"[view] initial camera view skipped: {exc}")


def _disable_episode_termination(env_cfg) -> None:
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None


def main() -> None:  # noqa: C901
    output_dir = os.path.dirname(args_cli.dataset_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    _disable_episode_termination(env_cfg)
    if args_cli.quality:
        env_cfg.sim.render.antialiasing_mode = "FXAA"
        env_cfg.sim.render.rendering_mode = "quality"
    if args_cli.enable_cameras:
        # Task 이름에 따라 카메라 주입 함수 선택 (PickCube → add_pick_cube_cameras, 그 외 → add_pick_pen_cameras)
        if args_cli.task and "Cube" in args_cli.task and add_pick_cube_cameras is not None:
            add_cameras_fn = add_pick_cube_cameras
        else:
            add_cameras_fn = add_pick_pen_cameras

        add_cameras_fn(
            env_cfg.scene,
            top_pos=args_cli.top_pos,
            top_target=args_cli.top_target,
            top_focal=args_cli.top_focal,
            wrist_local_pos=args_cli.wrist_pos,
            wrist_local_rot=args_cli.wrist_rot,
            wrist_focal=args_cli.wrist_focal,
        )
        # NOTE: 카메라 sensor update_period 는 건드리지 않는다(0.0 유지).
        # env 가 sim.dt=1/120·decimation=4·render_interval=4 라 카메라는 30 fps 로
        # 갱신되며, 이는 North Star observation.images.* fps 30 계약과 일치한다.
        # 실시간 성능은 보조 viewport docking 을 --tune_cameras 일 때만 켜서 확보한다.

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    camera_viewports: list[object] = []
    keyboard: GuiKeyboard | None = None
    controller: KeyboardJointController | SO101LeaderJointController | None = None
    recorder: Hdf5TeleopRecorder | None = None
    rate_limiter = RateLimiter(args_cli.step_hz)
    should_reset = False
    should_capture = bool(args_cli.capture_on_start)
    reset_success = False
    interrupted = False
    loop_count = 0

    def request_reset(success: bool) -> None:
        nonlocal should_reset, reset_success
        should_reset = True
        reset_success = success

    def request_capture() -> None:
        nonlocal should_capture
        should_capture = True

    def signal_handler(_signum, _frame) -> None:
        nonlocal interrupted
        interrupted = True
        print("\n[INFO] Ctrl+C detected. Cleaning up resources...")

    original_sigint_handler = signal.signal(signal.SIGINT, signal_handler)

    try:
        env.reset()
        env.sim.render()
        _set_initial_view()
        camera_tuner = None  # keep ref alive (GC 방지)
        if args_cli.enable_cameras and args_cli.tune_cameras:
            # 카메라 보정 모드에서만 2x2 docking viewport + 실시간 튜너 위젯을 띄운다.
            # (평상시엔 메인 viewport 만 렌더해 실시간 제어 성능을 확보)
            camera_viewports = create_camera_viewports()
            camera_tuner = create_camera_tuner(env)  # noqa: F841
        limits = _joint_limits(env.device)
        keyboard = GuiKeyboard()
        keyboard.add_callback("R", lambda: request_reset(False))
        keyboard.add_callback("N", lambda: request_reset(True))
        keyboard.add_callback("C", request_capture)
        keyboard.display_controls(args_cli.teleop_device)
        print(
            "[teleop] command speed limit: "
            f"arm<={args_cli.max_arm_speed:.3f} rad/s, "
            f"gripper<={args_cli.max_gripper_speed:.3f} rad/s, "
            f"loop={args_cli.step_hz} Hz"
        )

        if args_cli.teleop_device == "keyboard":
            controller = KeyboardJointController(env, keyboard, limits)
        else:
            controller = SO101LeaderJointController(env, keyboard, limits)

        if args_cli.record:
            recorder = Hdf5TeleopRecorder(Path(args_cli.dataset_file), fps=args_cli.lerobot_dataset_fps)

        while simulation_app.is_running() and not interrupted:
            with torch.inference_mode():
                if should_capture:
                    capture_camera_views(env, args_cli.capture_dir)
                    should_capture = False

                if should_reset:
                    if recorder is not None:
                        recorder.finalize_episode(success=reset_success)
                        if (
                            args_cli.num_demos > 0
                            and recorder.exported_successful_episode_count >= args_cli.num_demos
                        ):
                            print(f"[record] all {args_cli.num_demos} successful demonstrations recorded")
                            break
                    env.reset()
                    if controller is not None:
                        controller.reset()
                    keyboard.started = False
                    should_reset = False
                    continue

                action = controller.advance() if controller is not None else None
                if action is None:
                    env.sim.render()
                else:
                    env.step(action)
                    if recorder is not None:
                        recorder.record_step(env, action)
                rate_limiter.sleep(env)
                loop_count += 1
                if args_cli.max_steps > 0 and loop_count >= args_cli.max_steps:
                    print(f"[teleop] max_steps={args_cli.max_steps} reached")
                    break
    finally:
        signal.signal(signal.SIGINT, original_sigint_handler)
        if recorder is not None:
            recorder.close()
        if controller is not None:
            controller.close()
        if keyboard is not None:
            keyboard.close()
        camera_viewports.clear()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
