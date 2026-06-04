"""Local GUI joint teleop and camera tuning for SimToReal-SO101-PickPen-v0.

This is intentionally independent from the old LeIsaac teleop device layer.
It directly sends 6-dim joint-position actions to the current Isaac Lab env,
which makes it useful for scene/camera debugging before another rollout run.

Example:
    uv run --group isaac --locked python scripts/environments/teleoperation/pick_pen_joint_teleop.py \
        --task SimToReal-SO101-PickPen-v0 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time
import traceback

from isaaclab.app import AppLauncher


if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)


CAMERA_NAMES = ("top_camera", "front_camera", "wrist_camera")
KEY_BINDINGS = {
    "q": (0, 1.0, "shoulder_pan +"),
    "a": (0, -1.0, "shoulder_pan -"),
    "w": (1, 1.0, "shoulder_lift +"),
    "s": (1, -1.0, "shoulder_lift -"),
    "e": (2, 1.0, "elbow_flex +"),
    "d": (2, -1.0, "elbow_flex -"),
    "r": (3, 1.0, "wrist_flex +"),
    "f": (3, -1.0, "wrist_flex -"),
    "t": (4, 1.0, "wrist_roll +"),
    "g": (4, -1.0, "wrist_roll -"),
    "y": (5, 1.0, "gripper open +"),
    "h": (5, -1.0, "gripper close -"),
}


def _parse_floats(text: str, count: int, name: str) -> tuple[float, ...]:
    values = [float(v) for v in text.replace(",", " ").split()]
    if len(values) != count:
        raise argparse.ArgumentTypeError(f"{name} expects {count} floats, got {len(values)}: {text!r}")
    return tuple(values)


def _vec3(text: str) -> tuple[float, float, float]:
    return _parse_floats(text, 3, "vec3")  # type: ignore[return-value]


def _quat(text: str) -> tuple[float, float, float, float]:
    return _parse_floats(text, 4, "quat")  # type: ignore[return-value]


def _vec6(text: str) -> tuple[float, float, float, float, float, float]:
    return _parse_floats(text, 6, "vec6")  # type: ignore[return-value]


parser = argparse.ArgumentParser(description="GUI joint teleop and camera tuning for pick-pen scene")
parser.add_argument("--task", default="SimToReal-SO101-PickPen-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--step_hz", type=float, default=30.0)
parser.add_argument("--control_mode", choices=("keyboard", "leader"), default="keyboard")
parser.add_argument("--joint_step", type=float, default=0.035, help="Arm joint increment in radians per key press")
parser.add_argument("--gripper_step", type=float, default=0.05, help="Gripper increment per key press")
parser.add_argument("--max_steps", type=int, default=0, help="0 = run until Isaac window closes or Esc is pressed")
parser.add_argument("--snapshot_dir", type=Path, default=Path("outputs/camera_tuning"))
parser.add_argument("--snapshot_on_start", action="store_true", help="Save camera snapshots after initial warmup")
parser.add_argument("--snapshot_interval", type=int, default=0, help="Save snapshots every N steps; 0 disables")
parser.add_argument("--no_cameras", action="store_true", help="Do not inject top/front/wrist TiledCamera sensors")
parser.add_argument("--randomize_scene", action="store_true", help="Keep reset-time pen/cup randomization enabled")
parser.add_argument("--leader_port", default="COM5", help="SO-101 leader serial port for --control_mode leader")
parser.add_argument("--leader_id", default="so101_teleop", help="LeRobot calibration id for the SO-101 leader")
parser.add_argument("--leader_calibration_dir", type=Path, default=None, help="Optional LeRobot calibration directory")
parser.add_argument(
    "--leader_calibrate",
    action="store_true",
    help="Allow LeRobot's interactive leader calibration flow if calibration is missing or mismatched",
)
parser.add_argument(
    "--leader_gripper_divisor",
    type=float,
    default=100.0,
    help="Convert LeRobot leader gripper 0..100 value into Isaac 0..1 joint target",
)
parser.add_argument(
    "--leader_joint_signs",
    type=_vec6,
    default=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    help="Six signs applied after unit conversion, e.g. '1,-1,1,1,1,1'",
)
parser.add_argument(
    "--leader_joint_offsets",
    type=_vec6,
    default=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    help="Six offsets after sign conversion; arm in radians, gripper in 0..1",
)
parser.add_argument(
    "--leader_smoothing",
    type=float,
    default=0.0,
    help="Exponential smoothing factor for leader targets; 0.0 follows directly, 0.9 is very smooth",
)

# Camera overrides. Defaults come from pick_pen_env_cfg.py.
parser.add_argument("--top_pos", type=_vec3, default=None, help="x,y,z world position")
parser.add_argument("--top_target", type=_vec3, default=None, help="x,y,z world look-at target")
parser.add_argument("--top_focal", type=float, default=None, help="focal length in mm")
parser.add_argument("--front_pos", type=_vec3, default=None, help="x,y,z world position")
parser.add_argument("--front_target", type=_vec3, default=None, help="x,y,z world look-at target")
parser.add_argument("--front_focal", type=float, default=None, help="focal length in mm")
parser.add_argument("--wrist_pos", type=_vec3, default=None, help="x,y,z gripper-local position")
parser.add_argument("--wrist_rot", type=_quat, default=None, help="w,x,y,z gripper-local quaternion")
parser.add_argument("--wrist_focal", type=float, default=None, help="focal length in mm")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.no_cameras:
    args.enable_cameras = True

launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import sim_to_real  # noqa: E402  # registers the gym env

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from sim_to_real.tasks.pick_pen.pick_pen_env_cfg import (  # noqa: E402
    SO101_JOINT_ORDER,
    add_pick_pen_cameras,
)


class NonBlockingKeyboard:
    """Tiny cross-platform non-blocking terminal keyboard reader."""

    def __init__(self) -> None:
        self._is_windows = os.name == "nt"
        self._old_termios = None

    def __enter__(self) -> "NonBlockingKeyboard":
        if not self._is_windows and sys.stdin.isatty():
            import termios
            import tty

            self._old_termios = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, *_exc) -> None:
        if self._old_termios is not None:
            import termios

            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_termios)

    def read_key(self) -> str | None:
        if self._is_windows:
            import msvcrt

            if not msvcrt.kbhit():
                return None
            ch = msvcrt.getwch()
            # Swallow the second byte of Windows special-key sequences.
            if ch in ("\x00", "\xe0") and msvcrt.kbhit():
                msvcrt.getwch()
                return None
            return ch

        if not sys.stdin.isatty():
            return None
        import select

        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            return None
        return sys.stdin.read(1)


class RateLimiter:
    def __init__(self, hz: float) -> None:
        self.period = 1.0 / max(hz, 1e-6)
        self.next_time = time.perf_counter() + self.period

    def sleep(self) -> None:
        now = time.perf_counter()
        if self.next_time > now:
            time.sleep(self.next_time - now)
        self.next_time = max(self.next_time + self.period, time.perf_counter())


class LeaderArmSource:
    """LeRobot SO-101 leader reader converted into Isaac joint-position targets."""

    def __init__(
        self,
        port: str,
        leader_id: str,
        calibration_dir: Path | None,
        calibrate: bool,
        gripper_divisor: float,
        joint_signs: tuple[float, float, float, float, float, float],
        joint_offsets: tuple[float, float, float, float, float, float],
        device: str,
    ) -> None:
        if gripper_divisor <= 1e-6:
            raise ValueError("--leader_gripper_divisor must be > 0")
        self.port = port
        self.leader_id = leader_id
        self.calibration_dir = calibration_dir
        self.calibrate = calibrate
        self.gripper_divisor = gripper_divisor
        self.joint_signs = torch.tensor(joint_signs, dtype=torch.float32, device=device)
        self.joint_offsets = torch.tensor(joint_offsets, dtype=torch.float32, device=device)
        self.device = device
        self.teleop = None

    def connect(self) -> None:
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

        cfg = SO101LeaderConfig(
            port=self.port,
            id=self.leader_id,
            calibration_dir=self.calibration_dir,
            use_degrees=True,
        )
        self.teleop = SO101Leader(cfg)
        print(f"[leader] connecting SO-101 leader on {self.port} (id={self.leader_id})")
        self.teleop.connect(calibrate=self.calibrate)
        if not self.teleop.is_calibrated:
            raise RuntimeError(
                "SO-101 leader calibration is missing or mismatched. Run again with "
                "--leader_calibrate, or calibrate the leader with LeRobot first."
            )
        print("[leader] connected; arm joints use LeRobot degrees, gripper uses 0..100")

    def disconnect(self) -> None:
        if self.teleop is not None and self.teleop.is_connected:
            self.teleop.disconnect()

    def read_targets(self, limits: torch.Tensor) -> torch.Tensor:
        if self.teleop is None:
            raise RuntimeError("LeaderArmSource is not connected")

        raw_action = self.teleop.get_action()
        values: list[float] = []
        for joint_id, joint_name in enumerate(SO101_JOINT_ORDER):
            value = float(raw_action[f"{joint_name}.pos"])
            if joint_id < 5:
                value = math.radians(value)
            else:
                value = value / self.gripper_divisor
            values.append(value)

        targets = torch.tensor(values, dtype=torch.float32, device=self.device)
        targets = targets * self.joint_signs + self.joint_offsets
        return torch.maximum(torch.minimum(targets, limits[:, 1]), limits[:, 0])


def _disable_randomization(env_cfg) -> None:
    for name in (
        "randomize_pen_white",
        "randomize_pen_gray",
        "randomize_pen_black",
        "randomize_pen_blue",
        "randomize_pen_cup",
    ):
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)


def _disable_episode_termination(env_cfg) -> None:
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None


def _joint_limits(device: str) -> torch.Tensor:
    # Broad debug limits. The goal is camera/scene tuning, not enforcing final motor safety.
    limits = [
        (-math.pi, math.pi),
        (-math.pi, math.pi),
        (-math.pi, math.pi),
        (-math.pi, math.pi),
        (-math.pi, math.pi),
        (0.0, 1.0),
    ]
    return torch.tensor(limits, dtype=torch.float32, device=device)


def _render_env(env) -> None:
    env.unwrapped.sim.render()


def _rgb_to_u8(rgb: torch.Tensor) -> np.ndarray:
    arr = rgb[0].detach().cpu().numpy()
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _camera_metadata(env) -> dict:
    metadata: dict[str, dict] = {}
    for name in CAMERA_NAMES:
        try:
            cam = env.unwrapped.scene[name]
        except KeyError:
            continue

        item: dict[str, object] = {}
        try:
            item["pos_w"] = cam.data.pos_w[0].detach().cpu().tolist()
        except Exception:
            pass
        try:
            if hasattr(cam.data, "quat_w_world"):
                item["rot_w_wxyz"] = cam.data.quat_w_world[0].detach().cpu().tolist()
            elif hasattr(cam.data, "quat_w_ros"):
                item["rot_w_wxyz"] = cam.data.quat_w_ros[0].detach().cpu().tolist()
        except Exception:
            pass
        try:
            intr = cam.data.intrinsic_matrices[0].detach().cpu().tolist()
            item["intrinsics_3x3"] = intr
            fx = intr[0][0]
            item["fov_horizontal_deg"] = math.degrees(2.0 * math.atan(640.0 / (2.0 * fx)))
        except Exception:
            pass
        metadata[name] = item
    return metadata


def _save_snapshots(env, snapshot_dir: Path, step: int) -> None:
    from PIL import Image

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{step:06d}"
    saved: dict[str, str] = {}
    for name in CAMERA_NAMES:
        try:
            cam = env.unwrapped.scene[name]
        except KeyError:
            continue
        image = _rgb_to_u8(cam.data.output["rgb"])
        path = snapshot_dir / f"{name}_{stamp}.png"
        latest = snapshot_dir / f"{name}_latest.png"
        Image.fromarray(image).save(path)
        Image.fromarray(image).save(latest)
        saved[name] = str(path)

    meta = {
        "step": step,
        "saved": saved,
        "cameras": _camera_metadata(env),
        "cli_overrides": {
            "top_pos": args.top_pos,
            "top_target": args.top_target,
            "top_focal": args.top_focal,
            "front_pos": args.front_pos,
            "front_target": args.front_target,
            "front_focal": args.front_focal,
            "wrist_pos": args.wrist_pos,
            "wrist_rot": args.wrist_rot,
            "wrist_focal": args.wrist_focal,
        },
    }
    meta_path = snapshot_dir / f"camera_metadata_{stamp}.json"
    latest_meta = snapshot_dir / "camera_metadata_latest.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    latest_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[snapshot] {snapshot_dir.resolve()} ({', '.join(saved) or 'no cameras'})")


def _print_controls(control_mode: str) -> None:
    if control_mode == "leader":
        print(
            f"""
Controls:
  Move SO-101 Leader Arm on {args.leader_port}; its calibrated joint positions drive the sim arm.
  u reset scene               c save 3-camera snapshots
  p print joints + camera metadata
  Esc or Ctrl+C quit
""".strip()
        )
        return

    print(
        """
Controls (terminal window must have focus):
  q/a shoulder_pan    +/-       w/s shoulder_lift +/-
  e/d elbow_flex      +/-       r/f wrist_flex     +/-
  t/g wrist_roll      +/-       y/h gripper open/close
  [ / ] decrease/increase step size
  z zero all targets          u reset scene
  c save 3-camera snapshots   p print joints + camera metadata
  Esc or Ctrl+C quit
""".strip()
    )


def _print_state(targets: torch.Tensor, env) -> None:
    print("[targets]")
    for name, value in zip(SO101_JOINT_ORDER, targets.detach().cpu().tolist(), strict=True):
        print(f"  {name:16s} {value:+.4f}")
    print("[cameras]")
    print(json.dumps(_camera_metadata(env), indent=2))


def main() -> None:
    if args.num_envs != 1:
        raise ValueError("This manual teleop script currently supports --num_envs 1 only")

    device: str = args.device
    env = None
    try:
        env_cfg = parse_env_cfg(args.task, device=device, num_envs=1)
        env_cfg.seed = int(time.time())
        _disable_episode_termination(env_cfg)
        if not args.randomize_scene:
            _disable_randomization(env_cfg)
        if not args.no_cameras:
            add_pick_pen_cameras(
                env_cfg.scene,
                top_pos=args.top_pos,
                top_target=args.top_target,
                top_focal=args.top_focal,
                front_pos=args.front_pos,
                front_target=args.front_target,
                front_focal=args.front_focal,
                wrist_local_pos=args.wrist_pos,
                wrist_local_rot=args.wrist_rot,
                wrist_focal=args.wrist_focal,
            )

        env = gym.make(args.task, cfg=env_cfg)
        env.reset()
        robot = env.unwrapped.scene["robot"]
        targets = robot.data.joint_pos[0, :6].clone().to(device)
        limits = _joint_limits(device)
        step_count = 0
        arm_step = float(args.joint_step)
        gripper_step = float(args.gripper_step)
        rate = RateLimiter(args.step_hz)
        leader = None
        if args.control_mode == "leader":
            leader = LeaderArmSource(
                port=args.leader_port,
                leader_id=args.leader_id,
                calibration_dir=args.leader_calibration_dir,
                calibrate=args.leader_calibrate,
                gripper_divisor=args.leader_gripper_divisor,
                joint_signs=args.leader_joint_signs,
                joint_offsets=args.leader_joint_offsets,
                device=device,
            )
            leader.connect()
            targets = leader.read_targets(limits)

        _print_controls(args.control_mode)
        if args.snapshot_on_start and not args.no_cameras:
            for _ in range(5):
                env.step(targets.unsqueeze(0))
                if not args.headless:
                    _render_env(env)
            _save_snapshots(env, args.snapshot_dir, 0)

        with NonBlockingKeyboard() as keyboard:
            while simulation_app.is_running():
                key = keyboard.read_key()
                if key in ("\x1b", "\x03"):
                    break
                if key:
                    lower = key.lower()
                    if args.control_mode == "keyboard" and lower in KEY_BINDINGS:
                        joint_id, direction, label = KEY_BINDINGS[lower]
                        delta = gripper_step if joint_id == 5 else arm_step
                        targets[joint_id] += direction * delta
                        targets = torch.maximum(torch.minimum(targets, limits[:, 1]), limits[:, 0])
                        print(f"[key] {label}: {targets[joint_id].item():+.4f}")
                    elif lower == "[":
                        arm_step = max(0.002, arm_step * 0.75)
                        gripper_step = max(0.005, gripper_step * 0.75)
                        print(f"[step] arm={arm_step:.4f}, gripper={gripper_step:.4f}")
                    elif lower == "]":
                        arm_step = min(0.25, arm_step * 1.3333)
                        gripper_step = min(0.25, gripper_step * 1.3333)
                        print(f"[step] arm={arm_step:.4f}, gripper={gripper_step:.4f}")
                    elif lower == "z":
                        targets.zero_()
                        print("[target] zero")
                    elif lower == "u":
                        env.reset()
                        targets = robot.data.joint_pos[0, :6].clone().to(device)
                        print("[scene] reset")
                    elif lower == "c":
                        if not args.no_cameras:
                            _save_snapshots(env, args.snapshot_dir, step_count)
                    elif lower == "p":
                        _print_state(targets, env)

                if leader is not None:
                    leader_targets = leader.read_targets(limits)
                    smoothing = min(max(float(args.leader_smoothing), 0.0), 0.99)
                    if smoothing > 0.0:
                        targets = smoothing * targets + (1.0 - smoothing) * leader_targets
                    else:
                        targets = leader_targets

                action = targets.unsqueeze(0)
                env.step(action)
                if not args.headless:
                    _render_env(env)
                step_count += 1
                if args.snapshot_interval > 0 and step_count % args.snapshot_interval == 0 and not args.no_cameras:
                    _save_snapshots(env, args.snapshot_dir, step_count)
                if args.max_steps > 0 and step_count >= args.max_steps:
                    break
                rate.sleep()

    except KeyboardInterrupt:
        pass
    except Exception:
        print(traceback.format_exc())
        raise
    finally:
        if "leader" in locals() and leader is not None:
            leader.disconnect()
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
