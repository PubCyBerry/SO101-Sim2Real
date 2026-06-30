"""실 SO-101 joint-space 시퀀스(JSON)를 Isaac Sim 에서 재생하고 LeRobot v3 로 기록.

JSON 은 실 follower joint 궤적(arm degree, gripper [0,100], phase별 move/hold).
joint-space 라 IK·state machine 불필요 — env 기본 6D joint-position action term
(`PickCubeActionsCfg`, slew-limited)을 그대로 쓴다. 각 보간 waypoint 를 follower
calibration 으로 sim radian 변환해 env.step 하고, achieved state·카메라와 함께 기록한다.

데이터셋 action = 원본 waypoint(실 follower 단위, arm degree·gripper [0,100]) 이므로
그대로 실 SO-101 `lerobot-replay` 가 원궤적을 재현한다. observation.state 는 sim achieved
joint 을 실 follower 단위로 역변환(`sim_radians_to_real_follower`)해 action 과 동일 단위계로 둔다.

Usage:
    OMNI_KIT_ACCEPT_EULA=YES uv run --group isaac python scripts/datagen/record_real_sequence.py \
        --task SimToReal-SO101-PickCube-v0 --headless --enable_cameras \
        --sequence scripts/ece_4560/real/sequences/pick_place_demo.json \
        --dataset_dir ./datasets/pick_cube_test --fps 30

    # 단위/보간 self-check (Isaac Sim 불필요):
    python scripts/datagen/record_real_sequence.py --self_check
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from so101_contract.feature_codec import SO101_JOINT_ORDER, clamp_sim_joint_radians
from so101_contract.follower_calibration import (
    real_follower_to_sim_radians,
    sim_radians_to_real_follower,
)


# ── 시퀀스 로드·보간 (순수, sim 불필요) ──────────────────────────────────────
def load_sequence_phases(path: str | Path) -> list[tuple[np.ndarray, float, float]]:
    """JSON → [(target[6] 실 follower 단위, move_time, hold_time), ...]."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data["phases"] if isinstance(data, dict) else data
    phases = []
    for r in rows:
        target = np.asarray([float(r[j]) for j in SO101_JOINT_ORDER], dtype=np.float64)
        phases.append((target, float(r.get("move_time", 2.0)), float(r.get("hold_time", 0.0))))
    return phases


def build_waypoints(start: np.ndarray, phases, fps: int) -> np.ndarray:
    """start pose(실 follower 단위)에서 phase 들을 선형보간 → (T,6) waypoint 배열.

    phase 마다 move(round(move_time*fps) 프레임, prev→target LERP) + hold(round(hold_time*fps)
    프레임, target 유지). 실 GUI(so101_utils.move_to_pose) 와 동일한 LERP 규약.
    """
    wps: list[np.ndarray] = []
    prev = np.asarray(start, dtype=np.float64)
    for target, move_t, hold_t in phases:
        n_move = max(1, round(move_t * fps))
        for k in range(1, n_move + 1):
            a = k / n_move
            wps.append((1.0 - a) * prev + a * target)
        for _ in range(round(hold_t * fps)):
            wps.append(target.copy())
        prev = target
    return np.asarray(wps, dtype=np.float32)


def _self_check() -> None:
    phases = [
        (np.array([10, 20, -30, 40, -50, 60.0]), 2.0, 1.0),
        (np.array([-10, -20, 30, -40, 50, 11.0]), 2.0, 0.0),
    ]
    wps = build_waypoints(np.zeros(6), phases, fps=30)
    assert len(wps) == (60 + 30) + (60 + 0), f"보간 길이 {len(wps)}"
    assert np.allclose(wps[-1], phases[-1][0], atol=1e-4), "마지막 waypoint != 마지막 target"
    # 단위 round-trip: 실 follower → sim radian → 실 follower ≈ 원본.
    for wp in (wps[0], wps[44], wps[-1]):
        back = sim_radians_to_real_follower(real_follower_to_sim_radians(wp))
        assert np.allclose(back, wp, atol=1e-3), f"round-trip 불일치 {back} vs {wp}"
    # 실제 데모 JSON 도 로드 검증(있으면).
    demo = Path(__file__).resolve().parents[1] / "ece_4560/real/sequences/pick_place_demo.json"
    if demo.is_file():
        ph = load_sequence_phases(demo)
        assert len(ph) == 6 and all(t.shape == (6,) for t, _, _ in ph), "데모 JSON 형태 이상"
    print("[record_real_sequence] self-check OK")


if "--self_check" in sys.argv:
    _self_check()
    sys.exit(0)


# ── Isaac Sim 부팅 (다른 isaac import 보다 먼저) ─────────────────────────────
import argparse  # noqa: E402
import signal  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Replay a real SO-101 joint sequence in Isaac Sim → LeRobot v3.")
parser.add_argument("--task", type=str, default="SimToReal-SO101-PickCube-v0", help="Gym task id.")
parser.add_argument("--sequence", type=str, required=True, help="실 follower 시퀀스 JSON 경로.")
parser.add_argument("--dataset_dir", type=str, default="./datasets/pick_cube_test", help="LeRobot v3 출력 폴더.")
parser.add_argument("--fps", type=int, default=30, help="보간/기록 FPS (env control rate=30).")
parser.add_argument("--num_envs", type=int, default=1, help="환경 수(=1).")
parser.add_argument("--seed", type=int, default=0, help="환경 시드.")
parser.add_argument(
    "--task_desc",
    type=str,
    default="pick up the cube and place it in the bowl",
    help="LeRobot task 문자열(스키마 정합).",
)
# --headless·--enable_cameras·--device 등은 AppLauncher.add_app_launcher_args 가 추가한다.
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# AppLauncher 가 실제 쓰는 키만 전달(Windows access violation 회피 — AGENTS.md 규약).
_LAUNCHER_KEYS = {
    "headless", "device", "num_envs", "experience", "enable_cameras", "physics_dt",
    "rendering_dt", "enable_viewport", "viewport_camera_state",
}
app_launcher = AppLauncher({k: v for k, v in vars(args_cli).items() if k in _LAUNCHER_KEYS})
simulation_app = app_launcher.app


def main() -> None:
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    import sim_to_real  # noqa: F401  task 등록
    from sim_to_real.data.lerobot_recorder import LeRobotV3DatasetWriter
    from sim_to_real.data.lerobot_units import CAMERA_SCENE_NAMES, read_camera_rgb_u8
    from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import add_pick_cube_cameras
    from sim_to_real.utils.gripper_effort import dynamic_reset_gripper_effort_limit_sim

    phases = load_sequence_phases(args_cli.sequence)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    # 기본 action term(6D slew joint-position) 유지 + 로봇 링크 중력 off(떨림 없는 결정적 추종).
    env_cfg.use_teleop_device("so101_state_machine")
    add_pick_cube_cameras(env_cfg.scene)  # gym.make 전 top/wrist/front 카메라 주입
    env_cfg.seed = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    if hasattr(env, "initialize"):
        env.initialize()
    env.reset()

    robot = env.scene["robot"]
    device = env.device

    # 시작 pose(reset 후 실제 joint) → 실 follower 단위. 보간 시작점.
    start_rad = robot.data.joint_pos[0].detach().cpu().numpy()
    start_real = sim_radians_to_real_follower(start_rad)
    waypoints = build_waypoints(start_real, phases, args_cli.fps)
    print(f"[seq] {len(phases)} phases → {len(waypoints)} frames @ {args_cli.fps}fps", flush=True)

    writer = LeRobotV3DatasetWriter(
        Path(args_cli.dataset_dir), overwrite=True, enable_videos=True, robot_type="so_follower"
    )

    interrupted = False

    def _sig(*_):
        nonlocal interrupted
        interrupted = True
        print("\n[INFO] Ctrl+C — 정리 중...", flush=True)

    old_handler = signal.signal(signal.SIGINT, _sig)

    try:
        with torch.inference_mode():
            # warmup: 시작 pose 유지로 물리·카메라 버퍼 채우기.
            hold0 = torch.from_numpy(clamp_sim_joint_radians(start_rad.astype(np.float32))).unsqueeze(0).to(device)
            for _ in range(5):
                dynamic_reset_gripper_effort_limit_sim(env, "so101_state_machine")
                env.step(hold0)

            for i, wp in enumerate(waypoints):
                if interrupted or not simulation_app.is_running() or simulation_app.is_exiting():
                    break
                dynamic_reset_gripper_effort_limit_sim(env, "so101_state_machine")
                sim_rad = clamp_sim_joint_radians(real_follower_to_sim_radians(wp))
                action = torch.from_numpy(sim_rad).unsqueeze(0).to(device)
                env.step(action)

                achieved_rad = robot.data.joint_pos[0].detach().cpu().numpy()
                state_real = sim_radians_to_real_follower(achieved_rad)

                images = {}
                for cam_key, scene_name in CAMERA_SCENE_NAMES.items():
                    try:
                        images[cam_key] = read_camera_rgb_u8(env, scene_name)
                    except Exception as e:
                        print(f"[warn] {cam_key} 카메라 캡처 실패: {e}", flush=True)
                writer.add_frame(wp.astype(np.float32), state_real, images)

                if (i + 1) % 60 == 0:
                    print(f"  frame {i + 1}/{len(waypoints)}", flush=True)
    except Exception as e:
        import traceback

        print(f"\n[ERROR] {e}\n", flush=True)
        traceback.print_exc()
    finally:
        signal.signal(signal.SIGINT, old_handler)
        committed = writer.commit_episode(success=True, task_name=args_cli.task_desc)
        summary = writer.finalize()
        print(
            f"\n[seq] committed={committed} episodes={summary['total_episodes']} "
            f"frames={summary['total_frames']} → {summary['output_dir']}",
            flush=True,
        )
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
