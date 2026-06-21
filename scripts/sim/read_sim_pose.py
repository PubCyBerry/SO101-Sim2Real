"""Sim 포즈 구동 + joint 각도 출력 (sim↔real per-joint affine 측정용).

PickCube 환경의 SO-101 을 지정 joint config 들로 순환 구동하며 **데이터셋과 동일한
프레임**(arm deg, gripper [0,100]=rad×31.75; `lerobot_units.to_lerobot_units`)으로
achieved joint 값을 출력한다. GUI 로 포즈를 보고, 실기기에서 같은 포즈를 손으로
재현해 `scripts/test/read_position.py` 로 읽어 짝지으면 `measure_joint_affine.py` 가
per-joint affine(`AFFINE_<JOINT>_SIGN/OFFSET`, gripper anchor)을 피팅한다.

⚠ recorder(`rollout_to_lerobot.py`)와 동일하게 `robot.data.joint_pos[0]` 를 그대로
`to_lerobot_units` 에 넣는다 → 출력값이 학습 데이터 state 와 같은 단위·순서.

실행 (RT 코어 GPU 필요, Windows 워크스테이션):
    OMNI_KIT_ACCEPT_EULA=YES uv run --group isaac python scripts/sim/read_sim_pose.py
    # headless 로 값만:  ... read_sim_pose.py --headless
    # 포즈 직접 지정(rad, 'pan,lift,elbow,wflex,wroll,grip' 반복):
    #   ... --pose 0,0,0,0,0,-0.05 --pose 0,-1.3,1.2,-0.349,-1.571,0.85

각 포즈를 hold_steps 동안 유지하며 achieved 값을 주기 출력, 포즈 순환 반복(Ctrl-C 종료).
"""

from __future__ import annotations

import argparse
import faulthandler
import math
import os
from pathlib import Path

from isaaclab.app import AppLauncher


def _pose(s: str):
    vals = [float(x) for x in s.split(",")]
    if len(vals) != 6:
        raise argparse.ArgumentTypeError(f"포즈는 6값(rad) 'pan,lift,elbow,wflex,wroll,grip': {s!r}")
    return vals


parser = argparse.ArgumentParser(description="Sim 포즈 구동 + joint 각도 출력 (affine 측정용)")
parser.add_argument("--task", default="SimToReal-SO101-PickCube-v0")
parser.add_argument("--pose", type=_pose, action="append", default=None,
                    help="joint config(rad) 'pan,lift,elbow,wflex,wroll,grip'. 반복 지정. 미지정=기본 P1/P2/P3")
parser.add_argument("--hold_steps", type=int, default=150, help="포즈당 유지 step(30Hz, 150≈5s)")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# C-레벨 크래시 추적 (AGENTS: GUI 진입 스크립트 규약)
Path("outputs").mkdir(exist_ok=True)
faulthandler.enable(open("outputs/read_sim_pose_faulthandler.txt", "w"))

# Windows 워크스테이션 GUI 크래시 회피: 기본 GUI experience(isaaclab.python.kit)는 이 박스
# rtx.scenedb.plugin.dll 에서 0xc0000005 access violation(_prepare_ui)으로 즉사한다.
# --enable_cameras 면 AppLauncher 가 isaaclab.python.rendering.kit experience 를 골라 GUI 가 유지된다.
# (docs/TROUBLESHOOTING.md §RTX scene DB access violation / §_prepare_ui). 따라서 GUI(=non-headless)
# 에선 enable_cameras 를 강제한다. headless 면 불필요.
if not args.headless:
    args.enable_cameras = True

# 커스텀 인자는 AppLauncher 에 넘기지 않는다(Windows _prepare_ui access violation 회피).
_LAUNCHER_KEYS = {"headless", "enable_cameras", "experience", "device", "cpu",
                  "disable_fabric", "offscreen_render", "kit_args", "livestream"}
app_launcher = AppLauncher({k: v for k, v in vars(args).items() if k in _LAUNCHER_KEYS})
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import sim_to_real  # noqa: E402, F401  (Gym 환경 등록)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

# recorder 와 동일 변환·순서 (scripts/sim 동일 디렉터리)
from lerobot_units import JOINT_FEATURE_NAMES, to_lerobot_units  # noqa: E402

# PickCube gripper action term default offset (env init joint_pos["gripper"]).
# SlewLimitedJointPositionAction(use_default_offset=True): target = raw + offset → raw = target - offset.
GRIPPER_ACTION_OFFSET = 0.20

# 기본 포즈(rad): distinct 3개. 각 joint 가 포즈마다 충분히 달라야 affine 직선이 안정.
DEFAULT_POSES = [
    [0.0, 0.0, 0.0, 0.0, 0.0, -0.05],                              # P1 zero, grip 닫힘
    [0.0, -1.3, 1.2, math.radians(-20.0), math.radians(-90.0), 0.85],  # P2 READY, grip 열림
    [0.5, -0.7, 0.8, 0.3, 0.5, 0.40],                              # P3 distinct, grip 중간
]


def main() -> None:
    poses = args.pose if args.pose else DEFAULT_POSES
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg)
    env.reset()
    robot = env.unwrapped.scene["robot"]
    device = env.unwrapped.device

    print(f"[read_sim_pose] {len(poses)} 포즈 순환. 단위 = arm deg, gripper [0,100](rad×31.75) "
          f"= 데이터셋·모델 프레임. Ctrl-C 종료.", flush=True)

    pi = 0
    try:
        while simulation_app.is_running():
            target = poses[pi % len(poses)]
            # action: arm = target_rad(default 0), gripper = target_rad - offset.
            act = torch.tensor(
                [target[0], target[1], target[2], target[3], target[4], target[5] - GRIPPER_ACTION_OFFSET],
                dtype=torch.float32, device=device,
            ).unsqueeze(0)
            for _ in range(max(1, args.hold_steps)):
                env.step(act)
            achieved = to_lerobot_units(robot.data.joint_pos[0].detach().cpu().numpy())
            cmd_deg = [math.degrees(v) for v in target[:5]] + [target[5] * 31.75]
            print(f"\n[포즈 {pi % len(poses) + 1}] 명령(deg/그리퍼계)≈ "
                  + " ".join(f"{n.split('.')[0]}={c:.1f}" for n, c in zip(JOINT_FEATURE_NAMES, cmd_deg)),
                  flush=True)
            print("  achieved(=sim_deg, 이 값을 측정에 사용): "
                  + " ".join(f"{n.split('.')[0]}={a:.2f}" for n, a in zip(JOINT_FEATURE_NAMES, achieved)),
                  flush=True)
            pi += 1
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
