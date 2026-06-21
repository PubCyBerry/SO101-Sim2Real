# measure_joint_affine.py
#
# Sim↔Real per-joint frame 보정 측정·피팅 도구 (실기기 Windows 네이티브 실행).
#
# 목적: sim 학습 SmolVLA 를 실기기 추론할 때 필요한 joint 별 affine 을 측정한다.
#   model(sim) = A_j * real + B_j   (arm: A=±1 sign, B=offset / gripper: scale+offset)
# 결과를 `docker/policy-client-shim.py` 의 env 변수 형태로 출력한다(GRIPPER_AFFINE=1 경로).
#
# 원리: arm 은 DEGREES(절대각, scale 1:1)라 sim·real 차이는 영점 offset(+가끔 부호)뿐.
# 같은 **물리 포즈**를 sim·real 양쪽 단위로 읽어 2개 이상 모으면 per-joint 직선이 정해진다.
#
# 워크플로 (포즈 N개, N>=2 권장 3):
#   1) (SIM) 로봇을 어떤 config 로 둔다 → 그 sim joint_pos(deg) 6값을 안다(네가 set 한 값).
#   2) (REAL) 실기기 팔을 그 sim 포즈와 **똑같이** 손으로 맞춘다(토크 off).
#   3) 이 스크립트에서 Enter → real 읽음. 이어서 같은 포즈의 sim deg 6값 입력.
#   4) 포즈 바꿔 반복. 'q' 로 종료 → per-joint 피팅 + shim env 출력.
#
# 포즈 선택 팁: joint 마다 값이 충분히 다른 distinct 포즈(예: home / READY[0,-74.5,68.8,-20,-90]
# / 팔 앞으로 뻗기). 두 포즈에서 한 joint 값이 비슷하면 그 joint 피팅이 불안정.
#
# 빠른 추정(sim 없이): 각 joint 를 양쪽 기계 stop 까지 움직여 real 읽고, sim 값은 URDF 한계
#   pan ±110 / lift ±100 / elbow ±97 / wrist_flex ±95 / wrist_roll [-157,163] 로 입력.
#   (wrist_roll 은 연속회전이라 stop 없음 → distinct 포즈로 측정.)

import sys
import time

import numpy as np

from so101_utils import load_calibration, setup_motors

PORT_ID = "COM8"
ROBOT_NAME = "so101_robot"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def read_real(bus, samples=10):
    """Present_Position 평균 (arm deg, gripper [0,100]) — 추론과 동일 프레임."""
    acc = None
    for _ in range(samples):
        d = bus.sync_read("Present_Position")
        v = np.array([d[j] for j in JOINTS], dtype=np.float64)
        acc = v if acc is None else acc + v
        time.sleep(0.01)
    return acc / samples


def fit_and_print(reals, sims):
    reals = np.asarray(reals)  # (N,6) real
    sims = np.asarray(sims)    # (N,6) sim
    n = len(reals)
    print(f"\n=== 피팅 ({n} 포즈) — model(sim) = A * real + B ===")
    print(f"{'joint':14s}{'A(fit)':>9s}{'A(사용)':>9s}{'B(offset)':>11s}{'잔차°':>8s}")
    lines_arm, grip_line = [], ""
    for i, j in enumerate(JOINTS):
        x = reals[:, i]
        y = sims[:, i]
        A_fit, B_fit = np.polyfit(x, y, 1) if n >= 2 else (1.0, y[0] - x[0])
        resid = float(np.sqrt(np.mean((y - (A_fit * x + B_fit)) ** 2)))
        if j == "gripper":
            # gripper 는 scale 자유 → 피팅한 직선(model=A*real+B)을 shim 에 직접 넣는다.
            grip_line = f"GRIPPER_A={A_fit:.4f} GRIPPER_B={B_fit:.3f}"
            print(f"{j:14s}{A_fit:9.3f}{A_fit:9.3f}{B_fit:11.2f}{resid:8.2f}  (gripper=scale, A 자유)")
        else:
            # arm 은 부호만 의미(±1). 가까운 ±1 로 강제, offset 재계산.
            A_use = 1.0 if A_fit >= 0 else -1.0
            B_use = float(np.mean(y - A_use * x))
            resid = float(np.sqrt(np.mean((y - (A_use * x + B_use)) ** 2)))
            warn = "  ⚠ A_fit 이 ±1 에서 멀다(포즈 부족/매칭 오류?)" if abs(abs(A_fit) - 1) > 0.15 else ""
            print(f"{j:14s}{A_fit:9.3f}{A_use:9.0f}{B_use:11.2f}{resid:8.2f}{warn}")
            lines_arm.append(
                f"AFFINE_{j.upper()}_SIGN={A_use:.0f} AFFINE_{j.upper()}_OFFSET={B_use:.2f}")

    print("\n=== shim env (실기기 sim-모델 추론 시 명령 앞에 export) ===")
    print("GRIPPER_AFFINE=1 \\")
    for ln in lines_arm:
        print(f"{ln} \\")
    print(grip_line)
    print("# 포즈들에서 gripper 개도를 충분히 다르게(닫힘/중간/열림) 했을 때만 gripper 피팅 신뢰.")
    print("# 검증: sim 1ep lerobot-replay 로 재생 → 어긋난 joint 의 SIGN/OFFSET 조정.")


def main():
    calibration = load_calibration(ROBOT_NAME)
    bus = setup_motors(calibration, PORT_ID)
    bus.disable_torque()
    print("토크 OFF. 팔을 손으로 포즈에 맞추고 Enter → real 읽음. 종료 'q'.")
    reals, sims = [], []
    while True:
        cmd = input(f"\n[포즈 {len(reals) + 1}] 맞춘 뒤 Enter (종료 q): ").strip().lower()
        if cmd == "q":
            break
        rv = read_real(bus)
        print("  real(deg/[0,100]):", " ".join(f"{j}={v:.1f}" for j, v in zip(JOINTS, rv)))
        sraw = input("  같은 포즈의 sim deg 6값(space; gripper 는 rad×31.75계): ").split()
        if len(sraw) != 6:
            print("  6값 아님 → 이 포즈 버림"); continue
        reals.append(rv); sims.append([float(s) for s in sraw])
    if len(reals) < 2:
        print("포즈 2개 미만 → 피팅 불가"); sys.exit(1)
    fit_and_print(reals, sims)


if __name__ == "__main__":
    main()
