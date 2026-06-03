"""TA.1 SO-101 PD drive smoke test: hold stability + step response.

Usage:
    uv run python scripts/environments/drive_response_smoke.py \
        --task SimToReal-SO101-PickPen-v0 \
        --num_envs 1 --device cuda:0
"""

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import json
import sys
import traceback

import numpy as np
from isaaclab.app import AppLauncher

# -- 수용 임계값 -----------------------------------------------------------------
HOLD_TAIL_MAX_POS_RAD = 0.04   # 정적 hold: tail 위치 편차
HOLD_TAIL_RMS_VEL_RADS = 0.05  # 정적 hold: tail RMS 속도
HOLD_FINAL_VEL_RADS = 0.10     # 정적 hold: 최종 관절 속도
STEP_FINAL_ERR_RAD = 0.08 # step 응답: 최종 추종 오차
STEP_OVERSHOOT_RAD = 0.08 # step 응답: 최대 overshoot
STEP_VEL_FACTOR = 1.25    # step 응답: 최대 속도 ≤ velocity_limit_sim × factor

# step 응답에서 사용할 관절 목표 오프셋 (North Star 순서)
STEP_TARGET = np.array([0.15, -0.12, 0.12, -0.10, 0.10, 0.05], dtype=np.float32)

parser = argparse.ArgumentParser(description="TA.1 drive response smoke test")
parser.add_argument("--task", default="SimToReal-SO101-PickPen-v0")
parser.add_argument("--steps-hold", type=int, default=360)
parser.add_argument("--steps-step", type=int, default=240)
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym   # noqa: E402
import torch              # noqa: E402

import sim_to_real        # noqa: E402  # registers SimToReal-SO101-PickPen-v0
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def _check_finite(arr: np.ndarray, name: str) -> None:
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf: {arr}")


def main() -> None:
    device: str = args.device
    env = None
    result: dict = {}
    passed = True

    try:
        env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
        # 에피소드를 테스트 전 구간이 끝날 때까지 유지
        total_policy_steps = args.steps_hold + args.steps_step + 60
        env_cfg.episode_length_s = max(
            env_cfg.episode_length_s,
            total_policy_steps * env_cfg.sim.dt * env_cfg.decimation + 5.0,
        )

        # velocity_limit_sim 값을 임계값 계산에 사용
        try:
            arm_vel_limit = env_cfg.scene.robot.actuators["arm_joints"].velocity_limit_sim
        except (AttributeError, KeyError):
            arm_vel_limit = 5.5

        env = gym.make(args.task, cfg=env_cfg)
        robot = env.unwrapped.scene["robot"]

        # ── 초기화 ──────────────────────────────────────────────────────────
        obs_dict, _ = env.reset()
        zero_action = torch.zeros(args.num_envs, 6, device=device)
        step_action = torch.tensor(
            STEP_TARGET[None, :].repeat(args.num_envs, axis=0),
            dtype=torch.float32,
            device=device,
        )

        # ── 1. Hold 테스트 ────────────────────────────────────────────────
        hold_pos_list = []
        hold_vel_list = []
        for _ in range(args.steps_hold):
            env.step(zero_action)
            hold_pos_list.append(robot.data.joint_pos[0].cpu().numpy())
            hold_vel_list.append(robot.data.joint_vel[0].cpu().numpy())

        # 초반 settling 구간 제외. tail 구간으로 정적 안정성을 판정한다.
        settle_skip = min(120, args.steps_hold // 2)
        tail_window = min(120, max(1, args.steps_hold // 3))
        hold_pos = np.array(hold_pos_list[settle_skip:])
        hold_vel = np.array(hold_vel_list[settle_skip:])
        hold_pos_tail = np.array(hold_pos_list[-tail_window:])
        hold_vel_tail = np.array(hold_vel_list[-tail_window:])

        _check_finite(hold_pos, "hold_pos")
        _check_finite(hold_vel, "hold_vel")

        hold_max_pos = float(np.max(np.abs(hold_pos)))
        hold_max_vel = float(np.max(np.abs(hold_vel)))
        hold_tail_max_pos = float(np.max(np.abs(hold_pos_tail)))
        hold_tail_rms_vel = float(np.sqrt(np.mean(hold_vel_tail * hold_vel_tail)))
        hold_final_vel = float(np.max(np.abs(hold_vel_tail[-1])))

        hold_ok = (
            hold_tail_max_pos <= HOLD_TAIL_MAX_POS_RAD
            and hold_tail_rms_vel <= HOLD_TAIL_RMS_VEL_RADS
            and hold_final_vel <= HOLD_FINAL_VEL_RADS
        )
        if not hold_ok:
            passed = False

        # ── 2. Step 응답 테스트 ───────────────────────────────────────────
        step_pos_list = []
        step_vel_list = []
        for _ in range(args.steps_step):
            env.step(step_action)
            step_pos_list.append(robot.data.joint_pos[0].cpu().numpy())
            step_vel_list.append(robot.data.joint_vel[0].cpu().numpy())

        step_pos = np.array(step_pos_list)     # (T, 6)
        step_vel = np.array(step_vel_list)     # (T, 6)

        _check_finite(step_pos, "step_pos")
        _check_finite(step_vel, "step_vel")

        # 최종 추종 오차 (마지막 20 스텝 평균)
        tail = max(1, min(20, args.steps_step // 12))
        final_pos = step_pos[-tail:].mean(axis=0)   # (6,)
        final_err = np.abs(final_pos - STEP_TARGET)  # (6,)
        step_final_err_max = float(np.max(final_err))

        # Overshoot: 목표 방향을 기준으로 초과한 정도
        # target > 0: pos > target → overshoot; target < 0: pos < target → overshoot
        overshoot_per_joint = []
        for j in range(6):
            t = STEP_TARGET[j]
            if abs(t) < 1e-6:
                overshoot_per_joint.append(0.0)
                continue
            if t > 0:
                ov = float(np.max(step_pos[:, j] - t))
            else:
                ov = float(np.max(t - step_pos[:, j]))
            overshoot_per_joint.append(max(0.0, ov))
        step_overshoot_max = float(max(overshoot_per_joint))

        # 최대 절대 속도
        step_max_vel = float(np.max(np.abs(step_vel)))

        # 진동 프록시: 각 관절에서 (pos - target)의 부호 변화 횟수 합산
        sign_changes_total = 0
        for j in range(6):
            err_seq = step_pos[:, j] - STEP_TARGET[j]
            signs = np.sign(err_seq)
            sc = int(np.sum(np.abs(np.diff(signs)) > 0))
            sign_changes_total += sc

        vel_limit_threshold = arm_vel_limit * STEP_VEL_FACTOR
        step_ok = (
            step_final_err_max <= STEP_FINAL_ERR_RAD
            and step_overshoot_max <= STEP_OVERSHOOT_RAD
            and step_max_vel <= vel_limit_threshold
        )
        if not step_ok:
            passed = False

        env.close()

        # ── 결과 출력 ─────────────────────────────────────────────────────
        result = {
            "task_id": "TA.1",
            "task": args.task,
            "status": "passed" if passed else "failed",
            "hold": {
                "steps": args.steps_hold,
                "settle_skip": settle_skip,
                "tail_window": tail_window,
                "max_abs_pos_rad": round(hold_max_pos, 5),
                "max_abs_vel_rads": round(hold_max_vel, 5),
                "tail_max_abs_pos_rad": round(hold_tail_max_pos, 5),
                "tail_rms_vel_rads": round(hold_tail_rms_vel, 5),
                "final_abs_vel_rads": round(hold_final_vel, 5),
                "threshold_tail_pos_rad": HOLD_TAIL_MAX_POS_RAD,
                "threshold_tail_rms_vel_rads": HOLD_TAIL_RMS_VEL_RADS,
                "threshold_final_vel_rads": HOLD_FINAL_VEL_RADS,
                "ok": hold_ok,
            },
            "step": {
                "steps": args.steps_step,
                "target_rad": STEP_TARGET.tolist(),
                "final_err_per_joint_rad": [round(float(e), 5) for e in final_err],
                "final_err_max_rad": round(step_final_err_max, 5),
                "overshoot_per_joint_rad": [round(float(v), 5) for v in overshoot_per_joint],
                "overshoot_max_rad": round(step_overshoot_max, 5),
                "max_abs_vel_rads": round(step_max_vel, 5),
                "vel_limit_threshold_rads": round(float(vel_limit_threshold), 5),
                "oscillation_sign_changes": sign_changes_total,
                "threshold_final_err_rad": STEP_FINAL_ERR_RAD,
                "threshold_overshoot_rad": STEP_OVERSHOOT_RAD,
                "ok": step_ok,
            },
        }
        print(json.dumps(result, indent=2))
        sys.stdout.flush()

        if not passed:
            sys.exit(1)

    except Exception as exc:
        tb = traceback.format_exc()
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        result = {
            "task_id": "TA.1",
            "task": args.task,
            "status": "failed",
            "error": str(exc),
            "traceback": tb,
        }
        print(json.dumps(result, indent=2))
        sys.stdout.flush()
        sys.exit(1)

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
