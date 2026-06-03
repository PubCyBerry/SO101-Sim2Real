"""TA.2 scene spawn/physics smoke: pen ellipse regions, PenCup arc, y-separation, settle stability.

Usage:
    uv run python scripts/environments/scene_physics_smoke.py \
        --task SimToReal-SO101-PickPen-v0 \
        --resets 100 --settle-steps 30 --num_envs 1 --device cuda:0
"""

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import json
import math
import sys
import traceback

from isaaclab.app import AppLauncher

# -- 검증 임계값 ---------------------------------------------------------------
# 타원 (pens)
_PEN_X_RADIUS = 0.05
_PEN_Y_RADIUS = 0.02
_ELLIPSE_EPSILON = 0.02   # 수치 오차 여유; 타원 방정식 값 <= 1 + epsilon 허용

# 호 (PenCup)
_CUP_RADIUS = 0.44
_CUP_ANGLE_MAX_DEG = 20.0
_ARC_RADIUS_TOL = 0.015   # 호 반경 편차 허용 (m)
_ARC_ANGLE_TOL_DEG = 1.0  # 호 각도 초과 허용 (deg)

# y 분리 마진
# world-frame 펜 y 최대 ≈ -0.29, 컵 y 최소 ≈ -0.196.
# 이론 마진 ≈ 0.09 m; drift 여유를 두고 0.06 m 를 하한으로 사용.
_Y_SEP_THRESHOLD = 0.06

# 물리 안정성
_Z_DROP_TOL = 0.05        # default_z 대비 허용 하강량 (m)
_LIN_VEL_THRESH = 0.5     # 선속도 노름 상한 (m/s)
_ANG_VEL_THRESH = 5.0     # 각속도 노름 상한 (rad/s)
_XY_DRIFT_WARN = 0.20     # settle 후 root xy drift 참고 경고값 (m)

# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="TA.2 scene spawn/physics smoke")
parser.add_argument("--task", default="SimToReal-SO101-PickPen-v0")
parser.add_argument("--resets", type=int, default=100)
parser.add_argument("--settle-steps", type=int, default=30)
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

_PEN_NAMES = ["PenWhite", "PenGray", "PenBlack", "PenBlue"]
_CUP_NAME = "PenCup"


# ---------------------------------------------------------------------------
# 기하 검증 헬퍼
# ---------------------------------------------------------------------------


def _ellipse_val(pos_local: torch.Tensor, default_local: torch.Tensor) -> torch.Tensor:
    """(dx/Rx)^2 + (dy/Ry)^2 for each env. Returns shape (N,)."""
    dx = pos_local[:, 0] - default_local[:, 0]
    dy = pos_local[:, 1] - default_local[:, 1]
    return (dx / _PEN_X_RADIUS) ** 2 + (dy / _PEN_Y_RADIUS) ** 2


def _arc_deviations(
    pos_local: torch.Tensor,
    default_local: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """호 편차 반환: (반경 편차 m, 각도 deg).

    center = (default_x, default_y - CUP_RADIUS).
    angle = atan2(dx, dy) — +y 축 기준 반시계 양수.
    """
    center_x = default_local[:, 0]
    center_y = default_local[:, 1] - _CUP_RADIUS
    dx = pos_local[:, 0] - center_x
    dy = pos_local[:, 1] - center_y
    dist = torch.sqrt(dx ** 2 + dy ** 2)
    radius_dev = torch.abs(dist - _CUP_RADIUS)
    angle_deg = torch.atan2(dx, dy) * (180.0 / math.pi)
    return radius_dev, angle_deg


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    device: str = args.device
    env = None

    # 누적 통계
    per_pen_max_ellipse: dict[str, float] = {n: 0.0 for n in _PEN_NAMES}
    cup_max_radius_dev: float = 0.0
    cup_max_angle_overshoot_deg: float = 0.0  # max(0, |angle| - 20)
    min_spawn_y_sep: float = float("inf")
    min_settled_y_sep: float = float("inf")
    max_xy_drift: float = 0.0
    max_z_drop: float = 0.0
    max_lin_vel: float = 0.0
    max_ang_vel: float = 0.0
    worst_xy_drift: dict[str, object] | None = None
    worst_lin_vel: dict[str, object] | None = None
    worst_ang_vel: dict[str, object] | None = None
    default_xy_by_object: dict[str, list[float]] = {}
    first_spawn_xy_by_object: dict[str, list[float]] = {}
    first_settled_xy_by_object: dict[str, list[float]] = {}
    nan_triggered = False
    z_drop_triggered = False
    xy_drift_warn_triggered = False
    vel_triggered = False
    passed = True

    try:
        env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
        # 에피소드를 리셋 루프 전체를 수용할 만큼 늘림 (자동 리셋 방지)
        total_steps = args.resets * (args.settle_steps + 1) + 10
        env_cfg.episode_length_s = max(
            env_cfg.episode_length_s,
            total_steps * env_cfg.sim.dt * env_cfg.decimation + 5.0,
        )

        env = gym.make(args.task, cfg=env_cfg)
        scene = env.unwrapped.scene
        origins = scene.env_origins  # (num_envs, 3)

        pen_objects = {n: scene[n] for n in _PEN_NAMES}
        cup_object = scene[_CUP_NAME]
        all_named_objects = {**pen_objects, _CUP_NAME: cup_object}
        default_xy_by_object = {
            name: [round(float(v), 5) for v in obj.data.default_root_state[0, :2].tolist()]
            for name, obj in all_named_objects.items()
        }

        zero_action = torch.zeros(args.num_envs, 6, device=device)

        for reset_idx in range(args.resets):
            env.reset()

            # ── 스폰 직후: 타원 / 호 검증 ──────────────────────────────
            spawn_pos_by_name: dict[str, torch.Tensor] = {}
            spawn_pen_y_max_env = -float("inf")
            for name, obj in pen_objects.items():
                pos_local = obj.data.root_pos_w - origins          # (N, 3)
                spawn_pos_by_name[name] = pos_local.clone()
                if reset_idx == 0:
                    first_spawn_xy_by_object[name] = [
                        round(float(v), 5) for v in pos_local[0, :2].tolist()
                    ]
                default_local = obj.data.default_root_state[:, :3] # (N, 3)
                ev = _ellipse_val(pos_local, default_local)
                max_ev = float(ev.max().item())
                if max_ev > per_pen_max_ellipse[name]:
                    per_pen_max_ellipse[name] = max_ev
                if max_ev > 1.0 + _ELLIPSE_EPSILON:
                    passed = False
                py = float(pos_local[:, 1].max().item())
                if py > spawn_pen_y_max_env:
                    spawn_pen_y_max_env = py

            cup_local = cup_object.data.root_pos_w - origins
            spawn_pos_by_name[_CUP_NAME] = cup_local.clone()
            if reset_idx == 0:
                first_spawn_xy_by_object[_CUP_NAME] = [
                    round(float(v), 5) for v in cup_local[0, :2].tolist()
                ]
            cup_default_local = cup_object.data.default_root_state[:, :3]
            radius_dev, angle_deg = _arc_deviations(cup_local, cup_default_local)

            max_rad_dev = float(radius_dev.max().item())
            # 각도 초과량: max(0, |angle| - 20°)
            angle_overshoot = float(
                (torch.abs(angle_deg) - _CUP_ANGLE_MAX_DEG).clamp(min=0.0).max().item()
            )
            if max_rad_dev > cup_max_radius_dev:
                cup_max_radius_dev = max_rad_dev
            if angle_overshoot > cup_max_angle_overshoot_deg:
                cup_max_angle_overshoot_deg = angle_overshoot
            if max_rad_dev > _ARC_RADIUS_TOL:
                passed = False
            if angle_overshoot > _ARC_ANGLE_TOL_DEG:
                passed = False

            spawn_cup_y_min_env = float(cup_local[:, 1].min().item())
            spawn_sep = spawn_cup_y_min_env - spawn_pen_y_max_env
            if spawn_sep < min_spawn_y_sep:
                min_spawn_y_sep = spawn_sep
            if spawn_sep < _Y_SEP_THRESHOLD:
                passed = False

            # ── settle ──────────────────────────────────────────────────
            for _ in range(args.settle_steps):
                env.step(zero_action)

            # ── 정착 후: y 분리 참고값 + 물리 안정성 ───────────────────
            pen_y_max_env = -float("inf")
            for obj in pen_objects.values():
                pos_local = obj.data.root_pos_w - origins
                py = float(pos_local[:, 1].max().item())
                if py > pen_y_max_env:
                    pen_y_max_env = py

            cup_local_settled = cup_object.data.root_pos_w - origins
            cup_y_min_env = float(cup_local_settled[:, 1].min().item())
            sep = cup_y_min_env - pen_y_max_env
            if sep < min_settled_y_sep:
                min_settled_y_sep = sep
            if sep < _Y_SEP_THRESHOLD:
                passed = False

            # NaN/Inf + z 하강 + xy drift + 속도 점검
            for name, obj in all_named_objects.items():
                pos_w = obj.data.root_pos_w
                lin_vel = obj.data.root_lin_vel_w
                ang_vel = obj.data.root_ang_vel_w

                if (
                    not torch.isfinite(pos_w).all()
                    or not torch.isfinite(lin_vel).all()
                    or not torch.isfinite(ang_vel).all()
                ):
                    nan_triggered = True
                    passed = False

                pos_local = pos_w - origins
                if reset_idx == 0:
                    first_settled_xy_by_object[name] = [
                        round(float(v), 5) for v in pos_local[0, :2].tolist()
                    ]
                default_local = obj.data.default_root_state[:, :3]
                z_drop = default_local[:, 2] - pos_local[:, 2]
                z_drop_max = float(z_drop.max().item())
                if z_drop_max > max_z_drop:
                    max_z_drop = z_drop_max
                if z_drop_max > _Z_DROP_TOL:
                    z_drop_triggered = True
                    passed = False

                xy_drift = torch.linalg.norm(pos_local[:, :2] - spawn_pos_by_name[name][:, :2], dim=-1)
                xy_drift_max = float(xy_drift.max().item())
                if xy_drift_max > max_xy_drift:
                    max_xy_drift = xy_drift_max
                    worst_xy_drift = {"reset": reset_idx, "object": name, "xy_drift_m": round(xy_drift_max, 5)}
                if xy_drift_max > _XY_DRIFT_WARN:
                    xy_drift_warn_triggered = True

                lin_norm = float(torch.linalg.norm(lin_vel, dim=-1).max().item())
                ang_norm = float(torch.linalg.norm(ang_vel, dim=-1).max().item())
                if lin_norm > max_lin_vel:
                    max_lin_vel = lin_norm
                    worst_lin_vel = {"reset": reset_idx, "object": name, "lin_vel_ms": round(lin_norm, 5)}
                if ang_norm > max_ang_vel:
                    max_ang_vel = ang_norm
                    worst_ang_vel = {"reset": reset_idx, "object": name, "ang_vel_rads": round(ang_norm, 5)}
                if lin_norm > _LIN_VEL_THRESH or ang_norm > _ANG_VEL_THRESH:
                    vel_triggered = True
                    passed = False

        env.close()

        result = {
            "task_id": "TA.2",
            "task": args.task,
            "status": "passed" if passed else "failed",
            "config": {
                "resets": args.resets,
                "settle_steps": args.settle_steps,
                "num_envs": args.num_envs,
                "default_xy_by_object": default_xy_by_object,
                "first_spawn_xy_by_object": first_spawn_xy_by_object,
                "first_settled_xy_by_object": first_settled_xy_by_object,
            },
            "spawn_ellipse": {
                "x_radius_m": _PEN_X_RADIUS,
                "y_radius_m": _PEN_Y_RADIUS,
                "epsilon": _ELLIPSE_EPSILON,
                "max_ellipse_val_per_pen": {k: round(v, 5) for k, v in per_pen_max_ellipse.items()},
                "pass": all(v <= 1.0 + _ELLIPSE_EPSILON for v in per_pen_max_ellipse.values()),
            },
            "spawn_arc": {
                "radius_m": _CUP_RADIUS,
                "angle_range_deg": [-_CUP_ANGLE_MAX_DEG, _CUP_ANGLE_MAX_DEG],
                "radius_tol_m": _ARC_RADIUS_TOL,
                "angle_tol_deg": _ARC_ANGLE_TOL_DEG,
                "max_radius_deviation_m": round(cup_max_radius_dev, 5),
                "max_angle_overshoot_deg": round(cup_max_angle_overshoot_deg, 5),
                "pass": cup_max_radius_dev <= _ARC_RADIUS_TOL and cup_max_angle_overshoot_deg <= _ARC_ANGLE_TOL_DEG,
            },
            "y_separation": {
                "threshold_m": _Y_SEP_THRESHOLD,
                "note": "threshold=0.06: world pen_y_max≈-0.29, cup_y_min≈-0.196 → ≈0.09 m authored margin",
                "min_spawn_observed_m": round(min_spawn_y_sep, 5) if min_spawn_y_sep != float("inf") else None,
                "min_settled_observed_m": round(min_settled_y_sep, 5) if min_settled_y_sep != float("inf") else None,
                "pass": min_spawn_y_sep >= _Y_SEP_THRESHOLD and min_settled_y_sep >= _Y_SEP_THRESHOLD,
            },
            "physics_stability": {
                "z_drop_tol_m": _Z_DROP_TOL,
                "xy_drift_warn_m": _XY_DRIFT_WARN,
                "lin_vel_thresh_ms": _LIN_VEL_THRESH,
                "ang_vel_thresh_rads": _ANG_VEL_THRESH,
                "max_z_drop_m": round(max_z_drop, 5),
                "max_xy_drift_m": round(max_xy_drift, 5),
                "max_lin_vel_ms": round(max_lin_vel, 5),
                "max_ang_vel_rads": round(max_ang_vel, 5),
                "worst_xy_drift": worst_xy_drift,
                "worst_lin_vel": worst_lin_vel,
                "worst_ang_vel": worst_ang_vel,
                "nan_triggered": nan_triggered,
                "z_drop_triggered": z_drop_triggered,
                "xy_drift_warn_triggered": xy_drift_warn_triggered,
                "vel_triggered": vel_triggered,
                "pass": not nan_triggered and not z_drop_triggered and not vel_triggered,
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
            "task_id": "TA.2",
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
