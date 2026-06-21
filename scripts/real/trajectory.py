"""Cubic spline 관절 궤적 — ECE4560 SO-101 assignment9 (maegantucker.com).

clamped cubic(양끝 속도 0), per-DOF 독립. math 만 쓰는 순수 Python.

  p(t) = a0 + a1·t + a2·t² + a3·t³
  경계: p(0)=θ0, p(T)=θf, ṗ(0)=0, ṗ(T)=0
  계수: a0=θ0, a1=0, a2=3·Δ/T², a3=-2·Δ/T³   (Δ=θf-θ0)
  평가:  tlim = clip(t, 0, T)   (외삽 방지)

용도(실기기): 시작 자세→목표 자세를 부드러운 위치 프로파일로 재생(open-loop).
6축(arm 5 + gripper) 같은 duration 으로 동시 spline → grip 닫힘도 부드럽게.
속도(ṗ)는 feedforward 용으로 제공하나 SO-101 HW 는 위치만 받음(MuJoCo 전용).
"""

from __future__ import annotations


def cubic_coeffs(q0: list[float], qf: list[float], duration: float) -> list[tuple]:
    """per-DOF (a0,a1,a2,a3). duration<=0 이면 즉시 도달(step) 계수."""
    if duration <= 0:
        return [(pf, 0.0, 0.0, 0.0) for pf in qf]
    T2 = duration * duration
    T3 = T2 * duration
    out = []
    for p0, pf in zip(q0, qf):
        d = pf - p0
        out.append((p0, 0.0, 3.0 * d / T2, -2.0 * d / T3))
    return out


def _clip(t: float, duration: float) -> float:
    return 0.0 if t < 0.0 else (duration if t > duration else t)


def cubic_eval(coeffs: list[tuple], t: float, duration: float) -> list[float]:
    """시각 t 의 위치 벡터. t 는 [0,duration] 로 clip(외삽 방지)."""
    tl = _clip(t, duration)
    return [a0 + a1 * tl + a2 * tl * tl + a3 * tl * tl * tl for (a0, a1, a2, a3) in coeffs]


def cubic_vel(coeffs: list[tuple], t: float, duration: float) -> list[float]:
    """시각 t 의 속도 벡터(feedforward). HW 는 무시, 분석/sim 용."""
    tl = _clip(t, duration)
    return [a1 + 2.0 * a2 * tl + 3.0 * a3 * tl * tl for (a0, a1, a2, a3) in coeffs]


def sample_trajectory(q0: list[float], qf: list[float], duration: float,
                      dt: float) -> list[tuple[float, list[float]]]:
    """[(t, q)] 샘플(양끝 포함). 실시간 루프 대신 미리 펼칠 때 사용."""
    coeffs = cubic_coeffs(q0, qf, duration)
    if duration <= 0 or dt <= 0:
        return [(0.0, list(qf)), (max(duration, 0.0), list(qf))]
    n = max(1, int(round(duration / dt)))
    return [(min(i * dt, duration), cubic_eval(coeffs, min(i * dt, duration), duration))
            for i in range(n + 1)]
