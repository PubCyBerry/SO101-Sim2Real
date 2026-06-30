"""실 SO-101 follower(Feetech) ↔ Isaac Sim joint 간 **측정된** calibration contract.

`leader_calibration` 이 실 *leader* 정규화 ↔ sim 을 다룬다면, 이 모듈은 실 *follower*
관절값 ↔ sim 관절 의 per-joint affine 을 다룬다. 둘은 다른 장치·다른 영점이다.

배경: 실기기 녹화 데이터셋(arm degree, gripper [0,100])을 Isaac Sim 으로 replay 할 때,
실 follower 영점/스케일(`so101_robot.json` homing_offset)이 sim URDF 영점과 어긋나
같은 관절 숫자가 다른 물리 자세 → grasp 순간 EE 가 ~2.4cm 떠 큐브를 헛집는다.
(진단·측정 절차: `docs/SIM_REAL_REPLAY_CALIBRATION.md`.)

해결: 6축(arm 5 + gripper) per-joint affine 으로 real↔sim 을 보정. 변환 2개가 아니라
affine **1개 + 역산** 으로 양방향을 모두 지원한다.

    sim_deg_j = a_j * real_j + b_j          # forward (replay,  Real→Sim)
    real_j    = (sim_deg_j - b_j) / a_j     # inverse (배포,    Sim→Real)

- arm(0-4):  real_j = 실 follower degree,        sim_deg_j = Isaac URDF degree.
- gripper(5): real_j = 실 gripper [0,100],        sim_deg_j = sim gripper degree([-10,100]).

`FOLLOWER_AFFINE_A/B` 가 이 보정의 **단일 진실 소스**(device-specific). 재캘리브레이션·
로봇 교체 시 이 두 상수만 갱신한다. 기본 상수는 **현 feature_codec 동작과 동일**
(arm 1:1 degree, gripper [0,100]→[-10,100]°)이라 측정 전에도 무해한 no-op 보정이다.
"""

from __future__ import annotations

import math

import numpy as np

from .feature_codec import SO101_JOINT_ORDER

_DEG_TO_RAD = math.pi / 180.0
_RAD_TO_DEG = 180.0 / math.pi

# ── 측정된 per-joint affine (단일 진실 소스). sim_deg = A * real + B ──────────
# 순서 = SO101_JOINT_ORDER. arm(0-4) real=follower degree, gripper(5) real=[0,100].
#
# 측정 (2026-06-30, taehun 실기기 so101_robot, scripts/ece_4560/real/read_position.py):
#   arm = 각 joint 0(URDF-zero) 자세 2회 읽기 평균 → B[:5] = -real_home (a=1, offset only).
#     real_home = [-7.868, -4.484, 4.440, 4.176, -5.275]
#     ⚠ 손맞춤 재현오차: pan/wrist_roll 스냅샷 간 ~5-6° → reach_probe 잔차 크면 그 둘 재수집.
#   gripper = 끝점 2점 → real [0,100] 을 sim gripper [-10,100]° 로 매핑:
#     완전닫힘 real=6.085 → sim=-10° ,  완전열림 real=100.0 → sim=+100°
#     a_g = (100-(-10))/(100-6.085) = 1.17127 ,  b_g = -10 - a_g*6.085 = -17.127
#
# no-op 기준값(재캘리브레이션 출발점): arm A=1/B=0, gripper A=1.1/B=-10 (=feature_codec).
FOLLOWER_AFFINE_A = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.171267], dtype=np.float64)
FOLLOWER_AFFINE_B = np.array(
    [7.868132, 4.483516, -4.439560, -4.175824, 5.274725, -17.126755], dtype=np.float64
)


def _as_joint_array(values) -> np.ndarray:
    """dict{name: value} 또는 마지막 축이 6인 array 를 (..., 6) float64 로 정규화."""
    if isinstance(values, dict):
        array = np.asarray([values[j] for j in SO101_JOINT_ORDER], dtype=np.float64)
    else:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 0 or array.shape[-1] != len(SO101_JOINT_ORDER):
            raise ValueError(f"SO-101 joint array shape must end in 6, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("SO-101 joint array contains NaN or infinity")
    return array


def real_follower_to_sim_radians(joint_state) -> np.ndarray:
    """실 follower 관절값 → Isaac Sim joint radian (replay, Real→Sim).

    Args:
        joint_state: dict 또는 (..., 6) array. arm=follower degree, gripper=[0,100].
    Returns:
        (..., 6) radian array.
    """
    v = _as_joint_array(joint_state)
    sim_deg = FOLLOWER_AFFINE_A * v + FOLLOWER_AFFINE_B
    return (sim_deg * _DEG_TO_RAD).astype(np.float32)


def sim_radians_to_real_follower(values_rad) -> np.ndarray:
    """Isaac Sim joint radian → 실 follower 관절값 (배포, Sim→Real).

    `real_follower_to_sim_radians` 의 역.

    Args:
        values_rad: (..., 6) radian array.
    Returns:
        (..., 6) array. arm=follower degree, gripper=[0,100].
    """
    sim_deg = _as_joint_array(values_rad) * _RAD_TO_DEG
    return ((sim_deg - FOLLOWER_AFFINE_B) / FOLLOWER_AFFINE_A).astype(np.float32)


def fit_follower_affine(real, sim_deg):
    """매칭 포즈에서 per-joint affine (a, b) 적합 + 잔차 리포트.

    각 관절 독립 1차 최소제곱. 변하지 않는 관절(real range < 1e-6)은 offset-only
    (a=1, b=mean(sim-real)) 로 폴백하고 태그 — HOME-only 단일 포즈 수집도 그대로 처리한다.

    Args:
        real:    (N, 6) 실 follower 읽기. arm=degree, gripper=[0,100]. (N>=1)
        sim_deg: (N, 6) 매칭된 sim 관절 degree. arm=URDF degree, gripper=sim gripper degree.
    Returns:
        (A, B): 각 (6,) float64. FOLLOWER_AFFINE_A/B 에 붙여넣을 값.
    """
    real = np.atleast_2d(np.asarray(real, dtype=np.float64))
    sim_deg = np.atleast_2d(np.asarray(sim_deg, dtype=np.float64))
    if real.shape != sim_deg.shape or real.shape[-1] != 6:
        raise ValueError(f"real/sim shape must be (N,6) and equal, got {real.shape} {sim_deg.shape}")

    A, B = np.ones(6), np.zeros(6)
    print(f"[fit] N={len(real)} poses — joint: a, b, max|resid|(deg)")
    for j, name in enumerate(SO101_JOINT_ORDER):
        x, y = real[:, j], sim_deg[:, j]
        if x.max() - x.min() < 1e-6:  # 안 변함 → scale 미결정, offset only
            A[j], B[j], tag = 1.0, float(np.mean(y - x)), " (const→offset only)"
        else:
            A[j], B[j], tag = (*np.polyfit(x, y, 1), "")
        resid = float(np.max(np.abs(A[j] * x + B[j] - y)))
        print(f"  {name:14s} a={A[j]:+.4f} b={B[j]:+8.3f} resid={resid:.3f}{tag}")
    return A, B


def _self_check() -> None:
    from .feature_codec import policy_feature_to_sim_joint_radians

    global FOLLOWER_AFFINE_A, FOLLOWER_AFFINE_B
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.uniform(-90, 90, 5), [rng.uniform(0, 100)]]).astype(np.float32)

    A0, B0 = FOLLOWER_AFFINE_A.copy(), FOLLOWER_AFFINE_B.copy()
    try:
        # 1) no-op 상수(arm 1:1, gripper 1.1/-10)는 feature_codec 와 동일해야 한다.
        FOLLOWER_AFFINE_A = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.1])
        FOLLOWER_AFFINE_B = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -10.0])
        assert np.allclose(
            real_follower_to_sim_radians(x), policy_feature_to_sim_joint_radians(x), atol=1e-5
        ), "no-op 상수가 feature_codec 와 불일치"

        # 2) 임의 affine round-trip 항등 (forward → inverse → 원본).
        FOLLOWER_AFFINE_A = rng.uniform(0.8, 1.2, 6)
        FOLLOWER_AFFINE_B = rng.uniform(-20, 20, 6)
        back = sim_radians_to_real_follower(real_follower_to_sim_radians(x))
        assert np.allclose(back, x, atol=1e-4), f"round-trip 불일치: {back} vs {x}"
    finally:
        FOLLOWER_AFFINE_A, FOLLOWER_AFFINE_B = A0, B0

    # 3) fit 이 주입한 affine 을 복원하는지.
    real = rng.uniform(-80, 80, (5, 6))
    real[:, 5] = rng.uniform(0, 100, 5)
    A_true, B_true = rng.uniform(0.8, 1.2, 6), rng.uniform(-15, 15, 6)
    A_fit, B_fit = fit_follower_affine(real, real * A_true + B_true)
    assert np.allclose(A_fit, A_true, atol=1e-6) and np.allclose(B_fit, B_true, atol=1e-4), "fit 복원 실패"

    print("[follower_calibration] self-check OK")


if __name__ == "__main__":
    _self_check()
