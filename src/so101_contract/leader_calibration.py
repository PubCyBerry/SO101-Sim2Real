"""실 SO-101 leader(Feetech) ↔ Isaac Sim joint 간 calibration contract.

`feature_codec` 가 policy-feature ↔ sim radian 을 다룬다면, 이 모듈은 **실 leader 모터
정규화값 ↔ sim radian** 의 양방향 변환을 다룬다. 둘은 다른 변환이다:

- arm: 실 leader 는 ``RANGE_M100_100`` 정규화([-100, 100]) 를 내보낸다. USD joint 범위는
  관절별 **비대칭**(elbow -100/90, wrist_flex ±95 …) 이라 정규화→USD-degree 는 per-joint
  scale+offset 선형 remap 이 필요하다. (codec 의 arm 1:1 degree 처리로는 재현 불가.)
- gripper: 정규화 [0, 100] → USD degree [-10, 100]. 이 affine 은 `feature_codec` 의 gripper
  affine(``deg = feature/100*110 - 10``)과 **수식적으로 동일**하다.

원본: leisaac ``assets/robots/lerobot.py``(테이블) + ``devices/action_process.py``
(``convert_action_from_so101_leader``, 정규화→radian) + ``utils/robot_utils.py``
(``convert_leisaac_action_to_lerobot``, radian→정규화 역방향 / ``is_so101_at_rest_pose``).
leisaac 의 torch·``env.num_envs`` 결합을 제거하고 순수 numpy 6-vector 로 재작성 —
isaaclab/torch 불요라 host(uv) 와 isaac-sim Docker 양쪽에서 import 가능하다.

이 모듈이 세 테이블의 **단일 진실 소스**다. ``sim_to_real.assets.robots.lerobot`` 등
다른 곳은 여기서 import 한다.
"""

from __future__ import annotations

import math

import numpy as np

from .feature_codec import SO101_JOINT_ORDER

# joint limit written in USD (degree). 관절별 비대칭 범위.
SO101_FOLLOWER_USD_JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 90.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-160.0, 160.0),
    "gripper": (-10.0, 100.0),
}

# motor limit written in real device (normalized range). arm=RANGE_M100_100, gripper=RANGE_0_100.
SO101_FOLLOWER_MOTOR_LIMITS: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-100.0, 100.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 100.0),
    "wrist_flex": (-100.0, 100.0),
    "wrist_roll": (-100.0, 100.0),
    "gripper": (0.0, 100.0),
}

# rest pose 판정용 per-joint 허용 범위(degree). 중심값 ±30°.
SO101_FOLLOWER_REST_POSE_RANGE: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-30.0, 30.0),       # 0°
    "shoulder_lift": (-130.0, -70.0),    # -100°
    "elbow_flex": (60.0, 120.0),         # 90°
    "wrist_flex": (20.0, 80.0),          # 50°
    "wrist_roll": (-30.0, 30.0),         # 0°
    "gripper": (-40.0, 20.0),            # -10°
}

_DEG_TO_RAD = math.pi / 180.0
_RAD_TO_DEG = 180.0 / math.pi

# 테이블을 SO101_JOINT_ORDER 순서의 (6,) 배열로 펼쳐 벡터화 연산에 사용.
_MOTOR_LO = np.asarray([SO101_FOLLOWER_MOTOR_LIMITS[j][0] for j in SO101_JOINT_ORDER], dtype=np.float64)
_MOTOR_HI = np.asarray([SO101_FOLLOWER_MOTOR_LIMITS[j][1] for j in SO101_JOINT_ORDER], dtype=np.float64)
_USD_LO = np.asarray([SO101_FOLLOWER_USD_JOINT_LIMITS[j][0] for j in SO101_JOINT_ORDER], dtype=np.float64)
_USD_HI = np.asarray([SO101_FOLLOWER_USD_JOINT_LIMITS[j][1] for j in SO101_JOINT_ORDER], dtype=np.float64)
_MOTOR_RANGE = _MOTOR_HI - _MOTOR_LO
_USD_RANGE = _USD_HI - _USD_LO
_REST_LO = np.asarray([SO101_FOLLOWER_REST_POSE_RANGE[j][0] for j in SO101_JOINT_ORDER], dtype=np.float64)
_REST_HI = np.asarray([SO101_FOLLOWER_REST_POSE_RANGE[j][1] for j in SO101_JOINT_ORDER], dtype=np.float64)


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


def real_leader_to_sim_radians(joint_state) -> np.ndarray:
    """실 leader 모터 정규화값 → Isaac Sim joint radian.

    관절별 선형 remap: ``deg = (v - motor_lo)/motor_range * usd_range + usd_lo`` → radian.
    원본 leisaac ``convert_action_from_so101_leader`` 의 순수 numpy 버전.

    Args:
        joint_state: ``{joint_name: normalized_value}`` dict 또는 ``(..., 6)`` array
            (SO101_JOINT_ORDER 순서). arm 은 [-100, 100], gripper 는 [0, 100].
    Returns:
        ``(..., 6)`` radian array.
    """
    v = _as_joint_array(joint_state)
    deg = (v - _MOTOR_LO) / _MOTOR_RANGE * _USD_RANGE + _USD_LO
    return (deg * _DEG_TO_RAD).astype(np.float32)


def sim_radians_to_real_leader(values_rad) -> np.ndarray:
    """Isaac Sim joint radian → 실 leader 모터 정규화값 (정책 배포 방향).

    ``real_leader_to_sim_radians`` 의 역. 원본 leisaac ``convert_leisaac_action_to_lerobot``.

    Args:
        values_rad: ``(..., 6)`` radian array (SO101_JOINT_ORDER 순서).
    Returns:
        ``(..., 6)`` 정규화값 array (arm [-100, 100], gripper [0, 100]).
    """
    rad = _as_joint_array(values_rad)
    deg = rad * _RAD_TO_DEG
    v = (deg - _USD_LO) / _USD_RANGE * _MOTOR_RANGE + _MOTOR_LO
    return v.astype(np.float32)


def is_so101_at_rest_pose(values_rad) -> np.ndarray | bool:
    """모든 관절이 rest pose 허용 범위 안에 있는지 판정.

    원본 leisaac ``robot_utils.is_so101_at_rest_pose`` (torch→numpy).

    Args:
        values_rad: ``(..., 6)`` radian array (SO101_JOINT_ORDER 순서).
    Returns:
        ``(...)`` bool array (스칼라 입력이면 bool).
    """
    deg = _as_joint_array(values_rad) * _RAD_TO_DEG
    within = (deg > _REST_LO) & (deg < _REST_HI)
    result = np.all(within, axis=-1)
    return bool(result) if result.ndim == 0 else result
