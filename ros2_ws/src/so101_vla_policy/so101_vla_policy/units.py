"""SO-101 sim(rad) ↔ 실기기 LeRobot 단위 변환 + 상수 (numpy-only).

`scripts/sim/lerobot_units.py` 의 **미러**다(컨테이너 py3.12 격리 — isaac venv 의
scripts 를 import 하지 않기 위해 vendoring). 단위 규약이 바뀌면 두 파일을 같이 고친다.

- 팔 5축: radian(sim) ↔ degree(LeRobot)
- 그리퍼: radian(sim) ↔ [0,100] 정규화(LeRobot), scale = 31.75
"""

from __future__ import annotations

import math

import numpy as np

SO101_JOINT_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
JOINT_FEATURE_NAMES = [f"{j}.pos" for j in SO101_JOINT_ORDER]
CAMERA_KEYS = ("top", "wrist", "front")
GRIPPER_LEROBOT_SCALE = 31.75

_RAD_TO_DEG = 180.0 / math.pi
_DEG_TO_RAD = math.pi / 180.0

# joint position 한계(rad). teleop _joint_limits 와 동일 — arm ±pi, gripper [-10°,100°].
JOINT_LIMITS_RAD = np.array(
    [
        [-math.pi, math.pi],
        [-math.pi, math.pi],
        [-math.pi, math.pi],
        [-math.pi, math.pi],
        [-math.pi, math.pi],
        [math.radians(-10.0), math.radians(100.0)],
    ],
    dtype=np.float32,
)


def to_lerobot_units(values_rad: np.ndarray) -> np.ndarray:
    """sim joint radian → LeRobot SO-101 단위(arm deg, gripper [0,100])."""
    v = np.asarray(values_rad, dtype=np.float32).copy()
    v[:5] = v[:5] * _RAD_TO_DEG
    v[5] = v[5] * GRIPPER_LEROBOT_SCALE
    return v.astype(np.float32)


def from_lerobot_units(values_lerobot: np.ndarray) -> np.ndarray:
    """LeRobot SO-101 단위 → sim joint radian (to_lerobot_units 역변환)."""
    v = np.asarray(values_lerobot, dtype=np.float32).copy()
    v[:5] = v[:5] * _DEG_TO_RAD
    v[5] = v[5] / GRIPPER_LEROBOT_SCALE
    return v.astype(np.float32)


def clamp_joint_rad(values_rad: np.ndarray) -> np.ndarray:
    v = np.asarray(values_rad, dtype=np.float32)
    return np.minimum(np.maximum(v, JOINT_LIMITS_RAD[:, 0]), JOINT_LIMITS_RAD[:, 1]).astype(np.float32)
