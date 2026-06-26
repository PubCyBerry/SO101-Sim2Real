"""SO-101 canonical policy feature와 Isaac joint radian 사이의 변환."""

from __future__ import annotations

import math

import numpy as np

CODEC_VERSION = "so101_joint_position_v1"

FPS = 30
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
IMAGE_CHANNELS = 3
CAMERA_KEYS = ("top", "wrist", "front")

SO101_JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
JOINT_FEATURE_NAMES = tuple(f"{joint}.pos" for joint in SO101_JOINT_ORDER)

# Canonical PolicyFeature v1:
# - arm: physical joint angle in degrees
# - gripper: calibrated fraction [0, 100]
POLICY_GRIPPER_RANGE = (0.0, 100.0)
SIM_GRIPPER_RANGE_DEG = (-10.0, 100.0)

SIM_JOINT_LIMITS_RAD = np.asarray(
    [
        [-math.pi, math.pi],
        [-math.pi, math.pi],
        [-math.pi, math.pi],
        [-math.pi, math.pi],
        [-math.pi, math.pi],
        [math.radians(SIM_GRIPPER_RANGE_DEG[0]), math.radians(SIM_GRIPPER_RANGE_DEG[1])],
    ],
    dtype=np.float32,
)

_RAD_TO_DEG = 180.0 / math.pi
_DEG_TO_RAD = math.pi / 180.0


def _as_joint_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 0 or array.shape[-1] != len(SO101_JOINT_ORDER):
        raise ValueError(f"SO-101 joint array shape must end in 6, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("SO-101 joint array contains NaN or infinity")
    return array.copy()


def sim_joint_radians_to_policy_feature(values_rad: np.ndarray) -> np.ndarray:
    """Isaac joint radian을 canonical PolicyFeature v1으로 변환한다."""
    values = _as_joint_array(values_rad)
    values[..., :5] *= _RAD_TO_DEG

    gripper_deg = values[..., 5] * _RAD_TO_DEG
    sim_lo, sim_hi = SIM_GRIPPER_RANGE_DEG
    feature_lo, feature_hi = POLICY_GRIPPER_RANGE
    fraction = (gripper_deg - sim_lo) / (sim_hi - sim_lo)
    values[..., 5] = feature_lo + fraction * (feature_hi - feature_lo)
    return values.astype(np.float32)


def policy_feature_to_sim_joint_radians(values_feature: np.ndarray) -> np.ndarray:
    """Canonical PolicyFeature v1을 Isaac joint radian으로 변환한다."""
    values = _as_joint_array(values_feature)
    values[..., :5] *= _DEG_TO_RAD

    feature_lo, feature_hi = POLICY_GRIPPER_RANGE
    sim_lo, sim_hi = SIM_GRIPPER_RANGE_DEG
    fraction = (values[..., 5] - feature_lo) / (feature_hi - feature_lo)
    gripper_deg = sim_lo + fraction * (sim_hi - sim_lo)
    values[..., 5] = gripper_deg * _DEG_TO_RAD
    return values.astype(np.float32)


def clamp_sim_joint_radians(values_rad: np.ndarray) -> np.ndarray:
    """Isaac target을 현재 SO-101 articulation joint limit에 맞춘다."""
    values = _as_joint_array(values_rad)
    return np.clip(values, SIM_JOINT_LIMITS_RAD[:, 0], SIM_JOINT_LIMITS_RAD[:, 1]).astype(np.float32)


