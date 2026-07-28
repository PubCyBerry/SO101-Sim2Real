"""SO-101 policy I/O contract의 ROS 2 adapter.

공통 구현은 repo `src/so101_contract`에 있다. vla-ros entrypoint가 `/workspace/src`를
PYTHONPATH에 넣으므로 sim recorder, teleop, ROS가 같은 codec을 import한다.
"""

from __future__ import annotations

import numpy as np

from so101_contract.feature_codec import (
    CAMERA_KEYS,
    CODEC_VERSION,
    IMAGE_CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    JOINT_FEATURE_NAMES,
    SIM_JOINT_LIMITS_RAD,
    SO101_JOINT_ORDER,
    clamp_sim_joint_radians,
    policy_feature_to_sim_joint_radians,
    sim_joint_radians_to_policy_feature,
)
from so101_contract.eef_action_contract import CANONICAL_ACTION_NAMES

# policy-server RemotePolicyConfig.lerobot_features — 실 lerobot 의
# map_robot_keys_to_lerobot_features(SO101Follower(top/wrist/front)) 와 동일 스키마.
# 정적이라 하드코딩(SO101Follower import 회피 → lerobot 전체 의존 제거).
LEROBOT_FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (6,), "names": list(JOINT_FEATURE_NAMES)},
    **{
        f"observation.images.{camera}": {
            "dtype": "image",
            "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
            "names": ["height", "width", "channels"],
        }
        for camera in CAMERA_KEYS
    },
}

LEROBOT_EEF_FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (len(CANONICAL_ACTION_NAMES),),
        "names": list(CANONICAL_ACTION_NAMES),
    },
    **{
        f"observation.images.{camera}": {
            "dtype": "image",
            "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
            "names": ["height", "width", "channels"],
        }
        for camera in CAMERA_KEYS
    },
}

JOINT_LIMITS_RAD = SIM_JOINT_LIMITS_RAD


def to_lerobot_units(values_rad: np.ndarray) -> np.ndarray:
    """sim joint radian → canonical PolicyFeature v1."""
    return sim_joint_radians_to_policy_feature(values_rad)


def from_lerobot_units(values_lerobot: np.ndarray) -> np.ndarray:
    """Canonical PolicyFeature v1 → sim joint radian."""
    return policy_feature_to_sim_joint_radians(values_lerobot)


def clamp_joint_rad(values_rad: np.ndarray) -> np.ndarray:
    return clamp_sim_joint_radians(values_rad)
