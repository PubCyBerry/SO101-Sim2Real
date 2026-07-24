"""SO-101 sim-real policy I/O contract.

Isaac Sim, ROS 2, LeRobot recorder가 함께 사용하는 순수 Python 모듈이다.
Isaac Lab이나 ROS를 import하지 않으므로 각 실행 환경에서 같은 구현을 쓸 수 있다.
"""

from .action_queue import ACTION_AGGREGATE_NAMES, ActionChunkQueue, aggregate_actions
from .eef_kinematics import (
    ARM_JOINT_ORDER,
    EEF_KINEMATICS_VERSION,
    ROTATION_REPRESENTATION_DIMS,
    ROTATION_REPRESENTATIONS,
    SO101EndEffectorKinematics,
    decode_rotation_representation,
    encode_rotation_matrices,
)
from .feature_codec import (
    CAMERA_KEYS,
    CODEC_VERSION,
    FPS,
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

__all__ = [
    "ACTION_AGGREGATE_NAMES",
    "ARM_JOINT_ORDER",
    "ActionChunkQueue",
    "CAMERA_KEYS",
    "CODEC_VERSION",
    "EEF_KINEMATICS_VERSION",
    "FPS",
    "IMAGE_CHANNELS",
    "IMAGE_HEIGHT",
    "IMAGE_WIDTH",
    "JOINT_FEATURE_NAMES",
    "ROTATION_REPRESENTATION_DIMS",
    "ROTATION_REPRESENTATIONS",
    "SIM_JOINT_LIMITS_RAD",
    "SO101_JOINT_ORDER",
    "SO101EndEffectorKinematics",
    "aggregate_actions",
    "clamp_sim_joint_radians",
    "decode_rotation_representation",
    "encode_rotation_matrices",
    "policy_feature_to_sim_joint_radians",
    "sim_joint_radians_to_policy_feature",
]
