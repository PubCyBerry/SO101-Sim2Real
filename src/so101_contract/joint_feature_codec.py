"""Phase 16 correction — schema v2 canonical **joint** feature 계약.

v2 joint dataset/manifest는 다음 layout을 쓴다(legacy PolicyFeature v1과 **다르다**).

.. code-block:: text

    observation/action feature = arm[0:5] canonical radian + gripper[5] absolute policy feature [0,100]
    canonical platform joint state = 6 sim radian
    real follower 경계 = real_follower_to_sim_radians / sim_radians_to_real_follower
    sim 경계 = canonical sim radian 그대로

legacy ``to_lerobot_units``/``from_lerobot_units``(arm을 degree feature로 바꾸는 v1 codec)를
schema v2 joint state/action에 쓰면 안 된다. arm을 두 번 변환해 명령이 망가진다.

이 모듈은 그 경계 변환을 **한 곳**에 모으고, manifest joint topology의 단위 계약도 검증한다.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .feature_codec import (
    POLICY_GRIPPER_RANGE,
    SO101_JOINT_ORDER,
    clamp_sim_joint_radians,
    policy_feature_to_sim_joint_radians,
    sim_joint_radians_to_policy_feature,
)
from .follower_calibration import real_follower_to_sim_radians, sim_radians_to_real_follower

JOINT_FEATURE_CODEC_VERSION = "so101_canonical_joint_feature_v2"

#: v2 joint feature의 arm 단위. 이 값이 아니면 명령 이전에 실패한다.
CANONICAL_ARM_UNIT = "radian"
#: gripper group 의미(절대 policy feature).
CANONICAL_GRIPPER_SEMANTICS = "absolute_policy_feature_0_100"

ARM_DOF = 5
CANONICAL_DOF = len(SO101_JOINT_ORDER)
GRIPPER_INDEX = CANONICAL_DOF - 1


def _as_canonical(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape[-1] != CANONICAL_DOF:
        raise ValueError(f"{name} must end in {CANONICAL_DOF} joints, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def canonical_joint_state_to_feature(joint_radians: Any) -> np.ndarray:
    """canonical 6D sim radian → v2 joint feature(arm radian 5 + gripper feature 1).

    **arm은 변환하지 않는다.** gripper만 sim radian → policy feature ``[0,100]``.
    """
    joints = _as_canonical(joint_radians, "canonical joint state")
    gripper = sim_joint_radians_to_policy_feature(joints)[..., GRIPPER_INDEX : GRIPPER_INDEX + 1]
    return np.concatenate([joints[..., :ARM_DOF], gripper], axis=-1).astype(np.float32)


def feature_to_canonical_joint_state(feature: Any, *, clamp: bool = True) -> np.ndarray:
    """v2 joint feature → canonical 6D sim radian.

    arm은 그대로 두고 gripper feature만 sim radian으로 바꾼다.
    """
    values = _as_canonical(feature, "joint feature")
    gripper_feature = np.clip(
        values[..., GRIPPER_INDEX],
        POLICY_GRIPPER_RANGE[0],
        POLICY_GRIPPER_RANGE[1],
    )
    packed = np.zeros(values.shape, dtype=np.float32)
    packed[..., GRIPPER_INDEX] = gripper_feature
    gripper_radians = policy_feature_to_sim_joint_radians(packed)[..., GRIPPER_INDEX]
    canonical = np.concatenate(
        [values[..., :ARM_DOF], gripper_radians[..., None]],
        axis=-1,
    ).astype(np.float32)
    return clamp_sim_joint_radians(canonical) if clamp else canonical


def real_follower_state_to_feature(real_follower_state: Any) -> np.ndarray:
    """실 follower state → v2 joint feature. 경계 변환은 정확히 한 번이다."""
    canonical = real_follower_to_sim_radians(real_follower_state)
    return canonical_joint_state_to_feature(canonical)


def canonical_joint_state_to_real_follower(joint_radians: Any) -> np.ndarray:
    """canonical 6D sim radian → 실 follower command 단위(경계 변환 1회)."""
    return sim_radians_to_real_follower(_as_canonical(joint_radians, "canonical joint state"))


def sim_joint_command(joint_radians: Any) -> np.ndarray:
    """sim 경계: canonical radian을 그대로 쓴다(추가 codec 변환 없음)."""
    return _as_canonical(joint_radians, "canonical joint command").copy()


def sim_publish_command(platform_action: Any) -> np.ndarray:
    """router ``platform_actions`` 한 행 → sim publish 값.

    sim 경계에서는 **어떤 codec 변환도 하지 않고** joint limit clamp만 적용한다.
    ROS 노드와 validator가 같은 함수를 쓴다(경로 일치 보장).
    """
    return clamp_sim_joint_radians(sim_joint_command(platform_action))


# --- manifest 단위/의미 계약 ---------------------------------------------------
#
# joint runtime이 조용히 가정하지 않도록, **명시적 payload**를 manifest/transform에 싣는다.
# builder와 validator가 같은 모듈에 있는 단일 소스다.


def build_joint_feature_contract(
    joint_topology: dict[str, Any],
    *,
    gripper_index: int,
    gripper_group: str = "gripper_position",
) -> dict[str, Any]:
    """schema v2 joint feature 계약 payload를 만든다(단일 소스).

    manifest/serialized transform에 그대로 저장돼 fingerprint·manifest hash에 포함된다.
    """
    # arm_dof는 topology에서 유도한다(SO-101 runtime 강제는 validate 쪽 책임).
    names = _validate_joint_units(joint_topology, source="joint topology")
    return {
        "version": JOINT_FEATURE_CODEC_VERSION,
        "arm_unit": CANONICAL_ARM_UNIT,
        "gripper_semantics": CANONICAL_GRIPPER_SEMANTICS,
        "arm_dof": len(names),
        "gripper_index": int(gripper_index),
        "gripper_group": gripper_group,
        "gripper_range": [float(POLICY_GRIPPER_RANGE[0]), float(POLICY_GRIPPER_RANGE[1])],
        "joint_names": names,
    }


def validate_joint_feature_contract(
    payload: dict[str, Any] | None,
    *,
    joint_topology: dict[str, Any] | None = None,
    gripper_representation: str | None = None,
    action_groups: dict[str, Any] | None = None,
    action_dim: int | None = None,
    source: str = "manifest",
) -> dict[str, Any]:
    """명시적 joint feature 계약을 검증한다(누락/degree/의미·버전·index 불일치 거부).

    ``joint_topology``/``gripper_representation``/``action_groups``를 주면 교차 검증까지 한다.
    기본값을 합성해 통과시키지 않는다.
    """
    if not isinstance(payload, dict):
        raise ValueError(
            f"{source} has no explicit joint feature contract; schema v2 joint runtime requires "
            f"{JOINT_FEATURE_CODEC_VERSION} (arm_unit/gripper_semantics/arm_dof) and never "
            "assumes it"
        )
    expected = {
        "version": JOINT_FEATURE_CODEC_VERSION,
        "arm_unit": CANONICAL_ARM_UNIT,
        "gripper_semantics": CANONICAL_GRIPPER_SEMANTICS,
        "arm_dof": ARM_DOF,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"{source} joint feature contract {key}={payload.get(key)!r} != {value!r}"
            )
    gripper_index = payload.get("gripper_index")
    if not isinstance(gripper_index, int) or gripper_index != ARM_DOF:
        raise ValueError(
            f"{source} joint feature contract gripper_index={gripper_index!r} must be {ARM_DOF}"
        )
    gripper_range = payload.get("gripper_range")
    if list(gripper_range or []) != [float(POLICY_GRIPPER_RANGE[0]), float(POLICY_GRIPPER_RANGE[1])]:
        raise ValueError(
            f"{source} joint feature contract gripper_range={gripper_range!r} != "
            f"{list(POLICY_GRIPPER_RANGE)}"
        )

    if joint_topology is not None:
        units = validate_joint_unit_contract(
            joint_topology,
            source=source,
            expected_arm_dof=ARM_DOF,
        )
        if list(payload.get("joint_names") or []) != list(units["joint_names"]):
            raise ValueError(
                f"{source} joint feature contract names {payload.get('joint_names')} disagree "
                f"with the topology {units['joint_names']}"
            )
    if gripper_representation is not None and gripper_representation != "absolute":
        raise ValueError(
            f"{source} declares gripper_representation={gripper_representation!r}; schema v2 "
            "joint features carry an absolute gripper policy feature"
        )
    if action_groups is not None:
        group_name = payload.get("gripper_group", "gripper_position")
        bounds = action_groups.get(group_name)
        if bounds is None:
            raise ValueError(
                f"{source} action feature has no {group_name!r} group for the gripper contract"
            )
        start, end = (bounds["start"], bounds["end"]) if isinstance(bounds, dict) else bounds
        if int(end) - int(start) != 1:
            raise ValueError(
                f"{source} gripper group {group_name!r} must hold exactly one feature, got "
                f"{int(end) - int(start)}"
            )
        if int(start) != gripper_index:
            raise ValueError(
                f"{source} gripper group starts at {start} but the contract declares index "
                f"{gripper_index}"
            )
    if action_dim is not None and action_dim != ARM_DOF + 1:
        raise ValueError(
            f"{source} joint action dim {action_dim} != {ARM_DOF + 1} (arm + gripper)"
        )
    return dict(payload)


# --- manifest 단위 계약 -------------------------------------------------------


def _validate_joint_units(
    joint_topology: dict[str, Any] | None,
    *,
    source: str,
) -> list[str]:
    """joint별 단위만 검증하고 이름을 돌려준다(개수 제약 없음)."""
    if not isinstance(joint_topology, dict):
        raise ValueError(
            f"{source} has no joint topology; schema v2 joint runtime requires an explicit "
            "arm unit contract (radian) and cannot infer it"
        )
    joints = joint_topology.get("joints")
    if not isinstance(joints, list) or not joints:
        raise ValueError(f"{source} joint topology declares no joints")
    names: list[str] = []
    for joint in joints:
        if not isinstance(joint, dict):
            raise ValueError(f"{source} joint topology entries must be objects")
        unit = joint.get("unit")
        if unit is None:
            raise ValueError(
                f"{source} joint {joint.get('name')!r} has no unit; the arm unit contract "
                f"must be explicit ({CANONICAL_ARM_UNIT})"
            )
        if unit != CANONICAL_ARM_UNIT:
            raise ValueError(
                f"{source} joint {joint.get('name')!r} unit is {unit!r}; SO-101 schema v2 "
                f"canonical joint features are {CANONICAL_ARM_UNIT}. Refusing to interpret "
                "them as radians."
            )
        names.append(joint.get("name"))
    return names


def validate_joint_unit_contract(
    joint_topology: dict[str, Any] | None,
    *,
    source: str = "manifest",
    expected_arm_dof: int | None = ARM_DOF,
) -> dict[str, Any]:
    """joint topology가 v2 canonical 단위 계약을 만족하는지 검증한다.

    arm joint 단위가 ``radian``이 아니거나 topology가 없으면 **명령 이전에** 실패한다.
    degree를 radian으로 조용히 해석하지 않는다.
    """
    names = _validate_joint_units(joint_topology, source=source)
    if expected_arm_dof is not None and len(names) != expected_arm_dof:
        raise ValueError(
            f"{source} joint topology must declare {expected_arm_dof} arm joints, got {len(names)}"
        )
    return {
        "arm_unit": CANONICAL_ARM_UNIT,
        "gripper_semantics": CANONICAL_GRIPPER_SEMANTICS,
        "arm_dof": len(names),
        "joint_names": names,
    }


__all__ = [
    "ARM_DOF",
    "build_joint_feature_contract",
    "validate_joint_feature_contract",
    "CANONICAL_ARM_UNIT",
    "CANONICAL_DOF",
    "CANONICAL_GRIPPER_SEMANTICS",
    "GRIPPER_INDEX",
    "JOINT_FEATURE_CODEC_VERSION",
    "canonical_joint_state_to_feature",
    "canonical_joint_state_to_real_follower",
    "feature_to_canonical_joint_state",
    "real_follower_state_to_feature",
    "sim_joint_command",
    "sim_publish_command",
    "validate_joint_unit_contract",
]
