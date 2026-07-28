#!/usr/bin/env python3
"""Phase 13 — joint topology 계약과 topology-aware absolute↔relative 변환 검증.

확인 항목:

- ``revolute``/``continuous``/``prismatic`` metadata contract와 불가능한 조합 거부
- revolute wrap fixture가 **단순 subtraction과 다른 결과**를 내고, add가 canonical
  absolute target을 복원
- continuous는 반드시 periodic, prismatic은 선형 차이
- gripper passthrough bitwise 동일
- joint dimension을 5/6/7로 하드코딩하지 않고 feature metadata에서 resolve
- chunk 전체가 하나의 기준 state를 공유
- numpy/torch parity와 직렬화 round-trip

.. code-block:: bash

    python scripts/contract/validate_joint_representation.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from so101_contract.action_representation import (  # noqa: E402
    ActionRepresentationMode,
    ActionRepresentationSpec,
)
from so101_contract.action_transform import ActionRepresentationTransform  # noqa: E402
from so101_contract.joint_topology import (  # noqa: E402
    JOINT_TOPOLOGY_VERSION,
    TAU,
    JointSpec,
    JointTopology,
    JointType,
    absolute_joint_actions_to_relative,
    canonicalize_joint_actions,
    relative_joint_actions_to_absolute,
    so101_arm_joint_topology,
)

TOLERANCE_F64 = 1e-9
TOLERANCE_F32 = 3e-5


def _max_error(left, right) -> float:
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def _mixed_topology() -> JointTopology:
    """wrap/continuous/linear가 모두 들어간 4축 fixture."""
    return JointTopology(
        joints=(
            # ±π limit(span = 2π)이라 경계를 가로지르는 최단 경로가 존재한다.
            JointSpec("wrap_revolute", JointType.REVOLUTE, period=TAU, lower=-math.pi, upper=math.pi),
            # limit이 좁아 wrap이 의미 없는 revolute.
            JointSpec("limited_revolute", JointType.REVOLUTE, lower=-1.0, upper=1.0),
            JointSpec("spool", JointType.CONTINUOUS),
            JointSpec("rail", JointType.PRISMATIC, lower=0.0, upper=0.4),
        )
    )


def check_joint_spec_contract() -> None:
    """joint metadata contract의 필수 규칙."""
    continuous = JointSpec("spool", JointType.CONTINUOUS)
    if not continuous.is_periodic or continuous.period != TAU:
        raise AssertionError("continuous joint must always be periodic (default 2π)")
    prismatic = JointSpec("rail", JointType.PRISMATIC)
    if prismatic.is_periodic:
        raise AssertionError("prismatic joint must never be periodic")
    if prismatic.unit != "meter":
        raise AssertionError("prismatic joint unit must default to meter")
    limited = JointSpec("limited", JointType.REVOLUTE, lower=-1.0, upper=1.0)
    if limited.is_periodic:
        raise AssertionError("revolute without a declared period must stay linear")
    if abs(JointSpec("off", JointType.REVOLUTE, period=TAU, lower=0.0, upper=2.0).center - 1.0) > 0:
        raise AssertionError("canonical center must follow declared limits")

    rejects = [
        ("prismatic with period", lambda: JointSpec("r", JointType.PRISMATIC, period=TAU)),
        (
            "continuous with limits",
            lambda: JointSpec("c", JointType.CONTINUOUS, lower=-1.0, upper=1.0),
        ),
        ("non-positive period", lambda: JointSpec("r", JointType.REVOLUTE, period=0.0)),
        ("inverted limits", lambda: JointSpec("r", JointType.REVOLUTE, lower=1.0, upper=-1.0)),
        ("half limits", lambda: JointSpec("r", JointType.REVOLUTE, lower=1.0)),
        (
            "limit span beyond period",
            lambda: JointSpec("r", JointType.REVOLUTE, period=1.0, lower=-1.0, upper=1.0),
        ),
        ("unknown type", lambda: JointSpec("r", "ball")),
        ("empty name", lambda: JointSpec("", JointType.REVOLUTE)),
        ("empty topology", lambda: JointTopology(joints=())),
        (
            "duplicate joint names",
            lambda: JointTopology(
                joints=(JointSpec("a", JointType.REVOLUTE), JointSpec("a", JointType.PRISMATIC))
            ),
        ),
        ("unknown spec field", lambda: JointSpec.from_dict({"name": "a", "type": "revolute", "x": 1})),
    ]
    for label, call in rejects:
        try:
            call()
        except (TypeError, ValueError):
            continue
        raise AssertionError(f"invalid joint metadata was accepted: {label}")
    print(f"PASS: joint metadata contract ({len(rejects)} invalid declarations rejected)")


def check_wrap_differs_from_subtraction() -> None:
    """revolute wrap fixture가 단순 subtraction과 다르고, add가 canonical target을 복원."""
    topology = _mixed_topology()
    state = np.asarray([3.0, 0.4, 3.1, 0.10], dtype=np.float64)
    action = np.asarray([-3.0, -0.4, -3.1, 0.35], dtype=np.float64)

    naive = action - state
    delta = topology.difference(action, state)
    # 계약: q_absolute = topology_aware_add(q_state, delta) 가 곧 canonical target이다.
    restored = topology.add(state, delta)

    # wrap joint 두 개는 경계를 가로지르는 최단 경로를 택하므로 subtraction과 다르다.
    if abs(float(delta[0]) - float(naive[0])) < 1.0:
        raise AssertionError(
            f"revolute wrap did not differ from subtraction: {delta[0]} vs {naive[0]}"
        )
    if abs(float(delta[2]) - float(naive[2])) < 1.0:
        raise AssertionError("continuous wrap did not differ from subtraction")
    expected_wrap = -6.0 + TAU
    if abs(float(delta[0]) - expected_wrap) > TOLERANCE_F64:
        raise AssertionError(f"revolute shortest path is wrong: {delta[0]} != {expected_wrap}")
    if abs(float(delta[0])) > TAU / 2.0:
        raise AssertionError("wrapped delta must lie within [-period/2, period/2)")

    # 선형 joint는 정확히 subtraction이다.
    for index in (1, 3):
        if abs(float(delta[index]) - float(naive[index])) > TOLERANCE_F64:
            raise AssertionError(f"linear joint {index} must use plain subtraction")

    if _max_error(restored, action) > TOLERANCE_F64:
        raise AssertionError(
            f"topology_aware_add did not restore the canonical absolute target: {restored}"
        )
    # 내부/디버그용 raw 합은 canonical 범위를 벗어나며, 기본 add가 그것을 wrap한다.
    raw = topology.add(state, delta, canonicalize=False)
    if abs(float(raw[0]) - float(action[0])) < 1.0:
        raise AssertionError("fixture does not exercise the canonical wrap on add")
    if _max_error(topology.canonicalize(raw), restored) > TOLERANCE_F64:
        raise AssertionError("default add must equal canonicalize(raw add)")
    print("PASS: revolute/continuous wrap differs from subtraction and add restores the target")


def check_random_round_trip() -> None:
    """무작위 state/action에서 difference→add→canonicalize가 원 target을 복원."""
    topology = _mixed_topology()
    generator = np.random.default_rng(7)
    state = np.stack(
        [
            generator.uniform(-math.pi, math.pi, size=256),
            generator.uniform(-1.0, 1.0, size=256),
            generator.uniform(-3.0 * math.pi, 3.0 * math.pi, size=256),
            generator.uniform(0.0, 0.4, size=256),
        ],
        axis=-1,
    )
    action = np.stack(
        [
            generator.uniform(-math.pi, math.pi, size=256),
            generator.uniform(-1.0, 1.0, size=256),
            generator.uniform(-math.pi, math.pi, size=256),
            generator.uniform(0.0, 0.4, size=256),
        ],
        axis=-1,
    )
    delta = topology.difference(action, state)
    if float(np.max(np.abs(delta[:, 0]))) > TAU / 2.0 + TOLERANCE_F64:
        raise AssertionError("wrapped revolute delta escaped [-period/2, period/2)")
    restored = topology.add(state, delta)
    if _max_error(restored, topology.canonicalize(action)) > TOLERANCE_F64:
        raise AssertionError("randomized joint round-trip failed")

    for dtype, tolerance in ((np.float64, TOLERANCE_F64), (np.float32, TOLERANCE_F32)):
        typed_state = state.astype(dtype)
        typed_action = action.astype(dtype)
        typed_delta = topology.difference(typed_action, typed_state)
        typed_restored = topology.add(typed_state, typed_delta)
        if typed_restored.dtype != dtype:
            raise AssertionError(f"dtype {dtype} was not preserved")
        if _max_error(typed_restored, topology.canonicalize(typed_action)) > tolerance:
            raise AssertionError(f"{np.dtype(dtype).name} joint round-trip exceeded tolerance")
    print("PASS: randomized topology-aware round-trip (float32/float64)")


def check_chunk_transform_and_passthrough() -> None:
    """Full feature chunk 변환, gripper passthrough, 단일 기준 state."""
    topology = _mixed_topology()
    generator = np.random.default_rng(11)
    joint_dim = topology.dim
    action_dim = joint_dim + 1

    state = np.concatenate([generator.uniform(-3.0, 3.0, size=joint_dim), [42.0]])
    actions = np.concatenate(
        [
            generator.uniform(-3.0, 3.0, size=(6, joint_dim)),
            generator.uniform(0.0, 100.0, size=(6, 1)),
        ],
        axis=-1,
    )
    relative = absolute_joint_actions_to_relative(state, actions, topology)
    if not np.array_equal(relative[:, joint_dim:], actions[:, joint_dim:]):
        raise AssertionError("gripper passthrough was modified by the relative transform")
    restored = relative_joint_actions_to_absolute(state, relative, topology)
    if not np.array_equal(restored[:, joint_dim:], actions[:, joint_dim:]):
        raise AssertionError("gripper passthrough was modified by the absolute restore")
    expected = canonicalize_joint_actions(actions, topology)
    if _max_error(restored, expected) > TOLERANCE_F64:
        raise AssertionError("full-chunk joint round-trip failed")
    if relative.shape != actions.shape or restored.shape != actions.shape:
        raise AssertionError("chunk transform changed the action shape")

    # 모든 horizon이 같은 기준 state를 쓴다: 기준을 고정하면 delta는 누적 차분이 아니다.
    manual = topology.difference(actions[:, :joint_dim], state[None, :joint_dim])
    if _max_error(relative[:, :joint_dim], manual) > TOLERANCE_F64:
        raise AssertionError("chunk relative targets are not all anchored to the same state")

    # observation history가 있으면 마지막 관측만 기준이다.
    history = np.stack([np.full(action_dim, 9.0), state])
    from_history = absolute_joint_actions_to_relative(history, actions, topology)
    if _max_error(from_history, relative) > TOLERANCE_F64:
        raise AssertionError("observation history did not use the last state as reference")

    # in-place 수정 금지.
    if not np.array_equal(
        actions,
        np.concatenate([actions[:, :joint_dim], actions[:, joint_dim:]], axis=-1),
    ):
        raise AssertionError("input actions were mutated")
    print("PASS: chunk transform, gripper passthrough, single chunk reference")


def check_dimension_resolution() -> None:
    """joint dimension을 하드코딩하지 않고 feature metadata에서 resolve."""
    cases = {
        5: [f"arm.joint_{index}" for index in range(5)] + ["gripper.pos"],
        7: [f"arm.joint_{index}" for index in range(7)] + ["gripper.pos"],
    }
    for joint_dim, names in cases.items():
        groups = {"arm_joints": [0, joint_dim], "gripper_position": [joint_dim, joint_dim + 1]}
        metadata = {
            name: {"type": "revolute", "period": TAU, "lower": -math.pi, "upper": math.pi}
            for name in names[:joint_dim]
        }
        topology, indices = JointTopology.from_feature_metadata(
            names,
            groups,
            joint_group="arm_joints",
            joint_metadata=metadata,
        )
        if topology.dim != joint_dim or indices != tuple(range(joint_dim)):
            raise AssertionError(f"resolved joint dim mismatch: {topology.dim} != {joint_dim}")
        if topology.names != tuple(names[:joint_dim]):
            raise AssertionError("resolved joint names mismatch")

    names = cases[5]
    groups = {"arm_joints": [0, 5], "gripper_position": [5, 6]}
    rejects = [
        (
            "missing joint metadata",
            lambda: JointTopology.from_feature_metadata(
                names,
                groups,
                joint_group="arm_joints",
                joint_metadata={names[0]: {"type": "revolute"}},
            ),
        ),
        (
            "unknown group",
            lambda: JointTopology.from_feature_metadata(
                names,
                groups,
                joint_group="legs",
                joint_metadata={},
            ),
        ),
        (
            "out of range group",
            lambda: JointTopology.from_feature_metadata(
                names,
                {"arm_joints": [0, 99]},
                joint_group="arm_joints",
                joint_metadata={},
            ),
        ),
    ]
    for label, call in rejects:
        try:
            call()
        except (KeyError, ValueError):
            continue
        raise AssertionError(f"invalid joint metadata resolution was accepted: {label}")

    arm = so101_arm_joint_topology()
    if arm.dim != 5 or not arm.has_periodic_joint:
        raise AssertionError("SO-101 arm topology is wrong")
    if any(joint.type is not JointType.REVOLUTE for joint in arm.joints):
        raise AssertionError("SO-101 arm joints must be revolute")
    print("PASS: joint dimension resolved from feature metadata (5D and 7D), no hardcoding")


def check_torch_parity() -> None:
    """Torch/NumPy 결과 동일, dtype/device 보존."""
    topology = _mixed_topology()
    joint_dim = topology.dim
    generator = np.random.default_rng(13)
    state = np.concatenate([generator.uniform(-3.0, 3.0, size=joint_dim), [7.0]])
    actions = np.concatenate(
        [
            generator.uniform(-3.0, 3.0, size=(2, 5, joint_dim)),
            generator.uniform(0.0, 100.0, size=(2, 5, 1)),
        ],
        axis=-1,
    )
    batched_state = np.broadcast_to(state, (2, joint_dim + 1)).copy()

    numpy_relative = absolute_joint_actions_to_relative(batched_state, actions, topology)
    torch_relative = absolute_joint_actions_to_relative(
        torch.from_numpy(batched_state),
        torch.from_numpy(actions),
        topology,
    )
    if torch_relative.dtype is not torch.float64:
        raise AssertionError("torch float64 dtype was not preserved")
    if _max_error(torch_relative.numpy(), numpy_relative) > TOLERANCE_F64:
        raise AssertionError("torch/numpy joint relative mismatch")

    float32_relative = absolute_joint_actions_to_relative(
        torch.from_numpy(batched_state).float(),
        torch.from_numpy(actions).float(),
        topology,
    )
    if float32_relative.dtype is not torch.float32:
        raise AssertionError("torch float32 dtype was not preserved")
    if _max_error(float32_relative.numpy(), numpy_relative) > TOLERANCE_F32:
        raise AssertionError("torch float32 joint parity exceeded tolerance")

    torch_restored = relative_joint_actions_to_absolute(
        torch.from_numpy(batched_state),
        torch_relative,
        topology,
    )
    expected = canonicalize_joint_actions(actions, topology)
    if _max_error(torch_restored.numpy(), expected) > TOLERANCE_F64:
        raise AssertionError("torch joint round-trip failed")
    print("PASS: torch/numpy parity and dtype preservation")


def check_serialization() -> None:
    """Topology 직렬화 round-trip과 fingerprint 안정성."""
    topology = _mixed_topology()
    payload = topology.to_dict()
    restored = JointTopology.from_dict(payload)
    if restored != topology or restored.fingerprint() != topology.fingerprint():
        raise AssertionError("joint topology round-trip failed")
    if payload["version"] != JOINT_TOPOLOGY_VERSION:
        raise AssertionError("joint topology version was not recorded")

    changed = JointTopology(
        joints=(
            JointSpec("wrap_revolute", JointType.REVOLUTE, period=TAU, lower=-math.pi, upper=math.pi),
            JointSpec("limited_revolute", JointType.REVOLUTE, lower=-2.0, upper=2.0),
            JointSpec("spool", JointType.CONTINUOUS),
            JointSpec("rail", JointType.PRISMATIC, lower=0.0, upper=0.4),
        )
    )
    if changed.fingerprint() == topology.fingerprint():
        raise AssertionError("different joint limits must change the topology fingerprint")
    print("PASS: joint topology serialization and fingerprint")


def check_transform_modes() -> None:
    """joint_absolute는 target 변환을 건너뛰고 joint_relative만 state를 요구."""
    topology = _mixed_topology()
    joint_dim = topology.dim
    action_dim = joint_dim + 1
    generator = np.random.default_rng(17)
    state = np.concatenate([generator.uniform(-3.0, 3.0, size=joint_dim), [1.0]])
    actions = np.concatenate(
        [
            generator.uniform(-math.pi, math.pi, size=(4, joint_dim)),
            generator.uniform(0.0, 100.0, size=(4, 1)),
        ],
        axis=-1,
    )

    absolute_transform = ActionRepresentationTransform(
        spec=ActionRepresentationSpec(mode=ActionRepresentationMode.JOINT_ABSOLUTE),
        state_indices=tuple(range(joint_dim)),
        action_indices=tuple(range(joint_dim)),
        passthrough_action_indices=(joint_dim,),
        state_dim=action_dim,
        action_dim=action_dim,
        joint_topology=topology,
    )
    if absolute_transform.requires_state_reference:
        raise AssertionError("joint_absolute must not require a state reference")
    encoded = absolute_transform.encode(None, actions)
    if _max_error(encoded, canonicalize_joint_actions(actions, topology)) > TOLERANCE_F64:
        raise AssertionError("joint_absolute encode must only canonicalize")
    if _max_error(absolute_transform.decode(None, encoded), encoded) > TOLERANCE_F64:
        raise AssertionError("joint_absolute decode must be idempotent")

    relative_transform = ActionRepresentationTransform(
        spec=ActionRepresentationSpec(mode=ActionRepresentationMode.JOINT_RELATIVE),
        state_indices=tuple(range(joint_dim)),
        action_indices=tuple(range(joint_dim)),
        passthrough_action_indices=(joint_dim,),
        state_dim=action_dim,
        action_dim=action_dim,
        joint_topology=topology,
    )
    if not relative_transform.requires_state_reference:
        raise AssertionError("joint_relative must require a state reference")
    try:
        relative_transform.encode(None, actions)
    except ValueError:
        pass
    else:
        raise AssertionError("joint_relative encode without a state must fail")
    targets = relative_transform.encode(state, actions)
    restored = relative_transform.decode(state, targets)
    if _max_error(restored, canonicalize_joint_actions(actions, topology)) > TOLERANCE_F64:
        raise AssertionError("joint_relative transform round-trip failed")

    rejects = [
        (
            "joint mode without topology",
            lambda: ActionRepresentationTransform(
                spec=ActionRepresentationSpec(mode=ActionRepresentationMode.JOINT_RELATIVE),
                state_indices=(0, 1, 2, 3),
                action_indices=(0, 1, 2, 3),
                passthrough_action_indices=(4,),
                state_dim=5,
                action_dim=5,
            ),
        ),
        (
            "unclassified action dimension",
            lambda: ActionRepresentationTransform(
                spec=ActionRepresentationSpec(mode=ActionRepresentationMode.JOINT_ABSOLUTE),
                state_indices=tuple(range(joint_dim)),
                action_indices=tuple(range(joint_dim)),
                passthrough_action_indices=(joint_dim,),
                state_dim=action_dim,
                action_dim=action_dim + 1,
                joint_topology=topology,
            ),
        ),
    ]
    for label, call in rejects:
        try:
            call()
        except ValueError:
            continue
        raise AssertionError(f"invalid transform was accepted: {label}")

    payload = relative_transform.to_dict()
    if ActionRepresentationTransform.from_dict(payload) != relative_transform:
        raise AssertionError("joint transform serialization round-trip failed")
    print("PASS: joint_absolute canonical-only vs joint_relative state-anchored transforms")


CHECKS = (
    check_joint_spec_contract,
    check_wrap_differs_from_subtraction,
    check_random_round_trip,
    check_chunk_transform_and_passthrough,
    check_dimension_resolution,
    check_torch_parity,
    check_serialization,
    check_transform_modes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for check in CHECKS:
        check()
    print(f"PASS: joint action representation contract ({len(CHECKS)} checks)")


if __name__ == "__main__":
    main()
