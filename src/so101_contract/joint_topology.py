"""Phase 13 — joint topology 계약과 topology-aware absolute↔relative 변환.

``joint_relative`` mode의 target은 단순 subtraction이 아니다. revolute joint의 wrap과
continuous joint의 주기성 때문에 다음 계약이 필요하다.

.. code-block:: text

    Δq[h]         = topology_aware_difference(q_action[h], q_state)
    q_absolute[h] = topology_aware_add(q_state, Δq[h])

joint별 metadata(type, period/wrap, optional limits)를 **명시적 immutable contract**로
resolve하며, 다음을 MUST 지킨다.

- ``continuous``는 반드시 periodic이다(period 미지정 시 ``2π``).
- ``revolute``는 period가 있으면 최단 경로로 wrap하고, 없으면 선형 차이를 쓴다.
- ``prismatic``은 선형 차이이며 period를 가질 수 없다.
- gripper 등 passthrough group은 변환하지 않는다(absolute 유지).
- joint dimension은 5/6/7로 하드코딩하지 않고 feature metadata의 names/groups/indices에서
  resolve한다.

``topology_aware_add``는 계약대로 **canonical absolute target**을 돌려준다. periodic joint의
raw 합은 canonical 범위를 ``k·period``만큼 벗어날 수 있으므로 :meth:`JointTopology.add`가
그 자리에서 wrap한다(``canonicalize=False``는 내부/디버그 전용).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Any

from .array_backend import (
    Array,
    as_like,
    check_array,
    replace_indices,
    require_same_backend,
    resolve_chunk_reference,
    take_indices,
    validate_indices,
    where,
)

JOINT_TOPOLOGY_VERSION = "so101_joint_topology_v2"
TAU = 2.0 * math.pi

#: periodic이 아닌 joint의 period 자리표시자. 연산에서 mask로 걸러진다.
_NON_PERIODIC = 1.0


class JointType(str, Enum):
    """URDF joint type 중 action 계약에 영향을 주는 3종."""

    REVOLUTE = "revolute"
    CONTINUOUS = "continuous"
    PRISMATIC = "prismatic"


@dataclass(frozen=True)
class JointSpec:
    """한 joint의 topology 계약.

    Attributes:
        period: 주기. ``None``이면 선형(비주기) joint다. ``continuous``는 ``None``을 허용하지
            않으며 미지정 시 ``2π``가 들어간다.
        lower/upper: optional limit. periodic joint의 canonical 범위 중심을 정하는 데 쓴다.
    """

    name: str
    type: JointType
    period: float | None = None
    lower: float | None = None
    upper: float | None = None
    unit: str = "radian"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"joint name must be a non-empty string, got {self.name!r}")
        joint_type = self.type if isinstance(self.type, JointType) else JointType(str(self.type))
        object.__setattr__(self, "type", joint_type)

        period = self.period
        if joint_type is JointType.CONTINUOUS:
            # continuous는 정의상 periodic이다. 추정하지 않고 기본 2π를 명시적으로 채운다.
            period = TAU if period is None else float(period)
            if self.lower is not None or self.upper is not None:
                raise ValueError(
                    f"continuous joint {self.name!r} must not declare position limits"
                )
        elif joint_type is JointType.PRISMATIC:
            if period is not None:
                raise ValueError(f"prismatic joint {self.name!r} cannot be periodic")
        elif period is not None:
            period = float(period)
        if period is not None and (not math.isfinite(period) or period <= 0.0):
            raise ValueError(f"joint {self.name!r} period must be positive and finite, got {period!r}")
        object.__setattr__(self, "period", period)

        if self.lower is not None and self.upper is not None and self.lower >= self.upper:
            raise ValueError(
                f"joint {self.name!r} limits must satisfy lower < upper, "
                f"got [{self.lower}, {self.upper}]"
            )
        if (self.lower is None) != (self.upper is None):
            raise ValueError(f"joint {self.name!r} must declare both or neither limit")
        if period is not None and self.lower is not None:
            # URDF/float32 metadata에서 온 ±π 값은 period를 미세하게 넘길 수 있다.
            span_tolerance = 1e-6 * period
            if (self.upper - self.lower) - period > span_tolerance:
                raise ValueError(
                    f"joint {self.name!r} limit span exceeds its period: "
                    f"{self.upper - self.lower} > {period}"
                )
        if joint_type is JointType.PRISMATIC and self.unit == "radian":
            object.__setattr__(self, "unit", "meter")

    @property
    def is_periodic(self) -> bool:
        return self.period is not None

    @property
    def center(self) -> float:
        """Canonical 범위의 중심. limit이 있으면 그 중점, 없으면 0."""
        if self.lower is not None and self.upper is not None:
            return (self.lower + self.upper) / 2.0
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type.value,
            "period": self.period,
            "lower": self.lower,
            "upper": self.upper,
            "unit": self.unit,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> JointSpec:
        if not isinstance(payload, dict):
            raise TypeError(f"joint spec must be an object, got {type(payload).__name__}")
        unknown = set(payload) - {"name", "type", "period", "lower", "upper", "unit"}
        if unknown:
            raise ValueError(f"unknown joint spec fields: {sorted(unknown)}")
        return cls(
            name=payload["name"],
            type=JointType(payload["type"]),
            period=payload.get("period"),
            lower=payload.get("lower"),
            upper=payload.get("upper"),
            unit=payload.get("unit", "radian"),
        )


@dataclass(frozen=True)
class JointTopology:
    """Transform 대상 joint group 전체의 immutable topology 계약."""

    joints: tuple[JointSpec, ...]
    version: str = JOINT_TOPOLOGY_VERSION

    def __post_init__(self) -> None:
        joints = tuple(self.joints)
        if not joints:
            raise ValueError("joint topology must contain at least one joint")
        if not all(isinstance(joint, JointSpec) for joint in joints):
            raise TypeError("joint topology entries must be JointSpec instances")
        names = [joint.name for joint in joints]
        if len(set(names)) != len(names):
            raise ValueError(f"joint topology contains duplicate names: {names}")
        object.__setattr__(self, "joints", joints)
        if self.version != JOINT_TOPOLOGY_VERSION:
            raise ValueError(
                f"joint topology version mismatch: {self.version!r} != {JOINT_TOPOLOGY_VERSION!r}"
            )

    @property
    def dim(self) -> int:
        return len(self.joints)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(joint.name for joint in self.joints)

    @property
    def has_periodic_joint(self) -> bool:
        return any(joint.is_periodic for joint in self.joints)

    def _period_vectors(self, like: Array) -> tuple[Array, Array, Array]:
        """``(periodic mask, period, center)``를 입력과 같은 dtype/device로 만든다."""
        periodic = as_like(like, [1.0 if joint.is_periodic else 0.0 for joint in self.joints])
        period = as_like(
            like,
            [joint.period if joint.is_periodic else _NON_PERIODIC for joint in self.joints],
        )
        center = as_like(like, [joint.center for joint in self.joints])
        return periodic, period, center

    def difference(self, action: Array, state: Array) -> Array:
        """``topology_aware_difference(q_action, q_state)``.

        periodic joint는 ``[-period/2, period/2)``의 최단 경로로 wrap하고, 나머지는 선형
        차이를 그대로 쓴다.
        """
        resolved_action = check_array(action, "action", last_dim=self.dim)
        resolved_state = check_array(state, "state", last_dim=self.dim)
        require_same_backend(
            resolved_action,
            resolved_state,
            left_name="action",
            right_name="state",
        )
        periodic, period, _ = self._period_vectors(resolved_action)
        linear = resolved_action - resolved_state
        wrapped = (linear + period / 2.0) % period - period / 2.0
        return where(periodic > 0.0, wrapped, linear)

    def add(self, state: Array, delta: Array, *, canonicalize: bool = True) -> Array:
        """``q_absolute = topology_aware_add(q_state, Δq)``.

        기본값은 계약대로 **canonical absolute target**을 돌려준다. periodic joint는
        raw 합이 canonical 범위를 벗어날 수 있으므로 그 자리에서 wrap한다.

        ``canonicalize=False``는 wrap 이전 raw 합을 보기 위한 내부/디버그 용도다.
        """
        resolved_state = check_array(state, "state", last_dim=self.dim)
        resolved_delta = check_array(delta, "delta", last_dim=self.dim)
        require_same_backend(
            resolved_state,
            resolved_delta,
            left_name="state",
            right_name="delta",
        )
        total = resolved_state + resolved_delta
        return self.canonicalize(total) if canonicalize else total

    def canonicalize(self, values: Array) -> Array:
        """Periodic joint를 canonical 범위로 wrap한 absolute target."""
        resolved = check_array(values, "joint values", last_dim=self.dim)
        periodic, period, center = self._period_vectors(resolved)
        wrapped = (resolved - center + period / 2.0) % period - period / 2.0 + center
        return where(periodic > 0.0, wrapped, resolved)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "joints": [joint.to_dict() for joint in self.joints],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> JointTopology:
        if not isinstance(payload, dict):
            raise TypeError(f"joint topology must be an object, got {type(payload).__name__}")
        joints = payload.get("joints")
        if not isinstance(joints, list):
            raise ValueError("joint topology payload must contain a 'joints' list")
        return cls(
            joints=tuple(JointSpec.from_dict(joint) for joint in joints),
            version=payload.get("version", JOINT_TOPOLOGY_VERSION),
        )

    def fingerprint(self) -> str:
        canonical = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    @classmethod
    def from_feature_metadata(
        cls,
        feature_names: Sequence[str],
        groups: dict[str, Sequence[int]],
        *,
        joint_group: str,
        joint_metadata: dict[str, dict[str, Any]],
    ) -> tuple[JointTopology, tuple[int, ...]]:
        """Feature metadata에서 joint group의 topology와 index를 resolve.

        joint dimension을 하드코딩하지 않는다. group 안의 모든 joint 이름에 대한 metadata가
        없으면 추정하지 않고 실패한다.

        Returns:
            ``(topology, indices)``
        """
        names = tuple(str(name) for name in feature_names)
        if joint_group not in groups:
            raise KeyError(
                f"feature metadata has no joint group {joint_group!r}; "
                f"groups={sorted(groups)}"
            )
        bounds = tuple(int(value) for value in groups[joint_group])
        if len(bounds) != 2:
            raise ValueError(f"joint group bounds must be [start, end), got {bounds}")
        start, end = bounds
        if start < 0 or end <= start or end > len(names):
            raise ValueError(
                f"joint group range [{start},{end}) is invalid for dim {len(names)}"
            )
        indices = tuple(range(start, end))
        joint_names = names[start:end]

        missing = [name for name in joint_names if name not in joint_metadata]
        if missing:
            raise KeyError(
                "joint topology metadata is missing for "
                f"{missing}; declare type/period/limits explicitly"
            )
        joints = tuple(
            JointSpec.from_dict({"name": name, **joint_metadata[name]})
            for name in joint_names
        )
        return cls(joints=joints), indices


# --- chunk 단위 변환 ----------------------------------------------------------


def absolute_joint_actions_to_relative(
    state: Array,
    actions: Array,
    topology: JointTopology,
    *,
    state_joint_indices: Sequence[int] | None = None,
    action_joint_indices: Sequence[int] | None = None,
) -> Array:
    """Full feature action에서 joint group만 relative로 바꾸고 gripper는 passthrough.

    chunk의 모든 horizon이 같은 기준 state를 공유한다.
    """
    require_same_backend(state, actions, left_name="state", right_name="actions")
    default = tuple(range(topology.dim))
    state_indices = validate_indices(
        default if state_joint_indices is None else state_joint_indices,
        feature_dim=state.shape[-1],
        expected_count=topology.dim,
        name="state_joint_indices",
    )
    action_indices = validate_indices(
        default if action_joint_indices is None else action_joint_indices,
        feature_dim=actions.shape[-1],
        expected_count=topology.dim,
        name="action_joint_indices",
    )
    state_joints = take_indices(state, state_indices)
    action_joints = take_indices(actions, action_indices)
    reference = resolve_chunk_reference(state_joints, action_joints, topology.dim)
    relative = topology.difference(action_joints, reference[..., None, :])
    return replace_indices(actions, action_indices, relative)


def relative_joint_actions_to_absolute(
    state: Array,
    relative_actions: Array,
    topology: JointTopology,
    *,
    state_joint_indices: Sequence[int] | None = None,
    action_joint_indices: Sequence[int] | None = None,
    canonicalize: bool = True,
) -> Array:
    """Relative joint chunk를 absolute joint target으로 복원.

    기본 호출은 ``topology_aware_add`` 계약대로 canonical absolute target을 돌려주므로
    원 dataset action을 그대로 복원한다. ``canonicalize=False``는 wrap 이전 raw 합을 보기
    위한 내부/디버그 용도다.
    """
    require_same_backend(
        state,
        relative_actions,
        left_name="state",
        right_name="relative_actions",
    )
    default = tuple(range(topology.dim))
    state_indices = validate_indices(
        default if state_joint_indices is None else state_joint_indices,
        feature_dim=state.shape[-1],
        expected_count=topology.dim,
        name="state_joint_indices",
    )
    action_indices = validate_indices(
        default if action_joint_indices is None else action_joint_indices,
        feature_dim=relative_actions.shape[-1],
        expected_count=topology.dim,
        name="action_joint_indices",
    )
    state_joints = take_indices(state, state_indices)
    relative_joints = take_indices(relative_actions, action_indices)
    reference = resolve_chunk_reference(state_joints, relative_joints, topology.dim)
    absolute = topology.add(
        reference[..., None, :],
        relative_joints,
        canonicalize=canonicalize,
    )
    return replace_indices(relative_actions, action_indices, absolute)


def canonicalize_joint_actions(
    actions: Array,
    topology: JointTopology,
    *,
    action_joint_indices: Sequence[int] | None = None,
) -> Array:
    """``joint_absolute`` mode용. joint group만 canonical 범위로 wrap한다."""
    default = tuple(range(topology.dim))
    action_indices = validate_indices(
        default if action_joint_indices is None else action_joint_indices,
        feature_dim=actions.shape[-1],
        expected_count=topology.dim,
        name="action_joint_indices",
    )
    joints = take_indices(actions, action_indices)
    return replace_indices(actions, action_indices, topology.canonicalize(joints))


# --- SO-101 기본 topology -----------------------------------------------------
#
# arm 5축은 sim URDF에서 ±π limit(span = 2π)이라 경계를 가로지르는 최단 경로가 존재한다.
# 따라서 revolute + period=2π로 선언한다. 실제 limit 값의 단일 소스는
# ``feature_codec.SIM_JOINT_LIMITS_RAD``이며, 여기서는 그 값을 참조해 topology를 만든다.


def so101_arm_joint_topology(*, joint_names: Sequence[str] | None = None) -> JointTopology:
    """SO-101 arm 5축(gripper 제외) 기본 topology."""
    from .feature_codec import SIM_JOINT_LIMITS_RAD, SO101_JOINT_ORDER

    names = tuple(joint_names) if joint_names is not None else tuple(
        f"{joint}.rad" for joint in SO101_JOINT_ORDER[:5]
    )
    if len(names) != 5:
        raise ValueError(f"SO-101 arm topology needs 5 joint names, got {names}")
    joints = []
    for index, name in enumerate(names):
        lower, upper = (float(value) for value in SIM_JOINT_LIMITS_RAD[index])
        joints.append(
            JointSpec(
                name=name,
                type=JointType.REVOLUTE,
                period=TAU,
                lower=lower,
                upper=upper,
            )
        )
    return JointTopology(joints=tuple(joints))
