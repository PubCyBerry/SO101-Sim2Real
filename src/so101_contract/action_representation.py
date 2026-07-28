"""Phase 11 — 4-mode action representation config (schema v2).

v1은 ``mode ∈ {absolute, eef_relative}``라는 모호한 2-값 enum이었다. ``absolute``가
joint absolute인지 EEF absolute인지 config만 보고 알 수 없었기 때문에, v2는 space와
reference를 분리해 4개 mode로 고정한다.

.. code-block:: text

    joint_absolute   state = current absolute joint   target = future absolute joint
    joint_relative   state = current absolute joint   target = current joint 기준 상대 joint
    eef_absolute     state = current absolute EEF     target = future absolute EEF
    eef_relative     state = current absolute EEF     target = current EEF 기준 상대 EEF

핵심 계약:

- ``relative``는 **observation state가 아니라 action target의 표현**을 뜻한다.
  모든 mode에서 ``observation.state``는 current absolute 값이다.
- **dataset은 언제나 absolute를 저장한다.** relative target은 training processor가
  생성하고 inference postprocessor가 absolute로 복원한다. relative dataset을 별도
  영속 포맷으로 만들지 않는다.
- gripper는 모든 mode에서 absolute passthrough다.
- joint mode에서 ``pose_format``은 ``None``/``not_applicable``만 허용한다. EEF 기본값을
  암묵적으로 남기지 않는다.
- joint dimension은 6D/7D로 하드코딩하지 않고 dataset feature metadata에서 resolve한다.

v1 :class:`so101_contract.eef_action_contract.ActionRepresentationConfig`는 legacy
shim으로 남기고, :func:`from_legacy_v1_config`로만 v2 spec으로 승격한다.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any

ACTION_REPRESENTATION_SCHEMA_VERSION = 2
ACTION_REPRESENTATION_SPEC_VERSION = "so101_action_representation_v2"

#: dataset 영속 계약. mode와 무관하게 항상 absolute다.
DATASET_STORAGE_REFERENCE = "absolute"

#: 이 프로젝트가 지원하는 policy family. 8 representation × 3 policy = 24 조합.
SUPPORTED_POLICY_FAMILIES = ("act", "smolvla", "groot")

#: v1의 모호한 값. 신규 config에서 금지하고 명시적 4-mode로 유도한다.
_AMBIGUOUS_LEGACY_MODES = {
    "absolute": ("joint_absolute", "eef_absolute"),
    "relative": ("joint_relative", "eef_relative"),
}


class ActionSpace(str, Enum):
    """Action target이 사는 공간."""

    JOINT = "joint"
    EEF = "eef"


class ActionReference(str, Enum):
    """Action target의 기준. state 자체의 표현이 아니다."""

    ABSOLUTE = "absolute"
    RELATIVE = "relative"


class PoseFormat(str, Enum):
    """EEF pose 직렬화 형식. joint mode에서는 ``NOT_APPLICABLE``만 허용한다."""

    NOT_APPLICABLE = "not_applicable"
    XYZ_ROT6D_ROWS = "xyz_rot6d_rows"
    XYZ_QUATERNION_WXYZ = "xyz_quaternion_wxyz"
    XYZ_RPY = "xyz_rpy"


class ActionRepresentationMode(str, Enum):
    """(space, reference) 곱을 명시적 4-값으로 고정한 mode."""

    JOINT_ABSOLUTE = "joint_absolute"
    JOINT_RELATIVE = "joint_relative"
    EEF_ABSOLUTE = "eef_absolute"
    EEF_RELATIVE = "eef_relative"

    @property
    def space(self) -> ActionSpace:
        return ActionSpace.JOINT if self.value.startswith("joint_") else ActionSpace.EEF

    @property
    def reference(self) -> ActionReference:
        return (
            ActionReference.RELATIVE
            if self.value.endswith("_relative")
            else ActionReference.ABSOLUTE
        )

    @property
    def is_eef(self) -> bool:
        return self.space is ActionSpace.EEF

    @property
    def is_relative(self) -> bool:
        return self.reference is ActionReference.RELATIVE


#: EEF pose format별 회전 차원과 gripper 제외 pose 차원.
POSE_FORMAT_ROTATION_DIMS: dict[PoseFormat, int] = {
    PoseFormat.XYZ_ROT6D_ROWS: 6,
    PoseFormat.XYZ_QUATERNION_WXYZ: 4,
    PoseFormat.XYZ_RPY: 3,
}
EEF_POSE_FORMATS = tuple(POSE_FORMAT_ROTATION_DIMS)

#: mode별 state/action 의미. 문서와 코드가 갈라지지 않도록 여기 한 곳에만 둔다.
STATE_ACTION_SEMANTICS: dict[ActionRepresentationMode, dict[str, str]] = {
    ActionRepresentationMode.JOINT_ABSOLUTE: {
        "state": "current absolute joint",
        "target": "future absolute joint",
    },
    ActionRepresentationMode.JOINT_RELATIVE: {
        "state": "current absolute joint",
        "target": "future joint relative to current joint",
    },
    ActionRepresentationMode.EEF_ABSOLUTE: {
        "state": "current absolute EEF",
        "target": "future absolute EEF",
    },
    ActionRepresentationMode.EEF_RELATIVE: {
        "state": "current absolute EEF",
        "target": "future EEF relative to current EEF",
    },
}

#: mode별 추론 routing. checkpoint 하나는 하나의 representation에 고정된다.
INFERENCE_ROUTING: dict[ActionRepresentationMode, tuple[str, ...]] = {
    ActionRepresentationMode.JOINT_ABSOLUTE: ("joint_command",),
    ActionRepresentationMode.JOINT_RELATIVE: ("restore_absolute_joint", "joint_command"),
    ActionRepresentationMode.EEF_ABSOLUTE: ("ik", "joint_command"),
    ActionRepresentationMode.EEF_RELATIVE: ("restore_absolute_eef", "ik", "joint_command"),
}

DEFAULT_BASE_FRAME = "base_link"
DEFAULT_EEF_FRAME = "tcp_grasp"
DEFAULT_JOINT_GROUP = "arm_joints"
DEFAULT_GRIPPER_GROUP = "gripper_position"
DEFAULT_STATS_FILE = "meta/action_representation_stats.json"


def pose_format_dims(pose_format: PoseFormat | str) -> tuple[int, int]:
    """EEF pose format → ``(rotation_dim, pose_dim)``. pose_dim은 gripper를 제외한다."""
    resolved = coerce_pose_format(pose_format)
    rotation_dim = POSE_FORMAT_ROTATION_DIMS.get(resolved)
    if rotation_dim is None:
        raise ValueError(
            f"{resolved.value!r} is not an EEF pose format; "
            f"expected one of {[f.value for f in EEF_POSE_FORMATS]}"
        )
    return rotation_dim, 3 + rotation_dim


def coerce_mode(value: Any) -> ActionRepresentationMode:
    """문자열/enum을 4-mode enum으로 변환하고 모호한 legacy 값을 거부."""
    if isinstance(value, ActionRepresentationMode):
        return value
    if not isinstance(value, str):
        raise TypeError(
            f"action representation mode must be str or ActionRepresentationMode, "
            f"got {type(value).__name__}"
        )
    normalized = value.strip().lower()
    alternatives = _AMBIGUOUS_LEGACY_MODES.get(normalized)
    if alternatives is not None:
        raise ValueError(
            f"ambiguous legacy action representation mode {value!r} is rejected in schema v2; "
            f"choose an explicit mode among {list(alternatives)}"
        )
    try:
        return ActionRepresentationMode(normalized)
    except ValueError as exc:
        raise ValueError(
            f"unknown action representation mode {value!r}; expected one of "
            f"{[mode.value for mode in ActionRepresentationMode]}"
        ) from exc


def coerce_pose_format(value: Any) -> PoseFormat:
    """``None``은 ``not_applicable``로 정규화한다."""
    if value is None:
        return PoseFormat.NOT_APPLICABLE
    if isinstance(value, PoseFormat):
        return value
    if not isinstance(value, str):
        raise TypeError(
            f"pose_format must be str, PoseFormat or None, got {type(value).__name__}"
        )
    try:
        return PoseFormat(value.strip().lower())
    except ValueError as exc:
        raise ValueError(
            f"unknown pose format {value!r}; expected None or one of "
            f"{[fmt.value for fmt in PoseFormat]}"
        ) from exc


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def default_transform_group(
    mode: ActionRepresentationMode,
    pose_format: PoseFormat,
) -> str:
    """Relative 변환 대상 group의 canonical 이름."""
    if mode.is_eef:
        _, pose_dim = pose_format_dims(pose_format)
        return f"eef_{pose_dim}d"
    return DEFAULT_JOINT_GROUP


@dataclass(frozen=True)
class ActionRepresentationSpec:
    """세 policy가 공유하는 schema v2 action representation config.

    ``state_group``/``action_group``은 **변환 대상 group**을 가리킨다. EEF mode에서는
    pose group, joint mode에서는 arm joint group이다. gripper는 항상
    ``passthrough_action_groups``로 남아 absolute를 유지한다.
    """

    mode: ActionRepresentationMode
    pose_format: PoseFormat = PoseFormat.NOT_APPLICABLE
    reference_observation: str = "current_observation"
    state_group: str = ""
    action_group: str = ""
    passthrough_action_groups: tuple[str, ...] = (DEFAULT_GRIPPER_GROUP,)
    gripper_representation: str = "absolute"
    base_frame: str | None = None
    eef_frame: str | None = None
    stats_file: str = DEFAULT_STATS_FILE
    strict: bool = True

    def __post_init__(self) -> None:
        mode = coerce_mode(self.mode)
        pose_format = coerce_pose_format(self.pose_format)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "pose_format", pose_format)

        if self.reference_observation != "current_observation":
            raise ValueError(
                f"unsupported relative reference {self.reference_observation!r}; "
                "schema v2 requires 'current_observation'"
            )
        if self.gripper_representation != "absolute":
            raise ValueError(
                "gripper_representation must stay 'absolute' in every mode, "
                f"got {self.gripper_representation!r}"
            )
        if not self.passthrough_action_groups:
            raise ValueError("passthrough_action_groups must contain the absolute gripper group")
        if len(set(self.passthrough_action_groups)) != len(self.passthrough_action_groups):
            raise ValueError(
                f"passthrough_action_groups contains duplicates: {self.passthrough_action_groups}"
            )
        object.__setattr__(
            self,
            "passthrough_action_groups",
            tuple(self.passthrough_action_groups),
        )

        if mode.is_eef:
            if pose_format is PoseFormat.NOT_APPLICABLE:
                raise ValueError(
                    f"mode={mode.value!r} requires an explicit pose_format among "
                    f"{[fmt.value for fmt in EEF_POSE_FORMATS]}"
                )
            base_frame = self.base_frame or DEFAULT_BASE_FRAME
            eef_frame = self.eef_frame or DEFAULT_EEF_FRAME
            if not isinstance(base_frame, str) or not isinstance(eef_frame, str):
                raise TypeError("base_frame/eef_frame must be strings in EEF modes")
            object.__setattr__(self, "base_frame", base_frame)
            object.__setattr__(self, "eef_frame", eef_frame)
        else:
            if pose_format is not PoseFormat.NOT_APPLICABLE:
                raise ValueError(
                    f"mode={mode.value!r} is a joint mode; pose_format must be None or "
                    f"'not_applicable', got {pose_format.value!r}"
                )
            if self.base_frame is not None or self.eef_frame is not None:
                raise ValueError(
                    "joint modes have no EEF frame contract; base_frame/eef_frame must be None"
                )

        for field_name in ("state_group", "action_group"):
            value = getattr(self, field_name)
            if not value:
                object.__setattr__(
                    self,
                    field_name,
                    default_transform_group(mode, pose_format),
                )
            elif not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")
        if self.action_group in self.passthrough_action_groups:
            raise ValueError(
                f"action_group {self.action_group!r} cannot also be a passthrough group"
            )

        stats_path = Path(self.stats_file)
        if stats_path.is_absolute() or ".." in stats_path.parts:
            raise ValueError(
                f"stats_file must be a dataset-relative safe path: {self.stats_file!r}"
            )

    # --- 파생 계약 -----------------------------------------------------------

    @property
    def space(self) -> ActionSpace:
        return self.mode.space

    @property
    def reference(self) -> ActionReference:
        return self.mode.reference

    @property
    def is_eef(self) -> bool:
        return self.mode.is_eef

    @property
    def is_relative(self) -> bool:
        return self.mode.is_relative

    @property
    def rotation_dim(self) -> int:
        return pose_format_dims(self.pose_format)[0]

    @property
    def pose_dim(self) -> int:
        """gripper를 제외한 EEF pose 차원. joint mode에서는 오류."""
        if not self.is_eef:
            raise ValueError(f"mode={self.mode.value!r} has no EEF pose dimension")
        return pose_format_dims(self.pose_format)[1]

    @property
    def semantics(self) -> dict[str, str]:
        return dict(STATE_ACTION_SEMANTICS[self.mode])

    @property
    def inference_routing(self) -> tuple[str, ...]:
        return INFERENCE_ROUTING[self.mode]

    @property
    def stats_profile_kind(self) -> str:
        """Mode/pose-format별로 분리되는 stats profile 이름."""
        if not self.is_eef:
            return self.mode.value
        suffix = {
            PoseFormat.XYZ_ROT6D_ROWS: "rot6d",
            PoseFormat.XYZ_QUATERNION_WXYZ: "wxyz",
            PoseFormat.XYZ_RPY: "rpy",
        }[self.pose_format]
        return f"{self.mode.value}_{suffix}"

    def expected_action_dim(self, *, joint_dim: int | None = None) -> int:
        """Gripper를 포함한 전체 action 차원.

        EEF mode는 pose format으로 결정된다. joint mode는 dataset feature metadata에서
        resolve한 ``joint_dim``이 있어야 하며 6D/7D를 하드코딩하지 않는다.
        """
        gripper_dim = len(self.passthrough_action_groups)
        if self.is_eef:
            return self.pose_dim + gripper_dim
        if joint_dim is None:
            raise ValueError(
                f"mode={self.mode.value!r} action dimension must be resolved from dataset "
                "feature metadata; pass joint_dim"
            )
        if not isinstance(joint_dim, int) or joint_dim <= 0:
            raise ValueError(f"joint_dim must be a positive integer, got {joint_dim!r}")
        return joint_dim + gripper_dim

    def expected_state_dim(self, *, joint_dim: int | None = None) -> int:
        """State는 모든 mode에서 action과 같은 absolute layout을 쓴다."""
        return self.expected_action_dim(joint_dim=joint_dim)

    # --- 직렬화 --------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ACTION_REPRESENTATION_SCHEMA_VERSION,
            "spec_version": ACTION_REPRESENTATION_SPEC_VERSION,
            "mode": self.mode.value,
            "space": self.space.value,
            "reference": self.reference.value,
            "pose_format": self.pose_format.value,
            "reference_observation": self.reference_observation,
            "state_group": self.state_group,
            "action_group": self.action_group,
            "passthrough_action_groups": list(self.passthrough_action_groups),
            "gripper_representation": self.gripper_representation,
            "dataset_storage_reference": DATASET_STORAGE_REFERENCE,
            "base_frame": self.base_frame,
            "eef_frame": self.eef_frame,
            "stats_file": self.stats_file,
            "stats_profile_kind": self.stats_profile_kind,
            "strict": self.strict,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ActionRepresentationSpec:
        if not isinstance(payload, dict):
            raise TypeError(
                f"action representation payload must be a JSON object, "
                f"got {type(payload).__name__}"
            )
        schema_version = payload.get("schema_version", ACTION_REPRESENTATION_SCHEMA_VERSION)
        if schema_version != ACTION_REPRESENTATION_SCHEMA_VERSION:
            raise ValueError(
                f"action representation schema_version must be "
                f"{ACTION_REPRESENTATION_SCHEMA_VERSION}, got {schema_version!r}"
            )
        spec = cls(
            mode=coerce_mode(payload["mode"]),
            pose_format=coerce_pose_format(payload.get("pose_format")),
            reference_observation=payload.get("reference_observation", "current_observation"),
            state_group=payload.get("state_group", ""),
            action_group=payload.get("action_group", ""),
            passthrough_action_groups=tuple(
                payload.get("passthrough_action_groups", (DEFAULT_GRIPPER_GROUP,))
            ),
            gripper_representation=payload.get("gripper_representation", "absolute"),
            base_frame=payload.get("base_frame"),
            eef_frame=payload.get("eef_frame"),
            stats_file=payload.get("stats_file", DEFAULT_STATS_FILE),
            strict=bool(payload.get("strict", True)),
        )
        # 파생 필드가 저장된 값과 다르면 조용히 무시하지 않고 실패한다.
        for derived in ("space", "reference", "stats_profile_kind", "dataset_storage_reference"):
            if derived in payload and payload[derived] != spec.to_dict()[derived]:
                raise ValueError(
                    f"action representation derived field {derived!r} mismatch: "
                    f"{payload[derived]!r} != {spec.to_dict()[derived]!r}"
                )
        return spec

    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict())).hexdigest()

    def with_pose_format(self, pose_format: PoseFormat | str | None) -> ActionRepresentationSpec:
        return replace(self, pose_format=coerce_pose_format(pose_format))


# --- dataset 저장 원칙 -------------------------------------------------------


def validate_dataset_storage_reference(value: Any, *, source: str = "dataset") -> str:
    """Dataset이 relative 저장을 주장하면 거부한다.

    relative target은 training processor가 만들고 inference postprocessor가 되돌린다.
    영속 dataset은 mode와 무관하게 absolute다.
    """
    if value is None:
        return DATASET_STORAGE_REFERENCE
    if value != DATASET_STORAGE_REFERENCE:
        raise ValueError(
            f"{source} storage reference must be {DATASET_STORAGE_REFERENCE!r}; "
            f"relative datasets are not a persistence format, got {value!r}"
        )
    return DATASET_STORAGE_REFERENCE


def dataset_space_for_mode(mode: ActionRepresentationMode | str) -> str:
    """Mode → 필요한 absolute dataset 종류."""
    resolved = coerce_mode(mode)
    return "absolute_joint" if resolved.space is ActionSpace.JOINT else "absolute_eef"


# --- 조합 matrix -------------------------------------------------------------


def iter_representation_specs(**overrides: Any) -> Iterator[ActionRepresentationSpec]:
    """8개 (mode, pose_format) 조합 spec을 순서대로 생성."""
    for mode in ActionRepresentationMode:
        if mode.is_eef:
            for pose_format in EEF_POSE_FORMATS:
                yield ActionRepresentationSpec(
                    mode=mode,
                    pose_format=pose_format,
                    **overrides,
                )
        else:
            yield ActionRepresentationSpec(mode=mode, **overrides)


def iter_policy_combinations(
    policies: Sequence[str] = SUPPORTED_POLICY_FAMILIES,
    **overrides: Any,
) -> Iterator[tuple[str, ActionRepresentationSpec]]:
    """policy × representation 24-combination matrix."""
    for policy in policies:
        if policy not in SUPPORTED_POLICY_FAMILIES:
            raise ValueError(
                f"unsupported policy family {policy!r}; expected one of "
                f"{list(SUPPORTED_POLICY_FAMILIES)}"
            )
        for spec in iter_representation_specs(**overrides):
            yield policy, spec


def combination_id(policy: str, spec: ActionRepresentationSpec) -> str:
    """24-combination matrix에서 쓰는 안정적인 조합 식별자."""
    return f"{policy}__{spec.stats_profile_kind}"


# --- v1 legacy 호환 ----------------------------------------------------------


def from_legacy_v1_config(
    config: Any,
    *,
    allow_legacy_absolute: bool = False,
) -> ActionRepresentationSpec:
    """v1 :class:`ActionRepresentationConfig`를 v2 spec으로 승격.

    v1 ``mode='eef_relative'``는 정보 손실 없이 매핑된다. v1 ``mode='absolute'``는
    joint absolute를 의미했지만 config만으로는 모호하므로 명시적 opt-in에서만 허용한다.
    """
    mode = getattr(config, "mode", None)
    if mode == "eef_relative":
        return ActionRepresentationSpec(
            mode=ActionRepresentationMode.EEF_RELATIVE,
            pose_format=coerce_pose_format(getattr(config, "pose_format", None)),
            state_group=getattr(config, "state_pose_group", "") or "",
            action_group=getattr(config, "action_pose_group", "") or "",
            passthrough_action_groups=tuple(
                getattr(config, "passthrough_action_groups", (DEFAULT_GRIPPER_GROUP,))
            ),
            base_frame=getattr(config, "base_frame", None) or DEFAULT_BASE_FRAME,
            eef_frame=getattr(config, "eef_frame", None) or DEFAULT_EEF_FRAME,
            stats_file=getattr(config, "stats_file", DEFAULT_STATS_FILE),
            strict=bool(getattr(config, "strict", True)),
        )
    if mode == "absolute":
        if not allow_legacy_absolute:
            raise ValueError(
                "legacy mode='absolute' is ambiguous in schema v2; pass "
                "allow_legacy_absolute=True to interpret it as 'joint_absolute' and record "
                "the legacy opt-in in the checkpoint manifest"
            )
        return ActionRepresentationSpec(mode=ActionRepresentationMode.JOINT_ABSOLUTE)
    raise ValueError(f"cannot promote legacy action representation mode {mode!r} to schema v2")
