"""Phase 14 — schema v2 dataset action contract (4 mode × 3 pose format).

v1 :mod:`so101_contract.eef_action_contract`는 ``eef_relative + rot6d`` 10D 한 조합만
resolve했다. 이 모듈은 같은 역할을 4개 mode와 3개 EEF pose format으로 넓히고, resolve된
index/topology로 :class:`~so101_contract.action_transform.ActionRepresentationTransform`를
만든다.

metadata resolve 우선순위(추정 금지, 없으면 fail-fast):

1. ``meta/info.json``의 ``so101_action_representation`` 블록 — schema v2 dataset이 직접
   선언한 group/joint topology. joint dataset은 이 경로를 MUST 사용한다.
2. ``meta/modality.json`` group + ``meta/info.json``의 ``so101_eef_conversion`` —
   ``scripts/convert/joint_dataset_to_eef.py``가 현재 출력하는 EEF metadata.
3. 호출자가 명시적으로 주입한 ``joint_metadata`` (Phase 15 이전 joint dataset 보강용).

feature names는 format별 canonical 목록과 **정확히** 일치해야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .action_representation import (
    ActionRepresentationSpec,
    PoseFormat,
    validate_dataset_storage_reference,
)
from .action_transform import ActionRepresentationTransform
from .joint_topology import JointTopology

ACTION_DATASET_CONTRACT_VERSION = "so101_dataset_action_contract_v2"
DATASET_CONTRACT_BLOCK = "so101_action_representation"
EEF_CONVERSION_BLOCK = "so101_eef_conversion"

CANONICAL_STATE_KEY = "observation.state"
CANONICAL_ACTION_KEY = "action"
CANONICAL_GRIPPER_NAME = "gripper.pos"
CANONICAL_GRIPPER_GROUP = "gripper_position"

_XYZ_NAMES = ("tcp_grasp.x", "tcp_grasp.y", "tcp_grasp.z")

#: converter(`scripts/convert/joint_dataset_to_eef.py`)가 기록하는 rotation feature names.
EEF_ROTATION_NAMES: dict[PoseFormat, tuple[str, ...]] = {
    PoseFormat.XYZ_ROT6D_ROWS: (
        "tcp_grasp.rot6d.r0c0",
        "tcp_grasp.rot6d.r0c1",
        "tcp_grasp.rot6d.r0c2",
        "tcp_grasp.rot6d.r1c0",
        "tcp_grasp.rot6d.r1c1",
        "tcp_grasp.rot6d.r1c2",
    ),
    PoseFormat.XYZ_QUATERNION_WXYZ: (
        "tcp_grasp.quaternion.w",
        "tcp_grasp.quaternion.x",
        "tcp_grasp.quaternion.y",
        "tcp_grasp.quaternion.z",
    ),
    PoseFormat.XYZ_RPY: (
        "tcp_grasp.rpy.roll",
        "tcp_grasp.rpy.pitch",
        "tcp_grasp.rpy.yaw",
    ),
}

#: converter metadata의 ``rotation_representation`` ↔ v2 pose format.
ROTATION_REPRESENTATION_TO_POSE_FORMAT: dict[str, PoseFormat] = {
    "rot6d": PoseFormat.XYZ_ROT6D_ROWS,
    "wxyz": PoseFormat.XYZ_QUATERNION_WXYZ,
    "rpy": PoseFormat.XYZ_RPY,
}

#: converter metadata의 ``rotation_format`` 문자열.
POSE_FORMAT_METADATA_STRING: dict[PoseFormat, str] = {
    PoseFormat.XYZ_ROT6D_ROWS: "xyz+rot6d_rows",
    PoseFormat.XYZ_QUATERNION_WXYZ: "xyz+quaternion_wxyz_unit_canonical",
    PoseFormat.XYZ_RPY: "xyz+rpy_fixed_axis_radians",
}

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def eef_feature_names(pose_format: PoseFormat) -> tuple[str, ...]:
    """gripper를 포함한 EEF feature names."""
    try:
        rotation_names = EEF_ROTATION_NAMES[pose_format]
    except KeyError as exc:
        raise ValueError(f"{pose_format!r} is not an EEF pose format") from exc
    return _XYZ_NAMES + rotation_names + (CANONICAL_GRIPPER_NAME,)


@dataclass(frozen=True)
class ResolvedActionContract:
    """Dataset metadata에서 resolve한 processor/stats/checkpoint 공통 계약."""

    schema_version: str
    spec: ActionRepresentationSpec
    state_key: str
    action_key: str
    state_dim: int
    action_dim: int
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    state_groups: dict[str, tuple[int, int]]
    action_groups: dict[str, tuple[int, int]]
    transform: ActionRepresentationTransform
    base_frame: str | None
    eef_frame: str | None
    eef_kinematics_version: str | None
    urdf_sha256: str | None
    robot_yaml_sha256: str | None
    info_sha256: str
    modality_sha256: str | None
    fingerprint: str

    @property
    def joint_topology(self) -> JointTopology | None:
        return self.transform.joint_topology

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "action_representation": self.spec.to_dict(),
            "state_key": self.state_key,
            "action_key": self.action_key,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "state_names": list(self.state_names),
            "action_names": list(self.action_names),
            "state_groups": {name: list(bounds) for name, bounds in self.state_groups.items()},
            "action_groups": {name: list(bounds) for name, bounds in self.action_groups.items()},
            "transform": self.transform.to_dict(),
            "base_frame": self.base_frame,
            "eef_frame": self.eef_frame,
            "eef_kinematics_version": self.eef_kinematics_version,
            "urdf_sha256": self.urdf_sha256,
            "robot_yaml_sha256": self.robot_yaml_sha256,
            "info_sha256": self.info_sha256,
            "modality_sha256": self.modality_sha256,
            "fingerprint": self.fingerprint,
        }

    def feature_groups(self, key: str) -> dict[str, list[int]]:
        """:func:`so101_contract.action_manifest.build_feature_contract` 입력 형태."""
        groups = self.state_groups if key == self.state_key else self.action_groups
        return {name: [bounds[0], bounds[1]] for name, bounds in groups.items()}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(f"required dataset metadata not found: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"metadata root must be a JSON object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _feature_contract(info: dict[str, Any], key: str) -> tuple[int, tuple[str, ...]]:
    features = info.get("features")
    if not isinstance(features, dict) or key not in features:
        raise KeyError(f"meta/info.json features missing {key!r}")
    feature = features[key]
    if not isinstance(feature, dict):
        raise TypeError(f"feature metadata must be an object: {key!r}")
    if feature.get("dtype") != "float32":
        raise ValueError(f"{key!r} dtype must be 'float32', got {feature.get('dtype')!r}")
    shape = feature.get("shape")
    if not isinstance(shape, list) or len(shape) != 1 or not isinstance(shape[0], int):
        raise ValueError(f"{key!r} shape must be one-dimensional, got {shape!r}")
    names = feature.get("names")
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError(f"{key!r} names must be a string list, got {names!r}")
    if len(names) != shape[0]:
        raise ValueError(f"{key!r} names/shape mismatch: {len(names)} != {shape[0]}")
    return int(shape[0]), tuple(names)


def _normalize_groups(
    groups: Any,
    *,
    dim: int,
    label: str,
) -> dict[str, tuple[int, int]]:
    if not isinstance(groups, dict) or not groups:
        raise ValueError(f"{label} groups must be a non-empty object, got {groups!r}")
    resolved: dict[str, tuple[int, int]] = {}
    covered: set[int] = set()
    for name, bounds in groups.items():
        if isinstance(bounds, dict):
            start, end = bounds.get("start"), bounds.get("end")
        elif isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            start, end = bounds
        else:
            raise ValueError(f"{label}.{name} bounds must be [start,end) or {{start,end}}")
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError(f"{label}.{name} start/end must be integers, got {bounds!r}")
        if start < 0 or end <= start or end > dim:
            raise ValueError(f"{label}.{name} range [{start},{end}) is invalid for dim {dim}")
        indices = set(range(start, end))
        if covered & indices:
            raise ValueError(f"{label} groups overlap at {sorted(covered & indices)}")
        covered |= indices
        resolved[str(name)] = (start, end)
    if covered != set(range(dim)):
        raise ValueError(
            f"{label} groups do not partition the feature vector: "
            f"covered={sorted(covered)}, expected={list(range(dim))}"
        )
    return resolved


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest, got {value!r}")
    return value


def _validate_eef_metadata(
    info: dict[str, Any],
    spec: ActionRepresentationSpec,
    names: tuple[str, ...],
) -> dict[str, Any]:
    conversion = info.get(EEF_CONVERSION_BLOCK)
    if not isinstance(conversion, dict):
        raise KeyError(f"meta/info.json missing {EEF_CONVERSION_BLOCK!r} for an EEF mode")
    representation = conversion.get("rotation_representation")
    resolved_format = ROTATION_REPRESENTATION_TO_POSE_FORMAT.get(str(representation))
    if resolved_format is None:
        raise ValueError(
            f"unknown rotation_representation {representation!r}; expected one of "
            f"{sorted(ROTATION_REPRESENTATION_TO_POSE_FORMAT)}"
        )
    if resolved_format is not spec.pose_format:
        raise ValueError(
            f"dataset rotation representation {resolved_format.value!r} does not match "
            f"config pose_format {spec.pose_format.value!r}"
        )
    expected_pairs = {
        "base_frame": spec.base_frame,
        "eef_frame": spec.eef_frame,
        "rotation_format": POSE_FORMAT_METADATA_STRING[spec.pose_format],
        "gripper_format": "canonical_policy_feature_[0,100]",
        "keep_joints": False,
    }
    for key, expected in expected_pairs.items():
        if conversion.get(key) != expected:
            raise ValueError(
                f"{EEF_CONVERSION_BLOCK}.{key} mismatch: {conversion.get(key)!r} != {expected!r}"
            )
    expected_names = eef_feature_names(spec.pose_format)
    if names != expected_names:
        raise ValueError(
            f"EEF feature names mismatch for {spec.pose_format.value}: {names}; "
            f"expected {expected_names}"
        )
    kinematics_version = conversion.get("eef_kinematics_version")
    if not isinstance(kinematics_version, str) or not kinematics_version:
        raise ValueError(f"{EEF_CONVERSION_BLOCK}.eef_kinematics_version is missing")
    return {
        "eef_kinematics_version": kinematics_version,
        "urdf_sha256": _require_sha256(conversion.get("urdf_sha256"), "urdf_sha256"),
        "robot_yaml_sha256": _require_sha256(
            conversion.get("robot_yaml_sha256"),
            "robot_yaml_sha256",
        ),
    }


def _resolve_joint_topology(
    info: dict[str, Any],
    spec: ActionRepresentationSpec,
    *,
    action_names: tuple[str, ...],
    action_groups: dict[str, tuple[int, int]],
    joint_metadata: dict[str, dict[str, Any]] | None,
) -> tuple[JointTopology, tuple[int, ...]]:
    block = info.get(DATASET_CONTRACT_BLOCK)
    declared: dict[str, dict[str, Any]] | None = None
    if isinstance(block, dict) and isinstance(block.get("joints"), list):
        declared = {}
        for entry in block["joints"]:
            if not isinstance(entry, dict) or "name" not in entry:
                raise ValueError(f"{DATASET_CONTRACT_BLOCK}.joints entries must declare a name")
            declared[str(entry["name"])] = {
                key: value for key, value in entry.items() if key != "name"
            }
    metadata = joint_metadata if joint_metadata is not None else declared
    if metadata is None:
        raise KeyError(
            "joint mode requires explicit joint topology metadata; declare "
            f"meta/info.json['{DATASET_CONTRACT_BLOCK}']['joints'] or pass joint_metadata"
        )
    return JointTopology.from_feature_metadata(
        action_names,
        {name: list(bounds) for name, bounds in action_groups.items()},
        joint_group=spec.action_group,
        joint_metadata=metadata,
    )


def resolve_action_contract_from_metadata(
    info: dict[str, Any],
    modality: dict[str, Any] | None,
    spec: ActionRepresentationSpec,
    *,
    info_sha256: str,
    modality_sha256: str | None = None,
    joint_metadata: dict[str, dict[str, Any]] | None = None,
) -> ResolvedActionContract:
    """이미 읽어 둔 metadata dict에서 계약을 resolve."""
    if info.get("codebase_version") != "v3.0":
        raise ValueError(
            f"LeRobot codebase_version must be 'v3.0', got {info.get('codebase_version')!r}"
        )
    block = info.get(DATASET_CONTRACT_BLOCK)
    if isinstance(block, dict):
        validate_dataset_storage_reference(
            block.get("storage_reference"),
            source=DATASET_CONTRACT_BLOCK,
        )
        declared_space = block.get("space")
        if declared_space is not None and declared_space != spec.space.value:
            raise ValueError(
                f"dataset declares space {declared_space!r} but config mode is "
                f"{spec.mode.value!r}"
            )

    state_dim, state_names = _feature_contract(info, CANONICAL_STATE_KEY)
    action_dim, action_names = _feature_contract(info, CANONICAL_ACTION_KEY)
    if state_dim != action_dim:
        raise ValueError(
            "state/action must share the same absolute layout in every mode: "
            f"{state_dim} != {action_dim}"
        )
    if state_names != action_names:
        raise ValueError(
            "state/action feature names must match; the dataset stores the same absolute "
            f"layout for both: {state_names} != {action_names}"
        )

    group_source: Any = None
    if isinstance(block, dict) and isinstance(block.get("groups"), dict):
        group_source = {"state": block["groups"], "action": block["groups"]}
    elif modality is not None:
        group_source = modality
    if group_source is None:
        raise KeyError(
            "no group metadata found; provide meta/modality.json or declare "
            f"meta/info.json['{DATASET_CONTRACT_BLOCK}']['groups']"
        )
    state_groups = _normalize_groups(group_source.get("state"), dim=state_dim, label="state")
    action_groups = _normalize_groups(group_source.get("action"), dim=action_dim, label="action")

    if spec.state_group not in state_groups:
        raise KeyError(
            f"state metadata has no transform group {spec.state_group!r}; "
            f"groups={sorted(state_groups)}"
        )
    if spec.action_group not in action_groups:
        raise KeyError(
            f"action metadata has no transform group {spec.action_group!r}; "
            f"groups={sorted(action_groups)}"
        )
    passthrough_indices: list[int] = []
    for group in spec.passthrough_action_groups:
        if group not in action_groups:
            raise KeyError(f"action metadata has no passthrough group {group!r}")
        start, end = action_groups[group]
        passthrough_indices.extend(range(start, end))

    kinematics: dict[str, Any] = {
        "eef_kinematics_version": None,
        "urdf_sha256": None,
        "robot_yaml_sha256": None,
    }
    joint_topology: JointTopology | None = None
    if spec.is_eef:
        kinematics = _validate_eef_metadata(info, spec, action_names)
        start, end = action_groups[spec.action_group]
        if end - start != spec.pose_dim:
            raise ValueError(
                f"EEF pose group {spec.action_group!r} must be {spec.pose_dim}D for "
                f"{spec.pose_format.value}, got {end - start}D"
            )
        action_indices = tuple(range(start, end))
        state_start, state_end = state_groups[spec.state_group]
        state_indices = tuple(range(state_start, state_end))
        if len(state_indices) != spec.pose_dim:
            raise ValueError(
                f"state pose group must be {spec.pose_dim}D, got {len(state_indices)}D"
            )
    else:
        joint_topology, action_indices = _resolve_joint_topology(
            info,
            spec,
            action_names=action_names,
            action_groups=action_groups,
            joint_metadata=joint_metadata,
        )
        state_start, state_end = state_groups[spec.state_group]
        state_indices = tuple(range(state_start, state_end))
        if len(state_indices) != joint_topology.dim:
            raise ValueError(
                f"state joint group must be {joint_topology.dim}D, got {len(state_indices)}D"
            )
        if state_names[state_start:state_end] != joint_topology.names:
            raise ValueError(
                "state joint names differ from the action joint topology: "
                f"{state_names[state_start:state_end]} != {joint_topology.names}"
            )

    transform = ActionRepresentationTransform(
        spec=spec,
        state_indices=state_indices,
        action_indices=action_indices,
        passthrough_action_indices=tuple(sorted(passthrough_indices)),
        state_dim=state_dim,
        action_dim=action_dim,
        joint_topology=joint_topology,
    )

    fingerprint_payload = {
        "schema_version": ACTION_DATASET_CONTRACT_VERSION,
        "transform": transform.to_dict(),
        "state_key": CANONICAL_STATE_KEY,
        "action_key": CANONICAL_ACTION_KEY,
        "state_names": list(state_names),
        "action_names": list(action_names),
        "state_groups": {name: list(bounds) for name, bounds in state_groups.items()},
        "action_groups": {name: list(bounds) for name, bounds in action_groups.items()},
        "kinematics": kinematics,
        "info_sha256": info_sha256,
        "modality_sha256": modality_sha256,
    }
    return ResolvedActionContract(
        schema_version=ACTION_DATASET_CONTRACT_VERSION,
        spec=spec,
        state_key=CANONICAL_STATE_KEY,
        action_key=CANONICAL_ACTION_KEY,
        state_dim=state_dim,
        action_dim=action_dim,
        state_names=state_names,
        action_names=action_names,
        state_groups=state_groups,
        action_groups=action_groups,
        transform=transform,
        base_frame=spec.base_frame,
        eef_frame=spec.eef_frame,
        eef_kinematics_version=kinematics["eef_kinematics_version"],
        urdf_sha256=kinematics["urdf_sha256"],
        robot_yaml_sha256=kinematics["robot_yaml_sha256"],
        info_sha256=info_sha256,
        modality_sha256=modality_sha256,
        fingerprint=_sha256_json(fingerprint_payload),
    )


def resolve_action_representation_contract(
    dataset_root: str | Path,
    spec: ActionRepresentationSpec,
    *,
    joint_metadata: dict[str, dict[str, Any]] | None = None,
) -> ResolvedActionContract:
    """LeRobot v3 dataset 폴더에서 schema v2 action 계약을 resolve."""
    root = Path(dataset_root)
    info, info_sha256 = _read_json(root / "meta" / "info.json")
    modality_path = root / "meta" / "modality.json"
    modality: dict[str, Any] | None = None
    modality_sha256: str | None = None
    if modality_path.is_file():
        modality, modality_sha256 = _read_json(modality_path)
    return resolve_action_contract_from_metadata(
        info,
        modality,
        spec,
        info_sha256=info_sha256,
        modality_sha256=modality_sha256,
        joint_metadata=joint_metadata,
    )
