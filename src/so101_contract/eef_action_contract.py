"""Absolute EEF LeRobot v3 dataset의 action representation 계약.

학습 전에 ``meta/info.json``과 ``meta/modality.json``을 검증해
``xyz + Rot6D(rows) + absolute gripper`` 10D layout을 명시적으로 resolve한다.
추론 checkpoint에는 resolve 결과와 fingerprint를 저장해 dataset 없이도 같은
processor 구성을 복원할 수 있다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Literal

ACTION_REPRESENTATION_CONTRACT_VERSION = "so101_eef_action_contract_v1"
ACTION_REPRESENTATION_MODES = ("absolute", "eef_relative")

CANONICAL_BASE_FRAME = "base_link"
CANONICAL_EEF_FRAME = "tcp_grasp"
CANONICAL_POSE_FORMAT = "xyz_rot6d_rows"
CANONICAL_STATE_KEY = "observation.state"
CANONICAL_ACTION_KEY = "action"
CANONICAL_POSE_GROUP = "eef_9d"
CANONICAL_GRIPPER_GROUP = "gripper_position"
CANONICAL_EEF_NAMES = (
    "tcp_grasp.x",
    "tcp_grasp.y",
    "tcp_grasp.z",
    "tcp_grasp.rot6d.r0c0",
    "tcp_grasp.rot6d.r0c1",
    "tcp_grasp.rot6d.r0c2",
    "tcp_grasp.rot6d.r1c0",
    "tcp_grasp.rot6d.r1c1",
    "tcp_grasp.rot6d.r1c2",
)
CANONICAL_GRIPPER_NAMES = ("gripper.pos",)
CANONICAL_ACTION_NAMES = CANONICAL_EEF_NAMES + CANONICAL_GRIPPER_NAMES
CANONICAL_ACTION_DIM = len(CANONICAL_ACTION_NAMES)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ActionRepresentationConfig:
    """v1 nested action representation config (**legacy shim**).

    schema v2의 canonical config는
    :class:`so101_contract.action_representation.ActionRepresentationSpec`이다. 이 클래스는
    이미 배포된 v1 EEF-relative 학습/추론 경로가 계속 동작하도록 남겨 둔 호환 계층이며,
    ``mode='absolute'``는 v2에서 금지된 모호한 값이다. 신규 코드는 v2 spec을 쓰고,
    승격이 필요하면 :meth:`to_spec`을 사용한다.
    """

    mode: Literal["absolute", "eef_relative"] = "absolute"
    reference: Literal["current_observation"] = "current_observation"
    pose_format: Literal["xyz_rot6d_rows"] = "xyz_rot6d_rows"
    state_pose_group: str = CANONICAL_POSE_GROUP
    action_pose_group: str = CANONICAL_POSE_GROUP
    passthrough_action_groups: tuple[str, ...] = (CANONICAL_GRIPPER_GROUP,)
    base_frame: str = CANONICAL_BASE_FRAME
    eef_frame: str = CANONICAL_EEF_FRAME
    stats_file: str = "meta/relative_action_stats.json"
    strict: bool = True

    def __post_init__(self) -> None:
        if self.mode not in ACTION_REPRESENTATION_MODES:
            raise ValueError(
                f"unknown action representation mode {self.mode!r}; "
                f"expected one of {ACTION_REPRESENTATION_MODES}"
            )
        if self.reference != "current_observation":
            raise ValueError(
                f"unsupported relative reference {self.reference!r}; "
                "v1 requires 'current_observation'"
            )
        if self.pose_format != CANONICAL_POSE_FORMAT:
            raise ValueError(
                f"unsupported pose format {self.pose_format!r}; "
                f"v1 requires {CANONICAL_POSE_FORMAT!r}"
            )
        stats_path = Path(self.stats_file)
        if stats_path.is_absolute() or ".." in stats_path.parts:
            raise ValueError(f"stats_file must be a dataset-relative safe path: {self.stats_file!r}")
        if not self.passthrough_action_groups:
            raise ValueError("passthrough_action_groups must contain the absolute gripper group")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["passthrough_action_groups"] = list(self.passthrough_action_groups)
        return value

    def to_spec(self, *, allow_legacy_absolute: bool = False) -> Any:
        """v2 :class:`ActionRepresentationSpec`로 승격.

        ``mode='absolute'``는 joint absolute를 뜻했지만 config만으로는 EEF absolute와
        구분되지 않으므로 ``allow_legacy_absolute=True`` 명시적 opt-in에서만 허용한다.
        """
        from .action_representation import from_legacy_v1_config

        return from_legacy_v1_config(self, allow_legacy_absolute=allow_legacy_absolute)


@dataclass(frozen=True)
class ResolvedEEFActionContract:
    """Dataset metadata에서 resolve한 processor/checkpoint용 불변 계약."""

    schema_version: str
    config: ActionRepresentationConfig
    state_key: str
    action_key: str
    state_dim: int
    action_dim: int
    state_pose_indices: tuple[int, ...]
    action_pose_indices: tuple[int, ...]
    passthrough_action_indices: tuple[int, ...]
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    base_frame: str
    eef_frame: str
    eef_kinematics_version: str
    urdf_sha256: str
    robot_yaml_sha256: str
    info_sha256: str
    modality_sha256: str
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config": self.config.to_dict(),
            "state_key": self.state_key,
            "action_key": self.action_key,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "state_pose_indices": list(self.state_pose_indices),
            "action_pose_indices": list(self.action_pose_indices),
            "passthrough_action_indices": list(self.passthrough_action_indices),
            "state_names": list(self.state_names),
            "action_names": list(self.action_names),
            "base_frame": self.base_frame,
            "eef_frame": self.eef_frame,
            "eef_kinematics_version": self.eef_kinematics_version,
            "urdf_sha256": self.urdf_sha256,
            "robot_yaml_sha256": self.robot_yaml_sha256,
            "info_sha256": self.info_sha256,
            "modality_sha256": self.modality_sha256,
            "fingerprint": self.fingerprint,
        }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json_object(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(f"required dataset metadata not found: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"metadata root must be a JSON object: {path}")
    return value, _sha256_bytes(raw)


def _feature_contract(
    info: dict[str, Any],
    key: str,
) -> tuple[int, tuple[str, ...]]:
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


def _group_indices(
    modality: dict[str, Any],
    *,
    modality_name: str,
    group_name: str,
    feature_dim: int,
) -> tuple[int, ...]:
    modality_groups = modality.get(modality_name)
    if not isinstance(modality_groups, dict):
        raise KeyError(f"meta/modality.json missing {modality_name!r} groups")
    group = modality_groups.get(group_name)
    if not isinstance(group, dict):
        raise KeyError(f"meta/modality.json missing {modality_name}.{group_name!r}")
    start, end = group.get("start"), group.get("end")
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError(
            f"{modality_name}.{group_name} start/end must be integers, got {group!r}"
        )
    if start < 0 or end <= start or end > feature_dim:
        raise ValueError(
            f"{modality_name}.{group_name} range [{start},{end}) is invalid for dim {feature_dim}"
        )
    return tuple(range(start, end))


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest, got {value!r}")
    return value


def _fingerprint(value: dict[str, Any]) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(canonical)


def resolve_eef_action_contract(
    dataset_root: str | Path,
    config: ActionRepresentationConfig | None = None,
) -> ResolvedEEFActionContract:
    """Canonical 10D absolute EEF dataset metadata를 검증하고 index/fingerprint를 resolve."""
    config = config or ActionRepresentationConfig(mode="eef_relative")
    if config.mode != "eef_relative":
        raise ValueError(
            f"resolve_eef_action_contract requires mode='eef_relative', got {config.mode!r}"
        )

    root = Path(dataset_root)
    info, info_sha256 = _read_json_object(root / "meta" / "info.json")
    modality, modality_sha256 = _read_json_object(root / "meta" / "modality.json")
    if info.get("codebase_version") != "v3.0":
        raise ValueError(
            f"LeRobot codebase_version must be 'v3.0', got {info.get('codebase_version')!r}"
        )

    state_dim, state_names = _feature_contract(info, CANONICAL_STATE_KEY)
    action_dim, action_names = _feature_contract(info, CANONICAL_ACTION_KEY)
    if state_dim != CANONICAL_ACTION_DIM or action_dim != CANONICAL_ACTION_DIM:
        raise ValueError(
            f"v1 EEF-relative requires state/action dim {CANONICAL_ACTION_DIM}, "
            f"got {state_dim}/{action_dim}"
        )
    if state_names != CANONICAL_ACTION_NAMES:
        raise ValueError(
            f"{CANONICAL_STATE_KEY!r} names mismatch: {state_names}; "
            f"expected {CANONICAL_ACTION_NAMES}"
        )
    if action_names != CANONICAL_ACTION_NAMES:
        raise ValueError(
            f"{CANONICAL_ACTION_KEY!r} names mismatch: {action_names}; "
            f"expected {CANONICAL_ACTION_NAMES}"
        )

    state_pose_indices = _group_indices(
        modality,
        modality_name="state",
        group_name=config.state_pose_group,
        feature_dim=state_dim,
    )
    action_pose_indices = _group_indices(
        modality,
        modality_name="action",
        group_name=config.action_pose_group,
        feature_dim=action_dim,
    )
    if len(state_pose_indices) != 9 or len(action_pose_indices) != 9:
        raise ValueError(
            "state/action EEF pose groups must both be 9D, got "
            f"{len(state_pose_indices)}/{len(action_pose_indices)}"
        )

    passthrough_indices: list[int] = []
    for group_name in config.passthrough_action_groups:
        passthrough_indices.extend(
            _group_indices(
                modality,
                modality_name="action",
                group_name=group_name,
                feature_dim=action_dim,
            )
        )
    if len(set(passthrough_indices)) != len(passthrough_indices):
        raise ValueError(f"passthrough action groups overlap: {passthrough_indices}")
    if set(action_pose_indices).intersection(passthrough_indices):
        raise ValueError("action pose and passthrough groups overlap")
    classified = set(action_pose_indices).union(passthrough_indices)
    if classified != set(range(action_dim)):
        raise ValueError(
            f"unclassified or duplicate action dimensions: classified={sorted(classified)}, "
            f"expected={list(range(action_dim))}"
        )

    conversion = info.get("so101_eef_conversion")
    if not isinstance(conversion, dict):
        raise KeyError("meta/info.json missing 'so101_eef_conversion'")
    required_pairs = {
        "base_frame": config.base_frame,
        "eef_frame": config.eef_frame,
        "rotation_representation": "rot6d",
        "rotation_format": "xyz+rot6d_rows",
        "gripper_format": "canonical_policy_feature_[0,100]",
        "keep_joints": False,
    }
    for key, expected in required_pairs.items():
        if conversion.get(key) != expected:
            raise ValueError(
                f"so101_eef_conversion.{key} mismatch: "
                f"{conversion.get(key)!r} != {expected!r}"
            )
    eef_kinematics_version = conversion.get("eef_kinematics_version")
    if not isinstance(eef_kinematics_version, str) or not eef_kinematics_version:
        raise ValueError("so101_eef_conversion.eef_kinematics_version is missing")
    urdf_sha256 = _require_sha256(conversion.get("urdf_sha256"), "urdf_sha256")
    robot_yaml_sha256 = _require_sha256(
        conversion.get("robot_yaml_sha256"),
        "robot_yaml_sha256",
    )

    fingerprint_payload = {
        "schema_version": ACTION_REPRESENTATION_CONTRACT_VERSION,
        "config": config.to_dict(),
        "state_key": CANONICAL_STATE_KEY,
        "action_key": CANONICAL_ACTION_KEY,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "state_pose_indices": list(state_pose_indices),
        "action_pose_indices": list(action_pose_indices),
        "passthrough_action_indices": sorted(passthrough_indices),
        "state_names": list(state_names),
        "action_names": list(action_names),
        "base_frame": config.base_frame,
        "eef_frame": config.eef_frame,
        "eef_kinematics_version": eef_kinematics_version,
        "urdf_sha256": urdf_sha256,
        "robot_yaml_sha256": robot_yaml_sha256,
        "info_sha256": info_sha256,
        "modality_sha256": modality_sha256,
    }
    return ResolvedEEFActionContract(
        schema_version=ACTION_REPRESENTATION_CONTRACT_VERSION,
        config=config,
        state_key=CANONICAL_STATE_KEY,
        action_key=CANONICAL_ACTION_KEY,
        state_dim=state_dim,
        action_dim=action_dim,
        state_pose_indices=state_pose_indices,
        action_pose_indices=action_pose_indices,
        passthrough_action_indices=tuple(sorted(passthrough_indices)),
        state_names=state_names,
        action_names=action_names,
        base_frame=config.base_frame,
        eef_frame=config.eef_frame,
        eef_kinematics_version=eef_kinematics_version,
        urdf_sha256=urdf_sha256,
        robot_yaml_sha256=robot_yaml_sha256,
        info_sha256=info_sha256,
        modality_sha256=modality_sha256,
        fingerprint=_fingerprint(fingerprint_payload),
    )
