"""Phase 11 — universal ``action_representation.json`` (schema v2).

v1은 EEF-relative checkpoint에서만 manifest를 만들었고 joint absolute checkpoint에는
아무 계약 파일이 없었다. v2는 **mode와 무관하게 모든 신규 checkpoint가 manifest를
포함**하며, 없는 checkpoint는 추정하지 않고 fail-fast한다.

이 모듈은 LeRobot을 import하지 않는 core API다. 학습 runtime(LeRobot patch/factory)과
migration CLI 연결은 Phase 15~16 범위이며, 여기서는 재사용 가능한 schema/validation/
hash API만 제공한다.

계약 요약:

- ``schema_version = 2``이고 전체 내용 hash ``manifest_sha256``으로 tamper를 거부한다.
- 추론 CLI는 checkpoint representation을 **override하지 않고 assert**한다
  (:func:`assert_manifest_matches_cli`).
- manifest가 없는 legacy checkpoint는 ``--allow-legacy-joint-absolute-checkpoint``
  명시적 opt-in에서만 허용하고, 허용 사실을 새 manifest ``legacy`` 절에 기록한다.
- v1 manifest(``schema_version`` 이 문자열)는 :func:`manifest_schema_version`으로
  구분해 기존 EEF-relative 경로가 그대로 동작한다.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from .action_representation import (
    ACTION_REPRESENTATION_SCHEMA_VERSION,
    DATASET_STORAGE_REFERENCE,
    SUPPORTED_POLICY_FAMILIES,
    ActionRepresentationMode,
    ActionRepresentationSpec,
    PoseFormat,
    coerce_mode,
    coerce_pose_format,
    validate_dataset_storage_reference,
)

ACTION_REPRESENTATION_MANIFEST = "action_representation.json"
ACTION_MANIFEST_SCHEMA_VERSION = ACTION_REPRESENTATION_SCHEMA_VERSION
ACTION_MANIFEST_VERSION = "so101_action_representation_manifest_v2"
LEGACY_JOINT_ABSOLUTE_OPT_IN = "--allow-legacy-joint-absolute-checkpoint"

#: pose format별 회전 convention. metadata에 명시적으로 남긴다.
ROTATION_CONVENTIONS: dict[PoseFormat, str | None] = {
    PoseFormat.NOT_APPLICABLE: None,
    PoseFormat.XYZ_ROT6D_ROWS: "rotation_matrix_first_two_rows",
    PoseFormat.XYZ_QUATERNION_WXYZ: "unit_quaternion_scalar_first_canonical_sign",
    PoseFormat.XYZ_RPY: "fixed_axis_xyz_rpy_radians_wrapped",
}

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")

_REQUIRED_SECTIONS = ("action_representation", "features", "dataset", "stats", "policy", "runtime")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_manifest_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def so101_contract_source_sha256() -> str:
    """Checkpoint runtime에 실제 포함된 ``so101_contract`` Python source hash."""
    package_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    paths = sorted(package_root.rglob("*.py"))
    if not paths:
        raise RuntimeError(f"so101_contract source files not found under {package_root}")
    for path in paths:
        relative = path.relative_to(package_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        raw = path.read_bytes()
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest, got {value!r}")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _require_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object, got {type(value).__name__}")
    return value


# --- feature 계약 -------------------------------------------------------------


def build_feature_contract(
    key: str,
    names: list[str] | tuple[str, ...],
    groups: dict[str, tuple[int, int] | list[int]],
) -> dict[str, Any]:
    """State 또는 action feature의 names/groups/indices 계약."""
    resolved_names = tuple(str(name) for name in names)
    if not resolved_names:
        raise ValueError(f"{key} feature names must not be empty")
    resolved_groups = {
        str(group): [int(bounds[0]), int(bounds[1])] for group, bounds in groups.items()
    }
    _validate_group_partition(resolved_groups, dim=len(resolved_names), key=key)
    return {
        "key": key,
        "dim": len(resolved_names),
        "names": list(resolved_names),
        "groups": resolved_groups,
        "indices": {
            group: list(range(bounds[0], bounds[1]))
            for group, bounds in resolved_groups.items()
        },
    }


def _validate_group_partition(
    groups: dict[str, list[int]],
    *,
    dim: int,
    key: str,
) -> None:
    covered: set[int] = set()
    for group, bounds in groups.items():
        if len(bounds) != 2:
            raise ValueError(f"{key}.{group} bounds must be [start, end), got {bounds!r}")
        start, end = bounds
        if start < 0 or end <= start or end > dim:
            raise ValueError(f"{key}.{group} range [{start},{end}) is invalid for dim {dim}")
        indices = set(range(start, end))
        if covered & indices:
            raise ValueError(f"{key} groups overlap at {sorted(covered & indices)}")
        covered |= indices
    if covered != set(range(dim)):
        raise ValueError(
            f"{key} groups do not partition the feature vector: "
            f"covered={sorted(covered)}, expected={list(range(dim))}"
        )


# --- manifest 생성 ------------------------------------------------------------


def build_action_representation_manifest(
    spec: ActionRepresentationSpec,
    *,
    state_feature: dict[str, Any],
    action_feature: dict[str, Any],
    dataset: dict[str, Any],
    stats: dict[str, Any],
    policy: dict[str, Any],
    runtime: dict[str, Any],
    action_horizon: int,
    resolved_contract_fingerprint: str,
    transform: dict[str, Any] | None = None,
    kinematics: dict[str, Any] | None = None,
    legacy: dict[str, Any] | None = None,
    selective_reuse: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """모든 mode에서 동일한 schema v2 manifest payload를 만든다.

    호출자는 dataset/stats/policy/runtime identity를 직접 주입한다. 이 core API는
    LeRobot이나 Hugging Face runtime을 import하지 않는다.
    """
    if not isinstance(spec, ActionRepresentationSpec):
        raise TypeError(f"spec must be ActionRepresentationSpec, got {type(spec).__name__}")

    payload: dict[str, Any] = {
        "schema_version": ACTION_MANIFEST_SCHEMA_VERSION,
        "manifest_version": ACTION_MANIFEST_VERSION,
        # spec의 핵심 식별자는 도구가 파싱하기 쉽도록 최상위에도 평평하게 둔다.
        "mode": spec.mode.value,
        "space": spec.space.value,
        "reference": spec.reference.value,
        "pose_format": spec.pose_format.value,
        "state_dim": int(state_feature["dim"]),
        "action_dim": int(action_feature["dim"]),
        "gripper_representation": spec.gripper_representation,
        "base_frame": spec.base_frame,
        "eef_frame": spec.eef_frame,
        "rotation_convention": ROTATION_CONVENTIONS[spec.pose_format],
        "dataset_storage_reference": DATASET_STORAGE_REFERENCE,
        "action_horizon": _require_positive_int(action_horizon, "action_horizon"),
        # processor step과 manifest를 잇는 단일 명시적 필드. 두 곳이 어긋나면 load를 거부한다.
        "resolved_contract_fingerprint": _require_sha256(
            resolved_contract_fingerprint,
            "resolved_contract_fingerprint",
        ),
        "action_representation": spec.to_dict(),
        "transform": dict(transform) if transform is not None else None,
        "selective_reuse": dict(selective_reuse) if selective_reuse is not None else None,
        "features": {"state": state_feature, "action": action_feature},
        "dataset": dict(dataset),
        "stats": dict(stats),
        "policy": dict(policy),
        "runtime": dict(runtime),
        "kinematics": dict(kinematics) if kinematics is not None else None,
        "legacy": dict(legacy) if legacy is not None else {"allowed": False, "flag": None},
    }
    payload["manifest_sha256"] = canonical_manifest_sha256(payload)
    validate_action_representation_manifest(payload)
    return payload


# --- manifest 검증 ------------------------------------------------------------


def manifest_schema_version(manifest: dict[str, Any]) -> int:
    """v1(문자열 schema_version)과 v2(정수 2)를 구분."""
    _require_dict(manifest, "action representation manifest")
    version = manifest.get("schema_version")
    if version == ACTION_MANIFEST_SCHEMA_VERSION:
        return ACTION_MANIFEST_SCHEMA_VERSION
    if isinstance(version, str) and version.endswith("_v1"):
        return 1
    raise ValueError(f"unknown action representation manifest schema_version {version!r}")


def validate_action_representation_manifest(
    manifest: dict[str, Any],
    *,
    expected_spec: ActionRepresentationSpec | None = None,
    expected_policy_type: str | None = None,
    verify_runtime_source: bool = False,
) -> ActionRepresentationSpec:
    """Schema/hash/mode 정합을 검증하고 spec을 복원한다."""
    if manifest_schema_version(manifest) != ACTION_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "manifest is not schema v2; use the v1 EEF-relative loader or migrate it"
        )
    if manifest.get("manifest_version") != ACTION_MANIFEST_VERSION:
        raise ValueError(
            f"manifest_version must be {ACTION_MANIFEST_VERSION!r}, "
            f"got {manifest.get('manifest_version')!r}"
        )

    expected_hash = manifest.get("manifest_sha256")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if (
        not isinstance(expected_hash, str)
        or _SHA256_PATTERN.fullmatch(expected_hash) is None
        or canonical_manifest_sha256(unsigned) != expected_hash
    ):
        raise ValueError("action representation manifest content hash mismatch")

    missing = [name for name in _REQUIRED_SECTIONS if not isinstance(manifest.get(name), dict)]
    if missing:
        raise ValueError(f"action representation manifest is missing sections: {missing}")

    spec = ActionRepresentationSpec.from_dict(manifest["action_representation"])
    _validate_flat_identity(manifest, spec)
    _validate_features(manifest, spec)
    _validate_stats(manifest, spec)
    _validate_policy(manifest, expected_policy_type=expected_policy_type)
    _validate_dataset(manifest, spec)
    _validate_kinematics(manifest, spec)
    _validate_runtime(manifest, verify_runtime_source=verify_runtime_source)
    _validate_legacy(manifest, spec)

    if expected_spec is not None and expected_spec.fingerprint() != spec.fingerprint():
        raise ValueError(
            f"manifest action representation mismatch: {spec.mode.value}/"
            f"{spec.pose_format.value} != {expected_spec.mode.value}/"
            f"{expected_spec.pose_format.value}"
        )
    return spec


def _validate_flat_identity(manifest: dict[str, Any], spec: ActionRepresentationSpec) -> None:
    expected = {
        "mode": spec.mode.value,
        "space": spec.space.value,
        "reference": spec.reference.value,
        "pose_format": spec.pose_format.value,
        "gripper_representation": spec.gripper_representation,
        "base_frame": spec.base_frame,
        "eef_frame": spec.eef_frame,
        "rotation_convention": ROTATION_CONVENTIONS[spec.pose_format],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"manifest top-level {key!r} disagrees with action_representation: "
                f"{manifest.get(key)!r} != {value!r}"
            )
    validate_dataset_storage_reference(
        manifest.get("dataset_storage_reference"),
        source="manifest",
    )
    _require_positive_int(manifest.get("action_horizon"), "action_horizon")
    _require_sha256(
        manifest.get("resolved_contract_fingerprint"),
        "resolved_contract_fingerprint",
    )
    transform = manifest.get("transform")
    if transform is not None:
        if not isinstance(transform, dict):
            raise ValueError("manifest transform section must be an object or null")
        if transform.get("action_representation", {}).get("mode") != spec.mode.value:
            raise ValueError("manifest transform mode disagrees with action_representation")


def _validate_features(manifest: dict[str, Any], spec: ActionRepresentationSpec) -> None:
    features = manifest["features"]
    for key in ("state", "action"):
        feature = _require_dict(features.get(key), f"features.{key}")
        names = feature.get("names")
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise ValueError(f"features.{key}.names must be a string list")
        dim = _require_positive_int(feature.get("dim"), f"features.{key}.dim")
        if dim != len(names):
            raise ValueError(f"features.{key} dim/names mismatch: {dim} != {len(names)}")
        groups = _require_dict(feature.get("groups"), f"features.{key}.groups")
        _validate_group_partition(
            {group: list(bounds) for group, bounds in groups.items()},
            dim=dim,
            key=f"features.{key}",
        )

    action = features["action"]
    state = features["state"]
    if manifest.get("action_dim") != action["dim"] or manifest.get("state_dim") != state["dim"]:
        raise ValueError("manifest state_dim/action_dim disagree with the feature contract")

    transform_group = spec.action_group
    if transform_group not in action["groups"]:
        raise ValueError(
            f"action feature has no transform group {transform_group!r}; "
            f"groups={sorted(action['groups'])}"
        )
    if spec.state_group not in state["groups"]:
        raise ValueError(
            f"state feature has no transform group {spec.state_group!r}; "
            f"groups={sorted(state['groups'])}"
        )
    for group in spec.passthrough_action_groups:
        if group not in action["groups"]:
            raise ValueError(f"action feature has no passthrough group {group!r}")

    start, end = action["groups"][transform_group]
    transform_dim = int(end) - int(start)
    if spec.is_eef:
        if transform_dim != spec.pose_dim:
            raise ValueError(
                f"EEF pose group {transform_group!r} must be {spec.pose_dim}D for "
                f"{spec.pose_format.value}, got {transform_dim}D"
            )
        expected_action_dim = spec.expected_action_dim()
    else:
        # joint dimension은 하드코딩하지 않고 dataset feature metadata에서 resolve한다.
        expected_action_dim = spec.expected_action_dim(joint_dim=transform_dim)
    if action["dim"] != expected_action_dim:
        raise ValueError(
            f"action dim {action['dim']} disagrees with mode {spec.mode.value!r} "
            f"expectation {expected_action_dim}"
        )
    if state["dim"] != action["dim"]:
        raise ValueError(
            "state/action must share the same absolute layout in every mode: "
            f"{state['dim']} != {action['dim']}"
        )


def _validate_stats(manifest: dict[str, Any], spec: ActionRepresentationSpec) -> None:
    stats = manifest["stats"]
    profile_id = stats.get("profile_id")
    content_hash = stats.get("content_sha256")
    _require_sha256(content_hash, "stats.content_sha256")
    if profile_id != f"sha256:{content_hash}":
        raise ValueError("stats profile ID/hash mismatch")
    if stats.get("kind") != spec.stats_profile_kind:
        raise ValueError(
            f"stats.kind must be {spec.stats_profile_kind!r} for mode {spec.mode.value!r}, "
            f"got {stats.get('kind')!r}"
        )
    horizon = _require_positive_int(stats.get("horizon"), "stats.horizon")
    if horizon != manifest["action_horizon"]:
        raise ValueError(
            f"stats.horizon {horizon} disagrees with action_horizon {manifest['action_horizon']}"
        )


def _validate_policy(manifest: dict[str, Any], *, expected_policy_type: str | None) -> None:
    policy = manifest["policy"]
    policy_type = policy.get("type")
    if policy_type not in SUPPORTED_POLICY_FAMILIES:
        raise ValueError(
            f"unsupported policy family {policy_type!r}; expected one of "
            f"{list(SUPPORTED_POLICY_FAMILIES)}"
        )
    if expected_policy_type is not None and policy_type != expected_policy_type:
        raise ValueError(
            f"manifest policy type mismatch: {policy_type!r} != {expected_policy_type!r}"
        )
    if policy.get("prediction_api") != "predict_action_chunk":
        raise ValueError("policy.prediction_api must be 'predict_action_chunk'")
    if policy.get("full_chunk_postprocess_required") is not True:
        raise ValueError("checkpoint must require full-chunk postprocessing")
    # async client의 horizon 규칙(chunk_size 또는 execution_horizon)을 v2에서도 보존한다.
    chunk_size = _require_positive_int(policy.get("chunk_size"), "policy.chunk_size")
    execution_horizon = _require_positive_int(
        policy.get("execution_horizon"),
        "policy.execution_horizon",
    )
    if execution_horizon > chunk_size:
        raise ValueError(
            f"policy.execution_horizon {execution_horizon} exceeds chunk_size {chunk_size}"
        )
    if manifest["action_horizon"] != chunk_size:
        raise ValueError(
            f"action_horizon {manifest['action_horizon']} must equal the model chunk_size "
            f"{chunk_size}"
        )


def _validate_dataset(manifest: dict[str, Any], spec: ActionRepresentationSpec) -> None:
    dataset = manifest["dataset"]
    validate_dataset_storage_reference(dataset.get("storage_reference"), source="dataset")
    _require_sha256(dataset.get("fingerprint"), "dataset.fingerprint")
    repo_id = dataset.get("repo_id")
    revision = dataset.get("revision")
    if repo_id is not None and not isinstance(repo_id, str):
        raise ValueError("dataset.repo_id must be a string or null")
    if revision is not None and not isinstance(revision, str):
        raise ValueError("dataset.revision must be a string or null")
    expected_space = "absolute_eef" if spec.is_eef else "absolute_joint"
    if dataset.get("space") != expected_space:
        raise ValueError(
            f"dataset.space must be {expected_space!r} for mode {spec.mode.value!r}, "
            f"got {dataset.get('space')!r}"
        )


def _validate_kinematics(manifest: dict[str, Any], spec: ActionRepresentationSpec) -> None:
    kinematics = manifest.get("kinematics")
    if not spec.is_eef:
        return
    kinematics = _require_dict(kinematics, "kinematics")
    if not isinstance(kinematics.get("version"), str) or not kinematics["version"]:
        raise ValueError("kinematics.version is required for EEF modes")
    _require_sha256(kinematics.get("urdf_sha256"), "kinematics.urdf_sha256")
    _require_sha256(kinematics.get("robot_yaml_sha256"), "kinematics.robot_yaml_sha256")


def _validate_runtime(manifest: dict[str, Any], *, verify_runtime_source: bool) -> None:
    runtime = manifest["runtime"]
    if not isinstance(runtime.get("lerobot_version"), str) or not runtime["lerobot_version"]:
        raise ValueError("runtime.lerobot_version is required")
    if _GIT_SHA_PATTERN.fullmatch(str(runtime.get("lerobot_commit"))) is None:
        raise ValueError("runtime.lerobot_commit must be a full Git SHA")
    if _GIT_SHA_PATTERN.fullmatch(str(runtime.get("project_commit"))) is None:
        raise ValueError("runtime.project_commit must be a full Git SHA")
    source_hash = _require_sha256(
        runtime.get("so101_contract_source_sha256"),
        "runtime.so101_contract_source_sha256",
    )
    _require_sha256(runtime.get("processor_source_sha256"), "runtime.processor_source_sha256")
    clients = runtime.get("compatible_clients")
    if not isinstance(clients, list) or not clients:
        raise ValueError("runtime.compatible_clients must be a non-empty list")
    if verify_runtime_source and source_hash != so101_contract_source_sha256():
        raise ValueError("manifest/current so101_contract source hash mismatch")


def _validate_legacy(manifest: dict[str, Any], spec: ActionRepresentationSpec) -> None:
    legacy = _require_dict(manifest.get("legacy"), "legacy")
    allowed = legacy.get("allowed")
    if not isinstance(allowed, bool):
        raise ValueError("legacy.allowed must be a boolean")
    if not allowed:
        if legacy.get("flag") not in (None, ""):
            raise ValueError("legacy.flag must be null when legacy opt-in was not used")
        return
    if legacy.get("flag") != LEGACY_JOINT_ABSOLUTE_OPT_IN:
        raise ValueError(
            f"legacy opt-in must record {LEGACY_JOINT_ABSOLUTE_OPT_IN!r}, "
            f"got {legacy.get('flag')!r}"
        )
    if spec.mode is not ActionRepresentationMode.JOINT_ABSOLUTE:
        raise ValueError(
            "legacy opt-in only covers joint_absolute checkpoints, "
            f"got {spec.mode.value!r}"
        )


# --- CLI assertion ------------------------------------------------------------


def assert_manifest_matches_cli(
    manifest: dict[str, Any],
    *,
    mode: str | ActionRepresentationMode | None = None,
    pose_format: str | PoseFormat | None = None,
    policy_type: str | None = None,
) -> ActionRepresentationSpec:
    """추론 CLI 인자를 override가 아닌 **assertion**으로 검증한다.

    CLI와 checkpoint manifest가 다르면 시작을 거부한다.
    """
    spec = validate_action_representation_manifest(manifest, expected_policy_type=policy_type)
    if mode is not None and coerce_mode(mode) is not spec.mode:
        raise ValueError(
            f"CLI action representation mode {coerce_mode(mode).value!r} does not match the "
            f"checkpoint manifest {spec.mode.value!r}; a checkpoint is fixed to one representation"
        )
    if pose_format is not None and coerce_pose_format(pose_format) is not spec.pose_format:
        raise ValueError(
            f"CLI pose_format {coerce_pose_format(pose_format).value!r} does not match the "
            f"checkpoint manifest {spec.pose_format.value!r}"
        )
    return spec


# --- I/O ----------------------------------------------------------------------


def write_action_representation_manifest(
    checkpoint_dir: str | Path,
    manifest: dict[str, Any],
) -> Path:
    """Checkpoint 폴더에 manifest를 atomic하게 저장."""
    validate_action_representation_manifest(manifest)
    root = Path(checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    output = root / ACTION_REPRESENTATION_MANIFEST
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return output


def read_action_representation_manifest(checkpoint_dir: str | Path) -> dict[str, Any] | None:
    """Local checkpoint에서 manifest를 읽는다. 없으면 ``None``."""
    path = Path(checkpoint_dir) / ACTION_REPRESENTATION_MANIFEST
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid action representation manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"action representation manifest root must be an object: {path}")
    return value


def require_action_representation_manifest(
    checkpoint_dir: str | Path,
    *,
    allow_legacy_joint_absolute: bool = False,
) -> dict[str, Any] | None:
    """신규 checkpoint는 manifest를 MUST 포함한다. 없으면 추정하지 않고 실패한다.

    ``allow_legacy_joint_absolute=True``(CLI
    ``--allow-legacy-joint-absolute-checkpoint``)에서만 ``None``을 돌려주며, 호출자는
    이 사실을 새 manifest ``legacy`` 절에 기록해야 한다.
    """
    manifest = read_action_representation_manifest(checkpoint_dir)
    if manifest is not None:
        return manifest
    if allow_legacy_joint_absolute:
        return None
    raise FileNotFoundError(
        f"checkpoint is missing {ACTION_REPRESENTATION_MANIFEST}: {checkpoint_dir}. "
        "Action representation is never inferred; run the migration tool or pass "
        f"{LEGACY_JOINT_ABSOLUTE_OPT_IN}."
    )


def legacy_opt_in_record(reason: str, *, source: str | None = None) -> dict[str, Any]:
    """Legacy 허용 사실을 새 manifest에 남기기 위한 ``legacy`` 절."""
    return {
        "allowed": True,
        "flag": LEGACY_JOINT_ABSOLUTE_OPT_IN,
        "reason": reason,
        "source": source,
    }
