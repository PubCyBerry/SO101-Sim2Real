"""Phase 16 — legacy checkpoint → schema v2 migration core.

**원본은 절대 수정하지 않는다.** 새 디렉터리에 복사본을 만들고 거기에만 v2 processor
pair와 ``action_representation.json``을 추가한다.

지원하는 legacy 형태는 **두 가지뿐**이며, 각각 명시적 dispatch다.

1. ``manifest_absent`` — manifest가 없는 checkpoint. 정확한 flag
   ``--allow-legacy-joint-absolute-checkpoint``가 있을 때만 ``joint_absolute``로 취급한다.
   차원·config 이름·feature 이름으로 representation을 **추정하지 않는다**.
2. ``v1_eef_relative`` — v1 EEF-relative(`xyz_rot6d_rows`) manifest를 가진 checkpoint.
   v1 module로 읽어 v2 spec으로 변환하되, 완전한 v2 processor/manifest를 만들 만큼의
   dataset/stats 입력이 명시적으로 주어져야 한다. 없으면 실패한다.

그 밖의 형태는 지원하지 않는다고 명확히 실패한다.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

from .action_checkpoint_contract import resolve_checkpoint_contract
from .action_dataset_contract import resolve_action_representation_contract
from .action_manifest import (
    ACTION_REPRESENTATION_MANIFEST,
    LEGACY_JOINT_ABSOLUTE_OPT_IN,
    build_action_representation_manifest,
    build_feature_contract,
    manifest_schema_version,
    write_action_representation_manifest,
)
from .action_representation import (
    SUPPORTED_POLICY_FAMILIES,
    ActionRepresentationMode,
    ActionRepresentationSpec,
    PoseFormat,
    coerce_mode,
    coerce_pose_format,
)
from .action_representation_stats import (
    ActionStatsSampling,
    load_lerobot_v3_episodes,
    read_action_stats_artifact,
    select_stats_profile,
    serialize_stats_for_processor,
    source_columns_sha256,
)

ACTION_MIGRATION_VERSION = "so101_action_representation_migration_v1"

#: 이 도구가 인식하는 legacy source schema 상태.
SOURCE_SCHEMA_STATES = ("manifest_absent", "v1_eef_relative", "v2")


@dataclass(frozen=True)
class MigrationPlan:
    """Migration 전에 확정되는 입력. 모호하면 여기서 실패한다."""

    source: Path
    output: Path
    source_schema_state: str
    spec: ActionRepresentationSpec
    dataset_root: Path
    horizon: int
    legacy_opt_in: bool
    policy_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": str(self.source),
            "output": str(self.output),
            "source_schema_state": self.source_schema_state,
            "mode": self.spec.mode.value,
            "pose_format": self.spec.pose_format.value,
            "dataset_root": str(self.dataset_root),
            "horizon": self.horizon,
            "legacy_opt_in": self.legacy_opt_in,
            "policy_type": self.policy_type,
        }


@dataclass(frozen=True)
class MigrationResult:
    plan: MigrationPlan
    output: Path
    manifest: dict[str, Any]
    source_identity: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "output": str(self.output),
            "manifest_sha256": self.manifest["manifest_sha256"],
            "mode": self.manifest["mode"],
            "pose_format": self.manifest["pose_format"],
            "action_dim": self.manifest["action_dim"],
            "source_identity": self.source_identity,
        }


# --- source 식별 ---------------------------------------------------------------


def checkpoint_directory_sha256(root: Path) -> dict[str, Any]:
    """Checkpoint 파일 목록과 내용의 결정적 identity."""
    files: list[tuple[str, int, str]] = []
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        raw = path.read_bytes()
        file_hash = hashlib.sha256(raw).hexdigest()
        files.append((relative, len(raw), file_hash))
        digest.update(relative.encode("utf-8"))
        digest.update(file_hash.encode("utf-8"))
    return {
        "file_count": len(files),
        "tree_sha256": digest.hexdigest(),
        "files": [
            {"path": name, "bytes": size, "sha256": file_hash} for name, size, file_hash in files
        ],
    }


def detect_source_schema_state(source: Path) -> str:
    """``manifest_absent`` / ``v1_eef_relative`` / ``v2`` 중 하나. 그 외는 오류."""
    manifest_path = source / ACTION_REPRESENTATION_MANIFEST
    if not manifest_path.is_file():
        return "manifest_absent"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"source manifest is not valid JSON: {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise TypeError(f"source manifest root must be an object: {manifest_path}")
    version = manifest_schema_version(manifest)
    if version == 2:
        return "v2"
    representation = (manifest.get("representation") or {}).get("mode")
    pose_format = (manifest.get("representation") or {}).get("pose_format")
    if representation == "eef_relative" and pose_format in (None, "xyz_rot6d_rows"):
        return "v1_eef_relative"
    raise NotImplementedError(
        "unsupported legacy manifest: this tool migrates only manifest-less checkpoints "
        f"and v1 eef_relative+xyz_rot6d_rows checkpoints, got mode={representation!r} "
        f"pose_format={pose_format!r}"
    )


def _policy_type_of_checkpoint(source: Path) -> str:
    config_path = source / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"checkpoint has no config.json: {source}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    policy_type = config.get("type")
    if not isinstance(policy_type, str) or policy_type.lower() not in SUPPORTED_POLICY_FAMILIES:
        raise NotImplementedError(
            f"migration supports {sorted(SUPPORTED_POLICY_FAMILIES)}, got {policy_type!r}"
        )
    return policy_type.lower()


# --- plan ----------------------------------------------------------------------


def plan_migration(
    source: str | Path,
    output: str | Path,
    *,
    dataset_root: str | Path,
    horizon: int,
    mode: str | None = None,
    pose_format: str | None = None,
    allow_legacy_joint_absolute: bool = False,
) -> MigrationPlan:
    """입력을 검증하고 migration plan을 만든다. 모호하면 여기서 fail-fast."""
    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    if not source_path.is_dir():
        raise FileNotFoundError(f"source checkpoint directory not found: {source_path}")
    if output_path == source_path:
        raise ValueError("in-place migration is refused; choose a distinct output directory")
    if source_path in output_path.parents:
        raise ValueError(
            f"output must not live inside the source checkpoint: {output_path} ⊂ {source_path}"
        )
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")

    state = detect_source_schema_state(source_path)
    if state == "v2":
        raise ValueError(
            f"source already carries a schema v2 manifest; nothing to migrate: {source_path}"
        )
    policy_type = _policy_type_of_checkpoint(source_path)

    if state == "manifest_absent":
        if not allow_legacy_joint_absolute:
            raise PermissionError(
                f"checkpoint has no {ACTION_REPRESENTATION_MANIFEST} and its action "
                "representation is never inferred; pass "
                f"{LEGACY_JOINT_ABSOLUTE_OPT_IN} to declare it joint_absolute"
            )
        resolved_mode = coerce_mode(mode) if mode else ActionRepresentationMode.JOINT_ABSOLUTE
        if resolved_mode is not ActionRepresentationMode.JOINT_ABSOLUTE:
            raise ValueError(
                f"{LEGACY_JOINT_ABSOLUTE_OPT_IN} only declares joint_absolute, got "
                f"{resolved_mode.value!r}"
            )
        if pose_format:
            raise ValueError("joint_absolute migration must not declare a pose_format")
        spec = ActionRepresentationSpec(mode=ActionRepresentationMode.JOINT_ABSOLUTE)
        legacy_opt_in = True
    else:
        if allow_legacy_joint_absolute:
            raise ValueError(
                f"{LEGACY_JOINT_ABSOLUTE_OPT_IN} is only for manifest-less checkpoints; the "
                "source declares a v1 eef_relative manifest"
            )
        resolved_mode = coerce_mode(mode) if mode else ActionRepresentationMode.EEF_RELATIVE
        resolved_format = (
            coerce_pose_format(pose_format) if pose_format else PoseFormat.XYZ_ROT6D_ROWS
        )
        if resolved_mode is not ActionRepresentationMode.EEF_RELATIVE:
            raise ValueError(
                f"v1 source declares eef_relative; explicit mode {resolved_mode.value!r} conflicts"
            )
        if resolved_format is not PoseFormat.XYZ_ROT6D_ROWS:
            raise ValueError(
                "v1 EEF-relative checkpoints are xyz_rot6d_rows only; explicit pose_format "
                f"{resolved_format.value!r} conflicts"
            )
        spec = ActionRepresentationSpec(mode=resolved_mode, pose_format=resolved_format)
        legacy_opt_in = False

    dataset = Path(dataset_root).resolve()
    if not (dataset / "meta" / "info.json").is_file():
        raise FileNotFoundError(
            f"migration needs the training dataset to rebuild the v2 contract/stats: {dataset}"
        )
    return MigrationPlan(
        source=source_path,
        output=output_path,
        source_schema_state=state,
        spec=spec,
        dataset_root=dataset,
        horizon=int(horizon),
        legacy_opt_in=legacy_opt_in,
        policy_type=policy_type,
    )


# --- migration -----------------------------------------------------------------


def _copy_checkpoint(source: Path, staging: Path) -> None:
    shutil.copytree(source, staging, dirs_exist_ok=False)


def _build_processor_pair(plan: MigrationPlan, staging: Path) -> tuple[Any, dict[str, Any]]:
    """migrated checkpoint에 실제 로드 가능한 v2 processor pair를 기록한다."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import make_pre_post_processors

    from .lerobot_v2_integration import (
        action_representation_encode_step,
        build_manifest_from_processor,
    )

    config = PreTrainedConfig.from_pretrained(staging)
    config.device = "cpu"
    representation = getattr(config, "action_representation", None)
    if representation is None:
        raise RuntimeError(
            "installed LeRobot has no action_representation config; the v2 patch is missing"
        )
    representation.mode = plan.spec.mode.value
    representation.pose_format = (
        None
        if plan.spec.pose_format is PoseFormat.NOT_APPLICABLE
        else plan.spec.pose_format.value
    )
    representation.state_group = ""
    representation.action_group = ""
    representation.state_pose_group = None
    representation.action_pose_group = None
    representation.base_frame = None
    representation.eef_frame = None
    representation.stats_file = plan.spec.stats_file
    config.action_representation = representation

    dataset_stats, dataset_meta = _dataset_inputs(plan)
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        dataset_stats=dataset_stats,
        dataset_meta=dataset_meta,
    )
    if action_representation_encode_step(preprocessor) is None:
        raise RuntimeError("migration failed to attach a schema v2 processor pair")

    preprocessor.save_pretrained(staging)
    postprocessor.save_pretrained(staging)
    config.save_pretrained(staging)
    manifest = build_manifest_from_processor(config, preprocessor)
    return preprocessor, manifest


def _dataset_inputs(plan: MigrationPlan) -> tuple[dict[str, Any], Any]:
    """Dataset 계약·stats profile을 확인하고 factory 입력으로 만든다."""
    from types import SimpleNamespace

    import numpy as np

    contract = resolve_action_representation_contract(plan.dataset_root, plan.spec)
    sampling = ActionStatsSampling(action_delta_indices=tuple(range(plan.horizon)))
    artifact = read_action_stats_artifact(plan.dataset_root, output_file=plan.spec.stats_file)
    try:
        select_stats_profile(
            artifact,
            contract.transform,
            sampling,
            dataset_fingerprint=contract.fingerprint,
        )
    except KeyError as exc:
        raise KeyError(
            "migration requires a matching schema v2 stats profile; generate it with "
            "scripts/data/generate_action_representation_stats.py "
            f"--dataset-root {plan.dataset_root} --horizon {plan.horizon} "
            f"--mode {plan.spec.mode.value}"
        ) from exc

    episodes = load_lerobot_v3_episodes(
        plan.dataset_root,
        state_key=contract.state_key,
        action_key=contract.action_key,
        state_dim=contract.state_dim,
        action_dim=contract.action_dim,
    )
    states = np.concatenate([episode.states for episode in episodes])
    actions = np.concatenate([episode.actions for episode in episodes])

    def numeric(values: np.ndarray) -> dict[str, list[float]]:
        values = np.asarray(values, dtype=np.float64)
        return {
            "mean": values.mean(0).tolist(),
            "std": (values.std(0) + 1e-6).tolist(),
            "min": values.min(0).tolist(),
            "max": values.max(0).tolist(),
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q10": np.quantile(values, 0.10, axis=0).tolist(),
            "q50": np.quantile(values, 0.50, axis=0).tolist(),
            "q90": np.quantile(values, 0.90, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist(),
        }

    dataset_stats = {"observation.state": numeric(states), "action": numeric(actions)}
    dataset_meta = SimpleNamespace(
        root=plan.dataset_root,
        repo_id=f"local/{plan.dataset_root.name}",
        revision=None,
        features={
            "observation.state": {
                "dtype": "float32",
                "shape": (contract.state_dim,),
                "names": list(contract.state_names),
            },
            "action": {
                "dtype": "float32",
                "shape": (contract.action_dim,),
                "names": list(contract.action_names),
            },
        },
        stats=dataset_stats,
    )
    return dataset_stats, dataset_meta


def migrate_checkpoint(plan: MigrationPlan, *, overwrite: bool = False) -> MigrationResult:
    """Plan대로 새 디렉터리에 v2 checkpoint를 만든다(원본 불변, atomic publish)."""
    if plan.output.exists():
        if not overwrite:
            raise FileExistsError(
                f"output already exists: {plan.output}; pass --overwrite to replace it"
            )
        if any(plan.output.iterdir()) and not (
            plan.output / ACTION_REPRESENTATION_MANIFEST
        ).is_file():
            raise FileExistsError(
                f"refusing to overwrite a non-empty directory that is not a migrated checkpoint: "
                f"{plan.output}"
            )

    source_identity = checkpoint_directory_sha256(plan.source)
    plan.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{plan.output.name}.migrating-", dir=plan.output.parent)
    )
    try:
        checkpoint = staging / "checkpoint"
        _copy_checkpoint(plan.source, checkpoint)
        _, manifest = _build_processor_pair(plan, checkpoint)

        manifest = dict(manifest)
        manifest.pop("manifest_sha256", None)
        manifest["legacy"] = {
            "allowed": plan.legacy_opt_in,
            "flag": LEGACY_JOINT_ABSOLUTE_OPT_IN if plan.legacy_opt_in else None,
            "reason": (
                "manifest-less checkpoint declared joint_absolute by explicit opt-in"
                if plan.legacy_opt_in
                else None
            ),
            "source": str(plan.source),
        }
        manifest["migration"] = {
            "version": ACTION_MIGRATION_VERSION,
            "source_schema_state": plan.source_schema_state,
            "source_path": str(plan.source),
            "source_tree_sha256": source_identity["tree_sha256"],
            "source_file_count": source_identity["file_count"],
            "dataset_root": str(plan.dataset_root),
            "horizon": plan.horizon,
            "opt_in_flag": LEGACY_JOINT_ABSOLUTE_OPT_IN if plan.legacy_opt_in else None,
        }
        from .action_manifest import canonical_manifest_sha256

        manifest["manifest_sha256"] = canonical_manifest_sha256(manifest)
        write_action_representation_manifest(checkpoint, manifest)

        if plan.output.exists():
            shutil.rmtree(plan.output)
        checkpoint.replace(plan.output)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    # 원본이 그대로인지 다시 확인한다.
    if checkpoint_directory_sha256(plan.source)["tree_sha256"] != source_identity["tree_sha256"]:
        raise RuntimeError("source checkpoint changed during migration; aborting")

    contract = resolve_checkpoint_contract(plan.output, expected_policy_type=plan.policy_type)
    if contract.spec != plan.spec:
        raise RuntimeError(
            f"migrated manifest representation mismatch: {contract.spec.mode.value} != "
            f"{plan.spec.mode.value}"
        )
    return MigrationResult(
        plan=plan,
        output=plan.output,
        manifest=contract.manifest,
        source_identity={
            "tree_sha256": source_identity["tree_sha256"],
            "file_count": source_identity["file_count"],
        },
    )


def verify_source_unchanged(source: Path, expected: dict[str, Any]) -> bool:
    return checkpoint_directory_sha256(source)["tree_sha256"] == expected["tree_sha256"]


__all__ = [
    "ACTION_MIGRATION_VERSION",
    "SOURCE_SCHEMA_STATES",
    "MigrationPlan",
    "MigrationResult",
    "checkpoint_directory_sha256",
    "detect_source_schema_state",
    "migrate_checkpoint",
    "plan_migration",
    "verify_source_unchanged",
]
