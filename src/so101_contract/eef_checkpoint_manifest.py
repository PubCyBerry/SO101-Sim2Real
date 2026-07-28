"""EEF-relative LeRobot checkpoint의 self-contained manifest.

학습 시 serialized processor에 들어 있는 resolved dataset/stats 계약을
``action_representation.json``으로 materialize한다. 추론 시에는 이 파일과
processor config를 교차 검증해 잘못된 frame, stats, policy runtime을 즉시 거부한다.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

from .action_manifest import so101_contract_source_sha256
from .eef_action_contract import ACTION_REPRESENTATION_CONTRACT_VERSION
from .eef_relative_action import EEF_RELATIVE_ACTION_VERSION
from .lerobot_eef_processor import SE3RelativeActionsProcessorStep

ACTION_REPRESENTATION_MANIFEST = "action_representation.json"
EEF_CHECKPOINT_MANIFEST_VERSION = "so101_eef_checkpoint_manifest_v1"
PINNED_LEROBOT_VERSION = "0.6.0"
PINNED_LEROBOT_COMMIT = "30da8e687a6dfc617fcd94afc367ac7071c376ce"
SUPPORTED_POLICY_TYPES = frozenset({"act", "smolvla", "groot"})
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _source_tree_sha256() -> str:
    """Checkpoint runtime에 실제 포함된 so101_contract Python source hash."""
    # schema v2 manifest와 같은 구현을 공유한다(단일 소스).
    return so101_contract_source_sha256()


def _project_git_identity() -> tuple[str, bool | None]:
    configured = os.environ.get("SO101_PROJECT_GIT_COMMIT", "").strip().lower()
    if configured:
        if _GIT_SHA_PATTERN.fullmatch(configured) is None:
            raise ValueError(
                "SO101_PROJECT_GIT_COMMIT must be a full lowercase 40-character Git SHA"
            )
        return configured, None

    repo_root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().lower()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(repo_root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "SO101 project commit is unavailable; set SO101_PROJECT_GIT_COMMIT at build/runtime"
        ) from exc
    if _GIT_SHA_PATTERN.fullmatch(commit) is None:
        raise RuntimeError(f"invalid project Git commit resolved from {repo_root}: {commit!r}")
    return commit, dirty


def _lerobot_identity() -> tuple[str, str]:
    version = os.environ.get("LEROBOT_RUNTIME_VERSION", "").strip()
    if not version:
        version = importlib.metadata.version("lerobot")
    if version != PINNED_LEROBOT_VERSION:
        raise RuntimeError(
            f"EEF-relative manifest requires lerobot=={PINNED_LEROBOT_VERSION}, got {version}"
        )
    commit = os.environ.get("LEROBOT_GIT_COMMIT", PINNED_LEROBOT_COMMIT).strip().lower()
    if commit != PINNED_LEROBOT_COMMIT:
        raise RuntimeError(
            "LeRobot source commit mismatch: "
            f"{commit!r} != pinned v0.6.0 commit {PINNED_LEROBOT_COMMIT}"
        )
    return version, commit


def _relative_step(preprocessor: Any) -> SE3RelativeActionsProcessorStep | None:
    matches = [
        step
        for step in getattr(preprocessor, "steps", [])
        if isinstance(step, SE3RelativeActionsProcessorStep) and step.enabled
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"expected exactly one EEF-relative processor step, got {len(matches)}")
    return matches[0]


def _policy_type(policy_cfg: Any) -> str:
    value = getattr(policy_cfg, "type", None)
    if not isinstance(value, str) or not value:
        value = policy_cfg.__class__.__name__.removesuffix("Config").lower()
    value = value.lower()
    if value not in SUPPORTED_POLICY_TYPES:
        raise NotImplementedError(
            f"EEF-relative checkpoint does not support policy type {value!r}"
        )
    return value


def _policy_manifest(policy_cfg: Any) -> dict[str, Any]:
    policy_type = _policy_type(policy_cfg)
    chunk_size = getattr(policy_cfg, "chunk_size", None)
    n_action_steps = getattr(policy_cfg, "n_action_steps", None)
    base_model_path = getattr(policy_cfg, "base_model_path", None)
    if base_model_path is None:
        base_model_path = getattr(policy_cfg, "pretrained_path", None)
    return {
        "type": policy_type,
        "model_family": "GR00T-N1.7" if policy_type == "groot" else policy_type.upper(),
        "base_model_path": str(base_model_path) if base_model_path else None,
        "chunk_size": chunk_size if isinstance(chunk_size, int) else None,
        "execution_horizon": n_action_steps if isinstance(n_action_steps, int) else None,
        "prediction_api": "predict_action_chunk",
        "full_chunk_postprocess_required": True,
    }


def build_eef_action_representation_manifest(
    policy_cfg: Any,
    preprocessor: Any,
) -> dict[str, Any] | None:
    """EEF processor가 있으면 deterministic manifest를 만들고 absolute mode면 None."""
    step = _relative_step(preprocessor)
    if step is None:
        return None
    context = deepcopy(step.manifest_context)
    required_context = ("representation", "resolved_contract", "dataset", "relative_stats")
    missing = [key for key in required_context if not isinstance(context.get(key), dict)]
    if missing:
        raise ValueError(f"EEF processor manifest context is incomplete: {missing}")

    representation = context["representation"]
    contract = context["resolved_contract"]
    relative_stats = context["relative_stats"]
    if representation.get("mode") != "eef_relative":
        raise ValueError("processor manifest representation mode must be 'eef_relative'")
    if contract.get("fingerprint") != step.contract_fingerprint:
        raise ValueError("processor manifest contract fingerprint mismatch")
    profile_id = relative_stats.get("profile_id")
    profile_hash = relative_stats.get("content_sha256")
    if (
        not isinstance(profile_id, str)
        or not profile_id.startswith("sha256:")
        or profile_id.removeprefix("sha256:") != profile_hash
        or _SHA256_PATTERN.fullmatch(str(profile_hash)) is None
    ):
        raise ValueError("processor manifest contains an invalid relative stats profile ID/hash")

    lerobot_version, lerobot_commit = _lerobot_identity()
    project_commit, project_dirty = _project_git_identity()
    payload = {
        "schema_version": EEF_CHECKPOINT_MANIFEST_VERSION,
        "contract_schema_version": ACTION_REPRESENTATION_CONTRACT_VERSION,
        "transform_version": EEF_RELATIVE_ACTION_VERSION,
        "representation": representation,
        "resolved_contract": contract,
        "dataset": context["dataset"],
        "relative_stats": relative_stats,
        "kinematics": {
            "version": contract.get("eef_kinematics_version"),
            "urdf_sha256": contract.get("urdf_sha256"),
            "robot_yaml_sha256": contract.get("robot_yaml_sha256"),
        },
        "policy": _policy_manifest(policy_cfg),
        "runtime": {
            "lerobot_version": lerobot_version,
            "lerobot_commit": lerobot_commit,
            "project_commit": project_commit,
            "project_worktree_dirty": project_dirty,
            "so101_contract_source_sha256": _source_tree_sha256(),
            "compatible_clients": [
                "lerobot_async_full_chunk",
                "lerobot_sync_full_chunk",
                "so101_vla_policy_ros2",
                "so101_eef_robot_client",
            ],
        },
    }
    payload["manifest_sha256"] = _sha256_json(payload)
    validate_eef_action_representation_manifest(
        payload,
        policy_cfg=policy_cfg,
        preprocessor=preprocessor,
    )
    return payload


def validate_eef_action_representation_manifest(
    manifest: dict[str, Any],
    *,
    policy_cfg: Any | None = None,
    preprocessor: Any | None = None,
) -> None:
    """Manifest 자체 hash와 optional runtime policy/processor parity를 검증."""
    if not isinstance(manifest, dict):
        raise TypeError("action representation manifest must be a JSON object")
    if manifest.get("schema_version") != EEF_CHECKPOINT_MANIFEST_VERSION:
        raise ValueError("action representation manifest schema_version mismatch")
    if manifest.get("contract_schema_version") != ACTION_REPRESENTATION_CONTRACT_VERSION:
        raise ValueError("action representation contract schema_version mismatch")
    if manifest.get("transform_version") != EEF_RELATIVE_ACTION_VERSION:
        raise ValueError("action representation transform_version mismatch")

    expected_hash = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if (
        not isinstance(expected_hash, str)
        or _SHA256_PATTERN.fullmatch(expected_hash) is None
        or _sha256_json(unsigned) != expected_hash
    ):
        raise ValueError("action representation manifest content hash mismatch")

    representation = manifest.get("representation")
    contract = manifest.get("resolved_contract")
    stats = manifest.get("relative_stats")
    runtime = manifest.get("runtime")
    if not all(isinstance(value, dict) for value in (representation, contract, stats, runtime)):
        raise ValueError("action representation manifest has incomplete structured sections")
    if representation.get("mode") != "eef_relative":
        raise ValueError("manifest representation mode must be 'eef_relative'")
    if contract.get("config") != representation:
        raise ValueError("manifest representation/resolved contract config mismatch")
    contract_fingerprint = contract.get("fingerprint")
    if _SHA256_PATTERN.fullmatch(str(contract_fingerprint)) is None:
        raise ValueError("manifest dataset contract fingerprint is invalid")
    if stats.get("profile_id") != f"sha256:{stats.get('content_sha256')}":
        raise ValueError("manifest relative stats profile ID/hash mismatch")
    if runtime.get("lerobot_version") != PINNED_LEROBOT_VERSION:
        raise ValueError("manifest LeRobot version is not the pinned v0.6.0 runtime")
    if runtime.get("lerobot_commit") != PINNED_LEROBOT_COMMIT:
        raise ValueError("manifest LeRobot commit is not the pinned v0.6.0 source commit")
    if _GIT_SHA_PATTERN.fullmatch(str(runtime.get("project_commit"))) is None:
        raise ValueError("manifest project commit is not a full Git SHA")
    if _SHA256_PATTERN.fullmatch(str(runtime.get("so101_contract_source_sha256"))) is None:
        raise ValueError("manifest so101_contract source hash is invalid")
    if policy_cfg is not None or preprocessor is not None:
        current_lerobot_version, current_lerobot_commit = _lerobot_identity()
        current_project_commit, _ = _project_git_identity()
        if runtime["lerobot_version"] != current_lerobot_version:
            raise ValueError("manifest/current LeRobot version mismatch")
        if runtime["lerobot_commit"] != current_lerobot_commit:
            raise ValueError("manifest/current LeRobot source commit mismatch")
        if runtime["project_commit"] != current_project_commit:
            raise ValueError("manifest/current project commit mismatch")
        if runtime["so101_contract_source_sha256"] != _source_tree_sha256():
            raise ValueError("manifest/current so101_contract source hash mismatch")

    policy = manifest.get("policy")
    if not isinstance(policy, dict) or policy.get("type") not in SUPPORTED_POLICY_TYPES:
        raise ValueError("manifest policy compatibility is missing or unsupported")
    if policy.get("full_chunk_postprocess_required") is not True:
        raise ValueError("EEF-relative checkpoint must require full-chunk postprocessing")
    if policy_cfg is not None and policy.get("type") != _policy_type(policy_cfg):
        raise ValueError(
            f"manifest policy type mismatch: {policy.get('type')!r} != {_policy_type(policy_cfg)!r}"
        )
    if preprocessor is not None:
        step = _relative_step(preprocessor)
        if step is None:
            raise ValueError("EEF manifest was loaded without an EEF-relative processor step")
        if step.contract_fingerprint != contract_fingerprint:
            raise ValueError("manifest/processor contract fingerprint mismatch")
        if step.manifest_context != {
            "representation": representation,
            "resolved_contract": contract,
            "dataset": manifest.get("dataset"),
            "relative_stats": stats,
        }:
            raise ValueError("manifest and serialized processor context differ")


def write_eef_action_representation_manifest(
    pretrained_dir: str | Path,
    policy_cfg: Any,
    preprocessor: Any,
) -> Path | None:
    """Checkpoint pretrained_model 폴더에 manifest를 atomic하게 저장."""
    manifest = build_eef_action_representation_manifest(policy_cfg, preprocessor)
    if manifest is None:
        return None
    root = Path(pretrained_dir)
    root.mkdir(parents=True, exist_ok=True)
    output = root / ACTION_REPRESENTATION_MANIFEST
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return output


def load_eef_action_representation_manifest(
    pretrained_path: str | Path,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Local checkpoint 또는 Hub repo에서 manifest 하나를 로드."""
    path = Path(pretrained_path)
    if path.exists():
        manifest_path = path / ACTION_REPRESENTATION_MANIFEST
    else:
        from huggingface_hub import hf_hub_download

        manifest_path = Path(
            hf_hub_download(
                repo_id=str(pretrained_path),
                filename=ACTION_REPRESENTATION_MANIFEST,
                revision=revision,
                local_files_only=local_files_only,
            )
        )
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"EEF-relative checkpoint is missing {ACTION_REPRESENTATION_MANIFEST}: "
            f"{pretrained_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid action representation manifest {manifest_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"action representation manifest root must be an object: {manifest_path}")
    return value


def validate_eef_checkpoint_artifact(
    pretrained_path: str | Path,
    policy_cfg: Any,
    preprocessor: Any,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Inference startup에서 checkpoint manifest와 loaded processor를 교차 검증."""
    manifest = load_eef_action_representation_manifest(
        pretrained_path,
        revision=revision,
        local_files_only=local_files_only,
    )
    validate_eef_action_representation_manifest(
        manifest,
        policy_cfg=policy_cfg,
        preprocessor=preprocessor,
    )
    return manifest


def push_eef_action_representation_manifest(
    repo_id: str,
    policy_cfg: Any,
    preprocessor: Any,
) -> bool:
    """최종 model repo root에 manifest를 업로드. absolute mode면 아무 작업도 하지 않음."""
    manifest = build_eef_action_representation_manifest(policy_cfg, preprocessor)
    if manifest is None:
        return False
    from huggingface_hub import HfApi

    with tempfile.TemporaryDirectory(prefix="so101-eef-manifest-") as directory:
        path = Path(directory) / ACTION_REPRESENTATION_MANIFEST
        path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        HfApi().upload_file(
            path_or_fileobj=str(path),
            path_in_repo=ACTION_REPRESENTATION_MANIFEST,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add EEF-relative action representation manifest",
        )
    return True
