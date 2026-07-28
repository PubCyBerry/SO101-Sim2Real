"""LeRobot 비의존 EEF checkpoint↔platform kinematics 배포 계약 검증."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .eef_action_contract import ACTION_REPRESENTATION_CONTRACT_VERSION
from .eef_relative_action import EEF_RELATIVE_ACTION_VERSION

ACTION_REPRESENTATION_MANIFEST = "action_representation.json"
EEF_CHECKPOINT_MANIFEST_VERSION = "so101_eef_checkpoint_manifest_v1"
PINNED_LEROBOT_VERSION = "0.6.0"
PINNED_LEROBOT_COMMIT = "30da8e687a6dfc617fcd94afc367ac7071c376ce"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"kinematics source file not found: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def so101_contract_source_sha256() -> str:
    """현재 client가 실행하는 so101_contract Python source tree hash."""
    package_root = Path(__file__).resolve().parent
    paths = sorted(package_root.rglob("*.py"))
    if not paths:
        raise RuntimeError(f"so101_contract source files not found under {package_root}")
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(package_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        raw = path.read_bytes()
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def _canonical_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def load_deployment_manifest(
    pretrained_path: str | Path,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Local checkpoint/HF repo에서 작은 JSON manifest만 가져온다."""
    candidate = Path(pretrained_path)
    if candidate.exists():
        path = candidate / ACTION_REPRESENTATION_MANIFEST
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "Hub checkpoint deployment validation requires huggingface_hub"
            ) from exc
        path = Path(
            hf_hub_download(
                repo_id=str(pretrained_path),
                filename=ACTION_REPRESENTATION_MANIFEST,
                revision=revision,
                local_files_only=local_files_only,
            )
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"EEF checkpoint is missing {ACTION_REPRESENTATION_MANIFEST}: {pretrained_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid EEF checkpoint manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"EEF checkpoint manifest root must be an object: {path}")
    return value


def validate_deployment_manifest(
    manifest: dict[str, Any],
    *,
    urdf_path: str | Path,
    robot_yaml_path: str | Path,
    policy_type: str | None = None,
) -> None:
    """Client startup에서 frame/kinematics/policy compatibility를 fail-fast."""
    if manifest.get("schema_version") != EEF_CHECKPOINT_MANIFEST_VERSION:
        raise ValueError("deployment manifest schema version mismatch")
    if manifest.get("contract_schema_version") != ACTION_REPRESENTATION_CONTRACT_VERSION:
        raise ValueError("deployment action contract version mismatch")
    if manifest.get("transform_version") != EEF_RELATIVE_ACTION_VERSION:
        raise ValueError("deployment EEF transform version mismatch")
    expected = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if (
        not isinstance(expected, str)
        or _SHA256_PATTERN.fullmatch(expected) is None
        or _canonical_hash(unsigned) != expected
    ):
        raise ValueError("deployment manifest content hash mismatch")

    representation = manifest.get("representation")
    contract = manifest.get("resolved_contract")
    kinematics = manifest.get("kinematics")
    runtime = manifest.get("runtime")
    policy = manifest.get("policy")
    if not all(
        isinstance(value, dict)
        for value in (representation, contract, kinematics, runtime, policy)
    ):
        raise ValueError("deployment manifest has incomplete structured sections")
    if (
        representation.get("mode") != "eef_relative"
        or representation.get("base_frame") != "base_link"
        or representation.get("eef_frame") != "tcp_grasp"
        or representation.get("pose_format") != "xyz_rot6d_rows"
    ):
        raise ValueError("deployment manifest frame/pose representation mismatch")
    if contract.get("config") != representation:
        raise ValueError("deployment representation/resolved contract mismatch")
    if runtime.get("lerobot_version") != PINNED_LEROBOT_VERSION:
        raise ValueError("deployment checkpoint was not produced by LeRobot v0.6.0")
    if runtime.get("lerobot_commit") != PINNED_LEROBOT_COMMIT:
        raise ValueError("deployment checkpoint LeRobot commit mismatch")
    expected_source_hash = runtime.get("so101_contract_source_sha256")
    actual_source_hash = so101_contract_source_sha256()
    if expected_source_hash != actual_source_hash:
        raise ValueError(
            "checkpoint/client so101_contract source hash mismatch: "
            f"{expected_source_hash} != {actual_source_hash}"
        )
    if policy.get("full_chunk_postprocess_required") is not True:
        raise ValueError("deployment checkpoint does not require full-chunk postprocessing")
    if policy_type is not None and policy.get("type") != policy_type.lower():
        raise ValueError(
            f"deployment policy type mismatch: {policy.get('type')!r} != {policy_type!r}"
        )

    actual_urdf = sha256_file(urdf_path)
    actual_yaml = sha256_file(robot_yaml_path)
    if kinematics.get("urdf_sha256") != actual_urdf:
        raise ValueError(
            "checkpoint/client URDF hash mismatch: "
            f"{kinematics.get('urdf_sha256')} != {actual_urdf}"
        )
    if kinematics.get("robot_yaml_sha256") != actual_yaml:
        raise ValueError(
            "checkpoint/client robot YAML hash mismatch: "
            f"{kinematics.get('robot_yaml_sha256')} != {actual_yaml}"
        )
    if (
        contract.get("urdf_sha256") != actual_urdf
        or contract.get("robot_yaml_sha256") != actual_yaml
    ):
        raise ValueError("resolved contract and platform kinematics hashes differ")


def validate_checkpoint_for_platform(
    pretrained_path: str | Path,
    *,
    urdf_path: str | Path,
    robot_yaml_path: str | Path,
    policy_type: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
) -> dict[str, Any]:
    manifest = load_deployment_manifest(
        pretrained_path,
        revision=revision,
        local_files_only=local_files_only,
    )
    validate_deployment_manifest(
        manifest,
        urdf_path=urdf_path,
        robot_yaml_path=robot_yaml_path,
        policy_type=policy_type,
    )
    return manifest
