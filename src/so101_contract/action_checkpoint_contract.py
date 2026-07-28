"""Phase 16 — checkpoint action representation 계약의 단일 loader.

local checkpoint 디렉터리와 Hugging Face model revision/snapshot을 **하나의 API**로
해석하고, schema/hash/policy/runtime을 검증한 뒤 같은 resolved representation을 돌려준다.
server·sim client·real client가 각자 manifest를 파싱하지 않도록 여기로 모은다.

.. code-block:: text

    resolve_checkpoint_contract("/workspace/outputs/.../pretrained_model")
    resolve_checkpoint_contract("user/repo", revision="<sha>", local_files_only=True)

추론 CLI의 representation 인자는 **assertion**이다. :meth:`ResolvedCheckpointContract.assert_cli`
는 mode/pose_format/policy가 checkpoint와 다르면 로봇/sim 명령 이전에 즉시 실패한다.
생략된 인자는 manifest 값을 그대로 받아들인다(override 아님).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .action_manifest import (
    ACTION_REPRESENTATION_MANIFEST,
    LEGACY_JOINT_ABSOLUTE_OPT_IN,
    manifest_schema_version,
    validate_action_representation_manifest,
)
from .action_representation import (
    ActionRepresentationMode,
    ActionRepresentationSpec,
    PoseFormat,
    coerce_mode,
    coerce_pose_format,
)

CHECKPOINT_CONTRACT_VERSION = "so101_checkpoint_contract_v2"


@dataclass(frozen=True)
class ResolvedCheckpointContract:
    """검증된 checkpoint manifest에서 뽑은 배포용 계약."""

    source: str
    local_path: Path | None
    manifest_path: Path
    manifest: dict[str, Any]
    spec: ActionRepresentationSpec
    state_dim: int
    action_dim: int
    action_names: tuple[str, ...]
    action_groups: dict[str, tuple[int, int]]
    transform_indices: tuple[int, ...]
    passthrough_indices: tuple[int, ...]
    policy_type: str
    chunk_size: int
    execution_horizon: int
    kinematics: dict[str, Any] | None
    legacy: dict[str, Any]

    @property
    def mode(self) -> ActionRepresentationMode:
        return self.spec.mode

    @property
    def pose_format(self) -> PoseFormat:
        return self.spec.pose_format

    @property
    def routing(self) -> tuple[str, ...]:
        return self.spec.inference_routing

    @property
    def requires_ik(self) -> bool:
        return "ik" in self.spec.inference_routing

    @property
    def manifest_sha256(self) -> str:
        return str(self.manifest["manifest_sha256"])

    @property
    def stats_profile_id(self) -> str:
        return str(self.manifest["stats"]["profile_id"])

    @property
    def legacy_opt_in(self) -> bool:
        return bool(self.legacy.get("allowed", False))

    def assert_cli(
        self,
        *,
        mode: str | ActionRepresentationMode | None = None,
        pose_format: str | PoseFormat | None = None,
        policy_type: str | None = None,
    ) -> "ResolvedCheckpointContract":
        """CLI/env 인자를 assertion으로 검증한다(override 금지).

        생략(``None``)한 인자는 manifest 값을 그대로 수용한다. 값이 주어졌는데 다르면
        어떤 robot/sim 명령보다 먼저 실패한다.
        """
        if mode is not None:
            requested = coerce_mode(mode)
            if requested is not self.spec.mode:
                raise ValueError(
                    f"action representation mode assertion failed: CLI={requested.value!r} != "
                    f"checkpoint={self.spec.mode.value!r}; a checkpoint is fixed to one "
                    "representation and cannot be overridden"
                )
        if pose_format is not None:
            requested_format = coerce_pose_format(pose_format)
            if requested_format is not self.spec.pose_format:
                raise ValueError(
                    f"pose_format assertion failed: CLI={requested_format.value!r} != "
                    f"checkpoint={self.spec.pose_format.value!r}"
                )
        if policy_type is not None and policy_type.lower() != self.policy_type:
            raise ValueError(
                f"policy family assertion failed: CLI={policy_type!r} != "
                f"checkpoint={self.policy_type!r}"
            )
        return self

    def summary(self) -> str:
        legacy = f" legacy={self.legacy.get('flag')}" if self.legacy_opt_in else ""
        return (
            f"action_representation={self.spec.mode.value} "
            f"pose_format={self.spec.pose_format.value} "
            f"state/action={self.state_dim}/{self.action_dim} "
            f"policy={self.policy_type} horizon={self.chunk_size}/{self.execution_horizon} "
            f"routing={'->'.join(self.routing)} "
            f"stats={self.stats_profile_id} manifest={self.manifest_sha256[:12]}{legacy}"
        )


def _read_manifest_file(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"checkpoint is missing {ACTION_REPRESENTATION_MANIFEST}: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid action representation manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"action representation manifest root must be an object: {path}")
    return value


def _resolve_manifest_path(
    source: str | Path,
    *,
    revision: str | None,
    local_files_only: bool,
    cache_dir: str | Path | None = None,
) -> tuple[Path | None, Path]:
    """``(local checkpoint dir | None, manifest path)``.

    local 디렉터리면 그대로 쓰고, 아니면 Hugging Face repo id로 보고 manifest 파일 하나만
    받아 온다(가중치 다운로드 없음). ``local_files_only=True``면 캐시된 snapshot만 쓴다.
    """
    candidate = Path(source)
    if candidate.is_dir():
        return candidate, candidate / ACTION_REPRESENTATION_MANIFEST
    if candidate.is_file():
        return candidate.parent, candidate
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - 런타임 의존성
        raise ImportError(
            "resolving a Hub checkpoint contract requires huggingface_hub"
        ) from exc
    downloaded = Path(
        hf_hub_download(
            repo_id=str(source),
            filename=ACTION_REPRESENTATION_MANIFEST,
            revision=revision,
            local_files_only=local_files_only,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
    )
    # snapshot 디렉터리를 local_path로 노출해 processor/weight 경로도 같이 쓸 수 있게 한다.
    return downloaded.parent, downloaded


def resolve_checkpoint_contract(
    source: str | Path,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
    cache_dir: str | Path | None = None,
    expected_policy_type: str | None = None,
    verify_runtime_source: bool = False,
    allow_legacy_opt_in: bool = True,
) -> ResolvedCheckpointContract:
    """local 디렉터리 또는 Hub revision에서 schema v2 계약을 resolve한다.

    v1 manifest는 자동 승격하지 않는다. migration(:mod:`so101_contract.action_migration`)
    으로 v2 checkpoint를 만든 뒤 사용해야 한다.
    """
    local_path, manifest_path = _resolve_manifest_path(
        source,
        revision=revision,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
    )
    manifest = _read_manifest_file(manifest_path)
    version = manifest_schema_version(manifest)
    if version != 2:
        raise ValueError(
            f"checkpoint carries a v{version} action representation manifest: {source}. "
            "Automatic promotion is disabled; run "
            "scripts/convert/migrate_action_representation_checkpoint.py first."
        )
    spec = validate_action_representation_manifest(
        manifest,
        expected_policy_type=expected_policy_type,
        verify_runtime_source=verify_runtime_source,
    )
    legacy = dict(manifest.get("legacy") or {})
    if legacy.get("allowed") and not allow_legacy_opt_in:
        raise ValueError(
            f"checkpoint was migrated with {LEGACY_JOINT_ABSOLUTE_OPT_IN}; this deployment "
            "path refuses legacy opt-in checkpoints"
        )

    action_feature = manifest["features"]["action"]
    groups = {name: (int(b[0]), int(b[1])) for name, b in action_feature["groups"].items()}
    if spec.action_group not in groups:
        raise ValueError(
            f"manifest action feature has no transform group {spec.action_group!r}"
        )
    start, end = groups[spec.action_group]
    passthrough: list[int] = []
    for name in spec.passthrough_action_groups:
        if name not in groups:
            raise ValueError(f"manifest action feature has no passthrough group {name!r}")
        low, high = groups[name]
        passthrough.extend(range(low, high))

    policy = manifest["policy"]
    return ResolvedCheckpointContract(
        source=str(source),
        local_path=local_path,
        manifest_path=manifest_path,
        manifest=manifest,
        spec=spec,
        state_dim=int(manifest["state_dim"]),
        action_dim=int(manifest["action_dim"]),
        action_names=tuple(action_feature["names"]),
        action_groups=groups,
        transform_indices=tuple(range(start, end)),
        passthrough_indices=tuple(sorted(passthrough)),
        policy_type=str(policy["type"]),
        chunk_size=int(policy["chunk_size"]),
        execution_horizon=int(policy["execution_horizon"]),
        kinematics=manifest.get("kinematics"),
        legacy=legacy,
    )


def assert_checkpoint_representation(
    source: str | Path,
    *,
    mode: str | None = None,
    pose_format: str | None = None,
    policy_type: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
    cache_dir: str | Path | None = None,
) -> ResolvedCheckpointContract:
    """Startup용 one-shot helper: resolve + CLI assertion."""
    contract = resolve_checkpoint_contract(
        source,
        revision=revision,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
    )
    return contract.assert_cli(mode=mode, pose_format=pose_format, policy_type=policy_type)
