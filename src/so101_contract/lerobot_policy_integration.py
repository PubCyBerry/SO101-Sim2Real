"""LeRobot v0.6 policy factory에 EEF-relative processor를 연결하는 공통 계층.

ACT·SmolVLA·GR00T-N1.7 policy별 factory는 기존 image/tokenizer/model packing을
그대로 유지한다. 이 모듈은 factory 호출 전 relative stats를 선택하고, 생성된
pipeline의 normalization 경계 바깥에 공통 SE(3) pre/post step을 삽입한다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

from .eef_action_contract import (
    ActionRepresentationConfig,
    ResolvedEEFActionContract,
    resolve_eef_action_contract,
)
from .eef_relative_stats import (
    RelativeActionSamplingConfig,
    inject_relative_action_stats,
    load_relative_action_stats_profile,
)
from .lerobot_eef_processor import (
    SE3AbsoluteActionsProcessorStep,
    SE3RelativeActionsProcessorStep,
    make_eef_relative_processor_steps,
    reconnect_eef_relative_processor_steps,
)

SUPPORTED_EEF_RELATIVE_POLICIES = frozenset({"act", "smolvla", "groot"})
_PREPROCESS_INSERT_BEFORE = (
    "normalizer_processor",
    "groot_n1_7_pack_inputs_v1",
)
_POSTPROCESS_INSERT_BEFORE = ("device_processor",)


@dataclass(frozen=True)
class EEFRelativePolicyContext:
    """한 training run에서 resolve된 dataset/sampler/stats 계약."""

    policy_type: str
    representation: ActionRepresentationConfig
    contract: ResolvedEEFActionContract
    sampling: RelativeActionSamplingConfig
    stats_profile_id: str
    stats_profile: dict[str, Any]
    dataset_stats: dict[str, dict[str, Any]]
    manifest_context: dict[str, Any]


def coerce_action_representation_config(value: Any) -> ActionRepresentationConfig:
    if value is None:
        return ActionRepresentationConfig()
    if isinstance(value, ActionRepresentationConfig):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, dict):
        payload = dict(value)
        if "passthrough_action_groups" in payload:
            payload["passthrough_action_groups"] = tuple(payload["passthrough_action_groups"])
        return ActionRepresentationConfig(**payload)
    raise TypeError(
        "action_representation must be ActionRepresentationConfig or dict, got "
        f"{type(value).__name__}"
    )


def policy_action_representation(policy_cfg: Any) -> ActionRepresentationConfig:
    return coerce_action_representation_config(
        getattr(policy_cfg, "action_representation", None)
    )


def _policy_type(policy_cfg: Any) -> str:
    policy_type = getattr(policy_cfg, "type", None)
    if not isinstance(policy_type, str) or not policy_type:
        policy_type = getattr(policy_cfg, "name", None)
    if not isinstance(policy_type, str) or not policy_type:
        policy_type = policy_cfg.__class__.__name__.removesuffix("Config").lower()
    return policy_type.lower()


def _sampling_from_policy(policy_cfg: Any) -> RelativeActionSamplingConfig:
    observation_delta_indices = getattr(policy_cfg, "observation_delta_indices", None)
    if not observation_delta_indices:
        observation_delta_indices = (0,)
    action_delta_indices = getattr(policy_cfg, "action_delta_indices", None)
    if not action_delta_indices:
        chunk_size = getattr(policy_cfg, "chunk_size", None)
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("EEF-relative policy requires positive action_delta_indices/chunk_size")
        action_delta_indices = tuple(range(chunk_size))
    return RelativeActionSamplingConfig(
        observation_delta_indices=tuple(observation_delta_indices),
        action_delta_indices=tuple(action_delta_indices),
        reference_observation_index=-1,
    )


def _dataset_root(dataset_meta: Any) -> Path:
    root = getattr(dataset_meta, "root", None)
    if root is None:
        raise ValueError("EEF-relative training requires dataset_meta.root")
    return Path(root).resolve()


def _dataset_manifest_source(
    dataset_meta: Any,
    contract: ResolvedEEFActionContract,
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Machine-local path 대신 재현 가능한 dataset identity를 기록."""
    repo_id = getattr(dataset_meta, "repo_id", None)
    revision = getattr(dataset_meta, "revision", None)
    return {
        "repo_id": repo_id if isinstance(repo_id, str) and repo_id else None,
        "revision": revision if isinstance(revision, str) and revision else None,
        "local_fingerprint": contract.fingerprint,
        "source_columns_sha256": profile["dataset_contract"]["source_columns_sha256"],
        "info_sha256": contract.info_sha256,
        "modality_sha256": contract.modality_sha256,
    }


def prepare_eef_relative_policy_context(
    policy_cfg: Any,
    *,
    dataset_meta: Any,
    dataset_stats: dict[str, dict[str, Any]] | None,
    verify_source_columns: bool = True,
) -> EEFRelativePolicyContext | None:
    """Factory 호출 전 relative action stats와 dataset contract를 resolve."""
    representation = policy_action_representation(policy_cfg)
    if representation.mode == "absolute":
        return None

    policy_type = _policy_type(policy_cfg)
    if policy_type not in SUPPORTED_EEF_RELATIVE_POLICIES:
        raise NotImplementedError(
            f"EEF-relative action is only supported for "
            f"{sorted(SUPPORTED_EEF_RELATIVE_POLICIES)}, got {policy_type!r}"
        )
    if dataset_meta is None or dataset_stats is None:
        raise ValueError(
            "fresh EEF-relative processor construction requires dataset_meta and dataset_stats; "
            "checkpoint inference must load the serialized processor pipelines"
        )
    if policy_type == "act" and getattr(policy_cfg, "temporal_ensemble_coeff", None) is not None:
        raise NotImplementedError("ACT temporal ensemble is disabled for EEF-relative v1")
    if policy_type == "smolvla" and getattr(policy_cfg, "rtc_config", None) is not None:
        raise NotImplementedError("SmolVLA RTC is disabled for EEF-relative v1")
    if policy_type == "groot" and getattr(policy_cfg, "use_relative_actions", False):
        raise ValueError(
            "GrootConfig.use_relative_actions and action_representation.mode='eef_relative' "
            "cannot be enabled together; disable the legacy generic relative path"
        )

    root = _dataset_root(dataset_meta)
    contract = resolve_eef_action_contract(root, representation)
    sampling = _sampling_from_policy(policy_cfg)
    profile_id, profile = load_relative_action_stats_profile(
        root,
        contract,
        sampling,
        stats_file=representation.stats_file,
        require_production=True,
        verify_source_columns=verify_source_columns,
    )
    merged_stats = inject_relative_action_stats(dataset_stats, profile)
    return EEFRelativePolicyContext(
        policy_type=policy_type,
        representation=representation,
        contract=contract,
        sampling=sampling,
        stats_profile_id=profile_id,
        stats_profile=profile,
        dataset_stats=merged_stats,
        manifest_context={
            "representation": representation.to_dict(),
            "resolved_contract": contract.to_dict(),
            "dataset": _dataset_manifest_source(dataset_meta, contract, profile),
            "relative_stats": {
                "profile_id": profile_id,
                "content_sha256": profile_id.removeprefix("sha256:"),
                "sampling": sampling.to_dict(),
                "dataset_contract": profile["dataset_contract"],
            },
        },
    )


def _registry_name(step: Any) -> str | None:
    return getattr(step.__class__, "_registry_name", None)


def _insert_before_registry(
    pipeline: Any,
    step: Any,
    target_registry_names: tuple[str, ...],
    *,
    pipeline_name: str,
) -> None:
    steps = list(getattr(pipeline, "steps", []))
    for existing in steps:
        if isinstance(existing, step.__class__):
            raise ValueError(f"{pipeline_name} already contains {step.__class__.__name__}")
    target_index = next(
        (
            index
            for index, existing in enumerate(steps)
            if _registry_name(existing) in target_registry_names
        ),
        None,
    )
    if target_index is None:
        raise ValueError(
            f"{pipeline_name} has no insertion boundary among {target_registry_names}; "
            f"steps={[_registry_name(existing) for existing in steps]}"
        )
    steps.insert(target_index, step)
    pipeline.steps = steps


def attach_eef_relative_processor_steps(
    preprocessor: Any,
    postprocessor: Any,
    context: EEFRelativePolicyContext,
) -> tuple[Any, Any]:
    """Policy 고유 normalization/packing 바깥에 공통 SE(3) step을 삽입."""
    relative_step, absolute_step = make_eef_relative_processor_steps(
        state_pose_indices=context.contract.state_pose_indices,
        action_pose_indices=context.contract.action_pose_indices,
        passthrough_action_indices=context.contract.passthrough_action_indices,
        contract_fingerprint=context.contract.fingerprint,
        manifest_context=context.manifest_context,
        strict=context.representation.strict,
    )
    _insert_before_registry(
        preprocessor,
        relative_step,
        _PREPROCESS_INSERT_BEFORE,
        pipeline_name="policy preprocessor",
    )
    _insert_before_registry(
        postprocessor,
        absolute_step,
        _POSTPROCESS_INSERT_BEFORE,
        pipeline_name="policy postprocessor",
    )
    return preprocessor, postprocessor


def replace_eef_relative_processor_steps(
    preprocessor: Any,
    postprocessor: Any,
    context: EEFRelativePolicyContext,
) -> tuple[Any, Any]:
    """EEF checkpoint를 새 dataset/stats로 fine-tune할 때 serialized pair를 교체."""
    preprocessor.steps = [
        step
        for step in getattr(preprocessor, "steps", [])
        if not isinstance(step, SE3RelativeActionsProcessorStep)
    ]
    postprocessor.steps = [
        step
        for step in getattr(postprocessor, "steps", [])
        if not isinstance(step, SE3AbsoluteActionsProcessorStep)
    ]
    return attach_eef_relative_processor_steps(preprocessor, postprocessor, context)


def reconnect_loaded_eef_relative_processors(
    preprocessor: Any,
    postprocessor: Any,
) -> tuple[Any, Any]:
    """Checkpoint에서 pipeline을 로드한 뒤 state cache 연결을 복원."""
    reconnect_eef_relative_processor_steps(preprocessor, postprocessor)
    return preprocessor, postprocessor


def has_eef_relative_processor_steps(preprocessor: Any, postprocessor: Any) -> bool:
    has_pre = any(
        isinstance(step, SE3RelativeActionsProcessorStep)
        for step in getattr(preprocessor, "steps", [])
    )
    has_post = any(
        isinstance(step, SE3AbsoluteActionsProcessorStep)
        for step in getattr(postprocessor, "steps", [])
    )
    if has_pre != has_post:
        raise ValueError("checkpoint contains an incomplete EEF relative/absolute processor pair")
    return has_pre
