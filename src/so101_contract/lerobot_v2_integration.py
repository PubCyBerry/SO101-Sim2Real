"""Phase 15 — LeRobot v0.6 policy factory ↔ schema v2 action representation 연결.

v1 :mod:`so101_contract.lerobot_policy_integration`은 ``eef_relative`` 한 조합만
연결했다. 이 모듈은 4개 mode 전부에 대해 다음을 담당한다.

1. policy config의 nested ``action_representation`` → :class:`ActionRepresentationSpec`
2. dataset metadata → :class:`ResolvedActionContract` → mode/format/horizon stats profile
3. registered v2 pre/post step 삽입·교체·재연결(encode는 normalize 앞, decode는
   policy unnormalize/decode 뒤 device 앞)
4. pretrained config의 stale feature schema 교체와 selective-reuse/reinit report
5. policy family별 capacity(mask/slice) 검증과 GR00T native relative 중복 차단
6. schema v2 ``action_representation.json`` 생성/검증

**엄격성**: 대상 policy(ACT·SmolVLA·GR00T-N1.7)는 mode와 무관하게 v2 dataset 계약·stats
profile·processor·manifest를 MUST 갖춘다. 계약이 없으면 추정하지 않고 fail-fast한다.
legacy checkpoint 승격은 Phase 16 migration + 명시적 opt-in에서만 처리한다.
대상이 아닌 policy(pi0 등)는 ``joint_absolute`` 기본값에서만 stock 경로를 유지한다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import logging
from pathlib import Path
from typing import Any

from .action_dataset_contract import (
    ResolvedActionContract,
    resolve_action_representation_contract,
)
from .action_manifest import (
    ACTION_REPRESENTATION_MANIFEST,
    build_action_representation_manifest,
    build_feature_contract,
    canonical_manifest_sha256,
    manifest_schema_version,
    read_action_representation_manifest,
    so101_contract_source_sha256,
    validate_action_representation_manifest,
)
from .action_representation import (
    SUPPORTED_POLICY_FAMILIES,
    ActionRepresentationMode,
    ActionRepresentationSpec,
    coerce_mode,
    coerce_pose_format,
)
from .action_representation_stats import (
    ActionStatsSampling,
    inject_action_stats,
    load_lerobot_v3_episodes,
    read_action_stats_artifact,
    select_stats_profile,
    serialize_stats_for_processor,
    source_columns_sha256,
)
from .action_transform import ActionRepresentationTransform

LOGGER = logging.getLogger(__name__)

ACTION_REPRESENTATION_INTEGRATION_VERSION = "so101_lerobot_v2_integration_v1"

#: policy family별 padding capacity. dataset dimension이 이 값을 넘으면 학습이 조용히
#: 잘리므로 factory에서 미리 거부한다.
POLICY_CAPACITY_ATTRIBUTES = {
    "smolvla": ("max_state_dim", "max_action_dim"),
    "groot": ("max_state_dim", "max_action_dim"),
}

#: encode step은 policy normalization/packing **앞**에 들어간다.
PREPROCESS_INSERT_BEFORE = (
    "normalizer_processor",
    "groot_n1_7_pack_inputs_v1",
)
#: decode step은 policy unnormalize/decode **뒤**에 들어간다(실측 registry 이름).
POSTPROCESS_INSERT_AFTER = (
    "unnormalizer_processor",
    "groot_action_unpack_unnormalize_v2",
    "groot_n1_7_action_decode_v1",
)
#: 위 step이 없을 때의 하한: device 이동 앞.
POSTPROCESS_INSERT_BEFORE = ("device_processor",)


# --- config -------------------------------------------------------------------


def _config_payload(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, ActionRepresentationSpec):
        return value.to_dict()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    raise TypeError(
        f"action_representation must be a dataclass or dict, got {type(value).__name__}"
    )


def spec_from_policy_config(
    policy_cfg: Any,
    *,
    allow_default: bool = True,
) -> ActionRepresentationSpec:
    """Policy config의 nested ``action_representation``을 v2 spec으로 해석.

    Args:
        allow_default: fresh factory config처럼 nested config가 없을 수 있는 경로에서만
            ``True``. 이때만 명시적 기본값 ``joint_absolute`` + ``pose_format=None``을
            만든다. checkpoint load 경로는 ``False``로 호출해, config가 없는 legacy
            checkpoint의 representation을 **추정하지 않는다**.

    v1 checkpoint가 저장한 ``state_pose_group``/``action_pose_group``/``reference`` 키는
    읽어 들이지만, 모호한 ``mode='absolute'``와 mode/format 불일치는 거부한다.
    """
    payload = _config_payload(getattr(policy_cfg, "action_representation", None))
    if payload is None or not payload:
        if allow_default:
            return ActionRepresentationSpec(mode=ActionRepresentationMode.JOINT_ABSOLUTE)
        raise ValueError(
            "policy config has no action_representation section; schema v2 never infers the "
            "representation of a legacy checkpoint (use the Phase 16 migration tool)"
        )

    mode = coerce_mode(payload["mode"]) if "mode" in payload else None
    if mode is None:
        raise ValueError("action_representation is missing the required 'mode' field")
    pose_format = coerce_pose_format(payload.get("pose_format"))
    reference = payload.get("reference_observation") or payload.get("reference")
    state_group = payload.get("state_group") or payload.get("state_pose_group") or ""
    action_group = payload.get("action_group") or payload.get("action_pose_group") or ""
    # mode/format/group 불일치는 조용히 보정하지 않는다(ActionRepresentationSpec가 거부).
    return ActionRepresentationSpec(
        mode=mode,
        pose_format=pose_format,
        reference_observation=reference or "current_observation",
        state_group=state_group,
        action_group=action_group,
        passthrough_action_groups=tuple(
            payload.get("passthrough_action_groups") or ("gripper_position",)
        ),
        gripper_representation=payload.get("gripper_representation", "absolute"),
        base_frame=payload.get("base_frame"),
        eef_frame=payload.get("eef_frame"),
        stats_file=payload.get("stats_file") or "meta/action_representation_stats.json",
        strict=bool(payload.get("strict", True)),
    )


def policy_type_of(policy_cfg: Any) -> str:
    value = getattr(policy_cfg, "type", None)
    if not isinstance(value, str) or not value:
        value = getattr(policy_cfg, "name", None)
    if not isinstance(value, str) or not value:
        value = policy_cfg.__class__.__name__.removesuffix("Config").lower()
    return value.lower()


def is_target_policy(policy_cfg: Any) -> bool:
    """이 프로젝트가 v2 계약을 강제하는 policy family인지."""
    return policy_type_of(policy_cfg) in SUPPORTED_POLICY_FAMILIES


# --- context ------------------------------------------------------------------


@dataclass(frozen=True)
class ActionRepresentationPolicyContext:
    """한 training run에서 resolve된 dataset/sampler/stats/transform 계약."""

    policy_type: str
    spec: ActionRepresentationSpec
    contract: ResolvedActionContract
    sampling: ActionStatsSampling
    stats_profile_id: str
    stats_profile: dict[str, Any]
    stats_payload: dict[str, Any]
    dataset_stats: dict[str, dict[str, Any]]
    manifest_context: dict[str, Any]

    @property
    def transform(self) -> ActionRepresentationTransform:
        return self.contract.transform


def _sampling_from_policy(policy_cfg: Any) -> ActionStatsSampling:
    observation_delta_indices = getattr(policy_cfg, "observation_delta_indices", None) or (0,)
    action_delta_indices = getattr(policy_cfg, "action_delta_indices", None)
    if not action_delta_indices:
        chunk_size = getattr(policy_cfg, "chunk_size", None)
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError(
                "action representation training requires positive action_delta_indices/chunk_size"
            )
        action_delta_indices = tuple(range(chunk_size))
    return ActionStatsSampling(
        observation_delta_indices=tuple(observation_delta_indices),
        action_delta_indices=tuple(action_delta_indices),
        reference_observation_index=-1,
    )


def _dataset_identity(dataset_meta: Any, contract: ResolvedActionContract) -> dict[str, Any]:
    repo_id = getattr(dataset_meta, "repo_id", None)
    revision = getattr(dataset_meta, "revision", None)
    return {
        "repo_id": repo_id if isinstance(repo_id, str) and repo_id else None,
        "revision": revision if isinstance(revision, str) and revision else None,
        "fingerprint": contract.fingerprint,
        "space": "absolute_eef" if contract.spec.is_eef else "absolute_joint",
        "storage_reference": "absolute",
        "info_sha256": contract.info_sha256,
        "modality_sha256": contract.modality_sha256,
    }


def _validate_policy_capacity(policy_cfg: Any, contract: ResolvedActionContract) -> dict[str, Any]:
    """policy family별 padding capacity와 output slice 계약 검증."""
    policy_type = policy_type_of(policy_cfg)
    report: dict[str, Any] = {
        "policy_type": policy_type,
        "state_dim": contract.state_dim,
        "action_dim": contract.action_dim,
        "output_slice": [0, contract.action_dim],
    }
    attributes = POLICY_CAPACITY_ATTRIBUTES.get(policy_type)
    if attributes is None:
        # ACT는 padding이 없다. projection/head가 dataset dimension에 정확히 맞아야 한다.
        report["padding"] = False
        return report
    state_attribute, action_attribute = attributes
    max_state = getattr(policy_cfg, state_attribute, None)
    max_action = getattr(policy_cfg, action_attribute, None)
    if not isinstance(max_state, int) or not isinstance(max_action, int):
        raise ValueError(f"{policy_type} config is missing {state_attribute}/{action_attribute}")
    if contract.state_dim > max_state:
        raise ValueError(
            f"{policy_type} {state_attribute}={max_state} cannot hold state dim "
            f"{contract.state_dim}"
        )
    if contract.action_dim > max_action:
        raise ValueError(
            f"{policy_type} {action_attribute}={max_action} cannot hold action dim "
            f"{contract.action_dim}"
        )
    report.update({"padding": True, state_attribute: max_state, action_attribute: max_action})
    return report


def _validate_groot_modality(policy_cfg: Any, contract: ResolvedActionContract) -> None:
    """GR00T native relative 경로와 공통 v2 변환의 중복을 차단하고 modality를 교차 검증."""
    if getattr(policy_cfg, "use_relative_actions", False):
        raise ValueError(
            "GrootConfig.use_relative_actions duplicates the common v2 relative transform; "
            "disable the native path and use --policy.action_representation.mode instead"
        )
    modality = getattr(policy_cfg, "modality_config", None) or getattr(policy_cfg, "modality", None)
    if not isinstance(modality, dict):
        return
    for section in ("state", "action"):
        groups = modality.get(section)
        if not isinstance(groups, dict):
            continue
        declared = contract.state_groups if section == "state" else contract.action_groups
        for name, bounds in groups.items():
            if name not in declared:
                raise ValueError(
                    f"GR00T modality declares {section}.{name!r} which the dataset contract "
                    f"does not define: {sorted(declared)}"
                )
            start = bounds.get("start") if isinstance(bounds, dict) else bounds[0]
            end = bounds.get("end") if isinstance(bounds, dict) else bounds[1]
            if (int(start), int(end)) != declared[name]:
                raise ValueError(
                    f"GR00T modality {section}.{name} range {(start, end)} disagrees with the "
                    f"dataset contract {declared[name]}"
                )
            if isinstance(bounds, dict) and bounds.get("rep") == "relative":
                raise ValueError(
                    "GR00T modality declares a native relative group; the common v2 transform "
                    "would apply the relative conversion twice"
                )


def prepare_action_representation_context(
    policy_cfg: Any,
    *,
    dataset_meta: Any,
    dataset_stats: dict[str, dict[str, Any]] | None,
    verify_source_columns: bool = True,
) -> ActionRepresentationPolicyContext | None:
    """Factory 호출 전에 dataset 계약과 stats profile을 resolve.

    대상 policy에서는 mode와 무관하게 계약을 MUST 만족한다. ``None``은 대상이 아닌
    policy(+``joint_absolute``)에서만 반환되며, 그 경우 stock LeRobot 경로를 그대로 쓴다.
    """
    spec = spec_from_policy_config(policy_cfg, allow_default=True)
    policy_type = policy_type_of(policy_cfg)
    if policy_type not in SUPPORTED_POLICY_FAMILIES:
        if spec.mode is ActionRepresentationMode.JOINT_ABSOLUTE:
            return None
        raise NotImplementedError(
            f"action representation mode {spec.mode.value!r} is only supported for "
            f"{sorted(SUPPORTED_POLICY_FAMILIES)}, got {policy_type!r}"
        )
    if dataset_meta is None or dataset_stats is None:
        raise ValueError(
            f"{policy_type} requires dataset_meta and dataset_stats to build the schema v2 "
            "action representation contract; checkpoint inference must load the serialized "
            "processor pipelines instead"
        )

    root = getattr(dataset_meta, "root", None)
    if root is None:
        raise ValueError("action representation training requires dataset_meta.root")
    contract = resolve_action_representation_contract(Path(root).resolve(), spec)

    if policy_type == "act" and getattr(policy_cfg, "temporal_ensemble_coeff", None) is not None:
        raise NotImplementedError(
            "ACT temporal ensemble is not supported with full-chunk action representations"
        )
    if policy_type == "smolvla" and getattr(policy_cfg, "rtc_config", None) is not None:
        raise NotImplementedError(
            "SmolVLA RTC is not supported with full-chunk action representations"
        )
    if policy_type == "groot":
        _validate_groot_modality(policy_cfg, contract)
    capacity = _validate_policy_capacity(policy_cfg, contract)

    sampling = _sampling_from_policy(policy_cfg)
    artifact = read_action_stats_artifact(Path(root).resolve(), output_file=spec.stats_file)
    profile_id, profile = select_stats_profile(
        artifact,
        contract.transform,
        sampling,
        dataset_fingerprint=contract.fingerprint,
    )
    if verify_source_columns:
        _verify_stats_source_columns(Path(root).resolve(), contract, profile)
    merged_stats = inject_action_stats(dataset_stats, profile)
    stats_payload = serialize_stats_for_processor(profile)

    manifest_context = {
        "action_representation": spec.to_dict(),
        "resolved_contract": contract.to_dict(),
        "dataset": _dataset_identity(dataset_meta, contract),
        "stats": {
            "profile_id": profile_id,
            "content_sha256": profile_id.removeprefix("sha256:"),
            "kind": profile["kind"],
            "horizon": sampling.horizon,
            "sampling": sampling.to_dict(),
        },
        "capacity": capacity,
    }
    LOGGER.info(
        "[action-representation] mode=%s pose_format=%s state/action=%d/%d stats=%s policy=%s",
        spec.mode.value,
        spec.pose_format.value,
        contract.state_dim,
        contract.action_dim,
        profile_id,
        policy_type,
    )
    return ActionRepresentationPolicyContext(
        policy_type=policy_type,
        spec=spec,
        contract=contract,
        sampling=sampling,
        stats_profile_id=profile_id,
        stats_profile=profile,
        stats_payload=stats_payload,
        dataset_stats=merged_stats,
        manifest_context=manifest_context,
    )


def _verify_stats_source_columns(
    root: Path,
    contract: ResolvedActionContract,
    profile: dict[str, Any],
) -> None:
    """stats 생성 이후 dataset이 수정됐는지 checksum으로 확인."""
    expected = profile["dataset"].get("source_columns_sha256")
    if not expected:
        return
    episodes = load_lerobot_v3_episodes(
        root,
        state_key=contract.state_key,
        action_key=contract.action_key,
        state_dim=contract.state_dim,
        action_dim=contract.action_dim,
        max_episodes=profile["dataset"].get("max_episodes"),
    )
    if source_columns_sha256(episodes) != expected:
        raise ValueError(
            "action stats source state/action checksum mismatch; regenerate the stats artifact"
        )


# --- processor 삽입 -----------------------------------------------------------


def _registry_name(step: Any) -> str | None:
    return getattr(step.__class__, "_registry_name", None)


def _insert_before(pipeline: Any, step: Any, targets: tuple[str, ...], *, name: str) -> None:
    steps = list(getattr(pipeline, "steps", []))
    for existing in steps:
        if isinstance(existing, step.__class__):
            raise ValueError(f"{name} already contains {step.__class__.__name__}")
    index = next(
        (i for i, existing in enumerate(steps) if _registry_name(existing) in targets),
        None,
    )
    if index is None:
        raise ValueError(
            f"{name} has no insertion boundary among {targets}; "
            f"steps={[_registry_name(existing) for existing in steps]}"
        )
    steps.insert(index, step)
    pipeline.steps = steps


def _insert_after(
    pipeline: Any,
    step: Any,
    after: tuple[str, ...],
    before: tuple[str, ...],
    *,
    name: str,
) -> None:
    """policy unnormalize/decode 뒤(마지막 매치 다음), 없으면 device 이동 앞."""
    steps = list(getattr(pipeline, "steps", []))
    for existing in steps:
        if isinstance(existing, step.__class__):
            raise ValueError(f"{name} already contains {step.__class__.__name__}")
    index = None
    for position, existing in enumerate(steps):
        if _registry_name(existing) in after:
            index = position + 1
    if index is None:
        index = next(
            (i for i, existing in enumerate(steps) if _registry_name(existing) in before),
            None,
        )
    if index is None:
        raise ValueError(
            f"{name} has no insertion boundary among {after + before}; "
            f"steps={[_registry_name(existing) for existing in steps]}"
        )
    steps.insert(index, step)
    pipeline.steps = steps


def attach_action_representation_steps(
    preprocessor: Any,
    postprocessor: Any,
    context: ActionRepresentationPolicyContext,
) -> tuple[Any, Any]:
    """encode step은 normalize 앞, decode step은 policy unnormalize/decode 뒤에 삽입."""
    from .action_representation_processor import make_action_representation_processor_steps

    encode_step, decode_step = make_action_representation_processor_steps(
        context.transform,
        contract_fingerprint=context.contract.fingerprint,
        stats_payload=context.stats_payload,
        manifest_context=context.manifest_context,
        strict=context.spec.strict,
    )
    _insert_before(
        preprocessor,
        encode_step,
        PREPROCESS_INSERT_BEFORE,
        name="policy preprocessor",
    )
    _insert_after(
        postprocessor,
        decode_step,
        POSTPROCESS_INSERT_AFTER,
        POSTPROCESS_INSERT_BEFORE,
        name="policy postprocessor",
    )
    return preprocessor, postprocessor


def replace_action_representation_steps(
    preprocessor: Any,
    postprocessor: Any,
    context: ActionRepresentationPolicyContext,
) -> tuple[Any, Any]:
    """Checkpoint를 새 dataset/stats로 fine-tune할 때 serialized pair를 교체."""
    from .action_representation_processor import (
        ActionRepresentationDecodeStep,
        ActionRepresentationEncodeStep,
    )

    preprocessor.steps = [
        step
        for step in getattr(preprocessor, "steps", [])
        if not isinstance(step, ActionRepresentationEncodeStep)
    ]
    postprocessor.steps = [
        step
        for step in getattr(postprocessor, "steps", [])
        if not isinstance(step, ActionRepresentationDecodeStep)
    ]
    return attach_action_representation_steps(preprocessor, postprocessor, context)


def reconnect_action_representation_steps(preprocessor: Any, postprocessor: Any) -> tuple[Any, Any]:
    from .action_representation_processor import reconnect_action_representation_processor_steps

    reconnect_action_representation_processor_steps(preprocessor, postprocessor)
    return preprocessor, postprocessor


def has_action_representation_steps(preprocessor: Any, postprocessor: Any) -> bool:
    from .action_representation_processor import has_action_representation_processor_steps

    return has_action_representation_processor_steps(preprocessor, postprocessor)


def action_representation_encode_step(preprocessor: Any) -> Any | None:
    from .action_representation_processor import ActionRepresentationEncodeStep

    matches = [
        step
        for step in getattr(preprocessor, "steps", [])
        if isinstance(step, ActionRepresentationEncodeStep) and step.enabled
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"expected exactly one v2 encode step, got {len(matches)}")
    return matches[0]


def processor_step_order(pipeline: Any) -> list[str | None]:
    """검증/로그용 step registry 이름 순서."""
    return [_registry_name(step) for step in getattr(pipeline, "steps", [])]


# --- feature schema -----------------------------------------------------------


def override_policy_features(
    policy_cfg: Any,
    features: dict[str, Any],
    rename_map: dict[str, str] | None,
    *,
    feature_type_action: Any,
) -> bool:
    """Pretrained config의 stale input/output schema를 dataset schema로 교체.

    대상 policy의 **모든 v2 mode**에서 dataset shape가 진실이다(joint 6D ↔ EEF 10/8/7D).
    pretrained VLA config가 들고 있는 옛 schema를 그대로 두면 projection/head가 잘못된
    차원으로 만들어진다. Camera rename을 feature key에 먼저 적용하는 이유는 preprocessor가
    model 앞에서 같은 rename을 수행하기 때문이다.
    """
    if not is_target_policy(policy_cfg):
        return False
    renamed = {(rename_map or {}).get(key, key): value for key, value in features.items()}
    policy_cfg.output_features = {
        key: value for key, value in renamed.items() if value.type is feature_type_action
    }
    policy_cfg.input_features = {
        key: value for key, value in renamed.items() if key not in policy_cfg.output_features
    }
    return True


# --- selective reuse / reinit -------------------------------------------------


@dataclass(frozen=True)
class SelectiveReuseReport:
    """Pretrained weight 재사용/재초기화 결과."""

    reused: tuple[str, ...]
    reinitialized: tuple[tuple[str, str, str], ...]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    #: checkpoint에 없지만 **다른 로드된 tensor와 storage를 공유**하는 key.
    #: ``safetensors`` 저장이 shared tensor 중복을 의도적으로 버리기 때문에 생기며,
    #: 실제 누락이 아니라 tying으로 이미 복원된 값이다.
    tied: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "reused_parameter_count": len(self.reused),
            "reinitialized": [
                {"key": key, "checkpoint_shape": source, "model_shape": target}
                for key, source, target in self.reinitialized
            ],
            "missing_parameter_count": len(self.missing),
            "unexpected_parameter_count": len(self.unexpected),
            "missing": list(self.missing[:32]),
            "unexpected": list(self.unexpected[:32]),
            "tied_parameter_count": len(self.tied),
            "tied": list(self.tied[:32]),
        }

    @property
    def has_reinitialized_layers(self) -> bool:
        return bool(self.reinitialized)


def plan_selective_reuse(model: Any, state_dict: dict[str, Any]) -> SelectiveReuseReport:
    """모델 parameter와 checkpoint tensor의 shape를 비교해 재사용 계획을 만든다."""
    model_state = model.state_dict()
    reused: list[str] = []
    reinitialized: list[tuple[str, str, str]] = []
    for key, tensor in state_dict.items():
        target = model_state.get(key)
        if target is None:
            continue
        if tuple(target.shape) != tuple(tensor.shape):
            reinitialized.append((key, str(tuple(tensor.shape)), str(tuple(target.shape))))
        else:
            reused.append(key)
    missing = [key for key in model_state if key not in state_dict]
    unexpected = [key for key in state_dict if key not in model_state]

    # safetensors 저장은 shared tensor 중복을 버린다. 그래서 tying된 key는 checkpoint에
    # 없지만 실제로는 로드된 tensor와 같은 storage를 가리키므로 누락이 아니다.
    reused_storages = set()
    for key in reused:
        tensor = model_state.get(key)
        pointer = getattr(tensor, "data_ptr", None)
        if pointer is not None:
            reused_storages.add(pointer())
    tied = []
    for key in list(missing):
        tensor = model_state.get(key)
        pointer = getattr(tensor, "data_ptr", None)
        if pointer is not None and pointer() in reused_storages:
            tied.append(key)
    missing = [key for key in missing if key not in set(tied)]
    return SelectiveReuseReport(
        reused=tuple(sorted(reused)),
        reinitialized=tuple(sorted(reinitialized)),
        missing=tuple(sorted(missing)),
        unexpected=tuple(sorted(unexpected)),
        tied=tuple(sorted(tied)),
    )


def _load_state_dict_any(model_file: str | Path, device: str) -> dict[str, Any]:
    """단일/sharded safetensors를 모두 처리."""
    import json

    from safetensors.torch import load_file

    path = Path(model_file)
    if path.suffix == ".json" or path.name.endswith(".index.json"):
        index = json.loads(path.read_text(encoding="utf-8"))
        shards = sorted(set(index.get("weight_map", {}).values()))
        if not shards:
            raise ValueError(f"sharded safetensors index has no weight_map: {path}")
        state_dict: dict[str, Any] = {}
        for shard in shards:
            state_dict.update(load_file(str(path.parent / shard), device=device))
        return state_dict
    index_path = path.parent / f"{path.name}.index.json"
    if index_path.is_file():
        return _load_state_dict_any(index_path, device)
    return load_file(str(path), device=device)


def load_pretrained_with_selective_reuse(
    model: Any,
    model_file: str | Path,
    *,
    device: str | None = None,
    strict: bool = False,
) -> SelectiveReuseReport:
    """Shape가 맞는 parameter만 로드하고 나머지는 **명시적으로** 재초기화 상태로 남긴다.

    representation 전환(예: joint 6D → EEF 10D)에서 ACT의 state projection과 action head
    처럼 dimension에 종속된 layer 크기가 달라진다. stock loader는 이때 RuntimeError로
    죽거나 사용자가 눈치채지 못하게 넘어간다. 이 함수는 재사용/재초기화 목록을 report로
    남기고 model에 부착해 manifest까지 전달한다.

    같은 representation을 resume할 때는 재초기화가 0이어야 하며, ``strict=True``면 어떤
    불일치도 허용하지 않는다(엄격 parity).
    """
    resolved_device = device or "cpu"
    state_dict = _load_state_dict_any(model_file, resolved_device)
    report = plan_selective_reuse(model, state_dict)
    if strict and (report.reinitialized or report.missing or report.unexpected):
        raise RuntimeError(
            "strict pretrained load failed: "
            f"reinitialized={len(report.reinitialized)}, missing={len(report.missing)}, "
            f"unexpected={len(report.unexpected)}"
        )
    compatible = {key: state_dict[key] for key in report.reused}
    model.load_state_dict(compatible, strict=False)
    if report.reinitialized:
        LOGGER.warning(
            "[action-representation] %d pretrained layer(s) are dimension-incompatible and stay "
            "randomly initialized: %s",
            len(report.reinitialized),
            ", ".join(key for key, _, _ in report.reinitialized[:8]),
        )
    model._so101_selective_reuse_report = report
    return report


def selective_reuse_report_of(policy: Any) -> SelectiveReuseReport | None:
    return getattr(policy, "_so101_selective_reuse_report", None)


# --- manifest -----------------------------------------------------------------


def _policy_manifest_section(policy_cfg: Any, capacity: dict[str, Any]) -> dict[str, Any]:
    policy_type = policy_type_of(policy_cfg)
    if policy_type not in SUPPORTED_POLICY_FAMILIES:
        raise NotImplementedError(f"unsupported policy family for a v2 manifest: {policy_type!r}")
    chunk_size = getattr(policy_cfg, "chunk_size", None)
    execution_horizon = getattr(policy_cfg, "n_action_steps", None)
    base_model_path = getattr(policy_cfg, "base_model_path", None) or getattr(
        policy_cfg,
        "pretrained_path",
        None,
    )
    return {
        "type": policy_type,
        "model_family": "GR00T-N1.7" if policy_type == "groot" else policy_type.upper(),
        "base_model_path": str(base_model_path) if base_model_path else None,
        "chunk_size": int(chunk_size) if isinstance(chunk_size, int) else None,
        "execution_horizon": int(execution_horizon) if isinstance(execution_horizon, int) else None,
        "prediction_api": "predict_action_chunk",
        "full_chunk_postprocess_required": True,
        "capacity": capacity,
    }


def _runtime_section() -> dict[str, Any]:
    import importlib.metadata
    import os
    import re
    import subprocess

    version = os.environ.get("LEROBOT_RUNTIME_VERSION", "").strip()
    if not version:
        version = importlib.metadata.version("lerobot")
    commit = os.environ.get("LEROBOT_GIT_COMMIT", "").strip().lower()
    if not commit:
        from .eef_checkpoint_manifest import PINNED_LEROBOT_COMMIT

        commit = PINNED_LEROBOT_COMMIT
    project_commit = os.environ.get("SO101_PROJECT_GIT_COMMIT", "").strip().lower()
    if not project_commit:
        repo_root = Path(__file__).resolve().parents[2]
        try:
            project_commit = (
                subprocess.run(
                    ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                .stdout.strip()
                .lower()
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                "SO101 project commit is unavailable; set SO101_PROJECT_GIT_COMMIT"
            ) from exc
    if re.fullmatch(r"[0-9a-f]{40}", project_commit) is None:
        raise RuntimeError(f"invalid project commit: {project_commit!r}")
    source_hash = so101_contract_source_sha256()
    return {
        "lerobot_version": version,
        "lerobot_commit": commit,
        "project_commit": project_commit,
        "so101_contract_source_sha256": source_hash,
        "processor_source_sha256": source_hash,
        "integration_version": ACTION_REPRESENTATION_INTEGRATION_VERSION,
        "compatible_clients": [
            "lerobot_async_full_chunk",
            "lerobot_sync_full_chunk",
            "so101_vla_policy_ros2",
            "so101_eef_robot_client",
        ],
    }


def build_manifest_from_processor(
    policy_cfg: Any,
    preprocessor: Any,
    *,
    policy: Any | None = None,
) -> dict[str, Any]:
    """Serialized v2 encode step에서 schema v2 manifest를 만든다.

    대상 policy의 신규 checkpoint에는 encode step이 반드시 있어야 하며, 없으면 오류다.
    """
    step = action_representation_encode_step(preprocessor)
    if step is None:
        raise ValueError(
            "schema v2 manifest requires a serialized action representation encode step; "
            "the policy processor pipeline has none"
        )
    context = dict(step.manifest_context)
    required = ("action_representation", "resolved_contract", "dataset", "stats")
    missing = [key for key in required if not isinstance(context.get(key), dict)]
    if missing:
        raise ValueError(f"v2 processor manifest context is incomplete: {missing}")

    spec = ActionRepresentationSpec.from_dict(context["action_representation"])
    contract = context["resolved_contract"]
    if contract.get("fingerprint") != step.contract_fingerprint:
        raise ValueError("processor manifest contract fingerprint mismatch")

    state_feature = build_feature_contract(
        contract["state_key"],
        contract["state_names"],
        {name: tuple(bounds) for name, bounds in contract["state_groups"].items()},
    )
    action_feature = build_feature_contract(
        contract["action_key"],
        contract["action_names"],
        {name: tuple(bounds) for name, bounds in contract["action_groups"].items()},
    )
    kinematics = None
    if spec.is_eef:
        kinematics = {
            "version": contract["eef_kinematics_version"],
            "urdf_sha256": contract["urdf_sha256"],
            "robot_yaml_sha256": contract["robot_yaml_sha256"],
        }
    stats = dict(context["stats"])
    report = selective_reuse_report_of(policy) if policy is not None else None
    return build_action_representation_manifest(
        spec,
        state_feature=state_feature,
        action_feature=action_feature,
        dataset=context["dataset"],
        stats={
            "profile_id": stats["profile_id"],
            "content_sha256": stats["content_sha256"],
            "kind": stats["kind"],
            "horizon": stats["horizon"],
            "sampling": stats.get("sampling"),
        },
        policy=_policy_manifest_section(policy_cfg, context.get("capacity", {})),
        runtime=_runtime_section(),
        action_horizon=int(stats["horizon"]),
        resolved_contract_fingerprint=step.contract_fingerprint,
        transform=contract.get("transform"),
        kinematics=kinematics,
        legacy={"allowed": False, "flag": None},
        selective_reuse=report.to_dict() if report is not None else None,
    )


def write_action_representation_manifest_for(
    pretrained_dir: str | Path,
    policy_cfg: Any,
    preprocessor: Any,
    *,
    policy: Any | None = None,
) -> Path | None:
    """Local checkpoint(periodic·final)에 manifest 저장.

    대상 policy가 아니거나 v2 step이 없는 stock 경로에서는 ``None``을 돌려준다.
    """
    from .action_manifest import write_action_representation_manifest

    if action_representation_encode_step(preprocessor) is None:
        if is_target_policy(policy_cfg):
            raise ValueError(
                f"{policy_type_of(policy_cfg)} checkpoint has no schema v2 action representation "
                "processor; every new target checkpoint must carry one"
            )
        return None
    manifest = build_manifest_from_processor(policy_cfg, preprocessor, policy=policy)
    return write_action_representation_manifest(pretrained_dir, manifest)


def push_action_representation_manifest_for(
    repo_id: str,
    policy_cfg: Any,
    preprocessor: Any,
    *,
    policy: Any | None = None,
) -> bool:
    """Hub model root에 manifest 업로드."""
    import json
    import tempfile

    if action_representation_encode_step(preprocessor) is None:
        if is_target_policy(policy_cfg):
            raise ValueError(
                "cannot push a target-policy checkpoint without a schema v2 action representation"
            )
        return False
    manifest = build_manifest_from_processor(policy_cfg, preprocessor, policy=policy)
    from huggingface_hub import HfApi

    with tempfile.TemporaryDirectory(prefix="so101-action-manifest-") as directory:
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
            commit_message="Add schema v2 action representation manifest",
        )
    return True


def validate_checkpoint_manifest(
    pretrained_path: str | Path,
    policy_cfg: Any,
    preprocessor: Any,
) -> dict[str, Any]:
    """Checkpoint load 시 manifest 누락/불일치를 fail-fast로 거부.

    v1 manifest는 **자동 승격하지 않는다**(Phase 16 migration 또는 명시적 legacy opt-in 필요).
    """
    path = Path(pretrained_path)
    manifest = read_action_representation_manifest(path) if path.exists() else None
    if manifest is None:
        raise FileNotFoundError(
            f"checkpoint is missing {ACTION_REPRESENTATION_MANIFEST}: {pretrained_path}. "
            "Action representation is never inferred; run the Phase 16 migration tool."
        )
    if manifest_schema_version(manifest) != 2:
        raise ValueError(
            f"checkpoint carries a v1 action representation manifest: {pretrained_path}. "
            "Automatic promotion is disabled; use the Phase 16 migration tool or an explicit "
            "legacy opt-in."
        )
    spec = spec_from_policy_config(policy_cfg, allow_default=False)
    validate_action_representation_manifest(
        manifest,
        expected_spec=spec,
        expected_policy_type=policy_type_of(policy_cfg),
    )
    step = action_representation_encode_step(preprocessor)
    if step is None:
        raise ValueError(
            "schema v2 manifest was loaded without a serialized action representation processor"
        )
    if step.contract_fingerprint != manifest["resolved_contract_fingerprint"]:
        raise ValueError(
            "manifest/processor contract fingerprint mismatch: "
            f"{step.contract_fingerprint} != {manifest['resolved_contract_fingerprint']}"
        )
    stats_payload = step.stats_payload or {}
    if stats_payload.get("content_sha256") != manifest["stats"]["content_sha256"]:
        raise ValueError("manifest/processor stats hash mismatch")
    return manifest


def manifest_content_sha256(manifest: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return canonical_manifest_sha256(unsigned)
