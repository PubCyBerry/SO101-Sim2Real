"""Phase 13–14 — mode 중립 LeRobot processor step 쌍.

v1 :mod:`so101_contract.lerobot_eef_processor`는 ``eef_relative + rot6d`` 전용
SE(3) step이었다. 이 모듈은 :class:`~so101_contract.action_transform.ActionRepresentationTransform`
하나로 4개 mode를 모두 처리하는 registered pre/post step을 제공한다.

- ``joint_absolute``/``eef_absolute``: target 변환을 건너뛰고 canonical 정규화만 한다.
  기준 state 계약이 없으므로 **state cache를 만들지 않는다**(항상 ``None``).
- ``joint_relative``/``eef_relative``: prediction-time state를 cache하고, **full chunk**
  하나를 같은 기준 state로 복원한다.

processor는 dataset 없이 복원돼야 하므로 transform 계약과 stats payload를 config에 담는다.

Phase 15(policy factory/LeRobot patch 전면 배선)는 이 모듈을 쓰지 않고 v1 경로를 유지한다.
여기서는 protocol 호환 step과 factory helper까지만 제공한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

import torch

try:
    # LeRobot v0.6+
    from lerobot.configs import PipelineFeatureType, PolicyFeature
    from lerobot.types import EnvTransition, TransitionKey
except ImportError:
    from lerobot.configs.types import PipelineFeatureType, PolicyFeature
    from lerobot.processor.pipeline import EnvTransition, TransitionKey

from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from lerobot.utils.constants import OBS_STATE

from .action_representation_stats import restore_stats_from_processor
from .action_transform import ACTION_TRANSFORM_VERSION, ActionRepresentationTransform

ACTION_PROCESSOR_SCHEMA_VERSION = "so101_action_representation_processor_v2"
PREPROCESSOR_REGISTRY_NAME = "so101_action_representation_encode_v2"
POSTPROCESSOR_REGISTRY_NAME = "so101_action_representation_decode_v2"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _coerce_transform(value: Any) -> ActionRepresentationTransform:
    if isinstance(value, ActionRepresentationTransform):
        return value
    if isinstance(value, dict):
        return ActionRepresentationTransform.from_dict(value)
    raise TypeError(
        "transform must be ActionRepresentationTransform or its serialized dict, got "
        f"{type(value).__name__}"
    )


@ProcessorStepRegistry.register(PREPROCESSOR_REGISTRY_NAME)
@dataclass
class ActionRepresentationEncodeStep(ProcessorStep):
    """Absolute dataset action chunk → model target chunk.

    **relative mode에서만** observation state를 cache해 paired decode step이 같은 기준
    pose를 쓰게 한다. absolute mode는 기준 state 자체가 계약에 없으므로 cache를 만들지
    않고 항상 ``None``으로 유지한다(잘못된 re-anchoring 경로를 아예 없앤다).
    """

    transform: Any = None
    enabled: bool = True
    contract_fingerprint: str = ""
    stats_payload: dict[str, Any] = field(default_factory=dict)
    manifest_context: dict[str, Any] = field(default_factory=dict)
    strict: bool = True
    schema_version: str = ACTION_PROCESSOR_SCHEMA_VERSION
    transform_version: str = ACTION_TRANSFORM_VERSION
    _last_state: torch.Tensor | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.schema_version != ACTION_PROCESSOR_SCHEMA_VERSION:
            raise ValueError(
                f"processor schema version mismatch: {self.schema_version!r} != "
                f"{ACTION_PROCESSOR_SCHEMA_VERSION!r}"
            )
        if self.transform_version != ACTION_TRANSFORM_VERSION:
            raise ValueError(
                f"transform version mismatch: {self.transform_version!r} != "
                f"{ACTION_TRANSFORM_VERSION!r}"
            )
        self.transform = _coerce_transform(self.transform)
        if self.enabled and self.strict:
            if _SHA256_PATTERN.fullmatch(self.contract_fingerprint) is None:
                raise ValueError(
                    "strict action representation processor requires a dataset contract "
                    "SHA-256 fingerprint"
                )
            if not self.stats_payload:
                raise ValueError(
                    "strict action representation processor requires a self-contained stats "
                    "payload so inference works without the dataset"
                )
            # 저장된 stats가 조작됐으면 즉시 실패한다.
            restore_stats_from_processor(self.stats_payload)

    # --- runtime ------------------------------------------------------------

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(OBS_STATE) if observation else None
        if state is not None and not isinstance(state, torch.Tensor) and self.strict:
            raise TypeError(f"{OBS_STATE} must be a torch.Tensor, got {type(state).__name__}")
        if not self.transform.requires_state_reference:
            # absolute mode는 기준 state 계약이 없다. cache를 만들지 않고 비워 둔다.
            self._last_state = None
        elif isinstance(state, torch.Tensor):
            self._last_state = state

        if not self.enabled:
            return transition
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        if not isinstance(action, torch.Tensor):
            if self.strict:
                raise TypeError(
                    f"action must be a torch.Tensor, got {type(action).__name__}"
                )
            return transition
        if action.ndim != 3:
            raise ValueError(
                "action representation training target must be a full (B,H,D) chunk, got "
                f"{tuple(action.shape)}"
            )
        if self.transform.requires_state_reference and not isinstance(state, torch.Tensor):
            if self.strict:
                raise TypeError(
                    f"mode={self.transform.spec.mode.value!r} requires a torch state tensor"
                )
            return transition

        reference = state if isinstance(state, torch.Tensor) else None
        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = self.transform.encode(reference, action)
        return new_transition

    def get_cached_state(self) -> torch.Tensor | None:
        return self._last_state

    def reset(self) -> None:
        self._last_state = None

    def get_config(self) -> dict[str, Any]:
        return {
            "transform": self.transform.to_dict(),
            "enabled": self.enabled,
            "contract_fingerprint": self.contract_fingerprint,
            "stats_payload": self.stats_payload,
            "manifest_context": self.manifest_context,
            "strict": self.strict,
            "schema_version": self.schema_version,
            "transform_version": self.transform_version,
        }

    def transform_features(
        self,
        features: dict[PipelineFeatureType, dict[str, PolicyFeature]],
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(POSTPROCESSOR_REGISTRY_NAME)
@dataclass
class ActionRepresentationDecodeStep(ProcessorStep):
    """Model target chunk → absolute action chunk (chunk당 정확히 1회)."""

    transform: Any = None
    enabled: bool = True
    contract_fingerprint: str = ""
    strict: bool = True
    schema_version: str = ACTION_PROCESSOR_SCHEMA_VERSION
    transform_version: str = ACTION_TRANSFORM_VERSION
    encode_step: ActionRepresentationEncodeStep | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.schema_version != ACTION_PROCESSOR_SCHEMA_VERSION:
            raise ValueError(
                f"processor schema version mismatch: {self.schema_version!r} != "
                f"{ACTION_PROCESSOR_SCHEMA_VERSION!r}"
            )
        if self.transform_version != ACTION_TRANSFORM_VERSION:
            raise ValueError(
                f"transform version mismatch: {self.transform_version!r} != "
                f"{ACTION_TRANSFORM_VERSION!r}"
            )
        self.transform = _coerce_transform(self.transform)
        if (
            self.enabled
            and self.strict
            and _SHA256_PATTERN.fullmatch(self.contract_fingerprint) is None
        ):
            raise ValueError(
                "strict action representation decoder requires a dataset contract "
                "SHA-256 fingerprint"
            )

    def connect(self, encode_step: ActionRepresentationEncodeStep) -> None:
        if encode_step.contract_fingerprint != self.contract_fingerprint:
            raise ValueError(
                "pre/post action representation contract fingerprint mismatch: "
                f"{encode_step.contract_fingerprint} != {self.contract_fingerprint}"
            )
        if encode_step.transform.fingerprint() != self.transform.fingerprint():
            raise ValueError("pre/post action representation transform mismatch")
        self.encode_step = encode_step

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition
        if self.encode_step is None:
            raise RuntimeError(
                "action representation decoder is not connected to its encode step"
            )
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        if not isinstance(action, torch.Tensor):
            if self.strict:
                raise TypeError(
                    f"model action must be a torch.Tensor, got {type(action).__name__}"
                )
            return transition
        if action.ndim != 3:
            raise NotImplementedError(
                "action chunk must be decoded as a full (B,H,D) chunk before queue slicing; "
                f"got {tuple(action.shape)}"
            )

        state = self.encode_step.get_cached_state()
        if state is None:
            if self.transform.requires_state_reference:
                raise RuntimeError(
                    "relative decoder has no prediction-time state; run the paired encode step "
                    "first"
                )
            reference = None
        else:
            reference = state.to(device=action.device, dtype=action.dtype)

        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = self.transform.decode(reference, action)
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "transform": self.transform.to_dict(),
            "enabled": self.enabled,
            "contract_fingerprint": self.contract_fingerprint,
            "strict": self.strict,
            "schema_version": self.schema_version,
            "transform_version": self.transform_version,
        }

    def transform_features(
        self,
        features: dict[PipelineFeatureType, dict[str, PolicyFeature]],
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_action_representation_processor_steps(
    transform: ActionRepresentationTransform,
    *,
    contract_fingerprint: str,
    stats_payload: dict[str, Any],
    manifest_context: dict[str, Any] | None = None,
    strict: bool = True,
) -> tuple[ActionRepresentationEncodeStep, ActionRepresentationDecodeStep]:
    """같은 계약으로 연결된 encode/decode step 쌍을 생성."""
    encode_step = ActionRepresentationEncodeStep(
        transform=transform,
        enabled=True,
        contract_fingerprint=contract_fingerprint,
        stats_payload=dict(stats_payload),
        manifest_context=dict(manifest_context or {}),
        strict=strict,
    )
    decode_step = ActionRepresentationDecodeStep(
        transform=transform,
        enabled=True,
        contract_fingerprint=contract_fingerprint,
        strict=strict,
    )
    decode_step.connect(encode_step)
    return encode_step, decode_step


def reconnect_action_representation_processor_steps(
    preprocessor: Any,
    postprocessor: Any,
) -> None:
    """Serialized pipeline load 뒤 pre/post cache reference를 복구."""
    encode_steps = [
        step
        for step in getattr(preprocessor, "steps", [])
        if isinstance(step, ActionRepresentationEncodeStep) and step.enabled
    ]
    decode_steps = [
        step
        for step in getattr(postprocessor, "steps", [])
        if isinstance(step, ActionRepresentationDecodeStep) and step.enabled
    ]
    if not encode_steps and not decode_steps:
        return
    if len(encode_steps) != 1 or len(decode_steps) != 1:
        raise ValueError(
            "expected exactly one enabled action representation encode/decode pair, got "
            f"{len(encode_steps)}/{len(decode_steps)}"
        )
    decode_steps[0].connect(encode_steps[0])


def has_action_representation_processor_steps(
    preprocessor: Any,
    postprocessor: Any,
) -> bool:
    has_encode = any(
        isinstance(step, ActionRepresentationEncodeStep)
        for step in getattr(preprocessor, "steps", [])
    )
    has_decode = any(
        isinstance(step, ActionRepresentationDecodeStep)
        for step in getattr(postprocessor, "steps", [])
    )
    if has_encode != has_decode:
        raise ValueError("checkpoint contains an incomplete action representation step pair")
    return has_encode


def transform_from_v1_eef_contract(contract: Any) -> ActionRepresentationTransform:
    """v1 :class:`ResolvedEEFActionContract` → v2 transform (compatibility adapter).

    v1 ``eef_relative + rot6d`` checkpoint를 v2 step으로 재구성할 때 쓴다. v1 processor와
    수치가 동일해야 하며, 그 parity는
    ``scripts/contract/validate_action_representation_processor.py``가 검증한다.
    """
    spec = contract.config.to_spec()
    return ActionRepresentationTransform(
        spec=spec,
        state_indices=tuple(contract.state_pose_indices),
        action_indices=tuple(contract.action_pose_indices),
        passthrough_action_indices=tuple(contract.passthrough_action_indices),
        state_dim=int(contract.state_dim),
        action_dim=int(contract.action_dim),
    )
