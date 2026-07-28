"""LeRobot policy pipeline용 EEF-relative SE(3) processor steps.

LeRobot v0.6.0의 generic relative processor는 joint component별 ``action-state``
연산이다. 이 모듈은 canonical 10D ``xyz + Rot6D(rows) + gripper``에 대해
rigid transform compose를 수행하며, chunk 전체가 같은 prediction-time state를
기준으로 사용하도록 pre/post step을 연결한다.

이 모듈은 LeRobot processor가 필요한 실행 경로에서만 명시적으로 import한다.
따라서 ``so101_contract``의 순수 FK/codec 사용자는 LeRobot import에 의존하지 않는다.
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
    # 루트 Isaac/legacy host 환경에서도 순수 contract import를 허용한다.
    from lerobot.configs.types import PipelineFeatureType, PolicyFeature
    from lerobot.processor.pipeline import EnvTransition, TransitionKey

from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from lerobot.utils.constants import OBS_STATE

from .eef_relative_action import (
    EEF_RELATIVE_ACTION_VERSION,
    absolute_actions_to_relative,
    relative_actions_to_absolute,
)

SE3_RELATIVE_PROCESSOR_SCHEMA_VERSION = "so101_lerobot_se3_processor_v1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _as_index_tuple(values: tuple[int, ...] | list[int], name: str) -> tuple[int, ...]:
    indices = tuple(values)
    if not indices or not all(isinstance(index, int) for index in indices):
        raise ValueError(f"{name} must contain integer indices, got {values!r}")
    if len(set(indices)) != len(indices) or any(index < 0 for index in indices):
        raise ValueError(f"{name} must be unique and non-negative, got {indices}")
    return indices


def _validate_runtime_layout(
    state: torch.Tensor,
    action: torch.Tensor,
    *,
    state_pose_indices: tuple[int, ...],
    action_pose_indices: tuple[int, ...],
    passthrough_action_indices: tuple[int, ...],
) -> None:
    if state.shape[-1] <= max(state_pose_indices):
        raise ValueError(
            f"state dim {state.shape[-1]} does not cover pose indices {state_pose_indices}"
        )
    classified = set(action_pose_indices).union(passthrough_action_indices)
    expected = set(range(action.shape[-1]))
    if classified != expected:
        raise ValueError(
            "action indices do not classify the complete runtime action: "
            f"classified={sorted(classified)}, expected={sorted(expected)}"
        )


@ProcessorStepRegistry.register("so101_se3_relative_actions_processor")
@dataclass
class SE3RelativeActionsProcessorStep(ProcessorStep):
    """Absolute EEF action chunk를 current observation 기준 SE(3) relative로 변환."""

    enabled: bool = True
    state_pose_indices: tuple[int, ...] | list[int] = tuple(range(9))
    action_pose_indices: tuple[int, ...] | list[int] = tuple(range(9))
    passthrough_action_indices: tuple[int, ...] | list[int] = (9,)
    contract_fingerprint: str = ""
    manifest_context: dict[str, Any] = field(default_factory=dict)
    strict: bool = True
    transform_version: str = EEF_RELATIVE_ACTION_VERSION
    _last_state: torch.Tensor | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.state_pose_indices = _as_index_tuple(
            self.state_pose_indices,
            "state_pose_indices",
        )
        self.action_pose_indices = _as_index_tuple(
            self.action_pose_indices,
            "action_pose_indices",
        )
        self.passthrough_action_indices = _as_index_tuple(
            self.passthrough_action_indices,
            "passthrough_action_indices",
        )
        if len(self.state_pose_indices) != 9 or len(self.action_pose_indices) != 9:
            raise ValueError("SE(3) EEF pose index groups must both be 9D")
        if set(self.action_pose_indices).intersection(self.passthrough_action_indices):
            raise ValueError("action pose and passthrough indices overlap")
        if self.transform_version != EEF_RELATIVE_ACTION_VERSION:
            raise ValueError(
                f"transform version mismatch: {self.transform_version!r} != "
                f"{EEF_RELATIVE_ACTION_VERSION!r}"
            )
        if self.enabled and self.strict and _SHA256_PATTERN.fullmatch(
            self.contract_fingerprint
        ) is None:
            raise ValueError(
                "strict EEF-relative processor requires a dataset contract SHA-256 fingerprint"
            )
        if self.enabled and self.strict and not self.manifest_context:
            raise ValueError(
                "strict EEF-relative processor requires self-contained manifest context"
            )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(OBS_STATE) if observation else None
        if state is not None:
            if not isinstance(state, torch.Tensor):
                if self.strict:
                    raise TypeError(
                        f"{OBS_STATE} must be a torch.Tensor, got {type(state).__name__}"
                    )
            else:
                self._last_state = state

        if not self.enabled:
            return transition
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        if not isinstance(action, torch.Tensor) or not isinstance(state, torch.Tensor):
            if self.strict:
                raise TypeError("EEF-relative preprocessing requires torch action and state tensors")
            return transition
        if action.ndim != 3:
            raise ValueError(
                "EEF-relative training action must be a full (B,H,D) chunk, got "
                f"{tuple(action.shape)}"
            )
        _validate_runtime_layout(
            state,
            action,
            state_pose_indices=self.state_pose_indices,
            action_pose_indices=self.action_pose_indices,
            passthrough_action_indices=self.passthrough_action_indices,
        )

        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = absolute_actions_to_relative(
            state,
            action,
            state_pose_indices=self.state_pose_indices,
            action_pose_indices=self.action_pose_indices,
        )
        return new_transition

    def get_cached_state(self) -> torch.Tensor | None:
        return self._last_state

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "state_pose_indices": list(self.state_pose_indices),
            "action_pose_indices": list(self.action_pose_indices),
            "passthrough_action_indices": list(self.passthrough_action_indices),
            "contract_fingerprint": self.contract_fingerprint,
            "manifest_context": self.manifest_context,
            "strict": self.strict,
            "transform_version": self.transform_version,
        }

    def reset(self) -> None:
        self._last_state = None

    def transform_features(
        self,
        features: dict[PipelineFeatureType, dict[str, PolicyFeature]],
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("so101_se3_absolute_actions_processor")
@dataclass
class SE3AbsoluteActionsProcessorStep(ProcessorStep):
    """Full relative chunk를 prediction-time cached state 기준 absolute EEF로 복원."""

    enabled: bool = True
    contract_fingerprint: str = ""
    strict: bool = True
    transform_version: str = EEF_RELATIVE_ACTION_VERSION
    relative_step: SE3RelativeActionsProcessorStep | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.transform_version != EEF_RELATIVE_ACTION_VERSION:
            raise ValueError(
                f"transform version mismatch: {self.transform_version!r} != "
                f"{EEF_RELATIVE_ACTION_VERSION!r}"
            )
        if self.enabled and self.strict and _SHA256_PATTERN.fullmatch(
            self.contract_fingerprint
        ) is None:
            raise ValueError(
                "strict EEF-absolute processor requires a dataset contract SHA-256 fingerprint"
            )

    def connect(self, relative_step: SE3RelativeActionsProcessorStep) -> None:
        if relative_step.contract_fingerprint != self.contract_fingerprint:
            raise ValueError(
                "pre/post EEF processor contract fingerprint mismatch: "
                f"{relative_step.contract_fingerprint} != {self.contract_fingerprint}"
            )
        self.relative_step = relative_step

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition
        if self.relative_step is None:
            raise RuntimeError("EEF absolute postprocessor is not connected to its relative preprocessor")
        state = self.relative_step.get_cached_state()
        if state is None:
            raise RuntimeError(
                "EEF absolute postprocessor has no prediction-time state; "
                "run the paired preprocessor first"
            )
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        if not isinstance(action, torch.Tensor):
            if self.strict:
                raise TypeError(
                    f"EEF-relative model action must be a torch.Tensor, got {type(action).__name__}"
                )
            return transition
        if action.ndim != 3:
            raise NotImplementedError(
                "EEF-relative action must be decoded as a full (B,H,D) chunk before queue slicing; "
                f"got {tuple(action.shape)}"
            )

        cached_state = state.to(device=action.device, dtype=action.dtype)
        _validate_runtime_layout(
            cached_state,
            action,
            state_pose_indices=self.relative_step.state_pose_indices,
            action_pose_indices=self.relative_step.action_pose_indices,
            passthrough_action_indices=self.relative_step.passthrough_action_indices,
        )
        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = relative_actions_to_absolute(
            cached_state,
            action,
            state_pose_indices=self.relative_step.state_pose_indices,
            action_pose_indices=self.relative_step.action_pose_indices,
        )
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "contract_fingerprint": self.contract_fingerprint,
            "strict": self.strict,
            "transform_version": self.transform_version,
        }

    def transform_features(
        self,
        features: dict[PipelineFeatureType, dict[str, PolicyFeature]],
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_eef_relative_processor_steps(
    *,
    state_pose_indices: tuple[int, ...],
    action_pose_indices: tuple[int, ...],
    passthrough_action_indices: tuple[int, ...],
    contract_fingerprint: str,
    manifest_context: dict[str, Any] | None = None,
    strict: bool = True,
) -> tuple[SE3RelativeActionsProcessorStep, SE3AbsoluteActionsProcessorStep]:
    """동일 contract로 연결된 pre/post step 쌍을 생성."""
    relative_step = SE3RelativeActionsProcessorStep(
        enabled=True,
        state_pose_indices=state_pose_indices,
        action_pose_indices=action_pose_indices,
        passthrough_action_indices=passthrough_action_indices,
        contract_fingerprint=contract_fingerprint,
        manifest_context=dict(manifest_context or {}),
        strict=strict,
    )
    absolute_step = SE3AbsoluteActionsProcessorStep(
        enabled=True,
        contract_fingerprint=contract_fingerprint,
        strict=strict,
    )
    absolute_step.connect(relative_step)
    return relative_step, absolute_step


def reconnect_eef_relative_processor_steps(
    preprocessor: Any,
    postprocessor: Any,
) -> None:
    """Serialized pipeline load 뒤 pre/post cache reference를 복구."""
    relative_steps = [
        step
        for step in getattr(preprocessor, "steps", [])
        if isinstance(step, SE3RelativeActionsProcessorStep) and step.enabled
    ]
    absolute_steps = [
        step
        for step in getattr(postprocessor, "steps", [])
        if isinstance(step, SE3AbsoluteActionsProcessorStep) and step.enabled
    ]
    if not relative_steps and not absolute_steps:
        return
    if len(relative_steps) != 1 or len(absolute_steps) != 1:
        raise ValueError(
            "expected exactly one enabled EEF relative/absolute processor pair, got "
            f"{len(relative_steps)}/{len(absolute_steps)}"
        )
    absolute_steps[0].connect(relative_steps[0])
