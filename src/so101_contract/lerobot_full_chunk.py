"""LeRobot EEF-relative policy용 synchronous full-chunk inference runner.

``select_action()`` 내부 queue를 사용하면 다음 control tick의 observation이 preprocessor
cache를 덮어써 남은 relative action이 다른 state에 re-anchor될 수 있다. 이 runner는
queue refill 시에만 ``predict_action_chunk()``를 호출하고 full chunk를 정확히 한 번
postprocess한 뒤 absolute action FIFO를 소비한다.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import torch

from .lerobot_policy_integration import has_eef_relative_processor_steps
from .lerobot_v2_integration import has_action_representation_steps

FULL_CHUNK_INFERENCE_VERSION = "so101_eef_full_chunk_runner_v1"


@dataclass(frozen=True)
class FullChunkInferenceMetrics:
    chunks_predicted: int
    preprocessor_calls: int
    postprocessor_calls: int
    actions_dequeued: int


class FullChunkPolicyRunner:
    """Batch-preserving full-chunk FIFO. queue 중에는 새 observation을 preprocess하지 않는다."""

    def __init__(
        self,
        policy: Any,
        preprocessor: Any,
        postprocessor: Any,
        *,
        execution_horizon: int | None = None,
        require_eef_relative: bool = True,
    ) -> None:
        # schema v2 step(4 mode 공통) 또는 v1 SE(3) step 중 하나가 있으면 full-chunk 경로다.
        has_relative = has_action_representation_steps(
            preprocessor, postprocessor
        ) or has_eef_relative_processor_steps(preprocessor, postprocessor)
        if require_eef_relative and not has_relative:
            raise ValueError(
                "full-chunk runner requires a serialized action representation processor pair"
            )
        config = getattr(policy, "config", None)
        if has_relative:
            if getattr(config, "temporal_ensemble_coeff", None) is not None:
                raise NotImplementedError("ACT temporal ensemble is disabled for EEF-relative v1")
            if getattr(config, "rtc_config", None) is not None:
                raise NotImplementedError("RTC is disabled for EEF-relative v1")
            if getattr(config, "use_relative_actions", False):
                raise ValueError("GR00T legacy and common EEF-relative modes cannot be combined")

        if execution_horizon is None:
            execution_horizon = getattr(config, "n_action_steps", None)
        if execution_horizon is not None and (
            not isinstance(execution_horizon, int) or execution_horizon <= 0
        ):
            raise ValueError(f"execution_horizon must be positive, got {execution_horizon!r}")

        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.execution_horizon = execution_horizon
        self._queue: deque[torch.Tensor] = deque()
        self._chunks_predicted = 0
        self._preprocessor_calls = 0
        self._postprocessor_calls = 0
        self._actions_dequeued = 0
        self._chunk_anchor_state: torch.Tensor | None = None

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    @property
    def chunk_anchor_state(self) -> torch.Tensor | None:
        return self._chunk_anchor_state

    @property
    def metrics(self) -> FullChunkInferenceMetrics:
        return FullChunkInferenceMetrics(
            chunks_predicted=self._chunks_predicted,
            preprocessor_calls=self._preprocessor_calls,
            postprocessor_calls=self._postprocessor_calls,
            actions_dequeued=self._actions_dequeued,
        )

    def clear_queue(self) -> None:
        self._queue.clear()
        self._chunk_anchor_state = None

    def reset(self) -> None:
        self.clear_queue()
        for component in (self.policy, self.preprocessor, self.postprocessor):
            reset = getattr(component, "reset", None)
            if callable(reset):
                reset()

    def refill(self, observation: dict[str, Any]) -> torch.Tensor:
        """빈 queue를 새 chunk로 채우고 postprocessed full chunk를 반환."""
        if self._queue:
            raise RuntimeError("cannot refill a non-empty full-chunk queue")
        processed_observation = self.preprocessor(observation)
        self._preprocessor_calls += 1
        state = processed_observation.get("observation.state")
        if isinstance(state, torch.Tensor):
            self._chunk_anchor_state = state.detach().clone()
        else:
            self._chunk_anchor_state = None

        chunk = self.policy.predict_action_chunk(processed_observation)
        if not isinstance(chunk, torch.Tensor) or chunk.ndim != 3:
            raise ValueError(
                "predict_action_chunk must return torch.Tensor (B,H,D), got "
                f"{type(chunk).__name__} {getattr(chunk, 'shape', None)}"
            )
        self._chunks_predicted += 1
        absolute_chunk = self.postprocessor(chunk)
        self._postprocessor_calls += 1
        if not isinstance(absolute_chunk, torch.Tensor) or absolute_chunk.shape != chunk.shape:
            raise ValueError(
                "full-chunk postprocessor must preserve (B,H,D), got "
                f"{getattr(absolute_chunk, 'shape', None)} for input {tuple(chunk.shape)}"
            )
        horizon = absolute_chunk.shape[1]
        if self.execution_horizon is not None:
            horizon = min(horizon, self.execution_horizon)
        if horizon <= 0:
            raise ValueError("policy returned an empty action horizon")
        self._queue.extend(absolute_chunk[:, index, :] for index in range(horizon))
        return absolute_chunk

    def next_action(self, observation: dict[str, Any] | None) -> torch.Tensor:
        """FIFO가 비었을 때만 observation을 사용해 refill하고 다음 ``(B,D)``를 반환."""
        if not self._queue:
            if observation is None:
                raise ValueError("observation is required when the full-chunk queue is empty")
            self.refill(observation)
        action = self._queue.popleft()
        self._actions_dequeued += 1
        return action
