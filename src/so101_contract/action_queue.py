"""LeRobot 0.4.4 RobotClient와 같은 action chunk queue semantics."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
import threading

import numpy as np

ACTION_AGGREGATE_NAMES = ("weighted_average", "latest_only", "average", "conservative")

_AGGREGATE_WEIGHTS = {
    "weighted_average": (0.3, 0.7),
    "latest_only": (0.0, 1.0),
    "average": (0.5, 0.5),
    "conservative": (0.7, 0.3),
}


def aggregate_actions(old: np.ndarray, new: np.ndarray, name: str) -> np.ndarray:
    """Upstream `AGGREGATE_FUNCTIONS`와 동일한 두 action 결합."""
    if name not in _AGGREGATE_WEIGHTS:
        raise ValueError(f"unknown aggregate function {name!r}; expected one of {ACTION_AGGREGATE_NAMES}")
    old_array = np.asarray(old, dtype=np.float32)
    new_array = np.asarray(new, dtype=np.float32)
    if old_array.shape != new_array.shape:
        raise ValueError(f"action shape mismatch: old={old_array.shape}, new={new_array.shape}")
    old_weight, new_weight = _AGGREGATE_WEIGHTS[name]
    return (old_weight * old_array + new_weight * new_array).astype(np.float32)


class ActionChunkQueue:
    """Thread-safe numpy queue matching LeRobot `RobotClient`.

    incoming chunk를 받을 때 현재 queue와 timestep이 겹치는 action만 aggregate하고,
    실행이 끝난 timestep 이하의 action은 버린다. `must_go`와 refill 판단도 upstream
    client의 상태 전이를 따른다.
    """

    def __init__(self, aggregate_fn_name: str = "weighted_average") -> None:
        if aggregate_fn_name not in ACTION_AGGREGATE_NAMES:
            raise ValueError(
                f"unknown aggregate function {aggregate_fn_name!r}; "
                f"expected one of {ACTION_AGGREGATE_NAMES}"
            )
        self.aggregate_fn_name = aggregate_fn_name
        self.latest_action = -1
        self.action_chunk_size = -1
        self._must_go = True
        self._actions: OrderedDict[int, np.ndarray] = OrderedDict()
        self._lock = threading.RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._actions)

    def has_actions(self) -> bool:
        return len(self) > 0

    def timesteps(self) -> list[int]:
        with self._lock:
            return list(self._actions)

    def ready_to_send_observation(self, chunk_size_threshold: float) -> bool:
        if not 0.0 <= chunk_size_threshold <= 1.0:
            raise ValueError("chunk_size_threshold must be in [0, 1]")
        with self._lock:
            if self.action_chunk_size <= 0:
                return True
            return len(self._actions) / self.action_chunk_size <= chunk_size_threshold

    def observation_timestep(self) -> int:
        with self._lock:
            return max(self.latest_action, 0)

    def observation_must_go(self) -> bool:
        with self._lock:
            return self._must_go and not self._actions

    def mark_observation_sent(self, must_go: bool) -> None:
        if not must_go:
            return
        with self._lock:
            self._must_go = False

    def mark_request_failed(self) -> None:
        with self._lock:
            self._must_go = True

    def merge(self, incoming_actions: Iterable[tuple[int, np.ndarray]]) -> None:
        incoming = [(int(timestep), np.asarray(action, dtype=np.float32)) for timestep, action in incoming_actions]
        if not incoming:
            return

        with self._lock:
            self.action_chunk_size = max(self.action_chunk_size, len(incoming))
            current = dict(self._actions)
            future: dict[int, np.ndarray] = {}
            for timestep, action in incoming:
                if timestep <= self.latest_action:
                    continue
                if timestep in current:
                    action = aggregate_actions(current[timestep], action, self.aggregate_fn_name)
                future[timestep] = action.astype(np.float32)

            self._actions = OrderedDict(sorted(future.items()))
            self._must_go = True

    def pop_next(self) -> tuple[int, np.ndarray]:
        with self._lock:
            if not self._actions:
                raise IndexError("action queue is empty")
            timestep, action = self._actions.popitem(last=False)
            self.latest_action = timestep
            return timestep, action.copy()
