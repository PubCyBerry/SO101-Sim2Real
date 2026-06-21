"""sim/real 공통 deterministic single-flight chunk executor."""

from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import numpy as np


class ExecutorError(RuntimeError):
    """Chunk 순서·shape·request 상태 오류."""


def prefetch_lead_from_p99(p99_latency_seconds: float, fps: int = 30) -> int:
    if p99_latency_seconds < 0 or fps <= 0:
        raise ValueError("latency는 0 이상, fps는 양수여야 한다")
    return max(8, math.ceil(p99_latency_seconds * fps) + 2)


@dataclass(frozen=True)
class Chunk:
    request_id: int
    start_step: int
    actions: np.ndarray
    inference_latency_ms: float | None = None
    checkpoint_hash: str | None = None

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 6 or actions.shape[0] < 1:
            raise ExecutorError(f"chunk actions shape은 (N, 6)이어야 한다: {actions.shape}")
        if not np.all(np.isfinite(actions)):
            raise ExecutorError("chunk action에 NaN/Inf가 있다")
        object.__setattr__(self, "actions", np.ascontiguousarray(actions))

    @property
    def end_step(self) -> int:
        return self.start_step + len(self.actions)


@dataclass(frozen=True)
class RequestTicket:
    request_id: int
    observation_step: int
    requested_start_step: int
    started_monotonic_ns: int


@dataclass(frozen=True)
class TickResult:
    target: np.ndarray | None
    policy_step: int
    consumed: bool
    underrun: bool
    request_id: int | None
    chunk_offset: int | None


class SingleFlightChunkExecutor:
    """Chunk를 겹쳐 섞지 않고 정확한 경계에서만 교체한다."""

    def __init__(self, *, prefetch_lead: int = 8, request_timeout_ms: int = 2000) -> None:
        if prefetch_lead < 1 or request_timeout_ms <= 0:
            raise ValueError("prefetch_lead는 1 이상, request_timeout_ms는 양수여야 한다")
        self.prefetch_lead = int(prefetch_lead)
        self.request_timeout_ns = int(request_timeout_ms * 1_000_000)
        self.step = 0
        self.loop_tick = 0
        self._next_request_id = 1
        self._in_flight: RequestTicket | None = None
        self._current: Chunk | None = None
        self._next: Chunk | None = None
        self._last_target: np.ndarray | None = None
        self.underruns = 0
        self.stale_responses = 0
        self.timeouts = 0

    @property
    def in_flight(self) -> bool:
        return self._in_flight is not None

    @property
    def in_flight_ticket(self) -> RequestTicket | None:
        return self._in_flight

    @property
    def ready(self) -> bool:
        return self._current is not None and self._current.start_step <= self.step < self._current.end_step

    def seed(self, chunk: Chunk) -> None:
        if self._current is not None or self.step != 0:
            raise ExecutorError("initial chunk는 실행 시작 전에 한 번만 seed할 수 있다")
        if chunk.start_step != 0:
            raise ExecutorError(f"initial chunk start_step은 0이어야 한다: {chunk.start_step}")
        self._current = chunk

    def remaining(self) -> int:
        if not self.ready:
            return 0
        assert self._current is not None
        return self._current.end_step - self.step

    def should_request(self) -> bool:
        if self._in_flight is not None or self._next is not None:
            return False
        if self._current is None:
            return True
        return self.remaining() <= self.prefetch_lead

    def begin_request(self, *, now_ns: int | None = None) -> RequestTicket:
        if not self.should_request():
            raise ExecutorError("현재 상태에서는 새 inference request를 시작할 수 없다")
        requested_start = self.step if self._current is None else self._current.end_step
        ticket = RequestTicket(
            request_id=self._next_request_id,
            observation_step=self.step,
            requested_start_step=requested_start,
            started_monotonic_ns=time.monotonic_ns() if now_ns is None else int(now_ns),
        )
        self._next_request_id += 1
        self._in_flight = ticket
        return ticket

    def request_timed_out(self, *, now_ns: int | None = None) -> bool:
        if self._in_flight is None:
            return False
        current_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
        return current_ns - self._in_flight.started_monotonic_ns >= self.request_timeout_ns

    def fail_request(self, request_id: int, *, timeout: bool = False) -> None:
        if self._in_flight is None or self._in_flight.request_id != request_id:
            raise ExecutorError(f"알 수 없는 request 실패 응답: {request_id}")
        self._in_flight = None
        if timeout:
            self.timeouts += 1

    def accept(self, chunk: Chunk) -> bool:
        ticket = self._in_flight
        if ticket is None or ticket.request_id != chunk.request_id:
            raise ExecutorError(f"request_id가 현재 in-flight와 다르다: {chunk.request_id}")
        self._in_flight = None
        if chunk.start_step != ticket.requested_start_step:
            raise ExecutorError(
                f"chunk start_step 불일치: response={chunk.start_step}, "
                f"requested={ticket.requested_start_step}"
            )
        if chunk.start_step < self.step:
            self.stale_responses += 1
            return False
        if self._current is None:
            self._current = chunk
        else:
            self._next = chunk
        return True

    def tick(self) -> TickResult:
        self.loop_tick += 1
        if self._current is not None and self.step >= self._current.end_step:
            if self._next is not None and self._next.start_step == self.step:
                self._current, self._next = self._next, None
            else:
                self._current = None

        if not self.ready:
            self.underruns += 1
            held = None if self._last_target is None else self._last_target.copy()
            return TickResult(
                target=held,
                policy_step=self.step,
                consumed=False,
                underrun=True,
                request_id=None,
                chunk_offset=None,
            )

        assert self._current is not None
        offset = self.step - self._current.start_step
        target = self._current.actions[offset].copy()
        request_id = self._current.request_id
        policy_step = self.step
        self._last_target = target
        self.step += 1
        return TickResult(
            target=target,
            policy_step=policy_step,
            consumed=True,
            underrun=False,
            request_id=request_id,
            chunk_offset=offset,
        )


class MotionLimiter:
    """공통 velocity/acceleration/jerk 제한기."""

    def __init__(
        self,
        *,
        fps: float,
        max_velocity: np.ndarray,
        max_acceleration: np.ndarray,
        max_jerk: np.ndarray,
        position_gain_per_s: float = 1.0,
    ) -> None:
        self.dt = 1.0 / float(fps)
        if position_gain_per_s <= 0:
            raise ValueError("position_gain_per_s는 양수여야 한다")
        self.position_gain_per_s = float(position_gain_per_s)
        self.max_velocity = self._limits(max_velocity, "velocity")
        self.max_acceleration = self._limits(max_acceleration, "acceleration")
        self.max_jerk = self._limits(max_jerk, "jerk")
        self.position: np.ndarray | None = None
        self.velocity = np.zeros(6, dtype=np.float64)
        self.acceleration = np.zeros(6, dtype=np.float64)

    @staticmethod
    def _limits(values: np.ndarray, name: str) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (6,) or np.any(result <= 0):
            raise ValueError(f"max_{name} shape은 (6,)이고 모두 양수여야 한다")
        return result

    def reset(self, position: np.ndarray) -> None:
        value = np.asarray(position, dtype=np.float64)
        if value.shape != (6,) or not np.all(np.isfinite(value)):
            raise ValueError("limiter 초기 position은 finite (6,) vector여야 한다")
        self.position = value.copy()
        self.velocity.fill(0)
        self.acceleration.fill(0)

    def apply(self, target: np.ndarray) -> np.ndarray:
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (6,) or not np.all(np.isfinite(target)):
            raise ValueError("limiter target은 finite (6,) vector여야 한다")
        if self.position is None:
            self.reset(target)
            return target.astype(np.float32)
        position_error = target - self.position
        desired_velocity = np.clip(
            position_error * self.position_gain_per_s,
            -self.max_velocity,
            self.max_velocity,
        )
        desired_acceleration = np.clip(
            (desired_velocity - self.velocity) / self.dt,
            -self.max_acceleration,
            self.max_acceleration,
        )
        acceleration_delta = np.clip(
            desired_acceleration - self.acceleration,
            -self.max_jerk * self.dt,
            self.max_jerk * self.dt,
        )
        self.acceleration = np.clip(
            self.acceleration + acceleration_delta,
            -self.max_acceleration,
            self.max_acceleration,
        )
        self.velocity = np.clip(
            self.velocity + self.acceleration * self.dt,
            -self.max_velocity,
            self.max_velocity,
        )
        self.position = self.position + self.velocity * self.dt
        return self.position.astype(np.float32)


TObservation = TypeVar("TObservation")


@dataclass(frozen=True)
class InferenceWork(Generic[TObservation]):
    ticket: RequestTicket
    observation: TObservation


@dataclass(frozen=True)
class InferenceOutcome:
    chunk: Chunk | None
    error: BaseException | None


class SingleFlightInferenceWorker(Generic[TObservation]):
    """최대 한 건만 처리하는 background inference thread."""

    def __init__(
        self,
        infer: Callable[[RequestTicket, TObservation], Chunk],
        *,
        name: str = "so101-inference",
    ) -> None:
        self._infer = infer
        self._requests: queue.Queue[InferenceWork[TObservation] | None] = queue.Queue(maxsize=1)
        self._results: queue.Queue[InferenceOutcome] = queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._started = False

    def start(self) -> None:
        if not self._started:
            self._thread.start()
            self._started = True

    def submit(self, ticket: RequestTicket, observation: TObservation) -> None:
        if not self._started:
            self.start()
        try:
            self._requests.put_nowait(InferenceWork(ticket, observation))
        except queue.Full as exc:
            raise ExecutorError("background inference request가 이미 대기 중이다") from exc

    def poll(self) -> InferenceOutcome | None:
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None

    def close(self, timeout: float = 2.0) -> None:
        if not self._started:
            return
        try:
            self._requests.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while True:
            work = self._requests.get()
            if work is None:
                return
            try:
                chunk = self._infer(work.ticket, work.observation)
                outcome = InferenceOutcome(chunk=chunk, error=None)
            except BaseException as exc:  # noqa: BLE001
                outcome = InferenceOutcome(chunk=None, error=exc)
            self._results.put(outcome)
