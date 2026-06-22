"""Sim/real adapter가 공유하는 canonical 30 Hz 실행 loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from .executor import (
    Chunk,
    MotionLimiter,
    RequestTicket,
    SingleFlightChunkExecutor,
    SingleFlightInferenceWorker,
)
from .trace import JsonlTraceWriter


@dataclass(frozen=True)
class CanonicalObservation:
    state: np.ndarray
    images: dict[str, np.ndarray]

    def __post_init__(self) -> None:
        state = np.asarray(self.state, dtype=np.float32)
        if state.shape != (6,) or not np.all(np.isfinite(state)):
            raise ValueError("canonical observation state는 finite (6,)이어야 한다")
        expected = {"top", "wrist", "front"}
        if set(self.images) != expected:
            raise ValueError(f"camera key는 {sorted(expected)}여야 한다")
        checked = {}
        for name in sorted(expected):
            image = np.asarray(self.images[name])
            if image.shape != (480, 640, 3) or image.dtype != np.uint8:
                raise ValueError(
                    f"{name} image는 uint8 (480,640,3)이어야 한다: {image.shape} {image.dtype}"
                )
            checked[name] = np.ascontiguousarray(image)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "images", checked)


class RuntimeAdapter(Protocol):
    domain: str

    def read_state(self) -> np.ndarray: ...

    def capture(self, state: np.ndarray | None = None) -> CanonicalObservation: ...

    def canonical_to_native(self, target: np.ndarray) -> np.ndarray: ...

    def advance(self, native_command: np.ndarray) -> None: ...

    def safe_stop(self, reason: str) -> None: ...


InferChunk = Callable[[RequestTicket, CanonicalObservation], Chunk]


@dataclass(frozen=True)
class RuntimeHashes:
    contract_hash: str
    runtime_manifest_hash: str
    checkpoint_hash: str
    calibration_hash: str
    motor_profile_hash: str


class CanonicalRuntime:
    """두 client가 공유하는 deterministic chunk 소비와 trace 경로."""

    def __init__(
        self,
        *,
        adapter: RuntimeAdapter,
        infer: InferChunk,
        limiter: MotionLimiter,
        hashes: RuntimeHashes,
        trace: JsonlTraceWriter,
        prefetch_lead: int,
        request_timeout_ms: int,
    ) -> None:
        self.adapter = adapter
        self.infer = infer
        self.limiter = limiter
        self.hashes = hashes
        self.trace = trace
        self.executor = SingleFlightChunkExecutor(
            prefetch_lead=prefetch_lead,
            request_timeout_ms=request_timeout_ms,
        )
        self.worker = SingleFlightInferenceWorker(infer)
        self._last_inference_latency_ms: float | None = None
        self.observation_captures = 0

    def _capture_observation(
        self,
        state: np.ndarray | None = None,
    ) -> CanonicalObservation:
        self.observation_captures += 1
        return self.adapter.capture(state=state)

    def move_home(
        self,
        target: np.ndarray,
        *,
        max_steps: int = 300,
        tolerance: np.ndarray | None = None,
    ) -> CanonicalObservation:
        home = np.asarray(target, dtype=np.float32)
        if home.shape != (6,):
            raise ValueError("home target shape은 (6,)이어야 한다")
        state = self.adapter.read_state()
        self.limiter.reset(state)
        tolerance_value = (
            np.asarray(tolerance, dtype=np.float32)
            if tolerance is not None
            else np.array([np.deg2rad(5.0)] * 5 + [2.0], dtype=np.float32)
        )
        for _ in range(max_steps):
            limited = self.limiter.apply(home)
            native = self.adapter.canonical_to_native(limited)
            self.adapter.advance(native)
            state = self.adapter.read_state()
            if np.all(np.abs(state - home) <= tolerance_value):
                # Dynamics/backend speed can change which control tick first
                # satisfies the home tolerance. Re-anchor limiter state to the
                # exact canonical home so subsequent policy targets remain
                # bitwise identical across sim/real and CPU/GPU PhysX.
                self.limiter.reset(home)
                # Prime inference needs one synchronized RGB observation. During
                # homing, images are otherwise unused and are intentionally not
                # copied every 30 Hz control tick.
                return self._capture_observation(state=state)
        error = state - home
        raise TimeoutError(
            "canonical home target 도달 timeout: "
            f"state={state.tolist()}, error={error.tolist()}"
        )

    def prime(self, observation: CanonicalObservation | None = None) -> None:
        if observation is None:
            observation = self._capture_observation()
        ticket = self.executor.begin_request()
        try:
            chunk = self.infer(ticket, observation)
        except BaseException:
            self.executor.fail_request(ticket.request_id)
            raise
        self.executor.accept(chunk)
        self._last_inference_latency_ms = chunk.inference_latency_ms

    def run(
        self,
        steps: int | None,
        *,
        should_continue: Callable[[], bool] | None = None,
    ) -> None:
        """정해진 step 수 또는 외부 app이 종료될 때까지 실행한다.

        ``steps=None``은 livestream처럼 사용자가 원격 GUI에서 종료할 때까지
        같은 executor/worker를 유지해야 하는 경로에 사용한다.
        """
        if steps is not None and steps < 1:
            raise ValueError("steps는 1 이상이거나 None이어야 한다")
        if not self.executor.ready:
            self.prime()
        self.worker.start()
        completed = 0
        try:
            while steps is None or completed < steps:
                if should_continue is not None and not should_continue():
                    break
                self._poll_inference()
                if self.executor.request_timed_out():
                    ticket = self.executor.in_flight_ticket
                    assert ticket is not None
                    self.executor.fail_request(ticket.request_id, timeout=True)
                    self.adapter.safe_stop("inference_timeout")
                    self._trace_timeout(ticket.request_id)
                    raise TimeoutError(f"inference request {ticket.request_id} timeout")

                state = self.adapter.read_state()
                if self.executor.should_request():
                    ticket = self.executor.begin_request()
                    observation = self._capture_observation(state=state)
                    self.worker.submit(ticket, observation)

                tick = self.executor.tick()
                if tick.target is None:
                    self.adapter.safe_stop("executor_without_hold_target")
                    raise RuntimeError("executor가 hold할 canonical target이 없다")
                limited = self.limiter.apply(tick.target)
                native = self.adapter.canonical_to_native(limited)
                self.adapter.advance(native)
                self.trace.write(
                    loop_tick=self.executor.loop_tick,
                    policy_step=tick.policy_step,
                    request_id=tick.request_id,
                    chunk_offset=tick.chunk_offset,
                    raw_model_output=tick.target.tolist(),
                    limited_canonical_target=limited.tolist(),
                    native_command=np.asarray(native, dtype=np.float32).tolist(),
                    measured_canonical_state=state.tolist(),
                    inference_latency_ms=self._last_inference_latency_ms,
                    hold=not tick.consumed,
                    underrun=tick.underrun,
                    timeout=False,
                    contract_hash=self.hashes.contract_hash,
                    runtime_manifest_hash=self.hashes.runtime_manifest_hash,
                    checkpoint_hash=self.hashes.checkpoint_hash,
                    calibration_hash=self.hashes.calibration_hash,
                    motor_profile_hash=self.hashes.motor_profile_hash,
                    domain=self.adapter.domain,
                )
                completed += 1
        finally:
            self.worker.close()

    def _poll_inference(self) -> None:
        outcome = self.worker.poll()
        if outcome is None:
            return
        ticket = self.executor.in_flight_ticket
        if outcome.error is not None:
            if ticket is not None:
                self.executor.fail_request(ticket.request_id)
            raise outcome.error
        assert outcome.chunk is not None
        if ticket is None or ticket.request_id != outcome.chunk.request_id:
            return
        self.executor.accept(outcome.chunk)
        self._last_inference_latency_ms = outcome.chunk.inference_latency_ms

    def _trace_timeout(self, request_id: int) -> None:
        self.trace.write(
            loop_tick=self.executor.loop_tick,
            policy_step=self.executor.step,
            request_id=request_id,
            chunk_offset=None,
            raw_model_output=None,
            limited_canonical_target=None,
            native_command=None,
            measured_canonical_state=None,
            inference_latency_ms=None,
            hold=True,
            underrun=False,
            timeout=True,
            contract_hash=self.hashes.contract_hash,
            runtime_manifest_hash=self.hashes.runtime_manifest_hash,
            checkpoint_hash=self.hashes.checkpoint_hash,
            calibration_hash=self.hashes.calibration_hash,
            motor_profile_hash=self.hashes.motor_profile_hash,
            domain=self.adapter.domain,
        )
