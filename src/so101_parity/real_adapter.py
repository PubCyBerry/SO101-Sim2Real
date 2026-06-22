"""Windows native Feetech bus/camera ↔ canonical runtime adapter."""

from __future__ import annotations

import time

import numpy as np

from .calibration import CalibrationBundle
from .contract import JOINT_ORDER
from .runtime import CanonicalObservation


class RealSO101Adapter:
    domain = "real"

    def __init__(
        self,
        *,
        bus,
        cameras: dict,
        calibration: CalibrationBundle,
        enable_motion: bool,
        fps: int = 30,
    ) -> None:
        self.bus = bus
        self.cameras = cameras
        self.calibration = calibration
        self.enable_motion = enable_motion
        self.period_s = 1.0 / fps
        self._next_deadline = time.perf_counter()

    def read_state(self) -> np.ndarray:
        positions = self.bus.sync_read("Present_Position")
        native = np.asarray([positions[name] for name in JOINT_ORDER], dtype=np.float32)
        return self.calibration.real_to_canonical(native, clamp=False)

    def capture(self, state: np.ndarray | None = None) -> CanonicalObservation:
        if state is None:
            state = self.read_state()
        images = {}
        for name in ("top", "wrist", "front"):
            image = np.asarray(self.cameras[name].read_latest())
            # Camera backends commonly recycle their latest-frame buffer. Own
            # the request snapshot before the background inference thread uses it.
            images[name] = np.array(image, dtype=np.uint8, order="C", copy=True)
        return CanonicalObservation(state=state, images=images)

    def canonical_to_native(self, target: np.ndarray) -> np.ndarray:
        return self.calibration.canonical_to_real(target, clamp=False)

    def advance(self, native_command: np.ndarray) -> None:
        if not self.enable_motion:
            raise RuntimeError("real motion은 --enable-motion 없이 차단된다")
        native = np.asarray(native_command, dtype=np.float32)
        if native.shape != (6,):
            raise ValueError(f"real native command shape은 (6,)이어야 한다: {native.shape}")
        values = {name: float(native[index]) for index, name in enumerate(JOINT_ORDER)}
        # Real sink의 유일한 motion write 경로.
        self.bus.sync_write("Goal_Position", values)
        self._next_deadline += self.period_s
        time.sleep(max(0.0, self._next_deadline - time.perf_counter()))

    def safe_stop(self, reason: str) -> None:
        del reason
        if self.enable_motion:
            self.bus.disable_torque()
            self.enable_motion = False
