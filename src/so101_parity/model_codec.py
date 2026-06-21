"""Checkpoint model-native state/action과 canonical vector 사이 codec."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .calibration import CalibrationBundle, CalibrationError

ModelFrame = Literal["canonical", "sim_legacy_rad_scale_v1", "real_lerobot_range_v1"]


class ModelCodecError(ValueError):
    """Model frame provenance가 없거나 지원되지 않을 때 발생한다."""


@dataclass(frozen=True)
class ModelCodec:
    frame: ModelFrame
    calibration: CalibrationBundle
    legacy_gripper_scale: float = 31.75

    @staticmethod
    def _vector(values: np.ndarray | list[float]) -> np.ndarray:
        result = np.asarray(values, dtype=np.float32)
        if result.shape != (6,) or not np.all(np.isfinite(result)):
            raise ModelCodecError(f"model/canonical vector는 finite (6,)이어야 한다: {result.shape}")
        return result

    def canonical_to_model(self, values: np.ndarray | list[float]) -> np.ndarray:
        canonical = self._vector(values)
        if self.frame == "canonical":
            return canonical.copy()
        if self.frame == "sim_legacy_rad_scale_v1":
            sim_native = self.calibration.canonical_to_sim(canonical)
            result = sim_native.copy()
            result[:5] *= 180.0 / math.pi
            result[5] *= self.legacy_gripper_scale
            return result.astype(np.float32)
        if self.frame == "real_lerobot_range_v1":
            return self.calibration.canonical_to_real(canonical)
        raise ModelCodecError(f"지원하지 않는 model frame: {self.frame!r}")

    def model_to_canonical(self, values: np.ndarray | list[float]) -> np.ndarray:
        model = self._vector(values)
        if self.frame == "canonical":
            return model.copy()
        if self.frame == "sim_legacy_rad_scale_v1":
            sim_native = model.copy()
            sim_native[:5] *= math.pi / 180.0
            sim_native[5] /= self.legacy_gripper_scale
            return self.calibration.sim_to_canonical(sim_native)
        if self.frame == "real_lerobot_range_v1":
            return self.calibration.real_to_canonical(model)
        raise ModelCodecError(f"지원하지 않는 model frame: {self.frame!r}")

    def canonical_chunk_to_model(self, chunk: np.ndarray) -> np.ndarray:
        values = np.asarray(chunk, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 6:
            raise ModelCodecError(f"chunk shape은 (N, 6)이어야 한다: {values.shape}")
        return np.stack([self.canonical_to_model(row) for row in values])

    def model_chunk_to_canonical(self, chunk: np.ndarray) -> np.ndarray:
        values = np.asarray(chunk, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 6:
            raise ModelCodecError(f"chunk shape은 (N, 6)이어야 한다: {values.shape}")
        try:
            return np.stack([self.model_to_canonical(row) for row in values])
        except CalibrationError as exc:
            raise ModelCodecError(str(exc)) from exc
