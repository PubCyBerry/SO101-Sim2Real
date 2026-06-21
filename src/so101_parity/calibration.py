"""SO-101 canonical calibration bundle과 단조 PCHIP 변환."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contract import CANONICAL_SCHEMA, JOINT_ORDER, canonical_json


class CalibrationError(ValueError):
    """Calibration 누락·불일치·범위 오류."""


def _hash_without(raw: Mapping[str, Any], *keys: str) -> str:
    payload = {key: value for key, value in raw.items() if key not in keys}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class MonotonePchip:
    """외부 SciPy 의존성 없는 Fritsch-Carlson monotone cubic interpolation."""

    def __init__(self, x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> None:
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        if self.x.ndim != 1 or self.y.ndim != 1 or len(self.x) != len(self.y) or len(self.x) < 2:
            raise CalibrationError("PCHIP anchor는 동일 길이의 1-D 배열이며 2점 이상이어야 한다")
        if np.any(np.diff(self.x) <= 0):
            raise CalibrationError("PCHIP x anchor는 엄격히 증가해야 한다")
        delta_y = np.diff(self.y)
        if not (np.all(delta_y > 0) or np.all(delta_y < 0)):
            raise CalibrationError("PCHIP y anchor는 엄격히 단조여야 한다")
        self._increasing = bool(self.y[-1] > self.y[0])
        self._d = self._slopes()

    def _slopes(self) -> np.ndarray:
        h = np.diff(self.x)
        delta = np.diff(self.y) / h
        count = len(self.x)
        if count == 2:
            return np.array([delta[0], delta[0]], dtype=np.float64)
        slopes = np.zeros(count, dtype=np.float64)
        for index in range(1, count - 1):
            if delta[index - 1] * delta[index] <= 0:
                slopes[index] = 0.0
            else:
                weight_1 = 2.0 * h[index] + h[index - 1]
                weight_2 = h[index] + 2.0 * h[index - 1]
                slopes[index] = (weight_1 + weight_2) / (
                    weight_1 / delta[index - 1] + weight_2 / delta[index]
                )
        slopes[0] = self._endpoint_slope(h[0], h[1], delta[0], delta[1])
        slopes[-1] = self._endpoint_slope(h[-1], h[-2], delta[-1], delta[-2])
        return slopes

    @staticmethod
    def _endpoint_slope(h0: float, h1: float, delta0: float, delta1: float) -> float:
        slope = ((2.0 * h0 + h1) * delta0 - h0 * delta1) / (h0 + h1)
        if slope * delta0 <= 0:
            return 0.0
        if delta0 * delta1 < 0 and abs(slope) > abs(3.0 * delta0):
            return 3.0 * delta0
        return slope

    def __call__(self, values: float | np.ndarray, *, clamp: bool = False) -> np.ndarray:
        query = np.asarray(values, dtype=np.float64)
        if clamp:
            query = np.clip(query, self.x[0], self.x[-1])
        elif np.any((query < self.x[0]) | (query > self.x[-1])):
            raise CalibrationError(
                f"interpolation 범위 이탈: [{query.min()}, {query.max()}] not in "
                f"[{self.x[0]}, {self.x[-1]}]"
            )
        indices = np.searchsorted(self.x, query, side="right") - 1
        indices = np.clip(indices, 0, len(self.x) - 2)
        h = self.x[indices + 1] - self.x[indices]
        t = (query - self.x[indices]) / h
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        return (
            h00 * self.y[indices]
            + h10 * h * self._d[indices]
            + h01 * self.y[indices + 1]
            + h11 * h * self._d[indices + 1]
        )

    def inverse(self) -> "_InverseMonotonePchip":
        return _InverseMonotonePchip(self)


class _InverseMonotonePchip:
    """원 PCHIP 곡선을 이분 탐색해 수치적으로 정확히 역변환한다."""

    def __init__(self, forward: MonotonePchip) -> None:
        self.forward = forward

    def __call__(self, values: float | np.ndarray, *, clamp: bool = False) -> np.ndarray:
        query = np.asarray(values, dtype=np.float64)
        y_min = min(self.forward.y[0], self.forward.y[-1])
        y_max = max(self.forward.y[0], self.forward.y[-1])
        if clamp:
            query = np.clip(query, y_min, y_max)
        elif np.any((query < y_min) | (query > y_max)):
            raise CalibrationError(
                f"inverse interpolation 범위 이탈: [{query.min()}, {query.max()}] "
                f"not in [{y_min}, {y_max}]"
            )
        low = np.full(query.shape, self.forward.x[0], dtype=np.float64)
        high = np.full(query.shape, self.forward.x[-1], dtype=np.float64)
        for _ in range(52):
            middle = (low + high) * 0.5
            evaluated = self.forward(middle)
            go_right = evaluated < query if self.forward._increasing else evaluated > query
            low = np.where(go_right, middle, low)
            high = np.where(go_right, high, middle)
        return (low + high) * 0.5


@dataclass(frozen=True)
class ArmAffine:
    sign: float
    offset_rad: float
    real_deg_min: float
    real_deg_max: float

    def to_canonical(self, real_deg: float | np.ndarray) -> np.ndarray:
        return self.sign * np.asarray(real_deg) * (math.pi / 180.0) + self.offset_rad

    def from_canonical(self, canonical_rad: float | np.ndarray) -> np.ndarray:
        return (np.asarray(canonical_rad) - self.offset_rad) * (180.0 / math.pi) / self.sign


class CalibrationBundle:
    def __init__(self, raw: Mapping[str, Any]) -> None:
        supplied_hash = raw.get("calibration_hash")
        expected_hash = _hash_without(raw, "calibration_hash")
        if supplied_hash and supplied_hash != expected_hash:
            raise CalibrationError(
                f"calibration hash 불일치: supplied={supplied_hash}, expected={expected_hash}"
            )
        if raw.get("schema") != "so101-canonical-calibration-v1":
            raise CalibrationError(f"지원하지 않는 calibration schema: {raw.get('schema')!r}")
        if raw.get("policy_io_schema") != CANONICAL_SCHEMA:
            raise CalibrationError("calibration policy_io_schema가 canonical contract와 다르다")
        if tuple(raw.get("joint_order", [])) != JOINT_ORDER:
            raise CalibrationError("calibration joint_order가 canonical contract와 다르다")

        self.raw = dict(raw)
        self.calibration_hash = expected_hash
        self.validated = bool(raw.get("validated", False))
        self.calibration_id = str(raw.get("calibration_id", "unknown"))
        arm_raw = raw.get("arm", {})
        self.arm = tuple(
            ArmAffine(
                sign=float(arm_raw[name]["sign"]),
                offset_rad=float(arm_raw[name]["offset_rad"]),
                real_deg_min=float(arm_raw[name]["real_deg_range"][0]),
                real_deg_max=float(arm_raw[name]["real_deg_range"][1]),
            )
            for name in JOINT_ORDER[:5]
        )
        for name, affine in zip(JOINT_ORDER[:5], self.arm):
            if affine.sign not in (-1.0, 1.0):
                raise CalibrationError(f"{name} sign은 -1 또는 +1이어야 한다")
        gripper = raw.get("gripper", {})
        self._real_gripper = self._curve(gripper.get("real"), "real")
        self._sim_gripper = self._curve(gripper.get("sim"), "sim")

        motor = raw.get("motor_profile", {})
        expected_motor = motor.get("expected", {})
        self.motor_profile_hash = _hash_without(expected_motor) if expected_motor else ""
        supplied_motor_hash = motor.get("profile_hash")
        if supplied_motor_hash and supplied_motor_hash != self.motor_profile_hash:
            raise CalibrationError(
                f"motor profile hash 불일치: supplied={supplied_motor_hash}, "
                f"expected={self.motor_profile_hash}"
            )
        self.motor_profile_validated = bool(motor.get("readback_validated", False))

    @property
    def has_real_gripper_curve(self) -> bool:
        return self._real_gripper is not None

    @property
    def has_sim_gripper_curve(self) -> bool:
        return self._sim_gripper is not None

    @staticmethod
    def _curve(raw: Mapping[str, Any] | None, domain: str) -> MonotonePchip | None:
        if not raw or not raw.get("native") or not raw.get("aperture_mm"):
            return None
        if len(raw["native"]) != len(raw["aperture_mm"]):
            raise CalibrationError(f"{domain} gripper anchor 길이가 다르다")
        return MonotonePchip(raw["native"], raw["aperture_mm"])

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationBundle":
        with Path(path).open("r", encoding="utf-8") as stream:
            return cls(json.load(stream))

    def require_validated(self, *, require_motor_profile: bool = True) -> None:
        if not self.validated:
            raise CalibrationError(
                f"{self.calibration_id} calibration은 실측 검증 전이다. 실기기 motion을 차단한다."
            )
        if self._real_gripper is None or self._sim_gripper is None:
            raise CalibrationError("validated calibration에 real/sim gripper curve가 없다")
        if require_motor_profile and not self.motor_profile_validated:
            raise CalibrationError("motor profile EEPROM readback 검증 전이다. 실기기 motion을 차단한다")

    @staticmethod
    def _vector(values: np.ndarray | list[float]) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (6,):
            raise CalibrationError(f"SO-101 vector shape은 (6,)이어야 한다: {result.shape}")
        if not np.all(np.isfinite(result)):
            raise CalibrationError("SO-101 vector에 NaN/Inf가 있다")
        return result

    def real_to_canonical(self, values: np.ndarray | list[float], *, clamp: bool = False) -> np.ndarray:
        native = self._vector(values)
        result = np.empty(6, dtype=np.float64)
        for index, affine in enumerate(self.arm):
            value = native[index]
            if clamp:
                value = float(np.clip(value, affine.real_deg_min, affine.real_deg_max))
            elif not affine.real_deg_min <= value <= affine.real_deg_max:
                raise CalibrationError(f"{JOINT_ORDER[index]} real degree 범위 이탈: {value}")
            result[index] = affine.to_canonical(value)
        if self._real_gripper is None:
            raise CalibrationError("real gripper aperture calibration이 없다")
        result[5] = self._real_gripper(native[5], clamp=clamp)
        return result.astype(np.float32)

    def canonical_to_real(self, values: np.ndarray | list[float], *, clamp: bool = False) -> np.ndarray:
        canonical = self._vector(values)
        result = np.empty(6, dtype=np.float64)
        for index, affine in enumerate(self.arm):
            result[index] = affine.from_canonical(canonical[index])
            if clamp:
                result[index] = np.clip(result[index], affine.real_deg_min, affine.real_deg_max)
            elif not affine.real_deg_min <= result[index] <= affine.real_deg_max:
                raise CalibrationError(f"{JOINT_ORDER[index]} canonical target이 real 범위를 벗어난다")
        if self._real_gripper is None:
            raise CalibrationError("real gripper aperture calibration이 없다")
        result[5] = self._real_gripper.inverse()(canonical[5], clamp=clamp)
        return result.astype(np.float32)

    def sim_to_canonical(self, values: np.ndarray | list[float], *, clamp: bool = False) -> np.ndarray:
        native = self._vector(values)
        result = native.copy()
        if self._sim_gripper is None:
            raise CalibrationError("sim gripper aperture calibration이 없다")
        result[5] = self._sim_gripper(native[5], clamp=clamp)
        return result.astype(np.float32)

    def canonical_to_sim(self, values: np.ndarray | list[float], *, clamp: bool = False) -> np.ndarray:
        canonical = self._vector(values)
        result = canonical.copy()
        if self._sim_gripper is None:
            raise CalibrationError("sim gripper aperture calibration이 없다")
        result[5] = self._sim_gripper.inverse()(canonical[5], clamp=clamp)
        return result.astype(np.float32)

    @staticmethod
    def fit_arm_affine(
        canonical_rad: np.ndarray,
        real_deg: np.ndarray,
    ) -> dict[str, dict[str, float]]:
        canonical = np.asarray(canonical_rad, dtype=np.float64)
        real = np.asarray(real_deg, dtype=np.float64)
        if canonical.shape != real.shape or canonical.ndim != 2 or canonical.shape[1] != 5:
            raise CalibrationError("paired arm pose shape은 canonical/real 모두 (N, 5)여야 한다")
        if canonical.shape[0] < 3:
            raise CalibrationError("arm affine fitting에는 pose 3개 이상이 필요하다")
        real_rad = np.deg2rad(real)
        fitted: dict[str, dict[str, float]] = {}
        for index, name in enumerate(JOINT_ORDER[:5]):
            candidates = []
            for sign in (-1.0, 1.0):
                offset = float(np.mean(canonical[:, index] - sign * real_rad[:, index]))
                residual = canonical[:, index] - (sign * real_rad[:, index] + offset)
                rmse = float(np.sqrt(np.mean(residual**2)))
                candidates.append((rmse, sign, offset))
            rmse, sign, offset = min(candidates)
            fitted[name] = {"sign": sign, "offset_rad": offset, "fit_rmse_rad": rmse}
        return fitted

    @staticmethod
    def with_hash(raw: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(raw)
        motor = dict(result.get("motor_profile", {}))
        expected = motor.get("expected", {})
        if expected:
            motor["profile_hash"] = _hash_without(expected)
            result["motor_profile"] = motor
        result["calibration_hash"] = _hash_without(result, "calibration_hash")
        return result
