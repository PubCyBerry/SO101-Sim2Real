"""Canonical dynamics excitation 생성과 trace 기반 응답 식별."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from .contract import JOINT_ORDER


@dataclass(frozen=True)
class ExcitationRow:
    step: int
    time_s: float
    phase: str
    condition: str
    target: tuple[float, ...]


def _append_segment(
    rows: list[ExcitationRow],
    targets: Iterable[np.ndarray],
    *,
    fps: int,
    phase: str,
    condition: str,
) -> None:
    for target in targets:
        step = len(rows)
        rows.append(
            ExcitationRow(
                step=step,
                time_s=step / fps,
                phase=phase,
                condition=condition,
                target=tuple(float(value) for value in target),
            )
        )


def build_excitation_plan(
    home: np.ndarray,
    *,
    condition: str,
    fps: int = 30,
) -> list[ExcitationRow]:
    """READY pose 기준 hold/step/ramp/triangle/multisine/compound/gripper sweep."""

    base = np.asarray(home, dtype=np.float64)
    if base.shape != (6,):
        raise ValueError(f"home shape은 (6,)이어야 한다: {base.shape}")
    rows: list[ExcitationRow] = []

    def hold(target: np.ndarray, seconds: float, phase: str) -> None:
        _append_segment(
            rows,
            (target.copy() for _ in range(round(seconds * fps))),
            fps=fps,
            phase=phase,
            condition=condition,
        )

    hold(base, 3.0, "initial_hold")
    for joint in range(5):
        for amplitude_deg in (5.0, -5.0, 10.0, -10.0):
            target = base.copy()
            target[joint] += math.radians(amplitude_deg)
            hold(target, 1.5, f"step_{JOINT_ORDER[joint]}_{amplitude_deg:+g}deg")
            hold(base, 1.5, f"step_{JOINT_ORDER[joint]}_return_hold")

    for joint in range(5):
        for sign in (1.0, -1.0):
            count = round(2.0 * fps)
            targets = []
            for index in range(count):
                alpha = (index + 1) / count
                target = base.copy()
                target[joint] += sign * math.radians(10.0) * alpha
                targets.append(target)
            _append_segment(
                rows,
                targets,
                fps=fps,
                phase=f"ramp_{JOINT_ORDER[joint]}_{sign:+g}",
                condition=condition,
            )
            hold(base, 1.5, f"ramp_{JOINT_ORDER[joint]}_return_hold")

    triangle_seconds = 6.0
    count = round(triangle_seconds * fps)
    for joint in range(5):
        targets = []
        for index in range(count):
            phase = (index / fps) / 3.0
            triangle = 2.0 * abs(2.0 * (phase - math.floor(phase + 0.5))) - 1.0
            target = base.copy()
            target[joint] += math.radians(15.0) * triangle
            targets.append(target)
        _append_segment(
            rows,
            targets,
            fps=fps,
            phase=f"triangle_{JOINT_ORDER[joint]}_15deg",
            condition=condition,
        )
        hold(base, 2.0, f"triangle_{JOINT_ORDER[joint]}_return_hold")

    multisine_seconds = 12.0
    frequencies = (0.2, 0.35, 0.55, 0.8, 1.1, 1.5)
    targets = []
    for index in range(round(multisine_seconds * fps)):
        time_s = index / fps
        target = base.copy()
        for joint in range(5):
            target[joint] += math.radians(4.0) * math.sin(
                2.0 * math.pi * frequencies[joint] * time_s + joint * 0.7
            )
        target[5] += 8.0 * math.sin(2.0 * math.pi * frequencies[5] * time_s)
        targets.append(target)
    _append_segment(
        rows,
        targets,
        fps=fps,
        phase="multisine_0p2_to_1p5_hz",
        condition=condition,
    )
    hold(base, 2.0, "multisine_return_hold")

    compound_seconds = 8.0
    targets = []
    for index in range(round(compound_seconds * fps)):
        time_s = index / fps
        target = base.copy()
        for joint in range(5):
            target[joint] += math.radians(8.0) * math.sin(
                2.0 * math.pi * 0.25 * time_s + joint * math.pi / 5.0
            )
        target[5] += 12.0 * math.sin(2.0 * math.pi * 0.25 * time_s + math.pi / 3.0)
        targets.append(target)
    _append_segment(
        rows,
        targets,
        fps=fps,
        phase="compound_6axis",
        condition=condition,
    )
    hold(base, 2.0, "compound_return_hold")

    for aperture_mm in np.linspace(5.0, 100.0, 9):
        target = base.copy()
        target[5] = aperture_mm
        hold(target, 1.0, f"gripper_sweep_{aperture_mm:.3f}mm")
    hold(base, 2.0, "final_hold")
    return rows


def _estimate_delay_steps(target: np.ndarray, measured: np.ndarray, max_steps: int) -> int:
    target_velocity = np.diff(target)
    measured_velocity = np.diff(measured)
    best = (float("-inf"), 0)
    for delay in range(max_steps + 1):
        left = target_velocity[: len(target_velocity) - delay or None]
        right = measured_velocity[delay:]
        if len(left) < 8:
            continue
        left = left - np.mean(left)
        right = right - np.mean(right)
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        score = float(np.dot(left, right) / denominator) if denominator else float("-inf")
        best = max(best, (score, -delay))
    return -best[1]


def identify_joint_response(
    target: np.ndarray,
    measured: np.ndarray,
    *,
    fps: int = 30,
    max_delay_ms: float = 333.0,
) -> dict[str, float | int | None]:
    """2차 ARX와 직접 관측치로 delay/limit/deadband/backlash를 식별한다."""

    u = np.asarray(target, dtype=np.float64)
    y = np.asarray(measured, dtype=np.float64)
    if u.shape != y.shape or u.ndim != 1 or len(u) < 30:
        raise ValueError("target/measured는 동일 길이의 1-D이며 30 sample 이상이어야 한다")
    dt = 1.0 / fps
    delay = _estimate_delay_steps(u, y, round(max_delay_ms / 1000.0 * fps))
    start = max(2, delay + 1)
    rows = []
    values = []
    for index in range(start, len(y)):
        rows.append(
            [
                y[index - 1],
                y[index - 2],
                u[index - delay],
                u[index - delay - 1],
                1.0,
            ]
        )
        values.append(y[index])
    coefficients, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(values), rcond=None)
    a1, a2, b0, b1, bias = coefficients
    prediction = np.asarray(rows) @ coefficients
    residual_rmse = float(np.sqrt(np.mean((prediction - values) ** 2)))
    denominator = 1.0 - a1 - a2
    dc_gain = float((b0 + b1) / denominator) if abs(denominator) > 1e-9 else None

    poles = np.roots([1.0, -a1, -a2])
    stable = bool(np.all(np.abs(poles) < 1.0))
    natural_frequency = None
    damping_ratio = None
    if stable and np.all(np.abs(poles) > 1e-12):
        continuous = np.log(poles.astype(np.complex128)) / dt
        dominant = continuous[int(np.argmax(np.real(continuous)))]
        natural_frequency = float(abs(dominant))
        if natural_frequency > 1e-12:
            damping_ratio = float(-dominant.real / natural_frequency)

    velocity = np.diff(y) / dt
    acceleration = np.diff(velocity) / dt
    target_velocity = np.diff(u) / dt
    error = u - y
    moving = np.abs(target_velocity) > np.percentile(np.abs(target_velocity), 60)
    positive = error[1:][moving & (target_velocity > 0)]
    negative = error[1:][moving & (target_velocity < 0)]
    backlash = None
    if len(positive) and len(negative):
        backlash = float(abs(np.median(positive) - np.median(negative)))
    quiet = np.abs(target_velocity) < 1e-6
    deadband = float(np.percentile(np.abs(error[1:][quiet]), 90)) if np.any(quiet) else None

    return {
        "delay_steps": delay,
        "delay_ms": delay * 1000.0 / fps,
        "arx_a1": float(a1),
        "arx_a2": float(a2),
        "arx_b0": float(b0),
        "arx_b1": float(b1),
        "arx_bias": float(bias),
        "dc_gain": dc_gain,
        "natural_frequency_rad_s": natural_frequency,
        "damping_ratio": damping_ratio,
        "stable": stable,
        "residual_rmse": residual_rmse,
        "observed_velocity_limit_per_s": float(np.percentile(np.abs(velocity), 99.5)),
        "observed_acceleration_limit_per_s2": float(
            np.percentile(np.abs(acceleration), 99.5)
        ),
        "deadband": deadband,
        "backlash": backlash,
        "steady_state_bias": float(np.median(error[1:][quiet])) if np.any(quiet) else None,
    }


def identify_response(
    targets: np.ndarray,
    measured: np.ndarray,
    *,
    fps: int = 30,
) -> dict[str, dict[str, float | int | None]]:
    target_matrix = np.asarray(targets, dtype=np.float64)
    measured_matrix = np.asarray(measured, dtype=np.float64)
    if target_matrix.shape != measured_matrix.shape or target_matrix.ndim != 2:
        raise ValueError("targets/measured shape이 다르다")
    if target_matrix.shape[1] != 6:
        raise ValueError("SO-101 dynamics trace는 6축이어야 한다")
    return {
        name: identify_joint_response(
            target_matrix[:, index],
            measured_matrix[:, index],
            fps=fps,
        )
        for index, name in enumerate(JOINT_ORDER)
    }
