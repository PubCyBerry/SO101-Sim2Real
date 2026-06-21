"""Canonical parity JSONL trace 기록과 비교."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TraceRecord:
    schema: str
    monotonic_ns: int
    wall_time_ns: int
    loop_tick: int
    policy_step: int
    request_id: int | None
    chunk_offset: int | None
    raw_model_output: list[float] | None
    limited_canonical_target: list[float] | None
    native_command: list[float] | None
    measured_canonical_state: list[float] | None
    inference_latency_ms: float | None
    hold: bool
    underrun: bool
    timeout: bool
    contract_hash: str
    runtime_manifest_hash: str
    checkpoint_hash: str
    calibration_hash: str
    motor_profile_hash: str
    domain: str


class JsonlTraceWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a", encoding="utf-8", buffering=1)

    def write(self, **fields: Any) -> None:
        record = TraceRecord(
            schema="so101-parity-trace-v1",
            monotonic_ns=time.monotonic_ns(),
            wall_time_ns=time.time_ns(),
            **fields,
        )
        self._stream.write(json.dumps(asdict(record), ensure_ascii=False, separators=(",", ":")) + "\n")
        self._stream.flush()

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()

    def __enter__(self) -> "JsonlTraceWriter":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def load_trace(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def compare_canonical_targets(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> dict[str, Any]:
    left_by_step = {int(row["policy_step"]): row for row in left if not row.get("underrun")}
    right_by_step = {int(row["policy_step"]): row for row in right if not row.get("underrun")}
    common = sorted(set(left_by_step) & set(right_by_step))
    missing_left = sorted(set(right_by_step) - set(left_by_step))
    missing_right = sorted(set(left_by_step) - set(right_by_step))
    exact = True
    max_abs = 0.0
    for step in common:
        left_target = left_by_step[step]["limited_canonical_target"]
        right_target = right_by_step[step]["limited_canonical_target"]
        if left_target is None or right_target is None:
            exact = exact and left_target is right_target
            continue
        left_value = np.asarray(left_target, dtype=np.float32)
        right_value = np.asarray(right_target, dtype=np.float32)
        exact = exact and left_value.tobytes() == right_value.tobytes()
        max_abs = max(max_abs, float(np.max(np.abs(left_value - right_value))))
    return {
        "bitwise_equal": exact and not missing_left and not missing_right,
        "common_steps": len(common),
        "missing_left": missing_left,
        "missing_right": missing_right,
        "max_abs_error": max_abs,
        "left_underruns": sum(bool(row.get("underrun")) for row in left),
        "right_underruns": sum(bool(row.get("underrun")) for row in right),
    }
