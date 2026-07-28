#!/usr/bin/env python3
"""EEF-relative sim/real rollout JSONL을 acceptance criteria로 판정한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"metrics JSONL not found: {path}")
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: metric record must be an object")
        records.append(value)
    if not records:
        raise ValueError(f"metrics JSONL is empty: {path}")
    return records


def _load_eval(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"sim eval JSON not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"sim eval JSON must be an object: {path}")
    return value


def _number(record: dict[str, Any], key: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"final metric field {key!r} must be numeric, got {value!r}")
    return float(value)


def _require_at_least(
    failures: list[str],
    record: dict[str, Any],
    key: str,
    minimum: float,
) -> float:
    value = _number(record, key)
    if value < minimum:
        failures.append(f"{key}={value:g} < required {minimum:g}")
    return value


def _require_at_most(
    failures: list[str],
    record: dict[str, Any],
    key: str,
    maximum: float,
) -> float:
    value = _number(record, key)
    if value > maximum:
        failures.append(f"{key}={value:g} > allowed {maximum:g}")
    return value


def _evaluate_sim(
    final: dict[str, Any],
    failures: list[str],
    *,
    max_starvation_rate: float,
    max_empty_chunk_rate: float,
    max_stale_chunk_rate: float,
    max_position_residual_m: float,
    max_orientation_residual_rad: float,
) -> dict[str, float]:
    requests = _require_at_least(failures, final, "inference_requests", 1)
    _require_at_least(failures, final, "chunks_received", 1)
    commands = _require_at_least(failures, final, "commands_published", 1)
    _require_at_least(failures, final, "ik_steps", 1)
    _require_at_most(failures, final, "ik_failures", 0)
    _require_at_most(failures, final, "invalid_chunks", 0)
    _require_at_most(failures, final, "aborts", 0)
    _require_at_most(
        failures,
        final,
        "position_residual_max_m",
        max_position_residual_m,
    )
    _require_at_most(
        failures,
        final,
        "orientation_residual_max_rad",
        max_orientation_residual_rad,
    )

    starvation = _number(final, "queue_starvation_ticks")
    empty = _number(final, "empty_chunks")
    stale = _number(final, "stale_chunks")
    starvation_rate = starvation / max(starvation + commands, 1.0)
    empty_rate = empty / max(requests, 1.0)
    stale_rate = stale / max(requests, 1.0)
    if starvation_rate > max_starvation_rate:
        failures.append(
            f"queue_starvation_rate={starvation_rate:.6f} > {max_starvation_rate:.6f}"
        )
    if empty_rate > max_empty_chunk_rate:
        failures.append(f"empty_chunk_rate={empty_rate:.6f} > {max_empty_chunk_rate:.6f}")
    if stale_rate > max_stale_chunk_rate:
        failures.append(f"stale_chunk_rate={stale_rate:.6f} > {max_stale_chunk_rate:.6f}")
    return {
        "queue_starvation_rate": starvation_rate,
        "empty_chunk_rate": empty_rate,
        "stale_chunk_rate": stale_rate,
    }


def _evaluate_real_dry_run(
    final: dict[str, Any],
    failures: list[str],
) -> dict[str, float]:
    _require_at_least(failures, final, "chunks_received", 1)
    _require_at_least(failures, final, "dry_run_chunks", 1)
    _require_at_most(failures, final, "commands_sent", 0)
    _require_at_most(failures, final, "ik_failures", 0)
    _require_at_most(failures, final, "aborts", 0)
    return {}


def _evaluate_real(
    final: dict[str, Any],
    failures: list[str],
    *,
    max_position_residual_m: float,
    max_orientation_residual_rad: float,
) -> dict[str, float]:
    _require_at_least(failures, final, "chunks_received", 1)
    _require_at_least(failures, final, "commands_sent", 1)
    _require_at_least(failures, final, "residual_samples", 1)
    _require_at_most(failures, final, "ik_failures", 0)
    _require_at_most(failures, final, "aborts", 0)
    _require_at_most(
        failures,
        final,
        "position_residual_max_m",
        max_position_residual_m,
    )
    _require_at_most(
        failures,
        final,
        "orientation_residual_max_rad",
        max_orientation_residual_rad,
    )
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sim", "real-dry-run", "real"), required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--eval", dest="eval_path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-success-rate", type=float, default=1.0)
    parser.add_argument("--max-starvation-rate", type=float, default=0.10)
    parser.add_argument("--max-empty-chunk-rate", type=float, default=0.05)
    parser.add_argument("--max-stale-chunk-rate", type=float, default=0.10)
    parser.add_argument("--max-position-residual-m", type=float)
    parser.add_argument("--max-orientation-residual-rad", type=float)
    args = parser.parse_args()

    for name in (
        "min_success_rate",
        "max_starvation_rate",
        "max_empty_chunk_rate",
        "max_stale_chunk_rate",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0,1]")

    records = _load_jsonl(args.metrics)
    final = records[-1]
    failures: list[str] = []
    if final.get("event") != "final":
        failures.append(
            f"last metrics event must be 'final', got {final.get('event')!r}; "
            "rollout may not have shut down cleanly"
        )

    derived: dict[str, float]
    if args.mode == "sim":
        derived = _evaluate_sim(
            final,
            failures,
            max_starvation_rate=args.max_starvation_rate,
            max_empty_chunk_rate=args.max_empty_chunk_rate,
            max_stale_chunk_rate=args.max_stale_chunk_rate,
            max_position_residual_m=(
                5e-4
                if args.max_position_residual_m is None
                else args.max_position_residual_m
            ),
            max_orientation_residual_rad=(
                1e-2
                if args.max_orientation_residual_rad is None
                else args.max_orientation_residual_rad
            ),
        )
        if args.eval_path is None:
            failures.append("--eval is required in sim mode")
        else:
            eval_result = _load_eval(args.eval_path)
            success_rate = float(eval_result.get("all_cubes_success_rate", -1.0))
            derived["all_cubes_success_rate"] = success_rate
            if success_rate < args.min_success_rate:
                failures.append(
                    f"all_cubes_success_rate={success_rate:.6f} "
                    f"< required {args.min_success_rate:.6f}"
                )
    elif args.mode == "real-dry-run":
        derived = _evaluate_real_dry_run(final, failures)
    else:
        derived = _evaluate_real(
            final,
            failures,
            max_position_residual_m=(
                3e-2
                if args.max_position_residual_m is None
                else args.max_position_residual_m
            ),
            max_orientation_residual_rad=(
                2e-1
                if args.max_orientation_residual_rad is None
                else args.max_orientation_residual_rad
            ),
        )

    report = {
        "status": "PASS" if not failures else "FAIL",
        "mode": args.mode,
        "metrics_path": str(args.metrics),
        "eval_path": str(args.eval_path) if args.eval_path is not None else None,
        "records": len(records),
        "final": final,
        "derived": derived,
        "failures": failures,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if not failures else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
