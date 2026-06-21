#!/usr/bin/env python
"""동일 excitation plan의 real/sim trace를 비교하고 PhysX fitting 지표를 생성한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import numpy as np

from so101_parity.contract import JOINT_ORDER
from so101_parity.dynamics import identify_response


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _trace_matrix(path: Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rows = _load_jsonl(path)
    usable = [
        row
        for row in rows
        if row.get("limited_canonical_target") is not None
        and row.get("measured_canonical_state") is not None
        and not row.get("timeout", False)
    ]
    target = np.asarray([row["limited_canonical_target"] for row in usable], dtype=np.float64)
    measured = np.asarray([row["measured_canonical_state"] for row in usable], dtype=np.float64)
    return target, measured, usable


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--real-trace", type=Path, required=True)
    parser.add_argument("--sim-trace", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report: dict = {}
    try:
        plan = _load_jsonl(args.plan)
        real_target, real_measured, real_rows = _trace_matrix(args.real_trace)
        sim_target, sim_measured, sim_rows = _trace_matrix(args.sim_trace)
        hash_keys = ("contract_hash", "calibration_hash", "motor_profile_hash")
        trace_hashes = {
            "real": {key: real_rows[0].get(key) for key in hash_keys} if real_rows else {},
            "sim": {key: sim_rows[0].get(key) for key in hash_keys} if sim_rows else {},
        }
        if (
            not real_rows
            or not sim_rows
            or any(not trace_hashes["real"].get(key) for key in hash_keys)
            or trace_hashes["real"] != trace_hashes["sim"]
        ):
            raise RuntimeError(f"real/sim trace hash 불일치 또는 누락: {trace_hashes}")
        if any(
            any(row.get(key) != trace_hashes[domain][key] for row in rows)
            for domain, rows in (("real", real_rows), ("sim", sim_rows))
            for key in hash_keys
        ):
            raise RuntimeError("한 trace 내부에서 runtime hash가 변경됐다")
        count = min(len(plan), len(real_rows), len(sim_rows))
        if count < 30:
            raise RuntimeError(f"비교 가능한 dynamics sample이 부족하다: {count}")
        plan = plan[:count]
        real_target = real_target[:count]
        real_measured = real_measured[:count]
        sim_target = sim_target[:count]
        sim_measured = sim_measured[:count]

        target_bitwise_equal = (
            real_target.astype(np.float32).tobytes()
            == sim_target.astype(np.float32).tobytes()
        )
        real_identified = identify_response(real_target, real_measured)
        sim_identified = identify_response(sim_target, sim_measured)
        error = sim_measured - real_measured
        arm_error_deg = np.rad2deg(np.abs(error[:, :5]))
        gripper_error_mm = np.abs(error[:, 5])
        steady_mask = np.asarray(
            [
                "hold" in row["phase"] or row["phase"].startswith("gripper_sweep")
                for row in plan
            ],
            dtype=bool,
        )
        if not np.any(steady_mask):
            steady_mask = np.ones(count, dtype=bool)

        lag_diff_ms = {
            name: abs(
                float(real_identified[name]["delay_ms"])
                - float(sim_identified[name]["delay_ms"])
            )
            for name in JOINT_ORDER
        }
        suggestions = {}
        for name in JOINT_ORDER:
            real_joint = real_identified[name]
            sim_joint = sim_identified[name]
            real_wn = real_joint["natural_frequency_rad_s"]
            sim_wn = sim_joint["natural_frequency_rad_s"]
            real_zeta = real_joint["damping_ratio"]
            sim_zeta = sim_joint["damping_ratio"]
            suggestions[name] = {
                "delay_steps_target": real_joint["delay_steps"],
                "stiffness_scale_hint": (
                    float(real_wn / sim_wn) ** 2
                    if real_wn and sim_wn and sim_wn > 0
                    else None
                ),
                "damping_scale_hint": (
                    float(real_zeta / sim_zeta)
                    if real_zeta is not None and sim_zeta not in (None, 0)
                    else None
                ),
                "velocity_limit_target_per_s": real_joint[
                    "observed_velocity_limit_per_s"
                ],
                "acceleration_limit_target_per_s2": real_joint[
                    "observed_acceleration_limit_per_s2"
                ],
                "deadband_target": real_joint["deadband"],
                "backlash_target": real_joint["backlash"],
                "gravity_droop_target": real_joint["steady_state_bias"],
                "armature_and_friction": (
                    "ARX residual/overshoot를 최소화하도록 simulator sweep에서 공동 최적화"
                ),
            }

        metrics = {
            "arm_trajectory_p95_deg": float(np.percentile(arm_error_deg, 95)),
            "arm_steady_state_p95_deg": float(
                np.percentile(arm_error_deg[steady_mask], 95)
            ),
            "gripper_aperture_p95_mm": float(np.percentile(gripper_error_mm, 95)),
            "sim_real_lag_difference_max_ms": max(lag_diff_ms.values()),
        }
        gates = {
            "target_bitwise_equal": target_bitwise_equal,
            "arm_trajectory_p95_le_5deg": metrics["arm_trajectory_p95_deg"] <= 5.0,
            "arm_steady_state_p95_le_2deg": metrics[
                "arm_steady_state_p95_deg"
            ]
            <= 2.0,
            "gripper_p95_le_2mm": metrics["gripper_aperture_p95_mm"] <= 2.0,
            "lag_difference_le_33ms": metrics[
                "sim_real_lag_difference_max_ms"
            ]
            <= 33.0,
        }
        report = {
            "status": "passed" if all(gates.values()) else "failed",
            "samples": count,
            "trace_hashes": trace_hashes["real"],
            "metrics": metrics,
            "gates": gates,
            "lag_difference_ms": lag_diff_ms,
            "real_identified": real_identified,
            "sim_identified": sim_identified,
            "physx_fit_suggestions": suggestions,
        }
        _write_json(args.report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "passed" else 1
    except Exception as exc:
        report = {
            "status": "blocked",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(args.report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
