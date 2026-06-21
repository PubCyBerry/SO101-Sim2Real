#!/usr/bin/env python
"""동일 canonical excitation plan을 Isaac 또는 real SO-101에서 실행한다.

기본은 dry-run이다. Real motion은 검증된 calibration/motor profile과
--execute --enable-motion --confirm-emergency-cutoff-ready를 모두 요구한다.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("sim", "real"), required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--confirm-emergency-cutoff-ready", action="store_true")
    parser.add_argument("--confirm-payload-attached", action="store_true")
    parser.add_argument("--port", default="COM8")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("calibration/so101_canonical.json"),
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/parity/policy_io.json"),
    )
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=Path("configs/parity/runtime.json"),
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=Path("outputs/parity/dynamics/experiment_trace.jsonl"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/parity/dynamics/experiment_report.json"),
    )
    parser.add_argument("--device", default="cuda:0")
    return parser


def _write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.__stdout__, flush=True)


def _load_plan(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or any(row.get("schema") != "so101-dynamics-plan-v1" for row in rows):
        raise RuntimeError("지원하지 않는 dynamics plan")
    if any(int(row["step"]) != index for index, row in enumerate(rows)):
        raise RuntimeError("dynamics plan step이 연속적이지 않다")
    return rows


def _load_source_calibration(bundle):
    from lerobot.motors import MotorCalibration

    path = Path(bundle.raw["source_snapshots"]["lerobot"])
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected = bundle.raw["motor_profile"]["expected"]["joints"]
    if raw != expected:
        raise RuntimeError("LeRobot source snapshot과 calibration bundle이 다르다")
    return {name: MotorCalibration(**values) for name, values in raw.items()}


def _make_bus(port: str, calibration):
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    return FeetechMotorsBus(
        port=port,
        motors={
            "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
            "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
            "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
            "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
            "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
            "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
        },
        calibration=calibration,
    )


def _run(args: argparse.Namespace, simulation_app=None) -> int:
    from so101_parity.calibration import CalibrationBundle
    from so101_parity.contract import JOINT_ORDER, PolicyIOContract
    from so101_parity.executor import MotionLimiter
    from so101_parity.motor_profile import verify_motor_profile

    plan = _load_plan(args.plan)
    if args.max_steps is not None:
        plan = plan[: args.max_steps]
    contract = PolicyIOContract.load(args.contract)
    calibration = CalibrationBundle.load(args.calibration)
    runtime_cfg = json.loads(args.runtime_config.read_text(encoding="utf-8"))
    conditions = {row["condition"] for row in plan}
    if len(conditions) != 1:
        raise RuntimeError(f"plan condition은 하나여야 한다: {conditions}")
    condition = next(iter(conditions))

    blockers = []
    if not args.execute:
        blockers.append("--execute 없음")
    if args.domain == "real":
        if not args.enable_motion:
            blockers.append("--enable-motion 없음")
        if not args.confirm_emergency_cutoff_ready:
            blockers.append("--confirm-emergency-cutoff-ready 없음")
        if condition == "cube_payload" and not args.confirm_payload_attached:
            blockers.append("--confirm-payload-attached 없음")
        if not calibration.validated:
            blockers.append("canonical calibration 미검증")
        if not calibration.motor_profile_validated:
            blockers.append("motor profile readback 미검증")
    if blockers:
        _write_report(
            args.report,
            {
                "status": "blocked",
                "mode": "dry_run",
                "domain": args.domain,
                "condition": condition,
                "hardware_accessed": False,
                "motion_enabled": False,
                "steps": len(plan),
                "blockers": blockers,
            },
        )
        return 2

    limits = runtime_cfg["motion_limits"]
    limiter = MotionLimiter(
        fps=contract.fps,
        max_velocity=np.asarray(limits["velocity_per_s"], dtype=np.float64),
        max_acceleration=np.asarray(limits["acceleration_per_s2"], dtype=np.float64),
        max_jerk=np.asarray(limits["jerk_per_s3"], dtype=np.float64),
    )
    bus = None
    env = None
    adapter = None
    try:
        if args.domain == "sim":
            import gymnasium as gym

            import sim_to_real.isaac6  # noqa: F401
            from isaaclab_tasks.utils import parse_env_cfg
            from sim_to_real.isaac6.sim_adapter import Isaac6ParityAdapter

            if not calibration.has_sim_gripper_curve:
                raise RuntimeError("sim gripper curve가 없다")
            task_id = "SimToReal-SO101-PickCube-Isaac6Parity-v0"
            cfg = parse_env_cfg(task_id, device=args.device, num_envs=1)
            env = gym.make(task_id, cfg=cfg)
            env.reset()
            adapter = Isaac6ParityAdapter(env, calibration)

            def capture_state() -> np.ndarray:
                return adapter.capture().state

            def advance(canonical_target: np.ndarray) -> np.ndarray:
                native = adapter.canonical_to_native(canonical_target)
                adapter.advance(native)
                return native

        else:
            calibration.require_validated(require_motor_profile=True)
            source_calibration = _load_source_calibration(calibration)
            bus = _make_bus(args.port, source_calibration)
            bus.connect()
            bus.disable_torque()
            profile = verify_motor_profile(
                bus,
                calibration.raw["motor_profile"]["expected"],
            )
            if not profile["ok"]:
                raise RuntimeError(
                    "motor profile mismatch: " + "; ".join(profile["mismatches"])
                )
            bus.enable_torque()

            def capture_state() -> np.ndarray:
                positions = bus.sync_read("Present_Position")
                native = np.asarray(
                    [positions[name] for name in JOINT_ORDER],
                    dtype=np.float32,
                )
                return calibration.real_to_canonical(native)

            next_deadline = time.perf_counter()

            def advance(canonical_target: np.ndarray) -> np.ndarray:
                nonlocal next_deadline
                native = calibration.canonical_to_real(canonical_target)
                bus.sync_write(
                    "Goal_Position",
                    {
                        name: float(native[index])
                        for index, name in enumerate(JOINT_ORDER)
                    },
                )
                next_deadline += 1.0 / contract.fps
                time.sleep(max(0.0, next_deadline - time.perf_counter()))
                return native

        initial = capture_state()
        limiter.reset(initial)
        home = np.asarray(runtime_cfg["initial_home_target"], dtype=np.float64)
        home_tolerance = np.asarray(
            [math.radians(5.0)] * 5 + [2.0],
            dtype=np.float64,
        )
        for _ in range(contract.fps * 20):
            limited = limiter.apply(home)
            advance(limited)
            measured = capture_state()
            if np.all(np.abs(measured - home) <= home_tolerance):
                break
        else:
            raise RuntimeError("20초 안에 canonical READY pose에 도달하지 못했다")

        args.trace.parent.mkdir(parents=True, exist_ok=True)
        args.trace.unlink(missing_ok=True)
        limiter.reset(capture_state())
        with args.trace.open("w", encoding="utf-8", buffering=1) as stream:
            for row in plan:
                requested = np.asarray(row["canonical_target"], dtype=np.float64)
                limited = limiter.apply(requested)
                native = advance(limited)
                measured = capture_state()
                stream.write(
                    json.dumps(
                        {
                            "schema": "so101-dynamics-trace-v1",
                            "monotonic_ns": time.monotonic_ns(),
                            "wall_time_ns": time.time_ns(),
                            "loop_tick": int(row["step"]) + 1,
                            "policy_step": int(row["step"]),
                            "phase": row["phase"],
                            "condition": row["condition"],
                            "requested_canonical_target": requested.tolist(),
                            "limited_canonical_target": limited.astype(np.float32).tolist(),
                            "native_command": np.asarray(native, dtype=np.float32).tolist(),
                            "measured_canonical_state": measured.astype(np.float32).tolist(),
                            "timeout": False,
                            "domain": args.domain,
                            "contract_hash": contract.contract_hash,
                            "calibration_hash": calibration.calibration_hash,
                            "motor_profile_hash": calibration.motor_profile_hash,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

        _write_report(
            args.report,
            {
                "status": "passed",
                "domain": args.domain,
                "condition": condition,
                "steps": len(plan),
                "trace": str(args.trace),
                "contract_hash": contract.contract_hash,
                "calibration_hash": calibration.calibration_hash,
                "motor_profile_hash": calibration.motor_profile_hash,
            },
        )
        return 0
    finally:
        if adapter is not None:
            try:
                adapter.safe_stop("experiment_complete")
            except Exception:
                pass
        if bus is not None:
            try:
                bus.disable_torque()
            except Exception:
                pass
            try:
                if bus.is_connected:
                    bus.disconnect(disable_torque=True)
            except Exception:
                pass
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if simulation_app is not None:
            simulation_app.close()


def main() -> int:
    parser = _base_parser()
    args = parser.parse_args()
    simulation_app = None
    try:
        if args.domain == "sim" and args.execute:
            from isaaclab.app import AppLauncher

            launcher = AppLauncher(
                {
                    "visualizer": "none",
                    "device": args.device,
                    "enable_cameras": True,
                    "livestream": 0,
                }
            )
            simulation_app = launcher.app
        return _run(args, simulation_app)
    except Exception as exc:
        _write_report(
            args.report,
            {
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        if simulation_app is not None:
            simulation_app.close()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
