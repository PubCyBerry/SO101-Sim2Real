#!/usr/bin/env python
"""Windows native SO-101 canonical ROS client.

기본 실행은 파일/manifest만 검사하는 dry-run이다. Hardware readback은
``--inspect-readback``에서 torque-off로만 수행한다. Motion은 검증된 calibration,
검증된 motor profile, ``--enable-motion``, ``--confirm-emergency-cutoff-ready``가
모두 있어야 허용된다.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import traceback

import numpy as np

from so101_parity.calibration import CalibrationBundle
from so101_parity.contract import JOINT_ORDER, PolicyIOContract
from so101_parity.executor import Chunk, MotionLimiter
from so101_parity.manifest import RuntimeManifest, file_sha256
from so101_parity.motor_profile import verify_motor_profile
from so101_parity.real_adapter import RealSO101Adapter
from so101_parity.runtime import CanonicalRuntime, RuntimeHashes
from so101_parity.trace import JsonlTraceWriter


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM8")
    parser.add_argument("--top-camera", type=int, default=2)
    parser.add_argument("--wrist-camera", type=int, default=1)
    parser.add_argument("--front-camera", type=int, default=0)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--task-text", default="pick up the cube and place it in the bowl")
    parser.add_argument("--inspect-readback", action="store_true")
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--confirm-emergency-cutoff-ready", action="store_true")
    parser.add_argument("--contract", type=Path, default=Path("configs/parity/policy_io.json"))
    parser.add_argument("--calibration", type=Path, default=Path("calibration/so101_canonical.json"))
    parser.add_argument("--runtime-config", type=Path, default=Path("configs/parity/runtime.json"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/parity/runtime_manifest.mock.json"),
    )
    parser.add_argument("--pixi-lock", type=Path, default=Path("pixi.lock"))
    parser.add_argument("--p99-latency-ms", type=float, default=250.0)
    parser.add_argument("--trace", type=Path, default=Path("outputs/parity/real_trace.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("outputs/parity/real_client_report.json"))
    return parser


def _write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def _load_source_calibration(bundle: CalibrationBundle):
    from lerobot.motors import MotorCalibration

    path = Path(bundle.raw["source_snapshots"]["lerobot"])
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected = bundle.raw["motor_profile"]["expected"]["joints"]
    if raw != expected:
        raise RuntimeError(f"LeRobot source snapshot과 calibration bundle이 다르다: {path}")
    return {
        name: MotorCalibration(**values)
        for name, values in raw.items()
    }


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


def _make_cameras(args):
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.cameras.utils import make_cameras_from_configs

    configs = {
        "top": OpenCVCameraConfig(
            args.top_camera, fps=30, width=640, height=480, fourcc="MJPG"
        ),
        "wrist": OpenCVCameraConfig(
            args.wrist_camera, fps=30, width=640, height=480, fourcc="MJPG"
        ),
        "front": OpenCVCameraConfig(
            args.front_camera, fps=30, width=640, height=480, fourcc="MJPG"
        ),
    }
    return make_cameras_from_configs(configs)


def main() -> int:
    args = _parser().parse_args()
    bus = None
    cameras = {}
    adapter = None
    node = None
    client = None
    trace = None
    try:
        contract = PolicyIOContract.load(args.contract)
        calibration = CalibrationBundle.load(args.calibration)
        manifest = RuntimeManifest.load(args.manifest)
        runtime_cfg = json.loads(args.runtime_config.read_text(encoding="utf-8"))
        manifest.assert_hashes(
            contract_hash=contract.contract_hash,
            calibration_hash=calibration.calibration_hash,
            motor_profile_hash=calibration.motor_profile_hash,
            checkpoint_hash=str(manifest.raw["checkpoint_hash"]),
            pixi_lock_hash=file_sha256(args.pixi_lock),
            runtime_config_hash=file_sha256(args.runtime_config),
        )
        source_calibration = _load_source_calibration(calibration)

        blockers = []
        if not calibration.validated:
            blockers.append("paired arm pose와 real gripper aperture calibration 미검증")
        if not calibration.motor_profile_validated:
            blockers.append("EEPROM motor profile readback 미검증")
        if not calibration.has_real_gripper_curve:
            blockers.append("real gripper PCHIP curve 없음")
        if not args.enable_motion:
            blockers.append("--enable-motion 없음")
        if not args.confirm_emergency_cutoff_ready:
            blockers.append("--confirm-emergency-cutoff-ready 없음")

        if not args.inspect_readback and not args.enable_motion:
            _write_report(
                args.report,
                {
                    "status": "passed",
                    "mode": "dry_run",
                    "hardware_accessed": False,
                    "motion_allowed": False,
                    "blockers": blockers,
                    "contract_hash": contract.contract_hash,
                    "runtime_manifest_hash": manifest.manifest_hash,
                    "calibration_hash": calibration.calibration_hash,
                    "motor_profile_hash": calibration.motor_profile_hash,
                },
            )
            return 0

        if args.enable_motion:
            if not args.confirm_emergency_cutoff_ready:
                raise RuntimeError(
                    "비상 전원 차단 준비 확인 없이 --enable-motion을 사용할 수 없다"
                )
            calibration.require_validated(require_motor_profile=True)

        bus = _make_bus(args.port, source_calibration)
        bus.connect()
        bus.disable_torque()
        profile = verify_motor_profile(
            bus,
            calibration.raw["motor_profile"]["expected"],
        )
        if not profile["ok"]:
            raise RuntimeError("motor profile readback mismatch: " + "; ".join(profile["mismatches"]))

        if args.inspect_readback and not args.enable_motion:
            _write_report(
                args.report,
                {
                    "status": "passed",
                    "mode": "inspect_readback",
                    "hardware_accessed": True,
                    "torque_enabled": False,
                    "motion_allowed": False,
                    "profile": profile,
                    "motor_profile_hash": calibration.motor_profile_hash,
                },
            )
            return 0

        cameras = _make_cameras(args)
        for camera in cameras.values():
            camera.connect()
        bus.enable_torque()
        adapter = RealSO101Adapter(
            bus=bus,
            cameras=cameras,
            calibration=calibration,
            enable_motion=True,
            fps=contract.fps,
        )

        import rclpy
        from rclpy.node import Node
        from so101_vla_runtime.client import CanonicalVlaClient

        rclpy.init()
        node = Node("so101_real_parity_client")
        client = CanonicalVlaClient(node, "real-so101")
        client.connect(timeout_sec=20.0)
        expected_server = {
            "contract_hash": contract.contract_hash,
            "runtime_manifest_hash": manifest.manifest_hash,
            "checkpoint_hash": str(manifest.raw["checkpoint_hash"]),
            "calibration_hash": calibration.calibration_hash,
            "motor_profile_hash": calibration.motor_profile_hash,
            "fps": contract.fps,
            "chunk_size": int(manifest.raw["chunk_size"]),
        }
        mismatches = [
            f"{name}: server={getattr(client, name)!r}, expected={value!r}"
            for name, value in expected_server.items()
            if getattr(client, name) != value
        ]
        if mismatches:
            raise RuntimeError("ROS runtime info mismatch: " + "; ".join(mismatches))

        def infer(ticket, observation):
            response, actions = client.infer(
                request_id=ticket.request_id,
                observation_step=ticket.observation_step,
                start_step=ticket.requested_start_step,
                task=args.task_text,
                canonical_state=observation.state,
                images=observation.images,
                timeout_sec=runtime_cfg["executor"]["timeout_ms"] / 1000.0,
            )
            return Chunk(
                ticket.request_id,
                ticket.requested_start_step,
                actions,
                response.inference_latency_ms,
                response.checkpoint_hash,
            )

        limits = runtime_cfg["motion_limits"]
        limiter = MotionLimiter(
            fps=contract.fps,
            max_velocity=np.asarray(limits["velocity_per_s"], dtype=np.float32),
            max_acceleration=np.asarray(limits["acceleration_per_s2"], dtype=np.float32),
            max_jerk=np.asarray(limits["jerk_per_s3"], dtype=np.float32),
        )
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        args.trace.unlink(missing_ok=True)
        trace = JsonlTraceWriter(args.trace)
        runtime = CanonicalRuntime(
            adapter=adapter,
            infer=infer,
            limiter=limiter,
            hashes=RuntimeHashes(
                contract.contract_hash,
                manifest.manifest_hash,
                str(manifest.raw["checkpoint_hash"]),
                calibration.calibration_hash,
                calibration.motor_profile_hash,
            ),
            trace=trace,
            prefetch_lead=max(
                8,
                math.ceil(args.p99_latency_ms / 1000.0 * contract.fps) + 2,
            ),
            request_timeout_ms=int(runtime_cfg["executor"]["timeout_ms"]),
        )
        observation = runtime.move_home(
            np.asarray(runtime_cfg["initial_home_target"], dtype=np.float32)
        )
        runtime.prime(observation)
        runtime.run(args.steps)
        _write_report(
            args.report,
            {
                "status": "passed",
                "mode": "motion",
                "steps": args.steps,
                "underruns": runtime.executor.underruns,
                "timeouts": runtime.executor.timeouts,
                "trace": str(args.trace),
                "profile": profile,
            },
        )
        return 0
    except Exception as exc:
        if adapter is not None:
            try:
                adapter.safe_stop("client_exception")
            except Exception:
                pass
        _write_report(
            args.report,
            {
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        return 1
    finally:
        if trace is not None:
            trace.close()
        if client is not None:
            try:
                client.release()
            except Exception:
                pass
        if node is not None:
            node.destroy_node()
            import rclpy

            rclpy.shutdown()
        for camera in cameras.values():
            try:
                camera.disconnect()
            except Exception:
                pass
        if bus is not None and bus.is_connected:
            try:
                bus.disconnect(disable_torque=True)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
