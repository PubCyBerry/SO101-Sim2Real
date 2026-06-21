#!/usr/bin/env python
"""Isaac Lab 3 environment-step 기반 canonical ROS VLA client."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=64)
parser.add_argument("--task-text", default="pick up the cube and place it in the bowl")
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
parser.add_argument("--trace", type=Path, default=Path("outputs/parity/sim_trace.jsonl"))
parser.add_argument("--report", type=Path, default=Path("outputs/parity/sim_client_report.json"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

launcher = AppLauncher(
    {
        "visualizer": args.visualizer or "none",
        "device": args.device,
        "enable_cameras": True,
        "livestream": 0,
    }
)
simulation_app = launcher.app


def _report(payload: dict) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(text + "\n", encoding="utf-8")
    print(text, file=sys.__stdout__, flush=True)


def main() -> int:
    env = None
    node = None
    client = None
    trace = None
    adapter = None
    try:
        import gymnasium as gym
        import numpy as np
        import rclpy
        from rclpy.node import Node

        import sim_to_real.isaac6  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg
        from so101_parity.calibration import CalibrationBundle
        from so101_parity.contract import PolicyIOContract
        from so101_parity.executor import Chunk, MotionLimiter, prefetch_lead_from_p99
        from so101_parity.manifest import RuntimeManifest, file_sha256
        from so101_parity.runtime import CanonicalRuntime, RuntimeHashes
        from so101_parity.trace import JsonlTraceWriter
        from so101_vla_runtime.client import CanonicalVlaClient
        from sim_to_real.isaac6.sim_adapter import Isaac6ParityAdapter

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
        if not calibration.has_sim_gripper_curve:
            raise RuntimeError("sim gripper aperture calibration이 없어 motion을 시작할 수 없다")

        task_id = "SimToReal-SO101-PickCube-Isaac6Parity-v0"
        env_cfg = parse_env_cfg(task_id, device=args.device, num_envs=1)
        env = gym.make(task_id, cfg=env_cfg)
        env.reset()
        adapter = Isaac6ParityAdapter(env, calibration)

        rclpy.init()
        node = Node("so101_isaac6_parity_client")
        client = CanonicalVlaClient(node, "isaac-sim")
        client.connect(timeout_sec=20.0)
        expected = {
            "contract_hash": contract.contract_hash,
            "runtime_manifest_hash": manifest.manifest_hash,
            "checkpoint_hash": str(manifest.raw["checkpoint_hash"]),
            "calibration_hash": calibration.calibration_hash,
            "motor_profile_hash": calibration.motor_profile_hash,
            "fps": contract.fps,
            "chunk_size": int(manifest.raw["chunk_size"]),
        }
        actual = {name: getattr(client, name) for name in expected}
        mismatches = [
            f"{name}: server={actual[name]!r}, expected={value!r}"
            for name, value in expected.items()
            if actual[name] != value
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
                request_id=ticket.request_id,
                start_step=ticket.requested_start_step,
                actions=actions,
                inference_latency_ms=response.inference_latency_ms,
                checkpoint_hash=response.checkpoint_hash,
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
                contract_hash=contract.contract_hash,
                runtime_manifest_hash=manifest.manifest_hash,
                checkpoint_hash=str(manifest.raw["checkpoint_hash"]),
                calibration_hash=calibration.calibration_hash,
                motor_profile_hash=calibration.motor_profile_hash,
            ),
            trace=trace,
            prefetch_lead=max(
                8,
                math.ceil(args.p99_latency_ms / 1000.0 * contract.fps) + 2,
            ),
            request_timeout_ms=int(runtime_cfg["executor"]["timeout_ms"]),
        )
        home = np.asarray(runtime_cfg["initial_home_target"], dtype=np.float32)
        observation = runtime.move_home(home)
        runtime.prime(observation)
        runtime.run(args.steps)
        _report(
            {
                "status": "passed",
                "task": task_id,
                "steps": args.steps,
                "policy_steps": runtime.executor.step,
                "loop_ticks": runtime.executor.loop_tick,
                "underruns": runtime.executor.underruns,
                "timeouts": runtime.executor.timeouts,
                "stale_responses": runtime.executor.stale_responses,
                "prefetch_lead": runtime.executor.prefetch_lead,
                "trace": str(args.trace),
                "contract_hash": contract.contract_hash,
                "runtime_manifest_hash": manifest.manifest_hash,
                "checkpoint_hash": str(manifest.raw["checkpoint_hash"]),
                "calibration_hash": calibration.calibration_hash,
            }
        )
        return 0
    except Exception as exc:
        if adapter is not None:
            try:
                adapter.safe_stop("client_exception")
            except Exception:
                pass
        _report(
            {
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
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
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
