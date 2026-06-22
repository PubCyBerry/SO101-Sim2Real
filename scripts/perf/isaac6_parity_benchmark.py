#!/usr/bin/env python
"""Isaac Sim 6 / Isaac Lab 3 SO-101 parity hot-path benchmark.

This benchmark is intentionally independent from ROS so it can isolate:

* environment construction/reset cost;
* 30 Hz policy-step latency;
* state-only reads versus three-camera CPU capture;
* USD stage composition size;
* PhysX GPU buffer profile effects.

Use ``--capture-every 1`` to reproduce the pre-optimization runtime behavior,
and ``--capture-every 16`` to approximate one observation capture per action
chunk.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any

PROCESS_STARTED = time.perf_counter()

from isaaclab.app import AppLauncher


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summary_ms(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values, default=0.0),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _gpu_process_memory_mib() -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        current_pid = os.getpid()
        for line in result.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 2 and int(fields[0]) == current_pid:
                return int(fields[1])
    except Exception:
        return None
    return None


parser = argparse.ArgumentParser()
parser.add_argument("--task", default="SimToReal-SO101-PickCube-Isaac6Parity-v0")
parser.add_argument("--calibration", type=Path, default=Path("calibration/so101_canonical.json"))
parser.add_argument("--warmup-steps", type=int, default=20)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument(
    "--capture-every",
    type=int,
    default=1,
    help="Three-camera CPU capture cadence in policy steps. 0 means state-only.",
)
parser.add_argument(
    "--camera-pose-updates",
    choices=("config", "on", "off"),
    default="config",
)
parser.add_argument(
    "--physx-buffers",
    choices=("config", "compact"),
    default="config",
)
parser.add_argument(
    "--render-on-capture",
    action="store_true",
    help="Skip Kit camera rendering between observation captures.",
)
parser.add_argument(
    "--report",
    type=Path,
    default=Path("outputs/performance/isaac6_parity_benchmark.json"),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

launcher_cfg = {
    "visualizer": args.visualizer or "none",
    "device": args.device,
    "enable_cameras": True,
    "livestream": 0,
}
if args.kit_args:
    launcher_cfg["kit_args"] = args.kit_args

launcher = AppLauncher(launcher_cfg)
simulation_app = launcher.app
APP_READY = time.perf_counter()


def _emit(payload: dict[str, Any]) -> None:
    args.report.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_jsonable(payload), ensure_ascii=False, indent=2)
    args.report.write_text(text + "\n", encoding="utf-8")
    print(text, file=sys.__stdout__, flush=True)


def _read_state(adapter) -> Any:
    native = adapter.robot.data.joint_pos.torch[0].detach().cpu().numpy()
    return adapter.calibration.sim_to_canonical(native, clamp=True)


def _configure_candidate(env_cfg) -> None:
    if args.camera_pose_updates != "config":
        enabled = args.camera_pose_updates == "on"
        for name in ("top_camera", "wrist_camera", "front_camera"):
            getattr(env_cfg.scene, name).update_latest_camera_pose = enabled
    if args.physx_buffers == "compact":
        physics = env_cfg.sim.physics
        physics.gpu_max_rigid_contact_count = 2**20
        physics.gpu_max_rigid_patch_count = 5 * 2**14
        physics.gpu_found_lost_pairs_capacity = 2**19
        physics.gpu_found_lost_aggregate_pairs_capacity = 2**20
        physics.gpu_total_aggregate_pairs_capacity = 2**19
        physics.gpu_collision_stack_size = 2**25
        physics.gpu_heap_capacity = 2**25
        physics.gpu_temp_buffer_capacity = 2**23


def main() -> int:
    env = None
    try:
        import gymnasium as gym
        import torch
        from pxr import UsdUtils

        import sim_to_real.isaac6  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg
        from sim_to_real.isaac6.sim_adapter import Isaac6ParityAdapter
        from so101_parity.calibration import CalibrationBundle

        calibration = CalibrationBundle.load(args.calibration)
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
        env_cfg.episode_length_s = max(
            env_cfg.episode_length_s,
            (args.warmup_steps + args.steps + 30) * env_cfg.sim.dt * env_cfg.decimation,
        )
        _configure_candidate(env_cfg)

        make_started = time.perf_counter()
        env = gym.make(args.task, cfg=env_cfg)
        make_finished = time.perf_counter()
        reset_started = time.perf_counter()
        env.reset()
        reset_finished = time.perf_counter()
        adapter = Isaac6ParityAdapter(
            env,
            calibration,
            render_on_capture=args.render_on_capture,
        )
        action = adapter._last_native.unsqueeze(0)

        for step in range(args.warmup_steps):
            if args.capture_every > 0 and step % args.capture_every == 0:
                adapter.capture()
            else:
                _read_state(adapter)
            env.step(action)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        step_ms: list[float] = []
        capture_ms: list[float] = []
        state_ms: list[float] = []
        total_started = time.perf_counter()
        captured_frames = 0
        image_checksum = 0
        for step in range(args.steps):
            sample_started = time.perf_counter()
            should_capture = args.capture_every > 0 and step % args.capture_every == 0
            read_started = time.perf_counter()
            if should_capture:
                observation = adapter.capture()
                captured_frames += 1
                image_checksum = (
                    image_checksum
                    + sum(int(image[0, 0].sum()) for image in observation.images.values())
                ) % 2**32
            else:
                _read_state(adapter)
            read_finished = time.perf_counter()
            env.step(action)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            sample_finished = time.perf_counter()
            (capture_ms if should_capture else state_ms).append(
                (read_finished - read_started) * 1000.0
            )
            step_ms.append((sample_finished - sample_started) * 1000.0)
        total_finished = time.perf_counter()

        stage = env.unwrapped.scene.stage
        stage_stats = UsdUtils.ComputeUsdStageStats(stage)
        used_layers = stage.GetUsedLayers()
        cameras = {}
        for name in ("top_camera", "wrist_camera", "front_camera"):
            cfg = getattr(env_cfg.scene, name)
            rgb = env.unwrapped.scene[name].data.output["rgb"].torch
            cameras[name] = {
                "shape": list(rgb.shape),
                "dtype": str(rgb.dtype),
                "update_period": cfg.update_period,
                "update_latest_camera_pose": cfg.update_latest_camera_pose,
            }

        elapsed = total_finished - total_started
        physics = env_cfg.sim.physics
        _emit(
            {
                "status": "passed",
                "task": args.task,
                "device": args.device,
                "startup_s": {
                    "process_to_app": APP_READY - PROCESS_STARTED,
                    "gym_make": make_finished - make_started,
                    "reset": reset_finished - reset_started,
                },
                "benchmark": {
                    "warmup_steps": args.warmup_steps,
                    "steps": args.steps,
                    "capture_every": args.capture_every,
                    "captured_frames": captured_frames,
                    "render_on_capture": args.render_on_capture,
                    "elapsed_s": elapsed,
                    "policy_steps_per_s": args.steps / elapsed,
                    "step_ms": _summary_ms(step_ms),
                    "capture_ms": _summary_ms(capture_ms),
                    "state_read_ms": _summary_ms(state_ms),
                    "image_checksum": image_checksum,
                },
                "gpu_process_memory_mib": _gpu_process_memory_mib(),
                "cameras": cameras,
                "physx_buffers": {
                    "profile": args.physx_buffers,
                    "gpu_max_rigid_contact_count": physics.gpu_max_rigid_contact_count,
                    "gpu_max_rigid_patch_count": physics.gpu_max_rigid_patch_count,
                    "gpu_found_lost_pairs_capacity": physics.gpu_found_lost_pairs_capacity,
                    "gpu_found_lost_aggregate_pairs_capacity": (
                        physics.gpu_found_lost_aggregate_pairs_capacity
                    ),
                    "gpu_total_aggregate_pairs_capacity": (
                        physics.gpu_total_aggregate_pairs_capacity
                    ),
                    "gpu_collision_stack_size": physics.gpu_collision_stack_size,
                    "gpu_heap_capacity": physics.gpu_heap_capacity,
                    "gpu_temp_buffer_capacity": physics.gpu_temp_buffer_capacity,
                },
                "stage": {
                    "stats": stage_stats,
                    "used_layer_count": len(used_layers),
                    "used_layers": sorted(layer.identifier for layer in used_layers),
                },
                "kit_args": args.kit_args,
            }
        )
        return 0
    except Exception as exc:
        _emit(
            {
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return 1
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
