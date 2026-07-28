#!/usr/bin/env python3
"""SO-101 reachable workspace의 FK→sequential IK deterministic sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from so101_contract.eef_ik import IKConfig  # noqa: E402
from so101_contract.eef_policy_io import (  # noqa: E402
    EEFPlatformAdapterConfig,
    SO101EEFPolicyIO,
)
from so101_contract.follower_calibration import sim_radians_to_real_follower  # noqa: E402


def run_sweep(*, chunks: int, horizon: int, seed: int) -> dict:
    if chunks <= 0 or horizon <= 0:
        raise ValueError("chunks and horizon must be positive")
    adapter = SO101EEFPolicyIO.from_files(
        ROOT / "assets" / "robots" / "urdf" / "so_arm101.urdf",
        ROOT / "assets" / "robots" / "so101.yml",
        ik_config=IKConfig(
            position_tolerance_m=5e-4,
            orientation_tolerance_rad=1e-2,
        ),
        config=EEFPlatformAdapterConfig(
            max_arm_step_rad=0.08,
            max_gripper_step_rad=0.12,
            real_hardware_ik_validated=True,
        ),
    )
    rng = np.random.default_rng(seed)
    limits = adapter.ik.joint_limits_rad
    margin = 0.15 * (limits[:, 1] - limits[:, 0])
    lower = limits[:, 0] + margin
    upper = limits[:, 1] - margin

    position_max = 0.0
    orientation_max = 0.0
    joint_max = 0.0
    iterations: list[int] = []
    for _ in range(chunks):
        current_arm = rng.uniform(lower, upper)
        current = np.concatenate([current_arm, [rng.uniform(-0.10, 1.50)]]).astype(np.float32)
        targets: list[np.ndarray] = []
        previous = current.copy()
        for _step in range(horizon):
            target = previous.copy()
            target[:5] = np.clip(
                target[:5] + rng.uniform(-0.025, 0.025, size=5),
                lower,
                upper,
            )
            target[5] = float(np.clip(target[5] + rng.uniform(-0.06, 0.06), -0.10, 1.50))
            targets.append(target)
            previous = target
        target_joints = np.stack(targets).astype(np.float32)
        absolute_eef = np.stack(
            [adapter.observation_from_sim(joints) for joints in target_joints]
        )

        sim = adapter.action_chunk_to_sim(absolute_eef, current)
        if not sim.success or sim.platform_actions is None or sim.ik is None:
            raise AssertionError(
                f"reachable workspace chunk failed at {sim.failed_index}: {sim.reason}"
            )
        real = adapter.action_chunk_to_real(
            absolute_eef,
            sim_radians_to_real_follower(current),
        )
        if not real.success or real.platform_actions is None:
            raise AssertionError(f"matching real conversion failed: {real.reason}")
        np.testing.assert_allclose(
            real.platform_actions,
            sim_radians_to_real_follower(sim.platform_actions),
            atol=2e-4,
        )

        joint_max = max(
            joint_max,
            float(np.max(np.abs(sim.platform_actions[:, :5] - target_joints[:, :5]))),
        )
        for step in sim.ik.steps:
            position_max = max(position_max, float(step.position_residual_m))
            orientation_max = max(orientation_max, float(step.orientation_residual_rad))
            iterations.append(int(step.iterations))

    metrics = {
        "schema_version": "so101_eef_ik_workspace_sweep_v1",
        "seed": seed,
        "chunks": chunks,
        "horizon": horizon,
        "poses": chunks * horizon,
        "failures": 0,
        "position_residual_max_m": position_max,
        "orientation_residual_max_rad": orientation_max,
        "joint_reconstruction_max_rad": joint_max,
        "ik_iterations_mean": float(np.mean(iterations)),
        "ik_iterations_max": max(iterations),
        "acceptance": {
            "position_residual_max_m": 5e-4,
            "orientation_residual_max_rad": 1e-2,
            # 5-DoF arm의 near-singular equivalent solution 차이를 포함한 sweep 상한.
            "joint_reconstruction_max_rad": 3e-2,
        },
    }
    if position_max > 5e-4 + 1e-8:
        raise AssertionError(f"position residual exceeds acceptance: {position_max}")
    if orientation_max > 1e-2 + 1e-8:
        raise AssertionError(f"orientation residual exceeds acceptance: {orientation_max}")
    if joint_max > 3e-2:
        raise AssertionError(f"joint reconstruction exceeds acceptance: {joint_max}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metrics = run_sweep(chunks=args.chunks, horizon=args.horizon, seed=args.seed)
    rendered = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    print("PASS: reachable workspace FK→sequential IK sweep")


if __name__ == "__main__":
    main()
