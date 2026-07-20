#!/usr/bin/env python3
"""SkillGen/canonical LeRobot action의 joint limit·속도·tracking 품질을 검증한다."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

from sim_to_real.data.lerobot_units import FPS, JOINT_FEATURE_NAMES, to_lerobot_units
from so101_contract.leader_calibration import SO101_FOLLOWER_USD_JOINT_LIMITS


JOINT_NAMES = tuple(name.removesuffix(".pos") for name in JOINT_FEATURE_NAMES)
ACTION_LOWER = np.asarray(
    [SO101_FOLLOWER_USD_JOINT_LIMITS[name][0] for name in JOINT_NAMES[:5]] + [0.0],
    dtype=np.float64,
)
ACTION_UPPER = np.asarray(
    [SO101_FOLLOWER_USD_JOINT_LIMITS[name][1] for name in JOINT_NAMES[:5]] + [100.0],
    dtype=np.float64,
)
ARM_MAX_VELOCITY_RAD_S = 5.0
GRIPPER_MAX_VELOCITY_RAD_S = 2.5
MAX_DELTA = np.asarray(
    [ARM_MAX_VELOCITY_RAD_S * 180.0 / math.pi / FPS] * 5
    + [GRIPPER_MAX_VELOCITY_RAD_S * 180.0 / math.pi / FPS * 100.0 / 110.0],
    dtype=np.float64,
)


def _demo_names(data: h5py.Group) -> list[str]:
    return sorted(
        (name for name in data if name.startswith("demo_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )


def _hdf5_episodes(path: Path) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    with h5py.File(path, "r") as source:
        data = source["data"]
        for episode_index, name in enumerate(_demo_names(data)):
            demo = data[name]
            raw_action = demo["actions"]
            if raw_action.ndim != 2 or raw_action.shape[1] != 8:
                raise ValueError(f"{name}/actions must be 8D, got {raw_action.shape}")
            target = demo["joint_data/target"][:]
            state = demo["joint_data/state"][:]
            if target.shape != state.shape or target.ndim != 2 or target.shape[1] != 6:
                raise ValueError(f"{name}: invalid target/state shapes {target.shape}/{state.shape}")
            yield episode_index, to_lerobot_units(target), to_lerobot_units(state)


def _lerobot_episodes(root: Path) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    files = sorted((root / "data").glob("**/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no LeRobot parquet files under {root / 'data'}")
    table = pa.concat_tables([pq.read_table(path) for path in files])
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    episode_indices = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
    frame_indices = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
    indices = np.asarray(table["index"].to_numpy(), dtype=np.int64)
    if not np.array_equal(indices, np.arange(len(indices), dtype=np.int64)):
        raise ValueError("LeRobot global index is not contiguous and ordered")
    unique = np.unique(episode_indices)
    if not np.array_equal(unique, np.arange(len(unique), dtype=np.int64)):
        raise ValueError(f"LeRobot episode indices are not contiguous: {unique.tolist()}")
    for episode_index in unique:
        mask = episode_indices == episode_index
        expected_frames = np.arange(int(mask.sum()), dtype=np.int64)
        if not np.array_equal(frame_indices[mask], expected_frames):
            raise ValueError(f"episode {episode_index}: frame_index is not contiguous")
        yield int(episode_index), actions[mask], states[mask]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-hdf5", type=Path)
    source.add_argument("--lerobot-root", type=Path)
    parser.add_argument("--report-file", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--tolerance", type=float, default=1.0e-3)
    args = parser.parse_args()

    if args.tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if args.input_hdf5:
        input_path = args.input_hdf5.resolve()
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        episodes = _hdf5_episodes(input_path)
        source_kind = "skillgen_hdf5"
    else:
        input_path = args.lerobot_root.resolve()
        if not input_path.is_dir():
            raise FileNotFoundError(input_path)
        episodes = _lerobot_episodes(input_path)
        source_kind = "lerobot_v3"

    episode_count = 0
    frame_count = 0
    limit_counts = np.zeros(6, dtype=np.int64)
    velocity_counts = np.zeros(6, dtype=np.int64)
    max_limit_excess = np.zeros(6, dtype=np.float64)
    max_observed_delta = np.zeros(6, dtype=np.float64)
    tracking_errors: list[np.ndarray] = []
    examples: list[dict] = []

    for episode_index, action, state in episodes:
        if action.shape != state.shape or action.ndim != 2 or action.shape[1] != 6:
            raise ValueError(f"episode {episode_index}: invalid action/state shape")
        if not np.isfinite(action).all() or not np.isfinite(state).all():
            raise ValueError(f"episode {episode_index}: NaN or infinity")
        episode_count += 1
        frame_count += action.shape[0]
        tracking_errors.append(np.abs(action.astype(np.float64) - state.astype(np.float64)))

        lower_excess = np.maximum(ACTION_LOWER - action, 0.0)
        upper_excess = np.maximum(action - ACTION_UPPER, 0.0)
        excess = np.maximum(lower_excess, upper_excess)
        limit_mask = excess > args.tolerance
        limit_counts += limit_mask.sum(axis=0)
        max_limit_excess = np.maximum(max_limit_excess, excess.max(axis=0))

        if action.shape[0] > 1:
            delta = np.abs(np.diff(action.astype(np.float64), axis=0))
            velocity_mask = delta > (MAX_DELTA + args.tolerance)
            velocity_counts += velocity_mask.sum(axis=0)
            max_observed_delta = np.maximum(max_observed_delta, delta.max(axis=0))
            if len(examples) < 20:
                for frame_offset, joint_index in np.argwhere(velocity_mask):
                    examples.append(
                        {
                            "type": "velocity",
                            "episode": episode_index,
                            "frame_from": int(frame_offset),
                            "joint": JOINT_NAMES[int(joint_index)],
                            "delta": float(delta[frame_offset, joint_index]),
                            "allowed": float(MAX_DELTA[joint_index]),
                        }
                    )
                    if len(examples) >= 20:
                        break
        if len(examples) < 20:
            for frame_index, joint_index in np.argwhere(limit_mask):
                examples.append(
                    {
                        "type": "joint_limit",
                        "episode": episode_index,
                        "frame": int(frame_index),
                        "joint": JOINT_NAMES[int(joint_index)],
                        "value": float(action[frame_index, joint_index]),
                        "allowed": [
                            float(ACTION_LOWER[joint_index]),
                            float(ACTION_UPPER[joint_index]),
                        ],
                    }
                )
                if len(examples) >= 20:
                    break

    if args.expected_episodes is not None and episode_count != args.expected_episodes:
        raise ValueError(f"episode count {episode_count} != expected {args.expected_episodes}")
    if episode_count == 0:
        raise ValueError("no episodes")

    tracking = np.concatenate(tracking_errors, axis=0)
    total_limit_violations = int(limit_counts.sum())
    total_velocity_violations = int(velocity_counts.sum())
    status = "complete" if total_limit_violations == 0 and total_velocity_violations == 0 else "failed"
    report = {
        "status": status,
        "source_kind": source_kind,
        "input": str(input_path),
        "episodes": episode_count,
        "frames": frame_count,
        "fps": FPS,
        "joint_names": list(JOINT_NAMES),
        "action_limits": {
            "lower": ACTION_LOWER.tolist(),
            "upper": ACTION_UPPER.tolist(),
            "tolerance": args.tolerance,
            "violation_count": total_limit_violations,
            "violation_count_per_joint": limit_counts.tolist(),
            "max_excess_per_joint": max_limit_excess.tolist(),
        },
        "velocity_limits": {
            "arm_rad_s": ARM_MAX_VELOCITY_RAD_S,
            "gripper_rad_s": GRIPPER_MAX_VELOCITY_RAD_S,
            "max_delta_per_frame_policy_feature": MAX_DELTA.tolist(),
            "tolerance": args.tolerance,
            "violation_count": total_velocity_violations,
            "violation_count_per_joint": velocity_counts.tolist(),
            "max_observed_delta_per_joint": max_observed_delta.tolist(),
        },
        "tracking_error_abs_policy_feature": {
            "mean_per_joint": tracking.mean(axis=0).tolist(),
            "p95_per_joint": np.quantile(tracking, 0.95, axis=0).tolist(),
            "max_per_joint": tracking.max(axis=0).tolist(),
        },
        "violation_examples": examples,
    }
    report_path = args.report_file.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if status != "complete":
        raise RuntimeError(
            f"action QA failed: joint_limit={total_limit_violations}, "
            f"velocity={total_velocity_violations}"
        )


if __name__ == "__main__":
    main()
