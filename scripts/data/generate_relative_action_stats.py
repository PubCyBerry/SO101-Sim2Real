#!/usr/bin/env python3
"""Absolute EEF LeRobot v3 dataset의 horizon-aware relative action stats 생성."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from so101_contract.eef_action_contract import (  # noqa: E402
    CANONICAL_ACTION_NAMES,
    ActionRepresentationConfig,
    resolve_eef_action_contract,
)
from so101_contract.eef_relative_stats import (  # noqa: E402
    RELATIVE_ACTION_STATS_DEFAULT_PATH,
    RelativeActionSamplingConfig,
    calculate_relative_action_stats,
    inject_relative_action_stats,
    load_relative_action_stats_profile,
    validate_relative_action_stats_artifact,
    write_relative_action_stats_profile,
)
from so101_contract.eef_relative_action import (  # noqa: E402
    matrix_to_rot6d_rows,
    relative_actions_to_absolute,
)


def _parse_delta_indices(value: str) -> tuple[int, ...]:
    text = value.strip()
    if not text:
        raise argparse.ArgumentTypeError("delta indices must not be empty")
    try:
        if ":" in text:
            parts = text.split(":")
            if len(parts) not in {2, 3}:
                raise ValueError
            start, stop = int(parts[0]), int(parts[1])
            step = int(parts[2]) if len(parts) == 3 else 1
            if step == 0:
                raise ValueError
            result = tuple(range(start, stop, step))
        else:
            result = tuple(int(token.strip()) for token in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid delta indices {value!r}; use '0,1,2' or Python range syntax '0:3'"
        ) from exc
    if not result:
        raise argparse.ArgumentTypeError(f"delta indices resolve to an empty sequence: {value!r}")
    return result


def _rotations_z(angles: np.ndarray) -> np.ndarray:
    rotations = np.zeros((len(angles), 3, 3), dtype=np.float32)
    cosine, sine = np.cos(angles), np.sin(angles)
    rotations[:, 0, 0] = cosine
    rotations[:, 0, 1] = -sine
    rotations[:, 1, 0] = sine
    rotations[:, 1, 1] = cosine
    rotations[:, 2, 2] = 1.0
    return rotations


def _write_self_check_dataset(root: Path) -> np.ndarray:
    import pyarrow as pa
    import pyarrow.parquet as pq

    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    feature = {
        "dtype": "float32",
        "shape": [10],
        "names": list(CANONICAL_ACTION_NAMES),
    }
    info = {
        "codebase_version": "v3.0",
        "fps": 30,
        "total_episodes": 2,
        "total_frames": 12,
        "features": {
            "observation.state": feature,
            "action": feature,
        },
        "so101_eef_conversion": {
            "base_frame": "base_link",
            "eef_frame": "tcp_grasp",
            "eef_kinematics_version": "so101_base_tcp_grasp_fk_v2",
            "rotation_representation": "rot6d",
            "rotation_format": "xyz+rot6d_rows",
            "gripper_format": "canonical_policy_feature_[0,100]",
            "keep_joints": False,
            "urdf_sha256": "a" * 64,
            "robot_yaml_sha256": "b" * 64,
        },
    }
    modality = {
        name: {
            "eef_9d": {"start": 0, "end": 9},
            "gripper_position": {"start": 9, "end": 10},
        }
        for name in ("state", "action")
    }
    (root / "meta" / "info.json").write_text(
        json.dumps(info, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "meta" / "modality.json").write_text(
        json.dumps(modality, indent=2) + "\n",
        encoding="utf-8",
    )

    episode_indices = np.repeat(np.arange(2, dtype=np.int64), 6)
    frame_indices = np.tile(np.arange(6, dtype=np.int64), 2)
    angles = np.linspace(-0.4, 0.7, 12, dtype=np.float32)
    rotations = _rotations_z(angles)
    states = np.concatenate(
        [
            np.stack(
                [
                    frame_indices.astype(np.float32) * 0.01,
                    episode_indices.astype(np.float32) * 0.1,
                    np.full(12, 0.2, dtype=np.float32),
                ],
                axis=1,
            ),
            matrix_to_rot6d_rows(rotations),
            np.linspace(0.0, 100.0, 12, dtype=np.float32)[:, None],
        ],
        axis=1,
    ).astype(np.float32)

    # Known relative chunk를 absolute로 합성해 dataset action을 만든다.
    relative = np.zeros((12, 1, 10), dtype=np.float32)
    relative[:, 0, :3] = np.asarray([0.01, -0.02, 0.03], dtype=np.float32)
    relative[:, 0, 3:9] = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    relative[:, 0, 9] = np.linspace(100.0, 0.0, 12, dtype=np.float32)
    actions = relative_actions_to_absolute(states, relative)[:, 0]

    table = pa.table(
        {
            "episode_index": pa.array(episode_indices),
            "frame_index": pa.array(frame_indices),
            "observation.state": pa.array(states.tolist(), type=pa.list_(pa.float32(), 10)),
            "action": pa.array(actions.tolist(), type=pa.list_(pa.float32(), 10)),
        }
    )
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")
    return relative[:, 0]


def self_check() -> None:
    scratch = REPO_ROOT / "scratch"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="eef-stats-check-", dir=scratch) as directory:
        root = Path(directory)
        expected_step0 = _write_self_check_dataset(root)
        config = ActionRepresentationConfig(mode="eef_relative")

        result_h1 = calculate_relative_action_stats(
            root,
            RelativeActionSamplingConfig(action_delta_indices=(0,)),
            config=config,
            scratch_dir=scratch,
        )
        stats_h1 = result_h1.profile["stats"]["action"]
        if stats_h1["count"] != 12:
            raise AssertionError(f"H1 window count mismatch: {stats_h1['count']}")
        expected_mean = expected_step0.mean(axis=0)
        if not np.allclose(np.asarray(stats_h1["mean"])[0], expected_mean, atol=3e-6):
            raise AssertionError("H1 relative stats mean mismatch")

        result_h3 = calculate_relative_action_stats(
            root,
            RelativeActionSamplingConfig(action_delta_indices=(0, 1, 2)),
            config=config,
            scratch_dir=scratch,
        )
        stats_h3 = result_h3.profile["stats"]["action"]
        if stats_h3["count"] != 8:
            raise AssertionError(f"H3 episode-boundary window count mismatch: {stats_h3['count']}")
        output_path, changed = write_relative_action_stats_profile(root, result_h1)
        if not changed:
            raise AssertionError("first profile write was unexpectedly a no-op")
        _, changed = write_relative_action_stats_profile(root, result_h1)
        if changed:
            raise AssertionError("identical profile write was not idempotent")
        _, changed = write_relative_action_stats_profile(root, result_h3)
        if not changed:
            raise AssertionError("second horizon profile was not added")
        artifact = json.loads(output_path.read_text(encoding="utf-8"))
        validate_relative_action_stats_artifact(artifact)
        if len(artifact["profiles"]) != 2:
            raise AssertionError("multi-horizon profiles were not preserved")
        contract = resolve_eef_action_contract(root, config)
        profile_id, loaded_h3 = load_relative_action_stats_profile(
            root,
            contract,
            RelativeActionSamplingConfig(action_delta_indices=(0, 1, 2)),
        )
        if profile_id != result_h3.profile_id or loaded_h3 != result_h3.profile:
            raise AssertionError("relative stats profile selection mismatch")
        merged = inject_relative_action_stats(
            {"observation.state": {"mean": [1.0]}, "action": {"mean": [999.0]}},
            loaded_h3,
        )
        if merged["observation.state"]["mean"] != [1.0]:
            raise AssertionError("non-action dataset stats were not preserved")
        if merged["action"] != loaded_h3["stats"]["action"]:
            raise AssertionError("relative action stats were not injected")

        tampered = json.loads(json.dumps(artifact))
        tampered["profiles"][profile_id]["stats"]["action"]["mean"][0][0] += 1.0
        try:
            validate_relative_action_stats_artifact(tampered)
        except ValueError as exc:
            if "content hash mismatch" not in str(exc):
                raise
        else:
            raise AssertionError("tampered relative stats profile was accepted")

    print("PASS: horizon-aware EEF-relative action stats")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path)
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        "--horizon",
        type=int,
        help="action delta를 0:H로 생성",
    )
    action_group.add_argument(
        "--action-delta-indices",
        type=_parse_delta_indices,
        help="'0,1,2' 또는 stop-exclusive '0:3'",
    )
    parser.add_argument(
        "--observation-delta-indices",
        type=_parse_delta_indices,
        default=(0,),
    )
    parser.add_argument("--reference-observation-index", type=int, default=-1)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument(
        "--output-file",
        default=RELATIVE_ACTION_STATS_DEFAULT_PATH,
    )
    parser.add_argument("--overwrite-profile", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.self_check:
        self_check()
        return
    if args.dataset_dir is None:
        raise SystemExit("--dataset-dir is required unless --self-check is used")
    if args.horizon is None and args.action_delta_indices is None:
        raise SystemExit("one of --horizon or --action-delta-indices is required")
    if args.horizon is not None:
        if args.horizon <= 0:
            raise SystemExit("--horizon must be positive")
        action_delta_indices = tuple(range(args.horizon))
    else:
        action_delta_indices = args.action_delta_indices

    sampling = RelativeActionSamplingConfig(
        observation_delta_indices=args.observation_delta_indices,
        action_delta_indices=action_delta_indices,
        reference_observation_index=args.reference_observation_index,
    )
    result = calculate_relative_action_stats(
        args.dataset_dir,
        sampling,
        config=ActionRepresentationConfig(mode="eef_relative"),
        max_episodes=args.max_episodes,
        scratch_dir=REPO_ROOT / "scratch",
    )
    output_path, changed = write_relative_action_stats_profile(
        args.dataset_dir,
        result,
        output_file=args.output_file,
        overwrite_profile=args.overwrite_profile,
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "changed": changed,
                "profile_id": result.profile_id,
                "production": result.profile["production"],
                "sampling": result.profile["sampling"],
                "count": result.profile["stats"]["action"]["count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
