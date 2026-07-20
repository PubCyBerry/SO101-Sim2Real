#!/usr/bin/env python3
"""자동 SM source/annotated HDF5의 8D·boundary·physics 계약을 독립 검증한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


SIGNALS = (
    ("start", "grasp_cube"),
    ("term", "grasp_cube"),
    ("start", "place_cube"),
    ("term", "place_cube"),
)


def _demo_names(data: h5py.Group) -> list[str]:
    return sorted(data, key=lambda name: int(name.rsplit("_", 1)[1]))


def _transition_index(values: np.ndarray) -> int:
    binary = values.reshape(-1) > 0.5
    transitions = np.flatnonzero(binary & ~np.r_[False, binary[:-1]])
    if transitions.size != 1:
        raise ValueError(f"expected exactly one 0->1 transition, got {transitions.tolist()}")
    return int(transitions[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--annotated-file", type=Path, required=True)
    parser.add_argument("--report-file", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=10)
    args = parser.parse_args()

    source_path = args.source_file.resolve()
    annotated_path = args.annotated_file.resolve()
    episode_reports = []
    layouts = set()
    with h5py.File(source_path, "r") as source, h5py.File(annotated_path, "r") as annotated:
        source_names = _demo_names(source["data"])
        annotated_names = _demo_names(annotated["data"])
        if source_names != annotated_names:
            raise ValueError("source/annotated episode names differ")
        if len(source_names) != args.expected_episodes:
            raise ValueError(f"episode count {len(source_names)} != expected {args.expected_episodes}")
        for episode_index, name in enumerate(source_names):
            source_demo = source["data"][name]
            demo = annotated["data"][name]
            if "obs/datagen_info/subtask_start_signals" in source_demo or "obs/datagen_info/subtask_term_signals" in source_demo:
                raise ValueError(f"{name}: source file unexpectedly contains annotation signals")
            frames = int(demo.attrs["num_samples"])
            if not bool(source_demo.attrs.get("success")) or not bool(demo.attrs.get("success")):
                raise ValueError(f"{name}: non-success episode committed")
            action = demo["actions"][:]
            if action.shape != (frames, 8) or not np.isfinite(action).all():
                raise ValueError(f"{name}: invalid action shape/finite")
            quat_norm = np.linalg.norm(action[:, 3:7], axis=1)
            max_quat_error = float(np.max(np.abs(quat_norm - 1.0)))
            if max_quat_error > 1.0e-5:
                raise ValueError(f"{name}: quaternion norm error {max_quat_error}")
            frame_index = demo["frame_index"][:]
            timestamp = demo["timestamp"][:]
            if not np.array_equal(frame_index, np.arange(frames)):
                raise ValueError(f"{name}: non-monotonic frame_index")
            if not np.all(np.diff(timestamp) > 0):
                raise ValueError(f"{name}: non-monotonic timestamp")

            required_arrays = (
                "obs/policy",
                "obs/datagen_info/eef_pose/so101",
                "obs/datagen_info/target_eef_pose/so101",
                "obs/datagen_info/object_pose/Cube1",
                "obs/datagen_info/object_pose/Bowl",
            )
            for key in required_arrays:
                values = demo[key][:]
                if values.shape[0] != frames or not np.isfinite(values).all():
                    raise ValueError(f"{name}/{key}: invalid frame count or non-finite")

            boundaries = {}
            for kind, signal in SIGNALS:
                key = f"obs/datagen_info/subtask_{kind}_signals/{signal}"
                boundaries[f"{signal}_{kind}"] = _transition_index(demo[key][:])
            ordered = [
                boundaries["grasp_cube_start"],
                boundaries["grasp_cube_term"],
                boundaries["place_cube_start"],
                boundaries["place_cube_term"],
            ]
            if not (0 <= ordered[0] < ordered[1] < ordered[2] < ordered[3] < frames):
                raise ValueError(f"{name}: boundary order invalid: {ordered}")

            cube_states = demo["states/rigid_object/Cube1/root_pose"][:, :3]
            bowl_states = demo["states/rigid_object/Bowl/root_pose"][:, :3]
            grasp_term = boundaries["grasp_cube_term"]
            place_term = boundaries["place_cube_term"]
            lift_window_end = min(frames, grasp_term + 90)
            lifted_after_grasp = (
                float(cube_states[grasp_term:lift_window_end, 2].max() - cube_states[0, 2]) >= 0.04
            )
            gripper_closed_at_grasp = bool(action[grasp_term, -1] < 0.0)
            gripper_open_at_place = bool(action[place_term, -1] > 0.0)
            cube_bowl_distance = float(
                np.linalg.norm(cube_states[place_term, :2] - bowl_states[place_term, :2])
            )
            cube_in_bowl_at_place = cube_bowl_distance < 0.05
            if not (lifted_after_grasp and gripper_closed_at_grasp):
                raise ValueError(f"{name}: grasp boundary lacks physical grasp evidence")
            if not (gripper_open_at_place and cube_in_bowl_at_place):
                raise ValueError(f"{name}: place boundary lacks release/in-bowl evidence")

            layout = tuple(np.round(demo["initial_state/rigid_object/Cube1/root_pose"][0, :2], 6)) + tuple(
                np.round(demo["initial_state/rigid_object/Bowl/root_pose"][0, :2], 6)
            )
            layouts.add(layout)
            episode_reports.append(
                {
                    "episode": episode_index,
                    "name": name,
                    "seed": int(demo.attrs["seed"]),
                    "frames": frames,
                    "boundaries": boundaries,
                    "max_quaternion_norm_error": max_quat_error,
                    "lifted_after_grasp": lifted_after_grasp,
                    "gripper_closed_at_grasp": gripper_closed_at_grasp,
                    "gripper_open_at_place": gripper_open_at_place,
                    "cube_bowl_xy_distance_at_place_m": cube_bowl_distance,
                }
            )

    if len(layouts) != args.expected_episodes:
        raise ValueError(f"unique layouts {len(layouts)} != episodes {args.expected_episodes}")
    report = {
        "status": "complete",
        "source_file": str(source_path),
        "annotated_file": str(annotated_path),
        "episode_count": len(episode_reports),
        "unique_layout_count": len(layouts),
        "source_action_contract": "[x,y,z,qw,qx,qy,qz,gripper_binary] robot-base absolute",
        "subtask_signals": ["grasp_cube", "place_cube"],
        "annotation_method": "automatic State Machine phase and physics; annotate_demos.py --auto unused",
        "episodes": episode_reports,
    }
    report_path = args.report_file.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
