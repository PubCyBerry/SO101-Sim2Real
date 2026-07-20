#!/usr/bin/env python3
"""SkillGen generated HDF5의 success-only·layout·frame 계약을 검증한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--report-file", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--require-images", action="store_true")
    args = parser.parse_args()

    input_path = args.input_file.resolve()
    layouts: set[tuple[float, ...]] = set()
    episode_reports = []
    with h5py.File(input_path, "r") as source:
        data = source["data"]
        names = sorted(data, key=lambda name: int(name.rsplit("_", 1)[1]))
        if len(names) != args.expected_episodes:
            raise ValueError(f"episode count {len(names)} != expected {args.expected_episodes}")
        for episode_index, name in enumerate(names):
            demo = data[name]
            frames = int(demo.attrs["num_samples"])
            if not bool(demo.attrs.get("success", False)):
                raise ValueError(f"{name}: failed episode committed")
            required = {
                "actions": (frames, 8),
                "joint_data/state": (frames, 6),
                "joint_data/target": (frames, 6),
                "timing/frame_index": (frames,),
                "timing/timestamp": (frames,),
            }
            if args.require_images:
                required.update(
                    {f"images/{cam}": (frames, 480, 640, 3) for cam in ("top", "wrist", "front")}
                )
            for key, shape in required.items():
                if key not in demo or demo[key].shape != shape:
                    raise ValueError(f"{name}/{key}: shape={getattr(demo.get(key), 'shape', None)}, expected={shape}")
            action = demo["actions"][:]
            state = demo["joint_data/state"][:]
            target = demo["joint_data/target"][:]
            if not all(np.isfinite(values).all() for values in (action, state, target)):
                raise ValueError(f"{name}: NaN/Inf")
            max_quat_error = float(np.max(np.abs(np.linalg.norm(action[:, 3:7], axis=1) - 1.0)))
            if max_quat_error > 1.0e-5:
                raise ValueError(f"{name}: quaternion norm error {max_quat_error}")
            frame_index = demo["timing/frame_index"][:]
            timestamp = demo["timing/timestamp"][:]
            if not np.array_equal(frame_index, np.arange(1, frames + 1)):
                raise ValueError(f"{name}: frame index is not 1..N")
            if not np.all(np.diff(timestamp) > 0):
                raise ValueError(f"{name}: timestamp is not monotonic")
            cube_xy = demo["initial_state/rigid_object/Cube1/root_pose"][0, :2]
            bowl_xy = demo["initial_state/rigid_object/Bowl/root_pose"][0, :2]
            layout = tuple(np.round(np.concatenate([cube_xy, bowl_xy]), 6).tolist())
            layouts.add(layout)
            episode_reports.append(
                {
                    "episode": episode_index,
                    "frames": frames,
                    "max_quaternion_norm_error": max_quat_error,
                    "layout_cube_bowl_xy": list(layout),
                }
            )

    if len(layouts) != args.expected_episodes:
        raise ValueError(f"unique layouts {len(layouts)} != episodes {args.expected_episodes}")
    report = {
        "status": "complete",
        "input_file": str(input_path),
        "episode_count": len(episode_reports),
        "unique_layout_count": len(layouts),
        "success_only": True,
        "require_images": args.require_images,
        "episodes": episode_reports,
    }
    report_path = args.report_file.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
