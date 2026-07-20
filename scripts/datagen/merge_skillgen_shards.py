#!/usr/bin/env python3
"""검증된 SkillGen HDF5 shard를 resume 가능한 단일 full HDF5로 병합한다."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np


def _demo_names(group) -> list[str]:
    return sorted(group, key=lambda name: int(name.split("_")[-1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--report-file", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    shards = sorted(
        path
        for path in args.input_dir.resolve().glob("shard_*.hdf5")
        if not path.stem.endswith("_failed")
    )
    if not shards:
        raise FileNotFoundError(f"no shards found: {args.input_dir}")

    output_path = args.output_file.resolve()
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    if output_path.exists():
        raise FileExistsError(f"final output already exists: {output_path}")
    if partial_path.exists() and not args.resume:
        raise FileExistsError(f"partial merge exists; pass --resume: {partial_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_file.resolve().parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if partial_path.exists() else "w"
    origins: list[dict] = []
    layout_keys: set[tuple[float, ...]] = set()
    global_index = 0
    with h5py.File(partial_path, mode) as output:
        output_data = output.require_group("data")
        for temporary in [name for name in output_data if name.startswith("_copying_demo_")]:
            del output_data[temporary]

        first_env_args = output_data.attrs.get("env_args")
        total_frames = 0
        for shard in shards:
            with h5py.File(shard, "r") as source:
                source_data = source["data"]
                env_args = source_data.attrs["env_args"]
                if first_env_args is None:
                    first_env_args = env_args
                    output_data.attrs["env_args"] = env_args
                elif env_args != first_env_args:
                    raise ValueError(f"env_args mismatch: {shard}")

                for source_name in _demo_names(source_data):
                    source_demo = source_data[source_name]
                    target_name = f"demo_{global_index}"
                    sample_count = int(source_demo.attrs["num_samples"])
                    if not bool(source_demo.attrs.get("success", False)):
                        raise ValueError(f"non-success episode in canonical shard: {shard}:{source_name}")
                    required = (
                        "actions",
                        "joint_data/state",
                        "joint_data/target",
                        "images/top",
                        "images/wrist",
                        "images/front",
                    )
                    for key in required:
                        if key not in source_demo or int(source_demo[key].shape[0]) != sample_count:
                            raise ValueError(f"invalid frame contract: {shard}:{source_name}/{key}")
                    cube_xy = source_demo["initial_state/rigid_object/Cube1/root_pose"][0, :2]
                    bowl_xy = source_demo["initial_state/rigid_object/Bowl/root_pose"][0, :2]
                    layout_key = tuple(np.round(np.concatenate([cube_xy, bowl_xy]), 6).tolist())
                    if layout_key in layout_keys:
                        raise ValueError(f"duplicate initial layout: {shard}:{source_name}")
                    layout_keys.add(layout_key)
                    if target_name in output_data:
                        target_demo = output_data[target_name]
                        if int(target_demo.attrs["num_samples"]) != sample_count:
                            raise ValueError(f"resume sample mismatch: {target_name}")
                        if target_demo.attrs.get("skillgen_origin_shard") != shard.name:
                            raise ValueError(f"resume origin shard mismatch: {target_name}")
                        if target_demo.attrs.get("skillgen_origin_demo") != source_name:
                            raise ValueError(f"resume origin demo mismatch: {target_name}")
                    else:
                        temporary_name = f"_copying_{target_name}"
                        source.copy(source_demo, output_data, name=temporary_name)
                        copied = output_data[temporary_name]
                        copied.attrs["skillgen_origin_shard"] = shard.name
                        copied.attrs["skillgen_origin_demo"] = source_name
                        output.flush()
                        output_data.move(temporary_name, target_name)
                        output.flush()
                    origins.append(
                        {
                            "episode": global_index,
                            "shard": shard.name,
                            "source_demo": source_name,
                            "frames": sample_count,
                        }
                    )
                    total_frames += sample_count
                    global_index += 1
        output_data.attrs["total"] = total_frames
        output.flush()

    if global_index != args.expected_episodes:
        raise ValueError(f"episode count mismatch: {global_index} != {args.expected_episodes}")
    os.replace(partial_path, output_path)
    report = {
        "status": "complete",
        "input_dir": str(args.input_dir.resolve()),
        "output_file": str(output_path),
        "shard_count": len(shards),
        "episode_count": global_index,
        "unique_layout_count": len(layout_keys),
        "frame_count": total_frames,
        "episode_origins": origins,
    }
    args.report_file.resolve().write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "episode_origins"}, indent=2))


if __name__ == "__main__":
    main()
