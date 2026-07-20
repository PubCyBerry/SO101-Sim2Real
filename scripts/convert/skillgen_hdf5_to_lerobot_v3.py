#!/usr/bin/env python3
"""SkillGen HDF5를 SO-101 canonical LeRobot v3 데이터셋으로 변환한다.

SkillGen의 8D EE action은 학습 action으로 사용하지 않는다. 환경에 실제 적용된
``joint_data/target``과 관측된 ``joint_data/state``만 canonical 6D PolicyFeature v1
(arm degree, gripper [0, 100])로 변환한다.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import h5py
import numpy as np

from sim_to_real.data.lerobot_recorder import LeRobotV3DatasetWriter
from sim_to_real.data.lerobot_units import (
    CAMERA_KEYS,
    FPS,
    IMAGE_CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    to_lerobot_units,
)


TASK_NAME = "pick up the cube and place it in the bowl"


def _demo_names(data: h5py.Group) -> list[str]:
    names = [name for name in data if name.startswith("demo_")]
    return sorted(names, key=lambda name: int(name.rsplit("_", 1)[1]))


def _validate_demo(demo: h5py.Group, name: str) -> int:
    required = ["joint_data/state", "joint_data/target", *(f"images/{cam}" for cam in CAMERA_KEYS)]
    missing = [key for key in required if key not in demo]
    if missing:
        raise ValueError(f"{name}: required datasets missing: {missing}")

    lengths = {key: int(demo[key].shape[0]) for key in required}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{name}: frame lengths differ: {lengths}")
    frames = next(iter(lengths.values()))
    if frames <= 0:
        raise ValueError(f"{name}: empty episode")
    if int(demo.attrs.get("num_samples", -1)) != frames:
        raise ValueError(
            f"{name}: num_samples={demo.attrs.get('num_samples')} does not match frames={frames}"
        )
    if not bool(demo.attrs.get("success", False)):
        raise ValueError(f"{name}: non-success episode is forbidden in canonical output")

    for key in ("joint_data/state", "joint_data/target"):
        dataset = demo[key]
        if dataset.shape != (frames, 6):
            raise ValueError(f"{name}/{key}: shape={dataset.shape}, expected ({frames}, 6)")
        # 전체 joint array는 작으므로 여기서 finite를 엄격히 검사한다.
        if not np.isfinite(dataset[:]).all():
            raise ValueError(f"{name}/{key}: NaN or infinity detected")

    expected_image_shape = (frames, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
    for cam in CAMERA_KEYS:
        dataset = demo[f"images/{cam}"]
        if dataset.shape != expected_image_shape:
            raise ValueError(
                f"{name}/images/{cam}: shape={dataset.shape}, expected {expected_image_shape}"
            )
        if dataset.dtype != np.uint8:
            raise ValueError(f"{name}/images/{cam}: dtype={dataset.dtype}, expected uint8")
    return frames


def _safe_remove_output(path: Path) -> None:
    resolved = path.resolve()
    unsafe = {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve()}
    if resolved in unsafe or len(resolved.parts) < 4:
        raise ValueError(f"refusing to remove unsafe output: {resolved}")
    shutil.rmtree(resolved)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-file", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--max-episodes", type=int, help="변환기 smoke test용 앞 N개 제한")
    parser.add_argument("--task-name", default=TASK_NAME)
    parser.add_argument("--video-codec", choices=("libx264", "h264_nvenc"), default="h264_nvenc")
    parser.add_argument("--video-preset", default="p4")
    parser.add_argument("--video-quality", type=int, default=8)
    parser.add_argument("--video-nvenc-cq", type=int, default=23)
    parser.add_argument("--ffmpeg-exe", default="/usr/bin/ffmpeg")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_path = args.input_file.resolve()
    output_path = args.output_dir.resolve()
    partial_path = output_path.with_name(output_path.name + ".partial")
    report_path = args.report_file.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if args.max_episodes is not None and args.max_episodes < 1:
        raise ValueError("max-episodes must be positive")
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(output_path)
        _safe_remove_output(output_path)
    if partial_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"partial output exists; pass --overwrite: {partial_path}")
        _safe_remove_output(partial_path)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    input_episode_count = 0
    selected_names: list[str] = []
    input_frame_count = 0
    episode_frames: list[int] = []

    writer = LeRobotV3DatasetWriter(
        partial_path,
        overwrite=False,
        enable_videos=True,
        video_quality=args.video_quality,
        video_preset=args.video_preset,
        video_codec=args.video_codec,
        video_ffmpeg_exe=args.ffmpeg_exe,
        video_nvenc_cq=args.video_nvenc_cq,
    )
    try:
        with h5py.File(input_path, "r") as source:
            if "data" not in source:
                raise ValueError(f"missing HDF5 group: {input_path}:/data")
            data = source["data"]
            all_names = _demo_names(data)
            input_episode_count = len(all_names)
            selected_names = all_names[: args.max_episodes] if args.max_episodes else all_names
            if args.expected_episodes is not None and len(selected_names) != args.expected_episodes:
                raise ValueError(
                    f"selected episode count {len(selected_names)} != expected {args.expected_episodes}"
                )

            for output_episode, name in enumerate(selected_names):
                demo = data[name]
                frames = _validate_demo(demo, name)
                state = to_lerobot_units(demo["joint_data/state"][:])
                action = to_lerobot_units(demo["joint_data/target"][:])
                for frame_index in range(frames):
                    images = {
                        cam: np.ascontiguousarray(demo[f"images/{cam}"][frame_index])
                        for cam in CAMERA_KEYS
                    }
                    writer.add_frame(action[frame_index], state[frame_index], images)
                if not writer.commit_episode(success=True, task_name=args.task_name):
                    raise RuntimeError(f"writer did not commit {name}")
                episode_frames.append(frames)
                input_frame_count += frames
                print(
                    f"[convert] {output_episode + 1}/{len(selected_names)} "
                    f"source={name} frames={frames}",
                    flush=True,
                )
        writer_summary = writer.finalize(task_name=args.task_name)
    except BaseException:
        # 실패해도 ffmpeg child와 파일 descriptor를 닫아 partial을 재실행 가능하게 남긴다.
        writer._close_video_writers()
        raise

    if writer_summary["total_episodes"] != len(selected_names):
        raise RuntimeError(f"writer episode mismatch: {writer_summary}")
    if writer_summary["total_frames"] != input_frame_count:
        raise RuntimeError(f"writer frame mismatch: {writer_summary}")
    os.replace(partial_path, output_path)

    report = {
        "status": "complete",
        "input_file": str(input_path),
        "output_dir": str(output_path),
        "input_episode_count": input_episode_count,
        "converted_episode_count": len(selected_names),
        "input_frame_count": input_frame_count,
        "output_frame_count": int(writer_summary["total_frames"]),
        "episode_frame_min": min(episode_frames),
        "episode_frame_max": max(episode_frames),
        "fps": FPS,
        "cameras": list(CAMERA_KEYS),
        "camera_shape": [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS],
        "action_source": "joint_data/target",
        "state_source": "joint_data/state",
        "source_action_dim": 8,
        "canonical_action_dim": 6,
        "video_codec": args.video_codec,
        "video_preset": args.video_preset,
        "video_nvenc_cq": args.video_nvenc_cq,
        "task_name": args.task_name,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
