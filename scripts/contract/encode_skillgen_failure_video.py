#!/usr/bin/env python3
"""SkillGen failed HDF5의 top/wrist/front를 대표 합성 H.264 evidence로 인코딩한다."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import h5py
import imageio.v2 as imageio
import numpy as np


CAMERAS = ("top", "wrist", "front")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--report-file", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--ffmpeg-exe", default="/usr/bin/ffmpeg")
    parser.add_argument("--video-codec", choices=("libx264", "h264_nvenc"), default="h264_nvenc")
    args = parser.parse_args()

    input_path = args.input_file.resolve()
    output_path = args.output_video.resolve()
    report_path = args.report_file.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["IMAGEIO_FFMPEG_EXE"] = args.ffmpeg_exe

    with h5py.File(input_path, "r") as source:
        names = sorted(source["data"], key=lambda name: int(name.rsplit("_", 1)[1]))
        if args.episode >= len(names):
            raise IndexError(f"episode {args.episode} >= failed episodes {len(names)}")
        name = names[args.episode]
        demo = source["data"][name]
        if bool(demo.attrs.get("success", True)):
            raise ValueError(f"{name} is not marked failed")
        lengths = {cam: int(demo[f"images/{cam}"].shape[0]) for cam in CAMERAS}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"camera frame mismatch: {lengths}")
        frames = next(iter(lengths.values()))
        params = ["-pix_fmt", "yuv420p"]
        quality = 7
        if args.video_codec == "h264_nvenc":
            params = ["-preset", "p4", "-cq", "23", *params]
            quality = None
        writer = imageio.get_writer(
            output_path,
            fps=30,
            codec=args.video_codec,
            quality=quality,
            macro_block_size=1,
            ffmpeg_params=params,
        )
        try:
            for frame in range(frames):
                composite = np.concatenate(
                    [np.asarray(demo[f"images/{cam}"][frame], dtype=np.uint8) for cam in CAMERAS],
                    axis=1,
                )
                writer.append_data(np.ascontiguousarray(composite))
        finally:
            writer.close()

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    numerator, denominator = (int(value) for value in stream["r_frame_rate"].split("/"))
    encoded_fps = numerator / denominator
    encoded_frames = int(stream["nb_read_frames"])
    expected = {
        "codec_name": "h264",
        "width": 1920,
        "height": 480,
        "fps": 30.0,
        "frames": frames,
    }
    actual = {
        "codec_name": stream["codec_name"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": encoded_fps,
        "frames": encoded_frames,
    }
    if actual != expected:
        raise ValueError(f"encoded video contract mismatch: actual={actual}, expected={expected}")
    decode = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(output_path), "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    if decode.returncode != 0:
        raise RuntimeError(f"full decode failed: {decode.stderr.strip()}")

    report = {
        "status": "complete",
        "input_file": str(input_path),
        "failed_episode": args.episode,
        "source_name": name,
        "frames": frames,
        "fps": 30,
        "camera_order": list(CAMERAS),
        "resolution": [480, 1920],
        "video_codec": args.video_codec,
        "output_video": str(output_path),
        "ffprobe": actual,
        "full_decode": True,
        "decode_error": "",
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
