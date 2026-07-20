#!/usr/bin/env python3
"""LeRobot v3의 top/wrist/front H.264 비디오를 ffprobe/ffmpeg로 검증한다."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path
import subprocess


CAMERAS = ("top", "wrist", "front")


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True)


def _probe(path: Path, ffprobe: str) -> dict:
    result = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames,duration",
            "-of",
            "json",
            str(path),
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"{path}: expected one video stream, got {len(streams)}")
    return streams[0]


def _rate(value: str) -> float:
    return float(Fraction(value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--report-file", type=Path, required=True)
    parser.add_argument("--ffprobe", default="/usr/bin/ffprobe")
    parser.add_argument("--ffmpeg", default="/usr/bin/ffmpeg")
    parser.add_argument(
        "--full-decode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="모든 frame을 decode하여 bitstream 오류를 검사",
    )
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    expected_frames = int(info["total_frames"])
    expected_fps = int(info["fps"])
    reports: dict[str, dict] = {}
    errors: list[str] = []

    for camera in CAMERAS:
        directory = root / "videos" / f"observation.images.{camera}"
        files = sorted(directory.glob("**/file-*.mp4"))
        if not files:
            errors.append(f"{camera}: video file missing under {directory}")
            continue
        camera_frames = 0
        file_reports = []
        for path in files:
            stream = _probe(path, args.ffprobe)
            frame_text = stream.get("nb_read_frames") or stream.get("nb_frames")
            if frame_text in (None, "N/A"):
                errors.append(f"{camera}/{path.name}: frame count unavailable")
                frames = 0
            else:
                frames = int(frame_text)
            camera_frames += frames
            fps = _rate(stream["avg_frame_rate"])
            file_errors = []
            if stream.get("codec_name") != "h264":
                file_errors.append(f"codec={stream.get('codec_name')} expected h264")
            if int(stream.get("width", -1)) != 640 or int(stream.get("height", -1)) != 480:
                file_errors.append(
                    f"resolution={stream.get('width')}x{stream.get('height')} expected 640x480"
                )
            if abs(fps - expected_fps) > 1.0e-6:
                file_errors.append(f"fps={fps} expected {expected_fps}")
            decode_error = ""
            if args.full_decode:
                decoded = _run(
                    [
                        args.ffmpeg,
                        "-v",
                        "error",
                        "-i",
                        str(path),
                        "-map",
                        "0:v:0",
                        "-f",
                        "null",
                        "-",
                    ]
                )
                if decoded.returncode != 0 or decoded.stderr.strip():
                    decode_error = decoded.stderr.strip() or f"exit code {decoded.returncode}"
                    file_errors.append(f"decode failed: {decode_error}")
            errors.extend(f"{camera}/{path.name}: {message}" for message in file_errors)
            file_reports.append(
                {
                    "path": str(path),
                    "frames": frames,
                    "codec": stream.get("codec_name"),
                    "pix_fmt": stream.get("pix_fmt"),
                    "width": int(stream.get("width", -1)),
                    "height": int(stream.get("height", -1)),
                    "fps": fps,
                    "duration": float(stream.get("duration", 0.0)),
                    "full_decode": bool(args.full_decode),
                    "decode_error": decode_error,
                }
            )
        if camera_frames != expected_frames:
            errors.append(f"{camera}: frames={camera_frames}, expected={expected_frames}")
        reports[camera] = {"frames": camera_frames, "files": file_reports}

    report = {
        "status": "complete" if not errors else "failed",
        "dataset_root": str(root),
        "expected_frames_per_camera": expected_frames,
        "expected_fps": expected_fps,
        "required_cameras": list(CAMERAS),
        "full_decode": bool(args.full_decode),
        "cameras": reports,
        "errors": errors,
    }
    report_path = args.report_file.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise RuntimeError(f"video QA failed with {len(errors)} error(s)")


if __name__ == "__main__":
    main()
