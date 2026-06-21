#!/usr/bin/env python
"""Pixi environment와 고정 runtime stack을 fail-closed 검사한다."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

EXPECTED = {
    "python": (3, 12),
    "isaacsim": "6.0.0.1",
    "isaaclab": "6.1.11",
    "isaaclab_commit": "28a37cecdd433c22d9eabd6a5954add9f13a8951",
    "lerobot": "0.5.1",
    "torch": "2.10.0+cu128",
    "torch_cuda": "12.8",
}


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _command(args: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(args, capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _check_equal(errors: list[str], name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        errors.append(f"{name}: actual={actual!r}, expected={expected!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", choices=("sim", "real", "ros-tools"), required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    errors: list[str] = []
    _check_equal(errors, "python", sys.version_info[:2], EXPECTED["python"])
    lock_path = args.root / "pixi.lock"
    lock_text = lock_path.read_text(encoding="utf-8") if lock_path.exists() else ""
    if EXPECTED["isaaclab_commit"] not in lock_text:
        errors.append("pixi.lock에 고정 Isaac Lab commit이 없다")

    report: dict[str, Any] = {
        "environment": args.environment,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pixi_environment": os.getenv("PIXI_ENVIRONMENT_NAME"),
        "packages": {},
        "checks": {},
    }

    rclpy_version = _version("rclpy")
    rmw_prefix = _command(["ros2", "pkg", "prefix", "rmw_zenoh_cpp"])
    report["packages"]["rclpy"] = rclpy_version
    report["checks"]["rmw_zenoh_cpp"] = rmw_prefix
    if rclpy_version is None or not rmw_prefix["ok"]:
        errors.append("ROS 2 Jazzy rclpy/rmw_zenoh_cpp를 확인할 수 없다")

    if args.environment == "sim":
        versions = {
            "isaacsim": _version("isaacsim"),
            "isaaclab": _version("isaaclab"),
            "isaaclab-physx": _version("isaaclab-physx"),
            "isaaclab-tasks": _version("isaaclab-tasks"),
            "torch": _version("torch"),
        }
        report["packages"].update(versions)
        _check_equal(errors, "isaacsim", versions["isaacsim"], EXPECTED["isaacsim"])
        _check_equal(errors, "isaaclab", versions["isaaclab"], EXPECTED["isaaclab"])
        _check_equal(errors, "torch", versions["torch"], EXPECTED["torch"])
        try:
            import torch

            report["torch_cuda"] = torch.version.cuda
            _check_equal(errors, "torch CUDA", torch.version.cuda, EXPECTED["torch_cuda"])
        except ImportError as exc:
            errors.append(f"torch import 실패: {exc}")
        report["checks"]["nvidia_smi"] = _command(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
        )
        if not report["checks"]["nvidia_smi"]["ok"]:
            errors.append("nvidia-smi GPU/driver 확인 실패")

    if args.environment == "real":
        versions = {"lerobot": _version("lerobot"), "torch": _version("torch")}
        report["packages"].update(versions)
        _check_equal(errors, "lerobot", versions["lerobot"], EXPECTED["lerobot"])
        _check_equal(errors, "torch", versions["torch"], EXPECTED["torch"])

    report["ok"] = not errors
    report["errors"] = errors
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
