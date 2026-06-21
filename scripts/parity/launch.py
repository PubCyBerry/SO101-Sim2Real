#!/usr/bin/env python
"""SO-101 canonical parity 실행 경로 단일 launcher."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def _pixi() -> str:
    discovered = shutil.which("pixi")
    if discovered:
        return discovered
    runtime_root = Path(
        os.environ.get(
            "SO101_RUNTIME_ROOT",
            r"D:\SO101\isaac6_ros" if sys.platform == "win32" else "/DISK1/so101-sim2real/runtime/isaac6_ros",
        )
    )
    candidate = runtime_root / "bin" / ("pixi.exe" if sys.platform == "win32" else "pixi")
    if candidate.exists():
        return str(candidate)
    raise RuntimeError("Pixi 0.70.2 launcher를 찾을 수 없다")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=("validate", "mock-probe", "sim", "real-dry-run", "real-readback", "real-motion"),
    )
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--port", default="COM8")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--confirm-emergency-cutoff-ready", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    pixi = _pixi()
    environment = os.environ.copy()
    environment.setdefault("PYTHONUTF8", "1")
    environment.setdefault("PYTHONIOENCODING", "utf-8")
    if "ZENOH_SESSION_CONFIG_URI" not in environment:
        config = (
            Path("configs/zenoh/windows-client.json5")
            if sys.platform == "win32"
            else Path("configs/zenoh/server-client.json5")
        ).resolve()
        environment["ZENOH_SESSION_CONFIG_URI"] = str(config)

    if args.mode == "validate":
        commands = [
            [pixi, "lock", "--check"],
            [pixi, "run", "-e", "ros-tools", "core-test"],
            [pixi, "run", "-e", "real", "dataset-test"],
            [
                pixi,
                "run",
                "-e",
                "ros-tools",
                "python",
                "scripts/parity/validate_checkpoint.py",
                "--manifest",
                "configs/parity/runtime_manifest.mock.json",
                "--checkpoint",
                "configs/parity/replay_checkpoint.json",
            ],
        ]
    elif args.mode == "mock-probe":
        commands = [
            [
                pixi,
                "run",
                "-e",
                "ros-tools",
                "python",
                "-m",
                "so101_vla_runtime.integration_probe",
                "--samples",
                str(args.samples),
                "--warmup",
                "5",
                "--image-pattern",
                "gradient",
            ]
        ]
    elif args.mode == "sim":
        commands = [
            [
                pixi,
                "run",
                "-e",
                "sim",
                "python",
                "scripts/parity/run_sim_client.py",
                "--steps",
                str(args.steps),
                "--visualizer",
                "none",
            ]
        ]
    elif args.mode == "real-dry-run":
        commands = [
            [
                pixi,
                "run",
                "-e",
                "real",
                "python",
                "scripts/parity/run_real_client.py",
                "--port",
                args.port,
            ]
        ]
    elif args.mode == "real-readback":
        commands = [
            [
                pixi,
                "run",
                "-e",
                "real",
                "python",
                "scripts/parity/run_real_client.py",
                "--port",
                args.port,
                "--inspect-readback",
            ]
        ]
    else:
        if not args.enable_motion or not args.confirm_emergency_cutoff_ready:
            raise RuntimeError(
                "real-motion은 --enable-motion과 "
                "--confirm-emergency-cutoff-ready를 모두 요구한다"
            )
        commands = [
            [
                pixi,
                "run",
                "-e",
                "real",
                "python",
                "scripts/parity/run_real_client.py",
                "--port",
                args.port,
                "--steps",
                str(args.steps),
                "--enable-motion",
                "--confirm-emergency-cutoff-ready",
            ]
        ]

    for command in commands:
        result = subprocess.run(command, env=environment, check=False)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
