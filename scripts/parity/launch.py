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


def _pixi_command(subcommand: str, *arguments: str) -> list[str]:
    command = [_pixi(), subcommand]
    if sys.platform == "win32":
        project_root = Path(__file__).resolve().parents[2]
        manifest = project_root / "pixi.toml"
        lock = project_root / "pixi.lock"
        if not manifest.is_file() or not lock.is_file():
            raise RuntimeError(
                f"tracked Pixi manifest/lock을 찾을 수 없다: {project_root}"
            )
        command += ["--manifest-path", str(manifest)]
    if subcommand == "run":
        command.append("--frozen")
    return [*command, *arguments]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=("validate", "mock-probe", "sim", "real-dry-run", "real-readback", "real-motion"),
    )
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--port", default="COM8")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument(
        "--sim-device",
        default="cpu" if sys.platform == "win32" else "cuda:0",
        choices=("cpu", "cuda:0"),
        help="단일-env parity 권장값: Windows=cpu, Linux=cuda:0",
    )
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--confirm-emergency-cutoff-ready", action="store_true")
    parser.add_argument(
        "--manifest",
        default="configs/parity/runtime_manifest.mock.json",
        help="sim client가 검증할 runtime manifest",
    )
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--livestream", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--public-ip", default="")
    parser.add_argument("--camera-viewports", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
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
            _pixi_command("lock", "--check"),
            _pixi_command("run", "-e", "ros-tools", "core-test"),
            _pixi_command("run", "-e", "real", "dataset-test"),
            _pixi_command(
                "run",
                "-e",
                "ros-tools",
                "python",
                "scripts/parity/validate_checkpoint.py",
                "--manifest",
                "configs/parity/runtime_manifest.mock.json",
                "--checkpoint",
                "configs/parity/replay_checkpoint.json",
            ),
        ]
    elif args.mode == "mock-probe":
        commands = [
            _pixi_command(
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
            )
        ]
    elif args.mode == "sim":
        sim_arguments = [
            "run",
            "-e",
            "sim",
            "python",
            "scripts/parity/run_sim_client.py",
            "--steps",
            str(args.steps),
            "--device",
            args.sim_device,
            "--manifest",
            args.manifest,
            "--visualizer",
            "kit" if args.livestream else "none",
            "--livestream",
            str(args.livestream),
        ]
        if args.continuous:
            sim_arguments.append("--continuous")
        if args.public_ip:
            sim_arguments += ["--public-ip", args.public_ip]
        if args.camera_viewports:
            sim_arguments.append("--camera-viewports")
        commands = [_pixi_command(*sim_arguments)]
    elif args.mode == "real-dry-run":
        commands = [
            _pixi_command(
                "run",
                "-e",
                "real",
                "python",
                "scripts/parity/run_real_client.py",
                "--port",
                args.port,
            )
        ]
    elif args.mode == "real-readback":
        commands = [
            _pixi_command(
                "run",
                "-e",
                "real",
                "python",
                "scripts/parity/run_real_client.py",
                "--port",
                args.port,
                "--inspect-readback",
            )
        ]
    else:
        if not args.enable_motion or not args.confirm_emergency_cutoff_ready:
            raise RuntimeError(
                "real-motion은 --enable-motion과 "
                "--confirm-emergency-cutoff-ready를 모두 요구한다"
            )
        commands = [
            _pixi_command(
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
            )
        ]

    for command in commands:
        result = subprocess.run(command, env=environment, check=False)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
