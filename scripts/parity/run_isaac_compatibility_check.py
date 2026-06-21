#!/usr/bin/env python
"""Isaac Sim 6 공식 compatibility Checker를 headless로 실행해 JSON을 남긴다."""

from __future__ import annotations

import argparse
from enum import Enum
import faulthandler
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import tomllib
import traceback
import types

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument(
    "--report",
    type=Path,
    default=Path("outputs/parity/isaac_compatibility_check.json"),
)
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()

args.report.parent.mkdir(parents=True, exist_ok=True)
_stack_stream = args.report.with_suffix(".stack.txt").open("w", encoding="utf-8")
faulthandler.enable(file=_stack_stream)
faulthandler.dump_traceback_later(30, repeat=True, file=_stack_stream)
launcher = AppLauncher(
    {
        "visualizer": "none",
        "device": args.device,
        "enable_cameras": True,
        "livestream": 0,
    }
)
simulation_app = launcher.app


def _serialize(value):
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if hasattr(value, "__dict__"):
        return {key: _serialize(item) for key, item in value.__dict__.items()}
    return value


def _write(payload: dict) -> None:
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.__stdout__, flush=True)


def main() -> int:
    try:
        distribution_root = Path(
            importlib.metadata.distribution("isaacsim").locate_file("")
        )
        candidates = sorted(
            (distribution_root / "isaacsim" / "exts").glob(
                "isaacsim.app.compatibility_check-*"
            )
        )
        if len(candidates) != 1:
            raise RuntimeError(
                f"compatibility extension root를 하나로 결정하지 못했다: {candidates}"
            )
        extension_root = candidates[0]
        import isaacsim

        package_root = extension_root / "isaacsim" / "app" / "compatibility_check"
        package_specs = (
            ("isaacsim.app", extension_root / "isaacsim" / "app"),
            ("isaacsim.app.compatibility_check", package_root),
            ("isaacsim.app.compatibility_check.impl", package_root / "impl"),
        )
        for package_name, package_path in package_specs:
            if package_name not in sys.modules:
                package = types.ModuleType(package_name)
                package.__path__ = [str(package_path)]
                sys.modules[package_name] = package
        setattr(isaacsim, "app", sys.modules["isaacsim.app"])

        native_stub_name = "isaacsim.app.compatibility_check._compatibility_check"
        native_stub = types.ModuleType(native_stub_name)
        sys.modules[native_stub_name] = native_stub
        setattr(
            sys.modules["isaacsim.app.compatibility_check"],
            "_compatibility_check",
            native_stub,
        )
        checker_path = package_root / "impl" / "compatibility_checker.py"
        module_name = (
            "isaacsim.app.compatibility_check.impl.compatibility_checker"
        )
        module_spec = importlib.util.spec_from_file_location(module_name, checker_path)
        if module_spec is None or module_spec.loader is None:
            raise RuntimeError(f"compatibility Checker module을 읽지 못했다: {checker_path}")
        checker_module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = checker_module
        module_spec.loader.exec_module(checker_module)
        Checker = checker_module.Checker

        with (extension_root / "config" / "extension.toml").open("rb") as stream:
            extension_config = tomllib.load(stream)
        specs = extension_config["settings"]["exts"][
            "isaacsim.app.compatibility_check"
        ]
        checker = Checker()
        print("compatibility: nvidia-smi", file=sys.__stdout__, flush=True)
        checker.check_nvidia_smi({})
        print("compatibility: driver", file=sys.__stdout__, flush=True)
        checker.check_driver_version(specs["gpu_driver"])
        print("compatibility: RTX GPU", file=sys.__stdout__, flush=True)
        checker.check_rtx_gpu({})
        print("compatibility: VRAM", file=sys.__stdout__, flush=True)
        checker.check_vram(specs["gpu_vram"])
        print("compatibility: CPU", file=sys.__stdout__, flush=True)
        checker.check_cpu(specs["cpu_cores"])
        checker.check_cpu_cores(specs["cpu_cores"])
        checker.check_cpu_power_governor({})
        print("compatibility: RAM", file=sys.__stdout__, flush=True)
        checker.check_ram(specs["ram"])
        print("compatibility: storage", file=sys.__stdout__, flush=True)
        original_exists = checker_module.os.path.exists
        if sys.platform == "win32":
            runtime_root = Path(
                os.environ.get("SO101_RUNTIME_ROOT", str(Path.cwd()))
            ).resolve()
            runtime_drive = runtime_root.drive.upper()

            def _runtime_drive_only(path: str) -> bool:
                if len(path) == 2 and path[1] == ":":
                    return path.upper() == runtime_drive
                return original_exists(path)

            checker_module.os.path.exists = _runtime_drive_only
        try:
            checker.check_storage(specs["storage"])
        finally:
            checker_module.os.path.exists = original_exists
        print("compatibility: OS/display", file=sys.__stdout__, flush=True)
        checker.check_operating_system(specs["operating_system"])
        checker.check_display()

        checks = {
            "nvidia_smi": _serialize(checker.nvidia_smi),
            "gpu_driver_version": _serialize(checker.gpu_driver_version),
            "gpu_rtx": _serialize(checker.gpu_rtx),
            "gpu_vram": _serialize(checker.gpu_vram),
            "cpu": _serialize(checker.cpu),
            "cpu_cores": _serialize(checker.cpu_cores),
            "cpu_power_governor": _serialize(checker.cpu_power_governor),
            "ram": _serialize(checker.ram),
            "storage": _serialize(checker.disk_storage),
            "operating_system": _serialize(checker.operating_system),
            "display": _serialize(checker.display),
        }
        required = (
            checker.compatibility_check_status
            and checker.nvidia_smi.status
            and all(result.status for result in checker.gpu_rtx)
            and all(result.status for result in checker.gpu_vram)
        )
        payload = {
            "status": "passed" if required else "failed",
            "checker": "isaacsim.app.compatibility_check.Checker",
            "extension_config": str(extension_root / "config" / "extension.toml"),
            "headless_display_is_informational": True,
            "checks": checks,
        }
        _write(payload)
        return 0 if required else 1
    except Exception as exc:
        _write(
            {
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return 1
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
