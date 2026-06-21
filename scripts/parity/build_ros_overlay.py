#!/usr/bin/env python
"""ROS package를 runtime drive에 mirror한 뒤 isolated overlay를 빌드한다."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    runtime_root_text = os.environ.get("SO101_RUNTIME_ROOT")
    if not runtime_root_text:
        raise RuntimeError("SO101_RUNTIME_ROOT가 Pixi activation에 설정되지 않았다")
    runtime_root = Path(runtime_root_text).resolve()
    source_root = (runtime_root / "ros2_ws" / "src").resolve()
    if runtime_root not in source_root.parents:
        raise RuntimeError(f"ROS mirror가 runtime root 밖이다: {source_root}")

    package_names = ("so101_vla_interfaces", "so101_vla_runtime")
    source_root.mkdir(parents=True, exist_ok=True)
    mirrored = []
    for name in package_names:
        source = (project_root / "ros2_ws" / "src" / name).resolve()
        destination = (source_root / name).resolve()
        if source != destination:
            shutil.copytree(source, destination, dirs_exist_ok=True)
        mirrored.append(destination)

    overlay = project_root / ".pixi" / "ros2"
    command = [
        "colcon",
        "--log-base",
        str(overlay / "log"),
        "build",
        "--merge-install",
        "--build-base",
        str(overlay / "build"),
        "--install-base",
        str(overlay / "install"),
        "--base-paths",
        *(str(path) for path in mirrored),
        "--cmake-args",
        "-DBUILD_TESTING=OFF",
        "-DPython3_FIND_VIRTUALENV=ONLY",
    ]
    return subprocess.run(command, cwd=runtime_root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
