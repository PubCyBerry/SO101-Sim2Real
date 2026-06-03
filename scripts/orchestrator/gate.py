#!/usr/bin/env python3
"""Sim2Real 오케스트레이터 검증 게이트."""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = "datasets/pick_pen"
DEFAULT_GPU_LOCK = "/DISK1/so101-sim2real/run/gpu.lock"

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def run_command(command: list[str], *, cwd: Path = REPO_ROOT, timeout: int = 120) -> dict[str, Any]:
    """명령을 실행하고 JSON으로 남기기 좋은 결과만 반환한다."""
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "ok": completed.returncode == 0,
    }


def validate_lerobot_schema(dataset: str) -> dict[str, Any]:
    """T0.1 validator 게이트를 재실행한다."""
    checks = [
        run_command([sys.executable, "scripts/validate_lerobot_schema.py", dataset]),
        run_command([sys.executable, "scripts/validate_lerobot_schema.py", "--self-test"]),
    ]
    return {
        "gate": "validate_lerobot_schema",
        "dataset": dataset,
        "ok": all(check["ok"] for check in checks),
        "checks": checks,
    }


def gpu_lock_status(lock_path: str) -> dict[str, Any]:
    """GPU 중량 작업 직렬화 lock 경로 상태를 확인한다."""
    path = Path(lock_path)
    return {
        "gate": "gpu_lock_status",
        "lock_path": lock_path,
        "exists": path.exists(),
        "ok": True,
        "note": "로컬 경로 기준 상태다. 서버 GPU 작업은 ssh 경로에서 같은 파일을 사용한다.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Sim2Real gate runner")
    sub = parser.add_subparsers(dest="command", required=True)

    validate_parser = sub.add_parser("validate-lerobot-schema")
    validate_parser.add_argument("--dataset", default=DEFAULT_DATASET)

    lock_parser = sub.add_parser("gpu-lock-status")
    lock_parser.add_argument("--lock-path", default=DEFAULT_GPU_LOCK)

    args = parser.parse_args()

    if args.command == "validate-lerobot-schema":
        result = validate_lerobot_schema(args.dataset)
    elif args.command == "gpu-lock-status":
        result = gpu_lock_status(args.lock_path)
    else:  # pragma: no cover - argparse가 막는다.
        parser.error(f"unknown command: {args.command}")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
