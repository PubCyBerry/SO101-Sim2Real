#!/usr/bin/env python3
"""Codex용 Sim2Real 오케스트레이터 골격."""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
GATE = REPO_ROOT / "scripts" / "orchestrator" / "gate.py"
DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "sonnet[1m]")
DEFAULT_EFFORT = os.environ.get("CLAUDE_EFFORT", "high")
DEFAULT_PERMISSION_MODE = os.environ.get("CLAUDE_PERMISSION_MODE", "bypassPermissions")
DEFAULT_TOOLS = os.environ.get(
    "CLAUDE_TOOLS",
    "Skill, Read, Glob, Grep, Write, Edit, Bash, Agent, Monitor, TaskCreate, TaskGet, "
    "TaskList, TaskUpdate, TaskStop, WebFetch, WebSearch, Workflow",
)

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def run(command: list[str], *, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def find_claude() -> str:
    """로컬 Claude Code 실행 파일을 찾는다. Windows에서는 WSL을 거치지 않는다."""
    candidates: list[str | Path] = []
    if os.environ.get("CLAUDE_BIN"):
        candidates.append(os.environ["CLAUDE_BIN"])

    found = shutil.which("claude") or shutil.which("claude.exe")
    if found:
        candidates.append(found)

    candidates.extend(
        [
            Path.home() / ".local" / "bin" / "claude.exe",
            Path("C:/Users/taehunkim/.local/bin/claude.exe"),
        ]
    )

    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return str(path)

    raise FileNotFoundError("claude executable not found")


def dispatch_local(prompt_text: str) -> subprocess.CompletedProcess[str]:
    """로컬 worker를 Python subprocess로 직접 호출한다."""
    command = [
        find_claude(),
        "-p",
        prompt_text,
        "--output-format",
        "json",
        "--permission-mode",
        DEFAULT_PERMISSION_MODE,
        "--model",
        DEFAULT_MODEL,
        "--effort",
        DEFAULT_EFFORT,
        "--no-session-persistence",
        "--tools",
        DEFAULT_TOOLS,
        "--allowedTools",
        DEFAULT_TOOLS,
    ]
    return run(command, timeout=300)


def extract_worker_result(raw_stdout: str) -> dict[str, Any]:
    """Claude Code JSON의 result 필드에서 worker JSON을 꺼낸다."""
    outer = json.loads(raw_stdout)
    body = outer.get("result", "")
    if isinstance(body, dict):
        return body

    text = str(body).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    elif not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]

    parsed = json.loads(text)
    required = {"task_id", "status", "changed_files", "verification", "notes"}
    missing = required - set(parsed)
    if missing:
        raise ValueError(f"worker JSON missing keys: {sorted(missing)}")
    return parsed


def t01_probe_spec() -> str:
    return """\
Task: T0.1 dry-run verification only.

Rules:
- Do not edit files.
- Run:
  1. python scripts/validate_lerobot_schema.py datasets/pick_pen
  2. python scripts/validate_lerobot_schema.py --self-test
- Return only a JSON object with keys:
  task_id, status, changed_files, verification, notes
- task_id must be "T0.1".
- status must be "done" only if both commands pass.
"""


def dry_run_t01() -> dict[str, Any]:
    """T0.1을 worker DISPATCH 후 Codex VERIFY로 다시 확인한다."""
    dispatch = dispatch_local(t01_probe_spec())

    dispatch_result: dict[str, Any] = {
        "command": [
            "claude",
            "-p",
            "<T0.1 dry-run spec>",
            "--output-format",
            "json",
            "--permission-mode",
            DEFAULT_PERMISSION_MODE,
            "--model",
            DEFAULT_MODEL,
            "--effort",
            DEFAULT_EFFORT,
            "--tools",
            DEFAULT_TOOLS,
            "--allowedTools",
            DEFAULT_TOOLS,
        ],
        "exit_code": dispatch.returncode,
        "stdout": dispatch.stdout.strip(),
        "stderr": dispatch.stderr.strip(),
        "ok": dispatch.returncode == 0,
    }

    worker: dict[str, Any] | None = None
    parse_error: str | None = None
    if dispatch.returncode == 0:
        try:
            worker = extract_worker_result(dispatch.stdout)
        except Exception as exc:  # worker 출력 형식 오류도 게이트 실패다.
            parse_error = str(exc)

    verify = run([sys.executable, str(GATE), "validate-lerobot-schema"], timeout=120)
    verify_result = {
        "command": [sys.executable, str(GATE), "validate-lerobot-schema"],
        "exit_code": verify.returncode,
        "stdout": verify.stdout.strip(),
        "stderr": verify.stderr.strip(),
        "ok": verify.returncode == 0,
    }

    ok = dispatch_result["ok"] and worker is not None and verify_result["ok"]
    return {
        "task_id": "T0.4.dry_run_t0.1",
        "status": "done" if ok else "failed",
        "select": {"task_id": "T0.1", "reason": "T0.4 e2e dry run"},
        "dispatch": dispatch_result,
        "worker_result": worker,
        "worker_parse_error": parse_error,
        "verify": verify_result,
        "record": {
            "mode": "codex",
            "note": "dry run 결과는 호출자(Codex)가 TASKS.md/CONTEXT.md에 기록한다.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Sim2Real orchestrator loop skeleton")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("dry-run-t0.1", help="Claude dispatch + validator gate e2e dry run")
    args = parser.parse_args()

    if args.command == "dry-run-t0.1":
        result = dry_run_t01()
    else:  # pragma: no cover - argparse가 막는다.
        parser.error(f"unknown command: {args.command}")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
