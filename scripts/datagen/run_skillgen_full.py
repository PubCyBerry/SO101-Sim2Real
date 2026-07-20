#!/usr/bin/env python3
"""Docker SkillGen full generation을 검증 가능한 shard 단위로 실행·resume한다."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import h5py


def _episode_count(path: Path) -> int:
    with h5py.File(path, "r") as dataset:
        return len(dataset["data"])


def _valid_completed_shard(dataset: Path, report: Path, expected: int) -> bool:
    if not dataset.is_file() or not report.is_file():
        return False
    try:
        summary = json.loads(report.read_text(encoding="utf-8"))
        return (
            summary["status"] == "complete"
            and int(summary["successes"]) == expected
            and _episode_count(dataset) == expected
        )
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


def _ensure_action_qa(*, repo: Path, dataset: Path, report: Path, expected: int) -> dict:
    """완료 shard를 canonical unit으로 독립 검사하고 0-violation만 반환한다."""
    if report.is_file():
        try:
            existing = json.loads(report.read_text(encoding="utf-8"))
            if (
                existing["status"] == "complete"
                and existing["episodes"] == expected
                and existing["action_limits"]["violation_count"] == 0
                and existing["velocity_limits"]["violation_count"] == 0
            ):
                return existing
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            pass
    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/contract/validate_skillgen_action_quality.py"),
            "--input-hdf5",
            str(dataset),
            "--report-file",
            str(report),
            "--expected-episodes",
            str(expected),
        ],
        cwd=repo,
        check=True,
    )
    result = json.loads(report.read_text(encoding="utf-8"))
    if result["status"] != "complete":
        raise RuntimeError(f"action QA failed: {dataset}")
    return result


def _preserve_incomplete(paths: list[Path], incomplete_dir: Path) -> None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    incomplete_dir.mkdir(parents=True, exist_ok=True)
    for path in existing:
        destination = incomplete_dir / f"{path.name}.{stamp}"
        shutil.move(str(path), str(destination))


def _run_and_tee(command: list[str], *, env: dict[str, str], log_path: Path, container_name: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_file.write(line)
                log_file.flush()
            return_code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait(timeout=20)
            subprocess.run(["docker", "stop", container_name], check=False)
            raise
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--total-successes", type=int, default=1000)
    parser.add_argument("--shard-size", type=int, default=25)
    parser.add_argument("--base-seed", type=int, default=58000)
    parser.add_argument("--start-shard", type=int, default=0, help="병렬 worker의 시작 shard index")
    parser.add_argument("--stop-shard", type=int, help="병렬 worker의 끝 shard index(exclusive)")
    parser.add_argument("--max-failures", type=int, default=2500)
    parser.add_argument("--task", default="SimToReal-SO101-PickCube-SkillGen-DR-v0")
    parser.add_argument("--compose-file", type=Path, default=Path("docker/docker-compose.yaml"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--record-images", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.total_successes < 1 or args.shard_size < 1:
        raise ValueError("total-successes and shard-size must be positive")

    repo = Path.cwd().resolve()
    data_root = args.data_root.resolve()
    shard_dir = data_root / "full" / "shards"
    report_dir = data_root / "full" / "reports"
    log_dir = data_root / "full" / "logs"
    incomplete_dir = data_root / "full" / "incomplete"
    for directory in (shard_dir, report_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    shard_targets: list[int] = []
    remaining = args.total_successes
    while remaining:
        target = min(args.shard_size, remaining)
        shard_targets.append(target)
        remaining -= target
    stop_shard = len(shard_targets) if args.stop_shard is None else args.stop_shard
    if not 0 <= args.start_shard < stop_shard <= len(shard_targets):
        raise ValueError(
            f"invalid shard range [{args.start_shard}, {stop_shard}) for {len(shard_targets)} shards"
        )
    work_items = list(enumerate(shard_targets))[args.start_shard : stop_shard]
    is_full_range = len(work_items) == len(shard_targets)
    progress_path = (
        data_root / "full" / "run_progress.json"
        if is_full_range
        else data_root
        / "full"
        / f"run_progress_{args.start_shard:04d}_{stop_shard - 1:04d}.json"
    )

    environment = os.environ.copy()
    environment["SKILLGEN_DATA_ROOT"] = str(data_root)
    completed_reports: list[dict] = []
    action_qa_reports: list[dict] = []

    for shard_index, target in work_items:
        stem = f"shard_{shard_index:04d}"
        dataset_path = shard_dir / f"{stem}.hdf5"
        report_path = report_dir / f"{stem}.json"
        action_qa_path = report_dir / f"{stem}_action_qa.json"
        log_path = log_dir / f"{stem}.log"
        seed = args.base_seed + shard_index * 100_000

        if _valid_completed_shard(dataset_path, report_path, target):
            if not args.resume:
                raise FileExistsError(f"completed shard exists; pass --resume: {dataset_path}")
            print(f"[resume] 검증된 {stem} 건너뜀 ({target} episodes)")
            completed_reports.append(json.loads(report_path.read_text(encoding="utf-8")))
            action_qa_reports.append(
                _ensure_action_qa(
                    repo=repo,
                    dataset=dataset_path,
                    report=action_qa_path,
                    expected=target,
                )
            )
            continue

        if any(path.exists() for path in (dataset_path, report_path, log_path)):
            if not args.resume:
                raise FileExistsError(f"incomplete shard exists; pass --resume: {stem}")
            _preserve_incomplete([dataset_path, report_path, log_path], incomplete_dir)

        container_name = f"so101-skillgen-{stem}"
        command = [
            "docker",
            "compose",
            "--env-file",
            str(args.env_file),
            "-f",
            str(args.compose_file),
            "run",
            "--rm",
            "--name",
            container_name,
            "skillgen",
            "python",
            "/workspace/scripts/datagen/generate_skillgen.py",
            "--headless",
            "--task",
            args.task,
            "--input-file",
            "/workspace/datasets/skillgen/annotated/annotated.hdf5",
            "--output-file",
            f"/workspace/skillgen-data/full/shards/{stem}.hdf5",
            "--summary-file",
            f"/workspace/skillgen-data/full/reports/{stem}.json",
            "--generation-num-trials",
            str(target),
            "--seed",
            str(seed),
            "--max-failures",
            str(args.max_failures),
        ]
        if args.record_images:
            command.extend(["--enable_cameras", "--record-images"])

        print(f"[run] {stem}: target={target}, seed={seed}")
        _run_and_tee(command, env=environment, log_path=log_path, container_name=container_name)
        if not _valid_completed_shard(dataset_path, report_path, target):
            raise RuntimeError(f"shard validation failed: {stem}")
        completed_reports.append(json.loads(report_path.read_text(encoding="utf-8")))
        action_qa_reports.append(
            _ensure_action_qa(
                repo=repo,
                dataset=dataset_path,
                report=action_qa_path,
                expected=target,
            )
        )

        manifest = {
            "status": "in_progress",
            "total_target_successes": sum(target for _, target in work_items),
            "shard_size": args.shard_size,
            "completed_shards": len(completed_reports),
            "completed_successes": sum(int(report["successes"]) for report in completed_reports),
            "attempts": sum(int(report["attempts"]) for report in completed_reports),
            "failures": sum(int(report["failures"]) for report in completed_reports),
            "base_seed": args.base_seed,
            "task_id": args.task,
            "record_images": args.record_images,
        }
        manifest["success_rate"] = manifest["completed_successes"] / manifest["attempts"]
        progress_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    attempts = sum(int(report["attempts"]) for report in completed_reports)
    successes = sum(int(report["successes"]) for report in completed_reports)
    failures = sum(int(report["failures"]) for report in completed_reports)
    failure_categories = {
        category: sum(int(report["failure_categories"].get(category, 0)) for report in completed_reports)
        for category in ("planning", "ik_tracking", "grasp", "place", "timeout")
    }
    final_summary = {
        "status": "complete",
        "task_id": args.task,
        "shards": len(completed_reports),
        "shard_index_start": args.start_shard,
        "shard_index_stop": stop_shard,
        "successes": successes,
        "attempts": attempts,
        "failures": failures,
        "success_rate": successes / attempts,
        "failure_categories": failure_categories,
        "base_seed": args.base_seed,
        "shard_size": args.shard_size,
        "record_images": args.record_images,
        "use_skillgen": True,
        "num_envs": 1,
        "action_quality_reports": [
            str(report_dir / f"shard_{index:04d}_action_qa.json") for index, _ in work_items
        ],
        "data_root": str(data_root),
    }
    summary_path = (
        data_root / "full" / "generation_summary.json"
        if is_full_range
        else data_root
        / "full"
        / f"worker_summary_{args.start_shard:04d}_{stop_shard - 1:04d}.json"
    )
    summary_path.write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
