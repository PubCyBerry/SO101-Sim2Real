#!/usr/bin/env python3
"""SkillGen Gate A-J를 artifact로 판정하고 checksum·manifest·최종 보고서를 만든다."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def _load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _all_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(candidate for candidate in path.rglob("*") if candidate.is_file())
        else:
            raise FileNotFoundError(path)
    return sorted(set(candidate.resolve() for candidate in files))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-file", type=Path, default=Path("datasets/skillgen/source/source.hdf5"))
    parser.add_argument(
        "--annotated-file", type=Path, default=Path("datasets/skillgen/annotated/annotated.hdf5")
    )
    parser.add_argument("--smoke-file", type=Path, default=Path("datasets/skillgen/generated/smoke.hdf5"))
    parser.add_argument("--source-report", type=Path, default=Path("outputs/skillgen/reports/source_generation.json"))
    parser.add_argument("--source-replay-report", type=Path, default=Path("outputs/skillgen/reports/source_replay.json"))
    parser.add_argument("--smoke-report", type=Path, default=Path("outputs/skillgen/reports/smoke_generation.json"))
    parser.add_argument("--env-report", type=Path, default=Path("outputs/skillgen/reports/env_contract.json"))
    parser.add_argument(
        "--boundary-evidence-dir", type=Path, default=Path("outputs/skillgen/evidence/source_boundaries")
    )
    parser.add_argument("--failure-video", type=Path, required=True)
    parser.add_argument("--skip-checksums", action="store_true", help="finalizer 자체 smoke test용")
    args = parser.parse_args()

    repo = Path.cwd().resolve()
    data_root = args.data_root.resolve()
    report_dir = data_root / "reports"
    final_dir = data_root / "final"
    full_file = data_root / "full" / "full.hdf5"
    planner_visual_file = data_root / "evidence" / "planner_visual.hdf5"
    lerobot_root = data_root / "lerobot"
    report_paths = {
        "runtime": report_dir / "runtime_evidence.json",
        "planner_visual": report_dir / "planner_visual_generation.json",
        "source_annotation": report_dir / "source_annotation_validation.json",
        "smoke_hdf5": report_dir / "smoke_hdf5_validation.json",
        "generation": data_root / "full" / "generation_summary.json",
        "merge": report_dir / "full_merge.json",
        "full_hdf5": report_dir / "full_hdf5_validation.json",
        "conversion": report_dir / "conversion.json",
        "schema": report_dir / "schema_validation.json",
        "action": report_dir / "action_quality.json",
        "video": report_dir / "video_validation.json",
        "replay": report_dir / "replay_audit.json",
        "failure_generation": report_dir / "failure_physics_generation.json",
        "failure_video": report_dir / "failure_video.json",
    }
    reports = {name: _load(path) for name, path in report_paths.items()}
    source_report = _load(args.source_report.resolve())
    source_replay = _load(args.source_replay_report.resolve())
    smoke_report = _load(args.smoke_report.resolve())
    env_report = _load(args.env_report.resolve())
    info = _load(lerobot_root / "meta" / "info.json")
    boundary_images = sorted(args.boundary_evidence_dir.resolve().glob("*.png"))
    failure_video = args.failure_video.resolve()

    generation = reports["generation"]
    conversion = reports["conversion"]
    action = reports["action"]
    video = reports["video"]
    replay = reports["replay"]
    failure_generation = reports["failure_generation"]
    def replay_video_host_path(value: str) -> Path:
        container_prefix = "/workspace/skillgen-data/"
        if value.startswith(container_prefix):
            return data_root / value.removeprefix(container_prefix)
        return Path(value).resolve()

    successful_video_paths = [replay_video_host_path(value) for value in replay["successful_videos"]]
    required_failure_categories = {"planning", "ik_tracking", "grasp", "place", "timeout"}
    source_replay_successes = source_replay.get(
        "successes", round(source_replay["success_rate"] * source_replay.get("episodes", 10))
    )
    gates = {
        "A": (
            reports["runtime"]["status"] == "complete"
            and all(reports["runtime"]["checks"].values())
            and reports["runtime"]["git_commit_at_collection"] == _git("rev-parse", "HEAD")
            and reports["runtime"]["branch"].startswith("codex/")
            and Path(reports["runtime"]["worktree"]).resolve() == repo
        ),
        "B": (
            env_report["status"] == "complete"
            and all(env_report["checks"].values())
            and reports["planner_visual"]["status"] == "complete"
            and reports["planner_visual"]["record_images"]
            and reports["planner_visual"]["visualize_spheres"]
            and not reports["planner_visual"]["visualize_plan"]
            and reports["planner_visual"]["successes"] >= 1
            and reports["planner_visual"]["planner"]["planning_successes"] >= 2
            and reports["planner_visual"]["planner"]["plans_with_attached_cube"] >= 1
            and planner_visual_file.is_file()
        ),
        "C": (
            source_report["status"] == "complete"
            and source_report["successes"] >= 10
            and source_replay["status"] == "complete"
            and source_replay_successes >= 9
            and source_replay["success_rate"] >= 0.9
        ),
        "D": (
            reports["source_annotation"]["status"] == "complete"
            and reports["source_annotation"]["episode_count"] >= 10
            and len(boundary_images) >= 36
        ),
        "E": (
            smoke_report["status"] == "complete"
            and smoke_report["successes"] >= 10
            and smoke_report["success_rate"] >= 0.4
            and smoke_report["use_skillgen"]
            and smoke_report["num_envs"] == 1
            and set(smoke_report["failure_categories"]) == required_failure_categories
            and sum(smoke_report["failure_categories"].values()) == smoke_report["failures"]
            and smoke_report["planner"]["plans_with_attached_cube"] > 0
            and reports["smoke_hdf5"]["status"] == "complete"
            and reports["smoke_hdf5"]["episode_count"] == 10
            and reports["smoke_hdf5"]["unique_layout_count"] == 10
            and reports["smoke_hdf5"]["success_only"]
        ),
        "F": (
            generation["status"] == "complete"
            and generation["successes"] == 1000
            and generation["success_rate"] >= 0.4
            and generation["use_skillgen"]
            and generation["num_envs"] == 1
            and set(generation["failure_categories"]) == required_failure_categories
            and sum(generation["failure_categories"].values()) == generation["failures"]
            and generation["attempts"] == generation["successes"] + generation["failures"]
            and reports["merge"]["status"] == "complete"
            and reports["merge"]["episode_count"] == 1000
            and reports["merge"]["unique_layout_count"] == 1000
            and reports["full_hdf5"]["status"] == "complete"
            and reports["full_hdf5"]["episode_count"] == 1000
            and reports["full_hdf5"]["unique_layout_count"] == 1000
            and reports["full_hdf5"]["success_only"]
            and reports["full_hdf5"]["require_images"]
            and full_file.is_file()
        ),
        "G": (
            conversion["status"] == "complete"
            and conversion["input_episode_count"] == 1000
            and conversion["converted_episode_count"] == 1000
            and conversion["input_frame_count"] == conversion["output_frame_count"]
            and conversion["fps"] == 30
            and conversion["cameras"] == ["top", "wrist", "front"]
            and conversion["camera_shape"] == [480, 640, 3]
            and conversion["action_source"] == "joint_data/target"
            and conversion["state_source"] == "joint_data/state"
            and reports["schema"]["status"] == "complete"
            and reports["schema"]["require_all_cameras"]
            and reports["schema"]["error_count"] == 0
            and info["total_episodes"] == 1000
            and info["total_frames"] == conversion["output_frame_count"]
            and info["fps"] == 30
            and video["status"] == "complete"
            and video["expected_fps"] == 30
            and video["required_cameras"] == ["top", "wrist", "front"]
            and video["full_decode"]
            and not video["errors"]
        ),
        "H": (
            action["status"] == "complete"
            and action["action_limits"]["violation_count"] == 0
            and action["velocity_limits"]["violation_count"] == 0
        ),
        "I": (
            replay["status"] == "complete"
            and replay["sample_count"] >= 30
            and replay["success_rate"] >= 0.9
            and len(replay["successful_videos"]) >= 3
            and all(path.is_file() for path in successful_video_paths)
        ),
        "J": (
            args.source_file.resolve().is_file()
            and args.annotated_file.resolve().is_file()
            and args.smoke_file.resolve().is_file()
            and full_file.is_file()
            and lerobot_root.is_dir()
            and failure_video.is_file()
            and failure_generation["status"] == "complete"
            and failure_generation["record_images"]
            and failure_generation["keep_failed"]
            and failure_generation["failures"] >= 1
            and sum(failure_generation["failure_categories"].values())
            == failure_generation["failures"]
            and reports["failure_video"]["status"] == "complete"
            and reports["failure_video"]["full_decode"]
            and not reports["failure_video"]["decode_error"]
            and Path(reports["failure_video"]["input_file"]).is_file()
            and Path(reports["failure_video"]["output_video"]).resolve() == failure_video
        ),
    }
    if not all(gates.values()):
        failed = [gate for gate, passed in gates.items() if not passed]
        raise RuntimeError(f"cannot finalize; failed gates: {failed}")

    final_dir.mkdir(parents=True, exist_ok=True)
    checksum_path = final_dir / "SHA256SUMS"
    checksum_targets = _all_files(
        [
            args.source_file.resolve(),
            args.annotated_file.resolve(),
            args.smoke_file.resolve(),
            full_file,
            lerobot_root,
            report_dir,
            planner_visual_file,
            data_root / "full" / "reports",
            data_root / "full" / "generation_summary.json",
            args.source_report.resolve(),
            args.source_replay_report.resolve(),
            args.smoke_report.resolve(),
            args.env_report.resolve(),
            args.boundary_evidence_dir.resolve(),
            successful_video_paths[0].parent,
            failure_video.parent,
            failure_video,
        ]
    )
    checksums: dict[str, str] = {}
    if not args.skip_checksums:
        for index, path in enumerate(checksum_targets):
            print(f"[sha256] {index + 1}/{len(checksum_targets)} {path}", flush=True)
            checksums[str(path)] = _sha256(path)
        checksum_path.write_text(
            "".join(f"{digest}  {path}\n" for path, digest in checksums.items()), encoding="utf-8"
        )

    shard_reports = sorted(
        path
        for path in (data_root / "full" / "reports").glob("shard_*.json")
        if path.stem.removeprefix("shard_").isdigit()
    )
    shard_seeds = [_load(path)["seed"] for path in shard_reports]
    commands = [
        "docker compose --env-file .env -f docker/docker-compose.yaml build skillgen",
        "docker compose ... run --rm skillgen python /workspace/scripts/datagen/record_skillgen_source.py --headless --enable_cameras --num-demos 10 --seed 55000",
        "docker compose ... run --rm skillgen python /workspace/scripts/datagen/generate_skillgen.py --headless --generation-num-trials 10 --seed 56000",
        ".venv/bin/python scripts/datagen/run_skillgen_full.py --data-root /DISK1/so101-sim2real/skillgen --total-successes 1000 --shard-size 25 --base-seed 58000 --resume",
        ".venv/bin/python scripts/datagen/merge_skillgen_shards.py --expected-episodes 1000 --resume",
        ".venv/bin/python scripts/convert/skillgen_hdf5_to_lerobot_v3.py --expected-episodes 1000 --video-codec h264_nvenc",
        ".venv/bin/python scripts/contract/validate_lerobot_schema.py <lerobot> --require-all-cameras",
        ".venv/bin/python scripts/contract/validate_skillgen_action_quality.py --lerobot-root <lerobot> --expected-episodes 1000",
        ".venv/bin/python scripts/contract/validate_lerobot_videos.py <lerobot> --full-decode",
        "docker compose ... run --rm skillgen python /workspace/scripts/contract/replay_skillgen_lerobot_audit.py --enable_cameras --sample-count 30 --selection-seed 93017",
    ]
    manifest = {
        "status": "DONE",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "worktree": str(repo),
        "container_image_digest": reports["runtime"]["container_image_digest"],
        "isaac_sim_version": reports["runtime"]["isaac_sim_version"],
        "isaaclab_version": reports["runtime"]["isaaclab_version"],
        "curobo_commit": reports["runtime"]["curobo_commit"],
        "cuda_version": reports["runtime"]["cuda_toolkit"],
        "gpu_model": reports["runtime"]["gpu_model"],
        "task_id": generation["task_id"],
        "source_dataset_path": str(args.source_file.resolve()),
        "annotated_dataset_path": str(args.annotated_file.resolve()),
        "smoke_dataset_path": str(args.smoke_file.resolve()),
        "generated_dataset_path": str(full_file),
        "lerobot_dataset_path": str(lerobot_root),
        "source_episode_count": source_report["successes"],
        "generation_attempt_count": generation["attempts"],
        "generation_success_count": generation["successes"],
        "generation_success_rate": generation["success_rate"],
        "generation_failure_categories": generation["failure_categories"],
        "final_episode_count": info["total_episodes"],
        "final_frame_count": info["total_frames"],
        "fps": info["fps"],
        "camera_keys": ["top", "wrist", "front"],
        "random_seeds": {
            "source_start": source_report["seed_start"],
            "smoke": smoke_report["seed"],
            "full_shards": shard_seeds,
            "replay_selection": replay["selection_seed"],
        },
        "commands": commands,
        "reports": {name: str(path) for name, path in report_paths.items()},
        "success_replay_videos": [str(path) for path in successful_video_paths[:3]],
        "representative_failure_video": str(failure_video),
        "checksums_file": str(checksum_path),
        "checksums": checksums,
        "gate_results": {f"Gate {gate}": "PASS" for gate in gates},
    }
    manifest_path = final_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    final_report = f"""# SkillGen Data Generation Result

- Status: DONE
- Git commit: `{manifest['git_commit']}`
- Worktree: `{repo}`
- Runtime: Isaac Sim {manifest['isaac_sim_version']} / Isaac Lab {manifest['isaaclab_version']} / cuRobo `{manifest['curobo_commit']}`
- Source episodes: {manifest['source_episode_count']}
- SkillGen attempts: {manifest['generation_attempt_count']}
- SkillGen successes: {manifest['generation_success_count']}
- Generation success rate: {manifest['generation_success_rate']:.1%}
- Final LeRobot episodes: {manifest['final_episode_count']}
- Final frames: {manifest['final_frame_count']}
- Schema validation: PASS
- Velocity violations: 0
- Video decode failures: 0
- Replay audit: {replay['successes']}/{replay['sample_count']} ({replay['success_rate']:.1%})
- Dataset path: `{lerobot_root}`
- Manifest path: `{manifest_path}`
- Checksum path: `{checksum_path}`

## Gate Results

""" + "\n".join(f"- Gate {gate}: PASS" for gate in gates) + f"""

## Evidence

- Runtime: `{report_paths['runtime']}`
- Source/annotation: `{report_paths['source_annotation']}`
- Full generation: `{report_paths['generation']}`
- Conversion/schema/action/video: `{report_paths['conversion']}`, `{report_paths['schema']}`, `{report_paths['action']}`, `{report_paths['video']}`
- Replay: `{report_paths['replay']}`
- Representative failure: `{failure_video}`

## Remaining Issues

None
"""
    (final_dir / "RESULT.md").write_text(final_report, encoding="utf-8")
    print(json.dumps({"status": "DONE", "manifest": str(manifest_path), "checksums": str(checksum_path), "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
