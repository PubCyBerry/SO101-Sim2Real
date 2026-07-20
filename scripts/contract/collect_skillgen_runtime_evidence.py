#!/usr/bin/env python3
"""SkillGen 격리 runtime·worktree·reference 상태를 재현 가능한 JSON으로 수집한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


CUROBO_COMMIT = "ebb71702f3f70e767f40fd8e050674af0288abe8"


def _run(command: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(command, cwd=cwd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n{result.stderr.strip()}"
        )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-file", type=Path, required=True)
    parser.add_argument("--image", default="so101-skillgen:isaaclab-2.3.2-curobo-ebb7170")
    parser.add_argument(
        "--main-worktree",
        type=Path,
        default=Path("/home/konan147/Workspaces/SO101-Sim2Real"),
    )
    parser.add_argument(
        "--isaaclab-reference",
        type=Path,
        default=Path("/home/konan147/Workspaces/SO101-Sim2Real/ref_repos/IsaacLab"),
    )
    args = parser.parse_args()

    worktree = Path.cwd().resolve()
    image_info = json.loads(_run(["docker", "image", "inspect", args.image]))[0]
    labels = image_info["Config"].get("Labels", {})
    repo_digests = image_info.get("RepoDigests", [])
    import_smoke = _run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "/isaac-sim/python.sh",
            args.image,
            "-c",
            "import curobo,isaaclab_mimic; print('curobo+isaaclab_mimic imports OK')",
        ]
    )
    curobo_commit = _run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "git",
            args.image,
            "-C",
            "/opt/skillgen-curobo",
            "rev-parse",
            "HEAD",
        ]
    )
    nvcc = _run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "/usr/local/cuda-12.8/bin/nvcc",
            args.image,
            "--version",
        ]
    )
    ptx = _run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "bash",
            args.image,
            "-lc",
            "/usr/local/cuda-12.8/bin/cuobjdump --list-ptx "
            "/opt/skillgen-curobo/src/curobo/curobolib/line_search_cu*.so",
        ]
    )
    gpu_query = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()[0]
    gpu_model, driver_version, memory_mib = [part.strip() for part in gpu_query.split(",")]
    reference = args.isaaclab_reference.resolve()
    reference_status = _run(["git", "status", "--porcelain=v1"], cwd=reference)
    reference_tag = _run(["git", "describe", "--tags", "--exact-match", "HEAD"], cwd=reference)
    current_status = _run(["git", "status", "--porcelain=v1"], cwd=worktree)
    main_status = _run(["git", "status", "--porcelain=v1"], cwd=args.main_worktree.resolve())
    branch = _run(["git", "branch", "--show-current"], cwd=worktree)
    worktree_list = _run(["git", "worktree", "list", "--porcelain"], cwd=worktree)

    checks = {
        "codex_branch": branch.startswith("codex/"),
        "isolated_worktree": worktree != args.main_worktree.resolve(),
        "isaaclab_reference_clean": reference_status == "",
        "isaaclab_reference_v2_3_2": reference_tag in {"v2.3.2", "2.3.2"},
        "image_isaac_sim_5_1_0": labels.get("org.opencontainers.image.isaacsim.version") == "5.1.0",
        "image_isaaclab_2_3_2": labels.get("org.opencontainers.image.isaaclab.version") == "2.3.2",
        "exact_curobo_commit": curobo_commit == CUROBO_COMMIT,
        "compute_80_ptx_present": "sm_80.ptx" in ptx,
        "imports_ok": "imports OK" in import_smoke,
    }
    report = {
        "status": "complete" if all(checks.values()) else "failed",
        "checks": checks,
        "worktree": str(worktree),
        "branch": branch,
        "git_commit_at_collection": _run(["git", "rev-parse", "HEAD"], cwd=worktree),
        "worktree_status": current_status.splitlines(),
        "main_worktree": str(args.main_worktree.resolve()),
        "main_worktree_status_preserved": main_status.splitlines(),
        "git_worktree_list": worktree_list,
        "isaaclab_reference": str(reference),
        "isaaclab_reference_tag": reference_tag,
        "isaaclab_reference_status": reference_status.splitlines(),
        "image": args.image,
        "container_image_id": image_info["Id"],
        "container_image_digest": repo_digests[0] if repo_digests else image_info["Id"],
        "image_labels": labels,
        "isaac_sim_version": labels.get("org.opencontainers.image.isaacsim.version"),
        "isaaclab_version": labels.get("org.opencontainers.image.isaaclab.version"),
        "curobo_commit": curobo_commit,
        "cuda_toolkit": next((line.strip() for line in nvcc.splitlines() if "release" in line), nvcc),
        "ptx_entries": ptx.splitlines(),
        "gpu_model": gpu_model,
        "gpu_memory_mib": int(memory_mib),
        "driver_version": driver_version,
        "import_smoke": import_smoke,
    }
    report_path = args.report_file.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"worktree_status", "main_worktree_status_preserved", "git_worktree_list", "ptx_entries"}}, indent=2))
    if report["status"] != "complete":
        raise RuntimeError("runtime evidence gate failed")


if __name__ == "__main__":
    main()
