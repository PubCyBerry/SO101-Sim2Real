#!/usr/bin/env python
"""Contract/calibration/lock/checkpoint로 runtime manifest를 생성한다."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

from so101_parity.calibration import CalibrationBundle
from so101_parity.contract import PolicyIOContract
from so101_parity.manifest import RuntimeManifest, file_sha256, tree_sha256


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=Path("configs/parity/policy_io.json"))
    parser.add_argument("--calibration", type=Path, default=Path("calibration/so101_canonical.json"))
    parser.add_argument("--runtime-config", type=Path, default=Path("configs/parity/runtime.json"))
    parser.add_argument("--pixi-lock", type=Path, default=Path("pixi.lock"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backend", choices=("act", "smolvla", "groot_zmq", "replay"), required=True)
    parser.add_argument(
        "--model-frame",
        choices=("canonical", "sim_legacy_rad_scale_v1", "real_lerobot_range_v1"),
        required=True,
    )
    parser.add_argument("--task", default="pick up the cube and place it in the bowl")
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--rename-map",
        default="{}",
        help="ACT/SmolVLA observation rename map JSON",
    )
    parser.add_argument(
        "--checkpoint-ref",
        help="manifest 기준 checkpoint 위치. 생략하면 --checkpoint 문자열을 사용한다.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        rename_map = json.loads(args.rename_map)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--rename-map JSON 오류: {exc}") from exc
    if not isinstance(rename_map, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in rename_map.items()
    ):
        raise ValueError("--rename-map은 string→string JSON object여야 한다")

    contract = PolicyIOContract.load(args.contract)
    calibration = CalibrationBundle.load(args.calibration)
    checkpoint_hash = (
        file_sha256(args.checkpoint)
        if args.checkpoint.is_file()
        else tree_sha256(args.checkpoint, exclude_names=(".DS_Store",))
    )
    raw = {
        "schema": "so101-runtime-manifest-v1",
        "backend": args.backend,
        "checkpoint_ref": args.checkpoint_ref or str(args.checkpoint),
        "model_frame": args.model_frame,
        "task": args.task,
        "chunk_size": args.chunk_size,
        "lease_duration_ms": 30000,
        "contract_hash": contract.contract_hash,
        "calibration_hash": calibration.calibration_hash,
        "motor_profile_hash": calibration.motor_profile_hash,
        "checkpoint_hash": checkpoint_hash,
        "pixi_lock_hash": file_sha256(args.pixi_lock),
        "runtime_config_hash": file_sha256(args.runtime_config),
        "stack": {
            "python": platform.python_version(),
            "isaacsim": _version("isaacsim"),
            "isaaclab": _version("isaaclab"),
            "isaaclab_commit": "28a37cecdd433c22d9eabd6a5954add9f13a8951",
            "ros": "jazzy",
            "rmw": "rmw_zenoh_cpp",
            "pixi": "0.70.2",
            "torch": _version("torch"),
            "lerobot": _version("lerobot"),
            "physics": "physx",
        },
    }
    if args.backend in ("act", "smolvla"):
        raw["device"] = args.device
        raw["rename_map"] = rename_map
    manifest = RuntimeManifest.with_hash(raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
