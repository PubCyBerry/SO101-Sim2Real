#!/usr/bin/env python
"""Checkpoint hash와 server/runtime manifest 연결을 검증한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from so101_parity.manifest import RuntimeManifest, file_sha256, tree_sha256


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    manifest = RuntimeManifest.load(args.manifest)
    actual = (
        file_sha256(args.checkpoint)
        if args.checkpoint.is_file()
        else tree_sha256(args.checkpoint, exclude_names=(".DS_Store",))
    )
    expected = str(manifest.raw["checkpoint_hash"])
    report = {
        "ok": actual == expected,
        "checkpoint": str(args.checkpoint),
        "actual_checkpoint_hash": actual,
        "expected_checkpoint_hash": expected,
        "runtime_manifest_hash": manifest.manifest_hash,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
