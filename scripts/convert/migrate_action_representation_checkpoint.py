#!/usr/bin/env python3
"""Phase 16 — legacy checkpoint를 schema v2로 migration하는 CLI.

**원본 checkpoint는 바뀌지 않는다.** 새 디렉터리에 복사본을 만들고 거기에만 v2 processor
pair와 ``action_representation.json``을 추가한다(임시 디렉터리 → atomic publish).

지원 경로는 두 가지뿐이고 둘 다 명시적이다.

.. code-block:: bash

    # 1) manifest 없는 checkpoint를 joint_absolute로 선언(정확한 flag 필수)
    python scripts/convert/migrate_action_representation_checkpoint.py \\
        --source outputs/train/old/checkpoints/last/pretrained_model \\
        --output outputs/train/old/checkpoints/last/pretrained_model_v2 \\
        --dataset-root datasets/joint_v3 --horizon 50 \\
        --allow-legacy-joint-absolute-checkpoint

    # 2) v1 EEF-relative(xyz_rot6d_rows) checkpoint
    python scripts/convert/migrate_action_representation_checkpoint.py \\
        --source outputs/train/eef/checkpoints/last/pretrained_model \\
        --output outputs/train/eef/checkpoints/last/pretrained_model_v2 \\
        --dataset-root datasets/eef_v3 --horizon 50

차원·config 이름·feature 이름으로 representation을 추정하지 않는다. 지원하지 않는 legacy
형식, 잘못된 flag, in-place 대상, 모호한 입력은 모두 즉시 실패한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from so101_contract.action_manifest import LEGACY_JOINT_ABSOLUTE_OPT_IN  # noqa: E402
from so101_contract.action_migration import (  # noqa: E402
    detect_source_schema_state,
    migrate_checkpoint,
    plan_migration,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="legacy checkpoint 디렉터리")
    parser.add_argument("--output", type=Path, required=True, help="새로 만들 v2 checkpoint 디렉터리")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="v2 계약/stats를 재구성할 학습 dataset(LeRobot v3)",
    )
    parser.add_argument("--horizon", type=int, required=True, help="stats profile horizon")
    parser.add_argument("--mode", default=None, help="명시적 mode assertion(선택)")
    parser.add_argument("--pose-format", default=None, help="명시적 pose format assertion(선택)")
    parser.add_argument(
        LEGACY_JOINT_ABSOLUTE_OPT_IN,
        dest="allow_legacy_joint_absolute",
        action="store_true",
        help="manifest 없는 checkpoint를 joint_absolute로 선언(이 flag 없이는 거부)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="이미 migration된 output 디렉터리만 교체한다(임의의 비어 있지 않은 경로는 거부)",
    )
    parser.add_argument("--inspect", action="store_true", help="source schema 상태만 출력")
    args = parser.parse_args()

    if args.inspect:
        state = detect_source_schema_state(args.source.resolve())
        print(json.dumps({"source": str(args.source), "source_schema_state": state}, indent=2))
        return 0

    plan = plan_migration(
        args.source,
        args.output,
        dataset_root=args.dataset_root,
        horizon=args.horizon,
        mode=args.mode,
        pose_format=args.pose_format,
        allow_legacy_joint_absolute=args.allow_legacy_joint_absolute,
    )
    result = migrate_checkpoint(plan, overwrite=args.overwrite)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
