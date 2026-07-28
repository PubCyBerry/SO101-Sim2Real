#!/usr/bin/env python3
"""Phase 16 — checkpoint action representation startup assertion.

policy-server·sim client·real client가 기동 **전에** 같은 loader로 checkpoint 계약을
확인한다. CLI/env의 representation 값은 override가 아니라 **assertion**이며, 불일치하면
로봇/sim 명령을 만들기 전에 프로세스를 멈춘다. 값을 주지 않으면 manifest 값을 그대로 쓴다.

.. code-block:: bash

    python scripts/inference/assert_checkpoint_representation.py \\
        --checkpoint /workspace/outputs/train/run/checkpoints/last/pretrained_model \\
        --mode eef_relative --pose-format xyz_rot6d_rows --policy-type act

    # env(ACTION_REPRESENTATION_MODE/…)만으로도 동작한다.
    python scripts/inference/assert_checkpoint_representation.py --checkpoint <path> --from-env
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from so101_contract.inference_startup import (  # noqa: E402
    describe_plan,
    plan_inference_startup,
)


def _env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="local 디렉터리 또는 HF repo id")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--mode", default=None)
    parser.add_argument("--pose-format", default=None)
    parser.add_argument("--policy-type", default=None)
    parser.add_argument(
        "--from-env",
        action="store_true",
        help="ACTION_REPRESENTATION_MODE/ACTION_REPRESENTATION_POSE_FORMAT/POLICY_TYPE 사용",
    )
    parser.add_argument("--json", action="store_true", help="계약 요약을 JSON으로 출력")
    parser.add_argument(
        "--emit",
        choices=("mode", "pose_format", "client_kind", "policy_type", "action_dim"),
        help="machine-readable 단일 필드 출력(쉘 dispatch용)",
    )
    parser.add_argument("--urdf-path", default=None, help="EEF mode kinematics hash 검증용")
    parser.add_argument("--robot-yaml-path", default=None)
    parser.add_argument(
        "--skip-kinematics",
        action="store_true",
        help="URDF/YAML 없이 계약만 확인(dispatch 조회 용도)",
    )
    args = parser.parse_args()

    mode = args.mode
    pose_format = args.pose_format
    policy_type = args.policy_type
    if args.from_env:
        mode = mode or _env("ACTION_REPRESENTATION_MODE")
        pose_format = pose_format or _env("ACTION_REPRESENTATION_POSE_FORMAT")
        policy_type = policy_type or _env("POLICY_TYPE")

    plan = plan_inference_startup(
        args.checkpoint,
        mode=mode,
        pose_format=pose_format,
        policy_type=policy_type,
        revision=args.revision,
        local_files_only=args.local_files_only,
        urdf_path=args.urdf_path,
        robot_yaml_path=args.robot_yaml_path,
        verify_kinematics=not args.skip_kinematics,
    )
    contract = plan.contract

    if args.emit:
        print(plan.to_dict()[args.emit])
        return 0
    if args.json:
        print(
            json.dumps(
                {
                    "source": contract.source,
                    "mode": contract.spec.mode.value,
                    "pose_format": contract.spec.pose_format.value,
                    "state_dim": contract.state_dim,
                    "action_dim": contract.action_dim,
                    "policy_type": contract.policy_type,
                    "chunk_size": contract.chunk_size,
                    "execution_horizon": contract.execution_horizon,
                    "routing": list(contract.routing),
                    "requires_ik": contract.requires_ik,
                    "client_kind": plan.client_kind,
                    "legacy_opt_in": contract.legacy_opt_in,
                    "manifest_sha256": contract.manifest_sha256,
                },
                indent=2,
            )
        )
    else:
        print(f"[action-representation] {describe_plan(plan)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
