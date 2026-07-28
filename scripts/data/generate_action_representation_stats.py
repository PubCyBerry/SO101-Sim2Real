#!/usr/bin/env python3
"""Phase 15 — schema v2 universal action representation stats 생성 CLI.

LeRobot v3 dataset(absolute joint 또는 absolute EEF)에서 mode/pose-format/horizon별
stats profile을 생성해 ``meta/action_representation_stats.json`` 한 파일에 누적한다.
absolute와 relative profile이 공존하며, dataset을 수정하지 않고 metadata artifact만 만든다.

.. code-block:: bash

    # dataset이 지원하는 모든 representation
    python scripts/data/generate_action_representation_stats.py \\
        --dataset-root datasets/so101_pick_cube_eef --horizon 50 --all

    # 특정 조합만
    python scripts/data/generate_action_representation_stats.py \\
        --dataset-root datasets/so101_pick_cube_eef --horizon 50 \\
        --mode eef_relative --pose-format xyz_rot6d_rows

v1 전용 생성기는 ``scripts/data/generate_relative_action_stats.py``로 남아 있다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from so101_contract.action_dataset_contract import (  # noqa: E402
    resolve_action_representation_contract,
)
from so101_contract.action_representation import (  # noqa: E402
    EEF_POSE_FORMATS,
    ActionRepresentationMode,
    ActionRepresentationSpec,
    coerce_mode,
    coerce_pose_format,
)
from so101_contract.action_representation_stats import (  # noqa: E402
    ACTION_STATS_DEFAULT_PATH,
    ActionStatsSampling,
    calculate_action_representation_stats,
    empty_stats_artifact,
    load_lerobot_v3_episodes,
    read_action_stats_artifact,
    upsert_stats_profile,
    write_action_stats_artifact,
)


def _candidate_specs(args: argparse.Namespace) -> list[ActionRepresentationSpec]:
    if args.all:
        specs = [
            ActionRepresentationSpec(mode=ActionRepresentationMode.JOINT_ABSOLUTE),
            ActionRepresentationSpec(mode=ActionRepresentationMode.JOINT_RELATIVE),
        ]
        for pose_format in EEF_POSE_FORMATS:
            specs.extend(
                ActionRepresentationSpec(mode=mode, pose_format=pose_format)
                for mode in (
                    ActionRepresentationMode.EEF_ABSOLUTE,
                    ActionRepresentationMode.EEF_RELATIVE,
                )
            )
        return specs
    mode = coerce_mode(args.mode)
    pose_format = coerce_pose_format(args.pose_format) if args.pose_format else None
    return [ActionRepresentationSpec(mode=mode, pose_format=pose_format)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--horizon", type=int, required=True, help="action chunk 길이")
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in ActionRepresentationMode],
        help="--all이 없으면 필수",
    )
    parser.add_argument(
        "--pose-format",
        choices=[fmt.value for fmt in EEF_POSE_FORMATS],
        help="EEF mode에서 필수",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="dataset이 지원하는 representation을 모두 생성(지원하지 않는 조합은 건너뛰고 사유를 출력)",
    )
    parser.add_argument("--output-file", default=ACTION_STATS_DEFAULT_PATH)
    parser.add_argument("--max-episodes", type=int, default=None, help="디버그용 subset")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.all and not args.mode:
        parser.error("--mode is required unless --all is given")
    if args.horizon <= 0:
        parser.error("--horizon must be positive")

    root = args.dataset_root.resolve()
    sampling = ActionStatsSampling(action_delta_indices=tuple(range(args.horizon)))
    artifact = read_action_stats_artifact(root, output_file=args.output_file)
    if not artifact.get("profiles"):
        artifact = empty_stats_artifact()

    created: dict[str, str] = {}
    skipped: dict[str, str] = {}
    for spec in _candidate_specs(args):
        try:
            contract = resolve_action_representation_contract(root, spec)
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            if not args.all:
                raise
            skipped[spec.stats_profile_kind] = str(exc)
            continue
        episodes = load_lerobot_v3_episodes(
            root,
            state_key=contract.state_key,
            action_key=contract.action_key,
            state_dim=contract.state_dim,
            action_dim=contract.action_dim,
            max_episodes=args.max_episodes,
        )
        result = calculate_action_representation_stats(
            episodes,
            sampling,
            contract.transform,
            dataset_fingerprint=contract.fingerprint,
            max_episodes=args.max_episodes,
            production=args.max_episodes is None,
        )
        artifact, changed = upsert_stats_profile(artifact, result, overwrite=args.overwrite)
        created[spec.stats_profile_kind] = result.profile_id
        if not changed:
            skipped[f"{spec.stats_profile_kind}:unchanged"] = "identical profile already present"

    if not created:
        print(json.dumps({"created": {}, "skipped": skipped}, indent=2))
        raise SystemExit("no stats profile could be generated for this dataset")

    path = write_action_stats_artifact(root, artifact, output_file=args.output_file)
    print(
        json.dumps(
            {
                "artifact": str(path),
                "horizon": args.horizon,
                "created": created,
                "skipped": skipped,
                "production": args.max_episodes is None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
