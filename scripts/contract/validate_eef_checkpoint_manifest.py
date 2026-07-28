#!/usr/bin/env python3
"""EEF-relative checkpoint manifest의 save/load/tamper fail-fast 검증."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.processor.pipeline import DataProcessorPipeline  # noqa: E402

from so101_contract.eef_action_contract import (  # noqa: E402
    ACTION_REPRESENTATION_CONTRACT_VERSION,
)
from so101_contract.eef_checkpoint_manifest import (  # noqa: E402
    ACTION_REPRESENTATION_MANIFEST,
    load_eef_action_representation_manifest,
    validate_eef_action_representation_manifest,
    validate_eef_checkpoint_artifact,
    write_eef_action_representation_manifest,
)
from so101_contract.eef_deployment_contract import (  # noqa: E402
    sha256_file,
    validate_checkpoint_for_platform,
)
from so101_contract.eef_relative_action import EEF_RELATIVE_ACTION_VERSION  # noqa: E402
from so101_contract.lerobot_eef_processor import (  # noqa: E402
    make_eef_relative_processor_steps,
    reconnect_eef_relative_processor_steps,
)


def _manifest_context(
    fingerprint: str,
    profile_hash: str,
    urdf_sha256: str,
    robot_yaml_sha256: str,
) -> dict:
    representation = {
        "mode": "eef_relative",
        "reference": "current_observation",
        "pose_format": "xyz_rot6d_rows",
        "state_pose_group": "eef_9d",
        "action_pose_group": "eef_9d",
        "passthrough_action_groups": ["gripper_position"],
        "base_frame": "base_link",
        "eef_frame": "tcp_grasp",
        "stats_file": "meta/relative_action_stats.json",
        "strict": True,
    }
    return {
        "representation": representation,
        "resolved_contract": {
            "schema_version": ACTION_REPRESENTATION_CONTRACT_VERSION,
            "config": representation,
            "state_key": "observation.state",
            "action_key": "action",
            "state_dim": 10,
            "action_dim": 10,
            "state_pose_indices": list(range(9)),
            "action_pose_indices": list(range(9)),
            "passthrough_action_indices": [9],
            "state_names": [f"state.{index}" for index in range(10)],
            "action_names": [f"action.{index}" for index in range(10)],
            "base_frame": "base_link",
            "eef_frame": "tcp_grasp",
            "eef_kinematics_version": "fixture_v1",
            "urdf_sha256": urdf_sha256,
            "robot_yaml_sha256": robot_yaml_sha256,
            "info_sha256": "3" * 64,
            "modality_sha256": "4" * 64,
            "fingerprint": fingerprint,
        },
        "dataset": {
            "repo_id": "fixture/so101-eef",
            "revision": "5" * 40,
            "local_fingerprint": fingerprint,
            "source_columns_sha256": "6" * 64,
            "info_sha256": "3" * 64,
            "modality_sha256": "4" * 64,
        },
        "relative_stats": {
            "profile_id": f"sha256:{profile_hash}",
            "content_sha256": profile_hash,
            "sampling": {
                "observation_delta_indices": [0],
                "action_delta_indices": [0, 1, 2, 3],
                "reference_observation_index": -1,
                "reference_delta": 0,
                "horizon": 4,
            },
            "dataset_contract": {
                "contract_fingerprint": fingerprint,
                "source_columns_sha256": "6" * 64,
            },
        },
    }


def main() -> None:
    os.environ["LEROBOT_RUNTIME_VERSION"] = "0.6.0"
    fingerprint = "a" * 64
    profile_hash = "b" * 64
    urdf_path = REPO_ROOT / "assets" / "robots" / "urdf" / "so_arm101.urdf"
    robot_yaml_path = REPO_ROOT / "assets" / "robots" / "so101.yml"
    relative_step, absolute_step = make_eef_relative_processor_steps(
        state_pose_indices=tuple(range(9)),
        action_pose_indices=tuple(range(9)),
        passthrough_action_indices=(9,),
        contract_fingerprint=fingerprint,
        manifest_context=_manifest_context(
            fingerprint,
            profile_hash,
            sha256_file(urdf_path),
            sha256_file(robot_yaml_path),
        ),
    )
    preprocessor = DataProcessorPipeline(
        steps=[relative_step],
        name="policy_preprocessor",
    )
    postprocessor = DataProcessorPipeline(
        steps=[absolute_step],
        name="policy_postprocessor",
    )
    policy_cfg = SimpleNamespace(
        type="act",
        chunk_size=4,
        n_action_steps=4,
        base_model_path=None,
    )

    scratch = REPO_ROOT / "scratch"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="eef-manifest-check-", dir=scratch) as directory:
        root = Path(directory)
        preprocessor.save_pretrained(root, config_filename="policy_preprocessor.json")
        postprocessor.save_pretrained(root, config_filename="policy_postprocessor.json")
        output = write_eef_action_representation_manifest(root, policy_cfg, preprocessor)
        if output != root / ACTION_REPRESENTATION_MANIFEST or not output.is_file():
            raise AssertionError("action representation manifest was not written")

        loaded_pre = DataProcessorPipeline.from_pretrained(
            root,
            config_filename="policy_preprocessor.json",
            local_files_only=True,
        )
        loaded_post = DataProcessorPipeline.from_pretrained(
            root,
            config_filename="policy_postprocessor.json",
            local_files_only=True,
        )
        reconnect_eef_relative_processor_steps(loaded_pre, loaded_post)
        manifest = validate_eef_checkpoint_artifact(
            root,
            policy_cfg,
            loaded_pre,
            local_files_only=True,
        )
        if manifest["transform_version"] != EEF_RELATIVE_ACTION_VERSION:
            raise AssertionError("transform version was not preserved")
        if manifest["relative_stats"]["profile_id"] != f"sha256:{profile_hash}":
            raise AssertionError("relative stats profile was not preserved")
        validate_checkpoint_for_platform(
            root,
            urdf_path=urdf_path,
            robot_yaml_path=robot_yaml_path,
            policy_type="act",
            local_files_only=True,
        )

        tampered = json.loads(json.dumps(manifest))
        tampered["resolved_contract"]["eef_frame"] = "wrong_tcp"
        try:
            validate_eef_action_representation_manifest(tampered)
        except ValueError as exc:
            if "content hash" not in str(exc):
                raise
        else:
            raise AssertionError("tampered manifest was accepted")

        wrong_policy = SimpleNamespace(type="smolvla")
        try:
            validate_eef_action_representation_manifest(
                load_eef_action_representation_manifest(root),
                policy_cfg=wrong_policy,
                preprocessor=loaded_pre,
            )
        except ValueError as exc:
            if "policy type mismatch" not in str(exc):
                raise
        else:
            raise AssertionError("wrong policy runtime was accepted")

    print("PASS: self-contained EEF checkpoint manifest")


if __name__ == "__main__":
    main()
