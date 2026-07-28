#!/usr/bin/env python3
"""EEF checkpoint의 recorded absolute target 대비 open-loop full-chunk overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import default_collate

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata  # noqa: E402
from lerobot.datasets.factory import resolve_delta_timestamps  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies import get_policy_class, make_pre_post_processors  # noqa: E402

from so101_contract.eef_deployment_contract import validate_checkpoint_for_platform  # noqa: E402
from so101_contract.eef_relative_action import rot6d_rows_to_matrix  # noqa: E402
from so101_contract.lerobot_full_chunk import FullChunkPolicyRunner  # noqa: E402


def _rotation_error_rad(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    predicted_rotation = rot6d_rows_to_matrix(predicted[..., 3:9])
    target_rotation = rot6d_rows_to_matrix(target[..., 3:9])
    relative = target_rotation @ np.swapaxes(predicted_rotation, -1, -2)
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    return np.arccos(cosine)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--dataset-revision")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--urdf-path",
        type=Path,
        default=ROOT / "assets" / "robots" / "urdf" / "so_arm101.urdf",
    )
    parser.add_argument(
        "--robot-yaml-path",
        type=Path,
        default=ROOT / "assets" / "robots" / "so101.yml",
    )
    args = parser.parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")

    policy_cfg = PreTrainedConfig.from_pretrained(
        args.checkpoint,
        local_files_only=args.local_files_only,
    )
    manifest = validate_checkpoint_for_platform(
        args.checkpoint,
        urdf_path=args.urdf_path,
        robot_yaml_path=args.robot_yaml_path,
        policy_type=policy_cfg.type,
        local_files_only=args.local_files_only,
    )
    dataset_source = manifest["dataset"]
    repo_id = dataset_source.get("repo_id")
    if not isinstance(repo_id, str) or not repo_id:
        raise ValueError("checkpoint manifest has no dataset repo_id")
    revision = args.dataset_revision or dataset_source.get("revision")

    metadata = LeRobotDatasetMetadata(
        repo_id,
        root=args.dataset_root,
        revision=revision,
    )
    dataset = LeRobotDataset(
        repo_id,
        root=args.dataset_root,
        revision=revision,
        delta_timestamps=resolve_delta_timestamps(policy_cfg, metadata),
        return_uint8=True,
    )

    policy_class = get_policy_class(policy_cfg.type)
    policy = policy_class.from_pretrained(
        args.checkpoint,
        local_files_only=args.local_files_only,
    ).to(args.device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=args.checkpoint,
        pretrained_revision=None,
    )
    runner = FullChunkPolicyRunner(
        policy,
        preprocessor,
        postprocessor,
        execution_horizon=None,
    )

    sample_indices = np.linspace(
        0,
        max(0, len(dataset) - 1),
        num=min(args.num_samples, len(dataset)),
        dtype=np.int64,
    )
    overlays: list[dict] = []
    translation_errors: list[np.ndarray] = []
    rotation_errors: list[np.ndarray] = []
    gripper_errors: list[np.ndarray] = []
    with torch.inference_mode():
        for sample_index in sample_indices:
            batch = default_collate([dataset[int(sample_index)]])
            target = batch["action"].detach().cpu().numpy()[0]
            action_is_pad = batch.get("action_is_pad")
            valid = (
                np.ones(target.shape[0], dtype=bool)
                if action_is_pad is None
                else ~action_is_pad.detach().cpu().numpy()[0].astype(bool)
            )
            inference_batch = {
                key: value
                for key, value in batch.items()
                if key not in {"action", "action_is_pad"}
            }
            for camera_key in dataset.meta.camera_keys:
                if (
                    camera_key in inference_batch
                    and inference_batch[camera_key].dtype == torch.uint8
                ):
                    inference_batch[camera_key] = (
                        inference_batch[camera_key].to(dtype=torch.float32) / 255.0
                    )

            runner.reset()
            predicted = runner.refill(inference_batch).detach().cpu().numpy()[0]
            horizon = min(len(predicted), len(target), len(valid))
            predicted = predicted[:horizon]
            target = target[:horizon]
            valid = valid[:horizon]
            if not np.any(valid):
                continue
            translation = np.linalg.norm(predicted[:, :3] - target[:, :3], axis=-1)[valid]
            rotation = _rotation_error_rad(predicted, target)[valid]
            gripper = np.abs(predicted[:, 9] - target[:, 9])[valid]
            translation_errors.append(translation)
            rotation_errors.append(rotation)
            gripper_errors.append(gripper)
            overlays.append(
                {
                    "dataset_index": int(sample_index),
                    "valid_horizon": int(np.sum(valid)),
                    "predicted_absolute_eef": predicted[valid].tolist(),
                    "recorded_absolute_eef": target[valid].tolist(),
                }
            )

    if not overlays:
        raise RuntimeError("no valid open-loop samples were evaluated")
    translation = np.concatenate(translation_errors)
    rotation = np.concatenate(rotation_errors)
    gripper = np.concatenate(gripper_errors)
    report = {
        "schema_version": "so101_eef_open_loop_overlay_v1",
        "checkpoint": args.checkpoint,
        "manifest_sha256": manifest["manifest_sha256"],
        "policy_type": policy_cfg.type,
        "dataset": dataset_source,
        "samples": len(overlays),
        "valid_targets": int(len(translation)),
        "metrics": {
            "translation_error_mean_m": float(np.mean(translation)),
            "translation_error_max_m": float(np.max(translation)),
            "rotation_error_mean_rad": float(np.mean(rotation)),
            "rotation_error_max_rad": float(np.max(rotation)),
            "gripper_error_mean": float(np.mean(gripper)),
            "gripper_error_max": float(np.max(gripper)),
        },
        "overlays": overlays,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"PASS: EEF open-loop overlay → {args.output}")


if __name__ == "__main__":
    main()
