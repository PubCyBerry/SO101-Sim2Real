#!/usr/bin/env python3
"""저장된 EEF checkpoint를 LeRobot async server 실제 경로로 1회 추론 검증."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lerobot.async_inference.configs import PolicyServerConfig  # noqa: E402
from lerobot.async_inference.helpers import (  # noqa: E402
    RemotePolicyConfig,
    TimedObservation,
)
from lerobot.async_inference.policy_server import PolicyServer  # noqa: E402
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.transport import services_pb2  # noqa: E402

from so101_contract.eef_deployment_contract import (  # noqa: E402
    validate_checkpoint_for_platform,
)
from so101_contract.eef_relative_action import rot6d_rows_to_matrix  # noqa: E402


class _LocalContext:
    @staticmethod
    def peer() -> str:
        return "local-contract-test"


class _CountingPostprocessor:
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.calls: list[tuple[int, ...]] = []

    def __call__(self, action):
        self.calls.append(tuple(action.shape))
        return self.wrapped(action)


def _raw_observation(sample: dict, metadata: LeRobotDatasetMetadata) -> dict:
    state = sample["observation.state"].detach().cpu().numpy()
    names = metadata.features["observation.state"]["names"]
    raw = {name: float(state[index]) for index, name in enumerate(names)}
    for camera_key in metadata.camera_keys:
        image = sample[camera_key]
        if not isinstance(image, torch.Tensor) or image.ndim != 3:
            raise ValueError(f"camera sample must be CHW tensor: {camera_key}={type(image)}")
        raw[camera_key.removeprefix("observation.images.")] = (
            image.permute(1, 2, 0).detach().cpu().numpy()
        )
    raw["task"] = sample["task"]
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--policy-type", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--actions-per-chunk", type=int, default=2)
    parser.add_argument(
        "--rename-map",
        default="{}",
        help="RemotePolicyConfig camera rename JSON",
    )
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
    if args.actions_per_chunk <= 0:
        raise ValueError("--actions-per-chunk must be positive")
    rename_map = json.loads(args.rename_map)
    if not isinstance(rename_map, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in rename_map.items()
    ):
        raise ValueError("--rename-map must decode to a string-to-string object")

    manifest = validate_checkpoint_for_platform(
        args.checkpoint,
        urdf_path=args.urdf_path,
        robot_yaml_path=args.robot_yaml_path,
        policy_type=args.policy_type,
        local_files_only=True,
    )
    metadata = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    observation_features = {
        key: value
        for key, value in metadata.features.items()
        if key.startswith("observation.")
    }
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        return_uint8=True,
    )

    server = PolicyServer(
        PolicyServerConfig(
            host="127.0.0.1",
            port=8080,
            fps=30,
            inference_latency=0.0,
            obs_queue_timeout=1.0,
        )
    )
    context = _LocalContext()
    server.Ready(services_pb2.Empty(), context)
    instructions = RemotePolicyConfig(
        policy_type=args.policy_type,
        pretrained_name_or_path=str(args.checkpoint),
        lerobot_features=observation_features,
        actions_per_chunk=args.actions_per_chunk,
        device=args.device,
        rename_map=rename_map,
    )
    server.SendPolicyInstructions(
        services_pb2.PolicySetup(data=pickle.dumps(instructions)),
        context,
    )
    if not getattr(server, "_eef_relative_actions", False):
        raise AssertionError("async server did not activate EEF-relative mode")
    counting_postprocessor = _CountingPostprocessor(server.postprocessor)
    server.postprocessor = counting_postprocessor

    actions = server._predict_action_chunk(
        TimedObservation(
            timestamp=time.time(),
            timestep=7,
            observation=_raw_observation(dataset[0], metadata),
            must_go=True,
        )
    )
    if len(actions) != args.actions_per_chunk:
        raise AssertionError(
            f"async action chunk length mismatch: {len(actions)} != {args.actions_per_chunk}"
        )
    if len(counting_postprocessor.calls) != 1:
        raise AssertionError(
            "EEF checkpoint postprocessor was not called exactly once: "
            f"{counting_postprocessor.calls}"
        )
    full_chunk_shape = counting_postprocessor.calls[0]
    if len(full_chunk_shape) != 3 or full_chunk_shape[0] != 1 or full_chunk_shape[2] != 10:
        raise AssertionError(
            "EEF checkpoint postprocessor did not receive a canonical [1,H,10] chunk: "
            f"{full_chunk_shape}"
        )
    prediction_horizon = full_chunk_shape[1]
    allowed_horizons = {
        int(manifest["policy"][key])
        for key in ("chunk_size", "execution_horizon")
        if manifest["policy"].get(key) is not None
    }
    if prediction_horizon not in allowed_horizons:
        raise AssertionError(
            "EEF checkpoint returned a horizon not declared by its manifest: "
            f"{prediction_horizon} not in {sorted(allowed_horizons)}"
        )
    if prediction_horizon < args.actions_per_chunk:
        raise AssertionError(
            "policy full chunk is shorter than the requested external slice: "
            f"{prediction_horizon} < {args.actions_per_chunk}"
        )
    action_array = np.stack(
        [action.get_action().detach().cpu().numpy() for action in actions]
    )
    if action_array.shape != (args.actions_per_chunk, 10):
        raise AssertionError(f"async absolute EEF action shape mismatch: {action_array.shape}")
    if not np.all(np.isfinite(action_array)):
        raise AssertionError("async absolute EEF chunk contains non-finite values")
    rotations = rot6d_rows_to_matrix(action_array[:, 3:9])
    if not np.allclose(
        rotations @ np.swapaxes(rotations, -1, -2),
        np.eye(3),
        atol=2e-5,
        rtol=0.0,
    ):
        raise AssertionError("async absolute EEF chunk contains invalid Rot6D rotations")
    print(
        "PASS: async EEF checkpoint full-chunk "
        f"policy={args.policy_type} horizon={prediction_horizon} "
        f"manifest={manifest['manifest_sha256']}"
    )


if __name__ == "__main__":
    main()
