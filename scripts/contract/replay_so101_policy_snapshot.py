#!/usr/bin/env python3
"""저장된 동일 observation을 policy-server에 replay하고 action 차이를 비교한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle  # nosec
import sys
import time

import grpc
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "ros2_ws" / "src" / "so101_vla_policy" / "vendor"))

from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation  # noqa: E402
from lerobot.transport import services_pb2, services_pb2_grpc  # noqa: E402
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks  # noqa: E402
from so101_contract.feature_codec import policy_feature_to_sim_joint_radians  # noqa: E402
from so101_contract.policy_snapshot import load_policy_io_snapshot  # noqa: E402


def _online_replay(snapshot: dict, server_address: str, timeout: float) -> tuple[np.ndarray, np.ndarray]:
    manifest = snapshot["manifest"]
    metadata = manifest.get("metadata", {})
    required = (
        "policy_type",
        "pretrained_name_or_path",
        "actions_per_chunk",
        "policy_device",
        "lerobot_features",
    )
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"snapshot metadata missing online replay fields: {missing}")

    channel = grpc.insecure_channel(server_address, grpc_channel_options(initial_backoff="0.0333s"))
    stub = services_pb2_grpc.AsyncInferenceStub(channel)
    try:
        stub.Ready(services_pb2.Empty())
        policy_config = RemotePolicyConfig(
            metadata["policy_type"],
            metadata["pretrained_name_or_path"],
            metadata["lerobot_features"],
            int(metadata["actions_per_chunk"]),
            metadata["policy_device"],
            {},
        )
        stub.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(policy_config)))
        timed_observation = TimedObservation(
            timestamp=time.time(),
            timestep=int(manifest["request_timestep"]),
            observation=snapshot["observation"],
            must_go=bool(manifest["must_go"]),
        )
        iterator = send_bytes_in_chunks(
            pickle.dumps(timed_observation),
            services_pb2.Observation,
            log_prefix="[replay] observation",
            silent=True,
        )
        stub.SendObservations(iterator)

        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            response = stub.GetActions(services_pb2.Empty())
            if len(response.data) == 0:
                time.sleep(0.005)
                continue
            timed_actions = pickle.loads(response.data)  # nosec
            timesteps = np.asarray([action.get_timestep() for action in timed_actions], dtype=np.int64)
            actions = np.stack(
                [action.get_action().detach().cpu().numpy().astype(np.float32) for action in timed_actions]
            )
            return timesteps, actions
        raise TimeoutError(f"policy-server returned no actions within {timeout}s")
    finally:
        channel.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--server-address", default="", help="비우면 offline decode 검증만 수행")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=None, help="online replay action NPZ 저장 경로")
    args = parser.parse_args()

    snapshot = load_policy_io_snapshot(args.snapshot)
    decoded = policy_feature_to_sim_joint_radians(snapshot["actions_feature"])
    decode_max_abs = float(np.max(np.abs(decoded - snapshot["actions_sim_rad"])))
    result = {
        "snapshot": str(args.snapshot.resolve()),
        "request_timestep": snapshot["manifest"]["request_timestep"],
        "recorded_actions": int(snapshot["actions_feature"].shape[0]),
        "offline_decode_max_abs_rad": decode_max_abs,
    }

    if args.server_address:
        replay_timesteps, replay_actions = _online_replay(
            snapshot,
            args.server_address,
            args.timeout,
        )
        recorded_actions = snapshot["actions_feature"]
        comparable = min(len(recorded_actions), len(replay_actions))
        result.update(
            {
                "server_address": args.server_address,
                "replay_actions": int(len(replay_actions)),
                "timestep_equal": bool(
                    np.array_equal(snapshot["action_timesteps"][:comparable], replay_timesteps[:comparable])
                ),
                "action_max_abs": (
                    float(np.max(np.abs(recorded_actions[:comparable] - replay_actions[:comparable])))
                    if comparable else None
                ),
            }
        )
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.output,
                action_timesteps=replay_timesteps,
                actions_feature=replay_actions,
                actions_sim_rad=policy_feature_to_sim_joint_radians(replay_actions),
            )
            result["output"] = str(args.output.resolve())

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
