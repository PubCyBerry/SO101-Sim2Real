"""Replay backend typed service smoke client."""

from __future__ import annotations

import argparse

import numpy as np
import rclpy
from rclpy.node import Node

from so101_vla_runtime.client import CanonicalVlaClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", default="mock-client")
    parser.add_argument("--task", default="pick up the cube and place it in the bowl")
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = Node("so101_vla_mock_client")
    try:
        client = CanonicalVlaClient(node, args.client_id)
        client.connect()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        response, actions = client.infer(
            request_id=1,
            observation_step=0,
            start_step=0,
            task=args.task,
            canonical_state=np.zeros(6, dtype=np.float32),
            images={"top": image, "wrist": image, "front": image},
            timeout_sec=10.0,
        )
        print(
            f"ok count={len(actions)} latency_ms={response.inference_latency_ms:.3f} "
            f"manifest={response.runtime_manifest_hash}"
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
