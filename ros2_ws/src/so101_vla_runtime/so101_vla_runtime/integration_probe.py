"""Lease/hash/raw-image transport integration probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.node import Node

from so101_vla_runtime.client import CanonicalVlaClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--task", default="pick up the cube and place it in the bowl")
    parser.add_argument(
        "--image-pattern",
        choices=("zeros", "gradient", "random"),
        default="random",
        help="raw RGB transport payload pattern; random is the incompressible worst case",
    )
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = Node("so101_vla_integration_probe")
    report: dict = {}
    try:
        primary = CanonicalVlaClient(node, "probe-primary")
        primary.connect()

        secondary = CanonicalVlaClient(node, "probe-secondary")
        try:
            secondary.connect()
        except RuntimeError as exc:
            report["second_lease_rejected"] = True
            report["second_lease_error"] = str(exc)
        else:
            report["second_lease_rejected"] = False

        if args.image_pattern == "zeros":
            image = np.zeros((480, 640, 3), dtype=np.uint8)
        elif args.image_pattern == "gradient":
            x = np.arange(640, dtype=np.uint16)[None, :]
            y = np.arange(480, dtype=np.uint16)[:, None]
            image = np.stack(
                (
                    np.broadcast_to(x % 256, (480, 640)),
                    np.broadcast_to(y % 256, (480, 640)),
                    (x + y) % 256,
                ),
                axis=-1,
            ).astype(np.uint8)
        else:
            image = np.random.default_rng(101).integers(
                0, 256, size=(480, 640, 3), dtype=np.uint8
            )
        images = {"top": image, "wrist": image, "front": image}
        report["image_pattern"] = args.image_pattern
        report["raw_request_image_bytes"] = int(sum(value.nbytes for value in images.values()))
        packing_ms: list[float] = []
        latencies_ms: list[float] = []
        chunks: list[np.ndarray] = []
        for index in range(args.warmup + args.samples):
            prepare_started = time.perf_counter()
            request = primary.prepare_infer_request(
                request_id=index + 1,
                observation_step=index * primary.chunk_size,
                start_step=index * primary.chunk_size,
                task=args.task,
                canonical_state=np.zeros(6, dtype=np.float32),
                images=images,
            )
            call_started = time.perf_counter()
            _, actions = primary.call_prepared(request, timeout_sec=10.0)
            if index >= args.warmup:
                packing_ms.append((call_started - prepare_started) * 1000.0)
                latencies_ms.append((time.perf_counter() - call_started) * 1000.0)
                chunks.append(actions)
        report["warmup_samples"] = args.warmup
        report["samples"] = len(latencies_ms)
        report["packing_p50_ms"] = float(np.percentile(packing_ms, 50))
        report["packing_p99_ms"] = float(np.percentile(packing_ms, 99))
        report["transport_p50_ms"] = float(np.percentile(latencies_ms, 50))
        report["transport_p99_ms"] = float(np.percentile(latencies_ms, 99))
        report["chunks_bitwise_equal"] = all(
            chunk.tobytes() == chunks[0].tobytes() for chunk in chunks[1:]
        )

        saved = primary.contract_hash
        primary.contract_hash = "tampered"
        try:
            primary.infer(
                request_id=999,
                observation_step=0,
                start_step=0,
                task=args.task,
                canonical_state=np.zeros(6, dtype=np.float32),
                images=images,
                timeout_sec=10.0,
            )
        except RuntimeError as exc:
            report["contract_mismatch_rejected"] = True
            report["contract_mismatch_error"] = str(exc)
        else:
            report["contract_mismatch_rejected"] = False
        finally:
            primary.contract_hash = saved

        report["ok"] = (
            report["second_lease_rejected"]
            and report["chunks_bitwise_equal"]
            and report["contract_mismatch_rejected"]
        )
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["ok"]:
            raise SystemExit(1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
