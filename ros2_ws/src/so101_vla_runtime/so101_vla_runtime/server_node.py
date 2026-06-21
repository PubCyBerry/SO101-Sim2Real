"""Deterministic canonical VLA ROS 2 server."""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from so101_parity.calibration import CalibrationBundle
from so101_parity.contract import CAMERA_ORDER, PolicyIOContract
from so101_parity.manifest import RuntimeManifest, file_sha256
from so101_vla_interfaces.srv import GetRuntimeInfo, GetStatus, InferChunk
from so101_vla_runtime.backends import make_backend
from so101_vla_runtime.lease import LeaseError, MotionLease


def _image_to_numpy(message, expected_shape: tuple[int, int, int]) -> np.ndarray:
    if message.encoding.lower() != "rgb8":
        raise ValueError(f"image encoding은 rgb8이어야 한다: {message.encoding!r}")
    if (message.height, message.width, 3) != expected_shape:
        raise ValueError(
            f"image shape 불일치: {(message.height, message.width, 3)} != {expected_shape}"
        )
    expected_step = message.width * 3
    if message.step != expected_step or len(message.data) != message.height * expected_step:
        raise ValueError("image step/data 길이가 packed rgb8 계약과 다르다")
    return np.frombuffer(message.data, dtype=np.uint8).reshape(expected_shape).copy()


class VlaServer(Node):
    def __init__(
        self,
        *,
        manifest_path: Path,
        contract_path: Path,
        calibration_path: Path,
        runtime_config_path: Path,
        pixi_lock_path: Path,
    ) -> None:
        super().__init__("so101_vla_server")
        self.manifest = RuntimeManifest.load(manifest_path)
        self.contract = PolicyIOContract.load(contract_path)
        self.calibration = CalibrationBundle.load(calibration_path)
        self.manifest.assert_hashes(
            contract_hash=self.contract.contract_hash,
            calibration_hash=self.calibration.calibration_hash,
            motor_profile_hash=self.calibration.motor_profile_hash,
            checkpoint_hash=str(self.manifest.raw["checkpoint_hash"]),
            pixi_lock_hash=file_sha256(pixi_lock_path),
            runtime_config_hash=file_sha256(runtime_config_path),
        )
        self.backend = make_backend(dict(self.manifest.raw), manifest_path.parent, self.calibration)
        self.lease = MotionLease(int(self.manifest.raw.get("lease_duration_ms", 5000)))
        self._inference_lock = threading.Lock()
        self._inference_in_flight = False
        self._request_count = 0
        self._failure_count = 0

        self.create_service(
            GetRuntimeInfo,
            "/so101/vla/get_runtime_info",
            self._get_runtime_info,
        )
        self.create_service(InferChunk, "/so101/vla/infer_chunk", self._infer_chunk)
        self.create_service(GetStatus, "/so101/vla/status", self._get_status)
        self._status_publisher = self.create_publisher(String, "/so101/vla/status_json", 10)
        self.create_timer(1.0, self._publish_status)
        self.get_logger().info(
            f"VLA server ready backend={self.manifest.raw['backend']} "
            f"manifest={self.manifest.manifest_hash} checkpoint={self.backend.checkpoint_hash}"
        )

    def _get_runtime_info(self, request, response):
        try:
            lease_token = ""
            expires_at = 0
            if request.acquire_motion_lease and request.release_motion_lease:
                raise ValueError("lease acquire와 release를 동시에 요청할 수 없다")
            if request.release_motion_lease:
                self.lease.release(request.client_id, request.lease_token)
            if request.acquire_motion_lease:
                snapshot = self.lease.acquire(request.client_id)
                lease_token = snapshot.token
                expires_at = snapshot.expires_at_ns
            response.ok = True
            response.error = ""
            response.lease_token = lease_token
            response.lease_expires_at_ns = expires_at
        except Exception as exc:  # noqa: BLE001
            response.ok = False
            response.error = str(exc)
        response.contract_hash = self.contract.contract_hash
        response.runtime_manifest_hash = self.manifest.manifest_hash
        response.checkpoint_hash = self.backend.checkpoint_hash
        response.calibration_hash = self.calibration.calibration_hash
        response.motor_profile_hash = self.calibration.motor_profile_hash
        response.backend = str(self.manifest.raw["backend"])
        response.model_frame = str(self.manifest.raw["model_frame"])
        response.fps = self.contract.fps
        response.chunk_size = int(self.manifest.raw["chunk_size"])
        return response

    def _validate_request(self, request) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if request.contract_hash != self.contract.contract_hash:
            raise ValueError("contract hash mismatch")
        if request.runtime_manifest_hash != self.manifest.manifest_hash:
            raise ValueError("runtime manifest hash mismatch")
        if request.expected_checkpoint_hash != self.backend.checkpoint_hash:
            raise ValueError("checkpoint hash mismatch")
        if request.task != self.manifest.raw["task"]:
            raise ValueError("task mismatch")
        self.lease.validate_and_renew(request.client_id, request.lease_token)
        state = self.contract.validate_state(request.canonical_state)
        images = {
            "top": _image_to_numpy(request.top_image, self.contract.image_shape),
            "wrist": _image_to_numpy(request.wrist_image, self.contract.image_shape),
            "front": _image_to_numpy(request.front_image, self.contract.image_shape),
        }
        return state, self.contract.validate_images(images)

    def _infer_chunk(self, request, response):
        response.request_id = request.request_id
        response.start_step = request.start_step
        response.contract_hash = self.contract.contract_hash
        response.runtime_manifest_hash = self.manifest.manifest_hash
        response.checkpoint_hash = self.backend.checkpoint_hash
        self._request_count += 1
        try:
            canonical_state, images = self._validate_request(request)
            if not self._inference_lock.acquire(blocking=False):
                raise RuntimeError("inference request가 이미 in-flight다")
            self._inference_in_flight = True
            try:
                started = time.perf_counter()
                actions = self.backend.infer(
                    canonical_state=canonical_state,
                    images=images,
                    task=request.task,
                    start_step=request.start_step,
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
            finally:
                self._inference_in_flight = False
                self._inference_lock.release()
            if actions.ndim != 2 or actions.shape[1] != 6 or not np.all(np.isfinite(actions)):
                raise ValueError(f"backend action chunk shape/value 오류: {actions.shape}")
            response.ok = True
            response.error = ""
            response.action_count = len(actions)
            response.canonical_actions = actions.astype(np.float32).reshape(-1).tolist()
            response.inference_latency_ms = elapsed_ms
        except Exception as exc:  # noqa: BLE001
            self._failure_count += 1
            response.ok = False
            response.error = str(exc)
            response.action_count = 0
            response.canonical_actions = []
            response.inference_latency_ms = 0.0
            self.get_logger().error(f"infer request {request.request_id} 거부: {exc}")
        return response

    def _get_status(self, request, response):
        del request
        snapshot = self.lease.snapshot()
        response.ok = True
        response.error = ""
        response.active_client_id = "" if snapshot is None else snapshot.client_id
        response.lease_expires_at_ns = 0 if snapshot is None else snapshot.expires_at_ns
        response.inference_in_flight = self._inference_in_flight
        response.request_count = self._request_count
        response.failure_count = self._failure_count
        response.runtime_manifest_hash = self.manifest.manifest_hash
        return response

    def _publish_status(self) -> None:
        snapshot = self.lease.snapshot()
        payload = {
            "active_client_id": None if snapshot is None else snapshot.client_id,
            "lease_expires_at_ns": 0 if snapshot is None else snapshot.expires_at_ns,
            "inference_in_flight": self._inference_in_flight,
            "request_count": self._request_count,
            "failure_count": self._failure_count,
            "runtime_manifest_hash": self.manifest.manifest_hash,
        }
        message = String()
        message.data = json.dumps(payload, separators=(",", ":"))
        self._status_publisher.publish(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--pixi-lock", type=Path, required=True)
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = VlaServer(
        manifest_path=args.manifest.resolve(),
        contract_path=args.contract.resolve(),
        calibration_path=args.calibration.resolve(),
        runtime_config_path=args.runtime_config.resolve(),
        pixi_lock_path=args.pixi_lock.resolve(),
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
