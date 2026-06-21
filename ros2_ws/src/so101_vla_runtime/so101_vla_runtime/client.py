"""Typed ROS service client helper."""

from __future__ import annotations

import numpy as np
import rclpy
from sensor_msgs.msg import Image

from so101_vla_interfaces.srv import GetRuntimeInfo, InferChunk


def numpy_to_image(image: np.ndarray, stamp) -> Image:
    value = np.ascontiguousarray(image, dtype=np.uint8)
    if value.shape != (480, 640, 3):
        raise ValueError(f"RGB image shape은 (480, 640, 3)이어야 한다: {value.shape}")
    message = Image()
    message.header.stamp = stamp
    message.height = value.shape[0]
    message.width = value.shape[1]
    message.encoding = "rgb8"
    message.is_bigendian = False
    message.step = value.shape[1] * 3
    message.data = value.tobytes()
    return message


class CanonicalVlaClient:
    def __init__(self, node, client_id: str) -> None:
        self.node = node
        self.client_id = client_id
        self.runtime_client = node.create_client(
            GetRuntimeInfo, "/so101/vla/get_runtime_info"
        )
        self.infer_client = node.create_client(InferChunk, "/so101/vla/infer_chunk")
        self.lease_token = ""
        self.contract_hash = ""
        self.runtime_manifest_hash = ""
        self.checkpoint_hash = ""
        self.calibration_hash = ""
        self.motor_profile_hash = ""
        self.backend = ""
        self.model_frame = ""
        self.fps = 0
        self.chunk_size = 0

    def connect(self, timeout_sec: float = 10.0) -> None:
        if not self.runtime_client.wait_for_service(timeout_sec=timeout_sec):
            raise TimeoutError("get_runtime_info service 대기 timeout")
        request = GetRuntimeInfo.Request()
        request.client_id = self.client_id
        request.acquire_motion_lease = True
        request.release_motion_lease = False
        request.lease_token = ""
        response = self._call(self.runtime_client, request, timeout_sec)
        if not response.ok:
            raise RuntimeError(response.error)
        self.lease_token = response.lease_token
        self.contract_hash = response.contract_hash
        self.runtime_manifest_hash = response.runtime_manifest_hash
        self.checkpoint_hash = response.checkpoint_hash
        self.calibration_hash = response.calibration_hash
        self.motor_profile_hash = response.motor_profile_hash
        self.backend = response.backend
        self.model_frame = response.model_frame
        self.fps = response.fps
        self.chunk_size = response.chunk_size

    def release(self, timeout_sec: float = 2.0) -> None:
        if not self.lease_token:
            return
        request = GetRuntimeInfo.Request()
        request.client_id = self.client_id
        request.acquire_motion_lease = False
        request.release_motion_lease = True
        request.lease_token = self.lease_token
        response = self._call(self.runtime_client, request, timeout_sec)
        if not response.ok:
            raise RuntimeError(response.error)
        self.lease_token = ""

    def infer(
        self,
        *,
        request_id: int,
        observation_step: int,
        start_step: int,
        task: str,
        canonical_state: np.ndarray,
        images: dict[str, np.ndarray],
        timeout_sec: float,
    ):
        request = self.prepare_infer_request(
            request_id=request_id,
            observation_step=observation_step,
            start_step=start_step,
            task=task,
            canonical_state=canonical_state,
            images=images,
        )
        return self.call_prepared(request, timeout_sec)

    def prepare_infer_request(
        self,
        *,
        request_id: int,
        observation_step: int,
        start_step: int,
        task: str,
        canonical_state: np.ndarray,
        images: dict[str, np.ndarray],
    ) -> InferChunk.Request:
        request = InferChunk.Request()
        request.client_id = self.client_id
        request.lease_token = self.lease_token
        request.request_id = request_id
        request.observation_step = observation_step
        request.start_step = start_step
        request.contract_hash = self.contract_hash
        request.runtime_manifest_hash = self.runtime_manifest_hash
        request.expected_checkpoint_hash = self.checkpoint_hash
        request.task = task
        request.canonical_state = np.asarray(canonical_state, dtype=np.float32).tolist()
        stamp = self.node.get_clock().now().to_msg()
        request.top_image = numpy_to_image(images["top"], stamp)
        request.wrist_image = numpy_to_image(images["wrist"], stamp)
        request.front_image = numpy_to_image(images["front"], stamp)
        return request

    def call_prepared(self, request: InferChunk.Request, timeout_sec: float):
        response = self._call(self.infer_client, request, timeout_sec)
        if not response.ok:
            raise RuntimeError(response.error)
        actions = np.asarray(response.canonical_actions, dtype=np.float32).reshape(
            response.action_count, 6
        )
        return response, actions

    def _call(self, client, request, timeout_sec: float):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout_sec)
        if not future.done():
            raise TimeoutError("ROS service call timeout")
        return future.result()
