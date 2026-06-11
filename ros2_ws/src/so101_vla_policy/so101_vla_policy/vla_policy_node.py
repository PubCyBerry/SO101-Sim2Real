"""SO-101 VLA 추론 ROS 2 노드.

Isaac Sim bridge(`run_cube_desk_ros_bridge.py`)가 publish 하는 관측을 받아 LeRobot
async-inference `policy-server`(gRPC, SmolVLA/ACT)로 추론하고 6관절 joint target 을
publish 한다.

구독:
  /isaac_joint_states                 sensor_msgs/JointState  (6관절, rad)
  /camera/top/image_raw               sensor_msgs/Image       (rgb8 480x640)
  /camera/wrist/image_raw             sensor_msgs/Image
  /camera/front/image_raw             sensor_msgs/Image
발행:
  /isaac_joint_commands               sensor_msgs/JointState  (6관절 target, rad)
    (sim: bridge ArticulationController 가 직접 적용. 실기기: joint_command_to_trajectory shim 경유)

정책 파라미터는 실기기 policy-client 와 동일한 `.env`+`env/<POLICY_PROFILE>.env` 에서
읽고 ROS param 으로 override 한다. 전처리(rename/resize/normalize/추론)는 서버측이라
이 노드는 raw obs(state LeRobot 단위 + 이미지 uint8 + task)만 보낸다.
"""

from __future__ import annotations

import json
import os
import pickle  # nosec
import time
from collections import deque
from pathlib import Path

import grpc
import numpy as np
import rclpy
from cv_bridge import CvBridge
from dotenv import load_dotenv
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

from lerobot.async_inference.helpers import (
    RemotePolicyConfig,
    TimedObservation,
    map_robot_keys_to_lerobot_features,
)
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SO101Follower
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks

from so101_vla_policy.units import (
    CAMERA_KEYS,
    JOINT_FEATURE_NAMES,
    SO101_JOINT_ORDER,
    clamp_joint_rad,
    from_lerobot_units,
    to_lerobot_units,
)

_CAMERA_TOPICS = {
    "top": "/camera/top/image_raw",
    "wrist": "/camera/wrist/image_raw",
    "front": "/camera/front/image_raw",
}


def _load_env(env_file: str, env_dir: str) -> None:
    """`.env` → `env/<POLICY_PROFILE>.env` 순서로 os.environ 로드(나중 파일 override).

    .env 를 먼저 로드해 HF_USER 등이 들어가야 프로필의 ${HF_USER} 보간이 풀린다.
    """
    base = Path(env_file)
    if base.exists():
        load_dotenv(base)
        print(f"[vla] loaded {base}", flush=True)
    else:
        print(f"[vla] WARN: {base} 없음 (ROS param 또는 셸 env 사용)", flush=True)
    profile = os.getenv("POLICY_PROFILE")
    if profile:
        prof = Path(env_dir) / f"{profile}.env"
        if prof.exists():
            load_dotenv(prof, override=True)
            print(f"[vla] loaded profile {prof}", flush=True)


class PolicyServerSession:
    """policy-server gRPC 세션 — 핸드셰이크 + obs 송신 + action chunk 수신."""

    def __init__(self, *, server_address, policy_type, pretrained, actions_per_chunk,
                 policy_device, rename_map, poll_timeout):
        self._poll_timeout = float(poll_timeout)

        cams = {n: OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30)
                for n in ("top", "wrist", "front")}
        robot = SO101Follower(SO101FollowerConfig(port="", id="sim_vla_ros", cameras=cams))
        lerobot_features = map_robot_keys_to_lerobot_features(robot)

        self.policy_config = RemotePolicyConfig(
            policy_type, pretrained, lerobot_features, int(actions_per_chunk), policy_device,
            rename_map or {},
        )
        self.channel = grpc.insecure_channel(
            server_address, grpc_channel_options(initial_backoff="0.0333s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        print(f"[vla] connecting to policy-server at {server_address}", flush=True)
        self.stub.Ready(services_pb2.Empty())
        self.stub.SendPolicyInstructions(
            services_pb2.PolicySetup(data=pickle.dumps(self.policy_config))
        )
        print(f"[vla] sent instructions (type={policy_type}, model={pretrained}, "
              f"chunk={actions_per_chunk}, rename={bool(rename_map)})", flush=True)

    def predict_chunk(self, raw_obs: dict, timestep: int) -> list:
        timed_obs = TimedObservation(
            timestamp=time.time(), timestep=timestep, observation=raw_obs, must_go=True
        )
        observation_iterator = send_bytes_in_chunks(
            pickle.dumps(timed_obs), services_pb2.Observation, log_prefix="[vla] obs", silent=True
        )
        self.stub.SendObservations(observation_iterator)
        deadline = time.perf_counter() + self._poll_timeout
        while time.perf_counter() < deadline:
            chunk = self.stub.GetActions(services_pb2.Empty())
            if len(chunk.data) > 0:
                return pickle.loads(chunk.data)  # nosec
            time.sleep(0.005)
        return []

    def close(self):
        try:
            self.channel.close()
        except Exception:
            pass


class VlaPolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("vla_policy_node")

        # ── 파라미터 (ROS param > env > 기본값) ─────────────────────────────
        p = self.declare_parameter
        env_file = p("env_file", "/workspace/.env").value
        env_dir = p("env_dir", "/workspace/env").value
        _load_env(env_file, env_dir)

        def pe(name, env_key, default):
            v = p(name, "").value
            return v if v else os.getenv(env_key, default)

        self.server_address = pe("server_address", "POLICY_SERVER_ADDRESS", "127.0.0.1:8080")
        policy_type = pe("policy_type", "POLICY_TYPE", "smolvla")
        pretrained = (p("pretrained_name_or_path", "").value
                      or os.getenv("POLICY_REPO_ID")
                      or os.getenv("POLICY_BASE_MODEL_PATH")
                      or "lerobot/smolvla_base")
        actions_per_chunk = int(p("actions_per_chunk", 0).value or os.getenv("ACTIONS_PER_CHUNK") or 8)
        self.task_instruction = pe("task_instruction", "TASK",
                                   "pick up the cube and place it in the bowl")
        rename_raw = p("rename_map", "").value or os.getenv("RENAME_MAP", "")
        rename_map = json.loads(rename_raw) if rename_raw.strip() else {}
        policy_device = pe("policy_device", "POLICY_DEVICE", "cuda")
        self.chunk_size_threshold = float(
            p("chunk_size_threshold", 0.0).value or os.getenv("CHUNK_SIZE_THRESHOLD") or 0.5
        )
        poll_timeout = float(p("poll_timeout", 5.0).value)
        self.fps = int(p("fps", 30).value)
        self.joint_states_topic = p("joint_states_topic", "/isaac_joint_states").value
        self.joint_commands_topic = p("joint_commands_topic", "/isaac_joint_commands").value

        if "${" in (pretrained or ""):
            self.get_logger().warn(
                f"미해결 변수 보간 pretrained={pretrained!r}. HF_USER 설정 또는 "
                "ROS param pretrained_name_or_path 지정 필요."
            )

        # ── gRPC 세션 ───────────────────────────────────────────────────────
        self.session = PolicyServerSession(
            server_address=self.server_address, policy_type=policy_type, pretrained=pretrained,
            actions_per_chunk=actions_per_chunk, policy_device=policy_device,
            rename_map=rename_map, poll_timeout=poll_timeout,
        )
        self.actions_per_chunk = actions_per_chunk
        self._refill_floor = max(1, int(actions_per_chunk * self.chunk_size_threshold))

        # ── ROS I/O ─────────────────────────────────────────────────────────
        self.bridge = CvBridge()
        self._joint_rad: np.ndarray | None = None
        self._images: dict[str, np.ndarray] = {}
        self._queue: deque = deque()
        self._timestep = 0

        self.create_subscription(JointState, self.joint_states_topic, self._joint_cb, 10)
        for cam, topic in _CAMERA_TOPICS.items():
            self.create_subscription(Image, topic, self._make_image_cb(cam), 10)
        self.cmd_pub = self.create_publisher(JointState, self.joint_commands_topic, 10)

        self.timer = self.create_timer(1.0 / max(self.fps, 1), self._control_tick)
        self.get_logger().info(
            f"VLA node up. obs={self.joint_states_topic}+cameras → cmd={self.joint_commands_topic}, "
            f"task={self.task_instruction!r}, fps={self.fps}"
        )

    # ── 콜백 ────────────────────────────────────────────────────────────────
    def _joint_cb(self, msg: JointState) -> None:
        # name 기준 재정렬 → SO101_JOINT_ORDER (bridge=articulation 순, 실기기=알파벳순 모두 처리).
        idx = {n: i for i, n in enumerate(msg.name)}
        try:
            self._joint_rad = np.array(
                [msg.position[idx[j]] for j in SO101_JOINT_ORDER], dtype=np.float32
            )
        except (KeyError, IndexError):
            self.get_logger().warn(f"joint_states 이름 불일치: {list(msg.name)}", throttle_duration_sec=5.0)

    def _make_image_cb(self, cam: str):
        def _cb(msg: Image) -> None:
            try:
                self._images[cam] = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"{cam} image decode 실패: {exc}", throttle_duration_sec=5.0)
        return _cb

    # ── 제어 루프 ─────────────────────────────────────────────────────────
    def _ready(self) -> bool:
        return self._joint_rad is not None and all(c in self._images for c in CAMERA_KEYS)

    def _build_raw_obs(self) -> dict:
        state_lerobot = to_lerobot_units(self._joint_rad)
        obs: dict = {name: float(state_lerobot[i]) for i, name in enumerate(JOINT_FEATURE_NAMES)}
        for cam in CAMERA_KEYS:
            obs[cam] = self._images[cam]
        obs["task"] = self.task_instruction
        return obs

    def _control_tick(self) -> None:
        if not self._ready():
            return
        if len(self._queue) <= self._refill_floor:
            timed_actions = self.session.predict_chunk(self._build_raw_obs(), self._timestep)
            if not timed_actions:
                self.get_logger().warn("policy-server 빈 chunk (timeout)", throttle_duration_sec=5.0)
                return
            merged = {ts: act for ts, act in self._queue}
            for ta in timed_actions:
                ts = int(ta.get_timestep())
                if ts < self._timestep:
                    continue
                merged[ts] = ta.get_action().detach().cpu().numpy().astype(np.float32)
            self._queue = deque(sorted(merged.items(), key=lambda kv: kv[0]))

        if not self._queue:
            return
        ts, action_lerobot = self._queue.popleft()
        self._timestep = ts + 1
        target_rad = clamp_joint_rad(from_lerobot_units(action_lerobot))

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(SO101_JOINT_ORDER)
        msg.position = [float(v) for v in target_rad]
        self.cmd_pub.publish(msg)

    def destroy_node(self) -> bool:
        self.session.close()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = VlaPolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
