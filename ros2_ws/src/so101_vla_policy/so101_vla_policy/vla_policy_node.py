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

# lerobot.* 는 vendored mini-lerobot(pickle 호환 shim) — 실 lerobot 대신(import 체인이
# transformers/datasets 까지 끌어와 컨테이너 부적합). 경로는 entrypoint 가 PYTHONPATH 로 추가
# (/workspace/ros2_ws/src/so101_vla_policy/vendor). 미설정 환경 대비 __file__ 기준도 보강.
import sys as _sys
from pathlib import Path as _Path

_vendor = _Path(__file__).resolve().parents[1] / "vendor"
if _vendor.is_dir() and str(_vendor) not in _sys.path:
    _sys.path.insert(0, str(_vendor))

from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation  # noqa: E402
from lerobot.transport import services_pb2, services_pb2_grpc  # noqa: E402
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks  # noqa: E402

from so101_vla_policy.units import (  # noqa: E402
    CAMERA_KEYS,
    JOINT_FEATURE_NAMES,
    LEROBOT_FEATURES,
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
                 policy_device, lerobot_features, poll_timeout):
        self._poll_timeout = float(poll_timeout)

        # gRPC roundtrip 프로파일 윈도. N콜마다 total/obs_send/recv_wait min·mean·max 출력.
        self._rt_window: list[tuple[float, float]] = []  # (total_ms, obs_send_ms)
        self._rt_report_every = 10

        # ⚠ rename 은 **클라가 직접 적용**(features·obs 키를 policy 키로) → server rename_map 은 비움.
        # 이유: 0.5.1 server 는 raw_observation_to_observation 의 resize 에서
        # policy.config.image_features[key] 를 lerobot_features 키로 조회하는데, 이 단계는
        # preprocessor rename(rename_map) **이전**이다. 따라서 lerobot_features 이미지 키가
        # 모델 config 키(SmolVLA=camera1/2/3)와 일치해야 KeyError 가 안 난다.
        self.policy_config = RemotePolicyConfig(
            policy_type, pretrained, lerobot_features, int(actions_per_chunk), policy_device, {},
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
              f"chunk={actions_per_chunk}, image_keys={[k for k in lerobot_features if 'images' in k]})",
              flush=True)

    def predict_chunk(self, raw_obs: dict, timestep: int) -> list:
        _t0 = time.perf_counter()
        timed_obs = TimedObservation(
            timestamp=time.time(), timestep=timestep, observation=raw_obs, must_go=True
        )
        observation_iterator = send_bytes_in_chunks(
            pickle.dumps(timed_obs), services_pb2.Observation, log_prefix="[vla] obs", silent=True
        )
        self.stub.SendObservations(observation_iterator)
        _t_send = time.perf_counter()  # obs 직렬화+업로드(SendObservations 블로킹) 종료점
        deadline = time.perf_counter() + self._poll_timeout
        while time.perf_counter() < deadline:
            chunk = self.stub.GetActions(services_pb2.Empty())
            if len(chunk.data) > 0:
                self._record_roundtrip(_t0, _t_send)
                return pickle.loads(chunk.data)  # nosec
            time.sleep(0.005)
        return []

    def _record_roundtrip(self, t0: float, t_send: float) -> None:
        """gRPC 왕복 분해 기록. total = obs_send + recv_wait.
        recv_wait = 서버 추론 + action chunk 다운로드 + 폴 granularity(5ms 단위)."""
        now = time.perf_counter()
        self._rt_window.append(((now - t0) * 1e3, (t_send - t0) * 1e3))
        if len(self._rt_window) < self._rt_report_every:
            return
        tot = [a for a, _ in self._rt_window]
        snd = [b for _, b in self._rt_window]
        recv = [a - b for a, b in self._rt_window]
        n = len(tot)
        print(
            f"[vla] gRPC roundtrip ms (n={n}): "
            f"total min/mean/max={min(tot):.1f}/{sum(tot) / n:.1f}/{max(tot):.1f} · "
            f"obs_send mean={sum(snd) / n:.1f} · "
            f"recv_wait(infer+dl+poll) mean={sum(recv) / n:.1f}",
            flush=True,
        )
        self._rt_window.clear()

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
        # 그리퍼 command offset(rad) — sim cuRobo 데이터셋은 action 을 use_default_offset 의
        # pre-offset 값(grip_target - 0.20)으로 기록한다. 학습 env 의 action term 이 +0.20 을
        # 재적용하므로, sim bridge 추론 시 동일하게 그리퍼 target 에 더해줘야 한다(arm offset=0).
        # 실기기(절대각 기록)는 0. sim 추론은 GRIPPER_CMD_OFFSET=0.2.
        self.gripper_cmd_offset = float(
            p("gripper_cmd_offset", 0.0).value or os.getenv("GRIPPER_CMD_OFFSET") or 0.0
        )
        self._gripper_idx = list(SO101_JOINT_ORDER).index("gripper")

        if "${" in (pretrained or ""):
            self.get_logger().warn(
                f"미해결 변수 보간 pretrained={pretrained!r}. HF_USER 설정 또는 "
                "ROS param pretrained_name_or_path 지정 필요."
            )

        # rename 을 클라에서 적용: features 이미지 키 + obs 이미지 키를 policy 키로 만든다.
        # rename_map(dataset feat key → policy feat key), 예 observation.images.top→...camera1.
        # self._cam_obs_key[cam] = obs dict 에 넣을 bare 키(camera1 또는 top).
        lerobot_features = {"observation.state": dict(LEROBOT_FEATURES["observation.state"])}
        self._cam_obs_key: dict[str, str] = {}
        for cam in CAMERA_KEYS:
            ds_key = f"observation.images.{cam}"
            pol_key = rename_map.get(ds_key, ds_key)
            lerobot_features[pol_key] = {
                "dtype": "image", "shape": (480, 640, 3), "names": ["height", "width", "channels"],
            }
            self._cam_obs_key[cam] = pol_key.split("observation.images.")[-1]

        # ── gRPC 세션 ───────────────────────────────────────────────────────
        self.session = PolicyServerSession(
            server_address=self.server_address, policy_type=policy_type, pretrained=pretrained,
            actions_per_chunk=actions_per_chunk, policy_device=policy_device,
            lerobot_features=lerobot_features, poll_timeout=poll_timeout,
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
            # bare obs 키 = policy 이미지 키(camera1/2/3 또는 top/wrist/front) — features 와 정합.
            obs[self._cam_obs_key[cam]] = self._images[cam]
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
        raw_rad = from_lerobot_units(action_lerobot)
        # use_default_offset 재적용(그리퍼만; arm offset=0). sim 데이터의 pre-offset action → 절대 target.
        raw_rad[self._gripper_idx] += self.gripper_cmd_offset
        target_rad = clamp_joint_rad(raw_rad)

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
