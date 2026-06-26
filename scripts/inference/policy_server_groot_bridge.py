#!/usr/bin/env python
"""
scripts/policy_server_groot_bridge.py
=====================================
GR00T-N1.7 gRPC↔ZMQ bridge — LeRobot async-inference 컨트랙트를 그대로 유지한 채
추론만 NVIDIA Isaac-GR00T 네이티브 ZMQ 서버(run_gr00t_server.py, Gr00tPolicy)에
위임하는 PolicyServer 서브클래스.

────────────────────────────────────────────────────────
왜 bridge 인가
────────────────────────────────────────────────────────
GR00T-N1.7 은 transformers 4.57 / py3.10 을 요구해 policy-server:0.5.1 이미지
(transformers 5.3 / py3.12)와 한 venv 에 공존할 수 없다 → 별도 gr00t 이미지에서
ZMQ 서버로 돌린다(:5555). 반면 sim 폐루프(vla_policy_node)는 LeRobot gRPC
(AsyncInferenceStub, :8080)로만 말한다. 이 bridge 가 그 사이를 잇는다:

  vla_policy_node ──gRPC :8080──▶ [이 bridge] ──ZMQ :5555──▶ gr00t 이미지(Gr00tPolicy)

lerobot 의 PolicyServer 를 서브클래싱하므로 gRPC 핸드셰이크·관측 스트리밍·
청크 직렬화는 전부 재사용하고,
2개 지점만 오버라이드한다:
  1. SendPolicyInstructions — lerobot 정책 로드를 건너뛰고(GR00T 체크포인트는
     lerobot 포맷 아님) GR00T ZMQ 클라이언트를 연결한다.
  2. _predict_action_chunk — raw lerobot obs 를 GR00T modality dict 로 변환해
     ZMQ get_action 으로 추론하고, 결과를 LeRobot 단위 action chunk(TimedAction)
     로 되돌린다. lerobot 전/후처리(정규화·토크나이즈)는 건너뛴다(GR00T 가 내부에서
     자체 정규화/Eagle 토크나이즈 수행).

gr00t 패키지는 import 하지 않는다(transformers 충돌) — ZMQ wire 포맷
(msgpack + ndarray=np.save)만 아래에 최소 vendoring 한다.

────────────────────────────────────────────────────────
obs / action 변환 (vla_policy_node._build_raw_obs ↔ GR00T modality)
────────────────────────────────────────────────────────
raw obs(노드 송신)               → GR00T get_action 입력
  shoulder_pan.pos..wrist_roll.pos(deg)  → state.single_arm (1,1,5) f32
  gripper.pos([0,100])                   → state.gripper    (1,1,1) f32
  top/wrist/front(uint8 HWC 480x640x3)   → video.{top,wrist,front} (1,1,H,W,3) u8
  task(str)                              → language["annotation.human.task_description"]=[[task]]

GR00T action(반환)              → LeRobot action chunk
  single_arm (1,16,5) + gripper (1,16,1) f32(절대, 데이터셋 네이티브 = LeRobot 단위)
  → concat (16,6) → TimedAction 리스트(노드가 from_lerobot_units 로 rad 변환)

RTC 미적용(GR00T 는 init_rtc_processor 없음). 청크는 평문 서빙.

────────────────────────────────────────────────────────
실행 (policy-entrypoint.sh policy-server-groot 모드)
────────────────────────────────────────────────────────
  python /workspace/scripts/policy_server_groot_bridge.py \\
      --host 0.0.0.0 --port 8080 --fps 30 \\
      --groot_zmq_host 127.0.0.1 --groot_zmq_port 5555
"""

import argparse
import io
import logging
import pickle  # nosec — lerobot async gRPC 컨트랙트(서버/클라 모두 신뢰 호스트)
import time
from concurrent.futures import ThreadPoolExecutor
from pprint import pformat

import grpc
import msgpack
import numpy as np
import torch
import zmq

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.transport import services_pb2, services_pb2_grpc

logger = logging.getLogger(__name__)

# ── raw obs 키 (vla_policy_node.units.JOINT_FEATURE_NAMES / CAMERA_KEYS 와 동일) ──
_ARM_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
]
_GRIPPER_KEY = "gripper.pos"
# GR00T video modality key → raw obs 의 bare 이미지 키 (RENAME_MAP 비움 → 동일 이름)
_VIDEO_KEYS = ("front", "wrist", "top")
_DEFAULT_TASK = "pick up the cube and place it in the bowl"


# ─────────────────────────────────────────────────────────────────────────────
# GR00T ZMQ wire (gr00t.policy.server_client.MsgSerializer 의 ndarray 훅만 vendoring)
# ─────────────────────────────────────────────────────────────────────────────
def _encode(obj):
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    return obj


def _decode(obj):
    if isinstance(obj, dict) and "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


def _pack(data) -> bytes:
    return msgpack.packb(data, default=_encode)


def _unpack(data: bytes):
    return msgpack.unpackb(data, object_hook=_decode)


class GrootZmqClient:
    """GR00T 네이티브 PolicyServer(ZMQ REP) 로 향하는 최소 REQ 클라이언트."""

    def __init__(self, host: str, port: int, timeout_ms: int = 60000) -> None:
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self._ctx = zmq.Context.instance()
        self._connect()

    def _connect(self) -> None:
        self.sock = self._ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(f"tcp://{self.host}:{self.port}")

    def _reconnect(self) -> None:
        try:
            self.sock.close(0)
        except Exception:  # noqa: BLE001
            pass
        self._connect()

    def _call(self, endpoint: str, data: dict | None = None, requires_input: bool = True):
        req: dict = {"endpoint": endpoint}
        if requires_input:
            req["data"] = data
        try:
            self.sock.send(_pack(req))
            msg = self.sock.recv()
        except zmq.error.Again:
            # 타임아웃 → REQ 소켓이 잠긴 상태. 재생성 후 재던지기.
            self._reconnect()
            raise
        resp = _unpack(msg)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"GR00T server error: {resp['error']}")
        return resp

    def ping(self) -> bool:
        try:
            self._call("ping", requires_input=False)
            return True
        except Exception:  # noqa: BLE001
            self._reconnect()
            return False

    def get_action(self, observation: dict, options: dict | None = None):
        resp = self._call("get_action", {"observation": observation, "options": options})
        # gr00t get_action 은 (action, info) 튜플 반환 → msgpack 은 list 로 직렬화.
        action, info = resp[0], resp[1]
        return action, info

    def reset(self, options: dict | None = None):
        return self._call("reset", {"options": options})


# ─────────────────────────────────────────────────────────────────────────────
# Bridge PolicyServer
# ─────────────────────────────────────────────────────────────────────────────
class GrootBridgeServer(PolicyServer):
    """추론을 GR00T ZMQ 서버에 위임하는 PolicyServer."""

    def __init__(
        self,
        config: PolicyServerConfig,
        groot_host: str,
        groot_port: int,
        zmq_timeout_ms: int,
    ) -> None:
        super().__init__(config)
        self._groot_host = groot_host
        self._groot_port = groot_port
        self._zmq_timeout_ms = zmq_timeout_ms
        self._groot: GrootZmqClient | None = None
        self._chunk_count = 0
        self._default_task = _DEFAULT_TASK

    # ── 서버 리셋 시 GR00T 세션도 리셋 ──────────────────────────────────────────
    def _reset_server(self) -> None:
        super()._reset_server()
        self._chunk_count = 0
        if self._groot is not None:
            try:
                self._groot.reset()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[GR00T] reset 실패(무시): {exc}")

    # ── GR00T ZMQ 연결 (서버 부팅 중일 수 있어 ping 재시도) ──────────────────────
    def _connect_groot(self) -> None:
        self._groot = GrootZmqClient(self._groot_host, self._groot_port, self._zmq_timeout_ms)
        addr = f"{self._groot_host}:{self._groot_port}"
        for attempt in range(30):  # 모델 로드 대기 (최대 ~60s)
            if self._groot.ping():
                logger.info(f"[GR00T] ZMQ 서버 연결 확인 → {addr}")
                return
            logger.info(f"[GR00T] ZMQ 서버 대기 중… ({attempt + 1}/30) {addr}")
            time.sleep(2.0)
        logger.warning(
            f"[GR00T] ZMQ ping 미응답 {addr} — 첫 get_action 에서 블로킹/타임아웃될 수 있음. "
            "gr00t 컨테이너(zmq-server) 기동 여부 확인."
        )

    # ── lerobot 정책 로드 생략 + GR00T 연결 ──────────────────────────────────────
    def SendPolicyInstructions(self, request, context):  # noqa: N802
        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        specs = pickle.loads(request.data)  # nosec
        # RemotePolicyConfig 의 메타만 보관(정책 로드는 안 함). policy_type 은 무시한다
        # (lerobot SUPPORTED_POLICIES 에 groot_n17 미등록 — 그래서 super 호출 금지).
        self.device = getattr(specs, "device", "cuda")
        self.policy_type = getattr(specs, "policy_type", "groot_n17")
        self.lerobot_features = getattr(specs, "lerobot_features", {})
        self.actions_per_chunk = int(getattr(specs, "actions_per_chunk", 16))

        logger.info(
            f"[GR00T] instructions 수신 | type={self.policy_type} | "
            f"actions_per_chunk={self.actions_per_chunk} | "
            f"zmq={self._groot_host}:{self._groot_port}"
        )
        self._connect_groot()
        return services_pb2.Empty()

    # ── obs/action 변환 ──────────────────────────────────────────────────────────
    def _lerobot_obs_to_groot(self, raw: dict) -> dict:
        """raw lerobot obs(flat .pos + bare 이미지 + task) → GR00T modality dict."""
        arm = np.asarray([raw[k] for k in _ARM_KEYS], dtype=np.float32).reshape(1, 1, 5)
        grip = np.asarray([raw[_GRIPPER_KEY]], dtype=np.float32).reshape(1, 1, 1)

        video: dict[str, np.ndarray] = {}
        for key in _VIDEO_KEYS:
            if key not in raw:
                raise KeyError(
                    f"raw obs 에 이미지 키 '{key}' 없음. groot 프로필은 RENAME_MAP 을 비워 "
                    f"top/wrist/front bare 키를 기대한다. 받은 키: {sorted(raw.keys())}"
                )
            img = np.asarray(raw[key])
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            video[key] = img.reshape(1, 1, *img.shape)  # (B=1, T=1, H, W, C)

        task = raw.get("task") or self._default_task
        return {
            "video": video,
            "state": {"single_arm": arm, "gripper": grip},
            "language": {"annotation.human.task_description": [[str(task)]]},
        }

    @staticmethod
    def _groot_action_to_chunk(action: dict) -> np.ndarray:
        """GR00T action dict(single_arm + gripper) → (T, 6) f32 LeRobot 단위."""
        arm = np.asarray(action["single_arm"], dtype=np.float32)
        grip = np.asarray(action["gripper"], dtype=np.float32)
        if arm.ndim == 3:  # (B, T, 5) → (T, 5)
            arm = arm[0]
        if grip.ndim == 3:  # (B, T, 1) → (T, 1)
            grip = grip[0]
        if grip.ndim == 1:  # (T,) → (T, 1)
            grip = grip.reshape(-1, 1)
        return np.concatenate([arm, grip], axis=-1).astype(np.float32)  # (T, 6)

    # ── 추론: lerobot 전/후처리 우회, GR00T ZMQ 위임 ─────────────────────────────
    def _predict_action_chunk(self, observation_t):
        if self._groot is None:
            self._connect_groot()

        raw = observation_t.get_observation()
        groot_obs = self._lerobot_obs_to_groot(raw)

        t0 = time.perf_counter()
        action_dict, _info = self._groot.get_action(groot_obs)
        chunk = self._groot_action_to_chunk(action_dict)  # (T, 6)
        if self.actions_per_chunk:
            chunk = chunk[: self.actions_per_chunk]

        self._chunk_count += 1
        self.last_processed_obs = observation_t  # base 호환(sanity-check 참조 대비)
        logger.info(
            f"[GR00T] chunk #{self._chunk_count} | infer {(time.perf_counter() - t0) * 1e3:.0f}ms | "
            f"shape {tuple(chunk.shape)}"
        )

        chunk_t = torch.from_numpy(chunk)  # (T, 6)
        return self._time_action_chunk(
            observation_t.get_timestamp(), list(chunk_t), observation_t.get_timestep()
        )


def serve_groot(cfg: PolicyServerConfig, groot_host: str, groot_port: int, zmq_timeout_ms: int) -> None:
    logging.info(pformat({"server_config": cfg.__dict__, "groot_zmq": f"{groot_host}:{groot_port}"}))

    server_instance = GrootBridgeServer(cfg, groot_host, groot_port, zmq_timeout_ms)

    server = grpc.server(ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(server_instance, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    server_instance.logger.info(
        f"GrootBridgeServer (lerobot {_lerobot_ver()}) 기동 | gRPC {cfg.host}:{cfg.port} | "
        f"fps={cfg.fps} | GR00T ZMQ tcp://{groot_host}:{groot_port}"
    )
    server.start()
    server.wait_for_termination()
    server_instance.logger.info("Server terminated")


def _lerobot_ver() -> str:
    try:
        import importlib.metadata

        return importlib.metadata.version("lerobot")
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="GR00T-N1.7 gRPC↔ZMQ bridge policy server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = parser.add_argument_group("gRPC 서버 설정 (vla_policy_node 와 동일 컨트랙트)")
    g.add_argument("--host", default="0.0.0.0", help="gRPC bind 주소")
    g.add_argument("--port", type=int, default=8080, help="gRPC 포트")
    g.add_argument("--fps", type=int, default=30, help="제어 루프 FPS")
    g.add_argument("--inference_latency", type=float, default=0.033, help="목표 추론 레이턴시(초)")
    g.add_argument("--obs_queue_timeout", type=float, default=2.0, help="관측 큐 타임아웃(초)")

    r = parser.add_argument_group("GR00T ZMQ 백엔드")
    r.add_argument("--groot_zmq_host", default="127.0.0.1", help="gr00t 컨테이너 ZMQ 호스트")
    r.add_argument("--groot_zmq_port", type=int, default=5555, help="gr00t 컨테이너 ZMQ 포트")
    r.add_argument(
        "--groot_zmq_timeout_ms", type=int, default=60000,
        help="ZMQ recv/send 타임아웃(ms). 3B 추론 + 첫 호출 모델 워밍업 고려해 넉넉히.",
    )
    # actions_per_chunk 는 RemotePolicyConfig 에서 받으므로 CLI 인자는 받되 무시(로깅 호환).
    parser.add_argument("--actions_per_chunk", type=int, default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()

    server_config = PolicyServerConfig(
        host=args.host,
        port=args.port,
        fps=args.fps,
        inference_latency=args.inference_latency,
        obs_queue_timeout=args.obs_queue_timeout,
    )
    serve_groot(server_config, args.groot_zmq_host, args.groot_zmq_port, args.groot_zmq_timeout_ms)


if __name__ == "__main__":
    main()
