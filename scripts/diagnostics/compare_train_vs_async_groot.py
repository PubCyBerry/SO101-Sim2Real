"""GR00T-N1.7 — 학습(데이터+modality) 구조 입력 vs async sim 추론 파이프라인 비교.

`compare_train_vs_rtc.py`(SmolVLA)의 GR00T 판. 같은 에피소드의 recorded obs 를 두 경로로
통과시켜 정책 출력을 recorded(expert) 위에 overlay 하고, **모델 입력 timestamp 마다 마커**를 찍는다.

  Path A (training-structure / teacher-forced):
      gr00t ZMQ 서버(:5555)에 **직접** get_action. recorded frame 의 state+3cam 을 GR00T modality
      dict(single_arm/gripper/video/language) 로 변환(bridge `_lerobot_obs_to_groot` 와 동일)해
      n_action_steps(=feed_every) 마다 1번 입력, 16-step chunk 를 그대로 소비. = 모델이 학습 때
      최적화된 사용방식(드리프트 없는 이상적 소비).

  Path B (async sim deploy pipeline):
      gRPC async 서버(policy-server `GrootBridgeServer`, :8080) → bridge 가 obs 변환 → ZMQ → gr00t.
      소비는 `vla_policy_node` 로직(queue·refill·weighted merge, actions_per_chunk=16). = sim 폐루프
      배포 경로(ROS latency·sim-time desync 만 오프라인 재현 불가).

■ joint space 표현 / 보정 (사용자 질문):
    GR00T `single_arm` modality 는 **RELATIVE**(현재 state 기준 delta)지만, `Gr00tPolicy.get_action`
    내부 `decode_action(normalized_action, tag, batched_states)` 가 **입력 state 를 reference 로
    절대값 복원**(unnormalize + relative→absolute)한 뒤 반환한다. 따라서 ZMQ/gRPC 로 받는 action 은
    **이미 절대 LeRobot 단위(arm deg, gripper [0,100])** → recorded(절대)와 **직접 overlay, 별도 보정 불요**.
    teacher-forced(Path A)·async(Path B) 모두 같은 recorded state 를 reference 로 쓰므로 공정.
    (gripper +0.20 deploy offset 은 joint-space 적용 단계 것이라 action-vs-recorded overlay 엔 미적용.)

■ 사전 조건: gr00t zmq-server(:5555) + policy-server-groot bridge(:8080) 둘 다 같은 체크포인트로 기동.
  policy-server 이미지 컨테이너에서 실행(lerobot dataset + grpc + zmq + msgpack):
    docker compose --env-file .env -f docker/docker-compose.yaml run --rm --no-deps \
      --entrypoint bash policy-server -lc \
      'python /workspace/scripts/diagnostics/compare_train_vs_async_groot.py --episode -1 \
         --dataset taehunkim/so101_sim_pick_cube_smooth'
"""

from __future__ import annotations

import argparse
import io
import pickle  # nosec
import random
import time
from collections import deque

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import msgpack  # noqa: E402
import zmq  # noqa: E402
import grpc  # noqa: E402
from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation  # noqa: E402
from lerobot.transport import services_pb2, services_pb2_grpc  # noqa: E402
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM_KEYS = [f"{j}.pos" for j in JOINT_NAMES[:5]]
GRIPPER_KEY = "gripper.pos"
# 데이터셋 이미지 키(bare suffix) — GR00T modality video 키도 동일 이름(top/wrist/front).
CAM_SUFFIX = ["top", "wrist", "front"]
TASK = "pick up the cube and place it in the bowl"


# ── GR00T ZMQ wire (bridge _encode/_pack 와 동일) ────────────────────────────────
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


class GrootZmqClient:
    def __init__(self, host: str, port: int, timeout_ms: int = 60000) -> None:
        self._ctx = zmq.Context.instance()
        self.sock = self._ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(f"tcp://{host}:{port}")

    def _call(self, endpoint: str, data: dict | None = None, requires_input: bool = True):
        req: dict = {"endpoint": endpoint}
        if requires_input:
            req["data"] = data
        self.sock.send(msgpack.packb(req, default=_encode))
        resp = msgpack.unpackb(self.sock.recv(), object_hook=_decode)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"GR00T server error: {resp['error']}")
        return resp

    def get_action(self, observation: dict, options: dict | None = None):
        resp = self._call("get_action", {"observation": observation, "options": options})
        return resp[0], resp[1]  # (action_dict, info)

    def reset(self):
        try:
            self._call("reset", {"options": None})
        except Exception:  # noqa: BLE001
            pass


def _img_chw_to_u8_hwc(img) -> np.ndarray:
    """dataset frame 이미지(CHW float[0,1] torch) → HWC uint8."""
    arr = img.permute(1, 2, 0).numpy() if hasattr(img, "permute") else np.asarray(img)
    if arr.dtype != np.uint8:
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def _frame_to_groot_modality(frame) -> dict:
    """recorded frame → GR00T modality dict (bridge _lerobot_obs_to_groot 와 동일 형식)."""
    st = np.asarray(frame["observation.state"], dtype=np.float32)
    arm = st[:5].reshape(1, 1, 5)
    grip = st[5:6].reshape(1, 1, 1)
    video = {}
    for suf in CAM_SUFFIX:
        u8 = _img_chw_to_u8_hwc(frame[f"observation.images.{suf}"])
        video[suf] = u8.reshape(1, 1, *u8.shape)  # (B=1,T=1,H,W,C)
    return {
        "video": video,
        "state": {"single_arm": arm, "gripper": grip},
        "language": {"annotation.human.task_description": [[TASK]]},
    }


def _groot_action_to_chunk(action: dict) -> np.ndarray:
    arm = np.asarray(action["single_arm"], dtype=np.float32)
    grip = np.asarray(action["gripper"], dtype=np.float32)
    if arm.ndim == 3:
        arm = arm[0]
    if grip.ndim == 3:
        grip = grip[0]
    if grip.ndim == 1:
        grip = grip.reshape(-1, 1)
    return np.concatenate([arm, grip], axis=-1).astype(np.float32)  # (T,6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="(gr00t server checkpoint)", help="제목용 라벨")
    ap.add_argument("--dataset", default="taehunkim/so101_sim_pick_cube_smooth")
    ap.add_argument("--episode", type=int, default=-1, help="-1=무작위")
    ap.add_argument("--groot_zmq_host", default="127.0.0.1")
    ap.add_argument("--groot_zmq_port", type=int, default=5555)
    ap.add_argument("--server", default="127.0.0.1:8080", help="bridge gRPC")
    ap.add_argument("--actions_per_chunk", type=int, default=16)
    ap.add_argument("--feed_every", type=int, default=16, help="Path A teacher-forced 입력 주기(=n_action_steps)")
    ap.add_argument("--chunk_threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/workspace/outputs/compare_groot")
    args = ap.parse_args()

    # ── 에피소드 선택(무작위) ──
    ds = LeRobotDataset(args.dataset)
    ei = np.asarray(ds.hf_dataset["episode_index"])
    all_eps = sorted(set(int(e) for e in ei))
    if args.episode < 0:
        random.seed(args.seed)
        episode = random.choice(all_eps)
    else:
        episode = args.episode
    w = np.where(ei == episode)[0]
    f0, f1 = int(w[0]), int(w[-1]) + 1
    n = f1 - f0
    print(f"[gcmp] dataset={args.dataset} episode={episode} (전체 {len(all_eps)}개 중) frames={n}", flush=True)
    frames = [ds[f0 + t] for t in range(n)]
    recorded = np.stack([np.asarray(fr["action"], dtype=np.float32) for fr in frames])

    # ── Path A: gr00t ZMQ 직접 get_action, teacher-forced ──
    gz = GrootZmqClient(args.groot_zmq_host, args.groot_zmq_port)
    gz.reset()
    predA = []
    feedsA: list[int] = []
    chunk = None
    cstart = 0
    for t in range(n):
        if chunk is None or (t - cstart) >= args.feed_every:
            action, _ = gz.get_action(_frame_to_groot_modality(frames[t]))
            chunk = _groot_action_to_chunk(action)  # (16,6) 절대
            cstart = t
            feedsA.append(t)
        predA.append(chunk[min(t - cstart, chunk.shape[0] - 1)])
    predA = np.stack(predA)
    print(f"[gcmp] Path A (gr00t ZMQ teacher-forced, feed_every={args.feed_every}) done, {len(feedsA)} feeds", flush=True)

    # ── Path B: async gRPC bridge, vla_policy_node 소비로직 ──
    lerobot_features = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": list(f"{j}.pos" for j in JOINT_NAMES)},
    }
    for suf in CAM_SUFFIX:
        lerobot_features[f"observation.images.{suf}"] = {
            "dtype": "image", "shape": (480, 640, 3), "names": ["height", "width", "channels"]}
    pcfg = RemotePolicyConfig("groot_n17", args.model, lerobot_features, args.actions_per_chunk, "cuda", {})
    channel = grpc.insecure_channel(args.server, grpc_channel_options(initial_backoff="0.0333s"))
    stub = services_pb2_grpc.AsyncInferenceStub(channel)
    stub.Ready(services_pb2.Empty())
    stub.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(pcfg)))  # nosec
    print(f"[gcmp] bridge gRPC 연결({args.server}), instructions 전송", flush=True)

    def predict_chunk(raw_obs, timestep):
        to = TimedObservation(timestamp=time.time(), timestep=timestep, observation=raw_obs, must_go=True)
        it = send_bytes_in_chunks(pickle.dumps(to), services_pb2.Observation, log_prefix="", silent=True)  # nosec
        stub.SendObservations(it)
        deadline = time.perf_counter() + 10.0
        while time.perf_counter() < deadline:
            ch = stub.GetActions(services_pb2.Empty())
            if len(ch.data) > 0:
                return pickle.loads(ch.data)  # nosec
            time.sleep(0.005)
        return []

    def build_obs_B(fr):
        st = np.asarray(fr["observation.state"], dtype=np.float32)
        obs = {f"{j}.pos": float(st[i]) for i, j in enumerate(JOINT_NAMES)}
        for suf in CAM_SUFFIX:  # bare 키(top/wrist/front) — bridge _lerobot_obs_to_groot 가 기대
            obs[suf] = _img_chw_to_u8_hwc(fr[f"observation.images.{suf}"])
        obs["task"] = TASK
        return obs

    refill = max(1, int(args.actions_per_chunk * args.chunk_threshold))
    queue: deque = deque()
    timestep = 0
    predB = []
    feedsB: list[int] = []
    for t in range(n):
        if len(queue) <= refill:
            tas = predict_chunk(build_obs_B(frames[t]), timestep)
            feedsB.append(t)
            merged = {ts: a for ts, a in queue}
            for ta in tas:
                ts = int(ta.get_timestep())
                if ts < timestep:
                    continue
                merged[ts] = ta.get_action().detach().cpu().numpy().astype(np.float32)
            queue = deque(sorted(merged.items(), key=lambda kv: kv[0]))
        if not queue:
            predB.append(predB[-1] if predB else np.zeros(6, np.float32))
            continue
        ts, act = queue.popleft()
        timestep = ts + 1
        predB.append(act)
    predB = np.stack(predB)
    print(f"[gcmp] Path B (async bridge, apc={args.actions_per_chunk} refill={refill}) done, {len(feedsB)} feeds", flush=True)

    # ── metrics ──
    mae_A = np.abs(predA - recorded).mean(axis=0)
    mae_B = np.abs(predB - recorded).mean(axis=0)
    mae_AB = np.abs(predA - predB).mean(axis=0)
    print("[gcmp] ===== per-joint MAE (LeRobot 단위) =====", flush=True)
    for j, nm in enumerate(JOINT_NAMES):
        u = "[0,100]" if nm == "gripper" else "deg"
        print(f"[gcmp]   {nm:13s} A-rec={mae_A[j]:6.2f}  B-rec={mae_B[j]:6.2f}  A-B={mae_AB[j]:6.2f}  {u}", flush=True)
    print(f"[gcmp] overall: A-rec={mae_A.mean():.3f}  B-rec={mae_B.mean():.3f}  A-B(async효과)={mae_AB.mean():.3f}", flush=True)

    # ── overlay ──
    fig, axes = plt.subplots(3, 2, figsize=(16, 11))
    tt = np.arange(n)
    for j, nm in enumerate(JOINT_NAMES):
        ax = axes[j // 2][j % 2]
        ax.plot(tt, recorded[:, j], label="recorded (expert)", lw=2.2, color="tab:blue")
        ax.plot(tt, predA[:, j], label="A: training (gr00t ZMQ teacher-forced)", lw=1.3, color="tab:green", alpha=0.85)
        ax.plot(tt, predB[:, j], label="B: async sim (gRPC bridge)", lw=1.3, color="tab:red", alpha=0.85)
        # 모델 입력 timestamp 마커
        for fa in feedsA:
            ax.axvline(fa, color="tab:green", ls=":", lw=0.7, alpha=0.45)
        for fb in feedsB:
            ax.axvline(fb, color="tab:red", ls=":", lw=0.5, alpha=0.30)
        ax.scatter(feedsA, recorded[feedsA, j], marker="v", s=28, color="tab:green", zorder=5, label="A feed" if j == 0 else None)
        ax.scatter(feedsB, recorded[feedsB, j], marker="x", s=24, color="tab:red", zorder=5, label="B feed" if j == 0 else None)
        u = "[0,100]" if nm == "gripper" else "deg"
        ax.set_title(f"{nm} ({u})  A-rec={mae_A[j]:.2f} B-rec={mae_B[j]:.2f} A-B={mae_AB[j]:.2f}")
        ax.set_xlabel("frame")
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=7)
    fig.suptitle(f"GR00T-N1.7 training-struct vs async sim — {args.model.split('/')[-1]} ep{episode} "
                 f"(▽green=A feed, ×red=B feed, apc={args.actions_per_chunk}, n={n})  "
                 f"[single_arm RELATIVE→absolute by decode_action, 보정불요]", fontsize=11)
    fig.tight_layout()
    png = f"{args.out}_ep{episode}.png"
    fig.savefig(png, dpi=110)
    np.savez(f"{args.out}_ep{episode}.npz", recorded=recorded, predA=predA, predB=predB,
             feedsA=np.array(feedsA), feedsB=np.array(feedsB), mae_A=mae_A, mae_B=mae_B, mae_AB=mae_AB,
             episode=episode)
    print(f"[gcmp] 🎬 PNG → {png}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
