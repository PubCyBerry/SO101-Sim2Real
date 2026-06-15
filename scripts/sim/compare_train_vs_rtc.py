"""학습 입력방식 vs async+RTC+ROS 추론방식 — 같은 에피소드 obs 로 비교.

목적: 단위/정규화(=정합 확인)가 아니라 **소비 방식 차이**(chunk staleness·RTC guidance·
actions_per_chunk)가 정책 출력을 얼마나 바꾸는지 격리 측정. 같은 recorded obs(teacher-forced,
sim drift 없음)를 두 경로에 통과시켜 overlay.

  Path A (training): in-process predict_action_chunk(매 n_action_steps obs 1개) + postprocess.
                     = SmolVLA 가 학습 때 최적화된 사용방식.
  Path B (deploy)  : RTC policy-server(gRPC) + vla_policy_node 소비로직(queue·refill·must_go).
                     async chunk + RTC guidance(execution_horizon=8) + actions_per_chunk=24.

ROS latency·sim-time desync 는 오프라인 재현 불가(라이브 전용) — 이건 eval 성공률이 net 측정.
이 스크립트는 chunk+RTC 효과만 격리한다.

policy-server(RTC) 가 :8080 에 떠 있어야 함. policy-server 이미지 컨테이너에서 실행:
  docker compose --env-file .env -f docker/docker-compose.yaml run --rm --no-deps \
    --entrypoint bash policy-server -lc \
    'python /workspace/scripts/sim/compare_train_vs_rtc.py --episode 0'
"""

from __future__ import annotations

import argparse
import pickle  # nosec
import time
from collections import deque

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import grpc  # noqa: E402
from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation  # noqa: E402
from lerobot.transport import services_pb2, services_pb2_grpc  # noqa: E402
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
JOINT_FEATURE_NAMES = [f"{j}.pos" for j in JOINT_NAMES]
# dataset 키 → 정책 키(RENAME_MAP, smolvla.env 와 동일 순서)
CAM_DS = ["observation.images.top", "observation.images.wrist", "observation.images.front"]
CAM_POL = ["camera1", "camera2", "camera3"]
TASK = "pick up the cube and place it in the bowl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="taehunkim/so101_smolvla_sim_pick_cube")
    ap.add_argument("--dataset", default="taehunkim/so101_sim_pick_cube")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--server", default="127.0.0.1:8080")
    ap.add_argument("--actions_per_chunk", type=int, default=24)
    ap.add_argument("--chunk_threshold", type=float, default=0.5)
    ap.add_argument("--out", default="/workspace/outputs/compare_train_vs_rtc")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ── 데이터셋 1 에피소드 ──
    ds = LeRobotDataset(args.dataset)
    ei = np.asarray(ds.hf_dataset["episode_index"])
    w = np.where(ei == args.episode)[0]
    f0, f1 = int(w[0]), int(w[-1]) + 1
    n = f1 - f0
    print(f"[cmp] episode {args.episode}: {n} frames", flush=True)
    frames = [ds[f0 + t] for t in range(n)]
    recorded = np.stack([np.asarray(fr["action"], dtype=np.float32) for fr in frames])

    # ── Path A: training-style in-process ──
    policy = SmolVLAPolicy.from_pretrained(args.model).to(device).eval()
    cfg = policy.config
    pre, post = make_pre_post_processors(cfg, pretrained_path=args.model)
    feed_A = int(getattr(cfg, "n_action_steps", 50))
    policy.reset(); pre.reset(); post.reset()
    predA = []
    chunk = None; cstart = 0
    for t in range(n):
        if chunk is None or (t - cstart) >= feed_A:
            obs = {"observation.state": frames[t]["observation.state"], "task": TASK}
            for k in CAM_DS:
                obs[k] = frames[t][k]
            with torch.inference_mode():
                cn = policy.predict_action_chunk(pre(obs))
                chunk = np.stack([post(cn[:, h]).squeeze(0).float().cpu().numpy()
                                  for h in range(cn.shape[1])])
            cstart = t
        predA.append(chunk[min(t - cstart, chunk.shape[0] - 1)])
    predA = np.stack(predA)
    print(f"[cmp] Path A (training, feed_every={feed_A}) done", flush=True)

    # ── Path B: async + RTC + gRPC (vla_policy_node 소비로직) ──
    lerobot_features = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": list(JOINT_FEATURE_NAMES)},
    }
    for pk in CAM_POL:
        lerobot_features[f"observation.images.{pk}"] = {
            "dtype": "image", "shape": (480, 640, 3), "names": ["height", "width", "channels"]}
    pcfg = RemotePolicyConfig("smolvla", args.model, lerobot_features, args.actions_per_chunk, device, {})
    channel = grpc.insecure_channel(args.server, grpc_channel_options(initial_backoff="0.0333s"))
    stub = services_pb2_grpc.AsyncInferenceStub(channel)
    stub.Ready(services_pb2.Empty())
    stub.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(pcfg)))  # nosec
    print(f"[cmp] RTC server connected ({args.server}), instructions sent", flush=True)

    def predict_chunk(raw_obs, timestep):
        to = TimedObservation(timestamp=time.time(), timestep=timestep, observation=raw_obs, must_go=True)
        it = send_bytes_in_chunks(pickle.dumps(to), services_pb2.Observation, log_prefix="", silent=True)  # nosec
        stub.SendObservations(it)
        deadline = time.perf_counter() + 8.0
        while time.perf_counter() < deadline:
            ch = stub.GetActions(services_pb2.Empty())
            if len(ch.data) > 0:
                return pickle.loads(ch.data)  # nosec
            time.sleep(0.005)
        return []

    def build_obs_B(fr):
        st = np.asarray(fr["observation.state"], dtype=np.float32)
        obs = {name: float(st[i]) for i, name in enumerate(JOINT_FEATURE_NAMES)}
        for dsk, pk in zip(CAM_DS, CAM_POL):
            im = fr[dsk]  # CHW float[0,1] → HWC uint8 (vla node 가 보내는 형식)
            obs[pk] = (im.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        obs["task"] = TASK
        return obs

    refill = max(1, int(args.actions_per_chunk * args.chunk_threshold))
    queue: deque = deque(); timestep = 0
    predB = []; feedsB = []
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
        ts, act = queue.popleft(); timestep = ts + 1
        predB.append(act)
    predB = np.stack(predB)
    print(f"[cmp] Path B (async+RTC, apc={args.actions_per_chunk} refill={refill}) done, "
          f"{len(feedsB)} obs sent", flush=True)

    # ── metrics ──
    mae_A = np.abs(predA - recorded).mean(axis=0)
    mae_B = np.abs(predB - recorded).mean(axis=0)
    mae_AB = np.abs(predA - predB).mean(axis=0)   # async+RTC 가 유발한 차이
    print("[cmp] ===== per-joint MAE (lerobot units) =====", flush=True)
    for j, nm in enumerate(JOINT_NAMES):
        u = "[0,100]" if nm == "gripper" else "deg"
        print(f"[cmp]   {nm:13s} A-vs-rec={mae_A[j]:6.2f}  B-vs-rec={mae_B[j]:6.2f}  "
              f"A-vs-B={mae_AB[j]:6.2f}  {u}", flush=True)
    print(f"[cmp] overall: A-vs-rec={mae_A.mean():.3f}  B-vs-rec={mae_B.mean():.3f}  "
          f"A-vs-B(async+RTC효과)={mae_AB.mean():.3f}", flush=True)

    # ── overlay ──
    fig, axes = plt.subplots(3, 2, figsize=(16, 11))
    tt = np.arange(n)
    for j, nm in enumerate(JOINT_NAMES):
        ax = axes[j // 2][j % 2]
        ax.plot(tt, recorded[:, j], label="recorded (expert)", lw=2.2, color="tab:blue")
        ax.plot(tt, predA[:, j], label="A: training (in-proc chunk)", lw=1.3, color="tab:green", alpha=0.85)
        ax.plot(tt, predB[:, j], label="B: async+RTC (gRPC)", lw=1.3, color="tab:red", alpha=0.85)
        for fr in feedsB:
            ax.axvline(fr, color="red", ls=":", lw=0.5, alpha=0.35)
        u = "[0,100]" if nm == "gripper" else "deg"
        ax.set_title(f"{nm} ({u})  A-rec={mae_A[j]:.2f} B-rec={mae_B[j]:.2f} A-B={mae_AB[j]:.2f}")
        ax.set_xlabel("frame"); ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"training vs async+RTC — {args.model.split('/')[-1]} ep{args.episode} "
                 f"(red dotted=B obs sent, apc={args.actions_per_chunk}, n={n})", fontsize=13)
    fig.tight_layout()
    png = f"{args.out}_ep{args.episode}.png"
    fig.savefig(png, dpi=110)
    np.savez(f"{args.out}_ep{args.episode}.npz", recorded=recorded, predA=predA, predB=predB,
             feedsB=np.array(feedsB), mae_A=mae_A, mae_B=mae_B, mae_AB=mae_AB)
    print(f"[cmp] 🎬 PNG → {png}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
