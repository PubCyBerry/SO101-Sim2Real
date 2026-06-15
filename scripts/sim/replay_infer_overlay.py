"""학습 데이터 1 에피소드 → SmolVLA (학습 때 들어간 방식) 추론 + recorded overlay 진단.

⚠ sim·gRPC/RTC 추론 I/O 아님. **학습 때 데이터가 들어간 방식**을 그대로 재현한다:
  LeRobotDataset frame → 학습 preprocessor(rename top→camera1·normalize, 학습 stats 내장)
  → SmolVLA.predict_action_chunk → postprocessor(unnormalize). in-process.

chunk 단위 소비: 매 --horizon(기본 n_action_steps) 프레임마다 obs 1개를 정책에 넣어
chunk(예측 action 묶음)를 받고, 그 chunk 를 horizon 만큼 재생한다. **obs 가 정책에 들어가는
프레임(= chunk feed 시점)마다 marker** 로 표기 → 입력 cadence 가 보인다.

해석: 입력 직후(marker)에서 예측이 recorded 에 붙고 horizon 끝으로 갈수록 벌어지면(톱니) =
정책이 학습분포를 재현(units/normalize/image 정상). marker 직후도 안 맞으면 = 정규화/단위/
이미지/과소학습 문제.

lerobot 0.5.1 컨테이너(policy-server 이미지)에서 실행. 모델·데이터셋 = HF repo(hf_cache 캐시):
  docker compose --env-file .env -f docker/docker-compose.yaml run --rm --no-deps \
    --entrypoint bash policy-server -lc \
    'python /workspace/scripts/sim/replay_infer_overlay.py --episode 0'
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
TASK = "pick up the cube and place it in the bowl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="taehunkim/so101_smolvla_sim_pick_cube")
    ap.add_argument("--dataset", default="taehunkim/so101_sim_pick_cube")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="/workspace/outputs/vla_replay_overlay")
    ap.add_argument("--horizon", type=int, default=0,
                    help="obs 재입력 간격(프레임). 0=정책 n_action_steps 사용. chunk feed = marker.")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[replay] device={device} model={args.model} ds={args.dataset} ep={args.episode}", flush=True)

    # ── 정책 + 학습 pre/post processor (학습 stats·rename 내장) ──
    policy = SmolVLAPolicy.from_pretrained(args.model).to(device).eval()
    cfg = policy.config
    pre, post = make_pre_post_processors(cfg, pretrained_path=args.model)
    chunk_size = int(getattr(cfg, "chunk_size", 50))
    feed_every = args.horizon if args.horizon > 0 else int(getattr(cfg, "n_action_steps", chunk_size))
    print(f"[replay] chunk_size={chunk_size} n_action_steps={getattr(cfg,'n_action_steps','?')} "
          f"feed_every={feed_every}", flush=True)

    # ── 데이터셋 1 에피소드 프레임 범위 ──
    ds = LeRobotDataset(args.dataset)
    ei = np.asarray(ds.hf_dataset["episode_index"])
    where = np.where(ei == args.episode)[0]
    if where.size == 0:
        raise SystemExit(f"episode {args.episode} 없음 (num_episodes={ds.num_episodes})")
    f0, f1 = int(where[0]), int(where[-1]) + 1
    n = f1 - f0
    print(f"[replay] episode {args.episode}: frames [{f0},{f1}) = {n}", flush=True)

    img_keys = [k for k in ds.features if k.startswith("observation.images.")]

    def build_obs(frame: dict) -> dict:
        """학습 dataset frame → preprocessor 입력 dict (unbatched; AddBatchDim 이 배치화)."""
        obs = {"observation.state": frame["observation.state"], "task": TASK}
        for k in img_keys:
            obs[k] = frame[k]   # CHW float[0,1] = 학습 dataloader 와 동일 포맷
        return obs

    def predict_chunk(frame: dict) -> np.ndarray:
        with torch.inference_mode():
            pobs = pre(build_obs(frame))
            chunk_norm = policy.predict_action_chunk(pobs)        # [1, chunk_size, 6] (normalized)
            acts = [post(chunk_norm[:, h]).squeeze(0).float().cpu().numpy()
                    for h in range(chunk_norm.shape[1])]          # unnormalize per step
        return np.stack(acts)                                     # [chunk_size, 6] lerobot units

    policy.reset(); pre.reset(); post.reset()
    recorded, predicted, feed_frames = [], [], []
    cur_chunk, chunk_start = None, 0
    for t in range(n):
        frame = ds[f0 + t]
        if cur_chunk is None or (t - chunk_start) >= feed_every:
            cur_chunk = predict_chunk(frame)   # obs 입력 시점
            chunk_start = t
            feed_frames.append(t)
            print(f"[replay]   feed obs @ frame {t}/{n}", flush=True)
        off = min(t - chunk_start, cur_chunk.shape[0] - 1)
        predicted.append(cur_chunk[off])
        recorded.append(np.asarray(frame["action"], dtype=np.float32))

    recorded = np.stack(recorded)     # [n,6] lerobot units (arm deg, gripper [0,100])
    predicted = np.stack(predicted)
    mae = np.abs(predicted - recorded).mean(axis=0)
    # marker 직후(첫 스텝) 오차 — 입력 시점에서의 순간 예측 품질
    first = np.array(feed_frames)
    mae_first = np.abs(predicted[first] - recorded[first]).mean(axis=0)

    print("[replay] ===== per-joint MAE (lerobot units) =====", flush=True)
    for j, name in enumerate(JOINT_NAMES):
        unit = "[0,100]" if name == "gripper" else "deg"
        print(f"[replay]   {name:13s} MAE={mae[j]:7.3f} (feed-step MAE={mae_first[j]:7.3f}) {unit} "
              f"| recorded {recorded[:,j].min():.2f}..{recorded[:,j].max():.2f}", flush=True)

    # ── overlay plot + 입력(marker) ──
    fig, axes = plt.subplots(3, 2, figsize=(16, 11))
    tt = np.arange(n)
    for j, name in enumerate(JOINT_NAMES):
        ax = axes[j // 2][j % 2]
        ax.plot(tt, recorded[:, j], label="recorded (expert)", lw=2.0, color="tab:blue")
        ax.plot(tt, predicted[:, j], label="predicted (SmolVLA)", lw=1.4, color="tab:red", alpha=0.85)
        for fr in feed_frames:   # obs 입력 시점 marker
            ax.axvline(fr, color="green", ls="--", lw=0.7, alpha=0.5)
        ax.scatter(first, predicted[first, j], color="green", s=22, zorder=5,
                   label="obs fed (input)")
        unit = "[0,100]" if name == "gripper" else "deg"
        ax.set_title(f"{name}  ({unit})   MAE={mae[j]:.2f}  feed-step MAE={mae_first[j]:.2f}")
        ax.set_xlabel("frame"); ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"training-input replay overlay — {args.model.split('/')[-1]} ep{args.episode} "
                 f"(feed_every={feed_every}, green=obs fed, n={n})", fontsize=13)
    fig.tight_layout()
    png = f"{args.out}_ep{args.episode}.png"
    npz = f"{args.out}_ep{args.episode}.npz"
    fig.savefig(png, dpi=110)
    np.savez(npz, recorded=recorded, predicted=predicted, mae=mae, feed_frames=np.array(feed_frames),
             joints=JOINT_NAMES)
    print(f"[replay] 🎬 PNG → {png}\n[replay] NPZ → {npz}", flush=True)
    print(f"[replay] overall MAE={mae.mean():.3f} (arm5={mae[:5].mean():.3f}deg gripper={mae[5]:.3f}) | "
          f"feed-step overall MAE={mae_first.mean():.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
