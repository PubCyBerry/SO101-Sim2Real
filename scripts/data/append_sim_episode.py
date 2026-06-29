#!/usr/bin/env python3
"""sim replay 기록(replay_dataset_to_bridge.py --record_dir)을 기존 LeRobot v3 dataset 에
episode 1개로 append. 변환 없는 그대로의 sim observation(state·3cam)+action 을 추가한다.

흐름(2-환경 분리):
  ① vla-ros: replay 중 --record_dir 로 frames.npz(action·state, LeRobot 단위) + {cam}/*.png 기록.
  ② host uv(full lerobot): 이 스크립트가 기존 dataset 을 root 로 내려받아 LeRobotDataset.add_frame
     /save_episode 로 새 episode append. push 는 분리(scripts/data/upload_to_huggingface.py).

push 안 함(검증 후 별도). 사용:
  uv run python scripts/data/append_sim_episode.py \\
      --record_dir scratch/sim_ep1 --repo_id taehunkim/so101_pick_cube_test --root scratch/ds_work
  # 검증 후:
  uv run python scripts/data/upload_to_huggingface.py \\
      --dataset_dir scratch/ds_work --repo_id taehunkim/so101_pick_cube_test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record_dir", required=True, help="frames.npz + {cam}/*.png 가 있는 폴더")
    ap.add_argument("--repo_id", required=True, help="append 대상 dataset repo_id")
    ap.add_argument("--root", default=None, help="로컬 작업 dir(기존 dataset 내려받아 여기에 episode 추가)")
    args = ap.parse_args()

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception:  # noqa: BLE001  구버전 경로
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    rd = Path(args.record_dir)
    meta = np.load(rd / "frames.npz", allow_pickle=True)
    action = meta["action"].astype(np.float32)
    state = meta["state"].astype(np.float32)
    task = str(meta["task"])
    cams = [str(c) for c in meta["cameras"]]
    n = action.shape[0]
    print(f"[append] record_dir={rd} frames={n} cams={cams} task={task!r}", flush=True)

    ds = LeRobotDataset(args.repo_id, root=args.root)
    before = ds.num_episodes
    feats = list(ds.features)
    print(f"[append] 기존 episodes={before} fps={ds.fps} features={feats}", flush=True)
    for cam in cams:
        if f"observation.images.{cam}" not in feats:
            raise SystemExit(f"dataset 에 observation.images.{cam} 없음 — 스키마 불일치")

    for k in range(n):
        frame: dict = {"observation.state": state[k], "action": action[k], "task": task}
        for cam in cams:
            frame[f"observation.images.{cam}"] = np.asarray(
                imageio.imread(rd / cam / f"{k:06d}.png"), dtype=np.uint8
            )  # HWC RGB
        ds.add_frame(frame)
    ds.save_episode()
    print(f"[append] append 완료: episodes {before} → {ds.num_episodes}  root={ds.root}", flush=True)
    print(f"[append] push: uv run python scripts/data/upload_to_huggingface.py "
          f"--dataset_dir {ds.root} --repo_id {args.repo_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
