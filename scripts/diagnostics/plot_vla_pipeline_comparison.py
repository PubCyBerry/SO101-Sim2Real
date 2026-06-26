"""VLA expert data·open-loop·실제 closed-loop 파이프라인을 한 그림으로 비교한다.

open-loop NPZ의 recorded/pred와 ROS VLA node의 trajectory JSONL을 사용한다.
closed-loop 로그 앞에 warmup 구간이 있어도 eval JSON의 episode 수를 기준으로
마지막 N개 segment만 선택한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def _load_closedloop_segments(path: Path) -> list[list[dict]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"빈 closed-loop 로그: {path}")

    segments: list[list[dict]] = []
    start = 0
    previous_ts: int | float | None = None
    for index, row in enumerate(rows):
        current_ts = row["ts"]
        if previous_ts is not None and current_ts <= previous_ts:
            segments.append(rows[start:index])
            start = index
        previous_ts = current_ts
    segments.append(rows[start:])
    return [segment for segment in segments if segment]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openloop", type=Path, required=True)
    parser.add_argument("--closedloop", type=Path, required=True)
    parser.add_argument("--eval_json", type=Path, required=True)
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="VLA 4-cube pipeline comparison")
    args = parser.parse_args()

    openloop = np.load(args.openloop)
    recorded = np.asarray(openloop["recorded"], dtype=np.float32)
    predicted = np.asarray(openloop["pred"], dtype=np.float32)
    if recorded.shape != predicted.shape or recorded.shape[1] != len(JOINT_NAMES):
        raise ValueError(f"open-loop shape 불일치: recorded={recorded.shape}, pred={predicted.shape}")

    eval_result = json.loads(args.eval_json.read_text())
    n_episodes = int(eval_result["n_episodes"])
    segments = _load_closedloop_segments(args.closedloop)
    if len(segments) < n_episodes:
        raise ValueError(f"closed-loop segment 부족: {len(segments)} < eval episodes {n_episodes}")
    episode_segments = segments[-n_episodes:]
    if not 0 <= args.episode_index < len(episode_segments):
        raise ValueError(f"episode_index 범위 오류: {args.episode_index} / {len(episode_segments)}")

    closed_rows = episode_segments[args.episode_index]
    closed_state = np.asarray([row["state"] for row in closed_rows], dtype=np.float32)
    closed_action = np.asarray([row["action"] for row in closed_rows], dtype=np.float32)
    if closed_state.shape != closed_action.shape or closed_state.shape[1] != len(JOINT_NAMES):
        raise ValueError(
            f"closed-loop shape 불일치: state={closed_state.shape}, action={closed_action.shape}"
        )

    fig, axes = plt.subplots(6, 3, figsize=(20, 20), constrained_layout=True)
    for joint_index, joint_name in enumerate(JOINT_NAMES):
        unit = "[0,100]" if joint_name == "gripper" else "deg"

        data_ax = axes[joint_index, 0]
        data_ax.plot(recorded[:, joint_index], color="tab:green", linewidth=1.2)
        data_ax.set_title(f"{joint_name} ({unit}) · expert data")

        open_ax = axes[joint_index, 1]
        open_ax.plot(
            recorded[:, joint_index],
            color="tab:green",
            linewidth=1.2,
            label="recorded",
        )
        open_ax.plot(
            predicted[:, joint_index],
            color="tab:orange",
            linewidth=1.0,
            alpha=0.85,
            label="teacher-forced prediction",
        )
        joint_mae = float(np.mean(np.abs(predicted[:, joint_index] - recorded[:, joint_index])))
        open_ax.set_title(f"{joint_name} ({unit}) · open-loop MAE={joint_mae:.2f}")

        closed_ax = axes[joint_index, 2]
        closed_ax.plot(
            closed_action[:, joint_index],
            color="tab:red",
            linewidth=1.0,
            alpha=0.9,
            label="policy action",
        )
        closed_ax.plot(
            closed_state[:, joint_index],
            color="tab:blue",
            linewidth=1.0,
            alpha=0.75,
            label="achieved state",
        )
        tracking_mae = float(
            np.mean(np.abs(closed_action[:, joint_index] - closed_state[:, joint_index]))
        )
        closed_ax.set_title(f"{joint_name} ({unit}) · closed-loop tracking MAE={tracking_mae:.2f}")

        for axis in (data_ax, open_ax, closed_ax):
            axis.grid(alpha=0.25)
            axis.set_xlabel("step")
        if joint_index == 0:
            open_ax.legend(fontsize=8)
            closed_ax.legend(fontsize=8)

    episode_result = eval_result["episodes"][args.episode_index]
    fig.suptitle(
        f"{args.title}\n"
        f"open-loop {len(recorded)} frames · closed-loop episode {args.episode_index} "
        f"{len(closed_rows)} ticks · final {episode_result['n_final']}/"
        f"{eval_result['n_active_cubes']} · ever {episode_result['n_ever']}/"
        f"{eval_result['n_active_cubes']}",
        fontsize=15,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    plt.close(fig)
    print(f"[plot] 저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
