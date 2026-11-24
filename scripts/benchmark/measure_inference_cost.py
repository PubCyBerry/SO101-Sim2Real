#!/usr/bin/env python3
"""정책 1개의 **추론 비용**(chunk latency · VRAM)을 컨테이너 안에서 재고 JSON 을 출력한다.

`sweep_train_cost.py --infer` 가 정책별로 이 스크립트를 컨테이너에서 실행한다. 직접 실행할 일은
디버깅뿐이다.

무엇을 재는가
-------------
`select_action` 은 큐가 비었을 때만 실제 추론(chunk 전체 생성)을 하고, 그 다음 호출들은 큐에서
꺼내기만 한다. 그래서 **매 호출 전에 `policy.reset()`** 을 불러 항상 full-chunk 추론이 되게 하고
그 시간을 잰다 — async policy-server 가 실제로 감당하는 비용이 이것이다.

* ``chunk_p50_s`` / ``chunk_p95_s`` — full-chunk 추론 1회 latency
* ``per_step_ms`` — ``chunk_p50_s / n_action_steps`` (chunk 를 다 쓰는 경우의 스텝당 상환 비용)
* ``realtime_headroom`` — 30Hz 제어에서 chunk 하나가 벌어주는 시간 대비 추론 시간
  (``n_action_steps/30 / chunk_p50_s``; 1 보다 작으면 실시간 미달)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: E402
from lerobot.policies.factory import (  # noqa: E402
    make_policy,
    make_policy_config,
    make_pre_post_processors,
)


def _delta_timestamps(
    cfg: PreTrainedConfig, fps: int, dataset_keys: list[str]
) -> dict[str, list[float]] | None:
    """policy 가 요구하는 obs/action 시간 오프셋 → LeRobotDataset delta_timestamps.

    키는 **dataset 쪽 이름**이어야 한다 — rename 을 쓰는 정책의 `cfg.input_features` 는
    체크포인트 카메라 이름(예: exterior_1_left)이라 dataset 에 없다.
    """
    delta: dict[str, list[float]] = {}
    obs_indices = getattr(cfg, "observation_delta_indices", None)
    if obs_indices:
        for key in dataset_keys:
            delta[key] = [i / fps for i in obs_indices]
    action_indices = getattr(cfg, "action_delta_indices", None)
    if action_indices:
        delta["action"] = [i / fps for i in action_indices]
    return delta or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-json", required=True, help='{"type":..,"path":..,"overrides":{..}}')
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    spec = json.loads(args.policy_json)
    path = spec.get("path")
    if path:
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.pretrained_path = path
    else:
        cfg = make_policy_config(spec["type"])
    for key, value in (spec.get("overrides") or {}).items():
        if not hasattr(cfg, key):
            raise KeyError(f"{type(cfg).__name__} has no field {key!r}")
        setattr(cfg, key, value)
    cfg.device = args.device

    rename_map = spec.get("rename_map") or None
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    policy = make_policy(cfg, ds_meta=meta, rename_map=rename_map)
    # rename 은 processor 쪽에도 넣어야 한다 (train.py 와 같은 경로:
    # preprocessor_overrides["rename_observations_processor"]["rename_map"]).
    preprocessor, _ = make_pre_post_processors(
        cfg,
        pretrained_path=path,
        dataset_stats=meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": args.device},
            "normalizer_processor": {
                "stats": meta.stats,
                "features": {**cfg.input_features, **cfg.output_features},
                "norm_map": cfg.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": rename_map or {}},
        },
    )
    policy.eval()

    observation_keys = [key for key in meta.features if key.startswith("observation.")]
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.root,
        delta_timestamps=_delta_timestamps(cfg, meta.fps, observation_keys),
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=0, shuffle=False
    )
    batch = next(iter(loader))
    for camera_key in meta.camera_keys:
        if camera_key in batch and batch[camera_key].dtype == torch.uint8:
            batch[camera_key] = batch[camera_key].to(dtype=torch.float32) / 255.0
    batch = preprocessor(batch)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    latencies: list[float] = []
    with torch.inference_mode():
        for index in range(args.warmup + args.iters):
            # 매 호출을 full-chunk 추론으로 만든다 (큐 pop 은 비용이 아니다)
            policy.reset()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            policy.select_action(dict(batch))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if index >= args.warmup:
                latencies.append(elapsed)

    n_action_steps = int(getattr(cfg, "n_action_steps", 1) or 1)
    p50 = statistics.median(latencies)
    result = {
        "policy_type": cfg.type,
        "pretrained_path": path,
        "batch_size": args.batch_size,
        "device": args.device,
        "n_action_steps": n_action_steps,
        "chunk_p50_s": round(p50, 4),
        "chunk_p95_s": round(max(latencies) if len(latencies) < 20 else sorted(latencies)[int(0.95 * len(latencies))], 4),
        "chunk_min_s": round(min(latencies), 4),
        "per_step_ms": round(1000.0 * p50 / n_action_steps, 2),
        "realtime_headroom_30hz": round((n_action_steps / 30.0) / p50, 2),
        "mem_gb_allocated": (
            round(torch.cuda.max_memory_allocated() / 1024**3, 2) if torch.cuda.is_available() else None
        ),
        "num_total_params": sum(p.numel() for p in policy.parameters()),
        "iters": args.iters,
    }
    print("INFER_RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
