"""sim-프레임 LeRobot v3 데이터셋 → 실 SO-101 follower 프레임 in-place 변환 (Isaac 무의존).

sim 에서 녹화한 데이터셋(arm=sim URDF degree, gripper feature[0,100] = policy-feature)을
실기기 `lerobot-replay` 로 재생하려면, 각 관절을 실 follower 영점/스케일로 되돌려야 한다
(sim URDF 영점 ≠ 실 follower 영점 → 안 하면 EE ~2.4cm 오차·grasp 헛집음).

변환 = `so101_contract.follower_calibration.policy_feature_to_real_follower`
       (= feature_codec ∘ follower affine 역). `record_real_sequence`(real→sim)의 반대 방향.

**in-place**: data parquet 의 action·observation.state 값 + meta/stats.json 의 해당 통계만
갱신한다. videos·info.json·tasks·episodes 는 관절 프레임과 무관하므로 그대로 둔다.

    uv run python scripts/convert/sim_dataset_to_real_follower.py \
        --dataset_dir datasets/pink_ik_pickplace          # action+state 둘 다(기본)
    uv run python scripts/convert/sim_dataset_to_real_follower.py --self-check

⚠ 실기기 replay 는 잘못된 관절 타깃 = 충돌 위험. e-stop 준비하고 실행할 것.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from so101_contract.follower_calibration import policy_feature_to_real_follower  # noqa: E402

# meta/stats.json 통계 (lerobot_recorder._numeric_stats 와 동일 포맷 — 값만 새로 계산).
_STAT_QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}


def _numeric_stats(arr: np.ndarray) -> dict:
    """(N, D) → per-dim min/max/mean/std/count/q*. lerobot_recorder 포맷."""
    stats = {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [int(arr.shape[0])],
    }
    for name, q in _STAT_QUANTILES.items():
        stats[name] = np.quantile(arr, q, axis=0).tolist()
    return stats


def convert(dataset_dir: Path, cols: list[str]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    info = json.loads((dataset_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    for c in cols:
        if info["features"].get(c, {}).get("shape") != [6]:
            raise ValueError(f"'{c}' 가 6-dim feature 아님(info.json): {info['features'].get(c)}")

    data_files = sorted((dataset_dir / "data").rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"data parquet 없음: {dataset_dir/'data'}")

    fsl6 = pa.list_(pa.float32(), 6)
    accum: dict[str, list[np.ndarray]] = {c: [] for c in cols}
    for path in data_files:
        table = pq.read_table(path)
        for c in cols:
            sim = np.asarray(table.column(c).to_pylist(), dtype=np.float32)  # (N,6) policy-feature
            real = policy_feature_to_real_follower(sim).astype(np.float32)    # (N,6) follower
            accum[c].append(real)
            table = table.set_column(table.schema.get_field_index(c), c,
                                     pa.array(real.tolist(), type=fsl6))
        pq.write_table(table, path)
        print(f"[convert] {path.relative_to(dataset_dir)} — {', '.join(cols)} → real follower")

    stats_path = dataset_dir / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    for c in cols:
        stats[c] = _numeric_stats(np.concatenate(accum[c], axis=0))
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[convert] stats.json 갱신: {', '.join(cols)}")
    print(f"[convert] ✓ {dataset_dir} = 실 follower 프레임 (lerobot-replay 로 실기기 재생 가능)")


def self_check() -> int:
    # policy-feature 입력에 follower 역affine 이 올바른 방향으로 적용되는지(offset 부호 포함).
    from so101_contract.follower_calibration import FOLLOWER_AFFINE_B
    x = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 50.0], dtype=np.float32)  # arm deg + gripper feat
    real = policy_feature_to_real_follower(x)
    # arm(0-4): sim_deg = policy arm deg(1:1), real = sim_deg - B → x[:5] - B[:5].
    expect_arm = x[:5] - FOLLOWER_AFFINE_B[:5]
    assert np.allclose(real[:5], expect_arm, atol=1e-3), f"arm {real[:5]} != {expect_arm}"
    # 배치(N,6) 도 동일해야 함.
    batch = np.stack([x, x + 1.0])
    assert np.allclose(policy_feature_to_real_follower(batch)[0], real, atol=1e-4), "배치 불일치"
    print(f"[self-check] policy_feature({x.tolist()}) → real follower {np.round(real,3).tolist()}")
    print("[self-check] PASS — arm offset(부호) + 배치 정합")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--dataset_dir", help="LeRobot v3 데이터셋 폴더(meta/ data/ videos/)")
    ap.add_argument("--convert", choices=["action", "state", "both"], default="both",
                    help="변환할 컬럼(기본 both)")
    args = ap.parse_args()
    if args.self_check:
        return self_check()
    if not args.dataset_dir:
        ap.error("--dataset_dir 필요 (또는 --self-check)")
    cols = {"action": ["action"], "state": ["observation.state"],
            "both": ["action", "observation.state"]}[args.convert]
    convert(Path(args.dataset_dir).resolve(), cols)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
