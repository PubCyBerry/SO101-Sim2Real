#!/usr/bin/env python
"""LeRobot v3 데이터셋 joint·gripper 분포 진단 (Isaac 무의존, 서버 실행용).

sim→real 추론 parity 점검용. `observation.state` / `action` 6축의 분포
(min·q01·q50·q99·max·mean·std)를 뽑고, gripper(6번째)를 rad·실기기 [0,100] 개도로
환산해 **sim↔real gripper scale mismatch** 를 정량화한다. arm 5축은 degree 인지
(|값| > 100° 존재 여부로) 확인한다 — degree 면 real `use_degrees=True`(기본값)와 정합.

근거: `docs/SIM_REAL_INFERENCE_PARITY.md` §6 Option D. sim 은 gripper 를
`rad × GRIPPER_LEROBOT_SCALE(31.75)` 로 기록(`scripts/sim/lerobot_units.py`)하나,
실기기 `SOFollower` 의 gripper 는 항상 `MotorNormMode.RANGE_0_100` →
[0,100] 가 캘리브 full-travel 백분율. 두 척도가 캘리브된 적 없어 같은 모델 출력이
real 에서 덜 열린다.

사용:
  # 서버 로컬 데이터셋 (메인 리포 = /home/konan147/Workspaces/SO101-Sim2Real)
  uv run --no-project python scripts/sim/inspect_dataset_distribution.py \
      --root outputs/so101_sim_pick_cube_4cube_1024

  # HF repo (data/meta 만 다운로드, 비디오 제외)
  uv run --no-project python scripts/sim/inspect_dataset_distribution.py \
      --repo_id taehunkim/so101_sim_pick_cube_4cube_1024

  # 실측한 실기기 gripper open/close([0,100])를 주면 affine 권장값까지 출력
  ... --real_open 70 --real_close 6
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
GRIPPER_LEROBOT_SCALE = 31.75  # scripts/sim/lerobot_units.py 와 동일


def _load_columns(root: str) -> dict[str, np.ndarray]:
    """LeRobot v3 parquet 에서 observation.state·action 컬럼을 (N,6) ndarray 로."""
    import pyarrow.dataset as pads

    data_dir = os.path.join(root, "data")
    if os.path.isdir(data_dir):
        dataset = pads.dataset(data_dir, format="parquet")
    else:
        files = glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True)
        if not files:
            raise FileNotFoundError(f"{root} 아래 parquet 없음 (LeRobot v3 데이터셋 경로 확인)")
        dataset = pads.dataset(files, format="parquet")

    names = set(dataset.schema.names)
    cols = [c for c in ("observation.state", "action") if c in names]
    if not cols:
        raise KeyError(f"observation.state/action 컬럼 없음. 보유 컬럼: {sorted(names)}")

    table = dataset.to_table(columns=cols)
    out: dict[str, np.ndarray] = {}
    for c in cols:
        arr = np.asarray(table.column(c).to_pylist(), dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 6:
            raise ValueError(f"{c} shape {arr.shape}, (N,6) 기대 (6축 계약 위반)")
        out[c] = arr
    return out


def _stats_table(name: str, arr: np.ndarray) -> np.ndarray:
    """per-joint 분포 출력 + [q01,q50,q99] (3,6) 반환."""
    print(f"\n== {name}  shape={arr.shape}  (arm=deg, gripper=rad×{GRIPPER_LEROBOT_SCALE}) ==")
    print(f"{'joint':14s}{'min':>9s}{'q01':>9s}{'q50':>9s}{'q99':>9s}{'max':>9s}{'mean':>9s}{'std':>9s}")
    qs = np.percentile(arr, [1, 50, 99], axis=0)
    for i, j in enumerate(JOINTS):
        col = arr[:, i]
        print(f"{j:14s}{col.min():9.2f}{qs[0, i]:9.2f}{qs[1, i]:9.2f}"
              f"{qs[2, i]:9.2f}{col.max():9.2f}{col.mean():9.2f}{col.std():9.2f}")
    return qs


def main() -> None:
    ap = argparse.ArgumentParser(description="LeRobot v3 joint·gripper 분포 진단 (parity 점검)")
    ap.add_argument("--root", help="로컬 LeRobot v3 데이터셋 디렉터리")
    ap.add_argument("--repo_id", help="HF 데이터셋 repo id (data/meta 만 다운로드)")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--real_open", type=float, default=70.0, help="실기기 full open [0,100] 실측값")
    ap.add_argument("--real_close", type=float, default=6.0, help="실기기 full close [0,100] 실측값")
    args = ap.parse_args()

    root = args.root
    if root is None:
        if not args.repo_id:
            ap.error("--root 또는 --repo_id 중 하나 필요")
        from huggingface_hub import snapshot_download

        root = snapshot_download(
            args.repo_id, repo_type="dataset", revision=args.revision,
            allow_patterns=["data/**", "meta/**"],
        )
        print(f"[info] downloaded (data/meta only) → {root}", flush=True)

    data = _load_columns(root)
    grip: dict[str, tuple[float, float, float, float]] = {}  # name → (min,q01,q99,max)
    for name in ("action", "observation.state"):
        if name in data:
            qs = _stats_table(name, data[name])
            col = data[name][:, 5]
            grip[name] = (float(col.min()), float(qs[0, 5]), float(qs[2, 5]), float(col.max()))

    # ── arm degree 판정 ──────────────────────────────────────────────────────
    arm_abs_max = max(float(np.abs(data[n][:, :5]).max()) for n in data)
    if arm_abs_max > 100.0:
        verdict = "> 100° → DEGREE 확정. real 은 use_degrees=True(기본값) 필요 → arm 단위 정합."
    else:
        verdict = "≤ 100° → degree/RANGE_M100_100 구분 불가. 모델 config·real 캘리브 직접 확인 요망."
    print(f"\n[arm 단위] |arm 5축| 최대 = {arm_abs_max:.1f}°  {verdict}")
    print("           ⚠ 단위가 같아도 영점(real homing=가동범위 중점 vs sim URDF zero)·"
          "부호(drive_mode) 정합은 별도 확인 필요(절대자세 어긋남 리스크).")

    # ── gripper 환산·affine 권장 ─────────────────────────────────────────────
    if "action" in grip:
        gmin, gq01, gq99, gmax = grip["action"]
        print(f"\n[gripper action] [0,100]계(=rad×{GRIPPER_LEROBOT_SCALE}): "
              f"min={gmin:.2f} q01={gq01:.2f} q99={gq99:.2f} max={gmax:.2f}")
        print(f"  rad 환산(÷{GRIPPER_LEROBOT_SCALE}): close(q01)={gq01 / GRIPPER_LEROBOT_SCALE:.3f} rad, "
              f"open(q99)={gq99 / GRIPPER_LEROBOT_SCALE:.3f} rad")
        print(f"  real 해석: 모델 'open' 명령 ≈ {gq99:.1f} → real 그리퍼 개도 {gq99:.1f}% "
              f"(실측 full open = {args.real_open:.0f}%)")
        if args.real_open:
            ratio = gq99 / args.real_open * 100.0
            flag = ("⚠ scale mismatch — Option A(재학습) 또는 Option B(shim affine) 필요"
                    if ratio < 85 else "≈ 정합")
            print(f"  => real 그리퍼는 sim 의도의 약 {ratio:.0f}% 만 열림  {flag}")

        # 2점 affine: (모델 q01 → real close), (모델 q99 → real open)
        if gq99 != gq01:
            a = (args.real_open - args.real_close) / (gq99 - gq01)
            b = args.real_close - a * gq01
            print(f"\n[권장 Option B affine]  grip_real = {a:.3f} * grip_model + {b:.3f}")
            print(f"  앵커: 모델 close(q01)={gq01:.2f}→real {args.real_close:.0f}, "
                  f"open(q99)={gq99:.2f}→real {args.real_open:.0f}")
            print("  policy-client-shim.py 환경변수 (그대로 export 해서 추론):")
            print(f"    GRIPPER_AFFINE=1 \\\n"
                  f"    GRIPPER_SIM_CLOSE={gq01:.2f} GRIPPER_SIM_OPEN={gq99:.2f} \\\n"
                  f"    GRIPPER_REAL_CLOSE={args.real_close:.0f} GRIPPER_REAL_OPEN={args.real_open:.0f}")


if __name__ == "__main__":
    main()
