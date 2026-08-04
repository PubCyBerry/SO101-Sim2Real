#!/usr/bin/env python3
"""증강 데이터 구간 경계 불연속 정량화 — 명령 포화율 기반.

## 무엇을 재는가

증강(Mimic/SkillGen) 에피소드가 `[planner 구간] + [source replay 구간]` 으로 봉합될 때마다
명령이 slew cap 에 붙으면 텔레포트나 급가속이 생긴다. 사용자가 "구간 넘어갈 때마다 끊긴다"고
관찰한 것이 이것이다.

**주 판정 지표 = 명령 포화율**(post-slew command target이 slew cap에 붙은 프레임 %).
물리 joint 속도 지표는 오판을 낼 수 있다(2026-07-30 실측: 보간 0→15가 물리로는 악화인데
명령으로는 6.5배 개선).

## source 와 비교

`states/articulation/robot/joint_velocity` 는 둘 다 있으므로 arm joint 속도 노름으로 A/B.
증강본의 `eef_vel/speed` 는 진단용 별도 리포트.

사용:
    python scripts/contract/validate_boundary_continuity.py \\
        --generated /path/to/generated.hdf5 \\
        --source    /path/to/source.hdf5
    python scripts/contract/validate_boundary_continuity.py --self-check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

ARM_DOF = 5  # gripper 제외
DEFAULT_DT = 1.0 / 30.0  # control dt (sim dt 1/120 × decimation 4)

# arm 상한 import — 정본: src/sim_to_real/tasks/common/utils.py:11
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
try:
    from sim_to_real.tasks.common.utils import SO101_JOINT_TARGET_MAX_VELOCITY
    SLEW_CAP_RAD_S = SO101_JOINT_TARGET_MAX_VELOCITY["shoulder_pan"]  # arm 5축 공통 (5.0 rad/s)
except ImportError:
    # isaaclab 없으면 기본값 사용 (테스트 환경)
    SLEW_CAP_RAD_S = 5.0

# 명령 포화율이 주 판정 지표다 (2026-07-30 실측).
# 처음에는 물리 joint 속도의 "재개 충격"만 봤는데 그 지표가 **오판을 냈다**:
# num_interpolation_steps 0→15 가 물리 지표로는 25.1→27.4(악화)인데
# 명령 신호로는 포화 프레임 1.78%→0.38%(4.7배 개선)였다. 사람이 보는 "확 튐"은
# slew limiter 가 cap 에 붙어 최대속도 램프를 만드는 구간이므로 명령 쪽이 원인에 가깝다.
SATURATION_MARGIN = 0.998  # cap 의 99.8% 이상이면 clip 된 것으로 본다
TARGET_KEYS = ("joint_data/target", "applied_target")  # 증강 · source 레이아웃


def _demos(handle: h5py.File) -> list[str]:
    """demo 이름 목록을 번호순으로 반환."""
    return sorted(handle["data"], key=lambda name: int(name.split("_")[-1]))


def command_saturation(group: h5py.Group, *, dt: float = DEFAULT_DT) -> tuple[int, int, list[int]]:
    """post-slew 명령 target 이 per-joint cap 에 붙은 프레임 수 · 전체 · 프레임 index.

    `joint_data/target` 은 `applied_joint_targets`(slew 적용 후)다. 따라서 값이 cap 에
    정확히 붙는 것은 limiter 가 clip 했다는 직접 증거다.
    """
    key = next((k for k in TARGET_KEYS if k in group), None)
    if key is None:
        raise KeyError(f"명령 target 키가 없다 (찾은 것: {list(group.keys())}); 기대: {TARGET_KEYS}")
    target = np.asarray(group[key])[:, :ARM_DOF]
    rate = np.abs(np.diff(target, axis=0)) / dt
    saturated = (rate >= SLEW_CAP_RAD_S * SATURATION_MARGIN).any(axis=1)
    return int(saturated.sum()), int(len(saturated)), np.nonzero(saturated)[0].tolist()


def _arm_joint_speed(group: h5py.Group) -> np.ndarray:
    """arm 5축 joint 속도 노름 (rad/s) — source·generated 공통 키."""
    qdot = np.asarray(group["states/articulation/robot/joint_velocity"])
    return np.linalg.norm(qdot[:, :ARM_DOF], axis=-1)


def _wrist_roll_speed(group: h5py.Group) -> np.ndarray | None:
    """wrist_roll 축 속도 (rad/s) — 단독 분석용."""
    try:
        qdot = np.asarray(group["states/articulation/robot/joint_velocity"])
        # SO-101 arm joints: shoulder_pan(0), shoulder_lift(1), elbow_flex(2), wrist_flex(3), wrist_roll(4)
        return np.abs(qdot[:, 4])
    except Exception:
        return None


def find_interior_stalls(
    speed: np.ndarray, *, eps: float, min_len: int, edge_skip: int
) -> list[tuple[int, int]]:
    """속도가 eps 아래로 min_len 이상 머무는 내부 구간 [(start, end_exclusive), ...]."""
    low = speed < eps
    lo, hi = edge_skip, len(speed) - edge_skip
    stalls: list[tuple[int, int]] = []
    index = 0
    while index < len(low):
        if not low[index]:
            index += 1
            continue
        start = index
        while index < len(low) and low[index]:
            index += 1
        clipped_start, clipped_end = max(start, lo), min(index, hi)
        if clipped_end - clipped_start >= min_len:
            stalls.append((clipped_start, clipped_end))
    return stalls


def _jump_after(speed: np.ndarray, end: int, dt: float, window: int = 4) -> float:
    """stall 종료 직후 window 프레임 내 최대 가속 크기."""
    # ★ `end` 는 exclusive 다 — 창을 `end` 부터 열면 전이 스텝 자체가 빠진다.
    # 마지막 정지 프레임 → 첫 이동 프레임이 재개 충격의 본체이므로 `end-1` 부터 연다.
    tail = speed[max(end - 1, 0) : end + window]
    if len(tail) < 2:
        return 0.0
    return float(np.abs(np.diff(tail)).max() / dt)


def analyse(path: str, *, eps: float, min_len: int, edge_skip: int, label: str) -> dict:
    """HDF5 분석 — 포화율·stall·재개 가속."""
    sat_frames = sat_total = 0
    sat_positions: list[float] = []
    sat_demos = 0
    lengths: list[int] = []
    moving_speeds: list[np.ndarray] = []
    wrist_roll_speeds: list[np.ndarray] = []

    with h5py.File(path, "r") as handle:
        demos = _demos(handle)
        if not demos:
            raise ValueError(f"{path} 에 demo 가 0개 — 빈 입력은 통과가 아니다")
        for name in demos:
            group = handle["data"][name]
            speed = _arm_joint_speed(group)
            lengths.append(len(speed))
            moving_speeds.append(speed[speed >= eps])

            hits, frames, indices = command_saturation(group)
            sat_frames += hits
            sat_total += frames
            sat_demos += 1 if hits else 0
            sat_positions += [i / frames for i in indices]

            # wrist_roll 별도 추출
            wr_speed = _wrist_roll_speed(group)
            if wr_speed is not None:
                wrist_roll_speeds.append(wr_speed[wr_speed > 0.001])  # 정지 상태 제외

    moving = np.concatenate([s for s in moving_speeds if len(s)]) if moving_speeds else np.zeros(1)
    speed_median = float(np.median(moving)) if len(moving) else 0.0

    wrist_roll_all = np.concatenate([s for s in wrist_roll_speeds if len(s)]) if wrist_roll_speeds else np.zeros(1)

    result = {
        "label": label,
        "path": path,
        "demos": len(lengths),
        "frames_mean": float(np.mean(lengths)) if lengths else 0.0,
        "joint_speed_median": speed_median,
        # ★ 주 판정 지표
        "cmd_saturated_frames": sat_frames,
        "cmd_total_frames": sat_total,
        "cmd_saturation_pct": 100.0 * sat_frames / sat_total if sat_total else 0.0,
        "cmd_demos_with_saturation": sat_demos,
        "cmd_saturation_position_hist": (
            np.histogram(np.asarray(sat_positions), bins=10, range=(0.0, 1.0))[0].tolist()
            if sat_positions
            else [0] * 10
        ),
        # wrist_roll 통계 (source max 0.772 rad/s 기준)
        "wrist_roll_p50": float(np.percentile(wrist_roll_all, 50)) if len(wrist_roll_all) else 0.0,
        "wrist_roll_p95": float(np.percentile(wrist_roll_all, 95)) if len(wrist_roll_all) else 0.0,
        "wrist_roll_p99": float(np.percentile(wrist_roll_all, 99)) if len(wrist_roll_all) else 0.0,
        "wrist_roll_max": float(np.max(wrist_roll_all)) if len(wrist_roll_all) else 0.0,
    }
    return result


def _report(result: dict) -> None:
    """결과를 사람이 읽기 좋게 출력."""
    print(f"\n[{result['label']}] {result['path']}")
    print(f"  demo={result['demos']}  평균 프레임={result['frames_mean']:.0f}")
    print(
        f"  ★ 명령 포화: {result['cmd_saturated_frames']}/{result['cmd_total_frames']} 프레임 "
        f"({result['cmd_saturation_pct']:.2f}%) · 발생 demo {result['cmd_demos_with_saturation']}/{result['demos']}"
    )
    print(f"     에피소드 상대위치 분포(10분위): {result['cmd_saturation_position_hist']}")
    print(
        f"  wrist_roll 속도: p50 {result['wrist_roll_p50']:.3f} · p95 {result['wrist_roll_p95']:.3f} · "
        f"p99 {result['wrist_roll_p99']:.3f} · max {result['wrist_roll_max']:.3f} rad/s"
    )


def cmd_self_check() -> int:
    """명령 포화율 양성·음성 검증."""
    dt_frames = 200
    dt = DEFAULT_DT

    # 양성: 프레임 간 큰 점프로 포화 만들기
    # diff = |target[i+1] - target[i]| / dt 를 포화시키려면
    # 연속된 프레임의 차이가 SLEW_CAP × SATURATION_MARGIN 이상이어야 함
    positive = np.full((dt_frames, ARM_DOF), 0.0)
    # 프레임 99→100에서 점프: rate = 0.9 / (1/30) = 27 rad/s > 4.99 rad/s → 포화
    positive[100:108, :] = SLEW_CAP_RAD_S * 0.9
    hits, frames, indices = command_saturation({"joint_data/target": positive})
    if hits != 7:  # diff[99:106] = 7개 (frame 99→100, 100→101, ..., 105→106 공백 제외)
        # 다시 생각해보니 diff는 연속된 프레임의 차이다
        # target[0:100] = 0, target[100:108] = 0.9이면
        # diff[99:108]에서:
        # - diff[99] = (0.9 - 0) / dt → 포화
        # - diff[100:107] = (0.9 - 0.9) / dt = 0 → 미포화
        # - diff[107] = (0 - 0.9) / dt → 포화
        # 그래서 실제로는 diff[99]와 diff[107]만 포화다... 이건 맞지 않는다
        pass

    # 다시 설계: 매 프레임 연속 포화를 만들려면
    # diff[100:108]이 모두 포화되려면, target이 매 프레임 0.9씩 증가해야 한다
    positive2 = np.full((dt_frames, ARM_DOF), 0.0)
    for i in range(100, 108):
        positive2[i, :] = float(i - 100) * SLEW_CAP_RAD_S * 0.9
    hits2, frames2, indices2 = command_saturation({"joint_data/target": positive2})
    if hits2 != 8 or frames2 != 199:
        print(f"[self-check] FAIL — 양성 8프레임 포화 기대, 실제 {hits2}/{frames2}")
        return 1

    # 음성: 모두 정상 범위 (cap 미만) → 포화 0
    negative = np.full((dt_frames, ARM_DOF), SLEW_CAP_RAD_S * 0.3)
    hits_n, frames_n, _ = command_saturation({"joint_data/target": negative})
    if hits_n != 0:
        print(f"[self-check] FAIL — 음성 0포화 기대, 실제 {hits_n}")
        return 1

    print("[self-check] PASS — 양성 1 · 음성 1")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generated", help="증강 HDF5")
    parser.add_argument("--source", help="source HDF5 (main cuRobo SM) — 비교 기준")
    parser.add_argument("--stall-eps", type=float, default=0.05, help="정지 판정 임계 (rad/s)")
    parser.add_argument("--stall-min-len", type=int, default=3, help="정지 최소 길이 (프레임)")
    parser.add_argument("--edge-skip", type=int, default=30, help="앞뒤 제외 프레임")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    if args.self_check:
        return cmd_self_check()
    if not args.generated:
        parser.error("--generated 가 필요하다 (또는 --self-check)")

    kwargs = dict(eps=args.stall_eps, min_len=args.stall_min_len, edge_skip=args.edge_skip)
    generated = analyse(args.generated, label="generated", **kwargs)
    _report(generated)

    if not args.source:
        print("\n[warn] --source 미지정 — 기준선 없이는 PASS/FAIL 판정을 하지 않는다")
        return 0

    source = analyse(args.source, label="source", **kwargs)
    _report(source)

    print("\n=== 판정 ===")
    # ★ 판정은 명령 포화율로 한다. source(main SM)는 이음매가 없어 0% 여야 한다.
    src_pct = source["cmd_saturation_pct"]
    gen_pct = generated["cmd_saturation_pct"]
    print(f"  ★ 명령 포화율: source {src_pct:.2f}% → generated {gen_pct:.2f}%")
    limit = max(src_pct, 0.0) + 0.10
    verdict = gen_pct <= limit
    print(f"  임계 = source + 0.10%p = {limit:.2f}% → {'PASS' if verdict else 'FAIL'}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
