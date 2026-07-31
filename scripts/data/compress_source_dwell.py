"""IsaacLab HDF5 의 **짧은 전 축 정지(dwell)** 구간을 압축한다 — env 불요.

## 왜

cuRobo SM 은 6-phase 를 이어 붙이는데, phase 경계마다 앞 phase 의 끝과 뒤 phase 의 시작이
사실상 같은 자세라 **전 축이 멈춘 짧은 구간**이 생긴다. 증강본에서 "중간에 한 번씩 멈칫"으로
보이는 게 이것이다(실측: 에피소드당 4구간 25.6프레임, source 와 증강본이 동률 → source 고유).

긴 정지는 **기능적**이다 — 폐합 후 접촉 안정화(`GRASP_HOLD_STEPS`), release 전 그릇 상공 정지
(`SETTLE_STEPS`). 그건 성능을 위해 **일부러 늘린 값**이라(hold 5→15 로 grasp 개선) 건드리면
파지 성공률이 떨어진다. 그래서 **`--max-run` 이하의 짧은 구간만** 지운다.

## 어떻게

프레임별 |Δaction| 을 보고 전 축이 `--eps` rad/step 미만인 연속 구간을 찾아, 길이가
`--max-run` 이하면 그 구간을 **버린다**(구간 끝 프레임 1개는 남겨 자세 연속성 유지).
에피소드 머리 `--head` · 꼬리 `--tail` 프레임은 **의도된 pre-roll/post-hold** 라 보존한다.

demo 그룹의 **첫 차원이 T 인 모든 dataset** 을 같은 마스크로 자르므로 스트림 정합이 유지된다.
`initial_state` 같은 non-per-step 데이터는 그대로 복사한다.

⚠ **주석(annotate) 전에** 돌려라. 그러면 주석이 `datagen_info` 를 압축된 궤적 위에서 새로
만들어 subtask 경계까지 자동 정합된다. 주석 후 파일에 돌리면 신호 배열을 직접 맞춰야 한다.

사용:
    python scripts/data/compress_source_dwell.py --in src.hdf5 --out src_smooth.hdf5
    python scripts/data/compress_source_dwell.py --self-test
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np

DEFAULT_EPS = 0.0007       # rad/step ≈ 0.02 rad/s @30 Hz
#: 이 길이 이하 정지만 압축. 실측 스윕(source 16 demo, 5944 프레임):
#:   max_run  5 → −96 (1.6%) · 8 → −202 (3.4%) · 12 → −340 (5.7%) · 20 → −340 (포화)
#: 12 에서 포화 = 가장 긴 all-axis 정지가 ≤12 프레임이다. 8 은 phase junction 정지와 hold 앞부분을
#: 걷어내면서 접촉 정착 핵심(`GRASP_HOLD_STEPS`)은 남긴다. 12 로 올리면 dwell 이 사라지지만
#: 정착 시간을 통째로 없애 파지 성공률과 맞바꾸는 것이므로 A/B 없이 올리지 말 것.
DEFAULT_MAX_RUN = 8
DEFAULT_HEAD, DEFAULT_TAIL = 60, 30   # pre-roll 2 s / post-hold 1 s


def dwell_keep_mask(actions: np.ndarray, eps: float, max_run: int,
                    head: int, tail: int) -> np.ndarray:
    """(T,D) action → 유지할 프레임 bool 마스크 (T,)."""
    n_frames = len(actions)
    keep = np.ones(n_frames, dtype=bool)
    if n_frames <= head + tail + 2:
        return keep

    delta = np.abs(np.diff(actions, axis=0)).max(axis=1)   # (T-1,)
    stalled = delta < eps

    start = None
    for index in range(len(stalled) + 1):
        active = index < len(stalled) and stalled[index]
        if active and start is None:
            start = index
        elif not active and start is not None:
            length = index - start
            # 구간 [start+1 .. index] 프레임이 앞 프레임과 같다는 뜻. 마지막 1개만 남긴다.
            if length <= max_run and start + 1 >= head and index <= n_frames - tail:
                keep[start + 1: index] = False
            start = None
    return keep


def compress(path_in: str, path_out: str, eps: float, max_run: int,
             head: int, tail: int) -> None:
    with h5py.File(path_in, "r") as src, h5py.File(path_out, "w") as dst:
        data_in = src["data"]
        data_out = dst.create_group("data")
        for key, value in data_in.attrs.items():
            data_out.attrs[key] = value

        total_before = total_after = 0
        for name in sorted(data_in.keys(), key=lambda n: int(n.split("_")[-1])):
            demo_in = data_in[name]
            # ★기준 스트림은 `applied_target`(**슬루 적용 후** 명령) 이다. `actions`(raw 명령)로
            #   재면 hold 구간에서도 raw 가 미세하게 흔들려 정지로 안 잡힌다 — 실측: 같은 파일에서
            #   raw 기준 96 프레임(1.6%)만 검출, applied_target 기준 측정치는 ~410 프레임이었다.
            stream = "applied_target" if "applied_target" in demo_in else "actions"
            actions = np.asarray(demo_in[stream], dtype=np.float64)
            keep = dwell_keep_mask(actions, eps, max_run, head, tail)
            n_before, n_after = len(actions), int(keep.sum())
            total_before += n_before
            total_after += n_after

            demo_out = data_out.create_group(name)
            for key, value in demo_in.attrs.items():
                demo_out.attrs[key] = value
            demo_out.attrs["num_samples"] = n_after

            def copy(source: h5py.Group, target: h5py.Group) -> None:
                for key, item in source.items():
                    if isinstance(item, h5py.Group):
                        copy(item, target.create_group(key))
                    elif item.shape and item.shape[0] == n_before:
                        target.create_dataset(key, data=np.asarray(item)[keep],
                                              compression="lzf")
                    else:
                        target.create_dataset(key, data=np.asarray(item))

            copy(demo_in, demo_out)
            print(f"[dwell] {name}: {n_before} → {n_after} 프레임 (−{n_before - n_after})")

        print(f"[dwell] 합계 {total_before} → {total_after} 프레임 "
              f"(−{total_before - total_after}, {100 * (1 - total_after / total_before):.1f}%)")


def _self_test() -> None:
    """짧은 정지는 지우고 긴 정지·머리·꼬리는 보존하는지."""
    head, tail, max_run = 3, 2, 4
    # 머리3 + 이동2 + 짧은정지3 + 이동2 + 긴정지8 + 이동2 + 꼬리2
    parts = [np.zeros(3), np.arange(1, 3), np.full(3, 2.0), np.arange(3, 5),
             np.full(8, 4.0), np.arange(5, 7), np.full(2, 6.0)]
    actions = np.concatenate(parts).reshape(-1, 1)
    keep = dwell_keep_mask(actions, eps=0.5, max_run=max_run, head=head, tail=tail)
    dropped = int((~keep).sum())
    assert dropped > 0, "짧은 정지를 하나도 못 지웠다"
    # 긴 정지(8f) 는 보존돼야 한다 → 지운 수가 짧은정지(3f) 구간 범위 안이어야 한다
    assert dropped <= max_run, f"기능적 hold 까지 지웠다(dropped={dropped})"
    assert keep[:head].all(), "머리(pre-roll)를 지웠다"
    assert keep[-tail:].all(), "꼬리(post-hold)를 지웠다"
    # 단조 자세 열은 그대로 남아야 한다(자세 연속성)
    assert actions[keep][0, 0] == 0.0 and actions[keep][-1, 0] == 6.0
    print("compress_source_dwell self-test PASS "
          f"(총 {len(actions)} → {int(keep.sum())} 프레임, dropped={dropped})")


def main() -> None:
    parser = argparse.ArgumentParser(description="짧은 dwell 구간 압축 (env-free)")
    parser.add_argument("--in", dest="path_in", help="입력 HDF5")
    parser.add_argument("--out", dest="path_out", help="출력 HDF5")
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS,
                        help="이 값(rad/step) 미만이면 정지로 본다")
    parser.add_argument("--max-run", type=int, default=DEFAULT_MAX_RUN,
                        help="이 길이 이하 정지만 압축(그보다 길면 기능적 hold 로 보존)")
    parser.add_argument("--head", type=int, default=DEFAULT_HEAD, help="보존할 선두 프레임")
    parser.add_argument("--tail", type=int, default=DEFAULT_TAIL, help="보존할 말미 프레임")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return
    if not args.path_in or not args.path_out:
        parser.error("--in / --out 필요 (또는 --self-test)")
    compress(args.path_in, args.path_out, args.eps, args.max_run, args.head, args.tail)


if __name__ == "__main__":
    main()
