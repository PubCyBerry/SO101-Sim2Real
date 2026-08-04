#!/usr/bin/env python3
"""place 성공을 **종료 시점**에 재검증한다 — OR-latch 오탐(false success) 검출.

## 왜 필요한가 (실측 근거, 2026-07-30)

공식 SkillGen 생성기는 성공을 에피소드 전체에 대해 **OR-latch** 한다:
한 프레임이라도 조건을 만족하면 성공으로 굳고 **종료 시점에 재확인하지 않는다**.
"큐브가 그릇 반경을 잠깐 지나갔다가 팔에 밀려 나감" 이 성공으로 집계된다.

50 demo 실측 — 4건이 오탐이었다 (그릇 엎음, 밀고 엎음 등).

정상 demo 는 조건이 **끝까지 유지**되고, 오탐은 중간 구간에서만 True 다.
이 차이가 판별식이다.

사용:
    python scripts/contract/validate_place_success.py <generated.hdf5>
    python scripts/contract/validate_place_success.py --self-check
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_CFG = REPO_ROOT / "src" / "sim_to_real" / "tasks" / "pick_cube" / "pick_cube_env_cfg.py"

# 그릇 교란 임계 — 정상 46 demo 는 XY ≤3.4mm · tilt 0° 였고 오탐 4건은 6.1~32.7mm · 최대 41°.
# 정상 최대의 약 2배로 잡는다.
BOWL_SHIFT_LIMIT_M = 0.008
BOWL_TILT_LIMIT_DEG = 10.0


def literal_constants(py_path: Path) -> dict[str, object]:
    """모듈 최상위 리터럴 상수만 AST 로 읽는다 (import 불요, isaaclab 미필요)."""
    out: dict[str, object] = {}
    for node in ast.parse(py_path.read_text(encoding="utf-8")).body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        target = node.targets[0] if isinstance(node, ast.Assign) else node.target
        if not isinstance(target, ast.Name) or node.value is None:
            continue
        try:
            out[target.id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
    return out


def _quat_local_z(quat_wxyz: np.ndarray) -> np.ndarray:
    """wxyz 쿼터니언 → 로컬 +z 축의 world 성분."""
    w, x, y, z = quat_wxyz
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def analyse_demo(group: h5py.Group, *, radius: float, height_range: tuple[float, float]) -> dict:
    """place 조건을 프레임별로 재현하고 종료 시점 + 그릇 교란으로 재검증.

    규약: `src/sim_to_real/tasks/common/mdp/terminations.py::task_done` 와 같은 기하를
    numpy로 구현한다(isaaclab 불필요). 컨테이너 로컬 프레임 기준 판정.
    """
    cube = np.asarray(group["states/rigid_object/Cube1/root_pose"])
    bowl = np.asarray(group["states/rigid_object/Bowl/root_pose"])
    distance = np.hypot(cube[:, 0] - bowl[:, 0], cube[:, 1] - bowl[:, 1])
    inside = (
        (distance < radius)
        & (cube[:, 2] > bowl[:, 2] + height_range[0])
        & (cube[:, 2] < bowl[:, 2] + height_range[1])
    )
    shift = float(np.linalg.norm(bowl[-1, :2] - bowl[0, :2]))
    # 그릇 기울기: 로컬 +z 가 world +z 와 이루는 각도
    local_z = _quat_local_z(bowl[-1, 3:])
    tilt = float(np.degrees(np.arccos(np.clip(local_z[2], -1.0, 1.0))))
    fired = np.nonzero(inside)[0]
    return {
        "recorded_success": bool(group.attrs.get("success", False)),
        "frames": int(len(inside)),
        "inside_final": bool(inside[-1]),
        "inside_first": int(fired[0]) if len(fired) else -1,
        "inside_last": int(fired[-1]) if len(fired) else -1,
        "final_distance_m": float(distance[-1]),
        "bowl_shift_m": shift,
        "bowl_tilt_deg": tilt,
        # 판정: 종료 시점에도 그릇 안 + 그릇이 교란되지 않음
        "verified": bool(inside[-1]) and shift <= BOWL_SHIFT_LIMIT_M and tilt <= BOWL_TILT_LIMIT_DEG,
    }


def cmd_validate(args: argparse.Namespace) -> int:
    constants = literal_constants(ENV_CFG)
    missing = [k for k in ("BOWL_SUCCESS_RADIUS", "BOWL_HEIGHT_RANGE") if k not in constants]
    if missing:
        raise KeyError(f"{ENV_CFG} 에서 상수를 못 읽었다: {missing} — 이름이 바뀐 것 아닌지 확인하라")
    radius = float(constants["BOWL_SUCCESS_RADIUS"])
    height_range = tuple(float(v) for v in constants["BOWL_HEIGHT_RANGE"])  # type: ignore[arg-type]
    print(f"[cfg] radius={radius:.3f}m height_range={height_range} (pick_cube_env_cfg.py AST)")
    print(f"[cfg] bowl 교란 임계: shift≤{BOWL_SHIFT_LIMIT_M * 1000:.0f}mm tilt≤{BOWL_TILT_LIMIT_DEG:.0f}°")

    with h5py.File(args.hdf5, "r") as handle:
        names = sorted(handle["data"], key=lambda n: int(n.split("_")[-1]))
        if not names:
            print("[FAIL] demo 0개 — 빈 입력은 통과가 아니다")
            return 1
        results = {
            name: analyse_demo(handle["data"][name], radius=radius, height_range=height_range)
            for name in names
        }

    false_success = [n for n, r in results.items() if r["recorded_success"] and not r["verified"]]
    verified = [n for n, r in results.items() if r["verified"]]

    print(f"\n{'demo':>9} {'기록':>5} {'검증':>5} {'최종거리':>9} {'bowlΔ':>7} {'tilt':>6}  조건 True 구간")
    for name, r in results.items():
        mark = "" if r["recorded_success"] == r["verified"] else "  ← 오탐"
        span = f"f{r['inside_first']}~f{r['inside_last']}" if r["inside_first"] >= 0 else "없음"
        print(
            f"{name:>9} {str(r['recorded_success']):>5} {str(r['verified']):>5} "
            f"{r['final_distance_m'] * 1000:>8.1f}mm {r['bowl_shift_m'] * 1000:>6.1f}mm "
            f"{r['bowl_tilt_deg']:>5.1f}°  {span}{mark}"
        )

    total = len(results)
    print(f"\n[place] demo={total} 기록성공={sum(r['recorded_success'] for r in results.values())} "
          f"검증통과={len(verified)} 오탐={len(false_success)}")
    if false_success:
        print(f"[place] 오탐 목록: {false_success}")

    if args.fix_success_attr and false_success:
        # 공식 생성기는 성공을 OR-latch 하고 되돌릴 훅이 없다(data_generator.py:886).
        # env 쪽 termination 을 고쳐도 latch 자체는 못 막으므로, 데이터로 들어온 뒤
        # `success` attr 을 검증 결과로 정정하는 것이 유일한 차단점이다.
        with h5py.File(args.hdf5, "r+") as handle:
            for name in false_success:
                handle["data"][name].attrs["success"] = False
                handle["data"][name].attrs["place_unverified"] = True
        print(f"[place] success attr 정정: {len(false_success)}개 → False (+place_unverified=True)")
        with h5py.File(args.hdf5, "r") as handle:
            still = [n for n in false_success if bool(handle["data"][n].attrs.get("success", False))]
        if still:
            raise AssertionError(f"success attr 정정 실패: {still}")
        print("[place] 정정 확인 PASS")
        return 0

    print("[place] " + ("PASS" if not false_success else "FAIL"))
    return 1 if false_success else 0


def cmd_self_check() -> int:
    """판정식 양성·음성 검증."""
    frames = 60
    bowl = np.zeros((frames, 7), dtype=np.float64)
    bowl[:, 2] = 0.715
    bowl[:, 3] = 1.0  # 무회전
    radius, height_range = 0.06, (0.005, 0.12)

    def demo(cube_xy_final, cube_xy_mid=(0.0, 0.0), bowl_shift=0.0, tilt_quat=None):
        cube = np.zeros((frames, 7), dtype=np.float64)
        cube[:, 3] = 1.0
        cube[:, 2] = 0.728
        cube[:30, 0], cube[:30, 1] = cube_xy_mid
        cube[30:, 0], cube[30:, 1] = cube_xy_final
        b = bowl.copy()
        b[30:, 0] = bowl_shift
        if tilt_quat is not None:
            b[30:, 3:] = tilt_quat
        return cube, b

    class Fake(dict):
        attrs: dict = {"success": True}

        def __init__(self, cube, bowl):
            super().__init__(
                {
                    "states/rigid_object/Cube1/root_pose": cube,
                    "states/rigid_object/Bowl/root_pose": bowl,
                }
            )
            self.attrs = {"success": True}

    # 양성: 끝까지 그릇 안 + 그릇 정지 → verified
    good = analyse_demo(Fake(*demo((0.004, 0.0))), radius=radius, height_range=height_range)
    if not good["verified"]:
        print(f"[self-check] FAIL — 정상 케이스를 통과시키지 못했다: {good}")
        return 1

    # 음성 1: 중간엔 안, 끝엔 밖 (OR-latch 오탐 시나리오)
    transient = analyse_demo(Fake(*demo((0.20, 0.0))), radius=radius, height_range=height_range)
    if transient["verified"] or transient["inside_first"] != 0 or transient["inside_final"]:
        print(f"[self-check] FAIL — 일시적 성공을 잡지 못했다: {transient}")
        return 1

    # 음성 2: 끝까지 안이지만 그릇이 30mm 밀림
    pushed = analyse_demo(Fake(*demo((0.02, 0.0), bowl_shift=0.030)), radius=radius, height_range=height_range)
    if pushed["verified"]:
        print(f"[self-check] FAIL — 그릇 밀림을 잡지 못했다: {pushed}")
        return 1

    # 음성 3: 끝까지 안이지만 그릇이 90° 넘어짐 (로컬 z 가 world x 로) → tilt 90°
    tipped = analyse_demo(
        Fake(*demo((0.004, 0.0), tilt_quat=(0.7071, 0.0, 0.7071, 0.0))),
        radius=radius,
        height_range=height_range,
    )
    if tipped["verified"]:
        print(f"[self-check] FAIL — 그릇 넘어짐을 잡지 못했다: {tipped}")
        return 1

    print("[self-check] PASS — 양성 1 · 음성 3")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hdf5", nargs="?", help="검사할 generated HDF5")
    parser.add_argument(
        "--fix-success-attr",
        action="store_true",
        help="오탐 demo 의 success attr 을 False 로 정정한다(HDF5 in-place, demo 는 보존)",
    )
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        return cmd_self_check()
    if not args.hdf5:
        parser.error("hdf5 경로가 필요하다 (또는 --self-check)")
    return cmd_validate(args)


if __name__ == "__main__":
    sys.exit(main())
