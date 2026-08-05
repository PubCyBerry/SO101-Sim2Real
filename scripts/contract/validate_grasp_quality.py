#!/usr/bin/env python3
"""파지 기하 게이트 — main SM 의 `_gate_candidate` 를 HDF5 사후 계산으로 재현한다.

## 무엇을 재는가

폐합 시점에 **fixed jaw 내측면 중심**이 **큐브 face 중심**에서 얼마나 벗어났는지를
(normal, tangent, height, face_angle) 로 분해한다.

main SM 은 이 값으로 IK 해를 **거부하고 다른 후보로 넘어간다** — 그것이 main 이 100% 인
실제 기제다. SkillGen 에는 그 게이트·거부·피드백이 전부 없어서 오차 큰 시도를 그대로 실행한다.

## ★ 폐합 프레임 검출 — 이 한 줄이 두 번의 오진을 만들었다

source 초기 자세의 그리퍼는 이미 **닫힘**이다. 그래서 "처음 닫히는 프레임" 을 그대로 찾으면
**index 0(홈 자세)** 이 잡힌다. 올바른 규칙 = **열림(> 임계)을 거친 뒤 처음 닫히는 프레임**.

## 용도

**진단용이다.** 이미 `validate_place_success.py` 를 통과한 성공 데모를 이 게이트로 또
걸러내면 안 된다 — 큐브가 그릇에 제대로 들어간 데모는 파지 기하가 임계를 넘더라도 좋은
데이터다.

사용:
    python scripts/contract/validate_grasp_quality.py <hdf5> [--want-success/--want-failure]
    python scripts/contract/validate_grasp_quality.py --self-check
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

# ★ 진단용 분포 기준 (source p90 캘리브레이션, 게이트 아님).
# 플래너의 실제 게이트는 so101_contract.grasp_manifold 의 E_TANGENT_MAX/E_HEIGHT_MAX 상수다.
# ponytail: 단위는 미터; 키는 n/t/h (grasp_manifold 정본 규약). 경계에서 mm 변환.
E_TANGENT_P90_M = 0.0184  # source p90, 미터 단위
E_HEIGHT_P90_M = 0.0149
FACE_ANGLE_P90_DEG = 40.0  # 판별력 없음, main 값 그대로

# 그리퍼 열림 판정 임계. source 는 radian(열림 +1.27 / 닫힘 -0.08), 증강은 binary(±1).
GRIPPER_OPEN_THRESHOLD = {"joint": 0.5, "binary": 0.0}


def _rotation_from_quat_wxyz(quat) -> np.ndarray:
    """wxyz 쿼터니언 → 3x3 회전행렬."""
    w, x, y, z = (float(v) for v in quat)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def close_frame(gripper: np.ndarray, open_threshold: float) -> int | None:
    """열림을 거친 뒤 처음 닫히는 프레임. 없으면 None.

    ★ 단순히 "처음 닫히는 프레임" 을 찾으면 안 된다 — source 초기 자세가 이미 닫힘이라
    index 0 이 잡힌다. 이 실수가 게이트 재현을 두 번 무너뜨렸다.
    """
    opened = np.nonzero(gripper > open_threshold)[0]
    if not opened.size:
        return None
    after = np.nonzero(gripper[opened[0]:] < open_threshold)[0]
    return int(opened[0] + after[0]) if after.size else None


def gate_ok(metrics: dict[str, float]) -> bool:
    """진단용 분포 기준 통과 판정 (플래너 게이트 아님).

    키: n, t, h (grasp_manifold 정본)
    단위: 미터
    """
    return (
        abs(metrics.get("t", 0)) <= E_TANGENT_P90_M
        and abs(metrics.get("h", 0)) <= E_HEIGHT_P90_M
        and abs(metrics.get("face_angle", 0)) <= FACE_ANGLE_P90_DEG
    )


def analyse(path: str, *, want_success: bool) -> list[tuple[str, dict[str, float], bool]]:
    """HDF5 에서 파지 기하 기준 통과/불통과 판정."""
    try:
        from so101_contract.eef_kinematics import SO101EndEffectorKinematics
    except ImportError as e:
        raise ImportError(
            f"so101_contract.eef_kinematics 을 import 할 수 없다. "
            f"C1 커밋이 필요: {e}"
        )

    try:
        from so101_contract.grasp_manifold import grasp_face_error
        from so101_contract.curobo_frames import usd_to_urdf
        from so101_contract.grasp_geometry import FIXED_JAW_CLEAR_MIN, FIXED_JAW_CLEAR_MAX
    except ImportError as e:
        raise ImportError(f"C1 심볼 import 실패: {e}")

    # ponytail: cube_half 는 인자로 받아야 함 (크기 DR 대응)
    # 테스트용 기본값만 사용 (실제 HDF5는 CLI 인자/메타에서 읽어야 함)
    cube_half_m = 0.020  # 40mm 기본

    try:
        kinematics = SO101EndEffectorKinematics.from_files(
            REPO_ROOT / "assets" / "robots" / "urdf" / "so_arm101.urdf",
            REPO_ROOT / "assets" / "robots" / "so101.yml",
        )
    except Exception as e:
        raise RuntimeError(f"SO101EndEffectorKinematics 초기화 실패: {e}")

    rows: list[tuple[str, dict[str, float], bool]] = []

    with h5py.File(path, "r") as handle:
        group = handle["data"]
        for name in sorted(group, key=lambda n: int(n.split("_")[-1])):
            demo = group[name]
            if bool(demo.attrs.get("success", False)) != want_success:
                continue
            joint_key = next(
                (k for k in ("states/articulation/robot/joint_position", "obs_x/joint_pos") if k in demo),
                None,
            )
            if joint_key is None:
                raise KeyError(f"{name}: joint 상태 키가 없다 (기대: states/... 또는 obs_x/joint_pos)")
            actions = np.asarray(demo["actions"])
            kind = "binary" if actions.shape[-1] == 8 else "joint"
            index = close_frame(actions[:, -1], GRIPPER_OPEN_THRESHOLD[kind])
            if index is None:
                print(f"  [warn] {name}: 열림→닫힘 전이가 없다 — 제외")
                continue

            joints = np.asarray(demo[joint_key])
            cube = np.asarray(demo["states/rigid_object/Cube1/root_pose"])
            root = np.asarray(demo["states/articulation/robot/root_pose"])
            tcp = kinematics.forward_matrices(joints[index : index + 1, :5])[0]
            root_rot = _rotation_from_quat_wxyz(root[0, 3:])
            cube_urdf = np.asarray(usd_to_urdf(root_rot.T @ (cube[index, :3] - root[index, :3])))
            metrics = grasp_face_error(
                tcp[:3, 3], tcp[:3, :3], cube_urdf, cube_half=cube_half_m
            )
            rows.append((name, metrics, gate_ok(metrics)))
    return rows


def cmd_validate(args: argparse.Namespace) -> int:
    """HDF5 파일 검증 (진단용)."""
    try:
        rows = analyse(args.hdf5, want_success=not args.want_failure)
    except Exception as e:
        print(f"[FAIL] 분석 중 오류: {e}")
        return 1

    if not rows:
        print("[FAIL] 평가할 demo 0개 — 빈 입력은 통과가 아니다")
        return 1
    # ponytail: 진단용 분포 기준만 리포트 (플래너 게이트는 단일 소스 grasp_manifold)
    print(f"[cfg] 진단 분포(source p90): |t|<={E_TANGENT_P90_M*1000:.1f}mm "
          f"|h|<={E_HEIGHT_P90_M*1000:.1f}mm |face|<={FACE_ANGLE_P90_DEG:.0f}° (참고용)")
    print(f"\n{'demo':>9} {'n(mm)':>8} {'t(mm)':>8} {'h(mm)':>8} {'face':>8}  분포기준")
    for name, metrics, ok in rows:
        # 키: n, t, h (미터) → mm으로 변환
        e_n = metrics.get("n", 0) * 1000
        e_t = metrics.get("t", 0) * 1000
        e_h = metrics.get("h", 0) * 1000
        face = metrics.get("face_angle", 0)
        print(f"{name:>9} {e_n:+8.2f} {e_t:+8.2f} {e_h:+8.2f} "
              f"{face:+8.2f}  {'PASS' if ok else 'FAIL'}")
    array = np.array([
        [m.get("n", 0) * 1000, m.get("t", 0) * 1000, m.get("h", 0) * 1000, m.get("face_angle", 0)]
        for _, m, _ in rows
    ])
    passed = sum(1 for _, _, ok in rows if ok)
    print(f"\n[grasp] demo={len(rows)} 분포기준통과={passed} ({100 * passed / len(rows):.1f}%)")
    for i, key in enumerate(("n", "t", "h", "face")):
        unit = "mm" if i < 3 else "°"
        print(f"  {key:>5}: mean {array[:, i].mean():+7.2f}{unit}  std {array[:, i].std():6.2f}  "
              f"|p90| {np.percentile(np.abs(array[:, i]), 90):6.2f}{unit}")
    return 0


def cmd_self_check() -> int:
    """기하 계산 양성·음성 검증 + 폐합 프레임 검출 회귀 방지 (★)."""
    try:
        from so101_contract.grasp_manifold import grasp_face_error
        from so101_contract.grasp_geometry import FIXED_INNER_CENTER
    except ImportError as e:
        print(f"[self-check] FAIL — 필수 심볼 import 실패: {e}")
        return 1

    fixed_inner = np.asarray(FIXED_INNER_CENTER, dtype=np.float64)
    cube_half = 0.020  # 40mm 기본
    identity = np.eye(3)

    # 양성: fixed_inner 가 face 중심에 정확히 놓이는 tcp 위치를 역산 → 오차 0 (미터)
    normal = np.array([1.0, 0.0, 0.0])
    cube = np.array([0.30, 0.0, 0.05])
    tcp = cube + cube_half * normal - fixed_inner
    metrics = grasp_face_error(tcp, identity, cube, cube_half=cube_half)
    # 키: n, t, h (미터)
    if max(abs(metrics.get("n", 0)), abs(metrics.get("t", 0)), abs(metrics.get("h", 0))) > 1e-6:
        print(f"[self-check] FAIL — 완전 정합인데 오차가 남았다: {metrics}")
        return 1
    if not gate_ok(metrics):
        print(f"[self-check] FAIL — 완전 정합이 분포기준을 통과하지 못했다: {metrics}")
        return 1

    # 음성 1: tangent 로 0.030m 밀면 t 0.030m → 기각
    off = grasp_face_error(tcp + np.array([0.0, 0.030, 0.0]), identity, cube, cube_half=cube_half)
    if abs(off.get("t", 0) - 0.030) > 1e-6 or gate_ok(off):
        print(f"[self-check] FAIL — tangent 30mm 이탈을 잡지 못했다: {off}")
        return 1

    # 음성 2: 높이로 0.020m 밀면 h 0.020m → 기각 (p90=0.0149m)
    high = grasp_face_error(tcp + np.array([0.0, 0.0, 0.020]), identity, cube, cube_half=cube_half)
    if abs(high.get("h", 0) - 0.020) > 1e-6 or gate_ok(high):
        print(f"[self-check] FAIL — height 20mm 이탈을 잡지 못했다: {high}")
        return 1

    # 음성 3: ★ 폐합 프레임 검출 회귀 — 초기 닫힘 상태를 index 0으로 잡으면 안 됨
    gripper = np.array([-0.17, -0.17, 1.2, 1.2, 1.2, -0.08, -0.08])
    got = close_frame(gripper, 0.5)
    if got != 5:
        print(f"[self-check] FAIL — 폐합 프레임 5 기대, {got} (초기 닫힘 오진)")
        return 1

    # 음성 4: 한 번도 열리지 않으면 None
    if close_frame(np.array([-0.17, -0.17, -0.17]), 0.5) is not None:
        print("[self-check] FAIL — 열림이 없는데 프레임 반환")
        return 1

    print("[self-check] PASS — 양성 1 · 음성 4")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hdf5", nargs="?", help="검사할 HDF5 (source 또는 generated)")
    parser.add_argument("--want-failure", action="store_true", help="success=False demo 를 평가한다")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        return cmd_self_check()
    if not args.hdf5:
        parser.error("hdf5 경로가 필요하다 (또는 --self-check)")
    return cmd_validate(args)


if __name__ == "__main__":
    sys.exit(main())
