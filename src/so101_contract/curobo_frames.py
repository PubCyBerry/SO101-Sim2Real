"""cuRobo 계획 프레임·충돌 씬 기하 — **단일 소스**.

`scripts/cuRobo/curobo_batch_planner.py`(pick-place SM planner)와
`src/sim_to_real/datagen/mimic/curobo_planner.py`(SkillGen motion planner)가 **같은 값**을 써야
하는 것만 모았다. 둘이 갈리면 같은 씬을 서로 다르게 피해 계획이 조용히 어긋난다.

★ 이 모듈이 `so101_contract`(isaaclab 무의존 순수 패키지)에 있는 이유: SM planner 는 kit 부팅
없이 `/isaac-sim/python.sh` 로 단독 실행되는 프로세스라 `sim_to_real` 를 import 할 수 없다
(`sim_to_real/__init__` → `tasks` → isaaclab → `pxr`, kit 부팅 전엔 없음). 여기 두면 planner
프로세스·sim 프로세스·Windows 실기기 어디서나 같은 값을 읽는다.

두 축이 들어 있다.

1. **프레임 정합** — sim USD(`base_link` = `so101_new_calib` 규약) ↔ cuRobo URDF(`so_arm101`).
   두 체인의 `shoulder_pan` 프레임 일치 조건에서 ``T(urdf←usd) = Rz(90°) + BASE_T``.
   미보정 시 ~3 cm 빗나간다. 위치만 필요하면 :func:`usd_to_urdf`, 회전까지 필요하면
   :data:`T_URDF_FROM_USD`(4×4).

2. **충돌 씬 기하** — 책상은 obstacle 로 넣지 않고(로봇이 상판 위에 장착돼 base 구가 상판 안 →
   전 plan start-collision), 그릇은 **hollow rim ring**(오목 형상을 solid convex 로 넣으면 내부
   빈 공간이 허위충돌) 으로 근사한다. 두 결정 모두 실측으로 얻은 것이라 재발명 금지.
"""

from __future__ import annotations

import math

import numpy as np

# 책상 상판 z(base_link 프레임) 실측 단일 소스 — grasp 기하와 공용.
from so101_contract.grasp_geometry import TABLE_TOP_BASE

__all__ = [
    "BASE_YAW",
    "BASE_T",
    "T_URDF_FROM_USD",
    "T_USD_FROM_URDF",
    "TABLE_TOP",
    "CUBE_DIMS",
    "BOWL_RING_N",
    "BOWL_RING_RC",
    "BOWL_RING_H",
    "BOWL_RING_DIMS",
    "rotz",
    "usd_to_urdf",
    "urdf_to_usd",
    "transform_pose_usd_to_urdf",
    "bowl_ring",
    "assert_bowl_ring_sealed",
]

# ═══ 프레임 정합 (sim USD base_link ↔ cuRobo URDF solver) ═══════════════════════════
BASE_YAW = 90.0
BASE_T = (0.01576, -0.02079, -0.03248)


def rotz(x: float, y: float, deg: float) -> tuple[float, float]:
    """XY 를 z 축으로 ``deg`` 회전."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return c * x - s * y, s * x + c * y


def _rz_matrix(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


#: sim USD base_link 프레임 → cuRobo URDF solver 프레임 homogeneous 변환(4×4).
T_URDF_FROM_USD = np.eye(4, dtype=np.float64)
T_URDF_FROM_USD[:3, :3] = _rz_matrix(BASE_YAW)
T_URDF_FROM_USD[:3, 3] = BASE_T

#: 역변환.
T_USD_FROM_URDF = np.linalg.inv(T_URDF_FROM_USD)


def usd_to_urdf(p) -> tuple[float, float, float]:
    """sim USD base_link 좌표(위치) → cuRobo URDF solver 좌표: ``Rz(90) + BASE_T``."""
    x, y = rotz(p[0], p[1], BASE_YAW)
    return (x + BASE_T[0], y + BASE_T[1], p[2] + BASE_T[2])


def urdf_to_usd(p) -> tuple[float, float, float]:
    """:func:`usd_to_urdf` 의 역."""
    x, y = rotz(p[0] - BASE_T[0], p[1] - BASE_T[1], -BASE_YAW)
    return (x, y, p[2] - BASE_T[2])


def transform_pose_usd_to_urdf(pose: np.ndarray) -> np.ndarray:
    """USD base_link 프레임 SE(3) ``(...,4,4)`` → URDF solver 프레임 ``(...,4,4)``."""
    pose_arr = np.asarray(pose, dtype=np.float64)
    if pose_arr.shape[-2:] != (4, 4):
        raise ValueError(f"pose must end in (4,4), got {pose_arr.shape}")
    return T_URDF_FROM_USD @ pose_arr


# ═══ 충돌 씬 기하 ═══════════════════════════════════════════════════════════════════
#: 책상 상판 z (URDF solver 프레임). descend clamp·bowl ring 높이 기준.
TABLE_TOP = TABLE_TOP_BASE + BASE_T[2]

#: 큐브 obstacle/attach blob 한 변(m) — 50 mm 고정. ★cube_specs.py 의 DR 사다리
#: (25~40 mm, CUBE_SIZE_CHOICES) 보다 크게 두는 이유: collision 은 과대근사가 안전측이고,
#: world obstacle dims 는 planner 초기화 시 굳어 요청마다 못 바꾼다. 작은 큐브에는
#: 보수적 여유가 있어 안전하다. 상세 = cube_specs.py 주석 "사다리 상한".
CUBE_DIMS = 0.05

#: 그릇 rim = hollow octagon ring. cuRobo world obstacle 은 solid convex 뿐이라 오목 그릇을
#: solid box 로 넣으면 내부(빈 공간)가 hover 자세와 허위충돌한다. rim 벽만 N× cuboid 로 근사해
#: keep-out 유지 + 허위충돌 제거.
#: 박스 = 다각형 '변': tangential w ≈ 변길이+겹침, radial d 는 코너 azimuth 에 벽 구멍이 없을
#: 만큼 두껍게 — 0-gap 은 :func:`assert_bowl_ring_sealed` 가 강제한다.
BOWL_RING_N = 8
BOWL_RING_RC = 0.080   # box 중심 반경(m) — 벽 band [RC±d/2] 이 실제 rim 0.075 포함
BOWL_RING_H = 0.075    # ring 높이(테이블→rim)
BOWL_RING_DIMS = (0.030, 0.083, BOWL_RING_H)  # [radial, tangential, height]


def bowl_ring(bx: float, by: float) -> dict[str, dict]:
    """오목 그릇 rim → hollow octagon ring ``dict{name: {dims, pose}}``.

    각 box = 다각형 '변'(local-x=radial, quat=Rz(θ)), 배치 반경 ``BOWL_RING_RC`` →
    내부 hole + 상단 open. 중심 ``(bx, by)`` 이동(DR)마다 재계산한다.
    좌표는 URDF solver 프레임이다.
    """
    z_c = TABLE_TOP + BOWL_RING_H / 2
    ring: dict[str, dict] = {}
    for i in range(BOWL_RING_N):
        th = 2 * math.pi * i / BOWL_RING_N
        qw, qz = math.cos(th / 2), math.sin(th / 2)
        ring[f"bowl_{i}"] = {
            "dims": list(BOWL_RING_DIMS),
            "pose": [bx + BOWL_RING_RC * math.cos(th),
                     by + BOWL_RING_RC * math.sin(th), z_c, qw, 0.0, 0.0, qz]}
    return ring


def assert_bowl_ring_sealed(min_hole: float) -> None:
    """로드시 1-check: (a) hole 이 ``min_hole`` 이상 (b) rim 원둘레 전 azimuth 벽 연속.

    얇은 radial d 는 코너에 벽 구멍을 남겨 팔이 그릇을 파고든다 → 임포트 시 즉시 실패시킨다.

    Args:
        min_hole: 내부 hole 반경 하한(m). 호출자가 자기 용도(드롭 오프셋+pad 여유 등)로 정한다.
    """
    hole = BOWL_RING_RC - BOWL_RING_DIMS[0] / 2
    assert hole >= min_hole, f"bowl ring hole {hole:.3f}m < 요구 {min_hole:.3f}m"
    ring = bowl_ring(0.0, 0.0)

    def _inside(px, py, e):
        x, y, _z, qw, _qx, _qy, qz = e["pose"]
        dx, dy, _dz = e["dims"]
        th = 2 * math.atan2(qz, qw)
        c, s = math.cos(-th), math.sin(-th)
        return abs(c * (px - x) - s * (py - y)) <= dx / 2 and abs(s * (px - x) + c * (py - y)) <= dy / 2

    for k in range(720):
        ph = 2 * math.pi * k / 720
        px, py = 0.075 * math.cos(ph), 0.075 * math.sin(ph)  # 실제 rim 원
        assert any(_inside(px, py, e) for e in ring.values()), \
            f"bowl ring wall gap @ {math.degrees(ph):.0f}° — radial d↑ 또는 N↑"


def _self_check() -> None:
    """프레임 왕복·ring 봉인 자체검사."""
    for p in [(0.1, 0.2, 0.3), (-0.05, 0.4, 0.0), (0.0, 0.0, 0.0)]:
        back = urdf_to_usd(usd_to_urdf(p))
        assert max(abs(a - b) for a, b in zip(p, back)) < 1e-12, f"round-trip 실패: {p} → {back}"

    # 4×4 경로가 위치 경로와 같은 답을 내는지(회전 항 포함 정합)
    pose = np.eye(4)
    pose[:3, 3] = (0.1, 0.2, 0.3)
    got = transform_pose_usd_to_urdf(pose)[:3, 3]
    want = np.asarray(usd_to_urdf((0.1, 0.2, 0.3)))
    assert np.allclose(got, want), f"4×4 ↔ 위치 변환 불일치: {got} vs {want}"

    assert np.allclose(T_USD_FROM_URDF @ T_URDF_FROM_USD, np.eye(4)), "역변환 불일치"
    assert_bowl_ring_sealed(0.058)  # pick-place 용 하한(BOWL_PULL 0.03 + pad 0.028)
    print("curobo_frames self-check PASS")


if __name__ == "__main__":
    _self_check()
