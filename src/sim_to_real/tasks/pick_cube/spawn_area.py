"""SO-101 pick_cube DR 스폰 영역 — **단일 기하 소스** (isaaclab 무의존, 순수 python).

``randomize_cubes_scattered`` (Cube1, full 모드) 의 accept 조건을 그대로 미러한다::

    in_spawn_area(x,y) =  |x| <= bell(y)                     # 좌우대칭 종모양(최외곽)
                        ∧ (x,y) ∉ arm_exclude_box            # 로봇암 주변 배제
                        ∧ dist((x,y), bowl) >= MIN_BOWL_SEP  # 그릇 겹침 금지(내곽)
                        ∧ dist((x,y), base) >= MIN_BASE_SEP  # base 발치 배제(최내곽)

env_cfg(스폰 상수) · pickplace_sm(``--sweep_grid``) · plot_sweep(경계 오버레이) 세 곳이
이 모듈을 공유하므로 세 곳의 경계가 절대 어긋나지 않는다. **값을 바꿀 땐 여기 한 곳만.**

Cube1(40mm) 단일 큐브라 큐브간 이격(min_cube_sep)은 첫 배치라 무효 → 마스크에서 제외.
"""
from __future__ import annotations

import math

# 종 프로파일: (큐브중심 y, 좌우대칭 x 반너비) breakpoint. 사이 선형보간, 밖은 clamp.
CUBE_SCATTER_BELL: list[tuple[float, float]] = [
    (0.06, 0.24), (0.14, 0.24), (0.18, 0.20), (0.22, 0.16), (0.26, 0.08),
]
CUBE_SCATTER_X_RANGE: tuple[float, float] = (-0.24, 0.24)  # bell 최대 외접 사각형
CUBE_SCATTER_Y_RANGE: tuple[float, float] = (0.06, 0.26)
# 로봇암 주변 제외 박스 (x0,x1,y0,y1) env-local — full·base 공통.
CUBE_ARM_EXCLUDE: tuple[float, float, float, float] = (-0.09, 0.04, -0.045, 0.155)
# 배치 거리 제약 (Cube1 40mm) — env_cfg._make_randomize_cubes 인자와 동일 소스.
BOWL_CENTER_XY: tuple[float, float] = (-0.22, 0.265)  # bowl default (env-local)
MIN_BOWL_SEP: float = 0.14
MIN_BASE_SEP: float = 0.135
BASE_XY: tuple[float, float] = (0.0, 0.0)  # robot base (env-local)


def bell_halfwidth(y: float) -> float:
    """종 모양 x 반너비 w(y) — piecewise-linear (domain_randomization._bell_halfwidth 스칼라판)."""
    bp = sorted(CUBE_SCATTER_BELL)
    ys = [p[0] for p in bp]
    ws = [p[1] for p in bp]
    if y <= ys[0]:
        return ws[0]
    if y >= ys[-1]:
        return ws[-1]
    for i in range(len(ys) - 1):
        if ys[i] < y <= ys[i + 1]:
            f = (y - ys[i]) / (ys[i + 1] - ys[i])
            return ws[i] + f * (ws[i + 1] - ws[i])
    return ws[-1]


def in_spawn_area(x: float, y: float) -> bool:
    """(x,y) env-local 이 Cube1 스폰 가능영역 안인가 — DR accept 와 동일 판정."""
    if abs(x) > bell_halfwidth(y):
        return False
    ex0, ex1, ey0, ey1 = CUBE_ARM_EXCLUDE
    if ex0 <= x <= ex1 and ey0 <= y <= ey1:
        return False
    bx, by = BOWL_CENTER_XY
    if (x - bx) ** 2 + (y - by) ** 2 < MIN_BOWL_SEP ** 2:
        return False
    ax, ay = BASE_XY
    if (x - ax) ** 2 + (y - ay) ** 2 < MIN_BASE_SEP ** 2:
        return False
    return True


def _arc(cx, cy, r, th0_deg, th1_deg, n):
    """중심(cx,cy)·반경 r 원호 위 n점 [th0,th1]도."""
    return [(cx + r * math.cos(math.radians(t)), cy + r * math.sin(math.radians(t)))
            for t in (th0_deg + (th1_deg - th0_deg) * k / (n - 1) for k in range(n))]


def sweep_targets(nx: int = 15, ny: int = 8, boundary_n: int = 20,
                  inset: float = 0.006, dedup_r: float = 0.010) -> list[tuple[float, float, str]]:
    """스폰 영역 평가 타깃 (x, y, kind) 목록.

    interior = bounding box nx×ny grid ∩ in_spawn_area.
    boundary = 각 제약면 **바로 안쪽**(inset) 샘플 — 최외곽(bell·y edge)·최내곽
    (base_arc·bowl_arc·exclude_edge). 경계 타깃을 먼저 넣고 interior 를 dedup 로 얹어,
    사용자 요구인 "최외곽/내곽 경계면" 이 항상 평가에 포함된다.
    """
    xlo, xhi = CUBE_SCATTER_X_RANGE
    ylo, yhi = CUBE_SCATTER_Y_RANGE
    bx, by = BASE_XY
    wbx, wby = BOWL_CENTER_XY
    boundary: list[tuple[float, float, str]] = []

    # 최외곽 — bell 좌우 edge (|x| = w(y) 바로 안쪽)
    for k in range(boundary_n):
        y = ylo + (yhi - ylo) * k / (boundary_n - 1)
        w = bell_halfwidth(y)
        for x in (w - inset, -(w - inset)):
            if in_spawn_area(x, y):
                boundary.append((x, y, "bell"))
    # 최외곽 — 근/원 y edge
    for k in range(boundary_n):
        x = xlo + (xhi - xlo) * k / (boundary_n - 1)
        for y in (ylo + inset, yhi - inset):
            if in_spawn_area(x, y):
                boundary.append((x, y, "yedge"))
    # 최내곽 — base-sep 원호 (로봇 발치, +y 반구)
    for x, y in _arc(bx, by, MIN_BASE_SEP + inset, 15.0, 165.0, boundary_n * 2):
        if in_spawn_area(x, y):
            boundary.append((x, y, "base_arc"))
    # 내곽 — bowl-sep 원호 (그릇 좌상단, 작업영역 향한 호)
    for x, y in _arc(wbx, wby, MIN_BOWL_SEP + inset, -110.0, 20.0, boundary_n * 2):
        if in_spawn_area(x, y):
            boundary.append((x, y, "bowl_arc"))
    # 내곽 — arm exclude box 3변 바로 바깥
    ex0, ex1, ey0, ey1 = CUBE_ARM_EXCLUDE
    for k in range(boundary_n):
        t = k / (boundary_n - 1)
        for x, y in ((ex0 - inset, ey0 + (ey1 - ey0) * t),
                     (ex1 + inset, ey0 + (ey1 - ey0) * t),
                     (ex0 + (ex1 - ex0) * t, ey1 + inset)):
            if in_spawn_area(x, y):
                boundary.append((x, y, "exclude_edge"))

    interior: list[tuple[float, float, str]] = []
    for j in range(ny):
        y = ylo + (yhi - ylo) * j / (ny - 1)
        for i in range(nx):
            x = xlo + (xhi - xlo) * i / (nx - 1)
            if in_spawn_area(x, y):
                interior.append((x, y, "interior"))

    # dedup: 경계 타깃 우선 유지, 그 근방 interior 는 버림.
    kept: list[tuple[float, float, str]] = []
    r2 = dedup_r ** 2
    for pt in boundary + interior:
        if all((pt[0] - q[0]) ** 2 + (pt[1] - q[1]) ** 2 >= r2 for q in kept):
            kept.append(pt)
    return kept


def _self_check() -> None:
    """마스크·타깃 불변식 (GPU 불요). 실패 시 assert."""
    assert in_spawn_area(0.0, 0.20), "명백한 내부점이 거부됨"
    assert not in_spawn_area(0.30, 0.06), "bell 밖(|x|>w)이 허용됨"
    assert not in_spawn_area(-0.02, 0.10), "arm exclude box 안이 허용됨"
    assert not in_spawn_area(0.0, 0.08), "base-sep 원 안이 허용됨"
    assert not in_spawn_area(-0.20, 0.24), "bowl-sep 원 안이 허용됨"
    tg = sweep_targets()
    kinds = {k for _, _, k in tg}
    assert all(in_spawn_area(x, y) for x, y, _ in tg), "타깃에 스폰영역 밖 존재"
    for req in ("bell", "yedge", "base_arc", "exclude_edge", "interior"):
        assert req in kinds, f"경계/내부 kind 누락: {req}"
    n_bnd = sum(1 for _, _, k in tg if k != "interior")
    print(f"[spawn_area] OK  targets={len(tg)} (boundary={n_bnd}, interior={len(tg) - n_bnd}) "
          f"kinds={sorted(kinds)}")


if __name__ == "__main__":
    _self_check()
