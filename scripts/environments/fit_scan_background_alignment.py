#!/usr/bin/env python3
"""3DGS 스캔 ply 에서 바닥·책상 평면을 재서 ``SO101_BACKGROUND_*`` env var 를 뽑는다.

배경 스캔을 sim 책상에 맞추는 건 눈대중이 아니라 **측정**이다. 이 스크립트가 하는 일:

1. z 히스토그램에서 **수평 평면 후보**를 찾는다 (바닥 = 가장 낮은 유의미 peak).
2. 바닥 위 0.55~1.0 m 구간의 슬랩을 2D 격자로 flood-fill 해 **책상 후보**를 면적순으로 뽑고,
   각 후보의 중심·장축 yaw·크기를 PCA 로 잰다.
3. 고른 후보를 **sim 책상 상판**(``cube_desk/scene.usd`` 의 ``DeskTopSurface`` bbox 에서 읽음)에
   포갠 pos/rpy 를 계산해 그대로 붙여넣을 수 있는 env var 로 출력한다.

좌표 규약 — 배치 변환은 ``world = pos + S · Rz(yaw) · Ry(pitch) · Rx(roll) · p_ply``.
폰 스캔은 Y-up 이라 ``roll=90`` 이 고정이고(Isaac Z-up), yaw 가 책상 장축을 world X 에 맞춘다.

사용:
    .venv/bin/python scripts/environments/fit_scan_background_alignment.py \
        assets/test_3dgs_background.ply --list          # 책상 후보 나열
    .venv/bin/python scripts/environments/fit_scan_background_alignment.py \
        assets/test_3dgs_background.ply --desk 1 --flip --scale 1.0
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

_SH_C0 = 0.28209479177387814
_REPO_ROOT = Path(__file__).resolve().parents[2]

# sim 책상 상판은 scene.usd 에서 읽는다(상수 복제 금지). 씬 cfg 의 +y 1 cm shift 는
# 스캔 평면 피팅 오차(~cm)보다 작아 무시한다 — 필요하면 --sim-desk-offset 로 준다.
_CUBE_DESK_USD = _REPO_ROOT / "assets" / "scenes" / "cube_desk" / "scene.usd"
_DESK_TOP_PRIM = "/Scene/DeskTopSurface"


def load_ply_points(path: Path, min_opacity: float, max_sigma: float, max_extent: float) -> np.ndarray:
    """3DGS ply → Y-up→Z-up 적용한 가우시안 중심 (N,3). 저-alpha·거대·이상치 제거."""
    with open(path, "rb") as f:
        header = b""
        while b"end_header" not in header:
            chunk = f.read(4096)
            if not chunk:
                raise ValueError(f"{path}: end_header 없음")
            header += chunk
    off = header.index(b"end_header\n") + len(b"end_header\n")
    lines = header[:off].decode("ascii", errors="replace").splitlines()
    props = [ln.split()[2] for ln in lines if ln.startswith("property")]
    count = int(next(ln for ln in lines if ln.startswith("element vertex")).split()[2])
    a = np.fromfile(path, dtype=np.dtype([(n, "<f4") for n in props]), offset=off, count=count)

    xyz = np.stack([a["x"], a["y"], a["z"]], 1)
    sigma = np.exp(np.stack([a["scale_0"], a["scale_1"], a["scale_2"]], 1))
    alpha = 1.0 / (1.0 + np.exp(-a["opacity"]))
    keep = (
        (alpha > min_opacity)
        & (sigma.max(1) < max_sigma)
        & np.isfinite(xyz).all(1)
        & (np.abs(xyz) < max_extent).all(1)
    )
    p = xyz[keep]
    # Rx(+90): Y-up ply → Z-up. 이후 모든 좌표는 이 "w 프레임" 기준이다.
    return np.stack([p[:, 0], -p[:, 2], p[:, 1]], 1)


def find_floor_z(w: np.ndarray) -> float:
    """가장 낮은 유의미 수평 평면 = 바닥."""
    h, e = np.histogram(w[:, 2], bins=450, range=(-2.0, 2.5))
    zc = (e[:-1] + e[1:]) / 2
    peaks = [zc[i] for i in range(2, len(h) - 2)
             if h[i] == h[i - 2:i + 3].max() and h[i] > 0.15 * h.max()]
    if not peaks:
        raise SystemExit("수평 평면을 못 찾음 — 필터가 과하거나 스캔이 비었다")
    return float(min(peaks))


def desk_candidates(w: np.ndarray, floor_z: float, res: float = 0.03, min_cells: int = 100):
    """바닥 위 책상 높이대의 슬랩을 flood-fill → (면적순) 후보 리스트."""
    band = w[(w[:, 2] > floor_z + 0.55) & (w[:, 2] < floor_z + 1.0)]
    if len(band) == 0:
        raise SystemExit("책상 높이대에 점이 없다")
    hb, eb = np.histogram(band[:, 2], bins=90)
    desk_z = float((eb[:-1] + eb[1:])[np.argmax(hb)] / 2 + (eb[1] - eb[0]) / 4)
    desk_z = float(((eb[:-1] + eb[1:]) / 2)[np.argmax(hb)])

    slab = w[np.abs(w[:, 2] - desk_z) < 0.03][:, :2]
    xe = np.arange(slab[:, 0].min(), slab[:, 0].max() + res, res)
    ye = np.arange(slab[:, 1].min(), slab[:, 1].max() + res, res)
    H, _, _ = np.histogram2d(slab[:, 0], slab[:, 1], bins=[xe, ye])
    mask = H >= 3

    seen = np.zeros_like(mask)
    comps = []
    for si in range(mask.shape[0]):
        for sj in range(mask.shape[1]):
            if not mask[si, sj] or seen[si, sj]:
                continue
            q = deque([(si, sj)])
            seen[si, sj] = True
            cells = []
            while q:
                i, j = q.popleft()
                cells.append((i, j))
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        x, y = i + di, j + dj
                        if 0 <= x < mask.shape[0] and 0 <= y < mask.shape[1] \
                                and mask[x, y] and not seen[x, y]:
                            seen[x, y] = True
                            q.append((x, y))
            if len(cells) >= min_cells:
                comps.append(np.array(cells))

    out = []
    for c in sorted(comps, key=len, reverse=True):
        pts = np.stack([(xe[c[:, 0]] + xe[c[:, 0] + 1]) / 2,
                        (ye[c[:, 1]] + ye[c[:, 1] + 1]) / 2], 1)
        ctr = pts.mean(0)
        d = pts - ctr
        ev, evec = np.linalg.eigh(np.cov(d.T))
        k = int(np.argmax(ev))
        yaw = math.degrees(math.atan2(evec[1, k], evec[0, k])) % 180.0
        proj = d @ evec
        ext = proj.max(0) - proj.min(0)
        out.append({
            "center": (float(ctr[0]), float(ctr[1])),
            "top_z": desk_z,
            "yaw_deg": yaw,
            "length": float(ext[k]),
            "width": float(ext[1 - k]),
            "area_m2": len(c) * res * res,
        })
    return out


def sim_desk_top() -> tuple[tuple[float, float], float, tuple[float, float]]:
    """sim 책상 상판 (center_xy, top_z, (size_x, size_y)) — scene.usd 에서 읽는다."""
    from pxr import UsdGeom, Usd

    stage = Usd.Stage.Open(str(_CUBE_DESK_USD))
    prim = stage.GetPrimAtPath(_DESK_TOP_PRIM)
    if not prim or not prim.IsValid():
        raise SystemExit(f"{_CUBE_DESK_USD}: {_DESK_TOP_PRIM} 없음")
    b = UsdGeom.BBoxCache(0.0, ["default"]).ComputeWorldBound(prim).ComputeAlignedBox()
    lo, hi = b.GetMin(), b.GetMax()
    return ((float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)),
            float(hi[2]),
            (float(hi[0] - lo[0]), float(hi[1] - lo[1])))


def scale_from_desk_height(desk_height: float) -> float:
    """sim 책상(실측 자산)의 상판 높이를 기준자로 스캔 스케일을 잡는다.

    **extent(폭·길이)로 스케일을 잡으면 안 된다** — 스캔은 가장자리가 잘려 항상 작게 나온다
    (실측: 폭 0.57 vs 실제 0.80). 바닥면↔상판면은 둘 다 큰 평면이고 **차이**라서
    절단 편향이 없다. 이게 유일하게 믿을 수 있는 척도다.
    """
    _, sim_z, _ = sim_desk_top()
    return sim_z / desk_height


def refine_desk_frame(w: np.ndarray, desk: dict, scale: float, sim_size: tuple[float, float]):
    """책상 국소 프레임에서 sim 책상 사각형의 중심을 잡는다.

    component 무게중심은 못 쓴다 — 후보가 여러 책상/긴 카운터를 묶어버리면 중심이 엉뚱한
    곳에 간다. 대신 **파티션 쪽(+v) 뒷모서리**(스캐너가 정면에서 봐 잘 잡힌 쪽)에
    sim 책상 뒷변을 붙이고, 장축은 점유 구간 중앙에 둔다.
    """
    th = math.radians(desk["yaw_deg"])
    u = np.array([math.cos(th), math.sin(th)])
    v = np.array([-math.sin(th), math.cos(th)])
    q = w[:, :2] - np.array(desk["center"])
    su, sv = q @ u, q @ v
    slab = np.abs(w[:, 2] - desk["top_z"]) < 0.02

    length_s, width_s = sim_size[0] / scale, sim_size[1] / scale

    near = slab & (np.abs(su) < length_s / 2 + 0.4)
    h, e = np.histogram(sv[near], bins=70, range=(-0.9, 0.9))
    vc = (e[:-1] + e[1:]) / 2
    back_v = float(vc[np.where(h > 0.10 * h.max())[0][-1]])

    mid = slab & (np.abs(sv) < 0.45)
    h2, e2 = np.histogram(su[mid], bins=90, range=(-2.2, 2.2))
    uc = (e2[:-1] + e2[1:]) / 2
    on = np.where(h2 > 0.25 * h2.max())[0]
    center_u = float((uc[on[0]] + uc[on[-1]]) / 2)
    center_v = back_v - width_s / 2

    center_w = np.array(desk["center"]) + center_u * u + center_v * v
    return (float(center_w[0]), float(center_w[1])), back_v, (center_u, center_v)


def solve_placement(desk: dict, scale: float, flip: bool, center_xy: tuple[float, float] | None = None):
    """스캔 책상 → sim 책상 을 포개는 (pos, rpy_deg). 자체 검산 포함."""
    sim_c, sim_z, sim_size = sim_desk_top()
    yaw = -desk["yaw_deg"] + (180.0 if flip else 0.0)
    r = math.radians(yaw)
    cx, cy = center_xy if center_xy is not None else desk["center"]
    # world = pos + S·Rz(yaw)·c   →   pos = target - S·Rz(yaw)·c
    rx = math.cos(r) * cx - math.sin(r) * cy
    ry = math.sin(r) * cx + math.cos(r) * cy
    pos = (sim_c[0] - scale * rx, sim_c[1] - scale * ry, sim_z - scale * desk["top_z"])

    # 검산: 같은 변환으로 스캔 책상 중심이 sim 책상 중심에 오는가
    back = (pos[0] + scale * rx, pos[1] + scale * ry, pos[2] + scale * desk["top_z"])
    err = max(abs(back[0] - sim_c[0]), abs(back[1] - sim_c[1]), abs(back[2] - sim_z))
    assert err < 1e-6, f"검산 실패 err={err}"
    return pos, (90.0, 0.0, yaw), sim_c, sim_z, sim_size


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ply", type=Path)
    ap.add_argument("--list", action="store_true", help="책상 후보만 나열하고 종료")
    ap.add_argument("--desk", type=int, default=0, help="--list 의 후보 index (기본 0 = 최대 면적)")
    ap.add_argument("--flip", action="store_true", help="장축 180° 뒤집기(로봇이 책상 반대편일 때)")
    ap.add_argument("--scale", type=float, default=None,
                    help="스캔 등방 스케일. 생략하면 sim 책상 상판 높이(실측 자산)를 기준자로 자동 산출")
    ap.add_argument("--centroid", action="store_true",
                    help="뒷모서리 정렬 대신 component 무게중심을 쓴다(후보가 책상 하나일 때만)")
    ap.add_argument("--min-opacity", type=float, default=0.3)
    ap.add_argument("--max-sigma", type=float, default=0.5)
    ap.add_argument("--max-extent", type=float, default=12.0)
    args = ap.parse_args(argv)

    w = load_ply_points(args.ply, args.min_opacity, args.max_sigma, args.max_extent)
    floor_z = find_floor_z(w)
    cands = desk_candidates(w, floor_z)
    if not cands:
        raise SystemExit("책상 후보 0개")

    desk_h = cands[0]["top_z"] - floor_z
    auto_scale = scale_from_desk_height(desk_h)
    scale = args.scale if args.scale is not None else auto_scale
    _, sim_z_ref, sim_size = sim_desk_top()
    print(f"바닥 z={floor_z:+.3f}  ·  책상면 z={cands[0]['top_z']:+.3f}  "
          f"·  스캔 책상높이={desk_h:.3f} m  ·  점 {len(w)}개")
    print(f"sim 책상 상판 높이(실측 자산)={sim_z_ref:.3f} m  →  S={auto_scale:.4f}"
          f"{'' if args.scale is None else f'  (--scale {scale} 로 덮어씀)'}")

    print("\n책상 후보 (면적순):")
    for i, c in enumerate(cands):
        print(f"  [{i}] center=({c['center'][0]:+.3f},{c['center'][1]:+.3f}) "
              f"yaw={c['yaw_deg']:5.1f}°  {c['length']:.2f} × {c['width']:.2f} m  "
              f"area={c['area_m2']:.2f} m²")
    if args.list:
        return 0

    if not 0 <= args.desk < len(cands):
        raise SystemExit(f"--desk {args.desk} 범위 밖 (0..{len(cands) - 1})")

    d = cands[args.desk]
    center_xy = None
    if not args.centroid:
        center_xy, back_v, local = refine_desk_frame(w, d, scale, sim_size)
        print(f"\n뒷모서리 정렬: 파티션쪽 모서리 v={back_v:+.3f} → "
              f"국소 중심 (u {local[0]:+.3f}, v {local[1]:+.3f})")
    pos, rpy, sim_c, sim_z, sim_size = solve_placement(d, scale, args.flip, center_xy)

    print(f"\nsim 책상 상판: center=({sim_c[0]:+.3f},{sim_c[1]:+.3f}) z={sim_z:.3f} "
          f"{sim_size[0]:.2f} × {sim_size[1]:.2f} m  (scene.usd {_DESK_TOP_PRIM})")
    print(f"고른 스캔 책상 [{args.desk}]: {d['length']:.2f} × {d['width']:.2f} m"
          f"{'  (flip: 장축 180° 뒤집음)' if args.flip else ''}")
    print("\n─ env var ─────────────────────────────────────────────")
    print(f'SO101_BACKGROUND_POS="{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}"')
    print(f'SO101_BACKGROUND_RPY_DEG="{rpy[0]:.1f},{rpy[1]:.1f},{rpy[2]:.1f}"')
    print(f'SO101_BACKGROUND_SCALE="{scale:.4f}"')
    print("\n⚠ scale 을 바꾸면 POS 도 같이 바뀐다. 반드시 이 스크립트로 다시 뽑을 것.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
