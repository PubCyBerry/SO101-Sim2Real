"""cuRobo pick-place SM 스폰영역 sweep 결과 → matplotlib 성공맵 PNG.

``pickplace_sm.py sweep`` 이 쓴 ``sweep_results.json`` 을 읽어 DR 스폰영역 위에
per-cell place 성공률을 그린다. 사용자 요구인 **최외곽/내곽 경계 셀을 굵은 테두리로 강조**.

host ``.venv`` (matplotlib) 로 실행. ``sim_to_real`` 패키지 __init__ 은 isaac 을 import 하므로
피하고, 순수 python 인 ``spawn_area.py`` 만 importlib 로 파일 로드해 마스크 기하를 공유한다.

    uv run python scripts/cuRobo/plot_sweep.py \
        --results outputs/curobo_sweep/sweep_results.json

    # GPU 없이 렌더 파이프라인만 확인(합성 데이터):
    uv run python scripts/cuRobo/plot_sweep.py --demo --out outputs/curobo_sweep/demo_map.png
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402


def _load_spawn_area():
    """spawn_area.py 를 패키지 init 우회로 파일 로드(순수 python, isaac 불요)."""
    p = Path(__file__).resolve().parents[2] / "src/sim_to_real/tasks/pick_cube/spawn_area.py"
    spec = importlib.util.spec_from_file_location("spawn_area_standalone", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _demo_data(SA):
    """GPU 없이 렌더 검증용 합성 결과 — 좌우 절반 성공/실패 체크무늬."""
    cells = []
    for x, y, kind in SA.sweep_targets():
        planned = not (x < -0.18 and y > 0.22)          # 좌상단은 plan-fail 흉내
        placed = 1 if (planned and ((x + y) % 0.08) < 0.04) else 0
        cells.append({"x": x, "y": y, "kind": kind, "n": 1,
                      "n_planned": int(planned), "n_placed": int(placed),
                      "fails": [] if placed else (["synthetic"] if planned else ["no_curobo_solution"])})
    return {"task": "DEMO-synthetic", "num_envs": 0, "yaw": "0", "trials": 1,
            "spawn": {"bell": [list(p) for p in SA.CUBE_SCATTER_BELL],
                      "x_range": list(SA.CUBE_SCATTER_X_RANGE),
                      "y_range": list(SA.CUBE_SCATTER_Y_RANGE),
                      "exclude_box": list(SA.CUBE_ARM_EXCLUDE),
                      "bowl_center_xy": list(SA.BOWL_CENTER_XY), "bowl_sep": SA.MIN_BOWL_SEP,
                      "base_xy": list(SA.BASE_XY), "base_sep": SA.MIN_BASE_SEP},
            "n_targets": len(cells), "cells": cells}


def plot(data, SA, out):
    cells = data["cells"]
    if not cells:
        raise SystemExit("no cells in results — sweep 가 한 셀도 기록 못 함")
    sp = data.get("spawn", {})
    xr = sp.get("x_range", list(SA.CUBE_SCATTER_X_RANGE))
    yr = sp.get("y_range", list(SA.CUBE_SCATTER_Y_RANGE))

    # 스폰영역 마스크 fill/outline (spawn_area.in_spawn_area 단일 소스)
    gx = np.linspace(xr[0] - 0.03, xr[1] + 0.03, 300)
    gy = np.linspace(yr[0] - 0.05, yr[1] + 0.04, 200)
    MX, MY = np.meshgrid(gx, gy)
    mask = np.vectorize(SA.in_spawn_area)(MX, MY).astype(float)

    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    ax.contourf(MX, MY, mask, levels=[0.5, 1.5], colors=["#e9eff8"], zorder=0)
    ax.contour(MX, MY, mask, levels=[0.5], colors=["#37506e"], linewidths=1.6, zorder=1)

    # 제약면 오버레이(JSON meta)
    bx, by = sp.get("bowl_center_xy", list(SA.BOWL_CENTER_XY))
    ax.add_patch(Circle((bx, by), sp.get("bowl_sep", SA.MIN_BOWL_SEP), fill=False,
                        ls="--", ec="#c0392b", lw=1.1, zorder=2))
    a0, a1 = sp.get("base_xy", [0.0, 0.0])
    ax.add_patch(Circle((a0, a1), sp.get("base_sep", SA.MIN_BASE_SEP), fill=False,
                        ls="--", ec="#8e44ad", lw=1.1, zorder=2))
    ex0, ex1, ey0, ey1 = sp.get("exclude_box", list(SA.CUBE_ARM_EXCLUDE))
    ax.add_patch(Rectangle((ex0, ey0), ex1 - ex0, ey1 - ey0, fc="#f4d7d7", ec="#c0392b",
                           lw=1.0, hatch="///", alpha=0.7, zorder=1))
    ax.plot([bx], [by], marker="*", ms=20, color="#c0392b", zorder=5)
    ax.plot([a0], [a1], marker="^", ms=13, color="#8e44ad", zorder=5)

    xs = np.array([c["x"] for c in cells])
    ys = np.array([c["y"] for c in cells])
    frac = np.array([c["n_placed"] / max(c["n"], 1) for c in cells])
    planned = np.array([c["n_planned"] > 0 for c in cells])
    is_bnd = np.array([c["kind"] != "interior" for c in cells])

    pf = ~planned
    if pf.any():  # 계획 자체 실패 = 회색 X
        ax.scatter(xs[pf], ys[pf], marker="x", c="#7f8c8d", s=75, lw=2.2, zorder=6)
    ok = planned
    if ok.any():
        ax.scatter(xs[ok], ys[ok], c=frac[ok], cmap="RdYlGn", vmin=0.0, vmax=1.0,
                   s=np.where(is_bnd[ok], 160, 85),
                   edgecolors=np.where(is_bnd[ok], "black", "#555555"),
                   linewidths=np.where(is_bnd[ok], 1.9, 0.6), zorder=7)
        sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(0, 1))
        cb = fig.colorbar(sm, ax=ax, shrink=0.82)
        cb.set_label("place success fraction")

    tot = sum(c["n"] for c in cells)
    pls = sum(c["n_placed"] for c in cells)
    pln = sum(c["n_planned"] for c in cells)
    bnd = [c for c in cells if c["kind"] != "interior"]
    b_tot = sum(c["n"] for c in bnd)
    b_pls = sum(c["n_placed"] for c in bnd)
    ax.set_title(
        f"{data.get('task', '?')}  ·  spawn-area sweep  ·  cells={len(cells)}\n"
        f"place {pls}/{tot}={pls / max(tot, 1) * 100:.0f}%   "
        f"plan {pln}/{tot}={pln / max(tot, 1) * 100:.0f}%   "
        f"boundary place {b_pls}/{b_tot}={b_pls / max(b_tot, 1) * 100:.0f}%   "
        f"(yaw={data.get('yaw')} trials={data.get('trials')} envs={data.get('num_envs')})",
        fontsize=10)
    ax.set_xlabel("x  (env-local, m) — robot base at x=0")
    ax.set_ylabel("y  (env-local, m)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    handles = [
        Line2D([0], [0], marker="o", ls="", mfc="#2ecc71", mec="black", ms=12,
               label="boundary cell (thick edge)"),
        Line2D([0], [0], marker="o", ls="", mfc="#2ecc71", mec="#555", ms=8, label="interior cell"),
        Line2D([0], [0], marker="x", ls="", color="#7f8c8d", ms=9, label="plan fail"),
        Line2D([0], [0], marker="*", ls="", color="#c0392b", ms=14, label="bowl"),
        Line2D([0], [0], marker="^", ls="", color="#8e44ad", ms=11, label="robot base"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.92)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"[plot_sweep] wrote {out}  cells={len(cells)} place={pls}/{tot} "
          f"plan={pln}/{tot} boundary_place={b_pls}/{b_tot}")


def main():
    ap = argparse.ArgumentParser(description="spawn-area sweep 성공맵")
    ap.add_argument("--results", default="outputs/curobo_sweep/sweep_results.json")
    ap.add_argument("--out", default=None, help="PNG 경로(기본 results 옆 sweep_map.png)")
    ap.add_argument("--demo", action="store_true", help="GPU 불요 합성데이터로 렌더만 검증")
    args = ap.parse_args()
    SA = _load_spawn_area()
    if args.demo:
        data = _demo_data(SA)
        out = Path(args.out) if args.out else Path("outputs/curobo_sweep/demo_map.png")
    else:
        data = json.loads(Path(args.results).read_text(encoding="utf-8"))
        out = Path(args.out) if args.out else Path(args.results).with_name("sweep_map.png")
    plot(data, SA, out)


if __name__ == "__main__":
    main()
