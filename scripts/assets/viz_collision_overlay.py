"""Visual mesh vs Collision mesh(SDF source) 오버레이 시각화 — 큐브/jaw/gripper.

각 요소에 대해 (a) 시각 mesh(회색 반투명 표면), (b) SDF 충돌 source mesh(초록, SDF 가
그대로 따라가는 형상 = visual 과 일치), (c) 비교용 convex hull(빨강 wireframe = 기존
convexHull/Decomposition 이 부풀리던 근사)을 겹쳐 PNG 로 저장한다.

usd-core + trimesh + matplotlib(Agg) 만 사용 — Isaac/GPU 불요.

실행(재author·로봇 SDF 적용 후):
    .venv/bin/python scripts/assets/viz_collision_overlay.py
출력: outputs/collision_overlay/{cube,jaw,gripper}.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import trimesh
from pxr import Usd, UsdGeom, Gf

ROOT = Path(__file__).resolve().parents[2]
ROBOT_USD = ROOT / "assets" / "robots" / "so101_follower.usd"
CUBE_USD = ROOT / "assets" / "scenes" / "cube_desk" / "objects" / "Cube3" / "Cube3.usd"
OUT_DIR = ROOT / "outputs" / "collision_overlay"

# 요소별 (라벨, USD, [visual prim...], [collision prim...])
ELEMENTS = [
    ("cube", CUBE_USD,
     ["/Cube3/Visual"],
     ["/Cube3/Collision"]),
    ("jaw", ROBOT_USD,
     ["/visuals/jaw/moving_jaw_so101_v1/mesh"],
     ["/colliders/jaw/moving_jaw_so101_v1/mesh"]),
    ("gripper", ROBOT_USD,
     ["/visuals/gripper/sts3215_03a_v1/mesh",
      "/visuals/gripper/wrist_roll_follower_so101_v1/mesh"],
     ["/colliders/gripper/sts3215_03a_v1/mesh",
      "/colliders/gripper/wrist_roll_follower_so101_v1/mesh"]),
]


def _triangulate(counts, indices):
    """faceVertexCounts/Indices → (M,3) 삼각형 인덱스 (fan triangulation)."""
    tris = []
    off = 0
    for c in counts:
        face = indices[off:off + c]
        for k in range(1, c - 1):
            tris.append([face[0], face[k], face[k + 1]])
        off += c
    return np.asarray(tris, dtype=np.int64)


def _load_mesh(stage, prim_path, xform_cache):
    """Mesh prim → trimesh.Trimesh (world frame)."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        print(f"  [miss] {prim_path}")
        return None
    m = UsdGeom.Mesh(prim)
    pts = m.GetPointsAttr().Get()
    counts = m.GetFaceVertexCountsAttr().Get()
    idx = m.GetFaceVertexIndicesAttr().Get()
    if pts is None or counts is None or idx is None:
        return None
    P = np.array([[p[0], p[1], p[2]] for p in pts], dtype=np.float64)
    # world transform
    M = xform_cache.GetLocalToWorldTransform(prim)
    M = np.array(M, dtype=np.float64).reshape(4, 4)  # row-major, row-vector convention
    Ph = np.hstack([P, np.ones((len(P), 1))]) @ M
    P = Ph[:, :3]
    tris = _triangulate(list(counts), list(idx))
    return trimesh.Trimesh(vertices=P, faces=tris, process=False)


def _merge(meshes):
    meshes = [m for m in meshes if m is not None and len(m.faces)]
    if not meshes:
        return None
    return trimesh.util.concatenate(meshes)


def _decimate(mesh, target_faces):
    if mesh is None or len(mesh.faces) <= target_faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(target_faces)
    except Exception:
        return mesh


def _set_equal_aspect(ax, verts):
    mn = verts.min(axis=0); mx = verts.max(axis=0)
    c = (mn + mx) / 2.0
    r = (mx - mn).max() / 2.0 * 1.1
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)


def _draw(ax, visual, collision, hull, elev, azim):
    allv = []
    if visual is not None:
        pc = Poly3DCollection(visual.vertices[visual.faces], alpha=0.18)
        pc.set_facecolor((0.55, 0.55, 0.58)); pc.set_edgecolor("none")
        ax.add_collection3d(pc); allv.append(visual.vertices)
    if hull is not None:
        # convex hull = 기존(거부된) 근사 — 빨강 wireframe
        edges = hull.vertices[hull.edges_unique]
        lc = Line3DCollection(edges, colors=(0.85, 0.1, 0.1), linewidths=0.6, alpha=0.55)
        ax.add_collection3d(lc); allv.append(hull.vertices)
    if collision is not None:
        # SDF source mesh = 초록(visual 과 일치해야 함)
        pc = Poly3DCollection(collision.vertices[collision.faces], alpha=0.30)
        pc.set_facecolor((0.1, 0.7, 0.2)); pc.set_edgecolor((0.05, 0.35, 0.1, 0.4)); pc.set_linewidth(0.2)
        ax.add_collection3d(pc); allv.append(collision.vertices)
    if allv:
        _set_equal_aspect(ax, np.vstack(allv))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_box_aspect((1, 1, 1))


def render_element(label, usd, vis_paths, col_paths):
    print(f"[{label}] {usd.name}")
    stage = Usd.Stage.Open(str(usd))
    xc = UsdGeom.XformCache()
    visual = _merge([_load_mesh(stage, p, xc) for p in vis_paths])
    collision = _merge([_load_mesh(stage, p, xc) for p in col_paths])
    visual = _decimate(visual, 4000)
    collision_full = collision
    collision = _decimate(collision, 4000)
    hull = None
    if collision_full is not None:
        try:
            hull = collision_full.convex_hull
        except Exception as e:
            print("  [hull fail]", e)

    fig = plt.figure(figsize=(13, 5.2))
    fig.suptitle(
        f"{label}  —  visual(gray)  vs  SDF collision source(green, conforms)  vs  old convexHull(red, inflated)",
        fontsize=11,
    )
    for i, (elev, azim) in enumerate([(18, -60), (18, 30), (80, -90)]):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        _draw(ax, visual, collision, hull, elev, azim)
        ax.set_title(f"view {i+1}", fontsize=9)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{label}.png"
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    # 통계
    def _nf(m): return 0 if m is None else len(m.faces)
    print(f"  visual_faces={_nf(visual)} collision_faces(full)={_nf(collision_full)} hull_faces={_nf(hull)}")
    print(f"  saved -> {out}")
    return out


def main():
    for label, usd, vp, cp in ELEMENTS:
        render_element(label, usd, vp, cp)


if __name__ == "__main__":
    main()
