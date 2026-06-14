"""SO-101 cuRobo xrdf 빌드 — 공식 RobotBuilder API (PICKCUBE_CUROBO P1).

`ref_repos/curobo/curobo/examples/getting_started/build_robot_model.py` 흐름을 그대로 따른다
(커스텀 sphere 수학 없음, 전부 공식 MorphIt):
1. `fit_collision_spheres(sphere_density=2.0)` — 전 link MorphIt fit (base 는 z=0 clip).
2. thin/clip 으로 baseline 이 거친 link(front_cam·base)만 `refit_link_spheres(sphere_density=4.0)`
   — density↑ 면 MorphIt voxel 격자가 finer 해져 얇은 plate·clip base 도 정상 fit (진단 확인:
   front_cam 0.6→93.9%, base 16→92.6%).
3. `compute_collision_matrix` (self-collision ignore 자동) → `save_xrdf`.

⚠ cspace 에 gripper 포함(6축, 공식 그대로). cuRobo 계획 시 gripper 는 planner 레벨 lock_joints(P2).

실행:
    uv run --no-sync --group isaac python scripts/sim/build_so101_xrdf.py            # xrdf
    uv run --no-sync --group isaac python scripts/sim/build_so101_xrdf.py --visualize # + Viser
"""

from __future__ import annotations

import argparse
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
URDF = os.path.join(ROOT, "assets/robots/urdf/so_arm101.urdf")
ASSET = os.path.join(ROOT, "assets/robots/urdf")
XRDF = os.path.join(ROOT, "assets/robots/so101.xrdf")
YML = os.path.join(ROOT, "assets/robots/so101_curobo.yml")  # cuRobo native(mesh_link_names 포함, MPC/Viser용)
TOOL = "gripper_frame_link"

# baseline density 가 거친 link → 개별 density↑ refit (전부 MorphIt, 공식).
REFIT_DENSITY = {
    "front_cam_mount_link": 4.0,   # 5mm plate
    "base_link": 4.0,              # clip + 고정 mount
}

# 퇴화(degenerate) 작은 구 제거 임계 — clip 하드클램프·과할당이 남기는 r≈0.002 디커플 구.
# 0.005 미만은 충돌 기여 거의 0 + 시각적으로 떠다니는 잔여라 제거(정상 finger-tip 구 0.0077 등은 유지).
MIN_SPHERE_RADIUS = 0.005


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--viz_port", type=int, default=8085)
    ap.add_argument("--sphere_density", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import numpy as np
    import torch

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from curobo.robot_builder import RobotBuilder

    builder = RobotBuilder(urdf_path=URDF, asset_path=ASSET, tool_frames=[TOOL])
    builder.fit_collision_spheres(
        sphere_density=args.sphere_density,
        clip_links={"base_link": ("z", 0.0)},
        compute_metrics=True,
    )
    # 거친 link 만 density↑ 로 재fit (base 는 clip 유지)
    for link, dens in REFIT_DENSITY.items():
        clip = ((0.0, 0.0, 1.0), 0.0) if link == "base_link" else None
        builder.refit_link_spheres(
            link, sphere_density=dens, compute_metrics=True, clip_plane=clip
        )

    # 퇴화 작은 구 제거 — build() 전에 in-memory 필터(저장·Viser 둘 다 반영).
    # _collision_spheres[link] = [{"center":[x,y,z],"radius":r}, ...] 또는 [x,y,z,r] 리스트.
    def _radius(s):
        return float(s["radius"]) if isinstance(s, dict) else float(s[3])

    removed = 0
    cs = builder._collision_spheres or {}
    for ln in list(cs.keys()):
        keep = [s for s in cs[ln] if _radius(s) >= MIN_SPHERE_RADIUS]
        removed += len(cs[ln]) - len(keep)
        cs[ln] = keep
    if removed:
        print(f"[build] 퇴화 구 제거: {removed}개 (r<{MIN_SPHERE_RADIUS})")

    builder.compute_collision_matrix(prune_collisions=True, num_samples=1000)
    config = builder.build()
    builder.save_xrdf(config, XRDF)       # Isaac Sim/Lab 용
    builder.save(config, YML)             # cuRobo native(mesh_link_names) — MPC/Viser/planner 용

    print(f"\n[build] total {builder.num_spheres} spheres / {len(builder.collision_link_names)} links")
    for ln, m in builder.link_metrics.items():
        print(f"  {ln:<28s} n={m.num_spheres:3d} cover={m.coverage*100:5.1f}% "
              f"gap={m.surface_gap_mean*1000:5.2f}mm protr={m.protrusion*100:5.1f}%")
    print(f"[build] saved → {XRDF}")

    if args.visualize:
        print(f"[build] Viser: http://localhost:{args.viz_port}")
        builder.visualize(config, port=args.viz_port, show_meshes=True, show_spheres=True)
        import time

        try:
            while True:
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
