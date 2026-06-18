"""SO-101 follower USD 의 jaw/gripper 충돌 근사를 convexDecomposition → SDF Mesh 로 교체.

배경: jaw/gripper 는 동적 articulation 링크라 triangle mesh·meshSimplification 충돌이
PhysX 에서 convexHull 로 fallback 된다(부정확·부풀림). 동적 body 에서 오목/실제 형상을
정확히 표현하는 유일한 근사 = SDF(signed distance field). convexDecomposition(현재)은
손가락 사이 오목면을 메워 collision 이 visual 보다 크게 잡힌다(grasp 표면 비현실).

대상 prim(둘 다 Xform, UsdPhysics.MeshCollisionAPI 보유, /colliders/* 참조):
  /so101_new_calib/jaw/collisions
  /so101_new_calib/gripper/collisions

PhysxSDFMeshCollisionAPI 는 isaacsim 번들 스키마라, 여기서는 usd-core 만으로 동작하도록
AddAppliedSchema + CreateAttribute(raw 스키마 문자열) 로 author 한다(author_pick_cube_scene
의 bowl convexDecomposition 패턴과 동일). Isaac 부팅 불필요.

실행:
    .venv/bin/python scripts/assets/set_gripper_jaw_sdf_collision.py
    # 옵션: --resolution 256  --no-backup
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from pxr import Usd, UsdPhysics, Sdf

ROBOT_USD = Path(__file__).resolve().parents[2] / "assets" / "robots" / "so101_follower.usd"

# SDF 를 적용할 충돌 prim (jaw + gripper). 팔 링크(base/shoulder/upper_arm/lower_arm/wrist)는
# grasp 무관 → convexDecomposition 유지(저비용). 캠홀더(convexHull)도 단순형상이라 그대로.
TARGET_COLLISION_PRIMS = [
    "/so101_new_calib/jaw/collisions",
    "/so101_new_calib/gripper/collisions",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=256,
                        help="SDF 격자 해상도(긴 축 기준). 256=비용/정밀 균형(NVIDIA 기본).")
    parser.add_argument("--no-backup", action="store_true", help="원본 .bak 백업 생략")
    parser.add_argument("--usd", default=str(ROBOT_USD), help="대상 로봇 USD 경로")
    args = parser.parse_args()

    usd_path = Path(args.usd)
    if not usd_path.exists():
        raise FileNotFoundError(usd_path)

    if not args.no_backup:
        bak = usd_path.with_suffix(usd_path.suffix + ".preSDF.bak")
        if not bak.exists():
            shutil.copy2(usd_path, bak)
            print(f"[backup] {usd_path.name} -> {bak.name}")
        else:
            print(f"[backup] 이미 존재: {bak.name} (덮어쓰지 않음)")

    stage = Usd.Stage.Open(str(usd_path))
    changed = 0
    for path in TARGET_COLLISION_PRIMS:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            print(f"[MISS] prim 없음: {path}")
            continue
        if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            print(f"[WARN] MeshCollisionAPI 없음(그래도 적용 시도): {path}")

        # 1) approximation token: convexDecomposition → sdf
        mc = UsdPhysics.MeshCollisionAPI.Apply(prim)
        prev = prim.GetAttribute("physics:approximation").Get()
        mc.CreateApproximationAttr().Set("sdf")

        # 2) PhysxSDFMeshCollisionAPI (raw 스키마 문자열 — isaac 불요)
        if "PhysxSDFMeshCollisionAPI" not in prim.GetAppliedSchemas():
            prim.AddAppliedSchema("PhysxSDFMeshCollisionAPI")
        prim.CreateAttribute(
            "physxSDFMeshCollision:sdfResolution", Sdf.ValueTypeNames.Int
        ).Set(int(args.resolution))

        print(f"[ok] {path}: approx {prev} -> sdf  (sdfResolution={args.resolution})")
        changed += 1

    if changed == 0:
        print("[done] 변경 없음")
        return
    stage.GetRootLayer().Save()
    print(f"[done] {changed} prim SDF 적용 + 저장: {usd_path}")

    # 검증 재로드
    verify = Usd.Stage.Open(str(usd_path))
    for path in TARGET_COLLISION_PRIMS:
        p = verify.GetPrimAtPath(path)
        if p and p.IsValid():
            ap = p.GetAttribute("physics:approximation").Get()
            res = p.GetAttribute("physxSDFMeshCollision:sdfResolution").Get()
            print(f"[verify] {path}: approx={ap} sdfResolution={res} schemas={list(p.GetAppliedSchemas())}")


if __name__ == "__main__":
    main()
