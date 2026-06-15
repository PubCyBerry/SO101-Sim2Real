"""카메라 홀더 collision 추가 — WristCamMount / BellyCamMount.

WristCamMount/holder, BellyCamMount/holder mesh 에 직접 PhysicsCollisionAPI +
PhysicsMeshCollisionAPI 부여. 이 mesh 들은 각각 gripper / shoulder rigid body 의
자식이므로 PhysX 가 해당 body 충돌 shape 으로 자동 등록.

approximation=convexHull: 카메라 홀더는 단순 형상이라 convexDecomposition 불필요.
physxCollision contactOffset/restOffset 은 gripper/collisions 와 동일(0.002/0.0).

Usage:
    uv run python scripts/assets/add_cam_holder_collision.py
    uv run python scripts/assets/add_cam_holder_collision.py --dry_run
"""

import argparse
from pxr import Usd, UsdPhysics, Sdf

ROBOT_USD = "assets/robots/so101_follower.usd"

# (holder mesh 경로, 소속 rigid body link 경로)
CAM_HOLDER_MESHES = [
    (
        "/so101_new_calib/gripper/WristCamMount/holder",
        "/so101_new_calib/gripper",   # PhysicsRigidBodyAPI 있는 부모
    ),
    (
        "/so101_new_calib/shoulder/BellyCamMount/holder",
        "/so101_new_calib/shoulder",
    ),
]

CONTACT_OFFSET = 0.002
REST_OFFSET = 0.0
APPROXIMATION = "convexHull"  # 단순 형상 → convexHull 충분


def add_cam_holder_collision(usd_path: str, dry_run: bool = False) -> None:
    stage = Usd.Stage.Open(usd_path)

    for mesh_path, rb_path in CAM_HOLDER_MESHES:
        mesh_prim = stage.GetPrimAtPath(mesh_path)
        if not mesh_prim.IsValid():
            print(f"[SKIP] {mesh_path} 없음")
            continue

        rb_prim = stage.GetPrimAtPath(rb_path)
        if not rb_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            print(f"[WARN] {rb_path} 에 PhysicsRigidBodyAPI 없음 — 계속 진행")

        if mesh_prim.HasAPI(UsdPhysics.CollisionAPI):
            print(f"[SKIP] {mesh_path} 이미 CollisionAPI 존재")
            continue

        if not dry_run:
            # PhysicsCollisionAPI
            col_api = UsdPhysics.CollisionAPI.Apply(mesh_prim)
            col_api.CreateCollisionEnabledAttr(True)

            # PhysicsMeshCollisionAPI
            mesh_col_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
            mesh_col_api.CreateApproximationAttr(APPROXIMATION)

            # physxCollision 속성
            mesh_prim.CreateAttribute(
                "physxCollision:contactOffset", Sdf.ValueTypeNames.Float
            ).Set(CONTACT_OFFSET)
            mesh_prim.CreateAttribute(
                "physxCollision:restOffset", Sdf.ValueTypeNames.Float
            ).Set(REST_OFFSET)

        print(f"[ADD] {mesh_path}")
        print(f"      rigidBody={rb_path}")
        print(f"      approximation={APPROXIMATION}")
        print(f"      contactOffset={CONTACT_OFFSET}  restOffset={REST_OFFSET}")

    if dry_run:
        print("\ndry_run: USD 저장 안 함")
        return

    stage.GetRootLayer().Save()
    print(f"\n저장 완료: {usd_path}")


def verify(usd_path: str) -> None:
    stage = Usd.Stage.Open(usd_path)
    print("\n=== 검증 ===")
    for mesh_path, _ in CAM_HOLDER_MESHES:
        prim = stage.GetPrimAtPath(mesh_path)
        if not prim.IsValid():
            print(f"[FAIL] {mesh_path} 없음")
            continue
        has_col = prim.HasAPI(UsdPhysics.CollisionAPI)
        has_mesh = prim.HasAPI(UsdPhysics.MeshCollisionAPI)
        approx = prim.GetAttribute("physics:approximation").Get()
        enabled = prim.GetAttribute("physics:collisionEnabled").Get()
        contact = prim.GetAttribute("physxCollision:contactOffset").Get()
        status = "OK" if (has_col and has_mesh) else "FAIL"
        print(f"[{status}] {mesh_path}")
        print(f"     CollisionAPI={has_col}  MeshCollisionAPI={has_mesh}")
        print(f"     approximation={approx}  collisionEnabled={enabled}  contactOffset={contact}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    add_cam_holder_collision(ROBOT_USD, dry_run=args.dry_run)
    if not args.dry_run:
        verify(ROBOT_USD)
