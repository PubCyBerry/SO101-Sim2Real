"""ovphysx P1용 wrapper USD 생성 (메인 .venv 에서 실행).

robot + scene 을 단일 USD stage 에 합성해서 ovphysx 가 한 번의 add_usd 로 로드하게 함.
"""

from __future__ import annotations

import os
from pxr import Usd, UsdGeom, Sdf, Gf, UsdPhysics

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROBOT_USD = os.path.join(_REPO, "assets", "robots", "so101_follower.usd")
SCENE_USD = os.path.join(_REPO, "assets", "scenes", "cube_desk", "scene.usd")
OUTPUT_USD = os.path.join(_REPO, "outputs", "ovphysx_combined.usda")

# pick_cube_env_cfg.py 의 초기 위치 (recenter: robot base → world 원점 XY)
ROBOT_POS = (0.0, 0.0, 0.6749)
ROBOT_ROT = (0.0, 0.0, 0.0, 1.0)  # (w, x, y, z) identity


def main():
    print(f"[author] Creating combined USD at {OUTPUT_USD}")

    # 출력 디렉터리 확인
    os.makedirs(os.path.dirname(OUTPUT_USD), exist_ok=True)

    # 새 stage 생성 (기존 파일 있으면 덮어쓰기 위해 선삭제)
    if os.path.exists(OUTPUT_USD):
        os.remove(OUTPUT_USD)
    st = Usd.Stage.CreateNew(OUTPUT_USD)
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    st.SetMetadata("metersPerUnit", 1.0)

    # World root
    w = UsdGeom.Xform.Define(st, "/World")
    st.SetDefaultPrim(w.GetPrim())

    # Robot: reference (world-level prim, references robot.usd)
    # 주의: robot.usd 의 defaultPrim(/so101_new_calib) 가 references 되므로
    # prim path 는 /World/Robot/so101_new_calib/base 가 된다.
    # 부모 Xform 래퍼에 sim 배치 transform, 자식 prim 에 robot reference.
    # (robot 의 defaultPrim 이 이미 xformOp 를 가져 같은 prim 에 추가 시 충돌 → 분리)
    # 안 주면 로봇이 원점에 남아 큐브(world ~1.8m)와 멀어 팔이 못 닿음(round2 -1mm 원인).
    robot_xform = UsdGeom.Xform.Define(st, "/World/Robot")
    robot_xform.AddTranslateOp().Set(Gf.Vec3d(*ROBOT_POS))
    w, x, y, z = ROBOT_ROT  # (w,x,y,z)=(0,0,0,1) → z축 180° (sim init_state.rot 과 동일)
    robot_xform.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    robot_prim = st.DefinePrim("/World/Robot/inst")
    robot_prim.GetReferences().AddReference(ROBOT_USD)
    print(f"[author] Robot: /World/Robot(xform tr={ROBOT_POS} rot={ROBOT_ROT})/inst→ref")
    print(f"[author] articulation prim path: /World/Robot/inst/so101_new_calib/base")

    # Scene: reference (world-level prim, references scene.usd)
    # scene.usd 의 모든 물리 객체(큐브/그릇/조명)가 포함됨
    scene_prim = st.DefinePrim("/World/Scene")
    scene_prim.GetReferences().AddReference(SCENE_USD)
    print(f"[author] Scene reference added (prim path: /World/Scene/<objects>)")
    print(f"[author] Scene loaded (with internal SCENE_OFFSET)")

    # base 를 world 에 FixedJoint 로 앵커 = fixed-base articulation.
    # robot USD 는 floating base(is_fixed_base=False) — sim 에선 IsaacLab fix_root_link 이 고정.
    # 안 하면 중력·팔 반작용에 base 가 떠돌아 팔이 큐브 못 닿음(round3 진단: jaw↔cube 206mm).
    # 합성 USD 의 articulation root = /World/Robot/inst/base
    fj = UsdPhysics.FixedJoint.Define(st, "/World/Robot/baseFix")
    fj.CreateBody1Rel().SetTargets([Sdf.Path("/World/Robot/inst/base")])
    # body0 비움 → world. frame 을 base 현재 world pose 로 맞춰야(안 그러면 base 를
    # 원점으로 끌어 solver NaN). localPos0=base world pos, localRot0=base world rot(180°yaw).
    fj.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in ROBOT_POS]))
    w, x, y, z = ROBOT_ROT  # (0,0,0,1) = 180° yaw
    fj.CreateLocalRot0Attr().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    fj.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    fj.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    print("[author] FixedJoint(world ↔ /World/Robot/inst/base, frame=base world pose) — fixed base")

    # Save
    st.GetRootLayer().Save()
    print(f"[author] ✓ Combined USD saved: {OUTPUT_USD}")
    print(f"[author] Prim structure:")
    print(f"  /World")
    print(f"    /Robot (translated to {ROBOT_POS})")
    print(f"      /so101 (reference to robot.usd)")
    print(f"    /Scene")
    print(f"      /desk (reference to scene.usd)")


if __name__ == "__main__":
    main()
