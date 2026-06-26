"""USD 스테이지 파싱 헬퍼 — leisaac ``utils/general_assets.py`` 의 순수 pxr subset.

GPU·isaaclab·isaacsim 불요(pxr 만). author/collision 스크립트(USD 직접 편집)에서 재사용.
원본 하위의 ``parse_usd_and_create_subassets``/``spawn_from_prim_path`` 는 isaaclab+isaacsim 에
결합돼 있어 제외했다(우리 author 패턴은 self-contained USD 라 불요).
"""

from __future__ import annotations

from pxr import Usd, UsdGeom, UsdPhysics


def get_all_prims(stage, prim=None, prims_list=None):
    """스테이지(또는 prim) 하위 모든 prim 을 깊이우선으로 평탄화 수집."""
    if prims_list is None:
        prims_list = []
    if prim is None:
        prim = stage.GetPseudoRoot()
    for child in prim.GetChildren():
        prims_list.append(child)
        get_all_prims(stage, child, prims_list)
    return prims_list


def classify_prim(prim):
    """prim 을 Articulation / RigidBody / Normal 로 분류."""
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        return "Articulation"
    elif prim.HasAPI(UsdPhysics.RigidBodyAPI):
        return "RigidBody"
    else:
        return "Normal"


def is_articulation_root(prim):
    return prim.HasAPI(UsdPhysics.ArticulationRootAPI)


def is_rigidbody(prim):
    return prim.HasAPI(UsdPhysics.RigidBodyAPI)


def get_all_joints(stage):
    """스테이지 전체의 UsdPhysics.Joint prim 목록."""
    joints = []

    def recurse(prim):
        if UsdPhysics.Joint(prim):
            joints.append(prim)
        for child in prim.GetChildren():
            recurse(child)

    recurse(stage.GetPseudoRoot())
    return joints


def get_stage(usd_path):
    return Usd.Stage.Open(usd_path)


def get_prim_pos_rot(prim):
    """prim 의 world-frame 위치(list[3]) + 회전 quat(list[4] wxyz). 비-xformable 이면 (None, None)."""
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        return None, None
    matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    if matrix.Orthonormalize(issueWarning=True):
        rot = matrix.ExtractRotationQuat()
        rot_list = [rot.GetReal(), rot.GetImaginary()[0], rot.GetImaginary()[1], rot.GetImaginary()[2]]
    else:
        rot_list = [1, 0, 0, 0]
    pos = matrix.ExtractTranslation()
    pos_list = list(pos)
    return pos_list, rot_list


def get_articulation_joints(articulation_prim):
    """articulation prim 하위의 UsdPhysics.Joint 목록."""
    joints = []

    def recurse(prim):
        if UsdPhysics.Joint(prim):
            joints.append(prim)
        for child in prim.GetChildren():
            recurse(child)

    recurse(articulation_prim)
    return joints


def get_joint_type(joint_prim):
    return UsdPhysics.Joint(joint_prim).GetTypeName()


def is_fixed_joint(prim):
    return prim.GetTypeName() == "PhysicsFixedJoint"


def is_revolute_joint(prim):
    return prim.GetTypeName() == "PhysicsRevoluteJoint"


def is_prismatic_joint(prim):
    return prim.GetTypeName() == "PhysicsPrismaticJoint"


def get_joint_name_and_qpos(joint_prim):
    joint = UsdPhysics.Joint(joint_prim)
    return joint.GetName(), joint.GetPositionAttr().Get()


def get_all_joints_without_fixed(articulation_prim):
    joints = get_articulation_joints(articulation_prim)
    return [joint for joint in joints if not is_fixed_joint(joint)]
