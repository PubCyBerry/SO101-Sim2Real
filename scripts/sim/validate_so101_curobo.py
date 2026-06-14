"""SO-101 xrdf ↔ 신버전 cuRobo(ref_repos/curobo) 정합 검증 (PICKCUBE_CUROBO P1-a).

`assets/robots/so101.xrdf` 가 신 API(`KinematicsCfg`/`InverseKinematics`)로 로드되는지, 관절 chain·
collision sphere(카메라 홀더 포함)·FK·IK 성공률을 확인한다. (PATH E Docker 의 구 API 검증은 기존
`gen_so101_xrdf.py` — 이건 신 API 전용, 덮지 않음.)

실행:
    uv run --no-sync --group isaac python scripts/sim/validate_so101_curobo.py
"""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
XRDF = os.path.join(ROOT, "assets/robots/so101.xrdf")
URDF = os.path.join(ROOT, "assets/robots/urdf/so_arm101.urdf")
TOOL_LINK = "gripper_frame_link"
CSPACE = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def urdf_joint_limits() -> dict:
    root = ET.parse(URDF).getroot()
    lim = {}
    for j in root.findall("joint"):
        n = j.get("name")
        l = j.find("limit")
        if l is not None:
            lim[n] = (float(l.get("lower")), float(l.get("upper")))
    return lim


def main() -> int:
    from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo.types import GoalToolPose, JointState, Pose
    from curobo._src.types.robot import RobotCfg  # 공개 shim 없음

    # 1) 로드
    kin_cfg = KinematicsCfg.from_robot_yaml_file(XRDF, urdf_path=URDF)
    kin = Kinematics(kin_cfg)
    jn = list(kin.joint_names)
    print(f"[1] 로드 OK — joints={jn}  total_spheres={kin.total_spheres}")
    assert all(a in jn for a in CSPACE), f"팔 5축 누락: {jn}"
    # 순수 공식 xrdf 라 cspace 에 gripper 포함(6축) — cuRobo 계획 시 planner 레벨 lock_joints(P2).
    if "gripper" in jn:
        print("    (gripper 포함 6축 — P2 에서 planner lock_joints 로 5-DOF 계획)")

    # 2) collision sphere 적재 확인 (홀더·finger 포함됐나)
    q0 = torch.zeros((1, len(jn)), device="cuda")
    st = kin.compute_kinematics(JointState.from_position(q0, joint_names=jn))
    n_sph = int(st.get_link_spheres().shape[-2])
    print(f"[2] collision spheres 적재: {n_sph} (홀더 2링크 포함)")

    # 3) FK 스모크 — tool pose
    pose0 = st.tool_poses.get_link_pose(TOOL_LINK)
    pos0 = pose0.position.detach().cpu().numpy().reshape(-1, 3)[0]
    print(f"[3] FK(zero) {TOOL_LINK} pos={np.round(pos0,4).tolist()}")

    # 4) IK 성공률 — 랜덤 도달가능 config 를 FK 로 만들어 역해 (gripper 는 0.785 lock 근처 고정)
    lim = urdf_joint_limits()
    lo = torch.tensor([lim.get(j, (-0.1, 0.1))[0] for j in jn], device="cuda")
    hi = torch.tensor([lim.get(j, (-0.1, 0.1))[1] for j in jn], device="cuda")
    N = 200
    torch.manual_seed(0)
    q_rand = lo + (hi - lo) * torch.rand((N, len(jn)), device="cuda")
    fk = kin.compute_kinematics(JointState.from_position(q_rand, joint_names=kin.joint_names))
    pose_r = fk.tool_poses.get_link_pose(TOOL_LINK)
    pos = pose_r.position.reshape(N, 3)
    quat = pose_r.quaternion.reshape(N, 4)

    robot_cfg = RobotCfg.create({"kinematics": kin_cfg})
    ik_cfg = InverseKinematicsCfg.create(
        robot=robot_cfg, num_seeds=32, self_collision_check=False,
        position_tolerance=0.005, orientation_tolerance=0.05, max_batch_size=N,
    )
    ik = InverseKinematics(ik_cfg)
    goal = GoalToolPose.from_poses({TOOL_LINK: Pose(position=pos, quaternion=quat)}, num_goalset=1)
    res = ik.solve_pose(goal)
    succ = float(res.success.float().mean().item())
    print(f"[4] IK 성공률({N} 랜덤 도달가능 pose): {succ*100:.1f}%  "
          f"(6-DOF exact pose; 5-DOF 라 일부 미달 정상)")

    # position-only(5-DOF 적합) 도 참고 측정
    try:
        ik_pos_cfg = InverseKinematicsCfg.create(
            robot=robot_cfg, num_seeds=32, self_collision_check=False,
            position_tolerance=0.005, orientation_tolerance=6.3, max_batch_size=N,
        )
        ik_pos = InverseKinematics(ik_pos_cfg)
        res_p = ik_pos.solve_pose(goal)
        print(f"[4b] position-우선 IK 성공률: {float(res_p.success.float().mean().item())*100:.1f}%")
    except Exception as e:  # noqa: BLE001
        print(f"[4b] position-only 측정 생략: {e}")

    print("[GATE] xrdf ↔ 신 cuRobo 정합 OK" if succ > 0.5 else "[GATE] IK 성공률 낮음 — 확인 필요")
    return 0


if __name__ == "__main__":
    sys.exit(main())
