import time
import mujoco
import mujoco.viewer
from so101_mujoco_utils import set_initial_pose, send_position_command, move_to_pose, hold_position
from so101_forward_kinematics import get_forward_kinematics
import numpy as np

# 이 폴더엔 model/ 이 없어 test_forward_kinematics.py 와 동일하게 ref_repos scene.xml 사용
m = mujoco.MjModel.from_xml_path(r'C:\Users\taehunkim\Workspace\SO101-LeRobot-VLA\ref_repos\SO-ARM100\Simulation\SO101\scene.xml')
d = mujoco.MjData(m)

def show_cubes(viewer, starting_config, final_config, halfwidth=0.013):
    # Use forward kinematics
    starting_object_position, starting_object_orientation = get_forward_kinematics(starting_config)
    final_object_position, final_object_orientation = get_forward_kinematics(final_config)

    # Add starting cube
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[0],
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[halfwidth, halfwidth, halfwidth],
        pos=starting_object_position,
        mat=starting_object_orientation.flatten(),
        rgba=[1, 0, 0, 0.2]
    )
    # Add final cube
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[1],
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[halfwidth, halfwidth, halfwidth],
        pos=final_object_position,
        mat=final_object_orientation.flatten(),
        rgba=[0, 1, 0, 0.2]
    )
    viewer.user_scn.ngeom = 2
    viewer.sync()
    return

# ---- 6개 조인트 구성 (pick-and-place 순서) ----
# 1) 시작: pick 위치(pan=-45) 위에서 그리퍼 열림
starting_configuration = {
    'shoulder_pan': -45.0,   # in radians for mujoco!
    'shoulder_lift': 45.0,
    'elbow_flex': -45.00,
    'wrist_flex': 90.0,
    'wrist_roll': 0.0,
    'gripper': 50
}
# 2) pick 위치에서 그리퍼 닫음 (큐브 파지)
starting_configuration_closed = {
    'shoulder_pan': -45.0,
    'shoulder_lift': 45.0,
    'elbow_flex': -45.00,
    'wrist_flex': 90.0,
    'wrist_roll': 0.0,
    'gripper': 5
}
# 3) pick 쪽에서 팔을 들어올림 (책상/장애물 회피, 그리퍼 닫힘 유지)
#    shoulder_lift 45->0 으로 tool z 약 0.03 -> 0.23 m 상승. elbow/wrist 는 유지.
lift_pick = {
    'shoulder_pan': -45.0,
    'shoulder_lift': 0.0,
    'elbow_flex': 15.00,
    'wrist_flex': 120.0,
    'wrist_roll': 0.0,
    'gripper': 5
}
# 4) 들어올린 자세 그대로 place 쪽(pan=+45)으로 수평 이동
lift_place = {
    'shoulder_pan': 45.0,
    'shoulder_lift': 0.0,
    'elbow_flex': 15.00,
    'wrist_flex': 120.0,
    'wrist_roll': 0.0,
    'gripper': 5
}
# 5) place 위치로 하강 (그리퍼 아직 닫힘)
final_configuration_closed = {
    'shoulder_pan': 45.0,
    'shoulder_lift': 45.0,
    'elbow_flex': -45.00,
    'wrist_flex': 90.0,
    'wrist_roll': 0.0,
    'gripper': 5
}
# 6) 최종: place 위치에서 그리퍼 열어 큐브 놓음
final_configuration = {
    'shoulder_pan': 45.0,   # in radians for mujoco!
    'shoulder_lift': 45.0,
    'elbow_flex': -45.00,
    'wrist_flex': 90.0,
    'wrist_roll': 0.0,
    'gripper': 50
}

# 시작 자세로 초기화
set_initial_pose(d, starting_configuration)
send_position_command(d, starting_configuration)

with mujoco.viewer.launch_passive(m, d) as viewer:
    # 빨강=pick 큐브(시작), 초록=place 큐브(목표) 시각화
    show_cubes(viewer, starting_configuration, final_configuration)

    # 1) 시작 자세에서 잠시 정지
    hold_position(m, d, viewer, 1.0)

    # 1->2) 그리퍼 닫아 큐브 파지
    move_to_pose(m, d, viewer, starting_configuration_closed, 1.5)
    hold_position(m, d, viewer, 1.0)

    # 2->3) 팔 들어올림
    move_to_pose(m, d, viewer, lift_pick, 2.0)

    # 3->4) place 쪽으로 수평 이동
    move_to_pose(m, d, viewer, lift_place, 2.5)

    # 4->5) place 위치로 하강 (닫힘 유지)
    move_to_pose(m, d, viewer, final_configuration_closed, 2.0)
    hold_position(m, d, viewer, 1.0)

    # 5->6) 그리퍼 열어 큐브 내려놓음
    move_to_pose(m, d, viewer, final_configuration, 1.5)

    # 최종 자세 유지
    hold_position(m, d, viewer, 2.0)
