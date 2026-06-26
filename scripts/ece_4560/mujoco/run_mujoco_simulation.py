import mujoco
import mujoco.viewer
from so101_mujoco_utils import set_initial_pose, move_to_pose, hold_position

m = mujoco.MjModel.from_xml_path(r'C:\Users\taehunkim\Workspace\SO101-LeRobot-VLA\ref_repos\SO-ARM100\Simulation\SO101\scene.xml')
d = mujoco.MjData(m)

# 그림에 표시된 시작 자세 (각도=deg, gripper=0~100). 그림에 맞게 값 수정.
starting_position = {
    'shoulder_pan': 0.0,
    'shoulder_lift': -100.0,
    'elbow_flex': 100.0,
    'wrist_flex': 100.0,
    'wrist_roll': 0.0,
    'gripper': 0.0,
}

# 도달 목표 = 전부 0
desired_position = {
    'shoulder_pan': 0.0,
    'shoulder_lift': 0.0,
    'elbow_flex': 0.0,
    'wrist_flex': 0.0,
    'wrist_roll': 0.0,
    'gripper': 0.0,
}

move_duration = 2.0   # 이동 시간 (초)
hold_duration = 2.0   # 유지 시간 (초)

# 시작 자세로 초기화
set_initial_pose(d, starting_position)

with mujoco.viewer.launch_passive(m, d) as viewer:
    # 1) 목표(전부 0)로 2.0초 이동
    move_to_pose(m, d, viewer, desired_position, move_duration)

    # 2) 목표 자세 2.0초 유지
    hold_position(m, d, viewer, hold_duration)

    # 3) 시작 자세로 2.0초 복귀
    move_to_pose(m, d, viewer, starting_position, move_duration)
