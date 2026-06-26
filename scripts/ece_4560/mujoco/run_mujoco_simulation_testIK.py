import time
import mujoco
import mujoco.viewer
from so101_mujoco_utils import set_initial_pose, send_position_command, convert_to_dictionary
from so101_inverse_kinematics import get_inverse_kinematics
import numpy as np

# 이 폴더엔 model/ 이 없어 다른 스크립트와 동일하게 ref_repos scene.xml 사용
m = mujoco.MjModel.from_xml_path(r'C:\Users\taehunkim\Workspace\SO101-LeRobot-VLA\ref_repos\SO-ARM100\Simulation\SO101\scene.xml')
d = mujoco.MjData(m)

# Helper Function to show a cube at a given position and orientation
def show_cube(viewer, position, orientation, geom_num=0, rgba=[1, 0, 0, 0.4], halfwidth=0.013):
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[geom_num],
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[halfwidth, halfwidth, halfwidth],
        pos=np.asarray(position, float),
        mat=orientation.flatten(),
        rgba=rgba
    )
    viewer.user_scn.ngeom = geom_num + 1
    viewer.sync()
    return

# so101_mujoco_utils 의 move_to_pose/hold_position 은 wall-clock 기반이라
# 뷰어 Vertical Sync(보통 60fps)가 켜지면 1초에 물리가 ~0.12초만 진행돼 팔이
# 목표까지 못 가고, hold_position 은 그 "미도달 자세"를 그대로 동결한다
# (그래서 그리퍼가 큐브에서 한참 떨어진 채 멈춘다). 아래 로컬 버전은
#  (1) 프레임당 여러 번 mj_step(substeps) 해서 렌더 속도와 무관하게 충분한
#      물리 스텝을 보장하고,
#  (2) 정지 구간에서 스냅샷이 아니라 "목표 ctrl" 을 명령해 끝까지 수렴시킨다.
def move_to_pose(m, d, viewer, desired_position, duration, substeps=10):
    start = convert_to_dictionary(d.qpos.copy())
    nframes = max(1, int(duration / m.opt.timestep / substeps))
    for f in range(nframes):
        alpha = min((f + 1) / nframes, 1.0)
        pos = {j: (1 - alpha) * start[j] + alpha * desired_position[j] for j in desired_position}
        send_position_command(d, pos)
        for _ in range(substeps):
            mujoco.mj_step(m, d)
        viewer.sync()

def hold_position(m, d, viewer, duration, substeps=10):
    held_ctrl = d.ctrl.copy()   # 직전 move 의 목표 ctrl (현재 qpos 스냅샷 아님)
    nframes = max(1, int(duration / m.opt.timestep / substeps))
    for f in range(nframes):
        d.ctrl[:] = held_ctrl
        for _ in range(substeps):
            mujoco.mj_step(m, d)
        viewer.sync()

# Helper function to obtain random target position and yaw (rotation around the z-axis)
# from a given range.
#   x/y 범위 = workspace 분석(2a)에서 도달 가능(IK non-NaN)하고 joint limit 안에
#   드는 사각형. base 바로 위(특이점)와 먼 코너(NaN)를 피한 안전 영역이다.
#   특히 base 근처(x<0.15, y≈0)는 도달은 되지만 자세가 극단적이라 물리 정착이
#   잘 안 돼 그리퍼가 큐브에 안 맞는다 → x 하한을 0.15 로 올려 그 영역을 제외.
def get_random_position():
    x_pos_range = [0.15, 0.22]    # taken from workspace analysis
    y_pos_range = [-0.13, 0.13]   # taken from workspace analysis
    yaw_range = [0, np.pi / 2]    # 큐브 대칭이라 0~90도 밖은 중복
    x = np.random.uniform(x_pos_range[0], x_pos_range[1])
    y = np.random.uniform(y_pos_range[0], y_pos_range[1])
    yaw = np.random.uniform(yaw_range[0], yaw_range[1])
    return [x, y, 0.014], yaw

# Initial joint configuration at start of simulation
initial_config = {
    'shoulder_pan': 0.0,
    'shoulder_lift': 0.0,
    'elbow_flex': 0.00,
    'wrist_flex': 0.0,
    'wrist_roll': 0.0,
    'gripper': 0
}
set_initial_pose(d, initial_config)
send_position_command(d, initial_config)

# Start simulation with mujoco viewer
with mujoco.viewer.launch_passive(m, d) as viewer:

    # Pause for 10 seconds in order to make screen recording easier
    time.sleep(1)

    while viewer.is_running():

        for i in range(5):
            desired_position, desired_yaw = get_random_position()
            # 큐브가 대칭이라 자세는 항상 위에서 아래로(grasp-from-above) 고정.
            # IK 내부에서 tool z 를 월드 -z 로 맞추고 theta5=-theta1 로 푼다.
            desired_orientation = np.eye(3)

            # 빨강 큐브로 목표 위치 표시
            show_cube(viewer, desired_position, desired_orientation)

            # Get the inverse kinematics solution
            joint_configuration = get_inverse_kinematics(desired_position, desired_orientation)
            joint_configuration['gripper'] = 5  # 큐브 위에 정렬 보이게 거의 닫음

            # Move the robot to the desired pose (substeps 로 충분한 물리 스텝 보장)
            move_to_pose(m, d, viewer, joint_configuration, 1.5)

            # Hold for two seconds for easy visualization
            hold_position(m, d, viewer, 2.0)
