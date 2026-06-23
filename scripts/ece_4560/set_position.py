import time
from so101_utils import load_calibration, move_to_pose, hold_position, setup_motors

# CONFIGURATION VARIABLES
PORT_ID = "COM8" # REPLACE WITH YOUR PORT!
ROBOT_NAME = "so101_robot" # REPLACE WITH YOUR ROBOT NAME!

# --- Specified Parameters ---
'''
This is the format of the goal position dictionary used for goal position sync write.
The gripper command takes values of 0-100, while the other joints take values of degrees, based on
the settings specified in the bus initialization.
'''
desired_position = {
    'shoulder_pan': 0.0,   # degrees
    'shoulder_lift': 0.0,
    'elbow_flex': 0.0,
    'wrist_flex': 0.0,
    'wrist_roll': 0.0,
    'gripper': 10.0           # 0-100 range
}
move_time = 2.0  # seconds to reach desired position
hold_time = 2.0  # total time to hold at

# ------------------------

# 캘리브레이션 로드 + 모터 버스 셋업
calibration = load_calibration(ROBOT_NAME)
bus = setup_motors(calibration, PORT_ID)

# 시작 위치 기록 (마지막에 복귀용)
starting_pose = bus.sync_read("Present_Position")

# 1) desired_position 으로 move_time 동안 이동
move_to_pose(bus, desired_position, move_time)

# 2) desired_position 에서 hold_time 동안 유지
hold_position(bus, hold_time)

# 3) 시작 위치로 move_time 동안 복귀
move_to_pose(bus, starting_pose, move_time)

# 모터 토크 해제
bus.disable_torque()
