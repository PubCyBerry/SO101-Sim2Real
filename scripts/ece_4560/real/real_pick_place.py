import time
from so101_utils import load_calibration, move_to_pose, hold_position, setup_motors

# CONFIGURATION VARIABLES
PORT_ID = "COM8"            # REPLACE WITH YOUR PORT!
ROBOT_NAME = "so101_robot"  # REPLACE WITH YOUR ROBOT NAME!

# 실기기 명령 규약: 팔 5축 = degrees, gripper = 0-100 (bus 초기화 norm mode 기준).
# 시뮬과 동일한 6개 구성을 그대로 쓴다 (sim 은 dict 를 rad 로 변환하지만,
# 실기기 sync_write(normalize=True) 는 degrees/0-100 을 직접 받으므로 변환 불필요).

# ---- 6개 조인트 구성 (pick-and-place 순서, mujoco_pick_place.py 와 동일) ----
# 1) 시작: pick 위치(pan=-45) 위에서 그리퍼 열림
starting_configuration = {
    'shoulder_pan': -45.0,   # degrees
    'shoulder_lift': 45.0,
    'elbow_flex': -45.00,
    'wrist_flex': 90.0,
    'wrist_roll': 0.0,
    'gripper': 50            # 0-100
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
lift_pick = {
    'shoulder_pan': -45.0,
    'shoulder_lift': 0.0,
    'elbow_flex': 0.00,
    'wrist_flex': 900.0,
    'wrist_roll': 0.0,
    'gripper': 5
}
# 4) 들어올린 자세 그대로 place 쪽(pan=+45)으로 수평 이동
lift_place = {
    'shoulder_pan': 45.0,
    'shoulder_lift': 0.0,
    'elbow_flex': 0.00,
    'wrist_flex': 90.0,
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
    'shoulder_pan': 45.0,    # degrees
    'shoulder_lift': 45.0,
    'elbow_flex': -45.00,
    'wrist_flex': 90.0,
    'wrist_roll': 0.0,
    'gripper': 50
}

# ------------------------

# 캘리브레이션 로드 + 모터 버스 셋업
calibration = load_calibration(ROBOT_NAME)
bus = setup_motors(calibration, PORT_ID)

# 실제 시작 위치 기록 (마지막에 복귀해 책상에 부딪히는 것 방지)
starting_pose = bus.sync_read("Present_Position")

# 1) 첫 desired 구성(pick, 그리퍼 열림)으로 이동
move_to_pose(bus, starting_configuration, 2.0)
hold_position(bus, 1.0)

# 1->2) 그리퍼 닫아 큐브 파지
move_to_pose(bus, starting_configuration_closed, 1.5)
hold_position(bus, 1.0)

# 2->3) 팔 들어올림
move_to_pose(bus, lift_pick, 2.0)

# 3->4) place 쪽으로 수평 이동
move_to_pose(bus, lift_place, 2.5)

# 4->5) place 위치로 하강 (닫힘 유지)
move_to_pose(bus, final_configuration_closed, 2.0)
hold_position(bus, 1.0)

# 5->6) 그리퍼 열어 큐브 내려놓음
move_to_pose(bus, final_configuration, 1.5)
hold_position(bus, 1.0)

# 실제 시작 위치로 복귀 (종료 시 책상 충돌 방지)
move_to_pose(bus, starting_pose, 2.0)

# 모터 토크 해제
bus.disable_torque()
