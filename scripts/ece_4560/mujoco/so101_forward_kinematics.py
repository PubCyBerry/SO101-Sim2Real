import time
import mujoco
import numpy as np

def Rx(thetadeg):
    thetarad = np.deg2rad(thetadeg)
    c = np.cos(thetarad)
    s = np.sin(thetarad)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])

def Ry(thetadeg):
    thetarad = np.deg2rad(thetadeg)
    c = np.cos(thetarad)
    s = np.sin(thetarad)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])

def Rz(thetadeg):
    thetarad = np.deg2rad(thetadeg)
    c = np.cos(thetarad)
    s = np.sin(thetarad)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

# 각 g 는 부모 프레임 -> 자식 프레임 동차변환:
#   g = T(body.pos) @ R(고정 자세) @ Rz(joint_angle)
# 고정 자세 = MJCF body 의 quat 을 Rx/Ry/Rz 곱으로 분해한 것.
# MJCF 의 모든 hinge 축이 로컬 z(0 0 1)라 관절 회전은 항상 Rz(theta).
# theta 는 deg (테스트 설정 dict 가 deg, so101_mujoco_utils 가 deg 가정).

def get_gw1(theta1_deg):
    # body "shoulder" : quat(0,0,-1,0) == Rz(180)@Rx(180) == Ry(180)
    displacement = (0.0388353, 0.0, 0.0624)
    rotation = Rz(180) @ Rx(180) @ Rz(theta1_deg)
    pose = np.block([[rotation, np.array(displacement).reshape(3,1)], [0, 0, 0, 1]])
    return pose

def get_g12(theta2_deg):
    # body "upper_arm" : quat(0.5,-0.5,-0.5,-0.5) == Rx(-90)@Rz(-90)
    displacement = (-0.0303992, -0.0182778, -0.0542)
    rotation = Rx(-90) @ Rz(-90) @ Rz(theta2_deg)
    pose = np.block([[rotation, np.array(displacement).reshape(3,1)], [0, 0, 0, 1]])
    return pose

def get_g23(theta3_deg):
    # body "lower_arm" : quat(0.707107,0,0,0.707107) == Rz(90)
    displacement = (-0.11257, -0.028, 0.0)
    rotation = Rz(90) @ Rz(theta3_deg)
    pose = np.block([[rotation, np.array(displacement).reshape(3,1)], [0, 0, 0, 1]])
    return pose

def get_g34(theta4_deg):
    # body "wrist" : quat(0.707107,0,0,-0.707107) == Rz(-90)
    displacement = (-0.1349, 0.0052, 0.0)
    rotation = Rz(-90) @ Rz(theta4_deg)
    pose = np.block([[rotation, np.array(displacement).reshape(3,1)], [0, 0, 0, 1]])
    return pose

def get_g45(theta5_deg):
    # body "gripper" : quat(0.0172091,-0.0172091,0.706897,0.706897)
    #   == Rz(180)@Rx(90)@Rz(-2.78913075)  (-2.789° = SO-101 캘리브 틸트)
    displacement = (0.0, -0.0611, 0.0181)
    rotation = Rz(180) @ Rx(90) @ Rz(-2.78913075) @ Rz(theta5_deg)
    pose = np.block([[rotation, np.array(displacement).reshape(3,1)], [0, 0, 0, 1]])
    return pose

def get_g5t():
    # tool 프레임 = 두 jaw 사이 "파지 중심"(fixed/moving 손가락 끝 중점), 관절 없음.
    #   - 위치: MJCF site "gripperframe"(-0.0079,…)은 fixed jaw 끝이라 한쪽에 치우친다.
    #     두 손가락 끝 중점 (0.0128, -0.0002, -0.090) 으로 옮겨 그리퍼 정중앙에 둔다.
    #     (x=closing 축 방향 중앙, z=손가락 길이 중간 깊이; gripper=10 개도 기준)
    #   - 자세: Rx(90) 으로 tool z 를 gripper y축(= 닫는 축에 수직인 가로축)에 맞춘다.
    #     gripper x(닫는 축)면 실린더가 누워 닫는 축과 평행, gripper z면 수직으로 선다.
    #     tool z = gripper y 라야 실린더가 그리퍼 사이에 "장축으로" 물린다.
    displacement = (0.0128, -0.0002, -0.090)
    rotation = Rx(90)
    pose = np.block([[rotation, np.array(displacement).reshape(3,1)], [0, 0, 0, 1]])
    return pose

def get_forward_kinematics(position_dict):
    gw1 = get_gw1(position_dict['shoulder_pan'])
    g12 = get_g12(position_dict['shoulder_lift'])
    g23 = get_g23(position_dict['elbow_flex'])
    g34 = get_g34(position_dict['wrist_flex'])
    g45 = get_g45(position_dict['wrist_roll'])
    g5t = get_g5t()
    gwt = gw1 @ g12 @ g23 @ g34 @ g45 @ g5t
    position = gwt[0:3, 3]
    rotation = gwt[0:3, 0:3]
    return position, rotation
