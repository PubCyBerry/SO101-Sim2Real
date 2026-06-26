import numpy as np
from so101_forward_kinematics import (
    Rx, Ry, Rz,
    get_gw1, get_g12, get_g23, get_g34, get_g45, get_g5t,
)

# =====================================================================
# Redo Forward Kinematics (tool frame 재정의)
# ---------------------------------------------------------------------
# IK 유도를 쉽게 하려고 tool frame 의 자세를 wrist(frame4) 와 같은 방향으로
# 맞춘다. 즉 위에서 잡을 때(approach 가 수직 아래) tool z 가 월드 -z 와 나란해
# 지도록 한다. 위치(파지 중심)는 기존 get_g5t() 를 그대로 쓰고, 자세만
# R(g4t)=Rx(90) 이 되도록 회전을 교체한다. (FK 파일 자체는 건드리지 않아
#  기존 pick-place/시각화는 그대로 유지된다.)
#   g4t = get_g45(theta5) @ g5t_ik()  (frame4 -> tool)
# =====================================================================
_R_5T_IK = get_g45(0.0)[:3, :3].T @ Rx(90)   # R(g4t)=Rx(90) 이 되게 하는 g5t 회전
_TRANS_5T = get_g5t()[:3, 3]                  # 파지 중심 위치 (기존과 동일)

def get_g5t_ik():
    g = np.eye(4)
    g[:3, :3] = _R_5T_IK
    g[:3, 3] = _TRANS_5T
    return g

# ---- FK 로부터 IK 에 필요한 기하 상수 추출 (zero configuration 기준) ----
_P1 = get_gw1(0.0)[:3, 3]                      # joint1(shoulder_pan) 축 원점, world
_L23 = 0.116                                   # joint2->joint3 링크 길이 (upper arm)
_L34 = 0.135                                   # joint3->joint4 링크 길이 (forearm)

# frame1(=joint1 frame) 기준 joint2/3/4 원점 (zero config)
_inv_gw1_0 = np.linalg.inv(get_gw1(0.0))
def _to_frame1_at0(p_world):
    return (_inv_gw1_0 @ np.append(p_world, 1.0))[:3]
_g120 = get_gw1(0.0) @ get_g12(0.0)
_g130 = _g120 @ get_g23(0.0)
_g140 = _g130 @ get_g34(0.0)
_P2_F1 = _to_frame1_at0(_g120[:3, 3])
_P3_F1 = _to_frame1_at0(_g130[:3, 3])
_P4_F1 = _to_frame1_at0(_g140[:3, 3])

# 평면 2R 의 작업 평면 (joints 2,3,4 는 평행 pitch 라 한 평면에서 움직인다)
_N = np.cross(_P3_F1 - _P2_F1, _P4_F1 - _P3_F1)
_N /= np.linalg.norm(_N)                       # pitch 평면 법선 (frame1 기준 = +y)
_U = (_P3_F1 - _P2_F1) / np.linalg.norm(_P3_F1 - _P2_F1)   # zero 자세 upper-arm 방향
_WP = np.cross(_N, _U)                          # 평면 내 수직 기저
# wrist(frame4) 의 측면(법선 방향) offset = 상수 (theta1 풀이에 사용)
_LAT = float(np.dot(_P4_F1 - _P2_F1, _N) + np.dot(_P2_F1, _N))  # = frame1 기준 frame4 y
_LAT = float(_P4_F1[1])                          # 동일값, 명시적으로 frame1 y 성분
# elbow(theta3) zero 기준 각: theta3=0 일 때의 코사인법칙 내부각
_D0 = np.linalg.norm(_P4_F1 - _P2_F1)
_E0 = np.degrees(np.arccos((_D0**2 - _L23**2 - _L34**2) / (2 * _L23 * _L34)))


def _R_target(theta1_deg=None):
    """grasp-from-above(위에서 아래로) 시 desired tool 자세 (상수).

    tool z = (0,0,-1) (수직 아래), tool x = (-1,0,0) (월드 -x).
    theta5 = theta1 (1f) 이라 base 회전(theta1)과 roll(theta5=theta1)이 상쇄돼
    tool/gripper 자세가 theta1 과 무관하게 월드에 고정된다. 즉 gripper 의
    closing axis 가 항상 월드축(=축정렬 큐브 face 의 법선)과 평행해진다.
    """
    return np.array([[-1.0, 0.0,  0.0],
                     [0.0,  1.0,  0.0],
                     [0.0,  0.0, -1.0]])


# ---------------------------------------------------------------------
# 1c) wrist(joint4) frame 의 desired 위치
#     g_w4 = g_wt @ inv(g_4t),  g_4t = g_45(theta5) @ g5t_ik()
#     theta5 = theta1 이 1f 에서 정해지므로 g_45(theta1) 을 써서 roll 까지
#     정확히 반영한다 (g_45(0) 으로 두면 roll 만큼 wrist 위치 오차가 생긴다).
# ---------------------------------------------------------------------
def get_wrist_flex_position(target_position, theta1=None):
    if theta1 is None:
        theta1 = _solve_theta1(target_position)
    gwt = np.eye(4)
    gwt[:3, :3] = _R_target(theta1)
    gwt[:3, 3] = np.asarray(target_position, float)
    g4t = get_g45(theta1) @ get_g5t_ik()
    gw4 = gwt @ np.linalg.inv(g4t)
    wrist_flex_position = gw4[:3, 3]
    wrist_flex_orientation = gw4[:3, :3]
    return wrist_flex_position, wrist_flex_orientation


def _theta1_from_wrist(wrist_pos, guess_deg):
    """wrist 의 frame1 측면 offset 이 상수(_LAT)라는 제약으로 theta1 을 푼다.

    frame1 에서 wrist y 성분 = sin(t1)*A + cos(t1)*B = _LAT,
    여기서 A=wx-p1x, B=wy-p1y. 두 해 중 tool 방위각 추정치에 가까운 쪽 선택.
    """
    A = wrist_pos[0] - _P1[0]
    B = wrist_pos[1] - _P1[1]
    r = np.hypot(A, B)
    ratio = np.clip(_LAT / r, -1.0, 1.0)
    phi = np.arctan2(B, A)
    wrap = lambda a: (a + 180.0) % 360.0 - 180.0
    cands = [wrap(np.degrees(np.arcsin(ratio) - phi)),
             wrap(np.degrees(np.pi - np.arcsin(ratio) - phi))]
    return min(cands, key=lambda t: abs(wrap(t - guess_deg)))


# ---------------------------------------------------------------------
# 1b) theta1 (shoulder_pan)
#     desired EE 위치를 ground 에 투영해 방위각으로 푼다. joint1 축이 world
#     원점이 아니라 x=0.0388 만큼 떨어져 있으므로(p1) 반드시 빼줘야 한다.
#     wrist 의 측면 offset 때문에 EE 직접 투영은 수 도 오차가 있어, wrist
#     위치로 몇 번 refine 한다 (wrist 가 theta1 에 약하게 의존하기 때문).
# ---------------------------------------------------------------------
def _solve_theta1(target_position):
    px, py, _ = target_position
    theta1 = -np.degrees(np.arctan2(py - _P1[1], px - _P1[0]))   # 초기 추정
    for _ in range(8):
        wrist_pos, _o = get_wrist_flex_position(target_position, theta1)
        theta1 = _theta1_from_wrist(wrist_pos, theta1)
    return theta1


# ---------------------------------------------------------------------
# 1d) theta2(shoulder_lift), theta3(elbow_flex)
#     wrist 위치를 frame1 평면으로 옮겨 joint2-3-4 삼각형에 코사인법칙 적용.
# ---------------------------------------------------------------------
def _solve_shoulder_elbow(wrist_pos, theta1):
    pw_f1 = (np.linalg.inv(get_gw1(theta1)) @ np.append(wrist_pos, 1.0))[:3]
    v = pw_f1 - _P2_F1
    a = np.dot(v, _U)            # 평면 내 좌표 (upper-arm 축)
    b = np.dot(v, _WP)          # 평면 내 좌표 (수직 기저)
    D = np.hypot(a, b)          # joint2 -> wrist 거리
    # 도달 불가(D 가 링크 합/차 범위 밖)면 arccos 인자가 [-1,1] 을 벗어나
    # NaN 이 되어 그대로 전파된다 -> workspace 밖 판정 (2a). clip 하지 않음.
    cos_e = (D**2 - _L23**2 - _L34**2) / (2 * _L23 * _L34)
    e = np.degrees(np.arccos(cos_e))            # 코사인법칙 내부각 (불가시 NaN)
    theta3 = e - _E0
    phi = np.degrees(np.arctan2(b, a))           # wrist 방향 각
    psi = np.degrees(np.arctan2(_L34 * np.sin(np.radians(e)),
                                _L23 + _L34 * np.cos(np.radians(e))))  # 삼각형 joint2 각
    theta2 = phi - psi
    return theta2, theta3


def get_inverse_kinematics(target_position, target_orientation):
    "Geometric appraoch specific to the so-101 arms"
    # 위에서 잡기(grasp-from-above) 가정: tool 이 수직 아래를 향한다.
    # target_orientation 은 인터페이스용으로 받되, 표준 위-잡기에서는
    # theta5 = -theta1 (1f) 로 roll 을 정한다.

    # 1b) theta1 : EE 위치 투영
    theta1 = _solve_theta1(target_position)

    # 1c) wrist(joint4) 의 desired 위치
    target_wrist_position, _wo = get_wrist_flex_position(target_position, theta1)

    # 1d) theta2, theta3 : 코사인법칙
    theta2, theta3 = _solve_shoulder_elbow(target_wrist_position, theta1)

    # 1e) theta4 : tool z 를 world -z 에 맞춤. joints 2,3,4 가 평행 pitch 라
    #     pitch 합 theta2+theta3+theta4 = 90 이면 approach 가 수직이 된다.
    theta4 = 90.0 - theta2 - theta3

    # 1f) theta5 : roll 정렬. desired 자세=0 이면 우선 -theta1 이지만, wrist_roll
    #     축이 shoulder_pan 과 반대라 다시 부정 -> theta5 = theta1. 이래야 base
    #     회전이 상쇄돼 gripper closing axis 가 월드축(큐브 face 법선)과 평행해진다.
    theta5 = theta1

    joint_config = {
        'shoulder_pan': float(theta1),
        'shoulder_lift': float(theta2),
        'elbow_flex': float(theta3),
        'wrist_flex': float(theta4),
        'wrist_roll': float(theta5),
        'gripper': 0.0,
    }
    return joint_config
