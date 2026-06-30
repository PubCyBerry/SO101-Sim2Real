# so101_kinematics.py
"""SO-101 forward / inverse kinematics — real 폴더 self-contained 버전.

`scripts/ece_4560/mujoco/so101_forward_kinematics.py` 의 FK 수식을 **참고해 포팅**한
것(원본은 `import mujoco` 가 있어 mujoco 미설치 환경에서 import 불가). 여기서는
순수 numpy 만 쓰므로 Windows native uv(teleop/async, mujoco·isaac 없음)에서도
import 된다. mujoco 폴더 원본은 건드리지 않는다.

제공:
- `get_forward_kinematics(dict) -> (position[m], R[3x3])` : mujoco FK 와 동일 수식.
- `mat_to_euler_xyz` / `euler_xyz_to_mat` : R ↔ (roll,pitch,yaw) deg, 규약 `R = Rz@Ry@Rx`.
- `ee_pose_from_joints(dict) -> [x,y,z(m), roll,pitch,yaw(deg)]` : read-out 용 6D euler.
- `solve_ik_dls(...)` : 5-DOF damped least-squares 수치 IK. 임의 orientation target 을
  받되 SO-101 은 팔 5축이라 6-DOF 를 모두 만족 못 한다 → **position 우선·orientation
  best-effort** (가중 + 잔차 보고). AGENTS.md "position-only/best-effort, orientation
  hard constraint 금지" 준수.
"""

from __future__ import annotations

import math

import numpy as np

# joint 순서·limit (degree). 출처: src/so101_contract (feature_codec.SO101_JOINT_ORDER,
# leader_calibration.SO101_FOLLOWER_USD_JOINT_LIMITS). gripper 는 실기기 RANGE_0_100 [0,100].
JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
ARM_JOINTS = JOINT_ORDER[:5]

LIMITS = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-105.0, 105.0),
    "elbow_flex": (-100.0, 100.0),
    "wrist_flex": (-95.0, 105.0),
    "wrist_roll": (-160.0, 160.0),
    "gripper": (0.0, 100.0),
}


# ---------------------------------------------------------------------------
# 기본 회전 (degree 입력)
# ---------------------------------------------------------------------------
def Rx(thetadeg):
    t = np.deg2rad(thetadeg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(thetadeg):
    t = np.deg2rad(thetadeg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(thetadeg):
    t = np.deg2rad(thetadeg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# ---------------------------------------------------------------------------
# Forward kinematics — mujoco FK(so101_forward_kinematics.py) 수식 포팅.
# 각 g = 부모→자식 동차변환 (body.pos 변위 + 고정 자세 + Rz(joint)).
# ---------------------------------------------------------------------------
def _homog(rotation, displacement):
    return np.block([[rotation, np.asarray(displacement, float).reshape(3, 1)], [0, 0, 0, 1]])


def get_gw1(theta1_deg):
    return _homog(Rz(180) @ Rx(180) @ Rz(theta1_deg), (0.0388353, 0.0, 0.0624))


def get_g12(theta2_deg):
    return _homog(Rx(-90) @ Rz(-90) @ Rz(theta2_deg), (-0.0303992, -0.0182778, -0.0542))


def get_g23(theta3_deg):
    return _homog(Rz(90) @ Rz(theta3_deg), (-0.11257, -0.028, 0.0))


def get_g34(theta4_deg):
    return _homog(Rz(-90) @ Rz(theta4_deg), (-0.1349, 0.0052, 0.0))


def get_g45(theta5_deg):
    return _homog(Rz(180) @ Rx(90) @ Rz(-2.78913075) @ Rz(theta5_deg), (0.0, -0.0611, 0.0181))


def get_g5t():
    # tool 프레임 = 두 jaw 사이 파지 중심 (관절 없음, 고정).
    return _homog(Rx(90), (0.0128, -0.0002, -0.090))


def get_forward_kinematics(position_dict):
    """joint degree dict → (tool position[m] (3,), rotation[3x3]) world frame."""
    gwt = (
        get_gw1(position_dict["shoulder_pan"])
        @ get_g12(position_dict["shoulder_lift"])
        @ get_g23(position_dict["elbow_flex"])
        @ get_g34(position_dict["wrist_flex"])
        @ get_g45(position_dict["wrist_roll"])
        @ get_g5t()
    )
    return gwt[0:3, 3], gwt[0:3, 0:3]


# ---------------------------------------------------------------------------
# Euler (XYZ intrinsic, R = Rz(yaw) @ Ry(pitch) @ Rx(roll)) ↔ 회전행렬
# read-out 표시와 IK target 이 같은 규약을 쓰도록 한 쌍으로 정의.
# ---------------------------------------------------------------------------
def euler_xyz_to_mat(roll_deg, pitch_deg, yaw_deg):
    return Rz(yaw_deg) @ Ry(pitch_deg) @ Rx(roll_deg)


def mat_to_euler_xyz(R):
    """R(3x3) → (roll, pitch, yaw) degree. R = Rz(yaw)@Ry(pitch)@Rx(roll) 역산."""
    R = np.asarray(R, float)
    sy = -R[2, 0]
    sy = max(-1.0, min(1.0, sy))
    pitch = math.asin(sy)
    if abs(sy) < 0.99999:  # non-gimbal
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal lock (pitch ≈ ±90°)
        roll = math.atan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def ee_pose_from_joints(position_dict):
    """joint degree dict → [x, y, z (m), roll, pitch, yaw (deg)] (6D euler)."""
    pos, R = get_forward_kinematics(position_dict)
    roll, pitch, yaw = mat_to_euler_xyz(R)
    return np.array([pos[0], pos[1], pos[2], roll, pitch, yaw], dtype=float)


# ---------------------------------------------------------------------------
# 5-DOF damped least-squares IK (position 우선·orientation best-effort)
# ---------------------------------------------------------------------------
def _fk_arm(q5_deg):
    d = {ARM_JOINTS[i]: float(q5_deg[i]) for i in range(5)}
    d["gripper"] = 0.0
    return get_forward_kinematics(d)


def _rot_error_vec(R_target, R_cur):
    """world frame 회전 오차를 axis-angle 벡터(rad)로. R_err = R_target @ R_cur.T."""
    Re = np.asarray(R_target, float) @ np.asarray(R_cur, float).T
    cos = (np.trace(Re) - 1.0) / 2.0
    cos = max(-1.0, min(1.0, cos))
    angle = math.acos(cos)
    if angle < 1e-8:
        return np.zeros(3)
    v = np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
    n = np.linalg.norm(v)
    if n < 1e-9:  # angle ≈ π, 축 불안정 → best-effort 로 0 처리(드묾)
        return np.zeros(3)
    return v / n * angle


def _numeric_jacobian(q5_deg, w_pos, w_rot, eps=0.5):
    pos0, R0 = _fk_arm(q5_deg)
    J = np.zeros((6, 5))
    for i in range(5):
        dq = np.asarray(q5_deg, float).copy()
        dq[i] += eps
        pos1, R1 = _fk_arm(dq)
        J[0:3, i] = w_pos * (pos1 - pos0) / eps
        J[3:6, i] = w_rot * _rot_error_vec(R1, R0) / eps
    return J


def solve_ik_dls(
    target_xyz,
    target_rpy_deg,
    seed_deg,
    gripper=0.0,
    iters=300,
    lam=0.02,
    w_pos=10.0,
    w_rot=1.0,
    pos_tol_m=5e-4,
    rot_tol_deg=1.0,
    step_clamp_deg=15.0,
):
    """임의 6D pose target → joint degree dict (position 우선·orientation best-effort).

    Args:
        target_xyz: [x, y, z] (m, world).
        target_rpy_deg: [roll, pitch, yaw] (deg, R=Rz@Ry@Rx 규약).
        seed_deg: 초기 추정 (current pose dict 또는 6/5-vector). current pose 시드 권장.
        gripper: 출력 dict 에 그대로 넣을 gripper [0,100] (IK 미해결, passthrough).
    Returns:
        (joint_dict, residual_pos_mm, residual_rot_deg, reachable_bool).
        reachable = position 잔차 < 5 mm. orientation 잔차는 5-DOF 라 클 수 있음(정상).
    """
    if isinstance(seed_deg, dict):
        q = np.array([seed_deg[j] for j in ARM_JOINTS], float)
    else:
        q = np.asarray(seed_deg, float)[:5].copy()

    R_t = euler_xyz_to_mat(*target_rpy_deg)
    p_t = np.asarray(target_xyz, float)
    lo = np.array([LIMITS[j][0] for j in ARM_JOINTS])
    hi = np.array([LIMITS[j][1] for j in ARM_JOINTS])

    for _ in range(iters):
        pos, R = _fk_arm(q)
        e_pos = p_t - pos
        e_rot = _rot_error_vec(R_t, R)
        if np.linalg.norm(e_pos) < pos_tol_m and math.degrees(np.linalg.norm(e_rot)) < rot_tol_deg:
            break
        e = np.concatenate([w_pos * e_pos, w_rot * e_rot])
        J = _numeric_jacobian(q, w_pos, w_rot)
        JT = J.T
        try:
            dq = JT @ np.linalg.solve(J @ JT + (lam ** 2) * np.eye(6), e)
        except np.linalg.LinAlgError:
            break
        dq = np.clip(dq, -step_clamp_deg, step_clamp_deg)
        q = np.clip(q + dq, lo, hi)

    pos, R = _fk_arm(q)
    res_pos_mm = float(np.linalg.norm(p_t - pos) * 1000.0)
    res_rot_deg = float(math.degrees(np.linalg.norm(_rot_error_vec(R_t, R))))
    joint = {ARM_JOINTS[i]: float(q[i]) for i in range(5)}
    joint["gripper"] = float(gripper)
    return joint, res_pos_mm, res_rot_deg, (res_pos_mm < 5.0)
