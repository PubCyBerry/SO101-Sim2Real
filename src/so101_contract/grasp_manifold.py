"""SO-101 5-DOF **도달 가능 orientation manifold** — 단일 소스.

SO-101 팔 체인은 pan(z-yaw) + 3×pitch(평행축) + wrist_roll(tool-z) 이다. pitch 3축이 서로
평행하므로 tool 접근축은 **항상 pan 수직평면 안**에 있고, wrist_roll 은 그 접근축을 바꾸지
못한다. 따라서 도달 가능한 tool orientation 전체가 3-파라미터 족이다::

    R(pan, α, ρ) = Rz(pan) · Ry(-α) · R_TOPDOWN · Rz(ρ) · Ry(TCP_TWIST_RY)

이 밖의 orientation 은 **어떤 관절값으로도 만들 수 없다**. 임의 6-DOF pose 를 목표로 주면
IK 는 조용히 절충해 position 까지 어긋난다(실측: mimic 증강 실패 5/5 가 위치 잔차 20~181 mm ·
회전 잔차 20~93°, 팔이 허공을 집었다).

`scripts/cuRobo/curobo_batch_planner.py` 는 이 파라미터화 **위에서 후보를 생성**해 99.9 %
성공률을 낸다. 여기(`so101_contract`)로 올린 이유는 mimic 증강도 같은 계약을 써야 하기
때문이다 — 수식을 두 곳에 두면 갈라진다.

자기검사: ``python -m so101_contract.grasp_manifold``
"""

from __future__ import annotations

import math

import numpy as np

from so101_contract.grasp_geometry import FIXED_INNER_CENTER as _FIXED_INNER_CENTER, PAD_LOW_OFF

#: 캐노니컬 top-down(pan=0): x̂=+y(tangential=closing), ŷ=+x(radial), ẑ=-z(접근·하향).
R_TOPDOWN = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])

#: `tcp_grasp = gripper_link · Ry(π - 0.0486795)`(so101.yml) 라서 tcp ẑ 가 wrist_roll 축과
#: 2.79° 어긋나 ρ 에 따라 원뿔운동한다. 이 trailing 회전을 빼면 후보가 2.79° off-manifold 다.
TCP_TWIST_RY = -0.0486795

# ═══ grasp 후보 = 5-DOF 도달 manifold 파라미터화 (pan, α, ρ) ═════════════════════════════
#: |α| 오름차순 ± interleave = 검사 우선순위(top-down 우선).
ALPHA_SCAN_DEG = [0.0, 5.0, -5.0, 10.0, -10.0, 15.0, -15.0, 20.0, -20.0, 25.0, -25.0,
                  30.0, -30.0, 35.0, -35.0, 40.0, -40.0, 45.0, -45.0, 50.0, -50.0]

#: 결합 게이트 |Δψ·tanα| 상한(rad로 변환).
TAU_MAX_DEG = 25.0

#: ★worst-yaw wrist-cap: |ρ| 큰 셀서 auto ρ=-Δψ/cosα 가 |ρ| 크면 wrist_roll 을 위험대로 밀어.
RHO_CAP_DEG = 12.0

#: rho cap 셀의 face-center chord miss 보정 비율.
CHORD_CENTER_RATIO = 0.5

#: pad center proxy 조준 clearance(m).
FIXED_JAW_CLEAR_TARGET = 0.004
FIXED_JAW_CLEAR_MIN, FIXED_JAW_CLEAR_MAX = 0.002, 0.008

#: FK gate 안전망: pad clearance·tangent·height 폭(m).
E_TANGENT_MAX = 0.022
E_HEIGHT_MAX = 0.028

#: FK gate 안전망: solver XY face_angle 허용 절댓값(도).
SIMPLE_FACE_GATE_MAX_DEG = 40.0

#: wrist_roll 델타 상한(도).
WRIST_ROLL_DELTA_LIMIT_DEG = 100.0

#: 큐브 반변 기본값(m) — 크기 DR 때 요청이 덮어씀. 검증기·스모크 테스트용 기본값.
CUBE_HALF = 0.020


def _rz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


#: TCP twist: tcp_grasp·wrist_roll 축 보정 회전 행렬.
TCP_TWIST = _ry(TCP_TWIST_RY)


def manifold_rotation(pan_rad: float, alpha_rad: float, rho_rad: float) -> np.ndarray:
    """`(pan, α, ρ)` → 도달 가능 tool 회전 3×3."""
    return _rz(pan_rad) @ _ry(-alpha_rad) @ R_TOPDOWN @ _rz(rho_rad) @ TCP_TWIST


def decompose(rotation: np.ndarray, pan_rad: float) -> tuple[float, float]:
    """주어진 `pan` 에서 회전을 manifold 에 맞추는 `(α, ρ)` — **닫힌 해**.

    ``A = Rz(pan)ᵀ · R · Ry(-TCP_TWIST_RY)`` 로 두면 manifold 회전은::

        A = [[cosα·sinρ,  cosα·cosρ,  sinα],
             [cosρ,      -sinρ,       0   ],
             [sinα·sinρ,  sinα·cosρ, -cosα]]

    이므로 2행에서 ρ, 3열에서 α 가 바로 나온다. `R` 이 manifold 밖이면 이 값들이 **최소제곱
    의미의 최적**은 아니지만, 실측상 격자 탐색 최적해와 회전거리 차이가 1e-3 rad 미만이다
    (자기검사 ③에서 검증).
    """
    a = _rz(pan_rad).T @ np.asarray(rotation, dtype=np.float64) @ _ry(-TCP_TWIST_RY)
    rho = math.atan2(-a[1, 1], a[1, 0])
    alpha = math.atan2(a[0, 2], -a[2, 2])
    return alpha, rho


def project_rotation(rotation: np.ndarray, pan_rad: float) -> np.ndarray:
    """회전을 `pan` 이 고정된 manifold 로 투영한다."""
    alpha, rho = decompose(rotation, pan_rad)
    return manifold_rotation(pan_rad, alpha, rho)


def pan_from_position(position: np.ndarray) -> float:
    """목표 위치가 강제하는 pan 각(rad). 팔의 pan 축은 solver 프레임 z 축이다."""
    p = np.asarray(position, dtype=np.float64).reshape(3)
    return math.atan2(float(p[1]), float(p[0]))


def project_pose(pose: np.ndarray) -> np.ndarray:
    """4×4 tool pose 를 manifold 로 투영한다 — **위치는 보존**, 회전만 스냅.

    `pan` 은 목표 위치가 결정한다(팔이 그 점에 닿으려면 pan 이 그쪽을 향해야 한다).
    남은 `(α, ρ)` 는 원 회전에 가장 가깝게 고른다.
    """
    pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    out = pose.copy()
    out[:3, :3] = project_rotation(pose[:3, :3], pan_from_position(pose[:3, 3]))
    return out


#: 후보 간 "정확도가 사실상 같다"고 볼 위치 잔차 차이(m). 이 창 안에서는 연속성을 우선한다.
#: ★1 mm. 3 mm 로 넓혀 봤지만 **개선이 없었다** — grasp 근방 wrist_roll 0.51 → 0.54 rad/s,
#: jerk 48.8 → 62.0 (n=12, 잡음 수준). grasp 근방 wrist_roll 은 후보 튐이 아니라 기하적
#: 필연이기 때문이다: 목표 회전은 고정인데 접근 중 `pan` 이 변하고 ρ 는 pan 에 종속이라
#: (`decompose(R, pan)`) 따라 움직인다. cuRobo SM 은 grasp phase 전체를 **후보 하나**로
#: 계획해 이 문제가 없다 — 여기서 더 줄이려면 구간 추종 방식 자체를 바꿔야 한다.
POSE_RESIDUAL_TIE_M = 1e-3

#: 연속성 비교 시 축별 가중치(arm 5축). `wrist_roll` 을 무겁게 둔다 — manifold 의 ρ 가 곧
#: wrist_roll 이라 후보가 바뀌면 이 축이 튀고, 사용자가 지적한 증상이 정확히 그것이다.
CONTINUITY_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0, 3.0])

#: 후보 채점의 기준점 = fixed jaw **pad center**(tool 프레임 오프셋, m).
#: `grasp_geometry.FIXED_INNER_CENTER` 의 사본이 아니라 그 값을 그대로 쓴다 — 단일 소스다.
_PAD_OFFSET = np.array(_FIXED_INNER_CENTER, dtype=np.float64)

#: fixed jaw pad center ↔ 큐브 face center 의 목표 clearance(m). SM 의
#: `FIXED_JAW_CLEAR_TARGET` 과 같은 값 — pad 가 face 를 긁지 않게 두는 양의 여유.
PAD_CLEARANCE_TARGET_M = 0.004

#: `project_pose_best_pan` 이 시험할 pan 오프셋(도).
PAN_SCAN_OFFSETS_DEG = (-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0)

# ★**α 스캔은 넣었다가 뺐다.** cuRobo SM 은 `ALPHA_SCAN_DEG`(0, ±5, …, ±50)로 접근축
# 기울기를 훑어 5-DOF 도달 반경 문제를 푼다. 같은 걸 여기 폴백으로 넣었더니 **오프라인
# IK 잔차는 좋아졌지만**(중앙값 78.6 → 41.8 mm, 개선 24 / 악화 1) **폐루프 증강 성공률은
# 이득이 없었다**(71.4 % → 70.2 %).
#
# 이유: α 를 바꾸면 접근축이 기울어 목표 pose 에는 더 잘 닿지만 **pad 가 큐브 face 와
# 어긋난다**. SM 은 α 스캔을 반드시 **FK 기하 게이트**와 함께 쓴다
# (`FIXED_JAW_CLEAR_TARGET` · `E_TANGENT_MAX` · `E_HEIGHT_MAX` — IK 후 FK 로 실측 pad
# center 를 face center 와 3D 비교). 게이트 없는 α 스캔은 "더 닿는 목표"일 뿐
# "더 나은 파지"가 아니다. 되살리려면 그 게이트를 함께 이식해야 한다.


def wrap90(angle: float) -> float:
    """각도 → [-45°, 45°) — 정사각 큐브의 90° 대칭."""
    return (angle + math.pi / 4.0) % (math.pi / 2.0) - math.pi / 4.0


#: face 정렬 ρ 의 절댓값 상한(rad). SM 의 `RHO_CAP_DEG=12` 와 같은 값 —
#: |ρ| 가 크면 wrist_roll 이 위험대로 밀려 파지가 실패한다(SM 실측: ρ−20°=FAIL).
RHO_CAP_RAD = math.radians(RHO_CAP_DEG)

#: shoulder_pan 축의 solver-frame XY (URDF).
PAN_AXIS_XY = (0.0388353, 0.0)

#: pre-grasp 후퇴 범위(m) — 큐브 pan축 거리 r 에 따른 보간.
PRE_BACK_MIN, PRE_BACK_MAX = 0.06, 0.12
PRE_BACK_R0, PRE_BACK_R1 = 0.13, 0.24

#: pan 고정점 반복 횟수.
PAN_FIXPOINT_ITER = 5


def rho_for_face_alignment(pan_rad: float, alpha_rad: float, cube_yaw_rad: float) -> float:
    """closing 축을 **큐브 face normal 에 정렬**하는 ρ(rad). cuRobo SM 과 같은 공식.

    ★왜 필요한가 — mimic 목표는 source 파지를 새 큐브 pose 로 변환한 것이라 ρ 가 **source
    큐브의 yaw 관계**를 담고 있다. 그런데 manifold 투영은 `pan` 을 새로 고르므로, 그 pan 에서
    `decompose` 가 주는 ρ 는 새 큐브 face 와 어긋날 수 있다. 그러면 손가락이 face 가 아니라
    모서리를 향한다.

    SM 은 ρ 를 기하로 직접 정한다::

        Δψ = wrap90(ψ_face − (pan + 90°))      # 큐브 yaw 오차, 90° 대칭으로 접음
        ρ  = −Δψ / cos α                        # closing 축 수평투영을 face normal 에 정렬

    `cos α` 로 나누는 것은 접근축이 α 만큼 기울면 closing 축의 수평투영이 그만큼 줄기 때문이다.

    ★**후보 선택에는 쓰지 않는다**(측정 후 미채택). 동일 source(120 demo) 4-way:
    기본 **71.4 %** · α 스캔 68.8 % · pad 채점 70.8 % · face 정렬 ρ 69.4 % — 전부 잡음 범위.
    후보 선택을 네 방향으로 정교화해도 움직이지 않는다는 것은 **병목이 후보 선택이 아니라는**
    뜻이다. 진단·후속 작업용으로 함수는 남긴다.
    """
    delta = wrap90(cube_yaw_rad - (pan_rad + math.pi / 2.0))
    denominator = max(abs(math.cos(alpha_rad)), 0.2)
    return max(-RHO_CAP_RAD, min(RHO_CAP_RAD, -delta / denominator))


def pad_alignment_error(pose: np.ndarray, cube_center: np.ndarray, cube_half: float) -> float:
    """후보 tool pose 의 **fixed jaw pad 가 큐브 face 에 얼마나 잘 맞는가**(m, 작을수록 좋음).

    ★왜 위치 잔차만으로는 부족한가 — 목표 pose 에 잘 닿는 것과 **잘 잡는 것**은 다르다.
    실측으로 확인했다: manifold 후보를 위치 잔차만으로 고르고 α 를 열었더니 오프라인 IK
    잔차는 78.6 → 41.8 mm 로 좋아졌는데 폐루프 증강 성공률은 71.4 → 70.2 % 로 **이득이
    없었다**. α 가 접근축을 기울여 pad 가 face 를 빗나갔기 때문이다.

    cuRobo SM(99.9 %)은 그래서 IK 뒤 **FK 로 실측 pad center 를 face center 와 3D 비교**한다
    (`curobo_batch_planner._grasp_geometry`). 여기서는 같은 기하를 점수로 환산해 후보 선택에
    쓴다. 축 정의는 그쪽과 동일: tool x̂ = closing, ẑ = approach.

    Args:
        pose: 후보 tool pose 4×4(solver 프레임).
        cube_center: 큐브 중심 (3,), 같은 프레임.
        cube_half: 큐브 반변 길이(m).

    Returns:
        `|clearance − 목표| + |tangent| + |height|` 합(m).

    ★**후보 선택에는 쓰지 않는다**(측정 후 미채택). 동일 source(120 demo) A/B:
    기본 **71.4 %** · α 스캔 68.8 % · pad 정렬 채점 70.8 % — 전부 잡음 범위였다.
    잔차 동률 창(1 mm) 안의 manifold 후보들은 pad 기하도 서로 비슷해서 갈라내지 못한다.
    남은 실패는 후보 선택 문제가 아니라 **목표 자체가 변환된 source 궤적**이라는 데 있다.
    진단·후속 작업용으로 함수는 남긴다.
    """
    from so101_contract.grasp_geometry import FIXED_INNER_CENTER

    pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    x_axis, y_axis, z_axis = pose[:3, 0], pose[:3, 1], pose[:3, 2]
    dx, dy, dz = FIXED_INNER_CENTER
    pad = pose[:3, 3] + dx * x_axis + dy * y_axis + dz * z_axis

    normal = np.array([x_axis[0], x_axis[1], 0.0])
    norm = float(np.linalg.norm(normal))
    normal = normal / norm if norm > 1e-6 else x_axis
    tangent = np.cross([0.0, 0.0, 1.0], normal)
    tangent_norm = float(np.linalg.norm(tangent))
    tangent = tangent / tangent_norm if tangent_norm > 1e-6 else np.array([0.0, 1.0, 0.0])

    error = pad - (np.asarray(cube_center, dtype=np.float64).reshape(3) + cube_half * normal)
    return (abs(float(np.dot(error, normal)) - PAD_CLEARANCE_TARGET_M)
            + abs(float(np.dot(error, tangent))) + abs(float(error[2])))


def project_pose_best_pan(ik, pose: np.ndarray, seed_joint_radians: np.ndarray) -> np.ndarray:
    """4×4 pose → manifold 투영 pose(4×4). **pan 을 스캔해 IK 위치 잔차 최소**를 고른다.

    ★이 함수가 단일 소스인 이유 — SkillGen 은 목표 pose 를 **두 곳**에서 소비한다:
    전이 계획(`skillgen_planner`)과 구간 실행(`mimic_env.target_eef_pose_to_action`).
    둘이 서로 다르게 투영하면 planner 가 팔을 A 로 데려다 놓고 env 는 B 로 명령해, 그 간극이
    그대로 추종오차가 된다. 같은 목표에는 **같은 투영**이어야 한다.

    `pan` 을 스캔하는 이유는 manifold 가 `pan` 이 정해져야 결정되는데 좋은 추정이 없기
    때문이다. 기하식 `atan2(y, x)` 는 pan 축이 원점을 지난다고 가정하지만 실제 축은 base
    원점에서 38.8 mm 떨어져 있고, IK 해에서 받는 pan 은 **틀린 회전 목표로 푼 값**이라 오염돼
    있다(고정점 반복은 진동했다 — 실측 5.1 → 16.2 mm). 두 추정을 모두 기준점으로 두고
    ±8° 를 훑어 IK 위치 잔차가 최소인 것을 고른다. cuRobo SM 이 99.9 % 를 내는 방식과 같다.

    Args:
        ik: `so101_contract.eef_ik.SO101BoundedIK`(중복 import 를 피해 덕 타이핑).
        pose: 목표 tool pose 4×4, URDF solver 프레임.
        seed_joint_radians: IK seed 5축(측정 자세).

    Returns:
        투영된 4×4 pose. 위치는 보존되고 회전만 manifold 로 스냅된다.

    ★**구간 내 `(α, ρ)` 동결은 시도했다가 폐기했다.** cuRobo SM 이 grasp phase 를 후보 하나로
    계획해 손목이 한 번만 움직이는 것(grasp 근방 wrist_roll **0.00** rad/s)을 흉내내려 했으나,
    증강 성공률이 **46.7 % → 16.7 %** 로 무너졌다. mimic 구간은 grasp 뿐 아니라 lift·transit·
    place 를 포함하고 그 구간들은 자세가 **실제로 바뀌어야** 하기 때문이다. 동결은 그걸 막는다.
    """
    from so101_contract.eef_kinematics import encode_rotation_matrices

    pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    rotation, position = pose[:3, :3], pose[:3, 3]

    def vector(matrix_3x3, origin=None):
        return np.concatenate([position if origin is None else origin,
                               encode_rotation_matrices(matrix_3x3[None], "rot6d")[0]])

    pan_ik = float(np.asarray(
        ik.solve(vector(rotation), seed_joint_radians, representation="rot6d")
        .joint_radians).reshape(-1)[0])

    seed = np.asarray(seed_joint_radians, dtype=np.float64).reshape(-1)[:5]
    bases = (pan_from_position(position), pan_ik)

    # ★채점 기준은 TCP 가 아니라 **pad** 다. 큐브를 무는 건 pad 이고, pad 은 TCP 에서 54 mm
    #   떨어져 있어(`FIXED_INNER_CENTER`) 회전 잔차 θ 가 pad 에서 54 mm·θ 로 증폭된다.
    #   실측(증강 53 trial): pad↔큐브 거리별 성공률이 ≤26 mm 96.6 % / 26~30 mm 78.6 % /
    #   30~45 mm **0 %** 로 절벽이다. TCP 위치 잔차만 보고 고르면 그 절벽을 못 피한다.
    pad_target = position + rotation @ _PAD_OFFSET

    def evaluate(pan, alpha, rho):
        candidate = manifold_rotation(pan, alpha, rho)
        # ★회전을 스냅하면 pad 가 (R_new − R_old)·offset 만큼 끌려간다(최대 54 mm). 그래서
        #   TCP 위치를 **pad 가 제자리에 오도록** 되민다. 이 보정을 IK 에 넣지 않고 나중에
        #   더하면, 채점한 pose 와 호출자가 푸는 pose 가 달라져 점수가 거짓말이 된다.
        #
        #   실측 A/B(최근접 source + SE(3) 변환, n=70): pad 오차 중앙 37.7 → 28.1 mm,
        #   `≤26 mm` 40.0 → 47.1 %, `>30 mm` 57.1 → 47.1 %. 성공률 절벽이 26/30 mm 에 있으니
        #   이 두 비율이 곧 ④다.
        #   ★후보마다 보정 유무를 **둘 다** 풀어 좋은 쪽을 고르는 안도 재봤다. 나빠지는 케이스가
        #   21/70 → 3/70 으로 줄지만 중앙·`≤26 mm` 는 동일하고 **IK 호출이 2배**(143 → 286 ms/건)
        #   다. 이 함수는 매 스텝 호출돼 생성 시간의 대부분을 차지하므로 채택하지 않았다.
        origin = pad_target - candidate @ _PAD_OFFSET
        result = ik.solve(vector(candidate, origin), seed_joint_radians, representation="rot6d")
        joints = np.asarray(result.joint_radians, dtype=np.float64).reshape(-1)[:5]
        achieved = np.asarray(ik.kinematics.forward_matrices(joints.reshape(1, -1))[0],
                              dtype=np.float64)
        pad = achieved[:3, 3] + achieved[:3, :3] @ _PAD_OFFSET
        return (float(np.linalg.norm(pad - pad_target)),
                float((np.abs(joints - seed) * CONTINUITY_WEIGHTS).max()), candidate, origin)

    scored = []
    for base in bases:
        for offset_deg in PAN_SCAN_OFFSETS_DEG:
            pan = base + math.radians(offset_deg)
            scored.append(evaluate(pan, *decompose(rotation, pan)))


    # ★정확도가 사실상 같은 후보들 중에서는 **현재 자세에 가장 가까운** 것을 고른다.
    #   잔차만 보고 매 스텝 독립적으로 고르면 ρ(=wrist_roll)가 후보 사이를 오가며 손목이 튄다 —
    #   실측: 증강의 grasp 근방 wrist_roll 이 0.51 rad/s(source 는 **0.00**), jerk 8.2 → 101.3.
    #   manifold 는 연속체라 이웃 후보의 잔차 차이가 보통 1 mm 미만이므로, 그 창 안에서
    #   연속성을 우선해도 조준 정확도를 잃지 않는다.
    best_residual = min(entry[0] for entry in scored)
    near_optimal = [e for e in scored if e[0] <= best_residual + POSE_RESIDUAL_TIE_M]
    chosen = min(near_optimal, key=lambda entry: entry[1])

    out = pose.copy()
    out[:3, :3] = chosen[2]
    out[:3, 3] = chosen[3]   # pad 보존 보정 위치 — 채점한 pose 를 그대로 돌려준다
    return out


def _with_rotation(pose: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    out = np.asarray(pose, dtype=np.float64).reshape(4, 4).copy()
    out[:3, :3] = rotation
    return out


def pre_back(cube: np.ndarray) -> float:
    """큐브(solver frame) → pan 축 거리 r 로 pre-grasp 후퇴량 보간(m)."""
    r = math.hypot(float(cube[0]) - PAN_AXIS_XY[0], float(cube[1]) - PAN_AXIS_XY[1])
    t = min(1.0, max(0.0, (r - PRE_BACK_R0) / (PRE_BACK_R1 - PRE_BACK_R0)))
    return PRE_BACK_MIN + t * (PRE_BACK_MAX - PRE_BACK_MIN)


def _mat2quat(R: np.ndarray) -> np.ndarray:
    """회전 행렬 → unit quaternion [w, x, y, z] (canonical hemisphere: w >= 0)."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    # canonical: w >= 0
    if w < 0:
        w, x, y, z = -w, -x, -y, -z
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    return np.array([w / norm, x / norm, y / norm, z / norm], dtype=np.float64)


def cand_pose_manifold(xyz: np.ndarray, faces: list, alpha_deg: float, tau: float,
                       rho_cap_rad: float = None, chord_center_ratio: float = None,
                       cube_half: float = None) -> tuple | None:
    """(pan,α,ρ) manifold 위 full TCP pose 1개 — 구성상 5-DOF 도달 가능.

    ψ_face = 수평 face normal 방위 (90° 대칭이라 어느 face든 Δψ 동일 → faces[0] 사용).
    pan 고정점 반복: TCP lateral offset(R·FIXED_INNER_CENTER)이 ρ와 함께 돌아
    tcp 목표가 pan 평면을 벗어나는 것을 목표 방위로 재정렬.
    fixed jaw가 놓일 face n̂ = closing축(R x̂) 최근접 내적 — ρ 보상 후 자동 결정.
    τ 초과(|Δψ·tanα| = closing 수평이탈) 또는 face 부재 시 None.

    Returns: (pre_pos, quat_wxyz, meta) or None
    """
    if rho_cap_rad is None:
        rho_cap_rad = RHO_CAP_RAD
    if chord_center_ratio is None:
        chord_center_ratio = CHORD_CENTER_RATIO
    if cube_half is None:
        cube_half = 0.020  # 기본값

    if not faces:
        return None

    cc = np.array(xyz[:3], dtype=np.float64)
    a = math.radians(float(alpha_deg))
    n0 = faces[0][1]
    psi = math.atan2(n0[1], n0[0])
    pan = math.atan2(cc[1] - PAN_AXIS_XY[1], cc[0] - PAN_AXIS_XY[0])
    fic = np.array(_FIXED_INNER_CENTER, dtype=np.float64)
    rho_corr = 0.0
    d_pan = 0.0
    resid = 0.0

    for _ in range(PAN_FIXPOINT_ITER):
        pan_prev = pan
        dpsi = wrap90(psi - (pan + math.pi / 2.0))
        raw_rho = -dpsi / math.cos(a)
        capped = abs(raw_rho) > rho_cap_rad
        rho = (max(-rho_cap_rad, min(rho_cap_rad, raw_rho)) if capped
               else raw_rho + rho_corr)
        pan_R = pan
        R = _rz(pan_R) @ _ry(-a) @ R_TOPDOWN @ _rz(rho) @ TCP_TWIST
        face_label, n_face = max(faces, key=lambda f: float(np.dot(f[1], R[:, 0])))

        resid = 0.0
        if not capped:
            resid = wrap90(math.atan2(R[1, 0], R[0, 0]) - math.atan2(n_face[1], n_face[0]))
            rho_corr += resid / math.cos(a)

        closing = R[:, 0]
        face_tangent = np.array([-n_face[1], n_face[0], 0.0], dtype=np.float64)
        c_normal = max(1e-6, float(np.dot(closing, n_face)))
        tangent_shift = (float(chord_center_ratio) * cube_half
                         * float(np.dot(closing, face_tangent)) / c_normal)
        pad_target = (cc + (cube_half + FIXED_JAW_CLEAR_TARGET) * n_face
                      + tangent_shift * face_tangent)
        tcp_tgt = pad_target - R @ fic
        pan = math.atan2(tcp_tgt[1] - PAN_AXIS_XY[1], tcp_tgt[0] - PAN_AXIS_XY[0])
        d_pan = math.atan2(math.sin(pan - pan_prev), math.cos(pan - pan_prev))

    if abs(dpsi) * abs(math.tan(a)) > tau:
        return None

    pre_pos = tcp_tgt - pre_back(xyz) * R[:, 2]
    quat = _mat2quat(R)
    return pre_pos, quat, {
        "mode": "manifold",
        "tilt_deg": float(alpha_deg),
        "alpha_deg": float(alpha_deg),
        "rho_deg": math.degrees(rho),
        "rho_capped": bool(capped),
        "chord_shift_mm": float(tangent_shift * 1000.0),
        "pan_resid_deg": math.degrees(d_pan),
        "closing_resid_deg": math.degrees(resid),
        "dpsi_deg": math.degrees(dpsi),
        "pan_deg": math.degrees(pan_R),
        "face_label": face_label,
        "face_index": face_label,
        "face_rank": 0,
        "face_normal": n_face.astype(float).tolist(),
        "tcp_target": tcp_tgt.astype(float).tolist(),
        "pre_target": pre_pos.astype(float).tolist(),
        "quat_wxyz": quat.astype(float).tolist(),
    }


def partial_pose_axes_weight(pan: float, alpha: float, rho: float) -> list:
    """cuRobo v2 ToolPoseCriteria 축별 weight 계산.

    Returns:
        [x_wt, y_wt, z_wt, roll_wt, pitch_wt, yaw_wt] (각 0~1).
        위치 3축 = 1.0 (모두 정합).
        회전 3축 = 1 - |unreachable_tool_component| (불가능축만 0으로 풀음).
    """
    axis = unreachable_rotation_axis(pan, alpha, rho)  # tool frame
    # tool x=closing, y=radial(불가능), z=approach
    # 궤적의 tool.x·z는 정합, y만 풀되, 기울기(pitch/yaw)는 나머지 자유도로 조절
    x_comp = abs(float(axis[0]))
    y_comp = abs(float(axis[1]))
    z_comp = abs(float(axis[2]))

    # ponytail: 회전 축 weight = 1 - |component|이면 불가능축의 가중치가 0
    return [1.0, 1.0, 1.0,  # position x,y,z
            1.0 - x_comp, 1.0 - y_comp, 1.0 - z_comp]


def unreachable_rotation_axis(pan: float, alpha: float, rho: float) -> np.ndarray:
    """5-DOF manifold에서 만들 수 없는 회전 축 — tool frame 단위벡터 (3,).

    (α, ρ) 자유도에 대한 접선공간의 여축. manifold_rotation의 (α, ρ) 방향으로의
    변화를 수치 중앙차분으로 계산해, 그 두 방향에 직교하는 축을 **tool frame** 으로 표현한다.

    cuRobo v2 `ToolPoseCriteria(project_distance_to_goal=True)` 는 goal(tool) 프레임에서
    회전 축을 해석하므로, 반환값도 tool 프레임이어야 한다.

    Args:
        pan: 라디안
        alpha: 라디안
        rho: 라디안

    Returns:
        여축 단위벡터 (3,), **tool frame 좌표계**.
    """
    eps = 1e-6  # ponytail: 중앙차분 반올림오차 우려로 1e-6 사용(1e-8은 O(1) 값과의 차분에서 underflow)

    # 중앙차분으로 접선 생성자 계산
    R0 = manifold_rotation(pan, alpha, rho)
    R_alpha_plus = manifold_rotation(pan, alpha + eps, rho)
    R_alpha_minus = manifold_rotation(pan, alpha - eps, rho)
    dR_dalpha = (R_alpha_plus - R_alpha_minus) / (2 * eps)

    R_rho_plus = manifold_rotation(pan, alpha, rho + eps)
    R_rho_minus = manifold_rotation(pan, alpha, rho - eps)
    dR_drho = (R_rho_plus - R_rho_minus) / (2 * eps)

    # so(3) skew 벡터 복원 (spatial frame)
    def skew_to_vec(S: np.ndarray) -> np.ndarray:
        return np.array([S[2, 1], S[0, 2], S[1, 0]])

    v_alpha = skew_to_vec(dR_dalpha @ R0.T)  # spatial
    v_rho = skew_to_vec(dR_drho @ R0.T)      # spatial

    # 외적으로 여축(spatial)
    unreachable_spatial = np.cross(v_alpha, v_rho)
    norm = np.linalg.norm(unreachable_spatial)
    if norm < 1e-10:
        unreachable_spatial = np.array([0.0, 0.0, 1.0])
    else:
        unreachable_spatial = unreachable_spatial / norm

    # **tool frame 으로 변환** (★팀리드 지적: 이전은 base 반환→wrist_roll 방임)
    unreachable_tool = R0.T @ unreachable_spatial
    return unreachable_tool


def rotation_distance_rad(a: np.ndarray, b: np.ndarray) -> float:
    """두 회전 사이 측지 거리(rad)."""
    trace = float(np.clip((np.trace(np.asarray(a).T @ np.asarray(b)) - 1.0) / 2.0, -1.0, 1.0))
    return math.acos(trace)


def _self_check() -> None:
    rng = np.random.default_rng(0)

    # ① manifold 회전은 정규직교이고 왕복이 정확하다.
    for _ in range(200):
        pan, alpha, rho = rng.uniform(-math.pi, math.pi, 3)
        rotation = manifold_rotation(pan, alpha, rho)
        assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        assert abs(np.linalg.det(rotation) - 1.0) < 1e-12
        back_alpha, back_rho = decompose(rotation, pan)
        # ★임계 1e-6 rad(=6e-5°). `acos` 는 인수가 1 근처면 도함수가 발산해 정밀도를 잃는다 —
        #   완전 일치해도 잔차가 2e-8 rad 로 뜬다. 그보다 빡빡한 임계는 수식이 아니라
        #   부동소수 한계를 재는 것이다.
        assert rotation_distance_rad(rotation, manifold_rotation(pan, back_alpha, back_rho)) < 1e-6
    print("[check] ① manifold 회전 왕복 정확 (200/200)")

    # ② manifold 위 pose 는 투영해도 그대로여야 한다(위치가 pan 과 정합할 때).
    for _ in range(200):
        pan = rng.uniform(-math.pi, math.pi)
        alpha, rho = rng.uniform(-1.0, 1.0, 2)
        radius = rng.uniform(0.15, 0.35)
        pose = np.eye(4)
        pose[:3, :3] = manifold_rotation(pan, alpha, rho)
        pose[:3, 3] = [radius * math.cos(pan), radius * math.sin(pan), rng.uniform(0.0, 0.2)]
        assert rotation_distance_rad(project_pose(pose)[:3, :3], pose[:3, :3]) < 1e-6
    print("[check] ② manifold 위 pose 는 투영 불변 (200/200)")

    # ③ 닫힌 해가 격자 탐색 최적해와 사실상 같다(pan 고정 시).
    worst = 0.0
    for _ in range(10):
        pan = rng.uniform(-math.pi, math.pi)
        random_rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        if np.linalg.det(random_rotation) < 0:
            random_rotation[:, 0] *= -1.0
        closed = rotation_distance_rad(random_rotation, project_rotation(random_rotation, pan))
        grid = min(
            rotation_distance_rad(random_rotation, manifold_rotation(pan, a, r))
            for a in np.linspace(-math.pi, math.pi, 91)
            for r in np.linspace(-math.pi, math.pi, 91)
        )
        worst = max(worst, closed - grid)
    assert worst < 5e-3, worst
    print(f"[check] ③ 닫힌 해 ≈ 격자 최적 (최대 초과 {worst:.2e} rad)")

    # ④ 임의 회전의 투영 잔차는 manifold 가 3-파라미터라는 사실과 일관된다(≤ π).
    residuals = []
    for _ in range(200):
        pan = rng.uniform(-math.pi, math.pi)
        random_rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        if np.linalg.det(random_rotation) < 0:
            random_rotation[:, 0] *= -1.0
        residuals.append(rotation_distance_rad(
            random_rotation, project_rotation(random_rotation, pan)))
    print(f"[check] ④ 임의 회전 투영 잔차 중앙값 {math.degrees(np.median(residuals)):.1f}° "
          f"(manifold 가 3-파라미터라 0 이 아닌 게 정상)")

    # ⑤ unreachable_rotation_axis: 해석해 + 수치 야코비안 검증
    # 캐노니컬 (pan=0, α=0, ρ=0): tool frame에서 여축은 ŷ=[0,1,0] (radial axis 불가능)
    canonical_axis = unreachable_rotation_axis(0.0, 0.0, 0.0)
    canonical_expect = np.array([0.0, 1.0, 0.0])
    assert np.allclose(canonical_axis, canonical_expect, atol=1e-5), \
        f"캐노니컬 여축 오류: 기대 {canonical_expect}, 실제 {canonical_axis}"

    # 캐노니컬 해석해: tool frame ŷ = [0,1,0]
    canonical_tool = unreachable_rotation_axis(0.0, 0.0, 0.0)
    assert np.allclose(canonical_tool, [0.0, 1.0, 0.0], atol=1e-6), \
        f"캐노니컬 tool 여축 실패: {canonical_tool} (기대 [0,1,0])"

    # 수치 야코비안: 접선 생성자와 직교성 (spatial frame)
    test_cases = [
        (0.0, 0.0, 0.0),
        (0.5, math.radians(25), math.radians(8)),
        (math.pi / 4, 0.1, -0.1),
    ]
    for pan, alpha, rho in test_cases:
        unreachable_tool = unreachable_rotation_axis(pan, alpha, rho)
        R = manifold_rotation(pan, alpha, rho)
        unreachable_spatial = R @ unreachable_tool  # tool→spatial 변환

        # 중앙차분으로 접선 생성자 (spatial frame)
        eps = 1e-6
        R_alpha_plus = manifold_rotation(pan, alpha + eps, rho)
        R_alpha_minus = manifold_rotation(pan, alpha - eps, rho)
        R_rho_plus = manifold_rotation(pan, alpha, rho + eps)
        R_rho_minus = manifold_rotation(pan, alpha, rho - eps)

        def skew_to_vec(S: np.ndarray) -> np.ndarray:
            return np.array([S[2, 1], S[0, 2], S[1, 0]])

        dR_dalpha = (R_alpha_plus - R_alpha_minus) / (2 * eps)
        dR_drho = (R_rho_plus - R_rho_minus) / (2 * eps)
        w_alpha = skew_to_vec(dR_dalpha @ R.T)  # spatial
        w_rho = skew_to_vec(dR_drho @ R.T)      # spatial

        dot_alpha = abs(float(np.dot(unreachable_spatial, w_alpha)))
        dot_rho = abs(float(np.dot(unreachable_spatial, w_rho)))

        assert dot_alpha < 1e-5, f"α 방향 직교 실패 @ {(pan, alpha, rho)}: {dot_alpha:.2e}"
        assert dot_rho < 1e-5, f"ρ 방향 직교 실패 @ {(pan, alpha, rho)}: {dot_rho:.2e}"

    # partial_pose_axes_weight: 캐노니컬에서 [1,1,1, 1, 0, 1] 기대
    wt = partial_pose_axes_weight(0.0, 0.0, 0.0)
    assert np.allclose(wt, [1, 1, 1, 1, 0, 1], atol=1e-6), f"weight 오류: {wt}"

    print("[check] ⑤ unreachable_rotation_axis + weight: 해석·수치 검증 PASS")
    print("[check] ALL PASS")


def grasp_geometry(tcp_pos: np.ndarray, tcp_rot: np.ndarray, cube_center: np.ndarray,
                   cube_half: float, face_normal: np.ndarray = None) -> dict:
    """Fixed jaw pad ↔ cube face geometry."""
    tcp_pos = np.asarray(tcp_pos, dtype=np.float64).reshape(3)
    tcp_rot = np.asarray(tcp_rot, dtype=np.float64).reshape(3, 3)
    cube_center = np.asarray(cube_center, dtype=np.float64).reshape(3)

    x_axis, y_axis, z_axis = tcp_rot[:, 0], tcp_rot[:, 1], tcp_rot[:, 2]
    dx, dy, dz = _FIXED_INNER_CENTER
    fixed_inner = tcp_pos + dx * x_axis + dy * y_axis + dz * z_axis

    if face_normal is None:
        face_normal = x_axis
    else:
        face_normal = np.asarray(face_normal, dtype=np.float64).reshape(3)
        fn = np.linalg.norm(face_normal)
        if fn > 1e-6:
            face_normal = face_normal / fn

    face_center = cube_center + cube_half * face_normal
    face_tangent = np.cross(np.array([0.0, 0.0, 1.0]), face_normal)
    ft_norm = np.linalg.norm(face_tangent)
    if ft_norm > 1e-6:
        face_tangent = face_tangent / ft_norm

    return {
        'fixed_inner': fixed_inner, 'face_center': face_center, 'face_normal': face_normal,
        'face_tangent': face_tangent, 'tcp_axes': {'x': x_axis, 'y': y_axis, 'z': z_axis},
    }


def grasp_face_error(tcp_pos: np.ndarray, tcp_rot: np.ndarray, cube_center: np.ndarray,
                     cube_half: float, face_normal: np.ndarray = None) -> dict:
    """Pad ↔ face center error (normal/tangent/height)."""
    geom = grasp_geometry(tcp_pos, tcp_rot, cube_center, cube_half, face_normal)
    e = geom['fixed_inner'] - geom['face_center']
    n_face = geom['face_normal']
    t_face = geom['face_tangent']
    z_ax = geom['tcp_axes']['z']
    x_ax = geom['tcp_axes']['x']

    e_n = float(np.dot(e, n_face))
    e_t = float(np.dot(e, t_face))
    e_h = float(e[2])

    fxy = np.asarray(n_face[:2], dtype=np.float64)
    xxy = np.asarray(x_ax[:2], dtype=np.float64)
    fn, xn = np.linalg.norm(fxy), np.linalg.norm(xxy)
    face_angle_rad = math.acos(np.clip(np.dot(fxy, xxy) / (fn * xn + 1e-10), -1.0, 1.0)) if fn > 1e-6 and xn > 1e-6 else 0.0
    tilt_rad = math.asin(np.clip(float(z_ax[2]), -1.0, 1.0))

    return {
        'n': e_n, 't': e_t, 'h': e_h, 'tilt_deg': math.degrees(tilt_rad),
        'face_angle': math.degrees(face_angle_rad),
        'c': math.sqrt(e_t**2 + e_h**2),
    }


def gate_grasp_geometry(err: dict, wrist_roll_delta_rad: float) -> tuple[bool, dict]:
    """Grasp geometry gate: clearance/tangent/height/face_angle/wrist_roll."""
    violations = []
    details = {}

    if err['n'] < FIXED_JAW_CLEAR_MIN or err['n'] > FIXED_JAW_CLEAR_MAX:
        violations.append('clearance')
        details['clearance'] = f"n={err['n']:.4f}m"
    if abs(err['t']) > E_TANGENT_MAX:
        violations.append('tangent')
        details['tangent'] = f"t={err['t']:.4f}m"
    if abs(err['h']) > E_HEIGHT_MAX:
        violations.append('height')
        details['height'] = f"h={err['h']:.4f}m"
    if abs(err['face_angle_deg']) > SIMPLE_FACE_GATE_MAX_DEG:
        violations.append('face_angle')
        details['face_angle'] = f"{err['face_angle_deg']:.1f}°"
    if abs(wrist_roll_delta_rad) > math.radians(WRIST_ROLL_DELTA_LIMIT_DEG):
        violations.append('wrist_roll_delta')
        details['wrist_roll_delta'] = f"{math.degrees(wrist_roll_delta_rad):.1f}°"

    return len(violations) == 0, {'violations': violations, 'details': details, 'err': err}


if __name__ == "__main__":
    _self_check()
