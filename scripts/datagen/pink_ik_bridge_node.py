#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SO-101 pick-place — pink IK state machine (2026-07-03 motion-planning 설계 보고서 구현).

핵심 진단(§2): IK 실패 대부분은 solver 가 아니라 target pose 가 kinematically inconsistent
해서다. 5-DOF SO-101 의 접근축은 항상 shoulder_pan 이 정의하는 수직 평면 안에 있어야 하므로,
**처음부터 풀리는 pose 만 생성**한다 — (pan, α, ρ) 3 스칼라 → 6D pose 조립(grasp_R).

- TCP(§3): IK 타겟 = gripper_frame_link 이 아니라 닫힘 지점 정적 OP_FRAME `tcp_grasp`
  (검증된 grasp 오프셋 역산, EE-local 기본 (-0.003,-0.019,-0.042)). 큐브 중심을 직접 조준.
- grasp 선택(§4·§6): pan=atan2(cube), α=±tilt 스캔(0, +5, -5, … — +α=wrist 를 base 쪽
  (원거리 2R 해소), -α=반대(근거리 최소반경 해소)), 후보를 grasp+hover+pre corridor err
  score 로 평가해 최적 채택(top-down 선호 tie-break). ρ=wrist_roll yaw 보상 = -Δψ/cosα,
  제약 |Δψ·tanα| ≤ τ_max. grasp z 하한 = 책상 + jaw_floor_clear + jaw_tip_drop(tip 바닥금지).
- 접근(§5·§7): pan-first 정렬(side-swipe 제거, pan=IK 해의 shoulder_pan — URDF pan 부호가
  azimuth 와 반대) → pre-grasp(접근축 -z 후퇴) → tool축 Cartesian 직선 강하(작은 step,
  직전 해 seed) → close → world 수직 lift(H_SAFE 도달 전 횡이동 금지) → 등고 transit
  (그릇 중심서 base 쪽 bowl_pull 당김) → release(leg 0.5×) → home(REST 자세).
- pan 고정점 보정: TCP lateral offset(R·r_tcp)이 ρ 와 함께 돌아 EE 가 pan 평면을 벗어나면
  QP 가 위치를 타협 → 양단(|pan| 큰 곳) systematic miss(윗면 찌름). EE 방위로 pan 재정렬.
- 그릇 회피(§8): 절두 원뿔 keep-out cone_radius(z)(▽, 위로 벌어짐 — 클리어런스가 테이블
  근처 최대·rim 에서 최소) + 스윕 세그먼트 체크. H_SAFE ≥ rim+매달림+margin 이라 transit 은
  구성상 clear.

프레임 보정(기존 유지): URDF base ↔ sim base 가 z축 base_yaw(기본 90°) 어긋남 → 모든 EE
목표를 Rz(base_yaw) 회전 후 IK. base 회전은 joint 불변이라 결과 joint 그대로 publish.

--sweep / --gen-traj (DR 범위 물리검증용)는 기존 고정 GRASP_ORIENT 경로 그대로 유지
(bridge --grasp_sweep 계약 불변).

자가검증:  python pink_ik_bridge_node.py --self-check   (ROS·sim 불요, pink 컨테이너)
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

_REPO_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if os.path.isdir(_REPO_SRC):
    sys.path.insert(0, os.path.abspath(_REPO_SRC))
try:
    from so101_contract.feature_codec import SO101_JOINT_ORDER
except Exception:
    SO101_JOINT_ORDER = (
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll", "gripper",
    )

import pinocchio as pin
import pink
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask
import qpsolvers

_HERE = os.path.dirname(__file__)
DEFAULT_URDF = os.path.abspath(
    os.path.join(_HERE, "..", "..", "assets", "robots", "urdf", "so_arm101.urdf")
)
EE_FRAME = "gripper_frame_link"
TCP_FRAME = "tcp_grasp"
BASE_FRAME = "base_link"
CUBE_FRAME = "Cube1"
BOWL_FRAME = "Bowl"

# TCP(§3.3): 닫힘 상태에서 큐브 중심이 놓이는 점, EE(gripper_frame_link)-local 정적 오프셋.
# 검증된 grasp 값(grasp_dx/dy/z=(0,-0.016,-0.043) + GRASP_ORIENT) 역산 — jaw 꺾임·roll 흡수.
DEFAULT_TCP_OFFSET = (-0.003, -0.019, -0.042)

# bridge(run_cube_desk_ros_bridge)가 publish 하는 카메라 RGB 토픽 (LeRobot v3 obs). key=CAMERA_KEYS.
CAMERA_TOPICS = {
    "top": "/camera/top/image_raw",
    "wrist": "/camera/wrist/image_raw",
    "front": "/camera/front/image_raw",
}

# sim base_link 좌표 fallback(tf 없을 때). cube=책상끝29.5/왼쪽42, bowl 은 이전 tf.
DEFAULT_CUBE_XYZ = (0.015, -0.255, 0.05)
DEFAULT_BOWL_XYZ = (0.22, -0.265, 0.025)
ROBOT_WORLD = np.array([0.0, 0.0, 0.6749])  # world 로봇 base(user 제공)

# grasp EE 방향(URDF frame) — 실 작업공간 성공 trajectory(pick_place_demo.json) FK 에서 추출.
# ※ --sweep/--gen-traj 전용(계약 불변). SM 본체는 grasp_R(pan,α,ρ) 파라미터화 사용.
GRASP_ORIENT = np.array([
    [0.1520, 0.9852, -0.0795],
    [0.9882, -0.1528, -0.0042],
    [-0.0163, -0.0780, -0.9968],
])

# 캐노니컬 top-down 회전(solver 프레임, pan=0): x_ee=+y(tangential=닫힘축), y_ee=+x(radial),
# z_ee=-z(접근축, 아래). GRASP_ORIENT 를 (pan,α,ρ) 파라미터화로 일반화한 기준점.
R_TOPDOWN = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


def base_to_world(p):
    """base_link → world: 회전 없이 robot 위치만 더함(user 지시). world = base + (0,0,0.6749)."""
    return np.asarray(p, dtype=float) + ROBOT_WORLD


def rotz(p, deg):
    """z축 deg 회전. URDF base ↔ sim base 90° 어긋남 보정."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    p = np.asarray(p, dtype=float)
    return np.array([c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def wrap90(a):
    """각도 → [-45°, 45°) wrap. 정사각 큐브 90° 대칭(§4.2)."""
    return (a + math.pi / 4.0) % (math.pi / 2.0) - math.pi / 4.0


def _wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def grasp_R(pan, alpha, rho):
    """(pan, α, ρ) → TCP 회전(solver 프레임). 접근축이 pan 수직평면 안 → 5-DOF 도달 가능(§4.1).

    - α: 접근축 tilt(0=top-down). Ry(-α) → z_ee 가 radial 바깥쪽으로 기울며 wrist 가 base 쪽
      으로 당겨짐(§6 원거리 도달). 닫힘축(x_ee)은 α 와 무관하게 수평 유지.
    - ρ: wrist_roll(접근축 기준 roll). z_ee 가 아래를 향하므로 닫힘축 수평 yaw 는
      (pan+90°) - ρ·cosα 로 회전 → 보상은 ρ = -Δψ/cosα (§4.2, 부호 주의)."""
    return _rz(pan) @ _ry(-alpha) @ R_TOPDOWN @ _rz(rho)


def axis_angle_deg(R_a, R_t):
    """두 회전의 접근축(z 컬럼) 사이 각[deg] — orientation 도달 판정."""
    d = float(np.dot(np.asarray(R_a)[:, 2], np.asarray(R_t)[:, 2]))
    return math.degrees(math.acos(max(-1.0, min(1.0, d))))


def _load_recorder():
    """sim_to_real 패키지 __init__(isaaclab 의존)을 우회해 data.lerobot_recorder/units 만 로드.

    pink 컨테이너엔 isaaclab 이 없어 `import sim_to_real` 이 실패한다(→ tasks → isaaclab_tasks).
    lerobot_recorder/units 자체는 numpy·so101_contract·pyarrow/imageio 만 쓰므로, 패키지 stub 을
    sys.modules 에 선등록해 __init__ 실행 없이 submodule 만 임포트한다(cube_specs author 와 동일 패턴)."""
    import importlib
    import types
    src = os.path.abspath(_REPO_SRC)
    for pkg in ("sim_to_real", "sim_to_real.data"):
        if pkg not in sys.modules:
            stub = types.ModuleType(pkg)
            stub.__path__ = [os.path.join(src, *pkg.split("."))]
            sys.modules[pkg] = stub
    lr = importlib.import_module("sim_to_real.data.lerobot_recorder")
    lu = importlib.import_module("sim_to_real.data.lerobot_units")
    return lr.LeRobotV3DatasetWriter, lu.to_lerobot_units


def feat_to_rad(feat):
    """gripper feature[0,100] → sim rad. (feature_codec: deg = -10 + feat/100*110)."""
    return math.radians(-10.0 + feat / 100.0 * 110.0)


def arm_deg(q, qidx):
    return [round(math.degrees(q[qidx[n]]), 1) for n in SO101_JOINT_ORDER[:5]]


def gripper_feat(q, qidx):
    return round((math.degrees(q[qidx["gripper"]]) + 10.0) / 110.0 * 100.0)


def smoothstep(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def _solver():
    for s in ("quadprog", "daqp", "osqp", "scs"):
        if s in qpsolvers.available_solvers:
            return s
    if not qpsolvers.available_solvers:
        raise RuntimeError("qpsolvers backend 없음 — `pip install quadprog`")
    return qpsolvers.available_solvers[0]


class PinkIK:
    def __init__(self, urdf_path, wrist_roll_deg, tcp_offset=DEFAULT_TCP_OFFSET):
        self.model = pin.buildModelFromUrdf(urdf_path)
        if not self.model.existFrame(EE_FRAME):
            raise RuntimeError(f"URDF 에 EE 프레임 '{EE_FRAME}' 없음")
        # TCP 정적 프레임(§3): parent joint 고정 OP_FRAME(pinocchio #1927 — live jaw 중앙 불가).
        fid = self.model.getFrameId(EE_FRAME)
        fr = self.model.frames[fid]
        try:
            pj = fr.parentJoint          # pinocchio 3.x
        except AttributeError:
            pj = fr.parent               # pinocchio 2.x
        placement = fr.placement * pin.SE3(np.eye(3), np.asarray(tcp_offset, dtype=float))
        self.model.addFrame(pin.Frame(TCP_FRAME, int(pj), int(fid), placement,
                                      pin.FrameType.OP_FRAME))
        self.data = self.model.createData()
        self.qidx = {n: self.model.joints[self.model.getJointId(n)].idx_q
                     for n in SO101_JOINT_ORDER}
        self.lo = self.model.lowerPositionLimit.copy()
        self.hi = self.model.upperPositionLimit.copy()
        self.solver = _solver()
        self.q_post = pin.neutral(self.model)
        self.q_post[self.qidx["wrist_roll"]] = math.radians(wrist_roll_deg)
        self.q_post = self.clamp(self.q_post)
        self.tcp_offset = np.asarray(tcp_offset, dtype=float)
        # pan joint ↔ TCP azimuth 이득 1회 FK 캘리브레이션 — URDF pan 부호가 solver azimuth 와
        # 반대(joint -61.8° ↔ azimuth +54.8°). pan_align·IK seed 를 큐브 방향으로 돌리는 근거.
        i_pan = self.qidx["shoulder_pan"]
        t0 = self.fk(self.q_post, TCP_FRAME).translation
        qp = self.q_post.copy(); qp[i_pan] += 0.3
        t1 = self.fk(qp, TCP_FRAME).translation
        self.pan_az0 = math.atan2(t0[1], t0[0])
        gain = _wrap_pi(math.atan2(t1[1], t1[0]) - self.pan_az0) / 0.3
        self.pan_gain = gain if abs(gain) > 0.1 else 1.0

    def pan_seed(self, azimuth):
        """TCP azimuth 목표 → shoulder_pan 만 돌린 posture seed(방향 보장·극단 pan 수렴)."""
        q = self.q_post.copy()
        q[self.qidx["shoulder_pan"]] = _wrap_pi(azimuth - self.pan_az0) / self.pan_gain
        return self.clamp(q)

    def clamp(self, q):
        return np.clip(q, self.lo + 1e-4, self.hi - 1e-4)

    def fk(self, q, frame=EE_FRAME):
        return pink.Configuration(self.model, self.data, q).get_transform_frame_to_world(frame)

    def solve(self, target_xyz, orient_R=None, ori_cost=0.0, q_seed=None, iters=4000,
              frame=EE_FRAME):
        """target_xyz(solver 프레임, 회전 적용됨) → (q, err[m]). orient_R 주면 그 방향도 타겟
        (ori_cost, 5-DOF best-effort). 없으면 position-only."""
        target = np.asarray(target_xyz, dtype=float)
        oc = float(ori_cost) if orient_R is not None else 0.0
        ee_task = FrameTask(frame, position_cost=1.0, orientation_cost=oc, lm_damping=1.0)
        post = PostureTask(cost=1e-2); post.set_target(self.q_post)
        q = self.clamp(self.q_post.copy() if q_seed is None else q_seed.copy())
        err = 9.0
        for _ in range(iters):
            cfg = pink.Configuration(self.model, self.data, q)
            Tc = cfg.get_transform_frame_to_world(frame)
            err = float(np.linalg.norm(target - Tc.translation))
            R = orient_R if orient_R is not None else Tc.rotation
            ee_task.set_target(pin.SE3(R, target))
            try:
                v = solve_ik(cfg, [ee_task, post], 1 / 30, solver=self.solver, damping=1e-6, safety_break=False)
            except Exception:
                v = np.zeros(self.model.nv)
            q = pin.integrate(self.model, q, v / 30)
            if err < 3e-4:
                if orient_R is None:
                    break
                # consistent target(§2)은 접근축도 수렴 — ~1° 정합 시 조기 종료
                ca = float(np.dot(np.asarray(Tc.rotation)[:, 2], np.asarray(orient_R)[:, 2]))
                if ca > 0.9998:
                    break
        return self.clamp(q), err


# ────────────────────────── grasp 선택·경로 계획 (설계 §4~§8) ──────────────────────────

def cone_radius(z, p):
    """그릇 절두 원뿔 keep-out 반경(§8.1). ▽ 라 위로 벌어짐 — 테이블 근처 클리어런스 최대."""
    t = min(max((z - p.table_z) / p.bowl_h_rim, 0.0), 1.0)
    return p.bowl_r_base + (p.bowl_r_rim - p.bowl_r_base) * t + p.bowl_margin


def swept_clear(p_a, p_b, bowl_xy, half_w, p, n=12):
    """TCP 스윕 세그먼트(캡슐 근사)가 그릇 keep-out 을 안 침범하는지(§8.3).

    매달림 하단(TCP-hang: 큐브 바닥/jaw tip)이 rim+margin 위로 벗어난 샘플은 무조건 clear."""
    p_a = np.asarray(p_a, float); p_b = np.asarray(p_b, float)
    rim_z = p.table_z + p.bowl_h_rim
    for k in range(n + 1):
        q = p_a + (p_b - p_a) * (k / n)
        if q[2] - p.hang > rim_z + p.bowl_margin:
            continue
        if math.hypot(q[0] - bowl_xy[0], q[1] - bowl_xy[1]) <= cone_radius(q[2], p) + half_w:
            return False
    return True


def cart_line(ik, p_from, p_to, orient_R, q_seed, p):
    """Cartesian 직선(§7): step 마다 IK(직전 해 seed) → (q 경로, err_max). 접근/강하는
    orient_R(고정 접근축)로 정면 진입, lift/transit 은 None(position-only, 5-DOF best-effort)."""
    p_from = np.asarray(p_from, float); p_to = np.asarray(p_to, float)
    n = max(1, int(math.ceil(float(np.linalg.norm(p_to - p_from)) / p.cart_step)))
    q = q_seed
    path, emax = [], 0.0
    for k in range(1, n + 1):
        t = p_from + (p_to - p_from) * (k / n)
        q, e = ik.solve(t, orient_R=orient_R,
                        ori_cost=p.ori_cost if orient_R is not None else 0.0,
                        q_seed=q, iters=p.cart_iters, frame=TCP_FRAME)
        emax = max(emax, e)
        path.append(q)
    # 인접 IK 해의 joint zigzag(특히 position-only 구간 wrist 잔떨림) 저감 —
    # 끝점 고정 이동평균 2회. lift/transit 이 눈에 띄게 부드러워진다(user).
    for _ in range(2):
        for i in range(1, len(path) - 1):
            path[i] = 0.25 * path[i - 1] + 0.5 * path[i] + 0.25 * path[i + 1]
    return path, emax


def select_grasp(ik, cube, psi, bowl, p, log=print):
    """(±α, ρ) 스캔(§4·§6): 후보를 corridor 도달성 score 로 평가해 최적 grasp 선택.

    α 부호(§6 양방향): +α = 접근축이 base 반대쪽으로 기울어 wrist 가 base 쪽으로 당겨짐
    (원거리 2R 도달 해소 — 원거리 undershoot 방지), -α = 반대로 wrist 를 base 에서 밀어냄
    (근거리 최소반경/관절한계 해소 — 근거리 overshoot 방지). 게이트=grasp+hover 점,
    score 에 pre(via) err 포함 → top-down 이 corridor 까지 정확하면 즉시 채택, 아니면
    tilt 로 feasible set 확장. 반환 dict 또는 None(dead zone)."""
    cube = np.asarray(cube, float)
    az_cube = math.atan2(cube[1], cube[0])
    near = math.hypot(cube[0] - bowl[0], cube[1] - bowl[1]) < p.bowl_r_rim + p.near_bowl_margin
    tau = math.radians(p.tau_max_deg)
    # |α| 오름차순 ± interleave: 0, +5, -5, +10, -10, … — top-down 우선 정책 유지.
    alphas = [0.0]
    a_step = p.alpha_step_deg
    while a_step <= p.alpha_max_deg + 1e-9:
        alphas += [a_step, -a_step]
        a_step += p.alpha_step_deg
    # grasp z 하한: fixed jaw tip(TCP 아래 jaw_tip_drop, sim 실효)이 책상 위 jaw_floor_clear
    # 를 지키게 TCP z 를 clamp — tip 바닥접촉 방지(user). 단 pinch 밴드 ≈ TCP(캘리브 정의)라
    # z_g 가 큐브 중심 위로 올라간 만큼 grasp 마진이 깎인다(1cm 초과 시 헛닫힘 위험).
    z_g = max(cube[2], p.table_z + p.jaw_floor_clear + p.jaw_tip_drop)
    if z_g > cube[2] + 0.012:
        log(f"[grasp] ⚠ pinch 가 큐브 중심보다 {int((z_g - cube[2]) * 1000)}mm 위(floor rule) — "
            f"헛닫힘 위험, --jaw-floor-clear/--jaw-tip-drop 점검")
    it = min(p.iters, 700)   # 스캔용 반복 상한(좋은 후보는 조기종료라 싸다)
    approaches = (p.d_approach, p.d_approach_short) if near else (p.d_approach,)
    best = None
    for d_app in approaches:
        for a_deg in alphas:
            a = math.radians(float(a_deg))
            # pan 고정점 보정(3회): TCP lateral offset(R·r_tcp)이 ρ 와 함께 돌아 EE 가 pan
            # 평면을 벗어나면 QP 가 위치를 타협(양단 lateral miss) → EE 목표 방위로 재정렬.
            pan = az_cube
            for _ in range(3):
                dpsi = wrap90(psi - (pan + math.pi / 2.0))   # 닫힘축(ρ=0) 대비 face 오차
                rho = -dpsi / math.cos(a) + math.radians(p.roll_base_deg)
                R = grasp_R(pan, a, rho)
                c_hat = np.array([-math.sin(pan), math.cos(pan), 0.0])  # 닫힘축(+=moving jaw)
                p_g = cube + p.delta_bias * c_hat  # §5.1 lateral bias(닫힘이 자체 해소)
                p_g[2] = z_g
                ee_t = p_g - R @ ik.tcp_offset
                pan = math.atan2(ee_t[1], ee_t[0])
            if abs(dpsi) * abs(math.tan(a)) > tau:   # §4.2 jaw 수평이탈 제약
                continue
            p_pre = p_g - d_app * R[:, 2]                    # 접근축 후퇴 = pre-grasp(via)
            p_hov = p_g - min(p.hover, d_app) * R[:, 2]      # 저고도 corridor 점(sweep gate 동일)
            q_pre, e_pre = ik.solve(p_pre, orient_R=R, ori_cost=p.ori_cost,
                                    q_seed=ik.pan_seed(pan), iters=it, frame=TCP_FRAME)
            q_hov, e_hov = ik.solve(p_hov, orient_R=R, ori_cost=p.ori_cost, q_seed=q_pre,
                                    iters=it, frame=TCP_FRAME)
            q_g, e_g = ik.solve(p_g, orient_R=R, ori_cost=p.ori_cost, q_seed=q_hov,
                                iters=it, frame=TCP_FRAME)
            ax = axis_angle_deg(np.asarray(ik.fk(q_g, TCP_FRAME).rotation), R)
            # 게이트: grasp + hover(마지막 접근 corridor). pre 는 높은 via 라 err 커도
            # 무해(§6 2R) — 대신 score 로 반영해 corridor 가 나쁜 α 를 자연 배제.
            if e_g > p.pos_tol or e_hov > p.pos_tol or ax > p.axis_tol_deg:
                continue
            if near and not swept_clear(p_pre, p_g, bowl[:2], p.gripper_half_w, p):
                continue
            score = 3.0 * e_g + e_hov + 0.3 * e_pre + 2e-4 * abs(a_deg)  # top-down 선호 tie-break
            if best is None or score < best["score"]:
                best = {"pan": pan, "alpha": a, "rho": rho, "dpsi": dpsi, "R": R,
                        "p_pre": p_pre, "p_hov": p_hov, "p_g": p_g,
                        "q_pre": q_pre, "q_hov": q_hov, "q_g": q_g,
                        "e": (e_pre, e_hov, e_g, ax), "score": score,
                        "d_app": d_app, "near": near}
            if e_g < 0.006 and e_hov < 0.008 and e_pre < 0.02:   # 전 구간 정확 → 즉시 채택
                break
        if best is not None and best["e"][0] < 0.02 and best["e"][2] < 0.006:
            break
    if best is None:
        return None
    e_pre, e_hov, e_g, ax = best["e"]
    log(f"[grasp] pan={math.degrees(best['pan']):+.1f}° α={math.degrees(best['alpha']):+.0f}° "
        f"ρ={math.degrees(best['rho']):+.1f}° Δψ={math.degrees(best['dpsi']):+.1f}° "
        f"d_app={best['d_app']:.3f} near_bowl={best['near']} z_g={best['p_g'][2]:.3f} "
        f"g_err={e_g * 1000:.1f}mm hov={e_hov * 1000:.1f} pre={e_pre * 1000:.1f} axis={ax:.1f}°")
    return best


def plan_pick_place(ik, cube_b, psi_b, bowl_b, q_start, p, base_yaw_deg, log=print):
    """base_link 입력(위치·큐브 yaw) → solver 프레임 SM 계획(§7).

    반환 [(tag, path[q...], leg_sec)] 또는 None. 실행기는 [q_from]+path 인접점 lerp 에
    세그먼트 smoothstep 을 씌워 재생(결정적, 폐루프 아님). gripper 는 각 q 에 포함 —
    세그먼트 경계 보간이 open/close ramp 를 겸한다."""
    cube = rotz(cube_b, base_yaw_deg)
    bowl = rotz(bowl_b, base_yaw_deg)
    psi = psi_b + math.radians(base_yaw_deg)
    g = select_grasp(ik, cube, psi, bowl, p, log=log)
    if g is None:
        log("[plan] grasp dead zone — (α,ρ)·d_approach 스캔 전패(§8.2 케이스 C 또는 도달 밖)")
        return None
    qi = ik.qidx
    OPEN, CLOSE = feat_to_rad(p.grip_open), feat_to_rad(p.grip_close)

    def wg(q, grip):
        q = q.copy()
        q[qi["gripper"]] = grip
        return q

    # ① pan-first(§7): EE 안전 자세 유지한 채 pan 만 + 개구 — side-swipe 제거. pan 각은
    #    IK 해(q_pre)의 shoulder_pan 사용: URDF pan joint 부호가 solver azimuth 와 반대라
    #    g["pan"](azimuth) 직접 대입 시 큐브 반대 방향으로 돎.
    q_pan = q_start.copy()
    q_pan[qi["shoulder_pan"]] = g["q_pre"][qi["shoulder_pan"]]
    segs = [("pan_align", [wg(q_pan, OPEN)], p.leg_sec),
            ("pre_grasp", [wg(g["q_pre"], OPEN)], p.leg_sec)]
    # ③ 접근 2분할(2026-07-03 실패로그 fix): pre(높은 via, err 큼)→grasp 한 번에 내리면
    #    수렴 잔차를 큐브 존 안에서 xy 슬라이드로 풀어 큐브를 밀어냄(bulldozing, |ρ| 클수록
    #    jaw 슬롯 수직성분↑). 게이트로 정확성 보장된 hover 에 먼저 수렴(descend) 후,
    #    마지막 hover→grasp 는 짧고 정확한 순수 접근축 강하(approach).
    dsc, e_dsc = cart_line(ik, g["p_pre"], g["p_hov"], g["R"], g["q_pre"], p)
    segs.append(("descend", [wg(q, OPEN) for q in dsc], p.leg_sec))
    app, e_app = cart_line(ik, g["p_hov"], g["p_g"], g["R"], dsc[-1], p)
    segs.append(("approach", [wg(q, OPEN) for q in app], p.leg_sec))
    # close: 자세 유지, gripper 만 ramp(§5.3 — 닫힘폭 판정·effort 는 sim 쪽 clamp 가 담당)
    q_g = app[-1]
    segs.append(("grasp", [wg(q_g, CLOSE)], p.leg_sec))
    # ④ world 수직 lift → H_SAFE(§7·§8.3: rim+매달림+margin 이상, 도달 전 횡이동 금지)
    h_safe = max(bowl[2] + p.bowl_z,
                 g["p_g"][2] + p.lift,
                 p.table_z + p.bowl_h_rim + p.hang + p.bowl_margin)
    p_lift = np.array([g["p_g"][0], g["p_g"][1], h_safe])
    if not swept_clear(g["p_g"], p_lift, bowl[:2], p.gripper_half_w, p):
        log("[plan] ⚠ lift 경로 keep-out 접촉 — 근접 큐브 rim 사고 위험(§8.3), margin 재조정")
    lift, e_lift = cart_line(ik, g["p_g"], p_lift, None, q_g, p)
    segs.append(("lift", [wg(q, CLOSE) for q in lift], p.leg_sec))
    # ⑤ 등고 transit → 그릇 상공. 목표 xy = 그릇 중심에서 base 쪽으로 bowl_pull 당김
    #    (원거리 그릇서 far-rim 쪽 overshoot 관측 보정, user). h_safe ≥ rim+hang+margin.
    bxy = np.array([bowl[0], bowl[1]])
    r_b = float(np.linalg.norm(bxy))
    if r_b > 1e-6:
        bxy = bxy * max(0.0, 1.0 - p.bowl_pull / r_b)
    p_over = np.array([bxy[0], bxy[1], h_safe])
    tra, e_tra = cart_line(ik, p_lift, p_over, None, lift[-1], p)
    segs.append(("transit", [wg(q, CLOSE) for q in tra], p.leg_sec))
    # ⑥ release: transit 높이서 바로 떨굼(drop 세그먼트 제거 — user; ▽ self-centering)
    segs.append(("release", [wg(tra[-1], OPEN)], p.leg_sec * 0.5))  # user: release 시간 절반
    # ⑦ home = REST 자세(leader_calibration REST 중심값 — 시작자세 아님, user)
    q_rest = q_start.copy()
    for n, v in zip(SO101_JOINT_ORDER, REST_Q6):
        q_rest[qi[n]] = v
    segs.append(("home", [ik.clamp(q_rest)], p.home_sec))           # retreat 없음

    log(f"[plan] α={math.degrees(g['alpha']):.0f}° ρ={math.degrees(g['rho']):+.1f}° "
        f"h_safe={h_safe:.3f} segs={[(t, len(pth)) for t, pth, _ in segs]} "
        f"cart_err[mm] descend={e_dsc * 1000:.1f} approach={e_app * 1000:.1f} "
        f"lift={e_lift * 1000:.1f} transit={e_tra * 1000:.1f}")
    return segs, g


# ── grasp-sweep 궤적 생성용 상수 (bridge world 프레임에서 replay·physics 검증) ──
# home = bridge reset 자세(_START_POSE_RAD = pick_cube_env_cfg robot.init_state.joint_pos) 동일.
_SWEEP_HOME_DEG = {"shoulder_pan": 0.0, "shoulder_lift": -100.0, "elbow_flex": 90.0,
                   "wrist_flex": 70.0, "wrist_roll": -100.0}  # gripper=0 rad
HOME_Q6 = [math.radians(_SWEEP_HOME_DEG[n]) for n in SO101_JOINT_ORDER[:5]] + [0.0]
# REST 자세 = leader_calibration REST_POSE_RANGE 중심값(단일 소스) — home 세그먼트 복귀 목표.
try:
    from so101_contract.leader_calibration import SO101_FOLLOWER_REST_POSE_RANGE as _REST_RANGE
    REST_Q6 = [math.radians((_REST_RANGE[n][0] + _REST_RANGE[n][1]) / 2.0)
               for n in SO101_JOINT_ORDER]
except Exception:
    REST_Q6 = [math.radians(v) for v in (0.0, -100.0, 90.0, 50.0, 0.0, -10.0)]
# user 지시: REST = 위 중심값 자세에서 wrist_roll 만 -90° 로 돌린 상태.
REST_Q6[SO101_JOINT_ORDER.index("wrist_roll")] = math.radians(-90.0)
DESK_TOP_Z = 0.705           # world 책상 상판 (bridge CUBE_DESK_TOP_Z 정합)
CUBE_WORLD_Z_40MM = 0.726    # 40mm 큐브 중심 world z (desk+half+slack, _CUBE_INIT_STATES 정합)
# bowl world = pick_cube_env_cfg._BOWL_INIT_STATE (bridge place_defaults 와 동일).
BOWL_WORLD = [-0.22, 0.265, 0.715]


def self_check(args):
    print(f"[self-check] URDF: {args.urdf}")
    ik = PinkIK(args.urdf, wrist_roll_deg=args.wrist_roll_deg,
                tcp_offset=(args.tcp_dx, args.tcp_dy, args.tcp_dz))
    print(f"[self-check] solver={ik.solver} "
          f"tcp_offset=({args.tcp_dx},{args.tcp_dy},{args.tcp_dz}) "
          f"pan_gain={ik.pan_gain:+.2f}(az0={math.degrees(ik.pan_az0):+.1f}°)")

    # ① yaw 보상 기하(§4.2): 닫힘축 수평 방위 = 큐브 face(ψ) 정렬 + 접근축 pan 평면 구속(§2)
    for pan_d, psi_d, a_d in [(0, 0, 0), (20, 0, 0), (0, 30, 0), (25, 15, 20), (-30, -20, 10)]:
        pan, psi, a = (math.radians(v) for v in (pan_d, psi_d, a_d))
        dpsi = wrap90(psi - (pan + math.pi / 2.0))
        rho = -dpsi / math.cos(a)
        R = grasp_R(pan, a, rho)
        resid = abs(math.degrees(wrap90(math.atan2(R[1, 0], R[0, 0]) - psi)))
        off_plane = abs(-math.sin(pan) * R[0, 2] + math.cos(pan) * R[1, 2])  # z_ee 의 tangential 성분
        tol = 0.01 if a_d == 0 else 3.0   # α=0 완전 보상, tilt 는 1차 보상 잔차 허용(§4.2)
        assert resid <= tol, f"yaw-comp 잔차 {resid:.2f}° > {tol}° @ {(pan_d, psi_d, a_d)}"
        assert off_plane < 1e-9, f"접근축 pan 평면 이탈 {off_plane:.2e} @ {(pan_d, psi_d, a_d)}"
        print(f"[self-check]   yaw-comp pan={pan_d:+3d}° ψ={psi_d:+3d}° α={a_d:2d}° "
              f"→ ρ={math.degrees(rho):+6.1f}° face잔차={resid:.2f}°")

    q_home = pin.neutral(ik.model)
    for n, v in zip(SO101_JOINT_ORDER, HOME_Q6):
        q_home[ik.qidx[n]] = v
    q_home = ik.clamp(q_home)
    log = lambda m: print(f"[self-check] {m}")

    # ② 명목 배치(ψ=0) / ③ 큐브 yaw 20°(ρ 보상 경로) / ④ 그릇 근처(tilt 우선·keep-out 게이트)
    cases = [("nominal ψ=0°", DEFAULT_CUBE_XYZ, 0.0),
             ("yaw ψ=20°", DEFAULT_CUBE_XYZ, math.radians(20.0)),
             # 그릇 최근접 = DR min_bowl_sep(0.14) — 그보다 가까운 배치는 DR 이 안 만든다
             ("near-bowl", np.asarray(DEFAULT_BOWL_XYZ) + np.array([0.0, 0.14, 0.025]), 0.0),
             # DR bell 양단(env-local (∓0.24, 0.10) → base) — 양단 lateral miss·pan 방향 회귀
             ("left-end", (0.24, -0.10, 0.05), 0.0),
             ("right-end", (-0.24, -0.10, 0.05), 0.0),
             # 근거리(r=0.128, -α 후보 기대)·원거리(bell 먼 모서리 r=0.272, +α 후보 기대)
             ("near-arm", (-0.10, -0.08, 0.05), 0.0),
             ("far-corner", (0.08, -0.26, 0.05), 0.0)]
    for name, cube, psi in cases:
        print(f"[self-check] ── plan: {name} cube={np.round(np.asarray(cube, float), 3)}")
        res = plan_pick_place(ik, cube, psi, DEFAULT_BOWL_XYZ, q_home, args,
                              args.base_yaw_deg, log=log)
        assert res is not None, f"plan 실패: {name}"
        segs, _g = res
        for tag, path, _leg in segs:
            q = path[-1]
            tcp = ik.fk(q, TCP_FRAME).translation
            print(f"[self-check]   {tag:<9} n={len(path):2d} TCP={np.round(tcp, 3)} "
                  f"q={arm_deg(q, ik.qidx)} g={gripper_feat(q, ik.qidx)}")
    print("[self-check] PASS — yaw-comp 기하 + plan 7 케이스"
          "(nominal/yaw/near-bowl/양단/near-arm/far-corner)")
    return 0


def sweep(args):
    """큐브 (x,y) grid 를 훑으며 SO-101 이 top-down 으로 grasp 가능한 범위 측정.

    각 셀에서 grasp waypoint 를 top-down 방향 + 위치로 IK 풀어(폐루프 아님, 순수 kinematic)
    ①위치 도달 err<pos_tol ②achieved TCP z축이 수직에서 tilt_tol 이내 → graspable.

    프레임: grid 는 **env-local**(DR 범위와 동일, robot base 원점). base_link = 180° z 회전
    (xb=-xa, yb=-ya). 그 뒤 bridge 와 동일하게 rotz(base_yaw) 로 URDF-solver 프레임 진입.
    GRASP_ORIENT 는 nominal(고정 큐브)용이라, 큐브 azimuth 변화(Δφ)만큼 solver-z 로 회전시켜
    top-down 을 유지한 채 grasp yaw 를 팔 방위에 맞춘다(shoulder_pan 이 하는 일과 동형).
    """
    ik = PinkIK(args.urdf, args.wrist_roll_deg)
    print(f"[sweep] solver={ik.solver} URDF={args.urdf}")

    gx, gy, gz = args.grasp_dx, args.grasp_dy, args.grasp_z
    cz = args.cube_z
    byaw = args.base_yaw_deg

    xs = np.arange(args.xmin, args.xmax + 1e-9, args.step)
    ys = np.arange(args.ymin, args.ymax + 1e-9, args.step)
    tol = args.tilt_tol
    # 로봇암 주변 제외 박스(env-local) — 사용자: 책상 왼쪽끝서 X[35,48]cm·Y[0,20]cm.
    #   desk_left=env x -0.44, desk_front=env y -0.045 → base(0,0) straddle.
    ex_x0, ex_x1, ex_y0, ex_y1 = args.ex_x0, args.ex_x1, args.ex_y0, args.ex_y1

    def solve_topdown(gb, seed):
        """grasp point(base_link) → (q, err, tilt°). **고정 GRASP_ORIENT** = gripper finger
        axis 를 world 축(=identity 큐브 face)에 정렬한 top-down (사용자 요청: 큐브 face 정렬 grasp).
        큐브가 이동해도 방향 고정 → shoulder_pan 은 reach, wrist_roll 이 face 정렬 유지."""
        target = rotz(gb, byaw)
        q, err = ik.solve(target, orient_R=GRASP_ORIENT, ori_cost=args.ori_cost, q_seed=seed, iters=args.iters)
        tcp_z = np.asarray(ik.fk(q).rotation)[:, 2]
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -tcp_z[2]))))  # 수직(-z)에서 각
        return q, err, tilt

    def q6(qvec, grip_feat):
        """pink q(nq) + gripper feature → 6-vec(SO101_JOINT_ORDER, rad)."""
        return [float(qvec[ik.qidx[n]]) for n in SO101_JOINT_ORDER[:5]] + [feat_to_rad(grip_feat)]

    N = max(1, int(round(args.traj_leg_sec * args.hz)))  # 세그먼트당 dense step
    cells = []   # gen-traj: graspable 셀의 world 좌표 + dense pick→lift 궤적

    grid = {}   # (ix,iy) -> (tilt_deg, err_m). tilt=None → 로봇암 제외 박스
    for xa in xs:
        for ya in ys:
            key = (round(xa, 4), round(ya, 4))
            # 로봇암 주변 제외(사용자 지정) — 이 박스 안 큐브는 sweep 대상 아님.
            if ex_x0 <= xa <= ex_x1 and ex_y0 <= ya <= ex_y1:
                grid[key] = (None, None)
                continue
            # env-local → base_link(180° z). grasp point + 그 위 hover(수직 접근 corridor 검증).
            base_xy = np.array([-xa + gx, -ya + gy])
            grasp_p = np.array([base_xy[0], base_xy[1], cz + gz])
            hover_p = grasp_p + np.array([0.0, 0.0, args.hover])
            # hover 먼저(q_post seed) → grasp(hover q seed) — 실 접근 순서.
            qh, eh, th = solve_topdown(hover_p, ik.q_post)
            qg, eg, tg = solve_topdown(grasp_p, qh)
            # 셀 판정값 = hover·grasp 중 나쁜 쪽(둘 다 top-down 도달해야 수직 grasp 성립).
            graspable = max(eh, eg) <= args.pos_tol and max(th, tg) < tol
            grid[key] = (max(th, tg), max(eh, eg))

            # ── gen-traj: graspable 셀의 dense 궤적(home→approach→hover→descend→grasp→lift + hold) ──
            #   bridge 가 world 프레임서 큐브를 (xa,ya) 로 teleport 후 이 궤적을 replay,
            #   물리로 잡히는지(cube 상승) 검증. 고정 GRASP_ORIENT = 큐브 face 정렬 grasp.
            if args.gen_traj and graspable:
                # 큐브 위 높은 pre-approach → hover → 수직 하강. hover 는 큐브 top 위여야
                # descend 가 윗면을 찌르지 않는다(hover TCP z = grasp_z + traj_hover > cube_top).
                # gate 의 args.hover(0.04, deployed 공유)와 별개로 gen-traj 전용 높이 사용.
                appr_p = grasp_p + np.array([0.0, 0.0, args.approach])
                hov_p = grasp_p + np.array([0.0, 0.0, args.traj_hover])
                q_appr, _, _ = solve_topdown(appr_p, ik.q_post)
                q_hov, _, _ = solve_topdown(hov_p, q_appr)
                lift_p = grasp_p + np.array([0.0, args.lift_back, args.lift])
                q_lift, _ = ik.solve(rotz(lift_p, byaw), orient_R=None, q_seed=qg, iters=args.iters)
                wp = [HOME_Q6,
                      q6(q_appr, args.grip_open),  # pre-approach: 큐브 위 안전고도(찌름 방지)
                      q6(q_hov, args.grip_open),   # hover: 큐브 top 위
                      q6(qg, args.grip_open),      # descend: grasp 점까지 수직(open)
                      q6(qg, args.grip_close),     # close in place
                      q6(q_lift, args.grip_close)]  # lift (close)
                dense = [list(wp[0])]
                for a6, b6 in zip(wp[:-1], wp[1:]):
                    for k in range(1, N + 1):
                        f = smoothstep(k / N)
                        dense.append([a6[j] + f * (b6[j] - a6[j]) for j in range(6)])
                # 캡처 = **gripper 닫는 순간**(seg4=grasp_open→grasp_close 끝, index 4N). 큐브는
                # 아직 책상 위·손가락이 막 감쌈. (lift 끝 아님 — 사용자 요청.) 궤적은 lift 까지
                # 계속 돌아 성공(cube 상승) 판정은 그대로. 6 waypoint→5 seg: seg4 끝 = 4·N.
                capture_idx = 4 * N
                dense += [list(wp[-1])] * args.traj_hold
                cells.append({
                    "cube_world": [round(float(xa), 4), round(float(ya), 4), CUBE_WORLD_Z_40MM],
                    "capture_idx": capture_idx,
                    "traj": [[round(v, 5) for v in q] for q in dense],
                })

    # ── ASCII map (행=y forward↑, 열=x) ──────────────────────────────────
    def cell(xa, ya):
        tilt, err = grid[(round(xa, 4), round(ya, 4))]
        if tilt is None:
            return "A"          # 로봇암 주변 제외
        if err > args.pos_tol:
            return "x"          # 위치 도달 실패(unreachable)
        if tilt < 15.0:
            return "#"          # 완전 top-down
        if tilt < tol:
            return "+"          # top-down 허용범위 내
        if tilt < 40.0:
            return "."          # 기울지만 근접
        return ":"              # 위치는 되나 top-down 불가

    print(f"\n[sweep] top-down grasp map  (env-local, step={args.step}m, "
          f"pos_tol={args.pos_tol}m, tilt_tol={tol}°, ori_cost={args.ori_cost})")
    print("  '#'<15° top-down  '+'<%g° ok  '.'<40° tilt  ':'flat  'x'unreachable  'A'로봇암제외" % tol)
    print("  ★=robot base(0,0)  ◎=bowl  ●=nominal cube\n")
    print("        x(cm) →  " + " ".join(f"{x*100:+3.0f}" for x in xs))
    for ya in reversed(ys):
        row = []
        for xa in xs:
            c = cell(xa, ya)
            # landmark overlay
            if abs(xa) < args.step / 2 and abs(ya) < args.step / 2:
                c = "★"
            elif abs(xa - (-0.22)) < args.step / 2 and abs(ya - 0.265) < args.step / 2:
                c = "◎"
            elif abs(xa - (-0.015)) < args.step / 2 and abs(ya - 0.255) < args.step / 2:
                c = "●"
            row.append(f" {c} ")
        print(f"  y={ya*100:+5.1f}cm " + "".join(row))

    # ── 통계: graspable(#/+) 셀 bbox + inner radius ──────────────────────
    good = [(xa, ya) for (xa, ya), (t, e) in grid.items()
            if t is not None and e <= args.pos_tol and t < tol]
    if not good:
        print("\n[sweep] graspable 셀 없음 — tilt_tol/pos_tol 완화 필요")
        return 0
    gx_arr = np.array([g[0] for g in good])
    gy_arr = np.array([g[1] for g in good])
    radii = np.sqrt(gx_arr ** 2 + gy_arr ** 2)
    print(f"\n[sweep] graspable(top-down, tilt<{tol}°) 셀 {len(good)}개")
    print(f"  x 범위: [{gx_arr.min():+.3f}, {gx_arr.max():+.3f}] m")
    print(f"  y 범위: [{gy_arr.min():+.3f}, {gy_arr.max():+.3f}] m")
    print(f"  base 거리 r: [{radii.min():.3f}, {radii.max():.3f}] m  (min_base_sep 후보={radii.min():.3f})")
    # 여러 tilt_tol 에서의 bbox (판단용)
    for tt in (15.0, 20.0, 25.0, 30.0):
        gg = [(xa, ya) for (xa, ya), (t, e) in grid.items() if t is not None and e <= args.pos_tol and t < tt]
        if gg:
            ax = np.array([g[0] for g in gg]); ay = np.array([g[1] for g in gg])
            print(f"  tilt<{tt:>2.0f}°: n={len(gg):3d}  x[{ax.min():+.3f},{ax.max():+.3f}] "
                  f"y[{ay.min():+.3f},{ay.max():+.3f}]")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x_env", "y_env", "tilt_deg", "err_m", "graspable"])
            for (xa, ya), (t, e) in sorted(grid.items()):
                if t is None:
                    w.writerow([xa, ya, "arm", "arm", 0])  # 로봇암 제외
                    continue
                w.writerow([xa, ya, round(t, 2), round(e, 4),
                            int(e <= args.pos_tol and t < tol)])
        print(f"\n[sweep] CSV → {args.csv}")

    if args.gen_traj:
        import json
        out = {
            "hz": args.hz,
            "joint_order": list(SO101_JOINT_ORDER),
            "desk_top_z": DESK_TOP_Z,
            "lift_delta": args.lift_success,   # 성공 판정: cube z 가 spawn+lift_delta 초과
            "bowl_world": BOWL_WORLD,
            "cells": cells,
        }
        with open(args.gen_traj, "w") as f:
            json.dump(out, f)
        nframes = sum(len(c["traj"]) for c in cells)
        print(f"[sweep] gen-traj → {args.gen_traj}  ({len(cells)} graspable 셀, {nframes} frames, "
              f"leg={args.traj_leg_sec}s×{args.hz}Hz)")
    return 0


def run_ros(args):
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, JointState
    try:
        import tf2_ros
    except Exception:
        tf2_ros = None

    class PickPlace(Node):
        def __init__(self):
            super().__init__("pink_ik_bridge")
            self.ik = PinkIK(args.urdf, args.wrist_roll_deg,
                             tcp_offset=(args.tcp_dx, args.tcp_dy, args.tcp_dz))
            self.dt = 1.0 / args.hz
            self.p = args
            self.base_yaw = float(args.base_yaw_deg)

            self.q_meas = None
            self.q_start = None
            self.seq = None          # [(tag, path[q...], leg_sec)] — plan_pick_place 출력
            self.idx = 0
            self.q_from = None
            self.t = 0.0
            self.done = False
            self.finished = False   # record 완료 시 True → spin 루프 종료
            self.log_ct = 0
            self._grasp = None       # 폐루프 강하용 grasp 메타(_build 에서 채움)
            self.sink_ct = 0
            self.sink_z = None       # 단조 강하 명령 z(solver frame)
            self._sink_locked = False

            self.sub = self.create_subscription(JointState, "/isaac_joint_states", self._on_state, 10)
            self.pub = self.create_publisher(JointState, "/isaac_joint_commands", 10)
            self.tf_buf = None
            if tf2_ros is not None and not args.no_tf:
                self.tf_buf = tf2_ros.Buffer()
                self.tf_listener = tf2_ros.TransformListener(self.tf_buf, self)

            # ── 녹화(LeRobot v3) ────────────────────────────────────────────
            # state = q_meas(수신), action = 이번 tick publish 한 q_cmd. 둘 다 to_lerobot_units.
            # 이미지 = 3 캠 최신 프레임. 3 캠 + q_meas 모두 준비된 tick 부터 프레임 누적.
            self.writer = None
            self.imgs = {}   # cam → (H,W,3) uint8 최신 프레임
            self.rec_frames = 0
            if args.record:
                LeRobotV3DatasetWriter, self._to_units = _load_recorder()
                self.writer = LeRobotV3DatasetWriter(args.dataset_dir, overwrite=True,
                                                     enable_videos=True, robot_type="so_follower")
                for cam, topic in CAMERA_TOPICS.items():
                    self.create_subscription(Image, topic,
                                             lambda m, c=cam: self._on_image(c, m), 10)
                self.get_logger().info(f"녹화 ON → {args.dataset_dir} (task='{args.task_desc}')")

            self.timer = self.create_timer(self.dt, self._tick)
            self.get_logger().info(
                f"pink-ik pick-place SM · solver={self.ik.solver} · {args.hz}Hz · "
                f"leg={args.leg_sec}s · base_yaw={self.base_yaw}° · loop={args.loop}")

        def _on_state(self, msg):
            q = np.zeros(self.ik.model.nq) if self.q_meas is None else self.q_meas.copy()
            for name, val in zip(msg.name, msg.position):
                if name in self.ik.qidx:
                    q[self.ik.qidx[name]] = val
            self.q_meas = q
            if self.q_start is None:
                self.q_start = q.copy()
                self.get_logger().info(f"sim 시작자세(deg): {arm_deg(q, self.ik.qidx)} g={gripper_feat(q, self.ik.qidx)}")

        def _on_image(self, cam, msg):
            """sensor_msgs/Image → (H,W,3) uint8. cv_bridge 없이 raw buffer 디코드(rgb8/bgr8)."""
            arr = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.width, -1)
            if arr.shape[2] >= 3:
                arr = arr[:, :, :3]
                if msg.encoding.startswith("bgr"):
                    arr = arr[:, :, ::-1]
            self.imgs[cam] = np.ascontiguousarray(arr)

        def _tf(self, child, fallback):
            """(xyz, yaw, source). yaw = 큐브 face 정렬용(§4) — fallback 은 0."""
            if self.tf_buf is not None:
                try:
                    t = self.tf_buf.lookup_transform(BASE_FRAME, child, rclpy.time.Time())
                    tr, ro = t.transform.translation, t.transform.rotation
                    yaw = math.atan2(2.0 * (ro.w * ro.z + ro.x * ro.y),
                                     1.0 - 2.0 * (ro.y * ro.y + ro.z * ro.z))
                    return np.array([tr.x, tr.y, tr.z]), yaw, "tf"
                except Exception:
                    pass
            return np.array(fallback), 0.0, "fallback"

        def _build(self):
            if self.q_start is None:
                return False
            cube, psi, cs = self._tf(CUBE_FRAME, DEFAULT_CUBE_XYZ)
            bowl, _, bs = self._tf(BOWL_FRAME, DEFAULT_BOWL_XYZ)
            # 기동 직후 tf buffer 미충전이면 최대 5s 대기 — fallback 좌표로
            # 오계획하는 것을 방지(eval harness 의 위양성 실패 차단).
            if self.tf_buf is not None and "fallback" in (cs, bs):
                self._tf_wait = getattr(self, "_tf_wait", 0) + 1
                if self._tf_wait < int(5.0 * self.p.hz):
                    return False
            self.get_logger().info(
                f"cube({cs})={np.round(cube, 3)} ψ={math.degrees(psi):.1f}° "
                f"bowl({bs})={np.round(bowl, 3)} base_yaw={self.base_yaw}°")
            res = plan_pick_place(self.ik, cube, psi, bowl, self.q_start, self.p,
                                  self.base_yaw, log=self.get_logger().info)
            if res is None:
                self.get_logger().error("plan 실패(dead zone) — 정지")
                self.done = True
                return False
            segs, g = res
            self.seq = segs
            self._grasp = g          # 폐루프 강하용 grasp 메타(solver frame: p_g·R·z_g)
            self.sink_ct = 0         # 강하 피드백 스텝 카운터(에피소드/재계획당 리셋)
            self.sink_z = None
            self._sink_locked = False
            self.q_from = self.q_start.copy()
            self.idx = 0
            self.t = 0.0
            return True

        def _publish(self, q):
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = list(SO101_JOINT_ORDER)
            msg.position = [float(q[self.ik.qidx[n]]) for n in SO101_JOINT_ORDER]
            self.pub.publish(msg)

        def _tick(self):
            if self.q_meas is None or self.done:
                return
            if self.seq is None and not self._build():
                return

            tag, path, leg = self.seq[self.idx]

            # ── grasp 진입 시 폐루프 강하(Mode-A fix) ─────────────────────────
            # 낮은 wrist stiffness(7/4) 로 open-loop q_g 추종이 큐브 위 ~2.5cm 로 정착
            # (steady-state droop) → 헛닫힘(g→5, 큐브 desk 잔류). 측정 q 의 pink-FK TCP z 가
            # z_g 초과면 오차만큼 아래로 overshoot 재-IK 해 실제 큐브 높이까지 내린 뒤 close.
            # STOP 조건 = 측정 z ≤ z_g+tol(도달) 또는 budget 소진 → 정착 정상 config(PASS)는
            # 첫 tick 에 즉시 통과(무개입, 회귀 0). close 는 강하한 자세에서 제자리 수행.
            if tag == "grasp" and self._grasp is not None and self.q_meas is not None \
                    and not self._sink_locked:
                qi = self.ik.qidx
                z_tgt = float(self._grasp["p_g"][2])
                z_meas = float(self.ik.fk(self.ik.clamp(self.q_meas), TCP_FRAME).translation[2])
                if z_meas > z_tgt + self.p.sink_tol and self.sink_ct < self.p.sink_max:
                    # 단조 강하: 매 tick 측정보다 한 step 아래를 명령(command 는 되돌리지
                    # 않음→droop 이 어디서 멈추든 계속 밀어 극복). step 이 작아 lateral 스윙·
                    # 큐브 밀림 최소(exp2 의 big-overshoot 스윙 bulldoze fix). xy = grasp 점 고정.
                    if self.sink_z is None:
                        self.sink_z = z_meas
                    self.sink_z = min(self.sink_z, z_meas) - self.p.sink_step
                    self.sink_z = max(self.sink_z, z_tgt - self.p.sink_floor)   # 안전 하한
                    tgt = self._grasp["p_g"].copy()
                    tgt[2] = self.sink_z
                    q_sink, _e = self.ik.solve(tgt, orient_R=self._grasp["R"],
                                               ori_cost=self.p.ori_cost, q_seed=self.q_meas,
                                               iters=150, frame=TCP_FRAME)
                    q_sink[qi["gripper"]] = feat_to_rad(self.p.grip_open)  # 강하 중 개구 유지
                    self._publish(self.ik.clamp(q_sink))
                    self.sink_ct += 1
                    self.log_ct += 1
                    if self.log_ct % max(1, int(round(self.p.hz / 5))) == 0:
                        self.get_logger().info(
                            f"[sink#{self.idx}/{self.sink_ct}] z_meas={z_meas:.3f} cmd={self.sink_z:.3f} "
                            f"z_tgt={z_tgt:.3f} q={arm_deg(self.ik.clamp(self.q_meas), qi)} "
                            f"g={gripper_feat(self.ik.clamp(self.q_meas), qi)}")
                    return
                # 강하 완료/budget → 현재(강하한) 자세에서 gripper 만 CLOSE(제자리 파지)
                q_hold = self.ik.clamp(self.q_meas).copy()
                q_close = q_hold.copy(); q_close[qi["gripper"]] = feat_to_rad(self.p.grip_close)
                self.seq[self.idx] = (tag, [q_close], leg)
                self.q_from = q_hold
                self._sink_locked = True
                tag, path, leg = self.seq[self.idx]

            # [q_from]+path 인접점 lerp + 세그먼트 smoothstep — Cartesian 세그먼트는 path 가
            # dense IK 해라 직선 추종, joint 세그먼트는 기존과 동일한 1구간 보간.
            pts = [self.q_from] + path
            s = smoothstep(self.t / leg) * (len(pts) - 1)
            i = min(int(s), len(pts) - 2)
            q_cmd = self.ik.clamp(pts[i] + (s - i) * (pts[i + 1] - pts[i]))
            self._publish(q_cmd)

            # 녹화: home(복귀) 세그먼트는 제외 — 에피소드는 release/retreat 에서 끝냄.
            if self.writer is not None and tag != "home" and len(self.imgs) == len(CAMERA_TOPICS):
                st = self._to_units(np.array([self.q_meas[self.ik.qidx[n]] for n in SO101_JOINT_ORDER]))
                ac = self._to_units(np.array([q_cmd[self.ik.qidx[n]] for n in SO101_JOINT_ORDER]))
                self.writer.add_frame(ac, st, {c: self.imgs[c] for c in CAMERA_TOPICS})
                self.rec_frames += 1

            self.log_ct += 1
            if self.log_ct % max(1, int(round(self.p.hz / 5))) == 0:  # 0.2s
                qi = self.ik.qidx
                tcp_w = base_to_world(rotz(self.ik.fk(self.ik.clamp(self.q_meas), TCP_FRAME).translation,
                                           -self.base_yaw))
                # approach·grasp 중엔 큐브 tf 도 같이 — 하강/닫힘이 큐브를 미는지(변위) 실측
                cube_s = ""
                if tag in ("descend", "approach", "grasp") and self.tf_buf is not None:
                    c, _, src = self._tf(CUBE_FRAME, DEFAULT_CUBE_XYZ)
                    cube_s = f" cube({src})={np.round(c, 3)}"
                self.get_logger().info(
                    f"[{tag}#{self.idx}] meas W_tcp={np.round(tcp_w, 3)} "
                    f"q={arm_deg(self.ik.clamp(self.q_meas), qi)} "
                    f"g={gripper_feat(self.ik.clamp(self.q_meas), qi)}{cube_s}")

            self.t += self.dt
            if self.t >= leg:
                self.t = 0.0
                self.q_from = path[-1]
                self.idx += 1
                if self.idx >= len(self.seq):
                    if self.writer is not None:
                        # 녹화 모드: 1 에피소드(성공) flush + finalize 후 종료(--loop 무시).
                        committed = self.writer.commit_episode(True, self.p.task_desc)
                        summary = self.writer.finalize(self.p.task_desc)
                        self.get_logger().info(
                            f"녹화 완료: {self.rec_frames} frames · committed={committed} · "
                            f"{summary['total_episodes']}ep/{summary['total_frames']}f → {summary['output_dir']}")
                        self.done = True
                        self.finished = True   # spin 루프가 감지해 정상 종료
                        return
                    if self.p.loop:
                        self.idx = 0
                        self.seq = None  # tf 재조회(큐브 새 위치·yaw → 재계획)
                        self.get_logger().info("cycle done · re-pick")
                    else:
                        self.done = True
                        self.get_logger().info("pick-place 완료 · 자세 유지")
                else:
                    self.get_logger().info(f"→ {self.seq[self.idx][0]}")

    rclpy.init()
    node = PickPlace()
    try:
        # record 모드는 node.finished(1 에피소드 완료) 시 종료. 아니면 무한 유지(기존 동작).
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-check", action="store_true")
    # ── sweep: 큐브 위치 grid 를 훑어 top-down grasp 가능 범위 측정(오프라인 kinematic) ──
    ap.add_argument("--sweep", action="store_true",
                    help="큐브 (x,y) grid top-down grasp reachability 측정 → ASCII map+CSV")
    ap.add_argument("--xmin", type=float, default=-0.35, help="sweep env-local x 최소[m]")
    ap.add_argument("--xmax", type=float, default=0.35, help="sweep env-local x 최대[m]")
    ap.add_argument("--ymin", type=float, default=0.06, help="sweep env-local y 최소[m]")
    ap.add_argument("--ymax", type=float, default=0.46, help="sweep env-local y 최대[m]")
    ap.add_argument("--step", type=float, default=0.02, help="sweep grid 간격[m]")
    ap.add_argument("--cube-z", dest="cube_z", type=float, default=0.05,
                    help="큐브 중심 z(base_link)[m] — 40mm 큐브=책상+반높이≈0.05")
    ap.add_argument("--tilt-tol", dest="tilt_tol", type=float, default=25.0,
                    help="top-down 판정: achieved TCP z축이 수직에서 이 각[deg] 이내")
    ap.add_argument("--pos-tol", dest="pos_tol", type=float, default=0.02,
                    help="위치 도달 판정 err 상한[m] (sweep·grasp 선택 공용)")
    ap.add_argument("--iters", type=int, default=1500, help="waypoint IK 반복 상한(수렴 시 조기종료)")
    ap.add_argument("--csv", default=None, help="셀별 tilt/err CSV 출력 경로")
    # 로봇암 주변 제외 박스(env-local m). 사용자: 책상 왼쪽끝(env x -0.44)서 X[35,48]cm,
    # 책상 앞모서리(env y -0.045)서 Y[0,20]cm → base(0,0) straddle.
    ap.add_argument("--ex-x0", dest="ex_x0", type=float, default=-0.09)
    ap.add_argument("--ex-x1", dest="ex_x1", type=float, default=0.04)
    ap.add_argument("--ex-y0", dest="ex_y0", type=float, default=-0.045)
    ap.add_argument("--ex-y1", dest="ex_y1", type=float, default=0.155)
    # gen-traj: graspable 셀마다 dense pick→lift 궤적을 dump → bridge 가 world 프레임서 replay·물리검증.
    ap.add_argument("--gen-traj", dest="gen_traj", default=None,
                    help="graspable 셀의 home→hover→descend→grasp→lift 궤적 JSON 출력 경로")
    ap.add_argument("--traj-leg-sec", dest="traj_leg_sec", type=float, default=2.0,
                    help="gen-traj 세그먼트당 이동 시간[s] (deployed node leg_sec 정합)")
    ap.add_argument("--approach", type=float, default=0.14,
                    help="gen-traj pre-approach 높이[m] (grasp점 위, 찌름 방지 안전고도)")
    ap.add_argument("--traj-hover", dest="traj_hover", type=float, default=0.10,
                    help="gen-traj hover 높이[m] (grasp점 위, 큐브 top 위여야: >0.064)")
    ap.add_argument("--traj-hold", dest="traj_hold", type=int, default=15,
                    help="gen-traj lift 후 정지 프레임(캡처 안정)")
    ap.add_argument("--lift-success", dest="lift_success", type=float, default=0.04,
                    help="grasp 성공 판정: cube z 가 spawn+이 값[m] 초과하면 잡아 든 것")
    ap.add_argument("--hover", type=float, default=0.04,
                    help="grasp점 위 hover corridor gate 높이[m] (SM grasp 선택·sweep 공용)")
    ap.add_argument("--lift-back", dest="lift_back", type=float, default=0.06,
                    help="(sweep gen-traj 전용) lift 시 로봇쪽(+y)으로 당기는 양[m]")
    ap.add_argument("--grasp-dx", dest="grasp_dx", type=float, default=0.0,
                    help="(sweep 전용) grasp 측면 오프셋 x[m] — SM 본체는 TCP 프레임 사용")
    ap.add_argument("--grasp-dy", dest="grasp_dy", type=float, default=-0.016,
                    help="(sweep 전용) grasp 측면 오프셋 y[m] (실=-0.016)")
    ap.add_argument("--grasp-z", dest="grasp_z", type=float, default=-0.043,
                    help="(sweep 전용) 큐브 중심 기준 grasp z[m] (실=-0.043)")
    # ── 공통 ──
    ap.add_argument("--urdf", default=DEFAULT_URDF)
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--leg-sec", dest="leg_sec", type=float, default=2.0, help="세그먼트당 이동 시간[s]")
    ap.add_argument("--home-sec", dest="home_sec", type=float, default=2.0,
                    help="완료 후 home 복귀 시간[s] (teleport 방지 ramp)")
    ap.add_argument("--base-yaw-deg", dest="base_yaw_deg", type=float, default=90.0,
                    help="URDF↔sim base z 보정[deg]. cube pan 이 sim 0 근처 되게(기본 90)")
    ap.add_argument("--wrist-roll-deg", dest="wrist_roll_deg", type=float, default=-99.0,
                    help="PostureTask wrist_roll bias[deg] (IK seed·정칙화)")
    ap.add_argument("--ori-cost", dest="ori_cost", type=float, default=1.0,
                    help="접근축 orientation 타겟 가중치(5-DOF best-effort)")
    # ── SM: TCP(§3) — 검증 grasp 역산 EE-local 오프셋, GPU 재검증 시 여기만 갱신 ──
    ap.add_argument("--tcp-dx", dest="tcp_dx", type=float, default=DEFAULT_TCP_OFFSET[0],
                    help="TCP(tcp_grasp) EE-local x[m]")
    ap.add_argument("--tcp-dy", dest="tcp_dy", type=float, default=DEFAULT_TCP_OFFSET[1],
                    help="TCP(tcp_grasp) EE-local y[m]")
    ap.add_argument("--tcp-dz", dest="tcp_dz", type=float, default=DEFAULT_TCP_OFFSET[2],
                    help="TCP(tcp_grasp) EE-local z[m]")
    # ── SM: grasp 선택(§4·§6) ──
    ap.add_argument("--delta-bias", dest="delta_bias", type=float, default=0.0,
                    help="§5.1 lateral bias[m], +=moving jaw(닫힘축 +x_ee) 쪽. 보고서 권장 2~4mm — "
                         "0=검증된 기존 조준 유지, GPU 재검증 후 상향")
    ap.add_argument("--tau-max-deg", dest="tau_max_deg", type=float, default=10.0,
                    help="§4.2 jaw 수평이탈 허용 |Δψ·tanα| 상한[deg]")
    ap.add_argument("--alpha-max-deg", dest="alpha_max_deg", type=float, default=45.0,
                    help="접근축 tilt 상한[deg] — 스캔은 ±양방향(+=wrist를 base쪽=원거리용, "
                         "-=반대=근거리용)")
    ap.add_argument("--alpha-step-deg", dest="alpha_step_deg", type=float, default=5.0,
                    help="α 스캔 간격[deg]")
    ap.add_argument("--roll-base-deg", dest="roll_base_deg", type=float, default=0.0,
                    help="ρ 에 더할 고정 roll offset[deg] (실측 grasp 미세정렬용)")
    ap.add_argument("--axis-tol-deg", dest="axis_tol_deg", type=float, default=15.0,
                    help="grasp feasibility: achieved 접근축이 타겟에서 이 각[deg] 이내")
    ap.add_argument("--d-approach", dest="d_approach", type=float, default=0.09,
                    help="§7 pre-grasp 후퇴 거리[m] (접근축 -z)")
    ap.add_argument("--d-approach-short", dest="d_approach_short", type=float, default=0.04,
                    help="§8.2 케이스 C fallback: 짧은 pre-grasp 후퇴[m]")
    ap.add_argument("--cart-step", dest="cart_step", type=float, default=0.007,
                    help="Cartesian 직선 IK step[m] (§7: 5~10mm)")
    ap.add_argument("--cart-iters", dest="cart_iters", type=int, default=200,
                    help="Cartesian step 당 IK 반복 상한(직전 해 seed 라 작게)")
    # ── SM: 그릇 keep-out(§8) — author_pick_cube_scene.py BOWL_* 실측 ──
    ap.add_argument("--table-z", dest="table_z", type=float, default=0.030,
                    help="책상 상판 z(base_link)[m] = world 0.705 - robot 0.6749")
    ap.add_argument("--bowl-r-base", dest="bowl_r_base", type=float, default=0.033,
                    help="그릇 바닥 반경[m] (BOWL_R_BOTTOM)")
    ap.add_argument("--bowl-r-rim", dest="bowl_r_rim", type=float, default=0.075,
                    help="그릇 rim 반경[m] (BOWL_R_TOP)")
    ap.add_argument("--bowl-h-rim", dest="bowl_h_rim", type=float, default=0.075,
                    help="rim 높이[m] (테이블 기준, BOWL_LOCAL z + z_base + depth)")
    ap.add_argument("--bowl-margin", dest="bowl_margin", type=float, default=0.01,
                    help="keep-out 안전 margin[m]")
    ap.add_argument("--near-bowl-margin", dest="near_bowl_margin", type=float, default=0.08,
                    help="근접 판정: cube-bowl 거리 < r_rim+이 값 → swept keep-out 게이트 + "
                         "short-approach fallback (DR min_bowl_sep=0.14 커버)")
    ap.add_argument("--gripper-half-w", dest="gripper_half_w", type=float, default=0.04,
                    help="gripper(+큐브) 스윕 캡슐 반폭[m]")
    ap.add_argument("--hang", dest="hang", type=float, default=0.03,
                    help="TCP 아래 최저 매달림[m] — sim 실효 jaw tip(~0.02)·파지 큐브 바닥(~0.03). "
                         "rim 통과·H_SAFE 판정")
    # ── SM: grasp 깊이·release 위치 (user 실측 피드백 반영) ──
    ap.add_argument("--jaw-tip-drop", dest="jaw_tip_drop", type=float, default=0.02,
                    help="TCP→fixed jaw tip 수직 거리[m], **sim 실효값**. URDF mesh 는 47mm 지만 "
                         "sim(USD so101_new_calib 영점 차)에선 TCP 0.05 명령 시 tip 책상접촉 "
                         "관측=실효 ~20mm. ⚠0.047 로 두면 pinch 밴드(≈TCP)가 큐브 위 허공 → "
                         "gripper 헛닫힘(2026-07-03 로그 재현)")
    ap.add_argument("--jaw-floor-clear", dest="jaw_floor_clear", type=float, default=0.010,
                    help="grasp 시 fixed jaw tip 의 책상 최소 이격[m] (tip 바닥접촉 방지)")
    ap.add_argument("--bowl-pull", dest="bowl_pull", type=float, default=0.03,
                    help="transit/release xy 를 그릇 중심에서 base 쪽으로 당기는 양[m] "
                         "(원거리 far-rim overshoot 보정)")
    # ── SM: 높이·gripper ──
    ap.add_argument("--lift", type=float, default=0.085, help="grasp 후 최소 들어올림 높이[m]")
    ap.add_argument("--bowl-z", dest="bowl_z", type=float, default=0.113,
                    help="그릇 중심 위 release TCP(=큐브 중심) 높이[m] — rim 상공 drop(§8.3)")
    ap.add_argument("--grip-open", dest="grip_open", type=float, default=47.0,
                    help="open gripper feature[0,100] (실=47; §5.2 개구 부등식은 실측 캘리브 대상)")
    ap.add_argument("--sink-tol", dest="sink_tol", type=float, default=0.005,
                    help="grasp 폐루프 강하 STOP 임계[m]: 측정 TCP z ≤ z_g+tol 이면 close")
    ap.add_argument("--sink-max", dest="sink_max", type=int, default=60,
                    help="grasp 폐루프 강하 최대 tick(30Hz). droop 못 이기면 도달치서 close")
    ap.add_argument("--sink-step", dest="sink_step", type=float, default=0.004,
                    help="단조 강하 tick 당 하강 명령 step[m]. 작을수록 lateral 스윙·큐브 밀림↓")
    ap.add_argument("--sink-floor", dest="sink_floor", type=float, default=0.045,
                    help="강하 명령 z 안전 하한 = z_g-floor[m] (jaw tip 책상 penetration 방지)")
    ap.add_argument("--grip-close", dest="grip_close", type=float, default=5.0,
                    help="close gripper feature[0,100] (실=5; effort 제한은 sim gripper clamp 담당)")
    ap.add_argument("--loop", action="store_true", help="완료 후 재집기 반복(tf 재조회·재계획)")
    ap.add_argument("--no-tf", dest="no_tf", action="store_true")
    # 녹화: 폐루프 궤적(state=/isaac_joint_states, action=publish 한 command, images=3 캠)을
    # LeRobot v3 로 기록. 1 에피소드 완료 시 finalize+종료. --loop 는 무시(1회만).
    ap.add_argument("--record", action="store_true", help="pick-place 1 에피소드를 LeRobot v3 로 기록")
    ap.add_argument("--dataset-dir", dest="dataset_dir",
                    default="/workspace/datasets/pink_ik_pickplace",
                    help="LeRobot v3 출력 폴더(--record)")
    ap.add_argument("--task-desc", dest="task_desc",
                    default="pick up the cube and place it in the bowl",
                    help="에피소드 task 문자열(--record)")
    args = ap.parse_args()
    if args.self_check:
        return self_check(args)
    if args.sweep:
        return sweep(args)
    return run_ros(args)


if __name__ == "__main__":
    raise SystemExit(main())
