#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SO-101 pick-place — pink IK(프레임 보정) + timed joint-space 보간으로 Isaac Sim bridge 구동.

waypoint 시퀀스를 각 1회 IK 로 풀어 관절각으로 만들고, smoothstep joint-space 보간으로
순차 이동한다(폐루프 IK 아님 → droop·watchdog 없이 결정적). gripper 는 waypoint 별 open/close.

순서: hover_cube → descend → grasp(47→5) → lift(위+뒤) → over_bowl(높이유지)
      → release → retreat → home(2s ramp) → (--loop 시 재집기, 아니면 정지).

프레임 보정(핵심): URDF base_link 이 sim(USD) base_link 보다 z축 base_yaw(기본 90°) 어긋남.
→ 모든 EE 목표를 Rz(base_yaw) 회전 후 IK. base 회전은 joint 불변이라 결과 joint 그대로 publish.
(cube tf 를 그냥 쓰면 shoulder_pan 이 sim 0° 대신 ~97° 로 나와 90° 빗나감.)

grasp: SO-101 한쪽 jaw 고정·한쪽 moving → top-down 접근이 필수. 방향/깊이/그리퍼 값은
실 작업공간 성공 trajectory(scripts/ece_4560/real/sequences/pick_place_demo.json)를
follower_calibration 으로 sim 변환·역산해 기본값 확정(GRASP_ORIENT·grasp_z·gripper 47/5 등).

자가검증:  python pink_ik_bridge_node.py --self-check
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
BASE_FRAME = "base_link"
CUBE_FRAME = "Cube1"
BOWL_FRAME = "Bowl"

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
# TCP z축 ≈ [-0.08,0,-0.997] = top-down(아래). 5-DOF 로 도달 가능(ori_cost 0.3+ 에서 0° 오차).
# 큐브 azimuth(pan~0) 기준 — 큐브 위치 크게 바뀌면 회전 필요(현재 고정 큐브용).
GRASP_ORIENT = np.array([
    [0.1520, 0.9852, -0.0795],
    [0.9882, -0.1528, -0.0042],
    [-0.0163, -0.0780, -0.9968],
])


def base_to_world(p):
    """base_link → world: 회전 없이 robot 위치만 더함(user 지시). world = base + (0,0,0.6749)."""
    return np.asarray(p, dtype=float) + ROBOT_WORLD


def rotz(p, deg):
    """z축 deg 회전. URDF base ↔ sim base 90° 어긋남 보정."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    p = np.asarray(p, dtype=float)
    return np.array([c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]])


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
    def __init__(self, urdf_path, wrist_roll_deg):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        if not self.model.existFrame(EE_FRAME):
            raise RuntimeError(f"URDF 에 EE 프레임 '{EE_FRAME}' 없음")
        self.qidx = {n: self.model.joints[self.model.getJointId(n)].idx_q
                     for n in SO101_JOINT_ORDER}
        self.lo = self.model.lowerPositionLimit.copy()
        self.hi = self.model.upperPositionLimit.copy()
        self.solver = _solver()
        self.q_post = pin.neutral(self.model)
        self.q_post[self.qidx["wrist_roll"]] = math.radians(wrist_roll_deg)
        self.q_post = self.clamp(self.q_post)

    def clamp(self, q):
        return np.clip(q, self.lo + 1e-4, self.hi - 1e-4)

    def fk(self, q):
        return pink.Configuration(self.model, self.data, q).get_transform_frame_to_world(EE_FRAME)

    def solve(self, target_xyz, orient_R=None, ori_cost=0.0, q_seed=None, iters=4000):
        """target_xyz(URDF base, 회전 적용됨) → (q, err[m]). orient_R 주면 그 방향도 타겟
        (ori_cost, 5-DOF best-effort). 없으면 position-only."""
        target = np.asarray(target_xyz, dtype=float)
        oc = float(ori_cost) if orient_R is not None else 0.0
        ee_task = FrameTask(EE_FRAME, position_cost=1.0, orientation_cost=oc, lm_damping=1.0)
        post = PostureTask(cost=1e-2); post.set_target(self.q_post)
        q = self.clamp(self.q_post.copy() if q_seed is None else q_seed.copy())
        err = 9.0
        for _ in range(iters):
            cfg = pink.Configuration(self.model, self.data, q)
            Tc = cfg.get_transform_frame_to_world(EE_FRAME)
            err = float(np.linalg.norm(target - Tc.translation))
            R = orient_R if orient_R is not None else Tc.rotation
            ee_task.set_target(pin.SE3(R, target))
            try:
                v = solve_ik(cfg, [ee_task, post], 1 / 30, solver=self.solver, damping=1e-6, safety_break=False)
            except Exception:
                v = np.zeros(self.model.nv)
            q = pin.integrate(self.model, q, v / 30)
            if err < 3e-4 and orient_R is None:
                break
        return self.clamp(q), err


def pickplace_waypoints(cube, bowl, p):
    """sim base_link EE 목표 + gripper feature 리스트. cube/bowl = sim base xyz."""
    gx, gy, h, lift, gz, bz, lb = (p.grasp_dx, p.grasp_dy, p.hover, p.lift,
                                   p.grasp_z, p.bowl_z, p.lift_back)
    OPEN, CLOSE = p.grip_open, p.grip_close
    cube = np.asarray(cube); bowl = np.asarray(bowl)
    grasp = cube + np.array([gx, gy, gz])   # 실제 TCP grasp 점(큐브 중심 아래)
    # hover 는 grasp 점 기준(+z). lift 는 위(+lift)+뒤(로봇쪽 +y, lift_back) — 실 lift 처럼
    #   팔 당겨 높이 올려야 그릇으로 이동 시 낮게 안 쓸고 넘어감(z_min≈0.09).
    # bowl 은 그릇 위 높이(bz≈0.11) 유지 — 내려가면 그릇 침(실도 안 내려감).
    # 4번째 = grasp 방향(top-down) 타겟. 큐브측 True(잡기), lift·그릇측 False(위치우선).
    return [
        ("hover_cube", grasp + np.array([0, 0, h]),        OPEN,  True),
        ("descend",    grasp,                               OPEN,  True),
        ("grasp",      grasp,                               CLOSE, True),
        ("lift",       grasp + np.array([0, lb, lift]),     CLOSE, False),
        ("over_bowl",  bowl + np.array([0, 0, bz]),         CLOSE, False),
        ("release",    bowl + np.array([0, 0, bz]),         OPEN,  False),
        ("retreat",    bowl + np.array([0, 0, bz + 0.03]),  OPEN,  False),
    ]


def self_check(urdf_path):
    print(f"[self-check] URDF: {urdf_path}")
    ik = PinkIK(urdf_path, wrist_roll_deg=-99.0)
    print(f"[self-check] solver={ik.solver}")
    p = argparse.Namespace(grasp_dx=0.0, grasp_dy=-0.016, hover=0.04, lift=0.085,
                           grasp_z=-0.043, grip_open=47.0, grip_close=5.0, bowl_z=0.113,
                           lift_back=0.06)
    wps = pickplace_waypoints(DEFAULT_CUBE_XYZ, DEFAULT_BOWL_XYZ, p)
    seed = ik.q_post.copy()
    ok = 0
    for tag, xyz, grip, use_o in wps:
        R = GRASP_ORIENT if use_o else None
        q, e = ik.solve(rotz(xyz, 90.0), orient_R=R, ori_cost=0.5, q_seed=seed)
        seed = q
        arm = arm_deg(q, ik.qidx)
        good = e < 0.03
        ok += good
        print(f"[self-check]   {tag:<11} err={e*1000:5.1f}mm q={arm} {'OK' if good else 'MISS'}")
    assert ok == len(wps), f"waypoint IK {ok}/{len(wps)}"
    print(f"[self-check] PASS — {ok}/{len(wps)} waypoint(grasp 방향 top-down, 실 trajectory 정렬)")
    return 0


# ── grasp-sweep 궤적 생성용 상수 (bridge world 프레임에서 replay·physics 검증) ──
# home = bridge reset 자세(_START_POSE_RAD = pick_cube_env_cfg robot.init_state.joint_pos) 동일.
_SWEEP_HOME_DEG = {"shoulder_pan": 0.0, "shoulder_lift": -100.0, "elbow_flex": 90.0,
                   "wrist_flex": 70.0, "wrist_roll": -100.0}  # gripper=0 rad
HOME_Q6 = [math.radians(_SWEEP_HOME_DEG[n]) for n in SO101_JOINT_ORDER[:5]] + [0.0]
DESK_TOP_Z = 0.705           # world 책상 상판 (bridge CUBE_DESK_TOP_Z 정합)
CUBE_WORLD_Z_40MM = 0.726    # 40mm 큐브 중심 world z (desk+half+slack, _CUBE_INIT_STATES 정합)
# bowl world = pick_cube_env_cfg._BOWL_INIT_STATE (bridge place_defaults 와 동일).
BOWL_WORLD = [-0.22, 0.265, 0.715]


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
    hdr = "      " + "".join(f"{x*100:+03.0f}"[:1] if False else "" for x in xs)
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
            self.ik = PinkIK(args.urdf, args.wrist_roll_deg)
            self.dt = 1.0 / args.hz
            self.p = args
            self.base_yaw = float(args.base_yaw_deg)

            self.q_meas = None
            self.q_start = None
            self.seq = None          # [(tag, q, leg), ...] 순차 이동 목표
            self.idx = 0
            self.q_from = None
            self.t = 0.0
            self.done = False
            self.finished = False   # record 완료 시 True → spin 루프 종료
            self.cube_xyz = None
            self.log_ct = 0

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
                f"pink-ik pick-place · solver={self.ik.solver} · {args.hz}Hz · "
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
            if self.tf_buf is not None:
                try:
                    t = self.tf_buf.lookup_transform(BASE_FRAME, child, rclpy.time.Time())
                    tr = t.transform.translation
                    return np.array([tr.x, tr.y, tr.z]), "tf"
                except Exception:
                    pass
            return np.array(fallback), "fallback"

        def _build(self):
            if self.q_start is None:
                return False
            cube, cs = self._tf(CUBE_FRAME, DEFAULT_CUBE_XYZ)
            bowl, bs = self._tf(BOWL_FRAME, DEFAULT_BOWL_XYZ)
            self.cube_xyz = cube
            wps = pickplace_waypoints(cube, bowl, self.p)
            qi = self.ik.qidx
            seed = self.q_start.copy()
            seq = []
            self.get_logger().info(f"cube({cs})={np.round(cube,3)} bowl({bs})={np.round(bowl,3)} base_yaw={self.base_yaw}°")
            for tag, xyz, grip, use_o in wps:
                R = GRASP_ORIENT if use_o else None
                q, e = self.ik.solve(rotz(xyz, self.base_yaw), orient_R=R,
                                     ori_cost=self.p.ori_cost, q_seed=seed)
                q[qi["gripper"]] = feat_to_rad(grip)
                seed = q
                seq.append((tag, q, float(self.p.leg_sec)))
                ee_w = base_to_world(rotz(self.ik.fk(q).translation, -self.base_yaw))
                flag = "" if e < 0.03 else " ⚠MISS"
                self.get_logger().info(
                    f"  {tag:<11} W_ee={np.round(ee_w,3)} pan={math.degrees(q[qi['shoulder_pan']]):5.1f}° "
                    f"g={gripper_feat(q,qi)} err={e*1000:.0f}mm{flag}")
            # home(=시작자세) 복귀를 항상 추가 — loop 여도 retreat→home 을 부드럽게 보간해야
            # 재시작 시 q_from=q_start 점프(teleport) 없이 이어짐(팔이 이미 home 에 있음).
            seq.append(("home", self.q_start.copy(), float(self.p.home_sec or self.p.leg_sec)))
            self.seq = seq
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

            tag, q_to, leg = self.seq[self.idx]
            frac = smoothstep(self.t / leg)
            q_cmd = self.ik.clamp(self.q_from + frac * (q_to - self.q_from))
            self._publish(q_cmd)

            # 녹화: home(복귀) 세그먼트는 제외 — 에피소드는 큐브가 그릇에 놓인 retreat 에서 끝냄.
            if self.writer is not None and tag != "home" and len(self.imgs) == len(CAMERA_TOPICS):
                st = self._to_units(np.array([self.q_meas[self.ik.qidx[n]] for n in SO101_JOINT_ORDER]))
                ac = self._to_units(np.array([q_cmd[self.ik.qidx[n]] for n in SO101_JOINT_ORDER]))
                self.writer.add_frame(ac, st, {c: self.imgs[c] for c in CAMERA_TOPICS})
                self.rec_frames += 1

            self.log_ct += 1
            if self.log_ct % max(1, int(round(self.p.hz / 5))) == 0:  # 0.2s
                qi = self.ik.qidx
                ee_w = base_to_world(rotz(self.ik.fk(self.ik.clamp(self.q_meas)).translation, -self.base_yaw))
                self.get_logger().info(
                    f"[{tag}#{self.idx}] meas W_ee={np.round(ee_w,3)} q={arm_deg(self.ik.clamp(self.q_meas),qi)} g={gripper_feat(self.ik.clamp(self.q_meas),qi)}")

            self.t += self.dt
            if self.t >= leg:
                self.t = 0.0
                self.q_from = q_to
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
                        self.seq = None  # tf 재조회(큐브 새 위치)
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
                    help="위치 도달 판정 err 상한[m]")
    ap.add_argument("--iters", type=int, default=1500, help="셀당 IK 반복(수렴)")
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
    ap.add_argument("--urdf", default=DEFAULT_URDF)
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--leg-sec", dest="leg_sec", type=float, default=2.0, help="waypoint 당 이동 시간[s]")
    ap.add_argument("--home-sec", dest="home_sec", type=float, default=2.0,
                    help="완료 후 home 복귀 시간[s] (teleport 방지 ramp)")
    ap.add_argument("--base-yaw-deg", dest="base_yaw_deg", type=float, default=90.0,
                    help="URDF↔sim base z 보정[deg]. cube pan 이 sim 0 근처 되게(기본 90)")
    # grasp 기본값 = 실 작업공간 성공 trajectory(pick_place_demo.json) 역산값.
    ap.add_argument("--wrist-roll-deg", dest="wrist_roll_deg", type=float, default=-99.0)
    ap.add_argument("--ori-cost", dest="ori_cost", type=float, default=0.5,
                    help="grasp 방향(top-down) 타겟 가중치(5-DOF best-effort, 실 grasp 재현)")
    ap.add_argument("--hover", type=float, default=0.04,
                    help="큐브 grasp점 위 hover 높이[m]. top-down 유지 가능 범위(5-DOF, ~0.05 한계)")
    ap.add_argument("--lift", type=float, default=0.085, help="grasp 후 들어올림 높이[m] (실≈0.084)")
    ap.add_argument("--lift-back", dest="lift_back", type=float, default=0.06,
                    help="lift 시 로봇쪽(+y)으로 당기는 양[m]. 팔 당겨 올려 그릇 이동 시 낮게 안 쓸음")
    ap.add_argument("--bowl-z", dest="bowl_z", type=float, default=0.113,
                    help="그릇 중심 위 release 높이[m] (실=0.113, 안 내려가고 떨궈 넣음)")
    ap.add_argument("--grasp-z", dest="grasp_z", type=float, default=-0.043,
                    help="큐브 중심 기준 grasp z[m] (실=-0.043, TCP 가 큐브 아래)")
    ap.add_argument("--grasp-dx", dest="grasp_dx", type=float, default=0.0, help="grasp 측면 오프셋 x[m]")
    ap.add_argument("--grasp-dy", dest="grasp_dy", type=float, default=-0.016, help="grasp 측면 오프셋 y[m] (실=-0.016)")
    ap.add_argument("--grip-open", dest="grip_open", type=float, default=47.0, help="open/hover gripper [0,100] (실=47)")
    ap.add_argument("--grip-close", dest="grip_close", type=float, default=5.0, help="close/grasp gripper [0,100] (실=5)")
    ap.add_argument("--loop", action="store_true", help="완료 후 재집기 반복(home 생략)")
    ap.add_argument("--no-tf", dest="no_tf", action="store_true")
    # 녹화: 폐루프 궤적(state=/isaac_joint_states, action=publish 한 command, images=3 캠)을
    # LeRobot v3 로 기록. 1 에피소드 완료(home 도달) 시 finalize+종료. --loop 는 무시(1회만).
    ap.add_argument("--record", action="store_true", help="pick-place 1 에피소드를 LeRobot v3 로 기록")
    ap.add_argument("--dataset-dir", dest="dataset_dir",
                    default="/workspace/datasets/pink_ik_pickplace",
                    help="LeRobot v3 출력 폴더(--record)")
    ap.add_argument("--task-desc", dest="task_desc",
                    default="pick up the cube and place it in the bowl",
                    help="에피소드 task 문자열(--record)")
    args = ap.parse_args()
    if args.self_check:
        return self_check(args.urdf)
    if args.sweep:
        return sweep(args)
    return run_ros(args)


if __name__ == "__main__":
    raise SystemExit(main())
