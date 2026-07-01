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


def run_ros(args):
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
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
            self.cube_xyz = None
            self.log_ct = 0

            self.sub = self.create_subscription(JointState, "/isaac_joint_states", self._on_state, 10)
            self.pub = self.create_publisher(JointState, "/isaac_joint_commands", 10)
            self.tf_buf = None
            if tf2_ros is not None and not args.no_tf:
                self.tf_buf = tf2_ros.Buffer()
                self.tf_listener = tf2_ros.TransformListener(self.tf_buf, self)
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
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-check", action="store_true")
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
    args = ap.parse_args()
    return self_check(args.urdf) if args.self_check else run_ros(args)


if __name__ == "__main__":
    raise SystemExit(main())
