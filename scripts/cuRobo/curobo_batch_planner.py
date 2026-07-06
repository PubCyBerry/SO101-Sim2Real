"""cuRobo pick-place ZMQ **batch** planner 서비스 (SO-101, 신 API v0.8).

★2-프로세스: 이 planner(so101-curobo-datagen, cuRobo+warp) ↔ Isaac executor
(so101-isaac-sim) ZMQ 분리(in-process 불가=warp ABI). executor 가 큐브들(base_link)을
보내면 planner 가 **num_envs>=1 배치**로 collision-aware full pick-place 궤적 반환.

계획: tool frame = **tcp_grasp**(손가락 사이 pinch 점, sphere 기하 유도, so101.yml extra_link).
★5-DOF 원칙: orientation 고정 시 도달 위치는 2D 면 → goal 은 항상 **bank pose 그대로**
(FK-harvest (pos,quat) 쌍 = 존재증명) 또는 그 tool-z 평행이동(pan-plane 불변)만 쓴다.
4-phase: ① pre-grasp = ray-anchored goalset(grip-valid bank pose 중 하강선이 큐브 중심을
지나는 것, perp 순 정렬) ② 도착 pre pose FK 추출(goalset_index seed 불일치 회피)
③ linear_motion descend → 하강선 위 큐브 최근접점 + contact-link collision off ④ linear
retreat 역행. carry/place = 무필터 bank pose goalset(bowl obstacle 회피).
프레임=base_link→solver rotz(90).

collision 구성(cuRobo 정석 미러): target 큐브 = world obstacle("cube", per-request pose 주입,
pre/descend 중 팔 링크 관통 방지 — contact link 는 pre~lift collision off) → grasp 후
attachment_manager.attach 로 큐브 blob 을 attached_object(=tcp_grasp 동일 프레임)에 부착
+ world "cube" disable → lift/carry 는 잡은 큐브 부피 포함 계획 → carry 후 detach 복원.

프로토콜(JSON REQ/REP, port 5599):
  {"cmd":"ping"}                                              → {"ok":true,"n_envs":N}
  {"cmd":"plan_pickplace","cubes":[[x,y,z(,qw,qx,qy,qz)]×B],"bowl":[x,y]} → {"ok":true,
        (base_link frame, B ≤ n_envs; quat 주면 yaw face-align)      "trajectories":[[[6]×T]|null ×B]}
  {"cmd":"shutdown"}                                          → {"ok":true}

실행: /isaac-sim/python.sh curobo_batch_planner.py [--port 5599] [--n-envs 64] [--self-test]
"""
import argparse
import json
import math
import tempfile

import numpy as np
import torch
import yaml
import zmq
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.geom.sphere_fit import SphereFitType
from curobo._src.geom.types import Cuboid, SceneCfg
from curobo._src.motion.motion_planner_batch import BatchMotionPlanner
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.motion_planner import MotionPlannerCfg
from curobo.types import GoalToolPose, JointState, Pose

ROBOT = "/workspace/assets/robots/so101.yml"
BASE_YAW = 90.0
# ★base 프레임 완전 정합: bridge TF(base_link)=USD so101_new_calib 규약, cuRobo URDF=so_arm101 규약.
# 두 체인의 shoulder_pan 조인트 프레임 일치 조건에서 T(urdf←usd) = Rz(90°)+BASE_T (실측 유도).
# Rz(90°)=기존 BASE_YAW, BASE_T 는 미보정이던 평행이동(≈(16,-21,-32)mm) — 실행층 ~3cm 빗나감의 진범.
BASE_T = (0.01576, -0.02079, -0.03248)
GRIP_OPEN, GRIP_CLOSE = 75.0, 5.0  # open 75=straddle 마진 확대. close 5=강파지 — face-align 전제
# open 60→75(2026-07-06): 조준 완벽(xy 3mm)·fold 정상인데 무파지인 straddle-마진 실패 모드
# (ep15 실측 — 동일 arrival 에서 open75 만으로 PASS, knob2 스윕) 해소. 60 은 pad 간격이
# 큐브+5mm 수준이라 tangential ~3mm 오차로 moving jaw 가 큐브 모서리 위로 하강 → squirt.
# (yaw 미정렬 대각 접촉은 어떤 close 목표(5/23/26.5)로도 squirt-out: js trace 20°→pop 실측 3회.
#  → executor 가 cube quat 전달, 후보 pose 를 ρ=−Δψ/cosα 로 큐브 face 정렬(pink 검증 공식).)
CLOSE_STEPS = 38                   # close ramp 프레임(30fps≈1.27s) — open 75 확대분 보상해
                                   # 폐합 각속도 유지(급폐합=drop 레버 실측). 60→75 비례 30→38
TAU_MAX_DEG = 8.0  # |Δψ·tanα| 허용(deg) — tilted face-align 근사 손실 게이트(pink 규약)
TABLE_TOP = 0.035 + BASE_T[2]  # 책상 상판 z (urdf 프레임)
K = 40
PRE_BACK = 0.08          # pre-grasp = grasp 서 tool -z(approach 역방향) 8cm 후퇴
GRASP_Z_OFF = -0.008     # grasp 깊이 미세보정(m): pinch 가 큐브 상단걸침(wrist캠 실측) → 8mm 하향
LIFT_BACK = 0.10         # lift = grasp 서 tool -z 최대 10cm 역행(approach 되감기)
ALPHAS = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]  # goalset fallback 용 tilt ladder(deg)
# 결정론 우선 후보 (α, sx): 성공 실측 = (10, +1)(sx 역산: 성공 자세 x̂ azim ∈ φ+90±45 만 부합).
# goalset 임의 선택은 접촉 마진 비결정(동일 α 성패 반전) → 상위 후보를 K=1 순차 강제.
LADDER = [(10, 1), (20, 1), (-10, 1), (30, 1), (-20, 1), (40, 1)]
PAN_AXIS_XY = (0.0388353, 0.0)  # shoulder_pan 축의 base_link(=solver) XY 오프셋 — pan 평면 기준점(URDF)
TCP_LAT = (0.0, -0.015, 0.0)    # tcp_grasp 의 wrist_roll 축 이탈 lateral(so101.yml y 와 동기) — pan 고정점 보정용
CONTACT_LINKS = ["gripper_link", "moving_jaw_so101_v1_link"]  # descend/retreat 중 collision off
# descend/lift(linear) 단계 추가 off: grasp 자세에서 wrist sphere 가 큐브 obstacle 과
# 모델상 겹침(실측 gap wrist −0.6/−5.6mm·cam_mount +0.4mm, scratch/probe_grasp_spheres.py)
# — SO-101 은 wrist 가 짧아 grasp envelope 에 들어옴. 물리 접촉은 없음(sphere 보수 근사).
DESCEND_EXTRA_OFF = ["wrist_link", "wrist_cam_mount_link"]
CUBE_DIMS = 0.05  # 큐브 obstacle/attach blob 한 변(m) — 보수적 최대값(단일소스 cube_specs.py 40/50mm)
# so_arm101.urdf arm 관절 limit(rad) — start clamp 용(USD ±105° 와 어긋나는 5° 캘리브 마진)
ARM_LIMITS = [(-1.91986, 1.91986), (-1.74533, 1.74533), (-1.69, 1.69),
              (-1.65806, 1.65806), (-2.74385, 2.84121)]
# REST(접힘 휴식자세, bridge 초기자세 클램프판) — 에피소드 간 로봇 미리셋이라 start 가
# 직전 release 자세(bowl 쪽)로 오면 pre trajopt 실패 다발 → REST 경유 세그먼트 전치.
REST_Q = [-0.0001, -1.74033, 1.5647, 1.2108, -1.739]


def rotz(x, y, deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    return c * x - s * y, s * x + c * y


def usd_to_urdf(p):
    """bridge TF(base_link, USD 규약) 좌표 → cuRobo URDF(so_arm101) 좌표: Rz(90)+BASE_T."""
    x, y = rotz(p[0], p[1], BASE_YAW)
    return (x + BASE_T[0], y + BASE_T[1], p[2] + BASE_T[2])


def _mat2quat(R):
    """회전행렬(3x3, 열=[x̂,ŷ,ẑ]) → quat wxyz."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s; y = (R[0, 2] - R[2, 0]) / s; z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s; y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] >= R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s; y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s; y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float32)


class PickPlacePlanner:
    def __init__(self, n_envs=1, bowl_bl=(0.22, -0.265)):
        self.n_envs = n_envs
        bx, by, _ = usd_to_urdf((bowl_bl[0], bowl_bl[1], 0.0))
        self.bowl_s = (bx, by)
        rim_z = TABLE_TOP + 0.075
        world = {"cuboid": {"bowl": {"dims": [0.15, 0.15, 0.075],
                                     "pose": [bx, by, (TABLE_TOP + rim_z) / 2, 1, 0, 0, 0]},
                            # target 큐브 world obstacle — per-request update_obstacle_pose 로
                            # 실좌표 주입(placeholder 는 far). approach 중 팔 링크 큐브 관통 방지.
                            "cube": {"dims": [CUBE_DIMS] * 3,
                                     "pose": [9.0, 9.0, 0.02, 1, 0, 0, 0]}}}
        wf = tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False)
        yaml.safe_dump(world, wf); wf.close()
        # REST 복귀 단계용 월드 스왑: release 잔여 start 가 bowl 솔리드 cuboid 와 허위 충돌
        # (실제 그릇은 오목) → start-collision 거부. REST 세그먼트만 빈 월드로 계획.
        self.scene_full = SceneCfg.create(world)
        self.scene_empty = SceneCfg.create(
            {"cuboid": {"far_dummy": {"dims": [0.01, 0.01, 0.01], "pose": [9, 9, 9, 1, 0, 0, 0]}}})

        kin = Kinematics(KinematicsCfg.from_robot_yaml_file(ROBOT))
        jn = kin.joint_names
        qs = np.array([[p, l, e, w, r] for p in np.linspace(-0.9, 1.0, 11)
                       for l in np.linspace(-0.3, 1.4, 9) for e in np.linspace(-0.2, 1.6, 9)
                       for w in np.linspace(-0.6, 1.7, 9) for r in np.linspace(-1.5, 1.5, 5)], dtype=np.float32)
        st = kin.compute_kinematics(JointState.from_position(torch.tensor(qs, device="cuda"), joint_names=jn))
        # FK bank = carry 등 일반 reach goalset 용(bank pose 그대로 = 도달성 존재증명)
        self.FK_POS_ALL = st.tool_poses.get_link_pose(kin.tool_frames[0]).position.cpu().numpy()
        self.FK_QUAT_ALL = st.tool_poses.get_link_pose(kin.tool_frames[0]).quaternion.cpu().numpy()
        print(f"[planner] FK bank {len(self.FK_POS_ALL)}", flush=True)

        self.p = BatchMotionPlanner(MotionPlannerCfg.create(
            robot=ROBOT, scene_model=wf.name, max_batch_size=n_envs, max_goalset=K, use_cuda_graph=False))
        self.p.warmup(enable_graph=False, num_warmup_iterations=2)
        self.tf = self.p.tool_frames
        self.nA = len(self.p.joint_names)

    def _goalset(self, xyzs):
        """xyzs 최근접 **bank pose 그대로** K개 goalset — 전부 정확히 도달가능(존재증명).

        5-DOF 는 orientation 고정 시 도달 위치가 2D 면이라 (외래 위치, bank quat) 조합은
        tolerance 도박 → bank (pos,quat) 쌍을 그대로 goal 로 쓴다(carry 등 ±2cm 무관 reach)."""
        B = len(xyzs)
        pos = np.zeros((B, K, 3), dtype=np.float32)
        qs = np.zeros((B, K, 4), dtype=np.float32)
        for i, x in enumerate(xyzs):
            idx = np.argsort(np.linalg.norm(self.FK_POS_ALL - np.array(x), axis=1))[:K]
            pos[i], qs[i] = self.FK_POS_ALL[idx], self.FK_QUAT_ALL[idx]
        return GoalToolPose(
            tool_frames=self.tf,
            position=torch.tensor(pos, device="cuda").view(B, 1, 1, K, 3),
            quaternion=torch.tensor(qs, device="cuda").view(B, 1, 1, K, 4))

    def _pregrasp_goalset(self, xyzs, yaws=None, strict=True):
        """분석적 pan-plane pre-grasp goalset — 5-DOF 도달 보장 구성 + 큐브 face 정렬.

        TCP 는 wrist_roll 축 위(so101.yml tcp_grasp x=y=0)·approach축은 pan 수직평면 안
        → 후보 pose (cube−PRE_BACK·ẑ, R[x̂=±t̂, ŷ, ẑ]·Rz_tool(ρ)) 는 구성상 도달가능.
        ẑ = −cosα·ê_z + sinα·r̂ (α ladder), closing축 기저 x̂ = ±t̂ (수평).
        yaw 주어지면 ρ = −Δψ/cosα 로 pads↔큐브 수직면 정렬(pink 공식), 근사 손실
        |Δψ·tanα| > TAU_MAX_DEG 후보는 제외(단 전멸 방지 최소-Δψ 유지)."""
        B = len(xyzs)
        nc = len(ALPHAS) * 2
        pos = np.zeros((B, nc, 3), dtype=np.float32)
        qs = np.zeros((B, nc, 4), dtype=np.float32)
        tcp_lat = np.array(TCP_LAT)
        for i, xx in enumerate(xyzs):
            cx, cy, _ = xx
            yaw = None if yaws is None else yaws[i]
            cands = []  # (loss, pose, quat)
            for a_deg in ALPHAS:
                a = math.radians(a_deg)
                for sx in (1.0, -1.0):
                    # pan 고정점 반복: TCP 가 roll 축 밖(TCP_LAT)이라 ρ 에 따라 pan 평면
                    # 기준 방위가 이동 — φ ← az(cube − R·TCP_LAT − PAN_AXIS) 2-3회 수렴(pink §pan 보정).
                    phi = math.atan2(cy - PAN_AXIS_XY[1], cx - PAN_AXIS_XY[0])
                    R = None; loss = 0.0; z_ax = None
                    for _ in range(3):
                        r_hat = np.array([math.cos(phi), math.sin(phi), 0.0])
                        t_hat = np.array([-math.sin(phi), math.cos(phi), 0.0])
                        z_ax = -math.cos(a) * np.array([0.0, 0.0, 1.0]) + math.sin(a) * r_hat
                        x_ax = sx * t_hat
                        y_ax = np.cross(z_ax, x_ax)
                        R = np.stack([x_ax, y_ax, z_ax], 1)
                        loss = 0.0
                        if yaw is not None:
                            azim = phi + (0.0 if sx > 0 else math.pi) + math.pi / 2  # x̂0=±t̂ 방위
                            dpsi = (yaw - azim + math.pi / 4) % (math.pi / 2) - math.pi / 4
                            rho = -dpsi / max(math.cos(a), 0.3)
                            cr, sr = math.cos(rho), math.sin(rho)
                            R = R @ np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
                            loss = abs(math.degrees(dpsi * math.tan(a)))
                        ax_pt = np.array(xx) - R @ tcp_lat  # roll 축 점(고정점)
                        phi = math.atan2(ax_pt[1] - PAN_AXIS_XY[1], ax_pt[0] - PAN_AXIS_XY[0])
                    cands.append((loss, np.array(xx) - PRE_BACK * z_ax, _mat2quat(R)))
            if strict:
                ok = [c for c in cands if c[0] <= TAU_MAX_DEG]
                if not ok:  # 전멸 방지: 손실 최소 4개
                    ok = sorted(cands, key=lambda c: c[0])[:4]
            else:  # relaxed: 게이트 없이 전 후보(ρ best-effort 정렬 유지) — coverage 우선
                ok = cands
            base = list(ok)
            while len(ok) < nc:  # goalset 크기 고정(패딩=사이클 반복)
                ok.append(base[len(ok) % len(base)])
            for k in range(nc):
                pos[i, k], qs[i, k] = ok[k][1], ok[k][2]
        return GoalToolPose(
            tool_frames=self.tf,
            position=torch.tensor(pos, device="cuda").view(B, 1, 1, nc, 3),
            quaternion=torch.tensor(qs, device="cuda").view(B, 1, 1, nc, 4))

    def _finish(self, r, B, start):
        """plan_pose 결과 → (list[B] traj(rad)|None, end JointState(B,dof))."""
        if r is None:
            return [None] * B, start
        succ = r.success.view(-1).cpu().numpy()
        pos = r.interpolated_trajectory.position.detach()  # (B,1,T,dof) batch-safe
        while pos.dim() > 3:
            pos = pos[:, 0]
        if pos.dim() == 2:
            pos = pos.unsqueeze(0)
        last = r.interpolated_last_tstep  # (B,1) per-env 유효 길이
        trajs, ends = [], []
        for i in range(B):
            ti = int(last[i].reshape(-1)[0].item()) if last is not None else pos.shape[1]
            qi = pos[i, :max(ti, 2), :self.nA]
            trajs.append(qi.cpu().numpy() if succ[i] else None)
            ends.append(qi[-1])
        end = JointState.from_position(torch.stack(ends), joint_names=self.p.joint_names)
        return trajs, end

    def _plan_to(self, goal, start, linear=False, attempts=3):
        """GoalToolPose 직접 계획. linear=True 면 tool-z 직선 corridor + contact link collision off.

        attempts: 파티클 IK 확률성으로 동일 문제도 호출마다 성패 요동(eval pre-fail 재현 불가
        관측) — 재시도로 안정화. 성공 시 조기 종료라 추가 비용은 실패 케이스만."""
        B = goal.position.shape[0]
        if linear:
            self.p.disable_link_collision(CONTACT_LINKS + DESCEND_EXTRA_OFF)
            lin = ToolPoseCriteria.linear_motion(axis="z", non_terminal_scale=1.0,
                                                 project_distance_to_goal=True)
            self.p.update_tool_pose_criteria({k: lin for k in self.tf})
        r = self.p.plan_pose(goal_tool_poses=goal, current_state=start, max_attempts=attempts)
        if linear:
            self.p.update_tool_pose_criteria({k: ToolPoseCriteria() for k in self.tf})
            self.p.enable_link_collision(CONTACT_LINKS + DESCEND_EXTRA_OFF)
        return self._finish(r, B, start)

    def _cand_pose(self, xx, yaw, a_deg, sx):
        """(α, sx) 단일 후보 pose — pan 고정점 반복 + face-align ρ."""
        cx, cy, _ = xx
        a = math.radians(a_deg)
        phi = math.atan2(cy - PAN_AXIS_XY[1], cx - PAN_AXIS_XY[0])
        tcp_lat = np.array(TCP_LAT)
        R = None; z_ax = None
        for _ in range(3):
            r_hat = np.array([math.cos(phi), math.sin(phi), 0.0])
            t_hat = np.array([-math.sin(phi), math.cos(phi), 0.0])
            z_ax = -math.cos(a) * np.array([0.0, 0.0, 1.0]) + math.sin(a) * r_hat
            x_ax = sx * t_hat
            y_ax = np.cross(z_ax, x_ax)
            R = np.stack([x_ax, y_ax, z_ax], 1)
            if yaw is not None:
                azim = phi + (0.0 if sx > 0 else math.pi) + math.pi / 2
                dpsi = (yaw - azim + math.pi / 4) % (math.pi / 2) - math.pi / 4
                rho = -dpsi / max(math.cos(a), 0.3)
                cr, sr = math.cos(rho), math.sin(rho)
                R = R @ np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
            ax_pt = np.array(xx) - R @ tcp_lat
            phi = math.atan2(ax_pt[1] - PAN_AXIS_XY[1], ax_pt[0] - PAN_AXIS_XY[0])
        return np.array(xx) - PRE_BACK * z_ax, _mat2quat(R)

    def _pre_ladder(self, cs, yaws, start, ladder=None):
        """결정론 후보 ladder: 검증 후보(α=10, sx=+1)부터 순차 K=1 계획, 실패 env 만 다음.

        cuRobo goalset 은 임의 도달해 선택 → 접촉 마진이 후보마다 달라 grip 재현 안 됨
        (동일 α 성패 반전 관측) → 후보 선택을 planner 가 결정론으로 통제.
        ladder: knob 스윕용 후보열 오버라이드(기본 LADDER)."""
        B = len(cs)
        pre = [None] * B
        ends = [start.position[i] for i in range(B)]
        remaining = set(range(B))
        for a_deg, sx in (ladder or LADDER):
            if not remaining:
                break
            pos = np.zeros((B, 1, 3), dtype=np.float32)
            qs = np.zeros((B, 1, 4), dtype=np.float32)
            for i in range(B):
                yaw = None if yaws is None else yaws[i]
                # sx 미러: 성공 실측 sx=+1 은 좌측(φ<0) 기준 — 우측은 거울상으로 jaw
                # 돌출 방향이 반전(테이블 타격 스턱) → φ 부호로 미러링해 항상 같은 물리 방향
                phi_i = math.atan2(cs[i][1] - PAN_AXIS_XY[1], cs[i][0] - PAN_AXIS_XY[0])
                sx_eff = sx if phi_i < 0 else -sx
                pos[i, 0], qs[i, 0] = self._cand_pose(tuple(cs[i]), yaw, a_deg, sx_eff)
            goal = GoalToolPose(
                tool_frames=self.tf,
                position=torch.tensor(pos, device="cuda").view(B, 1, 1, 1, 3),
                quaternion=torch.tensor(qs, device="cuda").view(B, 1, 1, 1, 4))
            trajs, q_end = self._plan_to(goal, start, attempts=2)
            solved = []
            for i in list(remaining):
                if trajs[i] is not None:
                    pre[i] = trajs[i]; ends[i] = q_end.position[i]; remaining.discard(i)
                    solved.append(i)
            try:
                with open("/workspace/autoresearch/planner_diag.log", "a") as fd:
                    fd.write(f"[ladder] cand=({a_deg},{sx}) solved={solved} remaining={sorted(remaining)}\n")
            except OSError:
                pass
        return pre, remaining, ends

    def _plan_batch(self, xyzs, start):  # xyz 목표(quat 은 FK bank 서 선택) 편의 래퍼
        return self._plan_to(self._goalset(xyzs), start)

    @staticmethod
    def _diag(msg):
        try:
            with open("/workspace/autoresearch/planner_diag.log", "a") as fd:
                fd.write(msg + "\n")
        except OSError:
            pass

    def _attachment_manager(self):
        """AttachmentManager 핸들 — 설치본은 TrajOptSolver 가 컴포지션(.core)이라 위임
        property 부재(ref_repos 신버전은 노출). 양쪽 호환 접근."""
        ts = self.p.trajopt_solver
        am = getattr(ts, "attachment_manager", None)
        return am if am is not None else ts.core.attachment_manager

    def _attach_cube(self, q_state):
        """잡은 큐브 blob 을 attached_object 링크에 attach(+world "cube" obstacle disable).

        world_objects_pose_offset=None(identity) — attached_object 가 tcp_grasp 와 동일
        transform 이라 blob 이 정확히 pinch 점(=물리 큐브 위치)에 놓임(FK offset 산술 불요,
        batch-safe). carry 가 잡은 큐브 부피를 collision 으로 고려. 실패 시 False(무부착 진행).
        """
        B = q_state.position.shape[0] if q_state.position.dim() > 1 else 1
        try:
            am = self._attachment_manager()
            n_env = am.kinematics_params.link_spheres.shape[0]
            q_att = q_state
            if B > n_env:  # 공유 link_spheres(kinematics 1-env): env0 대표 — 실행경로는 B=1
                q_att = JointState.from_position(q_state.position[0:1], joint_names=self.p.joint_names)
            am.attach(
                q_att,
                [Cuboid(name="attached_cube", pose=[0, 0, 0, 1, 0, 0, 0], dims=[CUBE_DIMS] * 3)],
                link_name="attached_object", num_spheres=10,
                sphere_fit_type=SphereFitType.VOXEL,
                world_objects_pose_offset=None,
                disable_obstacle_names=["cube"])
            self._diag(f"[attach] ok B={B} n_env={n_env} (identity@tcp_grasp)")
            return True
        except Exception as e:
            self._diag(f"[attach] FAIL {type(e).__name__}: {e} — 무부착 fallback")
            return False

    def plan_pickplace_batch(self, cubes_bl, bowl_bl=None, start_rad=None, knobs=None):
        # knobs: 물리 phase 스윕용 per-request 오버라이드(스윕 계측, 기본=상수 → 무변경).
        #   {"grasp_z_off": f, "grip_open": f, "grip_close": f, "ladder": [[α,sx],..]}
        kn = knobs or {}
        z_off = float(kn.get("grasp_z_off", GRASP_Z_OFF))
        g_open = float(kn.get("grip_open", GRIP_OPEN))
        g_close = float(kn.get("grip_close", GRIP_CLOSE))
        ladder = [tuple(c) for c in kn.get("ladder", [])] or None
        # 파티클 IK 는 process 누적 seed 상태 → 동일 입력도 회차별 성패 요동(planFAIL 9~13 널뜀).
        # 요청마다 seed 리셋 = 에피소드 결정론(재현성) 복원.
        self.p.reset_seed()
        B = len(cubes_bl)
        cs = [usd_to_urdf(c[:3]) for c in cubes_bl]
        cs = [(x, y, z + z_off) for x, y, z in cs]
        # 6D payload([x,y,z,qw,qx,qy,qz])면 큐브 yaw 추출(z-yaw) → urdf 프레임(+90°)
        yaws = []
        for c in cubes_bl:
            if len(c) >= 7:
                w, x, y, z = c[3], c[4], c[5], c[6]
                yaw_usd = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                yaws.append(yaw_usd + math.radians(BASE_YAW))
            else:
                yaws.append(None)
        if all(v is None for v in yaws):
            yaws = None
        if bowl_bl is not None:  # per-episode 실제 bowl(TF) — carry 목표 정확화
            bx, by, _ = usd_to_urdf((bowl_bl[0], bowl_bl[1], 0.0))
        else:
            bx, by = self.bowl_s
        if start_rad is not None:  # 현재(초기) 자세부터 planning — joint frame 은 base 회전 불변
            arm = torch.tensor([s[:self.nA] for s in start_rad], device="cuda", dtype=torch.float32)
            # USD limit(±105°) > URDF limit(±100°): bridge 초기자세가 URDF 밖일 수 있어 clamp
            # (예: lift -1.7525 vs -1.74533 — 0.4° 스냅은 executor ramp 가 흡수)
            lim = torch.tensor(ARM_LIMITS, device="cuda", dtype=torch.float32)
            arm = torch.clamp(arm, lim[:, 0] + 0.005, lim[:, 1] - 0.005)
            start = JointState.from_position(arm, joint_names=self.p.joint_names)
        else:
            start = JointState.from_position(
                self.p.default_joint_state.position.unsqueeze(0).repeat(B, 1), joint_names=self.p.joint_names)
        # ⓪ start 가 REST 서 멀면(직전 에피소드 잔여 자세) collision-aware 로 REST 복귀 —
        #    bowl-side start 발 pre trajopt 실패(eval planFAIL 8/20 재현적) 차단.
        rest_seg = [None] * B
        rest_t = torch.tensor(REST_Q, device="cuda", dtype=torch.float32)
        far = (start.position - rest_t.unsqueeze(0)).abs().sum(1) > 0.3
        if bool(far.any()):
            goal_states = JointState.from_position(
                rest_t.unsqueeze(0).repeat(B, 1), joint_names=self.p.joint_names)
            self.p.update_world(self.scene_empty)   # bowl 허위 start-collision 회피
            rc = self.p.plan_cspace(goal_states, start)
            self.p.update_world(self.scene_full)
            rtraj, r_end = self._finish(rc, B, start)
            new_pos = []
            for i in range(B):
                if bool(far[i]) and rtraj[i] is not None:
                    rest_seg[i] = rtraj[i]
                    new_pos.append(r_end.position[i])
                else:
                    new_pos.append(start.position[i])
            start = JointState.from_position(torch.stack(list(new_pos)), joint_names=self.p.joint_names)
        # ⓪b target 큐브 world obstacle 배치(공유 월드 env0) — pre/descend 중 팔 링크의
        #    큐브 관통 방지. 공유 collision world 라 B>1 상이 큐브는 env0 것만 반영(실행경로 B=1).
        #    pose=aim점(cs, z_off 포함 ≈ 실제 중심 −8mm) — 회피 용도라 충분. REST 월드스왑
        #    (update_world 전체 리로드=placeholder 복원) 뒤에 주입해야 유효.
        try:
            cpose = Pose(
                position=torch.tensor([list(cs[0])], device="cuda", dtype=torch.float32),
                quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32))
            self.p.scene_collision_checker.update_obstacle_pose("cube", cpose, env_idx=0)
            self.p.scene_collision_checker.enable_obstacle("cube", True, env_idx=0)
        except Exception as e:
            self._diag(f"[cube-obst] FAIL {type(e).__name__}: {e}")
        # ① pre-grasp 3단: 결정론 LADDER(K=1 순차, 접촉 마진 재현성) → strict goalset →
        #    relaxed goalset (coverage). relaxed_used 는 진단용.
        #    contact link(jaw·gripper)는 pre 단계도 collision off — cuRobo plan_grasp 정석
        #    (goalset 단계 disable)과 동일. PRE_BACK 8cm 서 jaw tip 이 aim 점 1.3cm 앞
        #    = 큐브 obstacle 내부라 off 없이는 pre 전멸.
        self.p.disable_link_collision(CONTACT_LINKS)
        try:
            pre, remaining, ends = self._pre_ladder(cs, yaws, start, ladder=ladder)
            relaxed_used = [False] * B
            if remaining:
                preS, qS = self._plan_to(self._pregrasp_goalset([tuple(c) for c in cs], yaws), start)
                for i in list(remaining):
                    if preS[i] is not None:
                        pre[i] = preS[i]; ends[i] = qS.position[i]; remaining.discard(i)
            if remaining:
                preR, qR = self._plan_to(
                    self._pregrasp_goalset([tuple(c) for c in cs], yaws, strict=False), start)
                for i in list(remaining):
                    if preR[i] is not None:
                        pre[i] = preR[i]; ends[i] = qR.position[i]; relaxed_used[i] = True
                        remaining.discard(i)
        finally:
            self.p.enable_link_collision(CONTACT_LINKS)
        q_pre = JointState.from_position(torch.stack([torch.as_tensor(e, device="cuda") for e in ends]),
                                         joint_names=self.p.joint_names)
        # ② 도착 pre pose FK → achieved (pos,quat,ẑ) (goalset_index seed 불일치 회피)
        kk = self.p.compute_kinematics(q_pre)
        ap = kk.tool_poses.get_link_pose(self.tf[0])
        aq = ap.quaternion.detach().view(B, 4)
        app = ap.position.detach().view(B, 3)
        w, x, y, z = aq[:, 0], aq[:, 1], aq[:, 2], aq[:, 3]
        zax = torch.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], 1)
        cube_t = torch.tensor(cs, device="cuda", dtype=torch.float32)
        # 달성 자세 전수 진단(grip 실패 패턴 채굴용): α(접근 기울기)·Δψ(face 정렬오차)·relaxed
        try:
            xcol = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], 1)
            with open("/workspace/autoresearch/planner_diag.log", "a") as fd:
                for i in range(B):
                    al = math.degrees(math.acos(max(-1.0, min(1.0, -float(zax[i, 2])))))
                    azim = math.atan2(float(xcol[i, 1]), float(xcol[i, 0]))
                    dps = "-"
                    if yaws is not None and yaws[i] is not None:
                        dps = round(math.degrees((yaws[i] - azim + math.pi / 4) % (math.pi / 2) - math.pi / 4), 1)
                    fd.write(f"[graspdiag] cube=({cubes_bl[i][0]:.3f},{cubes_bl[i][1]:.3f}) "
                             f"alpha={al:.1f} dpsi={dps} relaxed={relaxed_used[i]} pre_ok={pre[i] is not None}\n")
        except OSError:
            pass
        # ③ linear descend → 하강선(ray) 위 큐브 최근접점: tool-z 평행이동이라 항상 도달가능
        tstar = ((cube_t - app) * zax).sum(1, keepdim=True)
        gpos = app + tstar * zax
        down = GoalToolPose(tool_frames=self.tf,
                            position=gpos.view(B, 1, 1, 1, 3).contiguous(),
                            quaternion=aq.view(B, 1, 1, 1, 4).contiguous())
        desc, q_grasp = self._plan_to(down, q_pre, linear=True)
        # ④ linear retreat: approach 역행(잡은 채 이탈) — pre 지점(검증됨) 너머로 안 나감
        up_pos = gpos - torch.clamp(tstar, max=LIFT_BACK) * zax
        up = GoalToolPose(tool_frames=self.tf,
                          position=up_pos.view(B, 1, 1, 1, 3).contiguous(),
                          quaternion=aq.view(B, 1, 1, 1, 4).contiguous())
        lift, q_lift = self._plan_to(up, q_grasp, linear=True)
        # ★attach(cuRobo 예제 정석: lift 후·carry 전) — 잡은 큐브 blob + world "cube" disable.
        #    grasp 직후 attach 는 blob 시작상태가 bowl cuboid 마진과 겹쳐(근접 배치 실측
        #    sep 0.093<0.075+0.025) lift 가 허위 planFAIL → lift 는 무blob(순수 수직 역행이라
        #    큐브 충돌 가치 없음), carry 만 blob 포함 계획. carry 후 detach 복원.
        attached = self._attach_cube(q_lift)
        try:
            # ⑤ carry: 그릇 상공 (bowl obstacle 회피 + attached 큐브 blob) — z 는 urdf 프레임(rim+9cm)
            carry, _ = self._plan_batch([(bx, by, 0.20 + BASE_T[2])] * B, q_lift)
        finally:
            if attached:
                try:
                    self._attachment_manager().detach()
                except Exception as e:
                    self._diag(f"[attach] detach FAIL {type(e).__name__}: {e}")
        out = []
        for i in range(B):
            if pre[i] is None or desc[i] is None or lift[i] is None or carry[i] is None:
                msg = (f"[planner] env{i} phase-fail pre={pre[i] is not None} desc={desc[i] is not None} "
                       f"lift={lift[i] is not None} carry={carry[i] is not None} cube={cubes_bl[i]}")
                print(msg, flush=True)
                try:  # 컨테이너 소멸 후에도 남는 진단(호스트 마운트)
                    with open("/workspace/autoresearch/planner_diag.log", "a") as fdiag:
                        fdiag.write(msg + "\n")
                except OSError:
                    pass
                out.append(None); continue
            a, de, li, ca = (np.rad2deg(t) for t in (pre[i], desc[i], lift[i], carry[i]))
            hold = np.repeat(de[-1:], CLOSE_STEPS, 0)
            seq = []
            if rest_seg[i] is not None:  # REST 복귀 세그먼트(있으면 선행, gripper OPEN)
                rr = np.rad2deg(rest_seg[i])
                seq.append(np.hstack([rr, np.full((len(rr), 1), g_open)]))
            seq += [np.hstack([a, np.full((len(a), 1), g_open)]),
                   np.hstack([de, np.full((len(de), 1), g_open)]),
                   np.hstack([hold, np.linspace(g_open, g_close, CLOSE_STEPS).reshape(-1, 1)]),
                   np.hstack([li, np.full((len(li), 1), g_close)]),
                   np.hstack([ca, np.full((len(ca), 1), g_close)])]
            full = np.vstack(seq)
            rel = np.repeat(full[-1:], 12, 0); rel[:, 5] = g_open
            out.append(np.vstack([full, rel]).astype(np.float32))
        return out


def serve_loop(pl, sock):
    while True:
        req = json.loads(sock.recv())
        cmd = req.get("cmd")
        if cmd == "ping":
            sock.send_string(json.dumps({"ok": True, "n_envs": pl.n_envs}))
        elif cmd == "plan_pickplace":
            trajs = pl.plan_pickplace_batch(req["cubes"], req.get("bowl"), req.get("start"),
                                            req.get("knobs"))
            sock.send_string(json.dumps({"ok": True,
                                         "trajectories": [t.tolist() if t is not None else None for t in trajs]}))
        elif cmd == "shutdown":
            sock.send_string(json.dumps({"ok": True})); return
        else:
            sock.send_string(json.dumps({"ok": False, "err": f"unknown {cmd!r}"}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5599)
    ap.add_argument("--n-envs", dest="n_envs", type=int, default=1)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    pl = PickPlacePlanner(n_envs=a.n_envs)
    print(f"[planner] ready n_envs={pl.n_envs}", flush=True)
    if a.self_test:
        cubes = [[0.017, -0.253, 0.06]] * a.n_envs  # 배치 검증
        trajs = pl.plan_pickplace_batch(cubes)
        n_ok = sum(t is not None for t in trajs)
        print(f"self-test batch n_envs={a.n_envs}: {n_ok}/{a.n_envs} plans, shapes={[t.shape if t is not None else None for t in trajs]}")
        print("SELFTEST_OK" if n_ok == a.n_envs else "SELFTEST_CHECK")
        return
    sock = zmq.Context().socket(zmq.REP); sock.bind(f"tcp://*:{a.port}")
    print(f"[planner] ZMQ REP :{a.port}", flush=True)
    serve_loop(pl, sock)


if __name__ == "__main__":
    main()
