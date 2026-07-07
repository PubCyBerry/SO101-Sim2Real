"""cuRobo pick-place ZMQ planner 서비스 (SO-101, 신 API v0.8).

★2-프로세스: 이 planner(so101-curobo-datagen, cuRobo+warp) ↔ Isaac executor
(so101-isaac-sim) ZMQ 분리(in-process 불가=warp ABI). executor 가 큐브(base_link)를 보내면
planner 가 collision-aware full pick-place 궤적을 반환한다.

**현재 num_envs=1(단일 env)만 처리한다.** 배치(multi-env)는 추후 확장 — `n_envs` 파라미터와
BatchMotionPlanner 백엔드는 그 hook 으로 남겨두되, 계획 로직은 큐브 1개 기준으로 읽기 쉽게 쓴다.

계획: tool frame = **tcp_grasp**(손가락 사이 pinch 점, so101.yml extra_link).
★5-DOF 원칙: orientation 고정 시 도달 위치는 2D 면 → goal 은 항상 **bank pose 그대로**
(FK-harvest (pos,quat) 쌍 = 존재증명) 또는 그 tool-z 평행이동(pan-plane 불변)만 쓴다.

6-phase(approach·grasp·lift·transit·place·retreat):
  ① approach = pre-grasp goalset(pan-plane bank pose, 하강선이 큐브 중심 통과, contact off)
  ② grasp    = 도착 pre pose FK → linear descend(하강선 위 큐브 최근접점) + gripper 폐합 ramp
  ③ lift     = grasp 서 tool -z linear 역행(잡은 채 수직 이탈), attach 큐브 blob
  ④ transit  = 그릇 상공(TRANSIT_Z) 이동, bank goalset(bowl obstacle 회피 + blob)
  ⑤ place    = 그릇 중심서 수직 linear 하강(PLACE_DROP, bowl obstacle 잠깐 disable) + gripper 개방
  ⑥ retreat  = init(home=start) 자세로 cspace 복귀(empty-world, gripper open)
프레임=base_link→solver rotz(90).

collision 구성(cuRobo 정석 미러): target 큐브 = world obstacle("cube", per-request pose 주입,
pre/descend 중 팔 링크 관통 방지 — contact link 는 pre~lift collision off) → grasp 후
attachment_manager.attach 로 큐브 blob 을 attached_object(=tcp_grasp 동일 프레임)에 부착
+ world "cube" disable → transit/place 는 잡은 큐브 부피 포함 계획 → place 후 detach 복원.

프로토콜(JSON REQ/REP, port 5599):
  {"cmd":"ping"}                                                    → {"ok":true,"n_envs":N}
  {"cmd":"plan_pickplace","cubes":[[x,y,z(,qw,qx,qy,qz)]],"bowl":[x,y]} → {"ok":true,
        (base_link frame, cubes[0] 만 사용; quat 주면 yaw face-align)      "trajectories":[[[6]×T]|null]}
  {"cmd":"shutdown"}                                                → {"ok":true}

실행: /isaac-sim/python.sh curobo_batch_planner.py [--port 5599] [--self-test]
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
# Rz(90°)=BASE_YAW, BASE_T=미보정이던 평행이동(≈(16,-21,-32)mm) — 실행층 ~3cm 빗나감의 진범.
BASE_T = (0.01576, -0.02079, -0.03248)
GRIP_OPEN, GRIP_CLOSE = 75.0, 5.0  # open 75=straddle 마진 확대. close 5=강파지 — face-align 전제
# open 60→75(2026-07-06): 조준 완벽(xy 3mm)·fold 정상인데 무파지인 straddle-마진 실패 모드 해소.
# 60 은 pad 간격이 큐브+5mm 수준이라 tangential ~3mm 오차로 moving jaw 가 모서리 위로 하강 → squirt.
# (yaw 미정렬 대각 접촉은 어떤 close 목표로도 squirt-out → 후보 pose 를 ρ=−Δψ/cosα 로 face 정렬.)
CLOSE_STEPS = 38   # close ramp 프레임 — open 75 확대분 보상해 폐합 각속도 유지(급폐합=drop 레버)
OPEN_STEPS = 20    # ⑤ release: gripper 개방 ramp 프레임(정지 상태 투하)
TAU_MAX_DEG = 8.0  # |Δψ·tanα| 허용(deg) — tilted face-align 근사 손실 게이트(pink 규약)
TABLE_TOP = 0.035 + BASE_T[2]  # 책상 상판 z (urdf 프레임)
K = 40             # goalset 크기(bank reach·pregrasp)
PRE_BACK = 0.08    # pre-grasp = grasp 서 tool -z(approach 역방향) 8cm 후퇴
GRASP_Z_OFF = -0.008  # grasp 깊이 미세보정(m): pinch 가 큐브 상단걸침 → 8mm 하향
LIFT_BACK = 0.10   # lift = grasp 서 tool -z 최대 10cm 역행(approach 되감기)
TRANSIT_Z = 0.20   # ④ transit: 그릇 상공 안전고도(urdf, +BASE_T[2] 는 사용처)
PLACE_DROP = 0.08  # ⑤ place: transit 상공서 그릇 안으로 수직 하강량(m) — knob
ALPHAS = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]  # goalset fallback tilt ladder(deg)
# 결정론 우선 후보 (α, sx): 성공 실측 = (10, +1). goalset 임의 선택은 접촉 마진 비결정(동일 α
# 성패 반전) → 상위 후보를 K=1 순차 강제.
LADDER = [(10, 1), (20, 1), (-10, 1), (30, 1), (-20, 1), (40, 1)]
PAN_AXIS_XY = (0.0388353, 0.0)  # shoulder_pan 축의 base_link(=solver) XY 오프셋 — pan 평면 기준점(URDF)
TCP_LAT = (0.0, -0.015, 0.0)    # tcp_grasp 의 wrist_roll 축 이탈 lateral — pan 고정점 보정용
CONTACT_LINKS = ["gripper_link", "moving_jaw_so101_v1_link"]  # descend/retreat 중 collision off
# descend/lift(linear) 추가 off: grasp 자세서 wrist sphere 가 큐브 obstacle 과 모델상 겹침
# (SO-101 은 wrist 가 짧아 grasp envelope 에 들어옴; 물리 접촉 없음=sphere 보수 근사).
DESCEND_EXTRA_OFF = ["wrist_link", "wrist_cam_mount_link"]
CUBE_DIMS = 0.05   # 큐브 obstacle/attach blob 한 변(m) — 보수적 최대값(cube_specs 40/50mm)
# so_arm101.urdf arm 관절 limit(rad) — start clamp 용(USD ±105° 와 어긋나는 5° 캘리브 마진)
ARM_LIMITS = [(-1.91986, 1.91986), (-1.74533, 1.74533), (-1.69, 1.69),
              (-1.65806, 1.65806), (-2.74385, 2.84121)]
DIAG_LOG = "/workspace/autoresearch/planner_diag.log"  # 컨테이너 소멸 후에도 남는 진단(호스트 마운트)


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


def _grip(arm_deg, grip):
    """arm-deg 궤적에 gripper 열(scalar=상수 · array=per-step ramp)을 붙여 (T,6) 로."""
    g = np.full((len(arm_deg), 1), grip) if np.isscalar(grip) else np.asarray(grip).reshape(-1, 1)
    return np.hstack([arm_deg, g])


class PickPlacePlanner:
    """단일 env pick-place planner. plan_pickplace(cube) 가 6-phase 궤적을 반환한다."""

    def __init__(self, n_envs=1, bowl_bl=(0.22, -0.265)):
        self.n_envs = n_envs  # 배치 확장 hook (현재 로직은 단일 env)
        bx, by, _ = usd_to_urdf((bowl_bl[0], bowl_bl[1], 0.0))
        self.bowl_s = (bx, by)
        rim_z = TABLE_TOP + 0.075
        world = {"cuboid": {
            "bowl": {"dims": [0.15, 0.15, 0.075],
                     "pose": [bx, by, (TABLE_TOP + rim_z) / 2, 1, 0, 0, 0]},
            # target 큐브 world obstacle — per-request update_obstacle_pose 로 실좌표 주입
            # (placeholder 는 far). approach 중 팔 링크의 큐브 관통 방지.
            "cube": {"dims": [CUBE_DIMS] * 3, "pose": [9.0, 9.0, 0.02, 1, 0, 0, 0]}}}
        wf = tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False)
        yaml.safe_dump(world, wf); wf.close()
        # ⑥ retreat 용 월드 스왑: release 잔여 start 가 bowl 솔리드 cuboid 와 허위 충돌
        # (실제 그릇은 오목) → start-collision 거부. retreat 세그먼트만 빈 월드로 계획.
        self.scene_full = SceneCfg.create(world)
        self.scene_empty = SceneCfg.create(
            {"cuboid": {"far_dummy": {"dims": [0.01, 0.01, 0.01], "pose": [9, 9, 9, 1, 0, 0, 0]}}})

        # FK bank = transit 등 일반 reach goalset 용(bank pose 그대로 = 도달성 존재증명)
        kin = Kinematics(KinematicsCfg.from_robot_yaml_file(ROBOT))
        qs = np.array([[p, l, e, w, r] for p in np.linspace(-0.9, 1.0, 11)
                       for l in np.linspace(-0.3, 1.4, 9) for e in np.linspace(-0.2, 1.6, 9)
                       for w in np.linspace(-0.6, 1.7, 9) for r in np.linspace(-1.5, 1.5, 5)],
                      dtype=np.float32)
        st = kin.compute_kinematics(
            JointState.from_position(torch.tensor(qs, device="cuda"), joint_names=kin.joint_names))
        tp = st.tool_poses.get_link_pose(kin.tool_frames[0])
        self.FK_POS_ALL = tp.position.cpu().numpy()
        self.FK_QUAT_ALL = tp.quaternion.cpu().numpy()
        print(f"[planner] FK bank {len(self.FK_POS_ALL)}", flush=True)

        self.p = BatchMotionPlanner(MotionPlannerCfg.create(
            robot=ROBOT, scene_model=wf.name, max_batch_size=n_envs, max_goalset=K,
            use_cuda_graph=False))
        self.p.warmup(enable_graph=False, num_warmup_iterations=2)
        self.tf = self.p.tool_frames
        self.nA = len(self.p.joint_names)

    # ── small helpers ───────────────────────────────────────────────────────────
    @staticmethod
    def _diag(msg):
        try:
            with open(DIAG_LOG, "a") as fd:
                fd.write(msg + "\n")
        except OSError:
            pass

    def _goal(self, pos, quat):
        """(G,3)/(G,4) → 단일 env GoalToolPose (goalset G; 단일 goal 은 G=1)."""
        p = torch.as_tensor(np.asarray(pos), device="cuda", dtype=torch.float32).view(1, 1, 1, -1, 3)
        q = torch.as_tensor(np.asarray(quat), device="cuda", dtype=torch.float32).view(1, 1, 1, -1, 4)
        return GoalToolPose(tool_frames=self.tf, position=p, quaternion=q)

    def _goalset(self, xyz):
        """xyz 최근접 K개 **bank pose 그대로** goalset(전부 정확히 도달가능=존재증명). 5-DOF 는
        orientation 고정 시 도달 위치가 2D 면이라 (외래 위치, bank quat) 조합은 tolerance 도박 →
        bank (pos,quat) 쌍을 그대로 goal 로 쓴다(transit 등 ±2cm 무관 reach)."""
        idx = np.argsort(np.linalg.norm(self.FK_POS_ALL - np.array(xyz), axis=1))[:K]
        return self._goal(self.FK_POS_ALL[idx], self.FK_QUAT_ALL[idx])

    def _extract(self, r, start):
        """plan_pose 결과 → (traj(T,nA) np|None, end JointState(1,dof)). 실패/None 이면 traj=None."""
        if r is None:
            return None, start
        pos = r.interpolated_trajectory.position.detach()
        while pos.dim() > 2:          # (B,1,T,dof) → (T,dof)
            pos = pos[0]
        last = r.interpolated_last_tstep
        ti = int(last.reshape(-1)[0].item()) if last is not None else pos.shape[0]
        q = pos[: max(ti, 2), : self.nA]
        ok = bool(r.success.view(-1)[0].item())
        end = JointState.from_position(q[-1:].clone(), joint_names=self.p.joint_names)
        return (q.cpu().numpy() if ok else None), end

    def _plan_to(self, goal, start, linear=False, attempts=3):
        """GoalToolPose 직접 계획 → (traj|None, end). linear=True 면 tool-z 직선 corridor +
        contact link collision off. attempts: 파티클 IK 확률성 안정화(성공 시 조기 종료)."""
        if linear:
            self.p.disable_link_collision(CONTACT_LINKS + DESCEND_EXTRA_OFF)
            lin = ToolPoseCriteria.linear_motion(axis="z", non_terminal_scale=1.0,
                                                 project_distance_to_goal=True)
            self.p.update_tool_pose_criteria({k: lin for k in self.tf})
        r = self.p.plan_pose(goal_tool_poses=goal, current_state=start, max_attempts=attempts)
        if linear:
            self.p.update_tool_pose_criteria({k: ToolPoseCriteria() for k in self.tf})
            self.p.enable_link_collision(CONTACT_LINKS + DESCEND_EXTRA_OFF)
        return self._extract(r, start)

    def _ee_pose_axis(self, q):
        """FK of q → (ee_pos[3], ee_quat_wxyz[4], approach_axis ẑ[3]) numpy (단일 env)."""
        tp = self.p.compute_kinematics(q).tool_poses.get_link_pose(self.tf[0])
        pos = tp.position.detach().view(-1).cpu().numpy()[:3]
        quat = tp.quaternion.detach().view(-1).cpu().numpy()[:4]
        w, x, y, z = quat
        zax = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
        return pos, quat, zax

    # ── candidate pose (pan-plane) ──────────────────────────────────────────────
    def _cand_pose(self, xyz, yaw, a_deg, sx):
        """pan-plane pre-grasp 후보 1개 → (pre_pos[3], quat_wxyz[4], loss).

        TCP 는 wrist_roll 축 위·approach 축은 pan 수직평면 안 → 후보가 구성상 5-DOF 도달가능.
        ẑ = −cosα·ê_z + sinα·r̂ (α 틸트), closing축 x̂ = ±t̂ (sx). yaw 주면 ρ=−Δψ/cosα 로
        큐브 face 정렬; loss = |Δψ·tanα|°(정렬 근사 손실). pan 고정점 반복: TCP 가 roll 축
        밖(TCP_LAT)이라 ρ 에 따라 pan 기준 방위가 이동 → φ 2-3회 수렴."""
        a = math.radians(a_deg)
        tcp_lat = np.array(TCP_LAT)
        phi = math.atan2(xyz[1] - PAN_AXIS_XY[1], xyz[0] - PAN_AXIS_XY[0])
        R = z_ax = None
        loss = 0.0
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
            ax_pt = np.array(xyz) - R @ tcp_lat  # roll 축 점(고정점)
            phi = math.atan2(ax_pt[1] - PAN_AXIS_XY[1], ax_pt[0] - PAN_AXIS_XY[0])
        return np.array(xyz) - PRE_BACK * z_ax, _mat2quat(R), loss

    # ── ① approach ──────────────────────────────────────────────────────────────
    def _pre_ladder(self, cube, yaw, start, ladder=None):
        """결정론 후보 ladder: 검증 후보(α=10,sx=+1)부터 순차 K=1 계획, 첫 성공 반환.

        cuRobo goalset 은 임의 도달해 선택 → 접촉 마진이 후보마다 달라 grip 재현 안 됨 →
        순차 K=1 로 planner 가 결정론으로 후보 선택."""
        # sx 미러: 성공 실측 sx=+1 은 좌측(φ<0) 기준. 우측은 거울상으로 jaw 돌출 방향이 반전
        # (테이블 타격 스턱) → φ 부호로 미러링해 항상 같은 물리 방향.
        phi = math.atan2(cube[1] - PAN_AXIS_XY[1], cube[0] - PAN_AXIS_XY[0])
        for a_deg, sx in (ladder or LADDER):
            sx_eff = sx if phi < 0 else -sx
            pos, quat, _ = self._cand_pose(cube, yaw, a_deg, sx_eff)
            traj, end = self._plan_to(self._goal([pos], [quat]), start, attempts=2)
            self._diag(f"[ladder] cand=({a_deg},{sx}) solved={traj is not None}")
            if traj is not None:
                return traj, end
        return None, start

    def _pregrasp_goalset(self, cube, yaw, strict):
        """pan-plane 후보 전체(α×sx)를 goalset 으로. strict=True 면 face-align 손실 게이트
        (TAU_MAX_DEG), 전멸 시 손실 최소 4개. relaxed=coverage 우선(게이트 없음)."""
        nc = len(ALPHAS) * 2
        cands = [self._cand_pose(cube, yaw, a_deg, sx)   # (pos, quat, loss)
                 for a_deg in ALPHAS for sx in (1.0, -1.0)]
        if strict:
            ok = [c for c in cands if c[2] <= TAU_MAX_DEG] or sorted(cands, key=lambda c: c[2])[:4]
        else:
            ok = cands
        base = list(ok)                  # 패딩 전 스냅샷(사이클 기준 — ok 는 아래서 자람)
        while len(ok) < nc:              # goalset 크기 고정(패딩=base 사이클 반복)
            ok.append(base[len(ok) % len(base)])
        pos = np.array([c[0] for c in ok[:nc]], dtype=np.float32)
        quat = np.array([c[1] for c in ok[:nc]], dtype=np.float32)
        return self._goal(pos, quat)

    def _approach(self, cube, yaw, start, ladder):
        """① approach = ladder → strict goalset → relaxed goalset. → (traj|None, q_pre, relaxed).
        contact link off (jaw tip 이 PRE_BACK 서 큐브 obstacle 내부라 off 없이 전멸)."""
        self.p.disable_link_collision(CONTACT_LINKS)
        try:
            pre, q_pre = self._pre_ladder(cube, yaw, start, ladder)
            relaxed = False
            if pre is None:
                pre, q_pre = self._plan_to(self._pregrasp_goalset(cube, yaw, strict=True), start)
            if pre is None:
                pre, q_pre = self._plan_to(self._pregrasp_goalset(cube, yaw, strict=False), start)
                relaxed = pre is not None
        finally:
            self.p.enable_link_collision(CONTACT_LINKS)
        return pre, q_pre, relaxed

    # ── obstacle / attach helpers ───────────────────────────────────────────────
    def _place_cube_obstacle(self, cube):
        """target 큐브 실좌표를 world "cube" obstacle 에 주입(placeholder 는 far)."""
        try:
            cpose = Pose(position=torch.tensor([list(cube)], device="cuda", dtype=torch.float32),
                         quaternion=torch.tensor([[1.0, 0, 0, 0]], device="cuda", dtype=torch.float32))
            self.p.scene_collision_checker.update_obstacle_pose("cube", cpose, env_idx=0)
            self.p.scene_collision_checker.enable_obstacle("cube", True, env_idx=0)
        except Exception as e:
            self._diag(f"[cube-obst] FAIL {type(e).__name__}: {e}")

    def _enable_bowl(self, flag):
        try:
            self.p.scene_collision_checker.enable_obstacle("bowl", flag, env_idx=0)
        except Exception as e:
            self._diag(f"[place] bowl enable={flag} FAIL {type(e).__name__}: {e}")

    def _attachment_manager(self):
        """AttachmentManager 핸들 — 설치본은 TrajOptSolver 컴포지션(.core)이라 위임 property
        부재(신버전은 노출). 양쪽 호환 접근."""
        ts = self.p.trajopt_solver
        am = getattr(ts, "attachment_manager", None)
        return am if am is not None else ts.core.attachment_manager

    def _attach_cube(self, q_grasp):
        """잡은 큐브 blob 을 attached_object 링크에 attach(+world "cube" disable).

        world_objects_pose_offset=None(identity) — attached_object 가 tcp_grasp 와 동일
        transform 이라 blob 이 정확히 pinch 점에 놓임(FK offset 산술 불요). 실패 시 False."""
        try:
            self._attachment_manager().attach(
                q_grasp,
                [Cuboid(name="attached_cube", pose=[0, 0, 0, 1, 0, 0, 0], dims=[CUBE_DIMS] * 3)],
                link_name="attached_object", num_spheres=10,
                sphere_fit_type=SphereFitType.VOXEL,
                world_objects_pose_offset=None, disable_obstacle_names=["cube"])
            self._diag("[attach] ok (identity@tcp_grasp)")
            return True
        except Exception as e:
            self._diag(f"[attach] FAIL {type(e).__name__}: {e} — 무부착 fallback")
            return False

    def _detach(self):
        try:
            self._attachment_manager().detach()
        except Exception as e:
            self._diag(f"[attach] detach FAIL {type(e).__name__}: {e}")

    # ── request parsing ─────────────────────────────────────────────────────────
    def _start_state(self, start_rad):
        """요청 start 자세 → JointState(URDF limit clamp). 없으면 planner default.
        (USD limit ±105° > URDF ±100° 라 bridge 초기자세가 URDF 밖일 수 있어 clamp.)"""
        if start_rad is None:
            return JointState.from_position(self.p.default_joint_state.position.unsqueeze(0),
                                            joint_names=self.p.joint_names)
        arm = torch.tensor(start_rad[0][: self.nA], device="cuda", dtype=torch.float32)
        lim = torch.tensor(ARM_LIMITS, device="cuda", dtype=torch.float32)
        arm = torch.clamp(arm, lim[:, 0] + 0.005, lim[:, 1] - 0.005).unsqueeze(0)
        return JointState.from_position(arm, joint_names=self.p.joint_names)

    @staticmethod
    def _cube_yaw(cube_bl):
        """6D payload([x,y,z,qw,qx,qy,qz]) → 큐브 z-yaw → urdf 프레임(+90°). 아니면 None."""
        if len(cube_bl) < 7:
            return None
        w, x, y, z = cube_bl[3:7]
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)) + math.radians(BASE_YAW)

    def _grasp_diag(self, cube_bl, aq, zax, yaw, relaxed, pre_ok):
        """달성 자세 grip 진단: α(접근 기울기)·Δψ(face 정렬오차)·relaxed."""
        w, x, y, z = aq
        xcol = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)])
        al = math.degrees(math.acos(max(-1.0, min(1.0, -float(zax[2])))))
        azim = math.atan2(xcol[1], xcol[0])
        dps = "-" if yaw is None else round(
            math.degrees((yaw - azim + math.pi / 4) % (math.pi / 2) - math.pi / 4), 1)
        self._diag(f"[graspdiag] cube=({cube_bl[0]:.3f},{cube_bl[1]:.3f}) alpha={al:.1f} "
                   f"dpsi={dps} relaxed={relaxed} pre_ok={pre_ok}")

    # ── main entry ──────────────────────────────────────────────────────────────
    def plan_pickplace(self, cube_bl, bowl_bl=None, start_rad=None, knobs=None):
        """큐브 1개(base_link frame) full 6-phase pick-place 궤적 (T,6) [arm deg×5 + gripper
        feature] 또는 None(어느 phase 든 실패 시).

        knobs: 물리 phase 스윕용 per-request 오버라이드(기본=상수 → 무변경):
          {"grasp_z_off": f, "grip_open": f, "grip_close": f, "ladder": [[α,sx],..]}"""
        kn = knobs or {}
        z_off = float(kn.get("grasp_z_off", GRASP_Z_OFF))
        g_open = float(kn.get("grip_open", GRIP_OPEN))
        g_close = float(kn.get("grip_close", GRIP_CLOSE))
        ladder = [tuple(c) for c in kn.get("ladder", [])] or None
        # 파티클 IK 는 process 누적 seed 상태 → 요청마다 리셋해 에피소드 결정론 복원.
        self.p.reset_seed()

        cube = usd_to_urdf(cube_bl[:3])
        cube = (cube[0], cube[1], cube[2] + z_off)          # grasp 깊이 보정
        yaw = self._cube_yaw(cube_bl)
        bx, by = usd_to_urdf((bowl_bl[0], bowl_bl[1], 0.0))[:2] if bowl_bl is not None else self.bowl_s
        start = self._start_state(start_rad)
        q_home = start.clone()   # ⑥ retreat 가 여기로 복귀 → 다음 start 항상 init
        self._place_cube_obstacle(cube)

        # ① APPROACH
        pre, q_pre, relaxed = self._approach(cube, yaw, start, ladder)

        # ② GRASP — 도착 pre pose FK → 하강선(ẑ) 위 큐브 최근접점까지 linear descend
        app, aq, zax = self._ee_pose_axis(q_pre)
        self._grasp_diag(cube_bl, aq, zax, yaw, relaxed, pre is not None)
        tstar = float(np.dot(np.array(cube) - app, zax))
        gpos = app + tstar * zax
        desc, q_grasp = self._plan_to(self._goal([gpos], [aq]), q_pre, linear=True)

        # ③ LIFT — grasp 서 tool -z linear 역행(pre 지점 너머로 안 나감)
        up = gpos - min(tstar, LIFT_BACK) * zax
        lift, q_lift = self._plan_to(self._goal([up], [aq]), q_grasp, linear=True)

        # attach(lift 후·transit 전) — grasp 직후 attach 는 blob 이 bowl 마진과 겹쳐 lift 허위 FAIL
        attached = self._attach_cube(q_lift)

        transit_z = TRANSIT_Z + BASE_T[2]
        place_z = transit_z - PLACE_DROP
        # ④ TRANSIT — 그릇 상공(bank goalset, bowl obstacle + blob)
        transit, q_transit = self._plan_to(self._goalset((bx, by, transit_z)), q_lift)

        # ⑤ PLACE — 그릇 중심서 수직 linear 하강(transit orientation 유지) 후 release.
        #    bowl cuboid(오목 근사) 허위 충돌 회피 위해 하강 중 bowl disable(중앙 수직=벽 리스크 없음).
        _, tq, _ = self._ee_pose_axis(q_transit)
        self._enable_bowl(False)
        place, q_place = self._plan_to(self._goal([(bx, by, place_z)], [tq]), q_transit, linear=True)
        self._enable_bowl(True)
        if attached:  # release 직전 detach → retreat 는 빈 그리퍼로 계획
            self._detach()

        # ⑥ RETREAT — init(home) 자세로 cspace 복귀(gripper open). bowl 근접 잔여 자세라
        #    empty-world 로 계획(bowl 오목 허위 start-collision 회피).
        self.p.update_world(self.scene_empty)
        retreat, _ = self._extract(self.p.plan_cspace(q_home, q_place), q_place)
        self.p.update_world(self.scene_full)

        phases = {"approach": pre, "grasp": desc, "lift": lift,
                  "transit": transit, "place": place, "retreat": retreat}
        if any(t is None for t in phases.values()):
            msg = "[planner] phase-fail {} cube={}".format(
                " ".join(f"{k}={v is not None}" for k, v in phases.items()), cube_bl)
            print(msg, flush=True); self._diag(msg)
            return None
        return self._assemble(phases, g_open, g_close)

    def _assemble(self, phases, g_open, g_close):
        """6 phase 궤적(rad) → 단일 (T,6) 시퀀스[arm deg + gripper feature].
        gripper 폐합(grasp)·개방(release) ramp 을 정지 hold 에 삽입."""
        a, de, li, tr, pl, rt = (np.rad2deg(phases[k]) for k in
                                 ("approach", "grasp", "lift", "transit", "place", "retreat"))
        close_hold = np.repeat(de[-1:], CLOSE_STEPS, 0)   # grasp: 정지 상태서 폐합
        open_hold = np.repeat(pl[-1:], OPEN_STEPS, 0)     # release: 정지 상태서 개방
        seq = [
            _grip(a, g_open),                                        # ① approach
            _grip(de, g_open),                                       # ② grasp descend
            _grip(close_hold, np.linspace(g_open, g_close, CLOSE_STEPS)),  #   grasp close
            _grip(li, g_close),                                      # ③ lift
            _grip(tr, g_close),                                      # ④ transit
            _grip(pl, g_close),                                      # ⑤ place descend
            _grip(open_hold, np.linspace(g_close, g_open, OPEN_STEPS)),    #   release
            _grip(rt, g_open),                                       # ⑥ retreat
        ]
        return np.vstack(seq).astype(np.float32)


def serve_loop(pl, sock):
    while True:
        req = json.loads(sock.recv())
        cmd = req.get("cmd")
        if cmd == "ping":
            sock.send_string(json.dumps({"ok": True, "n_envs": pl.n_envs}))
        elif cmd == "plan_pickplace":
            traj = pl.plan_pickplace(req["cubes"][0], req.get("bowl"), req.get("start"),
                                     req.get("knobs"))
            sock.send_string(json.dumps(
                {"ok": True, "trajectories": [traj.tolist() if traj is not None else None]}))
        elif cmd == "shutdown":
            sock.send_string(json.dumps({"ok": True})); return
        else:
            sock.send_string(json.dumps({"ok": False, "err": f"unknown {cmd!r}"}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5599)
    ap.add_argument("--n-envs", dest="n_envs", type=int, default=1)  # 배치 확장 hook
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    pl = PickPlacePlanner(n_envs=a.n_envs)
    print(f"[planner] ready n_envs={pl.n_envs}", flush=True)
    if a.self_test:
        traj = pl.plan_pickplace([0.017, -0.253, 0.06])
        print(f"self-test: {'OK' if traj is not None else 'FAIL'} "
              f"shape={None if traj is None else traj.shape}")
        print("SELFTEST_OK" if traj is not None else "SELFTEST_CHECK")
        return
    sock = zmq.Context().socket(zmq.REP); sock.bind(f"tcp://*:{a.port}")
    print(f"[planner] ZMQ REP :{a.port}", flush=True)
    serve_loop(pl, sock)


if __name__ == "__main__":
    main()
