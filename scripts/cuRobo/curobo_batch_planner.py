"""cuRobo pick-place ZMQ planner 서비스 (SO-101, 신 API v0.8).

★2-프로세스: 이 planner(so101-curobo-datagen, cuRobo+warp) ↔ Isaac executor
(so101-isaac-sim) ZMQ 분리(in-process 불가=warp ABI). executor 가 큐브(base_link)를 보내면
planner 가 collision-aware full pick-place 궤적을 반환한다.

★multi-env: 요청의 `cubes` 리스트 길이 = env 수. executor 는 N envs 를 IsaacLab native
lockstep 으로 병렬 replay 한다. planner 도 cuRobo `BatchMotionPlanner(multi_env=True)`의
batch 차원을 env 차원으로 맞춰 N개 env를 한 번에 푼다. grasp 후보는 env별 goalset 차원이다.

계획: tool frame = **tcp_grasp**(손가락 사이 pinch 점, so101.yml extra_link).
★5-DOF 원칙: orientation 고정 시 도달 위치는 2D 면 → goal 은 항상 **bank pose 그대로**
(FK-harvest (pos,quat) 쌍 = 존재증명) 또는 그 tool-z 평행이동(pan-plane 불변)만 쓴다.

6-phase(approach·grasp·lift·transit·release·retreat):
  ① approach = simple face-normal grasp 후보 batch 검증 → top-down 우선 랜덤 선택
  ② grasp    = 도착 pre pose FK → approach 축 linear descend(table clamp) + gripper 폐합
  ③ lift     = grasp 서 tool -z linear 역행(잡은 채 수직 이탈), attach 큐브 blob
  ④ transit  = 그릇 상공(TRANSIT_Z) bank goalset — 드롭 XY 는 BOWL_PULL 만큼 base 쪽
  ⑤ release  = 그릇 상공 settle hold → detach + gripper 개방(하강 없음, 큐브 낙하)
  ⑥ retreat  = init(home=start) 자세로 cspace 복귀(full scene, gripper open)
프레임=base_link→solver rotz(90).

collision 구성(cuRobo 정석 미러): target 큐브 = world obstacle("cube", per-request pose 주입)
→ grasp 후 attach 로 큐브 blob 을 attached_object(=tcp_grasp 동일 프레임)에 부착 + "cube"
disable → transit 은 잡은 큐브 부피 포함 계획 → release 직전 detach(+stale cube 재-disable).
그릇 = hollow rim ring(오목 fit) — §bowl ring 상수 참조.

프로토콜(JSON REQ/REP, port 5599) — cube 는 base_link frame, quat 주면 yaw face-align:
  {"cmd":"ping"}                                          → {"ok":true}
  {"cmd":"plan_pickplace",
   "cubes":[[x,y,z(,qw,qx,qy,qz)]×N],                     → {"ok":true,
   "bowl":[x,y] | [[x,y]×N],                                  "trajectories":[[[6]×T]|null ×N]}
   "start":[[6 joint rad]×N] (선택), "knobs":{...}(선택)}
  {"cmd":"shutdown"}                                      → {"ok":true}

실행: /isaac-sim/python.sh curobo_batch_planner.py [--port 5599] [--self-test]
"""
import argparse
import json
import math
import tempfile
import time

import numpy as np
import torch
import yaml
import zmq
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.geom.sphere_fit import SphereFitType
from curobo._src.geom.types import Cuboid
from curobo._src.motion.motion_planner_batch import BatchMotionPlanner
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.motion_planner import MotionPlannerCfg
from curobo.types import GoalToolPose, JointState, Pose

ROBOT = "/workspace/assets/robots/so101.yml"
DIAG_LOG = "/workspace/outputs/planner_diag.log"  # 호스트 마운트(./outputs) — 컨테이너 소멸 후 잔존

# ═══ 프레임 정합 (bridge TF ↔ cuRobo URDF) ═══════════════════════════════════════
# bridge TF(base_link)=USD so101_new_calib 규약, cuRobo URDF=so_arm101 규약. 두 체인의
# shoulder_pan 프레임 일치 조건에서 T(urdf←usd)=Rz(90°)+BASE_T (실측 유도 — 미보정 시 ~3cm 빗나감).
BASE_YAW = 90.0
BASE_T = (0.01576, -0.02079, -0.03248)
TABLE_TOP = 0.035 + BASE_T[2]   # 책상 상판 z (urdf 프레임)

# ═══ 로봇/큐브 기하 (실측 단일 소스) ═══════════════════════════════════════════════
PAN_AXIS_XY = (0.0388353, 0.0)  # shoulder_pan 축의 solver-frame XY — pan 평면·거리 기준점(URDF)
TCP_LAT = (0.0, -0.015, 0.0)    # tcp_grasp(비대칭 jaw pinch 중심)의 wrist_roll 축 이탈 — pan 보정
# fixed jaw pad 접촉면 center 의 tcp-frame 오프셋(gripper_link pad sphere centroid 실측).
# pad 이 tcp 서 46mm 아래·15mm 옆 — tcp 조준 시 pad 이 face 서 크게 벗어나는(edge grip) 원인.
FIXED_INNER_CENTER = (0.0215, 0.0147, 0.0463)  # (closing, lateral, jaw 아래방향) m
PAD_LOW_OFF = 0.075   # tcp → 물리 pad 최저점(fixed jaw tip) approach축 거리 — so101.yml 실측
                      # (tip -0.100 − tcp -0.025). moving jaw 는 12mm 얕음 → fixed 기준이 지배.
CUBE_HALF = 0.020     # 큐브 반변(40mm) — face_center = cube_center + CUBE_HALF·closing_axis
CUBE_DIMS = 0.05      # 큐브 obstacle/attach blob 한 변(m) — 보수적 최대값(cube_specs 40/50mm)
CONTACT_LINKS = ["gripper_link", "moving_jaw_so101_v1_link"]   # descend 중 collision off
DESCEND_EXTRA_OFF = ["wrist_link", "wrist_cam_mount_link"]     # 〃 — grasp 자세서 wrist sphere 가
                      # 큐브 obstacle 과 모델상 겹침(짧은 wrist, 물리 접촉 없음=sphere 보수 근사)
# so_arm101.urdf arm 관절 limit(rad) — start clamp 용(USD ±105° 와 어긋나는 5° 캘리브 마진)
ARM_LIMITS = [(-1.91986, 1.91986), (-1.74533, 1.74533), (-1.69, 1.69),
              (-1.65806, 1.65806), (-2.74385, 2.84121)]

# ═══ grasp 후보 + face-center 게이트 ══════════════════════════════════════════════
# 후보 = cube face normal + pan-plane tilt α. wrist_roll 은 init 그대로 보지 않고,
# shoulder_pan 을 먼저 cube vertical plane 에 맞춘 FK 자세의 gripper closing axis 를 기준으로
# closer face 를 고른다. cuRobo 는 후보를 batch 차원으로 한 번에 풀고, FK face-center gate 를
# 통과한 후보 중 top-down 을 우선 랜덤 선택한다.
SIMPLE_TILTS_DEG = [-70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70]
SIMPLE_FACE_LOSS_MAX_DEG = 10.0
SIMPLE_FACE_COUNT = 4
SIMPLE_MAX_CANDIDATES = len(SIMPLE_TILTS_DEG) * SIMPLE_FACE_COUNT
TOPDOWN_ALPHA_MAX_DEG = 10.0
WRIST_ROLL_DELTA_LIMIT_DEG = 100.0
# 게이트(IK 성공만으론 불충분 — IK-후-FK 실측 pad center 를 face center 와 3D 비교):
#   e_normal(closing clearance) ∈ [1,6]mm · |e_tangent| ≤ 22mm · |e_height| ≤ 28mm.
# tangent/height 폭 = fingers-down branch 에서 실제 도달 가능한 범위. 이상(3/4mm)은 40mm 큐브서
# kinematically 불가(75mm jaw+책상 clearance+tilt<50°); 미러 branch 가 더 좋았지만(e_h 6mm)
# wrist ~223° 뒤집기라 금지(사용자). 좁히면 simple 후보가 전멸한다.
FIXED_JAW_CLEAR_TARGET = 0.003  # pad center 조준 clearance(게이트 1~6mm 중앙)
FIXED_JAW_CLEAR_MIN, FIXED_JAW_CLEAR_MAX = 0.001, 0.006
E_TANGENT_MAX = 0.022
E_HEIGHT_MAX = 0.028
# wrist_roll gate = pan-aligned init pose 의 wrist_roll 기준 ±100°. shoulder_pan 만 cube plane으로
# 돌린 자세를 기준으로 보기 때문에 wrist_roll 값 자체는 start와 동일하며, SO-101 5-DoF에서
# cube face normal 정렬은 이 상대 회전 범위 안에 머물러야 한다.
WRIST_ROLL_DELTA_LIMIT = math.radians(WRIST_ROLL_DELTA_LIMIT_DEG)

# ═══ phase 파라미터 ═══════════════════════════════════════════════════════════════
K = 40              # goalset 크기(bank reach)
GRASP_Z_OFF = -0.008  # grasp 깊이 미세보정(m): pinch 큐브 상단걸침 → 8mm 하향(clamp 우선)
TABLE_MARGIN = 0.004  # pad 최저점 정지 고도(table_top 위, m). 사용자 요구 "실제 ≥2mm 무접촉"
                      # — IK 잔차+tilt 투영오차가 먹으므로 4mm 조준 → 실제 ≥2mm.
LIFT_BACK = 0.10    # ③ lift: grasp 서 tool -z 최대 역행(approach 되감기, m)
TRANSIT_Z = 0.21    # ④ transit 그릇 상공 고도(urdf, m). 0.25→0.21: ring keep-out 이 rim 을
                    # 스스로 피해 인하(드롭 과고 해소). 스침 재발 시 ring dims 먼저, 그다음 ↑.
BOWL_PULL = 0.03    # ④ 드롭 XY 를 그릇 중심서 base 쪽으로 당김(m) — near-rim 착지
# pre-grasp 후퇴(=descend 거리, m) — 큐브 pan-축 거리 r 적응(사용자 제안). jaw tip 이 큐브
# obstacle 위로 떠 approach 를 jaw-collision ON 으로 계획. 고정 0.12 는 pad-center 조준(+4.6cm)
# 과 겹쳐 근거리(r≈0.12) pre IK 전멸(실측 pre z 0.186 unreachable / 0.146 solved).
PRE_BACK_MIN, PRE_BACK_MAX = 0.06, 0.12
PRE_BACK_R0, PRE_BACK_R1 = 0.13, 0.24   # r≤R0→MIN · r≥R1→MAX (사이 선형)

# ═══ gripper 시퀀스 (feature [0,100]) ═════════════════════════════════════════════
GRIP_OPEN, GRIP_CLOSE = 75.0, 5.0  # open 75=straddle 마진(60 은 tangential 3mm 오차로 squirt)
GRIP_INIT = 0.0     # ⑥ retreat 끝 복원값 = SM init(-10°=feature 0)
CLOSE_STEPS = 5     # ② grasp 폐합 ramp 프레임
OPEN_STEPS = 10     # ⑤ release 개방 ramp 프레임(정지 상태 투하)
SETTLE_STEPS = 5    # ⑤ release 전 그릇 상공 정지 hold 프레임

# ═══ bowl obstacle = hollow rim ring (오목 그릇 fit) ══════════════════════════════
# cuRobo world obstacle 은 solid convex 뿐 — 오목 그릇을 solid box 로 넣으면 내부(빈 공간)가
# retreat 자세와 허위충돌. rim 벽만 N× cuboid octagon 으로 근사(내부 hole+상단 open) →
# keep-out 유지·허위충돌 제거, empty-world swap 없이 full scene 계획.
# 박스 = 다각형 '변': tangential w≈변길이+겹침, radial d 는 코너 azimuth 벽 구멍이 없을 만큼
# 두껍게 — 0-gap 은 _assert_bowl_ring_sealed 가 임포트 시 강제(scratch/search_ring.py 근거).
BOWL_RING_N = 8
BOWL_RING_RC = 0.080   # box 중심 반경(m) — 벽 band [RC±d/2] 이 실제 rim 0.075 포함
BOWL_RING_H = 0.075    # ring 높이(테이블→rim)
BOWL_RING_DIMS = (0.030, 0.083, BOWL_RING_H)  # [radial, tangential, height]


def _pre_back(cube):
    """큐브(solver frame) → pan 축 거리 r 로 pre-grasp 후퇴량 보간(0.06~0.12m)."""
    r = math.hypot(cube[0] - PAN_AXIS_XY[0], cube[1] - PAN_AXIS_XY[1])
    t = min(1.0, max(0.0, (r - PRE_BACK_R0) / (PRE_BACK_R1 - PRE_BACK_R0)))
    return PRE_BACK_MIN + t * (PRE_BACK_MAX - PRE_BACK_MIN)


def _bowl_ring(bx, by):
    """오목 그릇 rim → hollow octagon ring(BOWL_RING_N× cuboid) dict{name:{dims,pose}}.
    각 box = 다각형 '변'(local-x=radial, quat=Rz(θ)), 배치 반경 RC → 내부 hole+상단 open.
    중심(bx,by) 이동(DR)마다 재계산."""
    z_c = TABLE_TOP + BOWL_RING_H / 2
    ring = {}
    for i in range(BOWL_RING_N):
        th = 2 * math.pi * i / BOWL_RING_N
        qw, qz = math.cos(th / 2), math.sin(th / 2)
        ring[f"bowl_{i}"] = {
            "dims": list(BOWL_RING_DIMS),
            "pose": [bx + BOWL_RING_RC * math.cos(th),
                     by + BOWL_RING_RC * math.sin(th), z_c, qw, 0.0, 0.0, qz]}
    return ring


def _assert_bowl_ring_sealed():
    """로드시 1-check: (a) hole 이 drop+pad 여유 (b) rim 원둘레 전 azimuth 벽 연속(gap 없음).
    얇은 radial d 는 코너에 벽 구멍→팔이 그릇 파고듦 → 임포트서 즉시 실패."""
    hole = BOWL_RING_RC - BOWL_RING_DIMS[0] / 2
    assert hole >= BOWL_PULL + 0.028, f"bowl ring hole {hole:.3f}m < drop+pad(~0.028)"
    ring = _bowl_ring(0.0, 0.0)

    def _inside(px, py, e):
        x, y, _z, qw, _qx, _qy, qz = e["pose"]
        dx, dy, _dz = e["dims"]
        th = 2 * math.atan2(qz, qw)
        c, s = math.cos(-th), math.sin(-th)
        return abs(c * (px - x) - s * (py - y)) <= dx / 2 and abs(s * (px - x) + c * (py - y)) <= dy / 2

    for k in range(720):
        ph = 2 * math.pi * k / 720
        px, py = 0.075 * math.cos(ph), 0.075 * math.sin(ph)  # 실제 rim 원
        assert any(_inside(px, py, e) for e in ring.values()), \
            f"bowl ring wall gap @ {math.degrees(ph):.0f}° — radial d↑ 또는 N↑"


_assert_bowl_ring_sealed()


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
    """pick-place planner. BatchMotionPlanner batch 차원 = IsaacLab env 차원.

    후보 grasp 는 goalset 으로 넣고, phase별 plan_pose/plan_cspace 는 N개 env를 한 번에 푼다.
    cuRobo `multi_env=True`에서는 batch index가 collision env index가 되므로 이 매핑을 유지해야
    DR로 서로 다른 cube/bowl obstacle을 병렬 계획에 올바르게 적용할 수 있다.
    """

    def __init__(self, bowl_bl=(0.22, -0.265), max_batch_size=64):
        self.default_bowl_bl = bowl_bl
        self.max_batch_size = int(max_batch_size)
        self.max_goalset = max(K, SIMPLE_MAX_CANDIDATES)
        bx, by, _ = usd_to_urdf((bowl_bl[0], bowl_bl[1], 0.0))
        self.bowl_s = (bx, by)
        # 책상은 world obstacle 로 넣지 않는다 — 로봇이 책상 위에 장착돼 base 구가 상판(TABLE_TOP)
        # 안에 들어가 매 plan 이 start-collision 으로 거부됨. 대신 grasp 깊이는 pad-frame clamp
        # (TABLE_TOP+TABLE_MARGIN)로 제한 → "책상 obstacle+너무 깊은 descend reject" 를 clamp 로 대체.
        # bowl = hollow rim ring(_bowl_ring) — 오목 그릇 내부를 비워 retreat hover 허위충돌 방지.
        world = {"cuboid": {
            **_bowl_ring(bx, by),
            # target 큐브 world obstacle — per-request update_obstacle_pose 로 실좌표 주입
            # (placeholder 는 far). approach 중 팔 링크의 큐브 관통 방지.
            "cube": {"dims": [CUBE_DIMS] * 3, "pose": [9.0, 9.0, 0.02, 1, 0, 0, 0]}}}
        wf = tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False)
        # cuRobo SceneCollisionCfg 는 scene_model 이 list 일 때만 collision world num_envs 를
        # list 길이로 잡는다. multi_env=True에서 batch index=env_idx 이므로 world도 N개 복제한다.
        yaml.safe_dump([json.loads(json.dumps(world)) for _ in range(self.max_batch_size)], wf)
        wf.close()
        # planner 는 wf.name(=world) 로 로드 → per-request 는 update_obstacle_pose 로 그릇/큐브
        # 좌표만 갱신. hollow ring 이라 retreat·self-check 모두 full scene(옛 empty-world swap 제거).

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
            robot=ROBOT, scene_model=wf.name,
            max_batch_size=self.max_batch_size,
            max_goalset=self.max_goalset,
            multi_env=True,
            use_cuda_graph=False))
        self.p.warmup(enable_graph=False, num_warmup_iterations=2)
        self.tf = self.p.tool_frames
        self.nA = len(self.p.joint_names)
        self.rng = np.random.default_rng()
        self.last_candidate_diag = None
        self.last_plan_diag = {}

    def _ensure_batch_size(self, batch_size):
        """BatchMotionPlanner는 config.max_batch_size가 실제 batch 크기다.

        cuRobo 내부 padding 경로는 JointState jerk 등에서 shape 문제가 있어, 요청 env 수와
        planner batch size를 정확히 맞춘다. 동시에 살아있는 BatchMotionPlanner는 항상 1개다.
        """
        batch_size = int(batch_size)
        if batch_size == self.max_batch_size:
            return
        try:
            self.p.destroy()
        except Exception:
            pass
        self.__init__(bowl_bl=self.default_bowl_bl, max_batch_size=batch_size)

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

    def _goalset_batch(self, pos, quat):
        """(B,G,3)/(B,G,4) → B개 env, env별 G개 goalset."""
        pos_arr = np.asarray(pos)
        quat_arr = np.asarray(quat)
        if pos_arr.shape[0] < self.max_batch_size:
            pad = self.max_batch_size - pos_arr.shape[0]
            pos_arr = np.concatenate([pos_arr, np.repeat(pos_arr[:1], pad, axis=0)], axis=0)
            quat_arr = np.concatenate([quat_arr, np.repeat(quat_arr[:1], pad, axis=0)], axis=0)
        b, g = pos_arr.shape[:2]
        p = torch.as_tensor(pos_arr, device="cuda", dtype=torch.float32).view(b, 1, 1, g, 3)
        q = torch.as_tensor(quat_arr, device="cuda", dtype=torch.float32).view(b, 1, 1, g, 4)
        return GoalToolPose(tool_frames=self.tf, position=p, quaternion=q)

    def _pose_batch(self, pos, quat):
        """(B,3)/(B,4) → B개 env, env별 단일 pose goal."""
        pos_arr = np.asarray(pos)
        quat_arr = np.asarray(quat)
        if pos_arr.shape[0] < self.max_batch_size:
            pad = self.max_batch_size - pos_arr.shape[0]
            pos_arr = np.concatenate([pos_arr, np.repeat(pos_arr[:1], pad, axis=0)], axis=0)
            quat_arr = np.concatenate([quat_arr, np.repeat(quat_arr[:1], pad, axis=0)], axis=0)
        p = torch.as_tensor(pos_arr, device="cuda", dtype=torch.float32).view(-1, 1, 1, 1, 3)
        q = torch.as_tensor(quat_arr, device="cuda", dtype=torch.float32).view(-1, 1, 1, 1, 4)
        return GoalToolPose(tool_frames=self.tf, position=p, quaternion=q)

    def _joint_state_batch(self, q):
        q = torch.as_tensor(q, device="cuda", dtype=torch.float32)
        if q.dim() == 1:
            q = q.unsqueeze(0)
        return JointState.from_position(q[:, :self.nA].clone(), joint_names=self.p.joint_names)

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

    def _extract_batch(self, r, start, n):
        """Batched plan_pose 결과를 후보별 (traj|None, end) 리스트로 분해한다."""
        if r is None:
            return [(None, start) for _ in range(n)]
        pos = r.interpolated_trajectory.position.detach()
        if pos.dim() == 4 and pos.shape[1] == 1:
            pos = pos[:, 0]
        while pos.dim() > 3:
            pos = pos[..., 0, :, :]
        success = r.success.detach().view(-1)
        last = r.interpolated_last_tstep
        last_flat = last.detach().view(-1) if last is not None else None
        out = []
        for bi in range(n):
            ok = bi < success.numel() and bool(success[bi].item())
            ti = int(last_flat[bi].item()) if last_flat is not None and bi < last_flat.numel() else pos.shape[1]
            q = pos[bi, : max(ti, 2), : self.nA]
            end = JointState.from_position(q[-1:].clone(), joint_names=self.p.joint_names)
            out.append((q.cpu().numpy() if ok else None, end))
        return out

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

    def _plan_to_batch(self, goal, start, n, linear=False, attempts=3):
        """Batched GoalToolPose 직접 계획 → [(traj|None,end)]×n."""
        if linear:
            self.p.disable_link_collision(CONTACT_LINKS + DESCEND_EXTRA_OFF)
            lin = ToolPoseCriteria.linear_motion(axis="z", non_terminal_scale=1.0,
                                                 project_distance_to_goal=True)
            self.p.update_tool_pose_criteria({k: lin for k in self.tf})
        r = self.p.plan_pose(goal_tool_poses=goal, current_state=start, max_attempts=attempts)
        if linear:
            self.p.update_tool_pose_criteria({k: ToolPoseCriteria() for k in self.tf})
            self.p.enable_link_collision(CONTACT_LINKS + DESCEND_EXTRA_OFF)
        return self._extract_batch(r, start, n)

    def _plan_cspace_batch(self, goal_states, current_state, n, attempts=1):
        r = self.p.plan_cspace(goal_states, current_state, max_attempts=attempts)
        return self._extract_batch(r, current_state, n)

    @staticmethod
    def _quat_axes(quat):
        w, x, y, z = quat
        xax = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)])
        yax = np.array([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)])
        zax = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
        return xax, yax, zax

    def _ee_pose_axis(self, q):
        """FK of q → (ee_pos[3], ee_quat_wxyz[4], approach_axis ẑ[3]) numpy (단일 env)."""
        tp = self.p.compute_kinematics(q).tool_poses.get_link_pose(self.tf[0])
        pos = tp.position.detach().view(-1).cpu().numpy()[:3]
        quat = tp.quaternion.detach().view(-1).cpu().numpy()[:4]
        _xax, _yax, zax = self._quat_axes(quat)
        return pos, quat, zax

    def _ee_pose_axis_batch(self, q):
        """FK of q(N) → pos(N,3), quat(N,4), z_axis(N,3)."""
        tp = self.p.compute_kinematics(q).tool_poses.get_link_pose(self.tf[0])
        pos = tp.position.detach().view(-1, 3).cpu().numpy()
        quat = tp.quaternion.detach().view(-1, 4).cpu().numpy()
        z_axes = []
        for qq in quat:
            _xax, _yax, zax = self._quat_axes(qq)
            z_axes.append(zax)
        return pos, quat, np.asarray(z_axes, dtype=np.float64)

    def _pan_aligned_closing_axis(self, start, cube):
        """start 에서 shoulder_pan 만 cube vertical plane 으로 돌린 뒤 gripper x축을 읽는다."""
        qpos = start.position.detach().clone()
        if qpos.dim() == 1:
            qpos = qpos.unsqueeze(0)
        pan_idx = self.p.joint_names.index("shoulder_pan")
        qpos[0, pan_idx] = math.atan2(cube[1] - PAN_AXIS_XY[1], cube[0] - PAN_AXIS_XY[0])
        q_pan = JointState.from_position(qpos[:1], joint_names=self.p.joint_names)
        tp = self.p.compute_kinematics(q_pan).tool_poses.get_link_pose(self.tf[0])
        quat = tp.quaternion.detach().view(-1).cpu().numpy()[:4]
        xax, _yax, _zax = self._quat_axes(quat)
        xax[2] = 0.0
        norm = np.linalg.norm(xax)
        if norm < 1e-6:
            return np.array([cube[0] - PAN_AXIS_XY[0], cube[1] - PAN_AXIS_XY[1], 0.0])
        return xax / norm

    def _grasp_face_error(self, q_pre, cube, face_normal=None):
        """IK-후-FK 실측 fixed jaw inner face center 를 cube face center 와 **3D** 비교(사용자 스펙).

        grasp 자세 = pre 자세서 approach축(tcp z) linear descend(table clamp; plan_pickplace grasp
        와 동일 식). descend 는 orientation 보존 → grasp 회전 = pre 회전이라 pad 방향 정확, 위치만
        하강 이동. fixed jaw inner face center = grasp_tcp + R·FIXED_INNER_CENTER(단순 tcp+offset·x̂ 아님).
        face_center = cube_center + CUBE_HALF·n(closing축). e 를 (normal, tangent, height)로 분해.
        returns {n:e_normal(clearance), t:e_tangent(face-plane lateral), h:e_height(world-z),
                 a:alpha(tilt°), c:centerline(√(t²+h²))}."""
        tp = self.p.compute_kinematics(q_pre).tool_poses.get_link_pose(self.tf[0])
        pos = tp.position.detach().view(-1).cpu().numpy()[:3]
        quat = tp.quaternion.detach().view(-1).cpu().numpy()[:4]
        xax, yax, zax = self._quat_axes(quat)  # x=closing, z=approach
        cc = np.array(cube[:3])
        # descend = pre 서 _pre_back(=pad-at-face-center 조준 backoff, r 적응) 하강, table clamp 동일
        tstar = _pre_back(cube); zaz = float(zax[2])
        if zaz < -1e-3:  # 하강 중 — pad 최저점이 table+margin 아래로 못 가게 clamp
            tstar = min(tstar, (TABLE_TOP + TABLE_MARGIN - float(pos[2])) / zaz - PAD_LOW_OFF)
        grasp_tcp = pos + tstar * zax
        dx, dy, dz = FIXED_INNER_CENTER
        fixed_inner = grasp_tcp + dx * xax + dy * yax + dz * zax   # FK 실측 pad center(world)
        n_face = np.array(face_normal, dtype=np.float64) if face_normal is not None else xax
        n_face[2] = 0.0
        n_norm = np.linalg.norm(n_face)
        n_face = n_face / n_norm if n_norm > 1e-6 else xax
        if float(np.dot(n_face, xax)) < 0.0:
            n_face = -n_face
        face_center = cc + CUBE_HALF * n_face                      # fixed jaw 가 닿는 실제 cube face 중심
        e = fixed_inner - face_center
        t = np.cross(np.array([0.0, 0.0, 1.0]), n_face); tn = np.linalg.norm(t)
        t = t / tn if tn > 1e-6 else np.array([0.0, 1.0, 0.0])     # face-plane tangent(수평, ⊥ closing)
        e_t, e_h = float(np.dot(e, t)), float(e[2])
        face_angle = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(xax, n_face))))))
        return {"n": float(np.dot(e, n_face)), "t": e_t, "h": e_h,
                "a": math.degrees(math.acos(max(-1.0, min(1.0, -zaz)))), "c": math.hypot(e_t, e_h),
                "face_angle": float(face_angle)}

    # ── candidate pose (pan-plane) ──────────────────────────────────────────────
    @staticmethod
    def _face_normals_from_yaw(yaw):
        """cube yaw → solver-frame horizontal face outward normals 4개."""
        if yaw is None:
            return []
        return [
            np.array([math.cos(yaw + k * math.pi / 2), math.sin(yaw + k * math.pi / 2), 0.0],
                     dtype=np.float64)
            for k in range(4)
        ]

    def _ordered_face_normals(self, xyz, yaw, start):
        """pan-align 후 gripper closing axis 에 가장 가까운 cube face normal 순서."""
        faces = self._face_normals_from_yaw(yaw)
        if not faces:
            return [], None
        ref = self._pan_aligned_closing_axis(start, xyz)
        rn = np.linalg.norm(ref)
        if rn < 1e-6:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = ref / rn
        ranked = []
        for face_idx, normal in enumerate(faces):
            dot = max(-1.0, min(1.0, float(np.dot(ref, normal))))
            ranked.append((face_idx, normal, math.degrees(math.acos(dot)), dot))
        ranked.sort(key=lambda item: item[2])
        return ranked, ref

    def _cand_pose_simple(self, xyz, face_normal, a_deg):
        """단순 후보: cube face normal + tilt α 만으로 full TCP pose(quat)를 만든다.

        - approach ẑ: shoulder pan axis↔cube center 수직평면 안.
        - closing x̂: 실제 cube face normal 을 ẑ 에 수직 투영한 방향.
        - TCP 위치: fixed jaw inner pad center 가 face-front offset 에 오도록 역산.
        """
        cc = np.array(xyz[:3], dtype=np.float64)
        f = np.array(face_normal, dtype=np.float64)
        f[2] = 0.0
        fn = np.linalg.norm(f)
        if fn < 1e-6:
            raise ValueError("face normal is degenerate")
        f = f / fn

        pan_to_cube = np.array([cc[0] - PAN_AXIS_XY[0], cc[1] - PAN_AXIS_XY[1], 0.0],
                               dtype=np.float64)
        rn = np.linalg.norm(pan_to_cube)
        if rn < 1e-6:
            raise ValueError("cube is too close to shoulder pan axis")
        r_hat = pan_to_cube / rn

        a = math.radians(a_deg)
        z_ax = -math.cos(a) * np.array([0.0, 0.0, 1.0]) + math.sin(a) * r_hat
        z_ax = z_ax / np.linalg.norm(z_ax)

        x_proj = f - float(np.dot(f, z_ax)) * z_ax
        x_norm = np.linalg.norm(x_proj)
        if x_norm < 1e-6:
            raise ValueError("face normal is parallel to approach axis")
        x_ax = x_proj / x_norm
        if float(np.dot(x_ax, f)) < 0.0:
            x_ax = -x_ax
        y_ax = np.cross(z_ax, x_ax)
        y_ax = y_ax / np.linalg.norm(y_ax)
        # 수치 오차 제거: x̂ = ŷ×ẑ 로 다시 직교화.
        x_ax = np.cross(y_ax, z_ax)
        x_ax = x_ax / np.linalg.norm(x_ax)
        R = np.stack([x_ax, y_ax, z_ax], 1)

        pad_target = cc + (CUBE_HALF + FIXED_JAW_CLEAR_TARGET) * f
        tcp_tgt = pad_target - R @ np.array(FIXED_INNER_CENTER, dtype=np.float64)
        pre_pos = tcp_tgt - _pre_back(xyz) * z_ax
        face_loss = math.degrees(math.asin(max(-1.0, min(1.0, abs(float(np.dot(f, z_ax)))))))
        pan_residual = float(np.dot(z_ax, np.cross(r_hat, np.array([0.0, 0.0, 1.0]))))
        return pre_pos, _mat2quat(R), {
            "mode": "simple",
            "alpha_deg": float(a_deg),
            "face_normal": f.astype(float).tolist(),
            "face_loss_deg": float(face_loss),
            "pan_residual": pan_residual,
            "tcp_target": tcp_tgt.astype(float).tolist(),
            "pre_target": pre_pos.astype(float).tolist(),
            "quat_wxyz": _mat2quat(R).astype(float).tolist(),
        }

    def _simple_candidates(self, xyz, yaw, start, face_count=SIMPLE_FACE_COUNT):
        """가까운 cube face normal×tilt 후보를 만든다."""
        ranked_faces, ref_axis = self._ordered_face_normals(xyz, yaw, start)
        if not ranked_faces:
            return []
        ranked_faces = [f for f in ranked_faces if f[2] <= WRIST_ROLL_DELTA_LIMIT_DEG]
        if not ranked_faces:
            return []
        face_count = max(1, min(int(face_count), len(ranked_faces)))
        face_msg = " ".join(
            f"#{idx}:angle={angle:.1f},dot={dot:.2f}" for idx, _n, angle, dot in ranked_faces
        )
        self._diag(f"[simple] pan-aligned closing={None if ref_axis is None else ref_axis.tolist()} faces={face_msg}")
        cands = []
        for face_rank, (face_idx, normal, ref_angle, ref_dot) in enumerate(ranked_faces[:face_count]):
            for a_deg in SIMPLE_TILTS_DEG:
                try:
                    pos, quat, meta = self._cand_pose_simple(xyz, normal, a_deg)
                except ValueError:
                    continue
                meta["face_index"] = face_idx
                meta["face_rank"] = face_rank
                meta["reference_face_angle_deg"] = float(ref_angle)
                meta["reference_dot"] = float(ref_dot)
                if meta["face_loss_deg"] <= SIMPLE_FACE_LOSS_MAX_DEG:
                    cands.append((pos, quat, meta))
        cands.sort(key=lambda c: (c[2]["face_rank"], abs(c[2]["alpha_deg"]),
                                  round(c[2]["face_loss_deg"], 3), c[2]["face_index"]))
        return cands[:SIMPLE_MAX_CANDIDATES]

    @staticmethod
    def _priority_shuffle_candidates(cands, rng):
        """top-down 가능 후보를 먼저, 그 안에서는 closest face 우선 + 랜덤 tie-break."""
        if not cands:
            return []
        topdown = [c for c in cands if abs(c[2]["alpha_deg"]) <= TOPDOWN_ALPHA_MAX_DEG]
        base = topdown if topdown else cands
        best_rank = min(c[2].get("face_rank", 0) for c in base)
        primary = [c for c in base if c[2].get("face_rank", 0) == best_rank]
        primary_idx = list(range(len(primary)))
        rng.shuffle(primary_idx)
        primary = [primary[i] for i in primary_idx]
        primary_keys = {id(c) for c in primary}
        rest = [c for c in cands if id(c) not in primary_keys]
        return primary + rest

    def _wrist_delta_ok(self, q, start):
        wr_idx = self.p.joint_names.index("wrist_roll")
        qpos = q.position.detach()
        spos = start.position.detach()
        wr = float(qpos.view(-1, self.nA)[0, wr_idx].item())
        swr = float(spos.view(-1, self.nA)[0, wr_idx].item())
        delta = wr - swr
        return abs(delta) <= WRIST_ROLL_DELTA_LIMIT, wr, delta

    # ── ① approach ──────────────────────────────────────────────────────────────
    def _plan_pregrasp(self, cube, yaw, start, knobs):
        """simple 후보를 goalset 으로 계획하고, cuRobo가 고른 후보를 FK gate로 검증한다."""
        kn = knobs or {}
        seed = kn.get("seed")
        rng = np.random.default_rng(int(seed)) if seed is not None else self.rng
        face_count = int(kn.get("simple_face_count", SIMPLE_FACE_COUNT))
        candidates = self._priority_shuffle_candidates(
            self._simple_candidates(cube, yaw, start, face_count=face_count), rng)
        yaw_msg = "None" if yaw is None else f"{math.degrees(yaw):.1f}"
        self._diag(f"[simple] candidates={len(candidates)} yaw={yaw_msg} face_count={face_count} seed={seed}")
        if not candidates:
            self.last_candidate_diag = {"mode": "simple", "fail": "no_candidates"}
            return None, start

        n_candidates = len(candidates)
        pos = np.asarray([c[0] for c in candidates], dtype=np.float32)
        quat = np.asarray([c[1] for c in candidates], dtype=np.float32)
        result = self.p.plan_pose(
            goal_tool_poses=self._goal(pos, quat),
            current_state=start,
            max_attempts=2,
            success_ratio=1.0,
        )
        (traj, end), = self._extract_batch(result, start, 1)
        if traj is None:
            self.last_candidate_diag = {
                "mode": "simple",
                "fail": "no_curobo_goalset_solution",
                "num_candidates": len(candidates),
            }
            return None, start

        cand_idx = 0
        if result is not None and result.goalset_index is not None:
            cand_idx = int(result.goalset_index.detach().view(-1)[0].item())
            cand_idx = max(0, min(cand_idx, n_candidates - 1))
        meta = candidates[cand_idx][2]
        wrist_ok, wr, wr_delta = self._wrist_delta_ok(end, start)
        fe = self._grasp_face_error(end, cube, meta.get("face_normal"))
        ok = (wrist_ok and FIXED_JAW_CLEAR_MIN <= fe["n"] <= FIXED_JAW_CLEAR_MAX
              and abs(fe["t"]) <= E_TANGENT_MAX and abs(fe["h"]) <= E_HEIGHT_MAX
              and fe["face_angle"] <= SIMPLE_FACE_LOSS_MAX_DEG)
        label = f"simple(face={meta.get('face_index')},rank={meta.get('face_rank')},alpha={meta['alpha_deg']:+.0f})"
        self._diag(f"[simple] selected_cand={label} wr={math.degrees(wr):.0f} "
                   f"wr_delta={math.degrees(wr_delta):+.1f} wrist_ok={wrist_ok} "
                   f"alpha={fe['a']:.0f} e_norm={fe['n'] * 1000:.1f} e_tan={fe['t'] * 1000:.1f} "
                   f"e_h={fe['h'] * 1000:.1f} centerline={fe['c'] * 1000:.1f}mm "
                   f"face_angle={fe['face_angle']:.1f} ok={ok}")
        if not ok:
            self.last_candidate_diag = {
                "mode": "simple",
                "fail": "selected_candidate_failed_fk_gate",
                "num_candidates": len(candidates),
                "candidate_index": int(cand_idx),
                "wrist_roll_deg": math.degrees(wr),
                "wrist_delta_deg": math.degrees(wr_delta),
                "fk_face_error": {k: float(v) for k, v in fe.items()},
            }
            return None, start

        score = (round(fe["c"] * 1000, 1),
                 round(abs(fe["n"] - 0.003) * 1000, 1),
                 round(meta["face_loss_deg"], 1),
                 round(abs(math.degrees(wr_delta))),
                 round(fe["a"]))
        diag = {**meta, "candidate_index": int(cand_idx), "score": list(score),
                "wrist_roll_deg": math.degrees(wr),
                "wrist_delta_deg": math.degrees(wr_delta),
                "fk_face_error": {k: float(v) for k, v in fe.items()}}
        diag["selection"] = {
            "policy": "goalset_topdown_closest_face_randomized_order",
            "num_candidates": len(candidates),
            "goalset_index": int(cand_idx),
            "seed": None if seed is None else int(seed),
        }
        self.last_candidate_diag = diag
        self._diag(f"[simple] selected={diag}")
        return traj, end

    def _plan_pregrasp_batch(self, cubes, yaws, starts, knobs):
        """N env simple 후보를 priority order로 검사한다.

        각 pass는 env별 후보 1개씩을 BatchMotionPlanner batch 차원으로 병렬 계획한다.
        후보 순서는 top-down/closest-face 우선이고 primary pool 내부만 seed 기반으로 섞는다.
        """
        n_env = len(cubes)
        kn = knobs or {}
        seed = kn.get("seed")
        face_count = int(kn.get("simple_face_count", SIMPLE_FACE_COUNT))
        start_pos, start_quat, _ = self._ee_pose_axis_batch(starts)
        per_env = []
        for i, (cube, yaw) in enumerate(zip(cubes, yaws)):
            start_i = JointState.from_position(
                starts.position[i: i + 1].detach().clone(), joint_names=self.p.joint_names)
            rng = np.random.default_rng(int(seed) + i) if seed is not None else self.rng
            cands = self._priority_shuffle_candidates(
                self._simple_candidates(cube, yaw, start_i, face_count=face_count), rng)
            if not cands:
                fallback = (start_pos[i], start_quat[i], {
                    "mode": "simple", "fail": "no_candidates", "env": i,
                })
                cands = [fallback]
            per_env.append(cands[:self.max_goalset])

        max_pass = max(len(c) for c in per_env)
        trajs = [None] * n_env
        ends = [JointState.from_position(starts.position[i: i + 1].detach().clone(),
                                         joint_names=self.p.joint_names)
                for i in range(n_env)]
        diagnostics = [{"mode": "simple", "fail": "not_attempted", "num_candidates": len(per_env[i])}
                       for i in range(n_env)]
        ok_mask = [False] * n_env
        plan_ms = 0.0
        for pass_idx in range(max_pass):
            active = [i for i in range(n_env) if not ok_mask[i] and pass_idx < len(per_env[i])]
            if not active:
                break
            pos = np.zeros((n_env, 3), dtype=np.float32)
            quat = np.zeros((n_env, 4), dtype=np.float32)
            pass_meta = []
            for i in range(n_env):
                if i in active:
                    p_i, q_i, m_i = per_env[i][pass_idx]
                else:
                    p_i, q_i, m_i = start_pos[i], start_quat[i], {"mode": "fallback"}
                pos[i] = p_i
                quat[i] = q_i
                pass_meta.append(m_i)
            t0 = time.perf_counter()
            result = self.p.plan_pose(
                goal_tool_poses=self._pose_batch(pos, quat),
                current_state=starts,
                max_attempts=2,
                success_ratio=1.0,
            )
            plan_ms += (time.perf_counter() - t0) * 1000.0
            planned = self._extract_batch(result, starts, n_env)
            for i in active:
                meta = pass_meta[i]
                traj, end = planned[i]
                if traj is None:
                    diagnostics[i] = {**meta, "fail": "no_curobo_solution",
                                      "candidate_index": pass_idx,
                                      "num_candidates": len(per_env[i])}
                    continue
                wrist_ok, wr, wr_delta = self._wrist_delta_ok(
                    end,
                    JointState.from_position(starts.position[i: i + 1].detach().clone(),
                                             joint_names=self.p.joint_names),
                )
                fe = self._grasp_face_error(end, cubes[i], meta.get("face_normal"))
                ok = (wrist_ok and FIXED_JAW_CLEAR_MIN <= fe["n"] <= FIXED_JAW_CLEAR_MAX
                      and abs(fe["t"]) <= E_TANGENT_MAX and abs(fe["h"]) <= E_HEIGHT_MAX
                      and fe["face_angle"] <= SIMPLE_FACE_LOSS_MAX_DEG)
                diag = {**meta,
                        "candidate_index": pass_idx,
                        "wrist_roll_deg": math.degrees(wr),
                        "wrist_delta_deg": math.degrees(wr_delta),
                        "fk_face_error": {k: float(v) for k, v in fe.items()},
                        "selection": {
                            "policy": "batched_priority_candidate_scan",
                            "num_candidates": len(per_env[i]),
                            "candidate_rank": int(pass_idx),
                            "seed": None if seed is None else int(seed) + i,
                        }}
                if ok:
                    trajs[i] = traj
                    ends[i] = end
                    diagnostics[i] = diag
                    ok_mask[i] = True
                else:
                    diag["fail"] = "candidate_failed_fk_gate"
                    diagnostics[i] = diag
            if all(ok_mask):
                break

        for i, ok in enumerate(ok_mask):
            diagnostics[i]["plan_ms"] = plan_ms
            if not ok and diagnostics[i].get("fail") == "not_attempted":
                diagnostics[i]["fail"] = "no_feasible_candidate"
        q_rows = [e.position.detach().view(1, -1) for e in ends]
        if starts.position.shape[0] > n_env:
            q_rows.append(starts.position[n_env:].detach())
        q_end = self._joint_state_batch(torch.cat(q_rows, dim=0))
        return trajs, q_end, diagnostics, ok_mask, {"approach_plan_ms": plan_ms, "candidate_passes": int(max_pass)}

    def _approach(self, cube, yaw, start, knobs):
        """① approach = simple 후보 batch 검증 → top-down 우선 랜덤 선택."""
        return self._plan_pregrasp(cube, yaw, start, knobs)

    # ── obstacle / attach helpers ───────────────────────────────────────────────
    def _place_cube_obstacle(self, cube, env_idx=0):
        """target 큐브 실좌표를 world "cube" obstacle 에 주입(placeholder 는 far)."""
        try:
            cpose = Pose(position=torch.tensor([list(cube)], device="cuda", dtype=torch.float32),
                         quaternion=torch.tensor([[1.0, 0, 0, 0]], device="cuda", dtype=torch.float32))
            self.p.scene_collision_checker.update_obstacle_pose("cube", cpose, env_idx=env_idx)
            self.p.scene_collision_checker.enable_obstacle("cube", True, env_idx=env_idx)
        except Exception as e:
            self._diag(f"[cube-obst] FAIL {type(e).__name__}: {e}")

    def _place_bowl_obstacle(self, bx, by, env_idx=0):
        """실 그릇 중심(urdf)을 8× rim-ring obstacle 에 주입 — 기본 pose 는 __init__ 값이라
        실제 그릇이 다른 곳(밀림·DR)이면 transit 이 엉뚱한 곳을 피함. 매 요청 동기화."""
        try:
            for name, ent in _bowl_ring(bx, by).items():
                x, y, z, qw, qx, qy, qz = ent["pose"]
                bpose = Pose(position=torch.tensor([[x, y, z]], device="cuda", dtype=torch.float32),
                             quaternion=torch.tensor([[qw, qx, qy, qz]], device="cuda", dtype=torch.float32))
                self.p.scene_collision_checker.update_obstacle_pose(name, bpose, env_idx=env_idx)
        except Exception as e:
            self._diag(f"[bowl-obst] FAIL {type(e).__name__}: {e}")

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

    def _detach(self, num_envs=1):
        """detach + stale "cube" obstacle 재-disable.

        cuRobo detach() 는 attach 때 disable한 world "cube" 를 **원래 pickup 좌표에** 재활성화
        하는데, release 후 큐브는 실제론 그릇 안 = pickup 자리는 빈 공간. pickup 이 base 정면
        (urdf x≈0.2)이면 이 유령 box 가 q_home jaw tip 과 margin 겹쳐 retreat 가
        "Start or End state in collision" 으로 전멸(bisect 실증: cube-off 만으로 retreat 회복).
        → 즉시 다시 disable. 다음 요청의 _place_cube_obstacle 이 재배치+재활성."""
        try:
            self._attachment_manager().detach()
            for env_idx in range(num_envs):
                self.p.scene_collision_checker.enable_obstacle("cube", False, env_idx=env_idx)
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

    def _start_states(self, starts, n):
        """요청 start 자세 N개 → JointState(N,dof)."""
        if starts is None:
            q = self.p.default_joint_state.position.unsqueeze(0).repeat(self.max_batch_size, 1)
            return JointState.from_position(q, joint_names=self.p.joint_names)
        lim = torch.tensor(ARM_LIMITS, device="cuda", dtype=torch.float32)
        rows = []
        for i in range(n):
            arm = torch.tensor(starts[i][: self.nA], device="cuda", dtype=torch.float32)
            rows.append(torch.clamp(arm, lim[:, 0] + 0.005, lim[:, 1] - 0.005))
        if n < self.max_batch_size:
            rows.extend([rows[0].clone() for _ in range(self.max_batch_size - n)])
        return JointState.from_position(torch.stack(rows, dim=0), joint_names=self.p.joint_names)

    def _merge_phase_ends(self, planned, fallback, prev_ok):
        """phase 결과에서 성공 env는 end, 실패 env는 fallback state를 유지."""
        rows, ok = [], []
        for i, (traj, end) in enumerate(planned):
            good = bool(prev_ok[i]) and traj is not None
            ok.append(good)
            src = end.position if good else fallback.position[i: i + 1]
            rows.append(src.detach().view(1, -1))
        if fallback.position.shape[0] > len(planned):
            rows.append(fallback.position[len(planned):].detach())
        return self._joint_state_batch(torch.cat(rows, dim=0)), ok

    @staticmethod
    def _cube_yaw(cube_bl):
        """6D payload([x,y,z,qw,qx,qy,qz]) → 큐브 옆면 face-normal 의 solver-frame azimuth(rad).

        ★큐브 USD 로컬 프레임은 rest 에 body축 하나가 **수직**(실측 eul pitch≈−84°, body-x z=0.994).
        naive z-yaw(atan2 공식)는 pitch±90 gimbal-lock 이라 near-vertical 축의 미세 XY 투영을 쫓아
        실제 face 서 ~28° 어긋난 쓰레기 → 가운데손가락 closing축이 큐브 대각 접촉(사용자 보고).
        → body 3축 중 **가장 수평인 축**(min|z|, = 옆면 normal)의 XY 방위를 쓴다. 큐브 face 4방(90°)
        이라 두 수평축은 mod 90 동일 → simple face-normal 후보가 나머지를 흡수."""
        if len(cube_bl) < 7:
            return None
        w, x, y, z = cube_bl[3:7]
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        c, s = math.cos(math.radians(BASE_YAW)), math.sin(math.radians(BASE_YAW))
        Rs = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ R  # base→solver 회전
        ax = Rs[:, int(np.argmin(np.abs(Rs[2, :])))]  # 가장 수평인 body축 = 옆면 normal
        return math.atan2(ax[1], ax[0])

    def _grasp_diag(self, cube_bl, aq, zax, yaw, pre_ok):
        """달성 자세 grip 진단: α(접근 기울기)·Δψ(face 정렬오차)."""
        w, x, y, z = aq
        xcol = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)])
        al = math.degrees(math.acos(max(-1.0, min(1.0, -float(zax[2])))))
        azim = math.atan2(xcol[1], xcol[0])
        dps = "-" if yaw is None else round(
            math.degrees((yaw - azim + math.pi / 4) % (math.pi / 2) - math.pi / 4), 1)
        self._diag(f"[graspdiag] cube=({cube_bl[0]:.3f},{cube_bl[1]:.3f}) alpha={al:.1f} "
                   f"dpsi={dps} pre_ok={pre_ok}")

    # ── main entry ──────────────────────────────────────────────────────────────
    def plan_pickplace_batch(self, cube_bls, bowl_bls=None, starts_rad=None, knobs=None):
        """N개 env full pick-place를 BatchMotionPlanner 1개로 병렬 계획."""
        n_env = len(cube_bls)
        self._ensure_batch_size(n_env)
        kn = knobs or {}
        z_off = float(kn.get("grasp_z_off", GRASP_Z_OFF))
        g_open = float(kn.get("grip_open", GRIP_OPEN))
        g_close = float(kn.get("grip_close", GRIP_CLOSE))
        b_pull = float(kn.get("bowl_pull", BOWL_PULL))
        t_all0 = time.perf_counter()
        profile = {}
        self.p.reset_seed()

        cubes = []
        yaws = []
        bowls = []
        per_env_bowl = bool(bowl_bls) and isinstance(bowl_bls[0], (list, tuple))
        for i, cube_bl in enumerate(cube_bls):
            cube = usd_to_urdf(cube_bl[:3])
            cube = (cube[0], cube[1], cube[2] + z_off)
            cubes.append(cube)
            yaws.append(self._cube_yaw(cube_bl))
            bsrc = bowl_bls[i] if per_env_bowl else bowl_bls
            bx, by = usd_to_urdf((bsrc[0], bsrc[1], 0.0))[:2] if bsrc is not None else self.bowl_s
            bowls.append((bx, by))

        starts = self._start_states(starts_rad, n_env)
        q_home = starts.clone()
        diagnostics = []
        for i, cube_bl in enumerate(cube_bls):
            diagnostics.append({
                "ok": False,
                "cube_base_link": [float(v) for v in cube_bl],
                "cube_solver": [float(v) for v in cubes[i]],
                "cube_yaw_deg": None if yaws[i] is None else float(math.degrees(yaws[i])),
                "bowl_solver_xy": [float(bowls[i][0]), float(bowls[i][1])],
                "candidate": None,
                "phases": {},
                "profile_ms": {},
            })
            self._place_cube_obstacle(cubes[i], env_idx=i)
            self._place_bowl_obstacle(bowls[i][0], bowls[i][1], env_idx=i)
        for env_idx in range(n_env, self.max_batch_size):
            try:
                self.p.scene_collision_checker.enable_obstacle("cube", False, env_idx=env_idx)
            except Exception:
                pass

        # ① APPROACH: batch=N, goalset=candidates.
        pre, q_pre, cand_diag, pre_ok, pre_prof = self._plan_pregrasp_batch(cubes, yaws, starts, kn)
        profile.update(pre_prof)

        # ② GRASP descend.
        app, aq, zaxes = self._ee_pose_axis_batch(q_pre)
        gpos, tstars = [], []
        for i, (cube, zax) in enumerate(zip(cubes, zaxes)):
            tstar = _pre_back(cube)
            zaz = float(zax[2])
            if zaz < -1e-3:
                tstar_cap = (TABLE_TOP + TABLE_MARGIN - float(app[i, 2])) / zaz - PAD_LOW_OFF
                tstar = min(tstar, tstar_cap)
            tstars.append(tstar)
            gpos.append(app[i] + tstar * zax)
        t0 = time.perf_counter()
        desc_planned = self._plan_to_batch(self._pose_batch(np.asarray(gpos), aq), q_pre, n_env, linear=True)
        profile["grasp_plan_ms"] = (time.perf_counter() - t0) * 1000.0
        q_grasp, grasp_ok = self._merge_phase_ends(desc_planned, q_pre, pre_ok)

        # ③ LIFT.
        up = np.asarray([gpos[i] - min(tstars[i], LIFT_BACK) * zaxes[i] for i in range(n_env)])
        t0 = time.perf_counter()
        lift_planned = self._plan_to_batch(self._pose_batch(up, aq), q_grasp, n_env, linear=True)
        profile["lift_plan_ms"] = (time.perf_counter() - t0) * 1000.0
        q_lift, lift_ok = self._merge_phase_ends(lift_planned, q_grasp, grasp_ok)

        attached = False
        if any(lift_ok):
            attached = self._attach_cube(q_lift)

        # ④ TRANSIT: env별 bowl 상공 FK-bank goalset.
        fk_pos, fk_quat, _ = self._ee_pose_axis_batch(q_lift)
        tr_pos = np.zeros((n_env, K, 3), dtype=np.float32)
        tr_quat = np.zeros((n_env, K, 4), dtype=np.float32)
        transit_z = TRANSIT_Z + BASE_T[2]
        for i, (bx, by) in enumerate(bowls):
            if not lift_ok[i]:
                tr_pos[i] = np.repeat(fk_pos[i: i + 1], K, axis=0)
                tr_quat[i] = np.repeat(fk_quat[i: i + 1], K, axis=0)
                continue
            bd = math.hypot(bx, by)
            s = 1.0 - b_pull / bd if bd > 1e-6 else 1.0
            px, py = bx * s, by * s
            idx = np.argsort(np.linalg.norm(self.FK_POS_ALL - np.array((px, py, transit_z)), axis=1))[:K]
            tr_pos[i] = self.FK_POS_ALL[idx]
            tr_quat[i] = self.FK_QUAT_ALL[idx]
        t0 = time.perf_counter()
        transit_planned = self._plan_to_batch(self._goalset_batch(tr_pos, tr_quat), q_lift, n_env)
        profile["transit_plan_ms"] = (time.perf_counter() - t0) * 1000.0
        q_transit, transit_ok = self._merge_phase_ends(transit_planned, q_lift, lift_ok)

        if attached:
            self._detach(num_envs=self.max_batch_size)

        # ⑥ RETREAT.
        t0 = time.perf_counter()
        retreat_planned = self._plan_cspace_batch(q_home, q_transit, n_env)
        profile["retreat_plan_ms"] = (time.perf_counter() - t0) * 1000.0
        retreat_ok = [bool(transit_ok[i]) and retreat_planned[i][0] is not None for i in range(n_env)]
        profile["total_plan_ms"] = (time.perf_counter() - t_all0) * 1000.0

        trajectories = []
        for i in range(n_env):
            phases = {
                "approach": pre[i],
                "grasp": desc_planned[i][0],
                "lift": lift_planned[i][0],
                "transit": transit_planned[i][0],
                "retreat": retreat_planned[i][0],
            }
            diagnostics[i]["candidate"] = cand_diag[i]
            diagnostics[i]["phases"] = {k: v is not None for k, v in phases.items()}
            diagnostics[i]["attached"] = bool(attached)
            diagnostics[i]["profile_ms"] = {k: float(v) for k, v in profile.items()}
            ok = all(v is not None for v in phases.values()) and bool(retreat_ok[i])
            diagnostics[i]["ok"] = ok
            if ok:
                trajectories.append(self._assemble(phases, g_open, g_close))
            else:
                trajectories.append(None)
        self.last_plan_diag = diagnostics[0] if diagnostics else {}
        self.last_candidate_diag = cand_diag[0] if cand_diag else None
        return trajectories, diagnostics

    def plan_pickplace(self, cube_bl, bowl_bl=None, start_rad=None, knobs=None):
        """단일 env 호환 wrapper."""
        if start_rad is None:
            starts = None
        elif start_rad and isinstance(start_rad[0], (list, tuple)):
            starts = [start_rad[0]]
        else:
            starts = [start_rad]
        trajs, diagnostics = self.plan_pickplace_batch([cube_bl], [bowl_bl] if bowl_bl is not None else None,
                                                       starts, knobs)
        self.last_plan_diag = diagnostics[0]
        self.last_candidate_diag = diagnostics[0].get("candidate")
        return trajs[0]

    def _assemble(self, phases, g_open, g_close):
        """5 phase 궤적(rad) → 단일 (T,6) 시퀀스[arm deg + gripper feature].
        gripper 폐합(grasp)·개방(release=transit 상공) ramp 을 정지 hold 에 삽입."""
        a, de, li, tr, rt = (np.rad2deg(phases[k]) for k in
                             ("approach", "grasp", "lift", "transit", "retreat"))
        close_hold = np.repeat(de[-1:], CLOSE_STEPS, 0)   # grasp: 정지 상태서 폐합
        settle_hold = np.repeat(tr[-1:], SETTLE_STEPS, 0)  # 그릇 상공서 짧게 정지(안정)
        open_hold = np.repeat(tr[-1:], OPEN_STEPS, 0)     # release: 그릇 상공(transit)서 개방
        seq = [
            _grip(a, np.linspace(GRIP_INIT, g_open, len(a))),        # ① approach + gripper 개방(접근하며)
            _grip(de, g_open),                                       # ② grasp descend
            _grip(close_hold, np.linspace(g_open, g_close, CLOSE_STEPS)),  #   grasp close
            _grip(li, g_close),                                      # ③ lift
            _grip(tr, g_close),                                      # ④ transit
            _grip(settle_hold, g_close),                            #   settle over bowl (hold)
            _grip(open_hold, np.linspace(g_close, g_open, OPEN_STEPS)),    # ⑤ release over bowl
            _grip(rt, np.linspace(g_open, GRIP_INIT, len(rt))),      # ⑥ retreat + gripper→init 복원
        ]
        return np.vstack(seq).astype(np.float32)


def plan_batch(pl, req):
    """plan_pickplace 요청 1건 → per-env 궤적 리스트(list|None ×N).

    BatchMotionPlanner batch 차원으로 N env를 병렬 계획한다. bowl 은 [x,y] 공용 또는
    [[x,y]×N] per-env, start 는 [[6]×N] 또는 생략."""
    cubes = req["cubes"]
    bowl = req.get("bowl")
    starts = req.get("start")
    knobs = req.get("knobs")
    trajs, diagnostics = pl.plan_pickplace_batch(cubes, bowl, starts, dict(knobs or {}))
    out = [traj.tolist() if traj is not None else None for traj in trajs]
    return out, diagnostics


def serve_loop(pl, sock):
    while True:
        req = json.loads(sock.recv())
        cmd = req.get("cmd")
        if cmd == "ping":
            sock.send_string(json.dumps({"ok": True}))
        elif cmd == "plan_pickplace":
            trajs, diagnostics = plan_batch(pl, req)
            sock.send_string(json.dumps({"ok": True, "trajectories": trajs, "diagnostics": diagnostics}))
        elif cmd == "shutdown":
            sock.send_string(json.dumps({"ok": True})); return
        else:
            sock.send_string(json.dumps({"ok": False, "err": f"unknown {cmd!r}"}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5599)
    ap.add_argument("--max_batch_size", type=int, default=64,
                    help="cuRobo BatchMotionPlanner max_batch_size; batch dimension maps to env index.")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    pl = PickPlacePlanner(max_batch_size=a.max_batch_size)
    print("[planner] ready", flush=True)
    if a.self_test:
        q_identity = [1.0, 0.0, 0.0, 0.0]
        init = [math.radians(v) for v in (0.0, -100.0, 90.0, 50.0, -90.0, -10.0)]
        trajs, diagnostics = plan_batch(pl, {
            "cubes": [[0.017, -0.253, 0.06, *q_identity],
                      [0.167, -0.133, 0.06, *q_identity]],
            "start": [init, init],
        })
        ok = all(t is not None for t in trajs)
        for i, d in enumerate(diagnostics):
            print(f"self-test env{i}: ok={d.get('ok')} phases={d.get('phases')} "
                  f"candidate_fail={d.get('candidate', {}).get('fail')} "
                  f"profile_ms={d.get('profile_ms')}")
        print(f"self-test(2-env): {[len(t) if t else None for t in trajs]}")
        print("SELFTEST_OK" if ok else "SELFTEST_CHECK")
        return
    sock = zmq.Context().socket(zmq.REP); sock.bind(f"tcp://*:{a.port}")
    print(f"[planner] ZMQ REP :{a.port}", flush=True)
    serve_loop(pl, sock)


if __name__ == "__main__":
    main()
