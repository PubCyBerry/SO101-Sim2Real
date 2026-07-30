"""cuRobo pick-place ZMQ planner 서비스 (SO-101, 신 API v0.8).

★2-프로세스: 이 planner(so101-curobo-datagen, cuRobo+warp) ↔ Isaac executor
(so101-isaac-sim) ZMQ 분리(in-process 불가=warp ABI). executor 가 큐브(base_link)를 보내면
planner 가 collision-aware full pick-place 궤적을 반환한다.

★multi-env: 요청의 `cubes` 리스트 길이 = env 수. executor 는 N envs 를 IsaacLab native
lockstep 으로 병렬 replay 한다. planner 도 cuRobo `BatchMotionPlanner(multi_env=True)`의
batch 차원을 env 차원으로 맞춰 N개 env를 한 번에 푼다. grasp 후보는 env별 goalset 차원이다.

계획: tool frame = **tcp_grasp**(손가락 사이 pinch 점, so101.yml extra_link).
★5-DOF 원칙: grasp 후보는 도달 manifold 파라미터화 R=Rz(pan)·Ry(-α)·R_TOPDOWN·Rz(ρ)·TCP_TWIST
로 직접 생성한다(pink_ik_bridge_node §4·§6 이식). ρ=-Δψ/cosα 가 closing 축을 cube yaw face 에
정렬, 결합 게이트 |Δψ·tanα|≤τ. transit은 별도 reachability goalset(FK bank pose).

6-phase(approach·grasp·lift·transit·release·retreat):
  ① approach = manifold (pan,α,ρ) 후보를 |α| 오름차순 lockstep 검증(top-down 우선)
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

프로토콜(JSON REQ/REP, port 5599) — cube 는 base_link frame, quat 에서 face normal 직접 추출:
  {"cmd":"ping"}                                          → {"ok":true}
  {"cmd":"plan_pickplace",
   "cubes":[[x,y,z(,qw,qx,qy,qz)]×N],                     → {"ok":true,
   "bowl":[x,y] | [[x,y]×N],                                  "trajectories":[[[6]×T]|null ×N]}
   "start":[[6 joint rad]×N] (선택), "knobs":{...}(선택)}
  {"cmd":"shutdown"}                                      → {"ok":true}

실행: /isaac-sim/python.sh curobo_batch_planner.py [--port 5599] [--self-test]
"""
import argparse
import itertools
import json
import math
import tempfile
import time
import traceback

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

# ═══ 로봇/큐브 기하 (실측 단일 소스) ═══════════════════════════════════════════════
PAN_AXIS_XY = (0.0388353, 0.0)  # shoulder_pan 축의 solver-frame XY — face 선택 기준점(URDF)
# fixed jaw pad 기하 = so101_contract.grasp_geometry 단일 소스(SM 진단 로그와 같은 값 공유).
# 두 컨테이너 모두 PYTHONPATH=/workspace/src (Dockerfile.isaac_sim, cuRobo 이미지가 상속).
from so101_contract.grasp_geometry import FIXED_INNER_CENTER, PAD_LOW_OFF, TABLE_TOP_BASE  # noqa: E402
from so101_contract.feature_codec import POLICY_GRIPPER_RANGE, SIM_GRIPPER_RANGE_DEG  # noqa: E402
# 책상 상판 z (urdf 프레임) — base_link 실측 단일 소스에서 파생. descend clamp 가 쓴다.
TABLE_TOP = TABLE_TOP_BASE + BASE_T[2]
# 큐브 반변 — face_center = cube_center + half·closing_axis.
# ★실제 값은 요청의 `cube_half` 로 온다(SM 이 cube_specs 단일 소스에서 읽어 pose 와 함께 전송).
# 크기 DR 이 켜지면 **env 마다 다르므로 리스트**로 오고, 스칼라/누락은 하위호환 폴백이다.
CUBE_HALF = 0.020
# 큐브 obstacle/attach blob 한 변(m) — **크기 DR 상한**(cube_specs CUBE_SIZE_CHOICES max
# = authored 40 mm) 고정. collision 은 과대근사가 안전측이고, world obstacle dims 는 planner
# 초기화 시 굳어 요청마다 못 바꾼다 → 상한을 쓴다(25 mm 큐브면 뚱뚱한 박스로 계획).
# 예전 0.05 는 50 mm 큐브 씬 유물이라 25 mm 상대로 2배까지 벌어졌다.
CUBE_DIMS = 0.04
CONTACT_LINKS = ["gripper_link", "moving_jaw_so101_v1_link"]   # descend 중 collision off
DESCEND_EXTRA_OFF = ["wrist_link", "wrist_cam_mount_link"]     # 〃 — grasp 자세서 wrist sphere 가
                      # 큐브 obstacle 과 모델상 겹침(짧은 wrist, 물리 접촉 없음=sphere 보수 근사)
# so_arm101.urdf arm 관절 limit(rad) — start clamp 용(USD ±105° 와 어긋나는 5° 캘리브 마진)
ARM_LIMITS = [(-1.91986, 1.91986), (-1.74533, 1.74533), (-1.69, 1.69),
              (-1.65806, 1.65806), (-2.74385, 2.84121)]

# ═══ grasp 후보 = 5-DOF 도달 manifold 파라미터화 (pan, α, ρ) ═══════════════════════
# SO-101 체인 = pan(z-yaw)+3×pitch(평행축)+wrist_roll(tool-z) → tool 접근축은 항상 pan
# 수직평면 안(pitch 축이 평면 법선과 평행, roll 은 접근축 불변). 도달 가능 orientation 은
#   R = Rz(pan)·Ry(-α)·R_TOPDOWN·Rz(ρ)·TCP_TWIST
# 3-파라미터 manifold 가 전부다(scripts/datagen/pink_ik_bridge_node.py §4·§6 이식 — pink 는
# ROS/pinocchio 컨테이너라 import 불가, 수식 복사). world 고정 face-normal full pose(구판)는
# face 가 pan 평면과 어긋난 만큼 manifold 밖 → IK 전멸(실측 825 중 26 solve·gate 0)의 근본원인.
# - Δψ = wrap90(ψ_face - (pan+90°)) : 큐브 yaw face 오차(정사각 90° 대칭 → [-45°,45°))
# - ρ = -Δψ/cosα : closing 축 수평투영을 face normal 에 정렬(τ 내 잔차 <1°)
# - |Δψ·tanα| ≤ τ : closing 축 수평이탈각 결합 게이트
R_TOPDOWN = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
# 캐노니컬 top-down(pan=0): x̂=+y(tangential=closing), ŷ=+x(radial), ẑ=-z(접근·하향).
# TCP twist: tcp_grasp = gripper_link·Ry(π-0.0486795)(so101.yml) → tcp ẑ 가 wrist_roll 축과
# 2.79° 어긋나 ρ 에 따라 원뿔운동. trailing Ry(-0.0486795) 미포함 시 후보가 2.79° off-manifold
# (cuRobo 회전 수렴 예산 ~0.05rad 소진 + pad 조준 ~2.4mm 편향) → 상수로 bake.
TCP_TWIST_RY = -0.0486795
# |α| 오름차순 ± interleave = 검사 우선순위(top-down 우선). +α=접근축이 base 반대로 기울어
# wrist 를 base 쪽으로 당김(원거리 2R 해소), -α=반대(근거리 최소반경 해소).
ALPHA_SCAN_DEG = [0.0, 5.0, -5.0, 10.0, -10.0, 15.0, -15.0, 20.0, -20.0, 25.0, -25.0,
                  30.0, -30.0, 35.0, -35.0, 40.0, -40.0, 45.0, -45.0, 50.0, -50.0]
# 결합 게이트 |Δψ·tanα| 상한(knobs.tau_max_deg 로 조정). 전 도달구간 상수 —
# 옛 reach-adaptive 램프(10°→25°)의 최대치를 그대로 상수화했다(_manifold_candidates 주석 참조).
TAU_MAX_DEG = 25.0
# ★worst-yaw wrist-cap: |Δψ| 큰 셀서 auto ρ=-Δψ/cosα 가 |ρ| 크면 wrist_roll 을 위험대로 밀어
# grasp 실패(sm-eval 물리: ρ-10→wrist+4°=PASS 117mm · ρ-20→+13°=FAIL · ρ-41→+33°=FAIL).
# 54-sphere 모델 targeted replay(8건): cap 12/14/16/18° = 5/6/7/8 성공. 다만 18°는
# 64-env 첫 planning만 약 17분이 걸려 기본값으로 부적합했다. 기본 12°를 유지하고
# `rho_cap_deg` knob으로만 A/B한다. diagonal grasp miss는 아래 chord-center 로 보완한다.
# 정상(|auto ρ|≤이값) 셀 무영향 = face 정렬 보존.
RHO_CAP_DEG = 12.0
RHO_CAP_RAD = math.radians(RHO_CAP_DEG)
# rho cap 셀의 face-center chord miss 보정. baseline 실패 8건을 reset/plan seed 0/1/2로
# 반복한 결과 ratio=0은 5/8, ratio=0.5는 24/24 성공. 완전 보정보다 여유를 둔 반 보정을 사용.
CHORD_CENTER_RATIO = 0.5
# ρ+π 미러 branch 는 생성 안 함: τ 게이트 하 |ρ| ≤ 45°/cos50° ≈ 70° < 100° 라 기본 branch 가
# 항상 wrist gate 통과, 미러는 항상 탈락(Δ≥110° + 사용자 금지 이력 wrist ~223° 뒤집기).
# per-pass 1후보=1 plan_pose 라 보장된 탈락 후보는 latency 만 2배.
# pan/ρ 고정점 반복 횟수(실측 튜닝: 3회면 ~0.8° 잔차, 5회면 근거리도 수렴).
# 등거리 face(yaw 45°) 셀은 face 선택이 번갈리는 limit cycle 이라 수렴 자체가 불가능하다 —
# 그래서 수렴을 요구하지 않고 잔차만 meta(pan_resid_deg·closing_resid_deg)로 노출한다.
PAN_FIXPOINT_ITER = 5
# ★후보 채택 정책: 게이트를 **처음** 통과한 후보가 아니라, 앞쪽 이만큼의 pass 를 모두 검사한
# 뒤 `_candidate_score` 가 가장 낮은 후보를 쓴다(2026-07-30).
#   옛 동작: |α| 오름차순으로 돌다 첫 통과 즉시 채택 — score 는 계산해 diag 에 싣기만 했다.
#   게이트 창이 넓어서(e_norm 2~8 mm · |e_t| ≤22 mm) "통과했지만 조준이 나쁜" 후보가 그대로
#   쓰였고, 드물게 폐합 때 jaw 가 큐브를 밀어내 grasp 가 실패했다(측정 1/496, 큐브가 48 mm
#   밀린 채 +3 mm 만 들림). 근거·측정 = `09_TACIT_KNOWLEDGE.md §15.5`.
#   통과한 env 도 이 pass 수까지는 batch slot 을 계속 쓰므로 plan 호출은 늘지 않는다
#   (어차피 미통과 env 때문에 pass 를 더 돈다).
CANDIDATE_SCAN_PASSES = 3
SIMPLE_FACE_GATE_MAX_DEG = 40.0  # FK gate 안전망: solver XY face_angle 허용 절댓값
WRIST_ROLL_DELTA_LIMIT_DEG = 100.0
# 게이트(IK 성공만으론 불충분 — IK-후-FK 실측 pad center 를 face center 와 3D 비교):
#   e_normal(closing clearance) = 3~5mm 범위를 목표로 한다.
#   fixed_inner 는 pad centroid proxy 라서 IK/FK 잔차를 감안해도 face 를 긁지 않게 양의 여유를 둔다.
# tangent/height 폭 = fingers-down branch 실측 도달 범위(40mm 큐브 kinematic 한계) — manifold
# 후보로 통과율 확보 후 조임은 별도 튜닝.
FIXED_JAW_CLEAR_TARGET = 0.004  # pad center proxy 조준 clearance(4mm 부근)
FIXED_JAW_CLEAR_MIN, FIXED_JAW_CLEAR_MAX = 0.002, 0.008  # R3: 3-5mm→2-8mm 창 완화(D3, IK undershoot 수용)
E_TANGENT_MAX = 0.022
E_HEIGHT_MAX = 0.028
WRIST_ROLL_DELTA_LIMIT = math.radians(WRIST_ROLL_DELTA_LIMIT_DEG)

# ═══ phase 파라미터 ═══════════════════════════════════════════════════════════════
K = 40              # goalset 크기(bank reach)
# grasp 깊이 미세보정(m). 요청이 실어 보내는 큐브 z 는 2026-07-29 부터 실측 정합값
# (TABLE_TOP_BASE + half)이라, 이 offset 은 **중심 대비 조준 높이**만 뜻한다.
# +2.2 mm = 예전 조합(--grasp_z 0.060 과대 + 이 값 −8 mm)이 만들던 solver 조준점을 그대로
# 보존한 값이다 — 프레임 장부만 고치고 물리 working point 는 건드리지 않았다.
GRASP_Z_OFF = 0.0022
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
# 폐합 직후 정지: 접촉 안정화 후 lift. ★2026-07-30 5 → 15.
# gripper ramp 가 slew cap 에 맞춰 17 프레임으로 늘어난 뒤(§15), 마지막 프레임에야 목표
# 각도에 닿는다 — hold 5(0.17 s)는 PhysX 접촉력이 정착하기 전에 lift 를 시작시켰다.
# 잔여 실패 1건이 정확히 그 모습(큐브를 16 mm 들다 놓침 = slip, 밀어냄과 구분되는 유형)이라
# 정착 시간을 0.5 s 로 늘린다. 궤적 기하는 불변, 에피소드만 10 프레임(0.33 s) 길어진다.
GRASP_HOLD_STEPS = 15
SETTLE_STEPS = 5    # ⑤ release 전 그릇 상공 정지 hold 프레임

# ── gripper ramp 길이는 env 의 slew cap 에서 **유도**한다(하드코딩 금지) ──────────
# feature [0,100] → sim gripper rad 기울기. 단일 소스 = so101_contract.feature_codec.
_GRIP_RAD_PER_FEATURE = (
    math.radians(SIM_GRIPPER_RANGE_DEG[1] - SIM_GRIPPER_RANGE_DEG[0])
    / (POLICY_GRIPPER_RANGE[1] - POLICY_GRIPPER_RANGE[0]))
# env 가 gripper 명령에 거는 slew 상한(rad/s)과 제어 주파수.
#   = pick_cube_env_cfg._PICKCUBE_JOINT_MAX_VELOCITY["gripper"], 1/(sim.dt*decimation)
GRIPPER_SLEW_MAX_RAD_S = 2.5
CONTROL_HZ = 30.0
GRIP_RAMP_MIN_STEPS = 5   # 아주 짧은 Δ 에서도 이 정도는 나눠 보낸다


def grip_ramp_steps(g_from, g_to):
    """gripper feature 이동을 **slew cap 안에서** 소화하는 데 필요한 프레임 수.

    ★2026-07-29: 옛 상수(CLOSE_STEPS=5 · OPEN_STEPS=10)는 명령 속도 8.06 / 4.03 rad/s 로
    env cap(2.5)을 3.2×/1.6× 넘겼다. 명령이 cap 을 넘으면 SlewLimited action 이 잘라내므로
    **정해진 프레임 안에 폐합이 끝나지 않는다** — grasp 는 close(5)+hold(5)=10 프레임 예산에
    1.344 rad 중 0.833 rad 만 진행한 채 lift 로 넘어갔고, 남은 29° 는 팔이 올라가는 중에
    닫혔다. 큐브를 다 물기 전에 드는 셈이라 드물게(측정 3/2232 ≈ 0.13 %) jaw 가 큐브를
    밀어내고 grasp 가 실패했다(sweep 진단: max_cube_z 가 안착면 +5 mm 에서 멈춤).
    release 도 같은 구조 — 다 열리기 전에 retreat 이 시작되면 큐브를 끌고 나온다.

    ramp 를 cap 이내로 맞추면 **실제 폐합 속도는 그대로**(어차피 cap 이 지배)이고 lift/retreat
    타이밍만 폐합/개방 완료 뒤로 밀린다 = 순수 안정화. 기록되는 action 도 물리적으로 실현
    가능한 값이 된다(sim2real 데이터 품질).
    """
    delta_rad = abs(float(g_from) - float(g_to)) * _GRIP_RAD_PER_FEATURE
    return max(GRIP_RAMP_MIN_STEPS,
               int(math.ceil(delta_rad / (GRIPPER_SLEW_MAX_RAD_S / CONTROL_HZ))))

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


def _assert_pre_back_clears_cube(max_cube_half=0.025, clear=0.005):
    """로드시 1-check: 최소 pre-back(PRE_BACK_MIN)이 **jaw tip 을 큐브 위로 띄우는가**.

    ``_pre_back`` 의 r-램프는 도달거리별 planning 여유를 담은 실측 튜닝이지만, 그 존재 이유는
    "pre-grasp 자세에서 fixed jaw tip 이 큐브 obstacle 위에 있어야 approach 를 jaw-collision ON
    으로 계획할 수 있다"는 기하 조건이다. 그 조건은 **큐브 크기에 의존**하는데 램프 공식에는
    큐브 크기가 안 들어간다 → 큐브를 키우면 조용히 조건이 깨진다.

    pre 자세 jaw tip 고도 = tcp_tgt_z + (t − PAD_LOW_OFF)·cosα, tcp_tgt_z − cube_top
    = FIXED_INNER_CENTER[2] − half (조준식에서 pad center 가 face center 높이에 오므로).
    ⇒ 필요한 t ≥ PAD_LOW_OFF − (FIXED_INNER_CENTER[2] − half − clear)/cosα.
    cosα ≤ 1 이라 α=0(top-down)이 최악 — 그 값으로 검사한다.

    ponytail: 램프를 이 유도식으로 **대체**하지 않는다. 실측상 필요치(≈59 mm @50 mm 큐브)가
    PRE_BACK_MIN(60 mm)보다 작아 램프가 이미 조건을 만족하고, 램프는 그 위에 원거리 planning
    여유까지 담고 있어 교체하면 근거 없는 회귀 위험만 산다. 조건이 깨지면 여기서 터뜨린다.
    """
    need = PAD_LOW_OFF - (FIXED_INNER_CENTER[2] - max_cube_half - clear)
    assert PRE_BACK_MIN >= need, (
        f"PRE_BACK_MIN {PRE_BACK_MIN:.3f}m < jaw-tip clearance 필요치 {need:.3f}m "
        f"(cube half {max_cube_half}, clear {clear}) — pre-grasp 서 jaw 가 큐브에 박힌다")


_assert_pre_back_clears_cube()


def _descend_tstar(pre_tcp_z, zaz, cube):
    """pre-grasp → grasp 하강 거리(approach 축 t, m). table clamp 포함.

    ★게이트(``_grasp_geometry``)와 실행(``plan_pickplace_batch`` ②)이 **같은 값**을 써야 한다 —
    갈리는 순간 FK 게이트는 실제로 실행되지 않을 자세를 통과시킨다. 그래서 공식은 이 함수 하나뿐.
    table 은 world obstacle 이 아니라(로봇이 상판 위에 장착 → 전 plan start-collision) 이 clamp 가
    대신한다: pad 최저점(tcp + PAD_LOW_OFF·ẑ)이 TABLE_TOP+TABLE_MARGIN 아래로 못 내려간다.
    """
    tstar = _pre_back(cube)
    if zaz < -1e-3:  # 하강 중일 때만 clamp 의미 있음
        tstar = min(tstar, (TABLE_TOP + TABLE_MARGIN - float(pre_tcp_z)) / zaz - PAD_LOW_OFF)
    return tstar


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


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def wrap90(a):
    """각도 → [-45°, 45°) wrap — 정사각 큐브 90° 대칭(pink §4.2)."""
    return (a + math.pi / 4.0) % (math.pi / 2.0) - math.pi / 4.0


TCP_TWIST = _ry(TCP_TWIST_RY)  # tcp_grasp ↔ wrist_roll 축 2.79° 원뿔 보정(상수 블록 참조)


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


def _quat_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("zero quaternion")
    return q / n


def _quat_mul(a, b):
    """Hamilton product, wxyz."""
    aw, ax, ay, az = np.asarray(a, dtype=np.float64)
    bw, bx, by, bz = np.asarray(b, dtype=np.float64)
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def _quat_conj(q):
    q = _quat_normalize(q)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_rotate(q, v):
    """Quaternion으로 3D 벡터를 직접 회전한다. Euler/yaw 중간 표현을 만들지 않는다."""
    qn = _quat_normalize(q)
    vw = np.array([0.0, *np.asarray(v, dtype=np.float64)], dtype=np.float64)
    return _quat_mul(_quat_mul(qn, vw), _quat_conj(qn))[1:]


def _quat_z(theta):
    return np.array([math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0)], dtype=np.float64)


def _quat_to_euler_xyz_deg(q):
    """wxyz quaternion → XYZ/RPY deg. 진단 출력용; planning 계산에는 쓰지 않는다."""
    w, x, y, z = _quat_normalize(q)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


def _grip(arm_deg, grip):
    """arm-deg 궤적에 gripper 열(scalar=상수 · array=per-step ramp)을 붙여 (T,6) 로."""
    g = np.full((len(arm_deg), 1), grip) if np.isscalar(grip) else np.asarray(grip).reshape(-1, 1)
    return np.hstack([arm_deg, g])


def cand_pose_manifold(xyz, faces, alpha_deg, tau, rho_cap_rad=RHO_CAP_RAD,
                       chord_center_ratio=CHORD_CENTER_RATIO, cube_half=CUBE_HALF):
    """(pan,α,ρ) manifold 위 full TCP pose 1개 — 구성상 5-DOF 도달 가능(상수 블록 §manifold).

    ψ_face = 수평 face normal 방위(90° 대칭이라 어느 face 든 Δψ 동일 → faces[0] 사용).
    pan 고정점 ×3: TCP lateral offset(R·FIXED_INNER_CENTER)이 ρ 와 함께 돌아 tcp 목표가
    pan 평면을 벗어나는 것을 목표 방위로 재정렬(pink select_grasp l.347-358 이식).
    fixed jaw 가 놓일 face n̂ = closing 축(R x̂) 최근접 내적 — ρ 보상 후 자동 결정.
    τ 초과(|Δψ·tanα| = closing 수평이탈) 또는 face 부재 시 None.
    returns (pre_pos, quat_wxyz, meta) — pre_pos = tcp 목표서 approach 축 _pre_back 후퇴."""
    if not faces:
        return None
    cc = np.array(xyz[:3], dtype=np.float64)
    a = math.radians(float(alpha_deg))
    n0 = faces[0][1]
    psi = math.atan2(n0[1], n0[0])
    pan = math.atan2(cc[1] - PAN_AXIS_XY[1], cc[0] - PAN_AXIS_XY[0])
    fic = np.array(FIXED_INNER_CENTER, dtype=np.float64)
    rho_corr = 0.0
    # 고정점 반복: 근거리(r≈0.10)는 tcp lateral offset 비중이 커 pan 이 감쇠진동(~0.3/iter)한다.
    # 반복 횟수는 실측 튜닝값 고정(PAN_FIXPOINT_ITER=5; 3회면 ~0.8° 잔차).
    # ★수렴을 **강제하지 않는다**: cube yaw 45° 처럼 두 face 가 등거리인 셀은 face 선택이
    #   매 반복 번갈려 limit cycle 이 된다(수렴 실패 시 후보를 버리게 했더니 근거리·yaw45 셀
    #   후보가 전멸 — self_check_geom 이 잡았다). 대신 **잔차를 meta 에 남겨** 조용한
    #   off-manifold 를 관측 가능하게만 만든다. 품질 판정은 FK 게이트가 독립적으로 한다.
    d_pan = 0.0
    resid = 0.0
    for _ in range(PAN_FIXPOINT_ITER):
        pan_prev = pan
        dpsi = wrap90(psi - (pan + math.pi / 2.0))
        raw_rho = -dpsi / math.cos(a)
        capped = abs(raw_rho) > rho_cap_rad   # ★worst-yaw wrist-cap: |ρ| 제한(상수/knob)
        rho = (max(-rho_cap_rad, min(rho_cap_rad, raw_rho)) if capped
               else raw_rho + rho_corr)
        pan_R = pan  # 이 반복의 R/tcp 구축에 쓴 pan — meta 는 이 값(갱신 전)을 기록해야 정합
        R = _rz(pan_R) @ _ry(-a) @ R_TOPDOWN @ _rz(rho) @ TCP_TWIST
        face_label, n_face = max(faces, key=lambda f: float(np.dot(f[1], R[:, 0])))
        # ρ 잔차 feedback: -Δψ/cosα 는 1차 근사 + TCP twist 가 closing 수평방위를 α 비례로
        # 끌어당김(α=50° 서 ~1.7°) → 실측 closing 방위 잔차를 다음 반복 ρ 에 흡수(수렴 <0.1°).
        # d(closing_az)/dρ = -cosα 이므로 잔차 상쇄 부호는 +.
        resid = 0.0
        if not capped:  # capped 셀은 의도적 미스얼라인 — 정렬 feedback 안 함
            resid = wrap90(math.atan2(R[1, 0], R[0, 0]) - math.atan2(n_face[1], n_face[0]))
            rho_corr += resid / math.cos(a)
        # closing 축이 face normal과 어긋나면 face-center에서 시작한 jaw chord가 cube center를
        # 비켜 moving jaw가 모서리를 밀어낸다. face tangent 방향으로 h*tan(theta)만큼 옮겨
        # closing chord를 cube center 쪽으로 통과시킨다(ratio=1 완전 보정, knob으로 A/B).
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
    pre_pos = tcp_tgt - _pre_back(xyz) * R[:, 2]
    quat = _mat2quat(R)
    return pre_pos, quat, {
        "mode": "manifold",
        # tilt_deg/face_rank = 레거시 score·로그 키 호환(_candidate_score, diag 문자열)
        "tilt_deg": float(alpha_deg),
        "alpha_deg": float(alpha_deg),
        "rho_deg": math.degrees(rho),
        "rho_capped": bool(capped),  # worst-yaw wrist-cap 트리거 여부(프리뷰)
        "chord_shift_mm": float(tangent_shift * 1000.0),
        # 고정점 잔차 — 크면 후보가 manifold 에서 그만큼 벗어나 있다(진단용, 게이트 아님).
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


def _resolve_cube_halves(cube_half, n_env):
    """요청의 ``cube_half`` → 길이 n_env 리스트.

    리스트(per-env 크기 DR) · 스칼라(전 env 동일) · None(구버전 요청 → 상수 폴백) 모두 수용.
    길이가 모자라면 마지막 값으로 패딩한다(요청이 잘려 들어와도 조준이 침묵하지 않게 —
    길이 불일치는 진단 로그에 남는다).
    """
    if cube_half is None:
        return [CUBE_HALF] * n_env
    if isinstance(cube_half, (list, tuple)):
        vals = [float(v) for v in cube_half] or [CUBE_HALF]
        if len(vals) < n_env:
            PickPlacePlanner._diag(
                f"[cube-half] 요청 길이 {len(vals)} < env {n_env} — 마지막 값으로 패딩")
            vals = vals + [vals[-1]] * (n_env - len(vals))
        return vals[:n_env]
    return [float(cube_half)] * n_env


class PickPlacePlanner:
    """pick-place planner. BatchMotionPlanner batch 차원 = IsaacLab env 차원.

    후보 grasp 는 goalset 으로 넣고, phase별 plan_pose/plan_cspace 는 N개 env를 한 번에 푼다.
    cuRobo `multi_env=True`에서는 batch index가 collision env index가 되므로 이 매핑을 유지해야
    DR로 서로 다른 cube/bowl obstacle을 병렬 계획에 올바르게 적용할 수 있다.
    """

    def __init__(self, bowl_bl=(0.22, -0.265), max_batch_size=64):
        self.default_bowl_bl = bowl_bl
        self.max_batch_size = int(max_batch_size)
        self.max_goalset = max(K, len(ALPHA_SCAN_DEG))
        # 요청마다 갱신되는 **per-env** 큐브 반변(cube_specs 단일 소스, 크기 DR 이면 env 마다
        # 다르다). 필드 없는 구버전 요청은 상수 폴백. len == 요청 env 수.
        self.cube_halves = [CUBE_HALF]
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
            num_ik_seeds=64, num_trajopt_seeds=8,  # R1: seed 예산 확대(기본 32/4, D1·D4)
            use_cuda_graph=False))
        self.p.warmup(enable_graph=False, num_warmup_iterations=2)
        self.tf = self.p.tool_frames
        self.nA = len(self.p.joint_names)
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

    def _grasp_face_error(self, q_pre, cube, cube_half, face_normal=None):
        """IK-후-FK 실측 fixed jaw inner face center 를 cube face center 와 **3D** 비교(사용자 스펙).

        grasp 자세 = pre 자세서 approach축(tcp z) linear descend(table clamp; plan_pickplace grasp
        와 동일 식). descend 는 orientation 보존 → grasp 회전 = pre 회전이라 pad 방향 정확, 위치만
        하강 이동. fixed jaw inner face center = grasp_tcp + R·FIXED_INNER_CENTER(단순 tcp+offset·x̂ 아님).
        face_center = cube_center + cube_half·n(closing축). e 를 (normal, tangent, height)로 분해.
        returns {n:e_normal(clearance), t:e_tangent(face-plane lateral), h:e_height(world-z),
                 tilt_deg:grasp pitch°, face_angle:signed solver-XY alpha°,
                 c:centerline(√(t²+h²))}."""
        geom = self._grasp_geometry(q_pre, cube, cube_half, face_normal)
        e = geom["fixed_inner"] - geom["face_center"]
        n_face = geom["face_normal"]
        t = geom["face_tangent"]
        zax = geom["tcp_axes"]["z"]
        xax = geom["tcp_axes"]["x"]
        e_t, e_h = float(np.dot(e, t)), float(e[2])
        fxy = np.asarray(n_face[:2], dtype=np.float64)
        xxy = np.asarray(xax[:2], dtype=np.float64)
        fn_xy = np.linalg.norm(fxy)
        xn_xy = np.linalg.norm(xxy)
        if fn_xy < 1e-6 or xn_xy < 1e-6:
            face_angle = 90.0
        else:
            fxy /= fn_xy
            xxy /= xn_xy
            # alpha: selected cube face normal -> actual TCP closing x-axis, solver XY.
            face_angle = math.degrees(math.atan2(
                fxy[0] * xxy[1] - fxy[1] * xxy[0],
                float(np.dot(fxy, xxy)),
            ))
        tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, -float(zax[2])))))
        return {"n": float(np.dot(e, n_face)), "t": e_t, "h": e_h,
                "tilt_deg": float(tilt_deg), "c": math.hypot(e_t, e_h),
                "face_angle": float(face_angle)}

    def _grasp_geometry(self, q_pre, cube, cube_half, face_normal=None):
        """선택된 pre-grasp IK 해에서 실제 grasp 순간의 TCP/fixed-jaw/cube 기하를 같은 frame에 모은다."""
        tp = self.p.compute_kinematics(q_pre).tool_poses.get_link_pose(self.tf[0])
        pos = tp.position.detach().view(-1).cpu().numpy()[:3]
        quat = tp.quaternion.detach().view(-1).cpu().numpy()[:4]
        xax, yax, zax = self._quat_axes(quat)  # x=closing, z=approach
        cc = np.array(cube[:3])
        # descend = 실행 phase ② 와 **같은 함수**로 계산(_descend_tstar 단일 공식)
        grasp_tcp = pos + _descend_tstar(pos[2], float(zax[2]), cube) * zax
        dx, dy, dz = FIXED_INNER_CENTER
        fixed_inner = grasp_tcp + dx * xax + dy * yax + dz * zax   # FK 실측 pad center(world)
        n_face = np.array(face_normal, dtype=np.float64) if face_normal is not None else xax
        n_face[2] = 0.0
        n_norm = np.linalg.norm(n_face)
        n_face = n_face / n_norm if n_norm > 1e-6 else xax
        face_center = cc + float(cube_half) * n_face               # fixed jaw 가 닿는 실제 cube face 중심
        t = np.cross(np.array([0.0, 0.0, 1.0]), n_face); tn = np.linalg.norm(t)
        t = t / tn if tn > 1e-6 else np.array([0.0, 1.0, 0.0])     # face-plane tangent(수평, ⊥ closing)
        fixed_tip = grasp_tcp + PAD_LOW_OFF * zax
        return {
            "frame": "curobo_solver_base",
            "pre_tcp": pos,
            "grasp_tcp": grasp_tcp,
            "grasp_quat_wxyz": quat,
            "fixed_inner": fixed_inner,
            "fixed_jaw_tip": fixed_tip,
            "cube_center": cc,
            "face_normal": n_face,
            "face_center": face_center,
            "face_tangent": t,
            "tcp_axes": {"x": xax, "y": yax, "z": zax},
        }

    def _grasp_geometry_diag(self, q_pre, cube, cube_quat_wxyz, cube_half, face_normal=None):
        geom = self._grasp_geometry(q_pre, cube, cube_half, face_normal)
        q_cube_solver = _quat_mul(_quat_z(math.radians(BASE_YAW)), _quat_normalize(cube_quat_wxyz))

        def vec(v):
            return [float(x) for x in np.asarray(v, dtype=np.float64)]

        return {
            "frame": geom["frame"],
            "units": {"position": "m", "euler_xyz": "deg"},
            "cube": {
                "half_extent": float(cube_half),
                "position": vec(geom["cube_center"]),
                "quat_wxyz": vec(q_cube_solver),
                "euler_xyz_deg": _quat_to_euler_xyz_deg(q_cube_solver),
            },
            "selected_face": {
                "normal": vec(geom["face_normal"]),
                "center": vec(geom["face_center"]),
                "tangent": vec(geom["face_tangent"]),
            },
            "tcp": {
                "pre_position": vec(geom["pre_tcp"]),
                "grasp_position": vec(geom["grasp_tcp"]),
                "quat_wxyz": vec(geom["grasp_quat_wxyz"]),
                "euler_xyz_deg": _quat_to_euler_xyz_deg(geom["grasp_quat_wxyz"]),
                "axis_x_closing": vec(geom["tcp_axes"]["x"]),
                "axis_z_approach": vec(geom["tcp_axes"]["z"]),
            },
            "fixed_jaw": {
                "tip_position": vec(geom["fixed_jaw_tip"]),
                "inner_center_position": vec(geom["fixed_inner"]),
                "tip_model": "grasp_tcp + PAD_LOW_OFF * tcp_z_axis",
            },
        }

    # ── candidate pose (5-DOF manifold) ──────────────────────────────────────────
    @staticmethod
    def _cube_face_normals(cube_bl):
        """cube quat(base_link) → solver-frame 수평 grasp face ±normal ≤4개 [(label, n̂)].

        base_link cube quaternion 을 solver-frame 으로 합성한 뒤 body basis 를 직접 회전해
        수평 성분이 큰 body axis 두 개의 ±normal 을 만든다(기울어진 큐브 gimbal-lock 안전).
        어느 face 를 잡을지는 후보 생성기(cand_pose_manifold)가 ρ 보상 후 closing 축
        최근접 내적으로 자동 결정하므로 여기선 선별하지 않는다."""
        try:
            q_cube = _quat_normalize(cube_bl[3:7] if len(cube_bl) >= 7 else [1.0, 0.0, 0.0, 0.0])
        except ValueError:
            return []
        q_solver = _quat_mul(_quat_z(math.radians(BASE_YAW)), q_cube)
        horizontal = []
        for axis in np.eye(3):
            world_axis = _quat_rotate(q_solver, axis)
            n = np.array([world_axis[0], world_axis[1], 0.0], dtype=np.float64)
            h = np.linalg.norm(n)
            if h > 1e-5:
                horizontal.append((h, n / h))
        if not horizontal:
            return []
        horizontal.sort(key=lambda item: item[0], reverse=True)
        faces = []
        for _h, n in horizontal[:2]:
            for sign in (1.0, -1.0):
                cand = sign * n
                if all(float(np.dot(cand, f)) < 0.98 for _lbl, f in faces):
                    faces.append((f"f{len(faces)}", cand))
        return faces

    def _manifold_candidates(self, xyz, faces, cube_half, knobs=None):
        """(pan,α,ρ) 후보 목록 — ALPHA_SCAN_DEG 순서(|α| 오름차순 ± interleave)가 곧 우선순위."""
        kn = knobs or {}
        rho_cap_rad = math.radians(float(kn.get("rho_cap_deg", RHO_CAP_DEG)))
        chord_center_ratio = float(kn.get("chord_center_ratio", CHORD_CENTER_RATIO))
        pan_r = math.hypot(float(xyz[0]) - PAN_AXIS_XY[0], float(xyz[1]) - PAN_AXIS_XY[1])
        # τ 는 상수다. 예전에는 도달반경 r 에 따라 10°→25° 로 여는 램프(R2/R2')를 썼는데,
        # 그 임계값(0.16/0.22/0.06)은 실패한 **개별 셀**에 맞춰 옮겨진 값이었다(주석 이력에
        # r=0.246 셀 하나 때문에 0.24→0.22 로 당긴 기록이 남아 있었다).
        # 램프의 근거였던 "τ 를 열어도 안전하다"는 논리는 r 과 무관하다 — 나쁜 후보는 FK 게이트가
        # 독립적으로 거르고, τ 는 후보 **수**만 늘린다. 그래서 전 구간 최대치로 고정한다.
        # 우선순위(|α| 오름차순)는 그대로라 기존에 통과하던 셀의 선택 후보는 바뀌지 않고,
        # 후보를 다 소진하던 셀에만 추가 시도가 생긴다.
        tau = math.radians(float(kn.get("tau_max_deg", TAU_MAX_DEG)))
        cands = []
        for a_deg in ALPHA_SCAN_DEG:
            cand = cand_pose_manifold(
                xyz, faces, a_deg, tau, rho_cap_rad=rho_cap_rad,
                chord_center_ratio=chord_center_ratio, cube_half=float(cube_half),
            )
            if cand is None:
                continue
            cand[2]["pan_radius"] = float(pan_r)
            cands.append(cand)
        msg = (f"[manifold] cube=({float(xyz[0]):+.3f},{float(xyz[1]):+.3f}) r={pan_r:.3f} "
               f"faces={len(faces)} cands={len(cands)}")
        if cands:
            msg += (f" dpsi={cands[0][2]['dpsi_deg']:+.1f} rho0={cands[0][2]['rho_deg']:+.1f} "
                    f"pan={cands[0][2]['pan_deg']:+.1f}")
        self._diag(msg)
        return cands

    def _gate_candidate(self, end, start_row, cube, cube_quat, cube_half, meta, cand_idx,
                        n_cands, seed_i, rescue=False):
        """cuRobo 성공 후보의 FK gate 판정 + 진단 dict — batch scan·rescue 공용."""
        wrist_ok, wr, wr_delta = self._wrist_delta_ok(end, start_row)
        fe = self._grasp_face_error(end, cube, cube_half, meta.get("face_normal"))
        ok = (wrist_ok and FIXED_JAW_CLEAR_MIN <= fe["n"] <= FIXED_JAW_CLEAR_MAX
              and abs(fe["t"]) <= E_TANGENT_MAX and abs(fe["h"]) <= E_HEIGHT_MAX
              and abs(fe["face_angle"]) <= SIMPLE_FACE_GATE_MAX_DEG)
        score = self._candidate_score(fe, meta, wr_delta)
        diag = {**meta,
                "candidate_index": int(cand_idx),
                # top-level 에도 둔다 — 소비자(plan_pickplace_batch·SM 로그)가 여기서 읽는데
                # selection 하위에만 있어 성공 env 는 늘 `candidates=0` 으로 찍혔다.
                "num_candidates": int(n_cands),
                "score": list(score),
                "wrist_roll_deg": math.degrees(wr),
                "wrist_delta_deg": math.degrees(wr_delta),
                "fk_face_error": {k: float(v) for k, v in fe.items()},
                "geometry": self._grasp_geometry_diag(end, cube, cube_quat, cube_half,
                                                      meta.get("face_normal")),
                "selection": {
                    "policy": ("homogeneous_batch_rescue" if rescue
                               else "batched_priority_candidate_scan"),
                    "num_candidates": int(n_cands),
                    "candidate_rank": int(cand_idx),
                    "seed": seed_i,
                }}
        if not ok:
            diag["fail"] = "candidate_failed_fk_gate"
        self._diag(
            f"[manifold]{' rescue' if rescue else ''} "
            f"candidate={meta.get('face_label', meta.get('face_index'))} "
            f"alpha={float(meta.get('alpha_deg', 0.0)):+.0f} "
            f"rho={float(meta.get('rho_deg', 0.0)):+.1f} solved=True "
            f"wrist_ok={wrist_ok} face_ok={abs(fe['face_angle']) <= SIMPLE_FACE_GATE_MAX_DEG} "
            f"face_alpha={fe['face_angle']:+.2f} "
            f"wrist_delta={math.degrees(wr_delta):+.2f} "
            f"e_norm={fe['n'] * 1000.0:+.2f}mm "
            f"e_tan={fe['t'] * 1000.0:+.2f}mm e_h={fe['h'] * 1000.0:+.2f}mm "
            f"clear_ok={FIXED_JAW_CLEAR_MIN <= fe['n'] <= FIXED_JAW_CLEAR_MAX} "
            f"t_ok={abs(fe['t']) <= E_TANGENT_MAX} h_ok={abs(fe['h']) <= E_HEIGHT_MAX}")
        return ok, diag

    @staticmethod
    def _bias_adjusted_goal(p_i, meta, diag):
        """clearance(e_normal)만 어긋난 후보 → 실측 오차만큼 pad 조준을 face 쪽으로 민 goal.

        원거리 IK undershoot 는 방향이 일정해(실측 e_norm 5.8~8.5 mm) 1회 bias 보정으로 살아난다.
        다른 게이트(tangent/height/face_angle/wrist)가 깨진 후보는 **자세 자체가 틀린** 것이라
        보정 대상이 아니다. 이미 보정된 후보는 재보정하지 않는다(후보 증식 상한 = 2×).

        → (p_adj, meta_adj) 또는 None. batch pass loop·rescue 양쪽이 같은 규칙을 쓴다
          (예전엔 rescue 에만 있어서 "batch 에서 굶었는가"가 성공 여부를 갈랐다).
        """
        if meta.get("aim_corrected_mm") is not None:
            return None
        fe = diag.get("fk_face_error", {})
        if not (abs(fe.get("t", 1.0)) <= E_TANGENT_MAX
                and abs(fe.get("h", 1.0)) <= E_HEIGHT_MAX
                and abs(fe.get("face_angle", 90.0)) <= SIMPLE_FACE_GATE_MAX_DEG
                and abs(diag.get("wrist_delta_deg", 999.0)) <= WRIST_ROLL_DELTA_LIMIT_DEG):
            return None
        err = fe.get("n", 0.0) - FIXED_JAW_CLEAR_TARGET
        n_hat = np.asarray(meta["face_normal"], dtype=np.float64)
        p_adj = np.asarray(p_i, dtype=np.float64) - err * n_hat
        return p_adj, {**meta, "aim_corrected_mm": float(err * 1000.0)}

    def _wrist_delta_ok(self, q, start):
        wr_idx = self.p.joint_names.index("wrist_roll")
        qpos = q.position.detach()
        spos = start.position.detach()
        wr = float(qpos.view(-1, self.nA)[0, wr_idx].item())
        swr = float(spos.view(-1, self.nA)[0, wr_idx].item())
        delta = wr - swr
        return abs(delta) <= WRIST_ROLL_DELTA_LIMIT, wr, delta

    @staticmethod
    def _candidate_score(fe, meta, wr_delta, clear_target=FIXED_JAW_CLEAR_TARGET):
        """정렬 먼저, 그 다음 낮고 중심에 가까운 grasp. 작을수록 물리 grasp 가 안정적이다.

        ★1순위 = closing 축 ↔ face normal 각도(``face_angle``)의 5° 버킷 (2026-07-30).
        비스듬한 closing 은 한쪽 jaw 가 face 모서리를 먼저 때려 **큐브를 밀어낸다** — 측정된
        grasp 실패가 정확히 그 모습이었다(face_angle +32.7°, 큐브 20 mm 밀린 채 +2.7 mm 만 들림.
        게이트 ``SIMPLE_FACE_GATE_MAX_DEG=40`` 은 통과한다).

        ★2순위 = ``|e_n − target|``(clearance). ``e_n`` 은 pad 이 face 에서 떨어진 거리 =
        **폐합이 이동해야 하는 거리**다. 게이트 창이 2~8 mm 로 넓어 8 mm 짜리 후보가 통과했고,
        그 케이스도 큐브를 15 mm 밀어낸 채 실패했다(e_n +7.9 mm, 정렬은 −12.9° 로 양호).
        멀리서 닫을수록 큐브를 칠 확률이 커지므로 정렬 다음으로 중요하다.

        예전 1순위였던 ``|e_h|`` 는 table clamp 가 지배해 후보 간 변동이 4 mm 안쪽(실측 p50 14.6 /
        p90 18.8 mm)이라 **변별력이 거의 없는데** 정렬(0~40° 변동)과 clearance(2~8 mm)를 눌러
        이겼다 → 4순위로 내렸다. face_angle 에 5° 버킷을 쓰는 이유는 연속값 1순위면 0.1° 차이로
        clearance 가 훨씬 나쁜 후보가 뽑히기 때문이다.
        근거·측정 = ``09_TACIT_KNOWLEDGE.md §15.5``.
        """
        return (
            int(round(abs(fe["face_angle"]) / 5.0)),   # ① 정렬(5° 버킷) — 밀어냄의 직접 원인
            round(abs(fe["n"] - clear_target) * 1000.0, 2),  # ② 폐합 이동거리
            round(abs(fe["t"]) * 1000.0, 2),                 # ③ face 내 lateral
            round(abs(fe["h"]) * 1000.0, 2),                 # ④ 높이(clamp 지배 → 변별력 낮음)
            round(abs(float(meta.get("tilt_deg", 0.0))), 2),
            int(meta.get("face_rank", 0)),
            round(abs(math.degrees(wr_delta)), 2),
        )

    def _plan_pregrasp_batch(self, cubes, face_sets, cube_quats, halves, starts, knobs):
        """N env manifold 후보를 priority order(|α| 오름차순)로 검사한다.

        각 pass는 env별 후보 1개씩을 BatchMotionPlanner batch 차원으로 병렬 계획한다.
        env별 후보 goalset은 이 5-DoF 문제에서 개별 후보 계획보다 `no_curobo_solution`이
        더 자주 발생했다. 따라서 batch dimension은 env index로 유지하고, 후보 축만 lockstep으로
        훑어 안정성을 우선한다.
        """
        n_env = len(cubes)
        kn = knobs or {}
        # ⚠ knobs.seed 는 **진단 라벨 전용**이다. cuRobo v0.8 `reset_seed()` 는 인자를 받지 않아
        #   planner 해를 외부 seed 로 흔들 수 없다(= planning 은 입력에 대해 결정적). SM 은 더 이상
        #   이 knob 을 보내지 않는다 — 옛 sweep JSON 재현용으로 받기만 한다.
        seed = kn.get("seed")
        max_attempts = 4       # R1: 2→4 (attempt 마다 Halton seed 전진=다양성, D1)
        rescue_attempts = 6    # R1: 동질 rescue batch 는 더 넓게(6) — 굶긴 원거리 row 구제
        disable_cube = bool(kn.get("disable_cube_obstacle_for_approach", False))
        start_pos, start_quat, _ = self._ee_pose_axis_batch(starts)
        per_env = []
        for cube, faces, half in zip(cubes, face_sets, halves):
            cands = self._manifold_candidates(cube, faces, half, kn)
            per_env.append(list(cands[:self.max_goalset]))

        trajs = [None] * n_env
        ends = [JointState.from_position(starts.position[i: i + 1].detach().clone(),
                                         joint_names=self.p.joint_names)
                for i in range(n_env)]
        diagnostics = []
        for i in range(n_env):
            fail = "not_attempted" if per_env[i] else "no_candidates"
            diagnostics.append({"mode": "manifold", "fail": fail, "num_candidates": len(per_env[i])})
        initial_goalset_size = max((len(c) for c in per_env), default=0)
        ok_mask = [False] * n_env
        # env 별 최선 후보 (score, traj, end, diag) — 앞 CANDIDATE_SCAN_PASSES pass 를 비교한다.
        best = [None] * n_env
        plan_ms = 0.0
        plan_calls = 0
        # pass 수는 후보 수에 따라 자란다: clearance 만 어긋난 후보는 bias 보정본을 **같은 env 의
        # 후보 목록 뒤에 덧붙여** 다음 pass(어차피 도는 batch)에 태운다 → 추가 plan 호출 0.
        # (rescue 전용이던 보정을 batch 경로에도 동일 규칙으로 적용 — B4)
        pass_idx = -1
        while True:
            pass_idx += 1
            # 미통과 env 는 물론, **이미 통과한 env 도** 스캔 폭 안에서는 계속 후보를 본다
            # (더 나은 score 가 나오면 교체). batch slot 을 놀리지 않으므로 공짜다.
            active = [i for i in range(n_env)
                      if pass_idx < len(per_env[i])
                      and (not ok_mask[i] or pass_idx < CANDIDATE_SCAN_PASSES)]
            if not active:
                break
            pos = np.zeros((n_env, 3), dtype=np.float32)
            quat = np.zeros((n_env, 4), dtype=np.float32)
            pass_meta = []
            active_set = set(active)
            for i in range(n_env):
                if i in active_set:
                    p_i, q_i, m_i = per_env[i][pass_idx]
                else:
                    p_i, q_i, m_i = start_pos[i], start_quat[i], {"mode": "fallback"}
                pos[i] = p_i
                quat[i] = q_i
                pass_meta.append(m_i)
            t0 = time.perf_counter()
            if disable_cube:
                self._set_cube_obstacle_enabled(False, n_env)
            self.p.disable_link_collision(CONTACT_LINKS + DESCEND_EXTRA_OFF)
            try:
                result = self.p.plan_pose(
                    goal_tool_poses=self._pose_batch(pos, quat),
                    current_state=starts,
                    max_attempts=max_attempts,
                    success_ratio=1.0,
                )
            finally:
                self.p.enable_link_collision(CONTACT_LINKS + DESCEND_EXTRA_OFF)
                if disable_cube:
                    self._set_cube_obstacle_enabled(True, n_env)
            plan_ms += (time.perf_counter() - t0) * 1000.0
            plan_calls += 1
            planned = self._extract_batch(result, starts, n_env)
            for i in active:
                meta = pass_meta[i]
                traj, end = planned[i]
                if traj is None:
                    self._diag(
                        f"[manifold] candidate={meta.get('face_label', meta.get('face_index'))} "
                        f"alpha={float(meta.get('alpha_deg', 0.0)):+.0f} "
                        f"rho={float(meta.get('rho_deg', 0.0)):+.1f} solved=False"
                    )
                    diagnostics[i] = {**meta, "fail": "no_curobo_solution",
                                      "candidate_index": pass_idx,
                                      "num_candidates": len(per_env[i])}
                    continue
                start_row = JointState.from_position(
                    starts.position[i: i + 1].detach().clone(), joint_names=self.p.joint_names)
                ok, diag = self._gate_candidate(
                    end, start_row, cubes[i], cube_quats[i], halves[i], meta, pass_idx,
                    len(per_env[i]), None if seed is None else int(seed) + i)
                if ok:
                    score = tuple(diag["score"])
                    if best[i] is None or score < best[i][0]:
                        best[i] = (score, traj, end, diag)
                        trajs[i] = traj
                        ends[i] = end
                        diagnostics[i] = diag
                    ok_mask[i] = True
                else:
                    if best[i] is None:
                        diagnostics[i] = diag   # 아직 통과본이 없을 때만 실패 진단을 남긴다
                    adj = self._bias_adjusted_goal(pos[i], meta, diag)
                    if adj is not None:  # clearance 만 어긋남 → 보정본을 뒤 pass 후보로 추가
                        p_adj, meta_adj = adj
                        per_env[i].append((p_adj, quat[i], meta_adj))
            if all(ok_mask) and pass_idx + 1 >= CANDIDATE_SCAN_PASSES:
                break

        # ── rescue: 혼합 batch 가 굶긴 env 를 동질 batch(동일 goal·start 전 row 복제)로 구제.
        # 실측(2026-07-10): 단독/동질 batch 가 attempts=2 로 푸는 pose 를 혼합 batch 는 전
        # pass 실패(attempts=10 도 무효) — cuRobo batch solver 가 이질 goal 에서 어려운 row
        # (원거리 r≈0.23)를 굶김. 복제 batch 의 다른 row 는 각자 env world(다른 obstacle)라
        # 실패할 수 있으나 row i 만 채택하므로 무해(obstacle 동기화 불요 실증).
        rescue_envs = 0
        for i in range(n_env):
            if ok_mask[i] or not per_env[i]:
                continue
            rescue_envs += 1
            start_row_t = starts.position[i: i + 1].detach()
            start_row = JointState.from_position(start_row_t.clone(),
                                                 joint_names=self.p.joint_names)
            dup_starts = self._joint_state_batch(start_row_t.repeat(self.max_batch_size, 1))
            seed_i = None if seed is None else int(seed) + i

            def _dup_plan(p_g, q_g, env_i=i, dup=dup_starts):
                """후보 1개를 전 row 복제 batch 로 계획 → env_i row 의 (traj|None, end)."""
                nonlocal plan_ms, plan_calls
                pos = np.repeat(np.asarray(p_g, dtype=np.float32)[None], n_env, axis=0)
                quat = np.repeat(np.asarray(q_g, dtype=np.float32)[None], n_env, axis=0)
                t0 = time.perf_counter()
                if disable_cube:
                    self._set_cube_obstacle_enabled(False, n_env)
                self.p.disable_link_collision(CONTACT_LINKS + DESCEND_EXTRA_OFF)
                try:
                    result = self.p.plan_pose(
                        goal_tool_poses=self._pose_batch(pos, quat),
                        current_state=dup, max_attempts=rescue_attempts,
                        success_ratio=1.0)
                finally:
                    self.p.enable_link_collision(CONTACT_LINKS + DESCEND_EXTRA_OFF)
                    if disable_cube:
                        self._set_cube_obstacle_enabled(True, n_env)
                plan_ms += (time.perf_counter() - t0) * 1000.0
                plan_calls += 1
                return self._extract_batch(result, dup, n_env)[env_i]

            for cand_idx, (p_i, q_i, meta) in enumerate(per_env[i]):
                traj, end = _dup_plan(p_i, q_i)
                if traj is None:
                    self._diag(
                        f"[manifold] rescue candidate={meta.get('face_label')} "
                        f"alpha={float(meta.get('alpha_deg', 0.0)):+.0f} solved=False")
                    continue
                ok, diag = self._gate_candidate(
                    end, start_row, cubes[i], cube_quats[i], halves[i], meta, cand_idx,
                    len(per_env[i]), seed_i, rescue=True)
                adj = None if ok else self._bias_adjusted_goal(p_i, meta, diag)
                if adj is not None:
                    # rescue 는 어차피 env 단위 순차라 즉시 재시도가 싸다(batch 경로는 뒤 pass 에 태움).
                    p_adj, meta_adj = adj
                    traj2, end2 = _dup_plan(p_adj, q_i)
                    if traj2 is not None:
                        ok2, diag2 = self._gate_candidate(
                            end2, start_row, cubes[i], cube_quats[i], halves[i], meta_adj,
                            cand_idx, len(per_env[i]), seed_i, rescue=True)
                        if ok2:
                            traj, end, ok, diag = traj2, end2, ok2, diag2
                diagnostics[i] = diag
                if ok:
                    trajs[i] = traj
                    ends[i] = end
                    ok_mask[i] = True
                    break

        for i, ok in enumerate(ok_mask):
            diagnostics[i]["plan_ms"] = plan_ms
            if not ok and diagnostics[i].get("fail") == "not_attempted":
                diagnostics[i]["fail"] = "no_feasible_candidate"
        q_rows = [e.position.detach().view(1, -1) for e in ends]
        if starts.position.shape[0] > n_env:
            q_rows.append(starts.position[n_env:].detach())
        q_end = self._joint_state_batch(torch.cat(q_rows, dim=0))
        return trajs, q_end, diagnostics, ok_mask, {
            "approach_plan_ms": plan_ms,
            "candidate_passes": int(plan_calls),
            "candidate_goalset_size": int(initial_goalset_size),
            "rescue_envs": int(rescue_envs),
        }

    # ── obstacle / attach helpers ───────────────────────────────────────────────
    def _set_cube_obstacle_enabled(self, enabled, num_envs):
        try:
            for env_idx in range(num_envs):
                self.p.scene_collision_checker.enable_obstacle("cube", bool(enabled), env_idx=env_idx)
        except Exception as e:
            self._diag(f"[cube-obst] enable={enabled} FAIL {type(e).__name__}: {e}")

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

    # ── main entry ──────────────────────────────────────────────────────────────
    def plan_pickplace_batch(self, cube_bls, bowl_bls=None, starts_rad=None, knobs=None,
                             cube_half=None):
        """N개 env full pick-place를 BatchMotionPlanner 1개로 병렬 계획.

        ``cube_half`` = 요청이 실어 보낸 큐브 반변(m, cube_specs 단일 소스). 크기 DR 이 켜지면
        env 마다 다르므로 **길이 N 리스트**로 온다. 스칼라(전 env 동일)와 None(구버전 SM →
        상수 ``CUBE_HALF`` 폴백)도 받는다.
        """
        n_env = len(cube_bls)
        self._ensure_batch_size(n_env)   # ※ 배치 크기 변경 시 __init__ 재실행 → cube_halves 재설정 후
        self.cube_halves = _resolve_cube_halves(cube_half, n_env)
        kn = knobs or {}
        z_off = float(kn.get("grasp_z_off", GRASP_Z_OFF))
        g_open = float(kn.get("grip_open", GRIP_OPEN))
        g_close = float(kn.get("grip_close", GRIP_CLOSE))
        b_pull = float(kn.get("bowl_pull", BOWL_PULL))
        cube_off_pick_contact = bool(kn.get("disable_cube_obstacle_for_pick_contact", False))
        t_all0 = time.perf_counter()
        profile = {}
        self.p.reset_seed()

        cubes = []
        face_sets = []
        cube_quats = []
        bowls = []
        per_env_bowl = bool(bowl_bls) and isinstance(bowl_bls[0], (list, tuple))
        for i, cube_bl in enumerate(cube_bls):
            cube = usd_to_urdf(cube_bl[:3])
            cube = (cube[0], cube[1], cube[2] + z_off)
            cubes.append(cube)
            face_sets.append(self._cube_face_normals(cube_bl))
            cube_quats.append(cube_bl[3:7] if len(cube_bl) >= 7 else [1.0, 0.0, 0.0, 0.0])
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
                "cube_face_normals_solver": [[float(v) for v in n] for _idx, n in face_sets[i]],
                "cube_grasp_faces_solver": [
                    {"label": label, "normal": [float(v) for v in n]}
                    for label, n in face_sets[i]
                ],
                "bowl_solver_xy": [float(bowls[i][0]), float(bowls[i][1])],
                "cube_half": float(self.cube_halves[i]),
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

        # ① APPROACH: batch=N. 후보 = (pan,α,ρ) manifold 파라미터화 — face 는 ρ 보상 후
        # closing 축 최근접으로 자동 결정, |α| 오름차순 lockstep 검사.
        pre, q_pre, cand_diag, pre_ok, pre_prof = self._plan_pregrasp_batch(
            cubes, face_sets, cube_quats, self.cube_halves, starts, kn)
        profile.update(pre_prof)

        # ② GRASP descend + ③ LIFT.
        # knob 으로 큐브 obstacle 을 끄면 **이 구간에서만** 꺼져야 한다 → finally 복원.
        # (옛 코드는 끄기만 하고 안 켜서 이후 phase·다음 요청까지 상태가 샜다. approach 경로는
        #  이미 같은 try/finally 규약을 쓴다 — 두 경로를 맞춘다.)
        if cube_off_pick_contact:
            self._set_cube_obstacle_enabled(False, n_env)
        try:
            app, aq, zaxes = self._ee_pose_axis_batch(q_pre)
            gpos, tstars = [], []
            for i, (cube, zax) in enumerate(zip(cubes, zaxes)):
                tstar = _descend_tstar(app[i, 2], float(zax[2]), cube)  # 게이트와 동일 공식
                tstars.append(tstar)
                gpos.append(app[i] + tstar * zax)
            t0 = time.perf_counter()
            desc_planned = self._plan_to_batch(self._pose_batch(np.asarray(gpos), aq), q_pre,
                                               n_env, linear=True)
            profile["grasp_plan_ms"] = (time.perf_counter() - t0) * 1000.0
            q_grasp, grasp_ok = self._merge_phase_ends(desc_planned, q_pre, pre_ok)

            up = np.asarray([gpos[i] - min(tstars[i], LIFT_BACK) * zaxes[i] for i in range(n_env)])
            t0 = time.perf_counter()
            lift_planned = self._plan_to_batch(self._pose_batch(up, aq), q_grasp, n_env, linear=True)
            profile["lift_plan_ms"] = (time.perf_counter() - t0) * 1000.0
            q_lift, lift_ok = self._merge_phase_ends(lift_planned, q_grasp, grasp_ok)
        finally:
            if cube_off_pick_contact:
                self._set_cube_obstacle_enabled(True, n_env)

        attached = False
        attach_failed = False
        if any(lift_ok):
            attached = self._attach_cube(q_lift)
            # attach 실패 = transit 을 "잡은 큐브 부피 없이" 계획한다는 뜻 → 조용히 넘기지 않는다.
            attach_failed = not attached
            if attach_failed:
                print("[planner] ⚠ attach_cube FAILED — transit 이 큐브 부피 없이 계획됨"
                      "(그릇 rim 스침 위험). diag: attach_failed=true", flush=True)
                self._diag("[attach] FAILED — transit planned without grasped cube volume")

        # ④ TRANSIT: env별 bowl 상공 FK-bank goalset.
        fk_pos, fk_quat, _ = self._ee_pose_axis_batch(q_lift)
        tr_pos = np.zeros((n_env, K, 3), dtype=np.float32)
        tr_quat = np.zeros((n_env, K, 4), dtype=np.float32)
        transit_z = float(kn.get("transit_z", TRANSIT_Z)) + BASE_T[2]
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
            approach_ok = bool(pre_ok[i]) and pre[i] is not None
            grasp_phase_ok = bool(grasp_ok[i]) and desc_planned[i][0] is not None
            lift_phase_ok = bool(lift_ok[i]) and lift_planned[i][0] is not None
            transit_phase_ok = bool(transit_ok[i]) and transit_planned[i][0] is not None
            phases = {
                "approach": approach_ok,
                "grasp": grasp_phase_ok,
                "lift": lift_phase_ok,
                "transit": transit_phase_ok,
                "retreat": bool(retreat_ok[i]),
            }
            diagnostics[i]["candidate"] = cand_diag[i]
            diagnostics[i]["approach_fail"] = cand_diag[i].get("fail") if cand_diag[i] else None
            diagnostics[i]["num_candidates"] = cand_diag[i].get("num_candidates", 0) if cand_diag[i] else 0
            # fail = **처음 실패한 phase**. 옛 코드는 approach 사유만 실었는데, grasp/lift/
            # transit/retreat 에서 죽은 env 는 fail=None 이 돼 sweep 집계가 approach 로 편향됐다.
            first_bad = next((name for name, good in phases.items() if not good), None)
            diagnostics[i]["fail"] = (
                (diagnostics[i]["approach_fail"] or "approach_plan_failed")
                if first_bad == "approach"
                else (f"{first_bad}_plan_failed" if first_bad else None))
            diagnostics[i]["failed_phase"] = first_bad
            diagnostics[i]["phases"] = phases
            diagnostics[i]["attached"] = bool(attached)
            diagnostics[i]["attach_failed"] = bool(attach_failed)
            diagnostics[i]["profile_ms"] = {k: float(v) for k, v in profile.items()}
            ok = all(phases.values())
            diagnostics[i]["ok"] = ok
            if ok:
                trajectories.append(self._assemble({
                    "approach": pre[i],
                    "grasp": desc_planned[i][0],
                    "lift": lift_planned[i][0],
                    "transit": transit_planned[i][0],
                    "retreat": retreat_planned[i][0],
                }, g_open, g_close))
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
        close_steps = grip_ramp_steps(g_open, g_close)     # slew cap 에서 유도(하드코딩 아님)
        open_steps = grip_ramp_steps(g_close, g_open)
        close_hold = np.repeat(de[-1:], close_steps, 0)    # grasp: 정지 상태서 폐합
        grasp_hold = np.repeat(de[-1:], GRASP_HOLD_STEPS, 0)  # 폐합 접촉 안정화 후 lift
        settle_hold = np.repeat(tr[-1:], SETTLE_STEPS, 0)  # 그릇 상공서 짧게 정지(안정)
        open_hold = np.repeat(tr[-1:], open_steps, 0)      # release: 그릇 상공(transit)서 개방
        seq = [
            _grip(a, np.linspace(GRIP_INIT, g_open, len(a))),        # ① approach + gripper 개방(접근하며)
            _grip(de, g_open),                                       # ② grasp descend
            _grip(close_hold, np.linspace(g_open, g_close, close_steps)),  #   grasp close
            _grip(grasp_hold, g_close),                              #   grasp settle (hold)
            _grip(li, g_close),                                      # ③ lift
            _grip(tr, g_close),                                      # ④ transit
            _grip(settle_hold, g_close),                            #   settle over bowl (hold)
            _grip(open_hold, np.linspace(g_close, g_open, open_steps)),    # ⑤ release over bowl
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
    trajs, diagnostics = pl.plan_pickplace_batch(cubes, bowl, starts, dict(knobs or {}),
                                                 cube_half=req.get("cube_half"))
    out = [traj.tolist() if traj is not None else None for traj in trajs]
    return out, diagnostics


def serve_loop(pl, sock):
    """REQ/REP 서비스 루프.

    ★불변식: **받은 요청에는 반드시 답한다.** REP 를 한 번 빠뜨리면 클라이언트(SM)는 영원히
    블록된다 — 그래서 처리 전체를 try 로 감싸고 예외도 `{"ok":false,"err":...}` 로 응답한다.
    (옛 코드는 plan 예외가 그대로 올라와 planner 프로세스가 죽고 SM 은 무한 대기했다.)
    """
    request_i = 0
    while True:
        raw = sock.recv()
        stop = False
        try:
            req = json.loads(raw)
            cmd = req.get("cmd")
            if cmd == "ping":
                rep = {"ok": True}
            elif cmd == "plan_pickplace":
                request_i += 1
                n_env = len(req.get("cubes") or [])
                print(f"[planner] recv plan_pickplace #{request_i}: envs={n_env}", flush=True)
                trajs, diagnostics = plan_batch(pl, req)
                ok_count = sum(1 for t in trajs if t is not None)
                fails = [d.get("fail") for d in diagnostics]
                print(f"[planner] done plan_pickplace #{request_i}: planned={ok_count}/{n_env} "
                      f"fails={fails}", flush=True)
                rep = {"ok": True, "trajectories": trajs, "diagnostics": diagnostics}
            elif cmd == "shutdown":
                rep, stop = {"ok": True}, True
            else:
                rep = {"ok": False, "err": f"unknown {cmd!r}"}
        except Exception as exc:  # noqa: BLE001 — 어떤 예외든 응답은 나가야 한다
            traceback.print_exc()
            rep = {"ok": False, "err": f"{type(exc).__name__}: {exc}"}
            PickPlacePlanner._diag(f"[serve] EXCEPTION {rep['err']}\n{traceback.format_exc()}")
        sock.send_string(json.dumps(rep))
        if stop:
            return


# 크기 DR 사다리의 반변(m) — cube_specs CUBE_SIZE_CHOICES 미러. planner 는 self-contained
# 컨테이너에서 도는 진단 스크립트라 sim_to_real 패키지(isaac 의존)를 import 하지 않는다.
# ⚠ cube_specs.CUBE_SIZE_CHOICES 를 바꾸면 여기도 같이 바꾼다(self-check 가 잡는다).
SELF_CHECK_HALVES = (0.0125, 0.015, 0.0175, 0.020)


def _check_grip_ramp_within_cap():
    """유도한 ramp 가 정말 slew cap 안인가 + 폐합이 lift 전에 끝나는가."""
    step_cap = GRIPPER_SLEW_MAX_RAD_S / CONTROL_HZ
    for g_from, g_to, label in ((GRIP_OPEN, GRIP_CLOSE, "close"), (GRIP_CLOSE, GRIP_OPEN, "open")):
        n = grip_ramp_steps(g_from, g_to)
        per_step = abs(g_from - g_to) * _GRIP_RAD_PER_FEATURE / n
        assert per_step <= step_cap + 1e-9, (
            f"{label} ramp {n} 프레임 = {per_step * CONTROL_HZ:.2f} rad/s > cap "
            f"{GRIPPER_SLEW_MAX_RAD_S} — 정해진 프레임 안에 끝나지 않는다")
    print(f"self-check-geom: grip ramp close={grip_ramp_steps(GRIP_OPEN, GRIP_CLOSE)} "
          f"open={grip_ramp_steps(GRIP_CLOSE, GRIP_OPEN)} 프레임 "
          f"(cap {GRIPPER_SLEW_MAX_RAD_S} rad/s @ {CONTROL_HZ:.0f} Hz)")


def _check_grasp_height_budget():
    """크기 사다리 전 구간에서 **table clamp 이후** pad 조준 높이가 FK 게이트 안인가.

    descend 는 jaw tip 이 ``TABLE_TOP + TABLE_MARGIN`` 아래로 못 가게 clamp 된다
    (``_descend_tstar``). 큐브가 작아질수록 중심이 낮아지는데 tip 은 못 내려가므로
    pad center 가 큐브 중심보다 **위로** 벌어진다. α=0(top-down, 최악)에서:

        e_h = TABLE_MARGIN + PAD_LOW_OFF − FIXED_INNER_CENTER[2] − half − GRASP_Z_OFF

    이 값이 ``E_HEIGHT_MAX`` 를 넘으면 그 크기는 FK 게이트에서 전멸한다 — 크기 하한을
    더 낮추거나 TABLE_MARGIN·pad 기하를 바꿀 때 **조용히** 깨지는 자리라 여기서 터뜨린다.
    """
    for half in SELF_CHECK_HALVES:
        e_h = TABLE_MARGIN + PAD_LOW_OFF - FIXED_INNER_CENTER[2] - half - GRASP_Z_OFF
        assert abs(e_h) <= E_HEIGHT_MAX, (
            f"cube half {half * 1000:.1f}mm: table clamp 후 pad 조준 오차 {e_h * 1000:+.1f}mm "
            f"> E_HEIGHT_MAX {E_HEIGHT_MAX * 1000:.1f}mm — 이 크기는 FK 게이트에서 전멸한다")
    worst = TABLE_MARGIN + PAD_LOW_OFF - FIXED_INNER_CENTER[2] - min(SELF_CHECK_HALVES) - GRASP_Z_OFF
    print(f"self-check-geom: grasp height budget worst e_h={worst * 1000:+.1f}mm "
          f"(limit {E_HEIGHT_MAX * 1000:.1f}mm, half={min(SELF_CHECK_HALVES) * 1000:.1f}mm)")


def self_check_geom():
    """오프라인 기하 self-check — 후보 생성만 검증(planner 생성·GPU plan 불요. 단 모듈
    top-level 이 torch/curobo import 라 datagen 컨테이너 안에서 실행해야 한다).

    DR bell 근사 격자(pan 방위 ±80° × r 0.10~0.28) × cube yaw {0°, 22.5°, 45°}:
      (a) pre-twist R = Rz(pan)·Ry(-α)·R_TOPDOWN·Rz(ρ) 의 접근축이 pan 수직평면 안
          (=5-DOF manifold 위 존재증명; twist 포함 최종 R 은 2.79° cone 이 정상이라 pre 로 확인)
      (b) 최종 R closing 축 수평방위 ↔ 선택 face normal 잔차 < 1° (ρ 보상+feedback 수렴)
      (c) τ 결합 게이트: 후보는 항상 ≥5개(top-down 부근은 τ 무관) + yaw 어긋난 셀에서
          큰 |α| 후보가 실제로 배제됨"""
    _check_grasp_height_budget()
    _check_grip_ramp_within_cap()
    tau = math.radians(TAU_MAX_DEG)
    n_total = n_kept = n_cells = 0
    pruned_any = False  # τ 게이트가 실제로 한 번이라도 후보를 걸렀는지
    for az_deg in range(-80, 81, 20):
        for r in (0.10, 0.16, 0.22, 0.28):
            az = math.radians(az_deg)
            cube = (PAN_AXIS_XY[0] + r * math.cos(az),
                    PAN_AXIS_XY[1] + r * math.sin(az), 0.03)
            for yaw_deg in (0.0, 22.5, 45.0):
                h = math.radians(yaw_deg) / 2.0
                cube_bl = [0.0, 0.0, 0.03, math.cos(h), 0.0, 0.0, math.sin(h)]
                faces = PickPlacePlanner._cube_face_normals(cube_bl)
                assert len(faces) == 4, f"faces={len(faces)} != 4 (yaw={yaw_deg})"
                n_cells += 1
                cell_kept = 0
                # 크기 DR 사다리 전체(25~40 mm)를 돈다 — cube_half 는 chord-shift·pad 조준에
                # 들어가므로 크기가 바뀌면 후보 기하도 바뀐다(옛 체크는 40 mm 만 봤다).
                for half, a_deg in itertools.product(SELF_CHECK_HALVES, ALPHA_SCAN_DEG):
                    n_total += 1
                    cand = cand_pose_manifold(cube, faces, a_deg, tau, cube_half=half)
                    if cand is None:
                        pruned_any = True
                        continue
                    cell_kept += 1
                    _pre, _quat, meta = cand
                    a_rad = math.radians(a_deg)
                    pan = math.radians(meta["pan_deg"])
                    rho = math.radians(meta["rho_deg"])
                    dpsi = math.radians(meta["dpsi_deg"])
                    assert abs(dpsi) * abs(math.tan(a_rad)) <= tau + 1e-9, "gate leak"
                    # (a) manifold 존재증명: pre-twist 접근축 ⊥ pan 평면 법선
                    R_pre = _rz(pan) @ _ry(-a_rad) @ R_TOPDOWN @ _rz(rho)
                    plane_n = np.array([-math.sin(pan), math.cos(pan), 0.0])
                    off = abs(float(np.dot(R_pre[:, 2], plane_n)))
                    assert off < 1e-9, f"approach off pan-plane {off:.2e}"
                    # (b) uncapped: closing 수평방위 ⊥ face(resid<1°). rho-cap 셀은 의도적 미스얼라인
                    #     (넓은 jaw 관용·물리검증) — wrap90 로 ≤45° 보장, 런타임 FK face-gate(±40°)가 필터.
                    if not meta.get("rho_capped"):
                        R = R_pre @ TCP_TWIST
                        nf = np.asarray(meta["face_normal"], dtype=np.float64)
                        resid = abs(math.degrees(wrap90(
                            math.atan2(R[1, 0], R[0, 0]) - math.atan2(nf[1], nf[0]))))
                        assert resid < 1.0, (
                            f"closing-face resid {resid:.2f}° az={az_deg} r={r} "
                            f"yaw={yaw_deg} alpha={a_deg} half={half}")
                # (c) top-down 부근(τ 게이트: |Δψ|≤45° 서 α≤10° 는 항상 통과) ≥5개 보장
                # (크기마다 ≥5 → 사다리 전체로 ≥5×len(SELF_CHECK_HALVES))
                assert cell_kept >= 5 * len(SELF_CHECK_HALVES), \
                    f"cell kept={cell_kept} az={az_deg} r={r} yaw={yaw_deg}"
                n_kept += cell_kept
    assert pruned_any, "tau gate never pruned — gate dead?"
    print(f"self-check-geom: cells={n_cells} candidates kept={n_kept}/{n_total} "
          f"(tau={TAU_MAX_DEG}deg halves={[round(h, 4) for h in SELF_CHECK_HALVES]})")
    print("GEOM_SELFCHECK_OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5599)
    ap.add_argument("--max_batch_size", type=int, default=64,
                    help="cuRobo BatchMotionPlanner max_batch_size; batch dimension maps to env index.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--self-check-geom", action="store_true",
                    help="후보 생성 기하만 오프라인 검증(planner/GPU plan 불요)")
    a = ap.parse_args()
    if a.self_check_geom:
        self_check_geom()
        return
    pl = PickPlacePlanner(max_batch_size=a.max_batch_size)
    print("[planner] ready", flush=True)
    if a.self_test:
        q_identity = [1.0, 0.0, 0.0, 0.0]

        def _qz_bl(deg):
            h = math.radians(deg) / 2.0
            return [math.cos(h), 0.0, 0.0, math.sin(h)]

        init = [math.radians(v) for v in (0.0, -100.0, 90.0, 50.0, -90.0, -10.0)]
        # yaw 0°/22.5°/45° 케이스 — manifold 후보의 cube-yaw 정렬(ρ 보상) 경로까지 커버
        trajs, diagnostics = plan_batch(pl, {
            # env 마다 다른 큐브 크기 = 크기 DR 배선(per-env cube_half) 스모크.
            # z 는 SM 과 같은 규칙(TABLE_TOP_BASE + half)으로 각자 계산한다.
            "cubes": [[0.017, -0.253, TABLE_TOP_BASE + 0.0125, *q_identity],
                      [0.167, -0.133, TABLE_TOP_BASE + 0.020, *q_identity],
                      [0.017, -0.253, TABLE_TOP_BASE + 0.015, *_qz_bl(22.5)],
                      [0.100, -0.200, TABLE_TOP_BASE + 0.0175, *_qz_bl(45.0)]],
            "cube_half": [0.0125, 0.020, 0.015, 0.0175],
            "start": [init, init, init, init],
        })
        ok = all(t is not None for t in trajs)
        for i, d in enumerate(diagnostics):
            cand = d.get("candidate") or {}
            print(f"self-test env{i}: half={d.get('cube_half')} ok={d.get('ok')} "
                  f"phases={d.get('phases')} "
                  f"candidate_fail={cand.get('fail')} alpha={cand.get('alpha_deg')} "
                  f"rho={cand.get('rho_deg')} dpsi={cand.get('dpsi_deg')} "
                  f"profile_ms={d.get('profile_ms')}")
        print(f"self-test(4-env): {[len(t) if t else None for t in trajs]}")
        print("SELFTEST_OK" if ok else "SELFTEST_CHECK")
        return
    sock = zmq.Context().socket(zmq.REP); sock.bind(f"tcp://*:{a.port}")
    print(f"[planner] ZMQ REP :{a.port}", flush=True)
    serve_loop(pl, sock)


if __name__ == "__main__":
    main()
