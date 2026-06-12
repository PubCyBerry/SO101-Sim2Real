"""SO-101 pick-and-place 컨트롤러 (cube_desk 씬) — 해석적 IK + Cartesian waypoint follower.

Lula/DiffIK 등 수치 솔버 없이 닫힌 해(closed-form) IK + joint position action 만으로
`SimToReal-SO101-PickCube-v0` 에서 큐브 4개(30/40mm)를 그릇에 담는다. VLA Expert 데이터
생성용이라 2048-env 병렬에서 최종 placement 100% + 재시도 발동 ~0% 가 목표.

설계 — 두 조각
--------------
1. **기구학(SO101Kinematics)**: URDF(so_arm101.urdf) origin 체인을 pan 회전 평면의
   2-link(+wrist) 해석적 FK/IK 로 환원. 관절 역할:
     q1 (shoulder_pan) : 방위각 (pan 축 base -z → 부호 반전)
     q2,q3 (lift/elbow): pitch 평면 2-link IK
     q4 (wrist_flex)   : 툴 피치 (top-down -90°, 도달 불가 시 ik_reach 가 점진 완화)
     q5 (wrist_roll)   : 닫힘축 정렬(큐브 90° 대칭 접기) + roll_offset
   좌표 정합·상수는 --calibrate 실측 검증 (FK err 1.5mm). **이 부분은 검증되어 그대로 유지.**

2. **컨트롤러(SO101PickPlace)**: per-env 가 **Cartesian waypoint 리스트**(plan)를 따른다.
   - executor: 매 step 현재 명령 pose 를 다음 waypoint 로 **위치·pitch·roll 동시 선형 보간**한
     뒤 IK 로 풀어 joint 명령. 자세 불연속이 없으므로 슬램덩크가 구조적으로 불가능하다(별도
     anti-slam 코드 없음). z/slide/transport/rot ramp 가 보간 하나로 통합된다.
   - planner: 큐브당 7-waypoint plan 을 생성(상공→하강→파지→lift→운반→release→후퇴).
     grasp 는 **top-down 기본**(jaw·gripper 중점을 큐브 중심에 정렬), 도달 불가/장애물 시
     **angled fallback**(pitch 완화로 비스듬히). 닫힘축 roll 은 ±90° 우선·그릇/이웃 큐브 회피.
   - 안전망(드물게만 발동): 운반 중 drop·하강 중 drift 감지 → 해당 큐브 replan(큐브당 2 round).

성능: 매 step env 루프 직전 상태를 **1회 배치로 cpu numpy 스냅샷**(per-env .item() 동기화 제거)
→ 2048-env 가능. IK 는 순수 Python(CPU)이라 그대로.

GUI: 4-env 관전 + **R**=동일 셋업 재시작 / **N**=새 시드(DR 재샘플) 재시작.

실행:
    OMNI_KIT_ACCEPT_EULA=YES uv run --group isaac python \\
        scripts/environments/pick_cube_state_machine.py --num_envs 4 --headless

GUI 관전 (4-env, R/N 키):
    OMNI_KIT_ACCEPT_EULA=YES uv run --group isaac python \\
        scripts/environments/pick_cube_state_machine.py --num_envs 4

기구학 캘리브레이션(1회 진단):
    OMNI_KIT_ACCEPT_EULA=YES uv run --group isaac python \\
        scripts/environments/pick_cube_state_machine.py --calibrate --headless
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import math
import os
import sys
from dataclasses import dataclass
from enum import IntEnum

from isaaclab.app import AppLauncher

_LOG_PATH = os.path.abspath("outputs/so101_sm_progress.txt")
os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
open(_LOG_PATH, "w").close()
# C 레벨 크래시(access violation 등) Python traceback 을 파일로 덤프.
_FH_FILE = open(os.path.abspath("outputs/so101_sm_faulthandler.txt"), "w")
faulthandler.enable(file=_FH_FILE)


def log(msg: str) -> None:
    """Isaac Sim 이 gym.make 후 stdout 을 carb 로 재바인딩해 print 가 묻히므로
    파일 append + (소량 run 만)fsync + stderr 출력.

    대량 env(예: 2048)는 per-env 로그가 수만 줄이라 매 줄 fsync 가 wall-clock 을
    지배한다 → num_envs 큰 run 은 fsync 생략(flush 만, 크래시 안전성↓ 수용)."""
    with open(_LOG_PATH, "a") as f:
        f.write(msg + "\n")
        f.flush()
        if args.num_envs <= 64:
            os.fsync(f.fileno())
    print(msg, file=sys.__stderr__, flush=True)


def _vec3(s: str) -> tuple[float, float, float]:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected 'x,y,z'")
    return (parts[0], parts[1], parts[2])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="SO-101 pick-and-place (cube_desk, 해석적 IK + waypoint)")
parser.add_argument("--task", default="SimToReal-SO101-PickCube-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--active_objects", type=int, default=4, choices=[1, 2, 3, 4])
parser.add_argument("--object_radius_scale", type=float, default=1.0,
                    help="큐브 scatter DR 강도 (0=고정 spawn, 1=전체 workspace)")
parser.add_argument("--scatter_far", type=float, default=0.0,
                    help="scatter 범위를 base 에서 먼 쪽으로 이만큼 확장 (m). **0 권장** — base "
                         "범위(_CUBE_SCATTER_*)가 이미 reach 가장자리로 calibrate 됨. >0 은 그 "
                         "calibrated 한계를 넘겨 도달불가 spawn(build_plan None churn) 유발")
parser.add_argument("--scatter_z", type=float, default=0.0,
                    help="큐브 spawn z 분산 상한 (m). 0~이값 띄워 낙하 → 쌓이는 경우 발생. "
                         "**0 = flat(땅바닥)**. flat 분포 100% 달성 후 z-stacking 단계로 켠다")
parser.add_argument("--cube_sep", type=float, default=0.04,
                    help="큐브 간 최소 중심거리 (m). 작을수록 가까이/쌓임 가능 (0=DR 기본 유지)")
parser.add_argument("--container_angle_scale", type=float, default=1.0,
                    help="그릇 arc DR 강도 (0=고정, 1=기본 각도범위)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--calibrate", action="store_true",
                    help="기구학 진단 모드: FK 예측 vs 시뮬 실측 비교 후 종료 (컨트롤러 미실행)")
parser.add_argument("--check_spawns", action="store_true",
                    help="DR 적용 후 초기 spawn 의 **도달가능성·뭉침**을 env 전체로 점검 후 종료 "
                         "(컨트롤러 미실행). 닿을 수 없는 곳/뭉쳐서만 스폰되는지 검증용")
parser.add_argument("--replay_spawn", default=None,
                    help="실패 env taxonomy JSON 의 spawn pose 를 직접 세팅(DR 우회)해 재현·관전")
parser.add_argument("--replay_env_idx", type=int, default=-1,
                    help="replay_spawn JSON 에서 재현할 env 인덱스. -1=첫 non-clean 자동 선택")
# waypoint 전이·정착 파라미터
parser.add_argument("--joint_tol", type=float, default=0.09,
                    help="거친 단계 관절 수렴 판정 max|q_goal-q_now| (rad)")
parser.add_argument("--fine_joint_tol", type=float, default=0.025,
                    help="정밀 단계(하강/파지) 관절 수렴 판정 (rad)")
parser.add_argument("--reach_tol", type=float, default=0.012,
                    help="정밀 위치 WP(slide)의 **Cartesian 도달 판정** (m). 관절 수렴만으론 TCP 가 "
                         "수 mm 못 미쳐 close 가 큐브를 jaw 가장자리서 놓침(grasp miss) → 실측 TCP 가 "
                         "목표 이 거리 안에 들 때만 도달. 너무 작으면 도달 대기 길어짐(8mm→12mm 완화)")
parser.add_argument("--max_wp_steps", type=int, default=240,
                    help="한 waypoint 에서 수렴 못해도 넘어가는 step 상한 (30 Hz 기준 8초)")
parser.add_argument("--close_dwell", type=int, default=8,
                    help="그리퍼 닫힘 정착 step (30 Hz). gripper_speed 5rad/s 면 ~6 step 에 완전 닫힘")
parser.add_argument("--gripper_speed", type=float, default=5.0,
                    help="그리퍼 닫힘/열림 slew 상한 (rad/s). 물리 상한 5. 빠를수록 close 시간↓")
parser.add_argument("--release_wait", type=int, default=3,
                    help="release 전 그릇 위에서 닫은 채 정지 step (감속만 — 정체 방지). 하강 없이 떨굼")
parser.add_argument("--release_dwell", type=int, default=10,
                    help="그리퍼 열림 후 정지 step (큐브 낙하·open 완료 대기)")
parser.add_argument("--settle_steps", type=int, default=10,
                    help="reset 후 큐브 정착 대기 step (z 분산 spawn 낙하·정착 여유 포함)")
parser.add_argument("--max_round", type=int, default=3,
                    help="큐브당 replan 라운드 상한. (6 은 실패 큐브가 6×200tick 재시도 → 가장 느린 "
                         "env 가 max_total_steps cap 도달 → 대량 env 미완·성공률 추락. 3 이 균형)")
parser.add_argument("--max_total_steps", type=int, default=4000,
                    help="배치 전체 step 상한(straggler 안전장치). 초과 시 미완료 env fail")
# 높이/오프셋 (m)
parser.add_argument("--safe_height", type=float, default=0.15,
                    help="책상 윗면 기준 횡이동/운반 안전 고도. lift 후 이 고도로 올려 그대로 "
                         "운반(도착 후 추가 상승 없음). 낮으면 pan 회전 중 다른 큐브 침")
parser.add_argument("--grip_height", type=float, default=0.012,
                    help="grasp 시 TCP 의 **큐브 바닥** 기준 높이 (m). 책상 큐브는 바닥≈DESK 라 "
                         "DESK+grip 과 동일, 쌓여 들린 큐브는 그 바닥 기준이라 정확(Z-stacking 대응). "
                         "작을수록 깊게(바닥 근처). grasp z ladder 의 **깊은 쪽 끝**.")
parser.add_argument("--min_grip_depth", type=float, default=0.016,
                    help="grasp 시 큐브 top 아래 **최소 침투 깊이** (m). gz 를 이 깊이(얇은 쪽)부터 "
                         "grip_height 깊이까지 ladder 로 IK 시도해 첫 도달 해 채택 — 정확 깊이 고집 "
                         "안 하되 최소 이상은 보장(얇으면 윗면 긁음 방지). 얇을수록 self-clip·reach실패↓")
parser.add_argument("--release_spread", type=float, default=0.012,
                    help="release xy 분산 반경 (m). 작게 — 그릇 중심 근처에 살짝만 흩어 떨굼"
                         "(크면 가장자리로 흘림)")
parser.add_argument("--side_offset", type=float, default=0.035,
                    help="side-approach 횡오프셋 하한 (m). _side_offset 가 큐브 크기로 더 키움. "
                         "비킨 수직 하강 중 열린 jaw 가 큐브 윗면을 안 치게 비킴 (검증값 0.035)")
parser.add_argument("--slide_stop", type=float, default=0.005,
                    help="SLIDE 종점의 큐브 중심 잔여 거리 (m). 작을수록 큐브가 jaw 중앙 깊이 들어옴 "
                         "(close miss↓). 손가락 분리축이 진입축 ⊥ 이라 깊이 들어와도 정면 ram 없음")
parser.add_argument("--slide_speed", type=float, default=0.30, help="SLIDE 수평 진입 속도 (m/s)")
parser.add_argument("--pregrasp_height", type=float, default=0.04,
                    help="하강 전 비킨 지점 위 hover 높이 (m). 여기서 dwell 하며 중력 처짐 "
                         "보상(q_bias)을 수렴시켜 손가락 수직 확보 → 하강 찌름 방지")
parser.add_argument("--pregrasp_dwell", type=int, default=5,
                    help="pre-grasp hover 정착 step (q_bias 적분 수렴용). 중심 직하강이라 짧게")
parser.add_argument("--release_height", type=float, default=0.07,
                    help="그릇 바닥(책상 윗면) 기준 release 높이 (m)")
parser.add_argument("--bowl_clear_height", type=float, default=0.18,
                    help="그릇 위 통과·release 고도 (책상 기준 m). 매달린 큐브 바닥이 그릇 rim 을 "
                         "넘게 올려 **arc over** → 직선 운반이 rim 치는 것 방지(사용자). 도달 불가 시 "
                         "safe_height 로 폴백")
parser.add_argument("--release_roll_deg", type=float, default=90.0,
                    help="release 자세 wrist roll offset (deg). jaw 수평 → 퍼올림 없이 떨굼")
parser.add_argument("--lift_check", type=float, default=0.03,
                    help="파지 검증: lift 후 큐브 최소 상승량 (m)")
parser.add_argument("--drift_tol", type=float, default=0.015,
                    help="하강 중 큐브 xy 이탈 허용 (m). 초과 시 replan")
parser.add_argument("--drop_tol", type=float, default=0.055,
                    help="파지 후 큐브-TCP 거리 이 값 초과 시 drop 으로 간주 → replan")
parser.add_argument("--bowl_clear", type=float, default=0.12,
                    help="grasp 닫힘축 연장선이 그릇 중심에서 이만큼 못 떨어지면 roll 대안 채택")
# nudge (paddle push) — 도달불가/밀집 큐브 재배치
parser.add_argument("--nudge", action="store_true",
                    help="도달불가(특히 base 발치)·밀집 큐브를 닫은 그리퍼로 옆에서 밀어(paddle) "
                         "도달가능/트인 곳으로 옮긴 뒤 grasp. far-pull 은 기구학상 best-effort")
parser.add_argument("--nudge_dist", type=float, default=0.06, help="paddle push 큐브 이동 거리 (m)")
parser.add_argument("--max_nudge", type=int, default=2, help="큐브당 nudge 시도 상한")
parser.add_argument("--nudge_r_near", type=float, default=0.17,
                    help="base frame 반경 이보다 작으면 inner-reach(발치) → base 반대로 밀기 (m)")
parser.add_argument("--nudge_r_far", type=float, default=0.32,
                    help="base frame 반경 이보다 크면 외측 reach 한계 → base 쪽으로 best-effort (m)")
parser.add_argument("--cube_clear", type=float, default=0.05,
                    help="grasp 닫힘축 연장선과 다른 큐브 사이 최소 거리 (m)")
# 그리퍼 명령 (joint target, rad)
parser.add_argument("--gripper_open", type=float, default=0.65,
                    help="30mm 큐브용 열림 joint target (rad)")
parser.add_argument("--gripper_open_large", type=float, default=0.85,
                    help="40mm 큐브용 열림 joint target (rad)")
parser.add_argument("--gripper_close", type=float, default=-0.05)
# 속도 (m/s, Cartesian)
parser.add_argument("--descend_speed", type=float, default=0.32, help="하강 속도")
parser.add_argument("--lift_speed", type=float, default=0.55, help="상승/후퇴 속도")
parser.add_argument("--transport_speed", type=float, default=0.70,
                    help="수평 운반 순항 속도 상한. 너무 빠르면 marginal 그립(특히 40mm)을 전단해 drop")
parser.add_argument("--accel", type=float, default=6.0,
                    help="사다리꼴 속도 프로파일 가·감속도 (m/s²). 등속 선형(텔레포트 느낌·급출발 "
                         "큐브 밀침) 대신 양끝 0속도로 가속→순항(*_speed)→감속. 클수록 순항 빨리 "
                         "도달(시간↓), 작을수록 더 부드럽고 느림")
parser.add_argument("--min_speed", type=float, default=0.04,
                    help="속도 프로파일 하한 (m/s). 감속 말단 creep-stall 방지 floor")
# GUI 초기 카메라(사이드뷰) — world 좌표. headless 에선 무시됨.
parser.add_argument("--view_eye", type=_vec3, default=(3.05, -0.78, 1.02),
                    help="GUI 카메라 위치 'x,y,z'")
parser.add_argument("--view_lookat", type=_vec3, default=(1.74, -0.38, 0.74),
                    help="GUI 카메라 주시점 'x,y,z'")
parser.add_argument("--video", action="store_true",
                    help="사이드뷰를 mp4 로 녹화해 docs/ 에 저장")
parser.add_argument("--video_length", type=int, default=2000, help="녹화 최대 프레임 수")
parser.add_argument("--video_name", default="so101_pick_place",
                    help="docs/ 에 저장할 mp4 파일명 prefix")
parser.add_argument("--taxonomy", default=None,
                    help="per-env outcome taxonomy JSON 저장 경로 (미지정 시 outputs/sm_scale_<N>_seed<S>.json)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# 녹화는 viewport rgb 렌더가 필요 → 카메라 활성화.
if args.video:
    args.enable_cameras = True

# AppLauncher 부팅 (isaac 모듈 import 전에).
# vars(args) 전체를 넘기면 view_eye 같은 tuple 커스텀 인자가 carb 설정으로 흘러가
# Windows 에서 _prepare_ui access violation 발생 → 실제 사용 키만 화이트리스트 필터.
_LAUNCHER_KEYS = {
    "headless", "enable_cameras", "experience", "device", "cpu",
    "disable_fabric", "offscreen_render", "kit_args",
}
_launcher_args = {k: v for k, v in vars(args).items() if k in _LAUNCHER_KEYS}
app_launcher = AppLauncher(_launcher_args)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# 부팅 이후 import
# ---------------------------------------------------------------------------

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import sim_to_real  # noqa: E402, F401  (Gym 환경 등록 트리거)
from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import (  # noqa: E402
    BOWL_HEIGHT_RANGE,
    BOWL_SUCCESS_RADIUS,
    PickCubeEnvCfg,
)
from sim_to_real.utils.constant import BOWL_NAME, CUBE_NAMES  # noqa: E402

# cube_desk 책상 윗면 world z (placed 판정 기준).
DESK_TOP_Z = 0.705

# 큐브 한 변 길이 (author 스크립트 CUBE_SCALES): Cube1/2=30mm, Cube3/4=40mm.
CUBE_SIZES = {"Cube1": 0.030, "Cube2": 0.030, "Cube3": 0.040, "Cube4": 0.040}

# gripper action 의 default offset (PickCubeEnvCfg init joint_pos["gripper"]).
# SlewLimitedJointPositionAction: desired = raw*scale(1.0) + offset → raw = target - offset.
GRIPPER_ACTION_OFFSET = 0.20


# ---------------------------------------------------------------------------
# Part A — 해석적 기구학 (검증됨 — 변경 없음)
# ---------------------------------------------------------------------------


class SO101Kinematics:
    """SO-101 5축 닫힌 해 FK/IK (robot base_link frame 기준).

    URDF(assets/robots/urdf/so_arm101.urdf) origin 체인을 base frame 에서 전개한 결과:
      · shoulder_pan 축: base (PAN_X, 0) 위치의 -z 축 → +q1 명령 = world yaw -q1
      · lift/elbow/wrist_flex 축: 모두 pan 회전 평면의 같은 pitch 축(+y 방향)
      · zero pose TCP = base (0.391, 0.000, 0.227) — pan 평면 위에 정확히 위치
      · wrist_roll(q5) 회전 시 TCP 가 roll 축 주위 반경 ROLL_RHO(7.9mm) 원을 돌므로
        q5 확정 후 lateral 1차 보정을 적용한다.
    """

    # pan 축 base 위치/높이 (URDF shoulder_pan origin)
    PAN_X = 0.0388353
    # pan 축 기준 lift 축 radial 오프셋·base 기준 lift 축 높이
    LIFT_R = 0.0303992
    LIFT_Z = 0.0624 + 0.0542  # = 0.1166
    # 평면 링크 길이: lift→elbow, elbow→wrist_flex, wrist_flex→TCP(gripper_frame)
    L1 = math.hypot(0.11257, 0.028)    # 0.11600
    L2 = math.hypot(0.1349, 0.0052)    # 0.13500
    L3 = math.hypot(0.1592, 0.0079)    # 0.15940 (wrist_roll origin + gripper_frame 합성)
    # zero-pose 평면각 (수평 기준, 위가 +)
    TH1_0 = math.atan2(0.11257, 0.028)   # 상완 76.0° 위
    TH2_0 = math.atan2(0.0052, 0.1349)   # 전완 2.2° 위
    TH3_0 = math.atan2(-0.0079, 0.1592)  # wrist→TCP -2.8°
    # wrist_roll 축 ↔ TCP lateral 반경 (gripper_frame offset 의 roll 축 수직 성분)
    ROLL_RHO = 0.0079

    # joint limits (URDF): pan, lift, elbow, wrist_flex, wrist_roll
    JOINT_LIMITS = [
        (-1.91986, 1.91986),
        (-1.74533, 1.74533),
        (-1.69, 1.69),
        (-1.65806, 1.65806),
        (-2.74385, 2.84121),
    ]

    def fk_tcp(self, q: list[float]) -> tuple[float, float, float]:
        """관절각 → TCP(gripper_frame) 위치, base frame. 검증·INIT 동기화용."""
        q1, q2, q3, q4, q5 = (float(v) for v in q[:5])
        th1 = self.TH1_0 - q2
        th2 = th1 + (self.TH2_0 - self.TH1_0) - q3
        th3 = th2 + (self.TH3_0 - self.TH2_0) - q4
        pr = (self.LIFT_R + self.L1 * math.cos(th1) + self.L2 * math.cos(th2)
              + self.L3 * math.cos(th3))
        pz = (self.LIFT_Z + self.L1 * math.sin(th1) + self.L2 * math.sin(th2)
              + self.L3 * math.sin(th3))
        lat = self.ROLL_RHO * math.sin(q5)
        x = self.PAN_X + pr * math.cos(q1) - lat * math.sin(q1)
        y = -(pr * math.sin(q1) + lat * math.cos(q1))
        z = pz
        return (x, y, z)

    def ik(self, tcp: tuple[float, float, float], grasp_yaw: float,
           pitch: float = -math.pi / 2,
           q_ref: list[float] | None = None,
           roll_offset: float = 0.0) -> list[float] | None:
        """TCP 목표(base frame) + 손가락 닫힘축 yaw → 관절각 5개. 도달 불가 시 None.

        pitch: 툴 접근 피치(wrist→TCP 평면각). -π/2 = 수직 top-down.
        roll_offset: q5 에 더하는 고정 회전 (rad). 큐브 90° 대칭을 이용해 닫힘축을
        ±90° 돌린 대안 grasp 자세 — 장애물 회피용.
        """
        x, y, z = tcp
        dx, dy = x - self.PAN_X, y
        r = math.hypot(dx, dy)
        if r < 1e-6:
            return None
        q1 = -math.atan2(dy, dx)  # pan 축 = base -z → 부호 반전

        q5 = self._fold_45(grasp_yaw + q1) + roll_offset

        # q5 lateral 보정: TCP 가 roll 축에서 ρ·sin(q5) 만큼 옆으로 벗어남 → radial 로 환원
        lat = self.ROLL_RHO * math.sin(q5)
        r_eff = math.sqrt(max(r * r - lat * lat, 1e-9))

        # wrist_flex 축 위치 역산 (pitch 방향으로 L3 제거)
        pr = r_eff - self.LIFT_R
        pz = z - self.LIFT_Z
        wr = pr - self.L3 * math.cos(pitch)
        wz = pz - self.L3 * math.sin(pitch)

        # 2-link planar IK
        d2 = wr * wr + wz * wz
        c_rel = (d2 - self.L1 * self.L1 - self.L2 * self.L2) / (2.0 * self.L1 * self.L2)
        if abs(c_rel) > 1.0:
            return None  # 작업 반경 밖
        rel = math.acos(c_rel)

        base_ang = math.atan2(wz, wr)
        candidates: list[list[float]] = []
        for sign in (-1.0, 1.0):
            th1 = base_ang - math.atan2(self.L2 * math.sin(sign * rel),
                                        self.L1 + self.L2 * math.cos(sign * rel))
            th2 = th1 + sign * rel
            th3 = pitch
            q2 = self.TH1_0 - th1
            q3 = (self.TH2_0 - self.TH1_0) - (th2 - th1)
            q4 = (self.TH3_0 - self.TH2_0) - (th3 - th2)
            q = [q1, q2, q3, q4, q5]
            if all(lo <= v <= hi for v, (lo, hi) in zip(q, self.JOINT_LIMITS)):
                candidates.append(q)
        if not candidates:
            return None
        if q_ref is None or len(candidates) == 1:
            return candidates[0]
        return min(candidates,
                   key=lambda q: sum(abs(a - b) for a, b in zip(q, q_ref)))

    def ik_reach(self, tcp: tuple[float, float, float], grasp_yaw: float,
                 pitch_min: float = math.radians(-90),
                 pitch_max: float = math.radians(-30),
                 q_ref: list[float] | None = None,
                 roll_offset: float = 0.0) -> tuple[list[float], float] | None:
        """top-down 우선, 도달 불가 시 pitch 를 점진 완화하며 첫 해와 채택 pitch 반환.

        5-DOF position 우선·orientation best-effort 규약 (AGENTS.md).
        pitch_max -30°: SO-101 reach 가장자리(그릇 등 r>0.33)는 비스듬해야만 닿는다.
        """
        n_steps = 13
        for i in range(n_steps):
            pitch = pitch_min + (pitch_max - pitch_min) * i / (n_steps - 1)
            q = self.ik(tcp, grasp_yaw, pitch=pitch, q_ref=q_ref, roll_offset=roll_offset)
            if q is not None:
                return q, pitch
        return None

    @staticmethod
    def _fold_45(a: float) -> float:
        return (a + math.pi / 4) % (math.pi / 2) - math.pi / 4


# ---------------------------------------------------------------------------
# Domain Randomization — events 만 강도 조정
# ---------------------------------------------------------------------------


def _apply_dr(env_cfg: PickCubeEnvCfg) -> None:
    """object_radius_scale / container_angle_scale 을 reset 이벤트에 in-place 반영."""
    from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import (
        _CUBE_SCATTER_CENTER,
        _CUBE_SCATTER_X_RANGE,
        _CUBE_SCATTER_Y_RANGE,
    )

    scatter = getattr(env_cfg.events, "randomize_cubes", None)
    s = max(0.0, float(args.object_radius_scale))
    if scatter is not None and s <= 0.0:
        env_cfg.events.randomize_cubes = None
    elif scatter is not None and s != 1.0:
        cx, cy = _CUBE_SCATTER_CENTER
        x_lo, x_hi = _CUBE_SCATTER_X_RANGE
        y_lo, y_hi = _CUBE_SCATTER_Y_RANGE
        scatter.params["x_range"] = (cx - (cx - x_lo) * s, cx + (x_hi - cx) * s)
        scatter.params["y_range"] = (cy - (cy - y_lo) * s, cy + (y_hi - cy) * s)

    # DR 확장 (expert 데이터 다양성): base 에서 먼 쪽 확장 + Z 분산(쌓임) + 간격 축소
    if scatter is not None and env_cfg.events.randomize_cubes is not None:
        f = float(args.scatter_far)
        xr = scatter.params["x_range"]
        yr = scatter.params["y_range"]
        scatter.params["x_range"] = (xr[0] - f, xr[1] + f)   # x 양극단(base 좌우 먼 쪽)
        scatter.params["y_range"] = (yr[0], yr[1] + f)       # +y = base 에서 먼 쪽
        scatter.params["z_range"] = (0.0, float(args.scatter_z))
        if args.cube_sep > 0.0:
            scatter.params["min_cube_sep"] = float(args.cube_sep)

    bowl = getattr(env_cfg.events, "randomize_bowl", None)
    a = float(args.container_angle_scale)
    if bowl is not None and a != 1.0:
        lo, hi = bowl.params["angle_range_deg"]
        bowl.params["angle_range_deg"] = (lo * a, hi * a)


# ---------------------------------------------------------------------------
# Part B — 좌표 정합 (검증됨 — 변경 없음) + numpy quat 헬퍼
# ---------------------------------------------------------------------------


def _quat_to_yaw(quat) -> float:
    """wxyz quat → world yaw (rad). torch tensor 또는 numpy/list 모두 허용."""
    w, x, y, z = (float(v) for v in quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _np_quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """wxyz quat 으로 벡터 v 회전 (numpy, CPU)."""
    w, qv = q[0], q[1:4]
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


# USD root body frame ↔ URDF base_link frame 정합 (캘리브레이션 8자세 실측 fit, 잔차<0.3mm):
#   URDF = rot(+90°) · (USD_root_local - BASE_XY_OFFSET),  z - BASE_Z_OFFSET
BASE_XY_OFFSET = (0.0204, 0.0157)
BASE_Z_OFFSET = 0.0325
BASE_YAW_OFFSET = math.pi / 2  # USD root frame yaw → URDF base frame yaw 가산값


def _world_to_base(p_w: torch.Tensor, robot, e: int) -> tuple[float, float, float]:
    """world 좌표(torch) → URDF base_link frame. 캘리브레이션 모드용."""
    root_p = robot.data.root_pos_w[e, :3]
    yaw = _quat_to_yaw(robot.data.root_quat_w[e])
    d = (p_w - root_p).detach().cpu().tolist()
    c, s = math.cos(-yaw), math.sin(-yaw)
    bx = c * d[0] - s * d[1]
    by = s * d[0] + c * d[1]
    return (-(by - BASE_XY_OFFSET[1]), bx - BASE_XY_OFFSET[0], d[2] - BASE_Z_OFFSET)


def _world_to_base_np(p_w: np.ndarray, root_p: np.ndarray, root_yaw: float
                      ) -> tuple[float, float, float]:
    """world 좌표(numpy) → URDF base_link frame. 스냅샷 기반 핫패스용 (sync 없음)."""
    d = p_w - root_p
    c, s = math.cos(-root_yaw), math.sin(-root_yaw)
    bx = c * d[0] - s * d[1]
    by = s * d[0] + c * d[1]
    return (-(by - BASE_XY_OFFSET[1]), bx - BASE_XY_OFFSET[0], float(d[2]) - BASE_Z_OFFSET)


# gripper body → TCP(gripper_frame) 오프셋 (캘리브레이션 실측, base frame FK 와 정합).
GRASP_OFF = np.array([-0.0079, -0.000218121, -0.0981274])


# ---------------------------------------------------------------------------
# Part C — 캘리브레이션 모드 (--calibrate, 검증됨 — 변경 없음)
# ---------------------------------------------------------------------------


def run_calibration(env) -> None:
    """zero pose + 단일 joint 스텝 응답으로 FK 모델(상수·부호) 검증. 1회 진단용."""
    from isaaclab.utils.math import quat_apply

    kin = SO101Kinematics()
    robot = env.scene["robot"]
    names = list(robot.data.body_names)
    g_idx = names.index("gripper")
    grasp_off = torch.tensor(GRASP_OFF.tolist(), device=env.device)

    def measured_tcp_base() -> tuple[float, float, float]:
        gp = robot.data.body_pos_w[0, g_idx]
        gq = robot.data.body_quat_w[0, g_idx]
        tcp_w = gp + quat_apply(gq.unsqueeze(0), grasp_off.unsqueeze(0)).squeeze(0)
        return _world_to_base(tcp_w, robot, 0)

    def settle(q5: list[float], n: int = 90) -> None:
        act = torch.zeros((env.num_envs, 6), device=env.device)
        act[0, :5] = torch.tensor(q5, device=env.device)
        act[0, 5] = 0.5 - GRIPPER_ACTION_OFFSET  # 반개방 고정
        for _ in range(n):
            if not simulation_app.is_running():
                return
            env.step(act)

    log("=" * 72)
    log("[CALIB] FK 예측 vs 시뮬 실측 (base frame, 단위 m)")
    rq = robot.data.root_quat_w[0].detach().cpu().tolist()
    log(f"[CALIB] root pos={robot.data.root_pos_w[0, :3].detach().cpu().tolist()} "
        f"quat(wxyz)={rq} yaw={math.degrees(_quat_to_yaw(robot.data.root_quat_w[0])):.1f}°")
    log(f"[CALIB] 모델 상수: L1={kin.L1:.5f} L2={kin.L2:.5f} L3={kin.L3:.5f} "
        f"LIFT=({kin.LIFT_R:.4f},{kin.LIFT_Z:.4f}) PAN_X={kin.PAN_X:.4f}")

    cases = [
        ("zero", [0.0, 0.0, 0.0, 0.0, 0.0]),
        ("pan+0.4", [0.4, 0.0, 0.0, 0.0, 0.0]),
        ("lift+0.3", [0.0, 0.3, 0.0, 0.0, 0.0]),
        ("elbow+0.4", [0.0, 0.0, 0.4, 0.0, 0.0]),
        ("wrist+0.4", [0.0, 0.0, 0.0, 0.4, 0.0]),
        ("roll+0.6", [0.0, 0.0, 0.0, 0.0, 0.6]),
        ("combo", [0.3, 0.4, 0.5, -0.3, 0.4]),
        ("grasp형", [0.0, 0.9, 1.2, -0.5, 0.0]),
    ]
    max_err = 0.0
    for name, q in cases:
        settle(q)
        q_now = robot.data.joint_pos[0, :5].detach().cpu().tolist()
        pred = kin.fk_tcp(q_now)
        meas = measured_tcp_base()
        err = math.dist(pred, meas)
        max_err = max(max_err, err)
        log(f"[CALIB] {name:10s} q={['%+.3f' % v for v in q_now]}")
        log(f"        pred=({pred[0]:+.4f},{pred[1]:+.4f},{pred[2]:+.4f})  "
            f"meas=({meas[0]:+.4f},{meas[1]:+.4f},{meas[2]:+.4f})  err={err * 1000:.1f}mm")

    log("=" * 72)
    log(f"[CALIB] max err = {max_err * 1000:.1f}mm  "
        f"({'OK (<5mm)' if max_err < 0.005 else 'FAIL — 상수/부호 재보정 필요'})")


# ---------------------------------------------------------------------------
# Part D — Waypoint + 컨트롤러
# ---------------------------------------------------------------------------


@dataclass
class WP:
    """Cartesian waypoint. pos=world xyz, pitch=툴 피치, yaw=닫힘축 world yaw,
    roll=q5 roll_offset, grip=그리퍼 target, speed=Cartesian 보간 속도(m/s)."""
    pos: tuple[float, float, float]
    pitch: float
    yaw: float
    roll: float
    grip: float
    speed: float
    tol: float
    settle: int
    tag: str


class Pose:
    """현재 명령 Cartesian 상태 (보간 누적용)."""
    __slots__ = ("x", "y", "z", "pitch", "yaw", "roll")

    def __init__(self, x, y, z, pitch, yaw, roll):
        self.x, self.y, self.z = x, y, z
        self.pitch, self.yaw, self.roll = pitch, yaw, roll


class Outcome(IntEnum):
    RUNNING = 0
    DONE = 1


class SO101PickPlace:
    """num_envs 병렬 pick-and-place 컨트롤러 (해석적 IK + Cartesian waypoint follower).

    매 step: (1) 상태 1회 배치 스냅샷 → (2) per-env executor 가 다음 waypoint 로 보간·IK →
    (3) env.step([N,6]) 1회.
    """

    # 시작/대기 자세: shoulder_lift 최대 up, elbow_flex 최대 fold, wrist_flex -20°,
    # wrist_roll +90°(기본 자세) — 작업영역 위로 높이 접은 ready pose.
    HOME_Q = [0.0, -1.72, 1.66, math.radians(-20.0), math.radians(90.0)]
    HOME_GRIP = 0.0  # 대기 시 그리퍼 0°(닫힘)
    BIAS_KI = 0.06
    BIAS_MAX = 0.35

    def __init__(self, env, active_cubes: list[str]) -> None:
        self.env      = env
        self.scene    = env.unwrapped.scene
        self.device   = env.unwrapped.device
        self.num_envs = env.unwrapped.num_envs
        self.robot    = self.scene["robot"]
        self.kin      = SO101Kinematics()
        self.cubes    = list(active_cubes)

        self._g_idx = list(self.robot.data.body_names).index("gripper")
        self._grasp_off_t = torch.tensor(GRASP_OFF.tolist(), device=self.device)

        # 중력 처짐 보상 적분기 [N,5] (PD stiffness 17.8 정적오차 제거).
        self.q_bias = torch.zeros((self.num_envs, 5), device=self.device)

        # 스냅샷 컨테이너 (매 step _snapshot 으로 채움)
        self.snap: dict = {}

        self.reset_all()

    # ---- per-env 상태 초기화 --------------------------------------------

    def reset_all(self) -> None:
        """reset 직후 호출: 전 env 의 컨트롤러 상태를 초기화."""
        N = self.num_envs
        self.remaining: list[list[str]] = [list(self.cubes) for _ in range(N)]
        self.cur_cube : list[str | None] = [None] * N
        self.rounds   : list[dict[str, int]] = [{c: 0 for c in self.cubes} for _ in range(N)]
        self.plan     : list[list[WP]] = [[] for _ in range(N)]
        self.idx      : list[int]   = [0] * N
        self.dwell    : list[int]   = [0] * N
        self.wp_steps : list[int]   = [0] * N
        self.settle_n : list[int]   = [0] * N      # reset 후 정착 카운터
        self.q_cmd    : list[list[float]] = [list(self.HOME_Q) for _ in range(N)]
        self.cur_pose : list[Pose | None] = [None] * N
        self.seg_v    : list[float] = [0.0] * N    # 현재 segment Cartesian 속도(사다리꼴 프로파일)
        self.grasp_z0 : list[float] = [0.0] * N
        self.grasp_ref: list[np.ndarray | None] = [None] * N  # plan 시점 큐브 xy (drift 기준)
        self.n_placed : list[int] = [0] * N  # 그릇에 담은 큐브 수 (release 분산 인덱스)
        self.drop_logged: list[bool] = [False] * N  # drop 징후 1회만 기록 (attempt 별)
        self.nudge_n  : list[dict[str, int]] = [{c: 0 for c in self.cubes} for _ in range(N)]
        self.is_nudge : list[bool] = [False] * N  # 현재 plan 이 nudge(재배치)인가
        self.outcome  : list[Outcome] = [Outcome.RUNNING] * N
        self.events   : list[list[dict]] = [[] for _ in range(N)]  # taxonomy: retry/나쁜궤적
        self.q_bias.zero_()

    # ---- 스냅샷 (배치 GPU→CPU, per-env sync 제거) -----------------------

    def _snapshot(self) -> None:
        d = self.robot.data
        s = self.snap
        s["jp"] = d.joint_pos[:, :6].detach().cpu().numpy()
        s["rp"] = d.root_pos_w[:, :3].detach().cpu().numpy()
        s["rq"] = d.root_quat_w.detach().cpu().numpy()
        s["gp"] = d.body_pos_w[:, self._g_idx, :].detach().cpu().numpy()
        s["gq"] = d.body_quat_w[:, self._g_idx, :].detach().cpu().numpy()
        op, oq = {}, {}
        for name in self.cubes + [BOWL_NAME]:
            od = self.scene[name].data
            op[name] = od.root_pos_w[:, :3].detach().cpu().numpy()
            oq[name] = od.root_quat_w.detach().cpu().numpy()
        s["op"], s["oq"] = op, oq

    # ---- 스냅샷 기반 쿼리 (numpy, sync 없음) ----------------------------

    def obj_pos(self, name: str, e: int) -> np.ndarray:
        return self.snap["op"][name][e]

    def obj_yaw(self, name: str, e: int) -> float:
        return _quat_to_yaw(self.snap["oq"][name][e])

    def root_yaw(self, e: int) -> float:
        return _quat_to_yaw(self.snap["rq"][e])

    def tcp_meas(self, e: int) -> np.ndarray:
        """gripper body + grasp offset → TCP(gripper_frame) world 실측."""
        return self.snap["gp"][e] + _np_quat_apply(self.snap["gq"][e], GRASP_OFF)

    def joints(self, e: int) -> np.ndarray:
        return self.snap["jp"][e]

    def _to_base(self, p_w: np.ndarray, e: int) -> tuple[float, float, float]:
        return _world_to_base_np(p_w, self.snap["rp"][e], self.root_yaw(e))

    def _safe_z(self) -> float:
        return DESK_TOP_Z + args.safe_height

    # ---- IK 래퍼 --------------------------------------------------------

    def _ik(self, e: int, pos_w, pitch: float, yaw_w: float, roll: float
            ) -> list[float] | None:
        """world 목표 + 자세 → base 변환 → 고정 pitch IK. q_ref=현재 명령(가지 연속)."""
        tb = self._to_base(np.asarray(pos_w, dtype=float), e)
        yaw_b = yaw_w - self.root_yaw(e) + BASE_YAW_OFFSET
        return self.kin.ik(tb, yaw_b, pitch=pitch, q_ref=self.q_cmd[e], roll_offset=roll)

    def _ik_reach(self, e: int, pos_w, yaw_w: float, roll: float
                  ) -> tuple[list[float], float] | None:
        """top-down 우선 pitch 스캔 → (q, pitch). 도달 불가 None."""
        tb = self._to_base(np.asarray(pos_w, dtype=float), e)
        yaw_b = yaw_w - self.root_yaw(e) + BASE_YAW_OFFSET
        return self.kin.ik_reach(tb, yaw_b, q_ref=self.q_cmd[e], roll_offset=roll)

    def _reachable(self, e: int, pos_w, yaw_w: float = 0.0, roll: float = 0.0) -> bool:
        return self._ik_reach(e, pos_w, yaw_w, roll) is not None

    # ---- 그리퍼/크기 헬퍼 ----------------------------------------------

    @staticmethod
    def _open_cmd(cube: str) -> float:
        """큐브 크기 → 그리퍼 열림 target (rad). 30/40mm 두 검증점 선형 보간·외삽 →
        임의 크기 일반화 (큐브가 손가락 사이로 들어올 만큼 벌림)."""
        size = CUBE_SIZES.get(cube, 0.030)
        frac = (size - 0.030) / (0.040 - 0.030)
        cmd = args.gripper_open + frac * (args.gripper_open_large - args.gripper_open)
        return max(0.2, min(1.74, cmd))

    @staticmethod
    def _side_offset(cube: str) -> float:
        """큐브 크기 → side-approach 비킴 거리 (m). 큐브 반폭 + 여유 → 임의 크기 일반화.
        충분히 비켜야 비킨 수직 하강 중 열린 jaw 가 큐브 윗면을 안 침(descend clip 방지)."""
        size = CUBE_SIZES.get(cube, 0.030)
        return max(args.side_offset, size * 0.5 + 0.018)

    # ---- planner --------------------------------------------------------

    def _closing_axis(self, e: int, cube: str, roll: float) -> float:
        """닫힘축(fixed finger) world 방위 (rad). **q1(pan) 의존 포함** — 정확값.

        q5(wrist_roll) = fold_45(yaw_b + q1) + roll 이고, 닫힘축 world 방위는 q1·q5 에
        함께 의존한다. q1 을 빼먹은 근사(fold_45(yaw)+roll)는 base 중심에서 벗어난 큐브의
        slide 방향을 틀어 fixed finger 가 큐브를 정면으로 밀어버린다.
        """
        p = self.obj_pos(cube, e)
        yaw = self.obj_yaw(cube, e)
        root_yaw = self.root_yaw(e)
        tb = self._to_base(p, e)
        q1 = -math.atan2(tb[1], tb[0] - SO101Kinematics.PAN_X)
        q5_base = self.kin._fold_45((yaw - root_yaw + BASE_YAW_OFFSET) + q1)
        yaw_fixed_w = (q5_base - q1) - BASE_YAW_OFFSET + root_yaw
        return yaw_fixed_w + roll

    def _grasp_setup(self, e: int, cube: str):
        """grasp 자세·진입 방향 결정. 반환 (roll, (sdx, sdy) slide 단위벡터, (bx0, by0) 비킴점).

        · roll 은 **±90° 만**(사용자 기본 자세). 큐브 90° 대칭이라 둘 다 동등 grasp 인데
          접근 방향이 180° 반대라, 그릇/이웃 큐브 없는 쪽을 골라 장애물 회피(=사용자가 말한
          "닫는 축에 장애물 있으면 90° 돌려").
        · slide(비킴 진입)는 `_closing_axis` 방향 = **jaw gap 이 열린 방향** → 큐브가 두 손가락
          사이로 들어옴. 손가락 분리축(사용자가 말한 닫는 축)은 이에 ⊥ → 닫는 축이 이동
          방향과 수직(정면 ram 없음). roll 0 은 닫는 축 ∥ 이동이라 ram → 제외.
        · 손가락 연장선·비킴점·slide 경로가 그릇/이웃 큐브와 충돌하면 점수↓.
        """
        p = self.obj_pos(cube, e)
        bowl = self.obj_pos(BOWL_NAME, e)
        others = [self.obj_pos(c, e) for c in self.remaining[e] if c != cube]
        so = self._side_offset(cube)   # 큐브 크기별 비킴 거리
        fh = 0.045   # 손가락 분리축(⊥) 연장 평가 반길이 (m)
        px, py = float(p[0]), float(p[1])
        bx, by = float(bowl[0]), float(bowl[1])
        best = None
        # ±90 = 닫는 축 ⊥ yaw(기본), 0/π = 90° 돌린 자세(나란한 큐브 회피용). 막힘 적은
        # 후보 채택, ±90 에 가산점(기본 자세 우선). 진입로(slide, ax)+손가락 끝(⊥) 둘 다 검사.
        for pref, roll in ((True, math.pi / 2), (True, -math.pi / 2),
                           (False, 0.0), (False, math.pi)):
            ax = self._closing_axis(e, cube, roll)      # gap-open(=slide·비킴) 방향
            dx, dy = math.cos(ax), math.sin(ax)
            fx, fy = -math.sin(ax), math.cos(ax)         # 손가락 분리축(⊥)
            bx0, by0 = px - so * dx, py - so * dy         # 비킨 하강점(진입축 따라)
            pts = [
                (px + fh * fx, py + fh * fy),             # 손가락 끝 A (⊥)
                (px - fh * fx, py - fh * fy),             # 손가락 끝 B (⊥)
                (bx0, by0),                               # 비킨 하강점(진입로)
                (0.5 * (bx0 + px), 0.5 * (by0 + py)),     # slide 중간(진입로)
            ]
            score = 1e9
            for cxx, cyy in pts:
                score = min(score, math.hypot(cxx - bx, cyy - by) - args.bowl_clear)
                for o in others:
                    score = min(score, math.hypot(cxx - float(o[0]), cyy - float(o[1]))
                                - args.cube_clear)
            adj = score + (0.04 if pref else 0.0)         # ±90 기본 선호
            if best is None or adj > best[0]:
                best = (adj, roll, (dx, dy), (bx0, by0))
        return best[1], best[2], best[3]

    def _grasp_pose(self, e: int, cube: str):
        """grasp 해 (gx, gy, gz, yaw, roll, pitch, bx0, by0) 또는 None.

        gz = 큐브 top 아래 침투 깊이를 **min_grip_depth(얇은 쪽)부터 grip_height 깊이까지**
        ladder 로 IK 시도해 첫 도달 해를 채택한다. 정확한 깊이를 고집하지 않고 '최소 이상이면
        grip'(사용자) → reach 가장자리에 강하고, 얇을수록 하강 self-clip 도 준다. 단 min_grip_depth
        floor 가 너무 얇아 손가락이 윗면을 긁는 것을 막는다.
        """
        roll, (sdx, sdy), (bx0, by0) = self._grasp_setup(e, cube)
        p = self.obj_pos(cube, e)
        yaw = self.obj_yaw(cube, e)
        size = CUBE_SIZES.get(cube, 0.030)
        top = float(p[2]) + size * 0.5
        gx = float(p[0]) - args.slide_stop * sdx
        gy = float(p[1]) - args.slide_stop * sdy
        deep = size - args.grip_height            # top 아래 최대 침투(= grip_height 깊이)
        lo = min(args.min_grip_depth, deep)       # 얇은 쪽 floor
        n = 4
        for i in range(n):
            depth = lo + (deep - lo) * i / (n - 1)   # 얇게(floor) → 깊게
            out = self._ik_reach(e, (gx, gy, top - depth), yaw, roll)
            if out is not None:
                return gx, gy, top - depth, yaw, roll, out[1], bx0, by0
        return None

    def _graspable(self, e: int, cube: str) -> bool:
        """도달 가능한 grasp 해(어떤 깊이로든)가 있는가 — 선택용 reach 판정."""
        return self._grasp_pose(e, cube) is not None

    def _place_xy(self, e: int) -> tuple[float, float]:
        """release 목표 xy — 그릇 중심을 base 쪽으로 살짝 당겨 reach 마진 확보 후,
        이미 담은 큐브 수(n_placed)별 ring 으로 분산 → 같은 자리 쌓임 방지.
        결정적(겹침 없음)이고 성공 반경 6cm 이내라 분산해도 in-bowl."""
        b = self.obj_pos(BOWL_NAME, e)
        root = self.snap["rp"][e]
        v = np.array([root[0] - b[0], root[1] - b[1]])
        d = max(float(np.linalg.norm(v)), 1e-6)
        cx = float(b[0]) + 0.008 * v[0] / d
        cy = float(b[1]) + 0.008 * v[1] / d
        ang = (math.pi / 2) * self.n_placed[e] + math.pi / 4  # 4분할 ring
        r = args.release_spread
        return (cx + r * math.cos(ang), cy + r * math.sin(ang))

    def _build_plan(self, e: int, cube: str) -> list[WP] | None:
        """큐브 1개 plan. 도달 불가 시 None(defer).

        grasp = side-approach: 비대칭 jaw 가 큐브 윗면을 찌르지 않게 닫힘축 방향으로
        side_offset 비켜 **수직 하강**(하강 중 큐브 무접촉) 후 **수평 slide 로 큐브 중심까지**
        진입해 close → 중심 파지. roll 은 ±90° 고정으로 접근부터 grasp 까지 불변(내려가서
        재정렬 안 함). grasp 깊이는 바닥 기준(grasp_floor)으로 깊게. release = 그릇 위
        안전고도에서 **하강 없이** n_placed 별 분산 지점에 0.3s 대기 후 떨굼.
        """
        p = self.obj_pos(cube, e)
        # grasp 해: 자세(roll)·진입(side/slide)·도달 가능 gz(얇게 우선 깊이 ladder) 일괄.
        gp = self._grasp_pose(e, cube)
        if gp is None:
            return None
        gx, gy, gz, yaw, roll, pitch, bx0, by0 = gp
        if self._ik(e, (bx0, by0, gz), pitch, yaw, roll) is None:
            return None  # 비킨 하강점 도달 불가
        sz = self._safe_z()
        # 그릇 위 통과·release 고도: 매달린 큐브 바닥이 그릇 rim 을 넘게 올림(arc over) →
        # 직선 운반이 rim 치는 것 방지(사용자). lift(sz)→transport(bz) 가 올라가며 그릇 위로
        # 호를 그린다. bz 도달 불가(그릇이 높이+먼 reach 한계)면 sz 로 폴백(큐브 포기보다 나음).
        bz = DESK_TOP_Z + args.bowl_clear_height
        bx, by = self._place_xy(e)
        rel = self._ik_reach(e, (bx, by, bz), yaw, roll)
        if rel is None:
            bz = sz
            rel = self._ik_reach(e, (bx, by, sz), yaw, roll)
            if rel is None:
                return None  # 그릇 도달 불가 (DR 상 거의 없음)
        pit_bowl = rel[1]

        self.grasp_ref[e] = p[:2].copy()  # 하강 drift 가드 기준
        oc = self._open_cmd(cube)
        cc = args.gripper_close
        ct, ft = args.joint_tol, args.fine_joint_tol
        hover_z = gz + args.pregrasp_height
        return [
            WP((bx0, by0, sz),      pitch,    yaw, roll, oc, args.transport_speed, ct, 0,                   "approach"),
            WP((bx0, by0, hover_z), pitch,    yaw, roll, oc, args.descend_speed,   ft, args.pregrasp_dwell, "hover"),
            WP((bx0, by0, gz),      pitch,    yaw, roll, oc, args.descend_speed,   ft, 0,                   "descend"),
            WP((gx, gy, gz),        pitch,    yaw, roll, oc, args.slide_speed,     ft, 0,                   "slide"),
            WP((gx, gy, gz),        pitch,    yaw, roll, cc, args.slide_speed,     ft, args.close_dwell,    "grasp"),
            WP((gx, gy, sz),        pitch,    yaw, roll, cc, args.lift_speed,      ct, 0,                   "lift"),
            WP((bx, by, bz),        pit_bowl, yaw, roll, cc, args.transport_speed, ct, args.release_wait,   "transport"),
            WP((bx, by, bz),        pit_bowl, yaw, roll, oc, args.lift_speed,      ct, args.release_dwell,  "release"),
        ]

    def _nudge_target(self, e: int, cube: str) -> tuple[float, float] | None:
        """재배치 push 방향(world 단위벡터) 또는 None. base 발치=base 반대, 밀집=이웃 반대,
        외측=base 쪽(best-effort). grasp 불가 큐브를 도달가능/트인 곳으로 옮기는 방향."""
        p = self.obj_pos(cube, e)
        tb = self._to_base(p, e)
        r = math.hypot(tb[0] - SO101Kinematics.PAN_X, tb[1])
        base = self.snap["rp"][e]
        rad = np.array([float(p[0]) - float(base[0]), float(p[1]) - float(base[1])])
        rn = float(np.linalg.norm(rad))
        rad = rad / max(rn, 1e-6)
        if r < args.nudge_r_near:                 # 발치 inner-reach → base 반대로 밀어냄
            return (float(rad[0]), float(rad[1]))
        others = [self.obj_pos(c, e) for c in self.remaining[e] if c != cube]
        if others:                                # 밀집 → 이웃 평균 반대로
            cen = np.mean([o[:2] for o in others], axis=0)
            away = np.array([float(p[0]) - cen[0], float(p[1]) - cen[1]])
            nd = float(np.linalg.norm(away))
            if nd < args.cube_clear:
                return (float(rad[0]), float(rad[1])) if nd < 1e-6 else \
                    (float(away[0] / nd), float(away[1] / nd))
        if r > args.nudge_r_far:                  # 외측 reach 한계 → base 쪽(best-effort pull)
            return (-float(rad[0]), -float(rad[1]))
        return None

    def _build_nudge_plan(self, e: int, cube: str, pdir: tuple[float, float]) -> list[WP] | None:
        """닫은 그리퍼로 큐브 옆을 짚어 pdir 로 nudge_dist 만큼 미는 paddle plan. 도달 불가 None.

        큐브 뒤쪽(push 반대)에 닫은 그리퍼를 내려 큐브 옆면 높이에서 pdir 로 sweep → 큐브를
        앞으로 민다(드래그 아닌 push 라 큐브가 튀지 않음). 끝나면 재선택(round 증가 없음)."""
        p = self.obj_pos(cube, e)
        size = CUBE_SIZES.get(cube, 0.030)
        px, py, cz = float(p[0]), float(p[1]), float(p[2])
        dx, dy = pdir
        off = size * 0.5 + 0.018                    # 큐브 뒤쪽 접촉 시작 오프셋
        bxp, byp = px - off * dx, py - off * dy      # paddle 시작(큐브 뒤)
        exp_, eyp = px + args.nudge_dist * dx, py + args.nudge_dist * dy  # 그리퍼 끝(큐브를 앞으로 push)
        sz = self._safe_z()
        o1 = self._ik_reach(e, (bxp, byp, cz), 0.0, 0.0)
        o2 = self._ik_reach(e, (exp_, eyp, cz), 0.0, 0.0)
        if o1 is None or o2 is None:
            return None                              # 짚는 자리·끝점 도달 불가 → nudge 포기
        pit = o1[1]
        cc = args.gripper_close
        ct, ft = args.joint_tol, args.fine_joint_tol
        return [
            WP((bxp, byp, sz), pit, 0.0, 0.0, cc, args.transport_speed, ct, 0, "nudge_appr"),
            WP((bxp, byp, cz), pit, 0.0, 0.0, cc, args.descend_speed,   ft, 0, "nudge_down"),
            WP((exp_, eyp, cz), pit, 0.0, 0.0, cc, args.slide_speed,    ft, 2, "nudge_push"),
            WP((exp_, eyp, sz), pit, 0.0, 0.0, cc, args.lift_speed,     ct, 0, "nudge_up"),
        ]

    def _select_next(self, e: int) -> str | None:
        """남은 큐브 중 다음 대상: clear(장애물 적음) > reachable > 그릇에서 먼 순.

        막힌 큐브는 자연히 뒤로, 앞 큐브가 치워지면 clear 해진다. round 상한 초과는 후순위.
        """
        rem = self.remaining[e]
        if not rem:
            return None
        if all(self.rounds[e][c] >= args.max_round for c in rem):
            log(f"[SM] env{e}: 남은 {rem} round 소진 → 종료 "
                f"(rounds={[self.rounds[e][c] for c in rem]})")
            return None
        bowl = self.obj_pos(BOWL_NAME, e)
        best, best_key = None, None
        for c in rem:
            p = self.obj_pos(c, e)
            reachable = self._graspable(e, c)
            d_bowl = float(np.linalg.norm(p[:2] - bowl[:2]))
            over = self.rounds[e][c] >= args.max_round
            # 우선순위: round 미소진 > 도달가능 > 높은 z(쌓인 stack top 먼저) > 그릇서 먼
            key = (not over, reachable, round(float(p[2]), 3), d_bowl)
            if best_key is None or key > best_key:
                best_key, best = key, c
        return best

    # ---- 안전망(taxonomy 기록) -----------------------------------------

    def _event(self, e: int, kind: str, cube: str, tag: str, detail: str = "") -> None:
        self.events[e].append({"kind": kind, "cube": cube, "wp": tag, "detail": detail})

    def _replan_cube(self, e: int, cube: str, reason: str) -> None:
        """현재 큐브 실패 → taxonomy 기록 후 종료(round 증가는 _end_cube 가 단일 처리)."""
        tag = self.plan[e][self.idx[e]].tag if self.plan[e] else "?"
        self._event(e, "retry", cube, tag, reason)
        log(f"[SM] env{e} {cube}: {reason} — replan "
            f"(round {self.rounds[e][cube] + 1}/{args.max_round})")
        self._end_cube(e, placed=False)

    def _end_cube(self, e: int, placed: bool) -> None:
        """현재 큐브 종료. placed=True 면 remaining 에서 제거, 아니면 round +1 하고
        round 소진 시에만 제거(그 전엔 유지 → 다른 큐브 처리 후 재시도)."""
        cube = self.cur_cube[e]
        if cube is not None:
            if placed:
                if cube in self.remaining[e]:
                    self.remaining[e].remove(cube)
                self.n_placed[e] += 1  # 다음 release 분산 인덱스
            else:
                self.rounds[e][cube] += 1
                if self.rounds[e][cube] >= args.max_round and cube in self.remaining[e]:
                    self.remaining[e].remove(cube)  # round 소진 — 포기
        self.cur_cube[e] = None
        self.plan[e] = []
        self.idx[e] = 0
        self.dwell[e] = 0
        self.wp_steps[e] = 0
        self.drop_logged[e] = False

    # ---- executor -------------------------------------------------------

    def _converged(self, e: int, tol: float) -> bool:
        q_now = self.joints(e)[:5]
        return max(abs(g - float(n)) for g, n in zip(self.q_cmd[e], q_now)) < tol

    def _init_pose(self, e: int) -> None:
        """현재 명령 pose 를 실측 TCP 로 초기화(보간 시작점)."""
        t = self.tcp_meas(e)
        # roll init = +90°(home wrist_roll 과 일치) → 첫 접근 wrist 스윙 제거
        self.cur_pose[e] = Pose(float(t[0]), float(t[1]), float(t[2]),
                                -math.pi / 2, 0.0, math.pi / 2)

    def _log_grasp(self, e: int, cube: str) -> None:
        """close 직후 기하 진단: 큐브가 TCP(jaw gap) 근처에 있는지·slide 가 큐브를
        밀쳐냈는지·그리퍼가 닫혔는지. grasp 실패 원인 분리용."""
        cp = self.obj_pos(cube, e)
        tcp = self.tcp_meas(e)
        gj = float(self.joints(e)[5])
        ref = self.grasp_ref[e]
        drift = float(np.linalg.norm(cp[:2] - ref)) if ref is not None else -1.0
        log(f"[GRASP] env{e} {cube}: d_cube_tcp={float(np.linalg.norm(cp - tcp)) * 1000:.0f}mm "
            f"d_xy={float(np.linalg.norm(cp[:2] - tcp[:2])) * 1000:.0f}mm "
            f"dz={(cp[2] - tcp[2]) * 1000:.0f}mm slide_drift={drift * 1000:.0f}mm "
            f"grip_j={gj:.3f} cube_z={cp[2]:.3f}")

    def _step_env(self, e: int) -> tuple[list[float], float]:
        """env e 의 (q_cmd[5], gripper_target) 반환 + waypoint 전이."""
        if self.outcome[e] == Outcome.DONE:
            return self.HOME_Q, self.HOME_GRIP

        # reset 후 큐브 정착 대기
        if self.settle_n[e] < args.settle_steps:
            self.settle_n[e] += 1
            self._init_pose(e)
            return list(self.HOME_Q), self.HOME_GRIP

        # 다음 큐브 선택 / plan 생성
        if self.cur_cube[e] is None:
            cube = self._select_next(e)
            if cube is None:
                self.outcome[e] = Outcome.DONE
                self._report(e)
                return list(self.HOME_Q), self.HOME_GRIP
            plan = self._build_plan(e, cube)
            if plan is None:
                # nudge: grasp 불가 큐브를 닫은 그리퍼로 밀어 도달가능/트인 곳으로 재배치(사용자).
                if args.nudge and self.nudge_n[e][cube] < args.max_nudge:
                    pdir = self._nudge_target(e, cube)
                    nplan = self._build_nudge_plan(e, cube, pdir) if pdir else None
                    if nplan is not None:
                        self.nudge_n[e][cube] += 1
                        self.cur_cube[e] = cube
                        self.is_nudge[e] = True
                        self.plan[e] = nplan
                        self.idx[e] = self.dwell[e] = self.wp_steps[e] = 0
                        self.seg_v[e] = 0.0
                        if self.cur_pose[e] is None:
                            self._init_pose(e)
                        self._event(e, "nudge", cube, "plan", f"({pdir[0]:+.2f},{pdir[1]:+.2f})")
                        log(f"[SM] env{e} {cube}: nudge {self.nudge_n[e][cube]}/{args.max_nudge} "
                            f"dir({pdir[0]:+.2f},{pdir[1]:+.2f})")
                        return self.q_cmd[e], args.gripper_close
                # plan 불가(IK 도달 불가 등). **영구 제외 대신 defer** — 한 번 실패로 graspable
                # 큐브를 버리지 않는다(사용자). round 소진 시에만 제거.
                self.rounds[e][cube] += 1
                self._event(e, "unreachable", cube, "plan", "build_plan None")
                if self.rounds[e][cube] >= args.max_round and cube in self.remaining[e]:
                    self.remaining[e].remove(cube)
                    log(f"[SM] env{e} {cube}: plan {args.max_round}회 불가 — 제외")
                else:
                    log(f"[SM] env{e} {cube}: plan 불가 — defer "
                        f"(round {self.rounds[e][cube]}/{args.max_round})")
                return self.q_cmd[e], args.gripper_open
            self.cur_cube[e] = cube
            self.plan[e] = plan
            self.idx[e] = 0
            self.dwell[e] = 0
            self.wp_steps[e] = 0
            if self.cur_pose[e] is None:
                self._init_pose(e)
            log(f"[SM] env{e}: cube={cube} roll={math.degrees(plan[0].roll):+.0f}° "
                f"(remaining {self.remaining[e]})")

        cube = self.cur_cube[e]
        wp = self.plan[e][self.idx[e]]
        self.wp_steps[e] += 1
        timeout = self.wp_steps[e] >= args.max_wp_steps
        tol = wp.tol

        # ----- 안전망 가드 -----
        holding = wp.grip <= 0.0  # 닫힘 명령 = 큐브 쥔 상태
        if wp.tag == "descend" and self.grasp_ref[e] is not None:
            # 비킨 하강이라 큐브를 건드리지 않아야 정상 — 원위치에서 밀리면 replan
            drift = float(np.linalg.norm(self.obj_pos(cube, e)[:2] - self.grasp_ref[e]))
            if drift > args.drift_tol:
                # [CLIP] 진단: self-clip(목표 큐브 자체) vs neighbor-clip(이웃) 구분.
                #   tcp-목표거리·최근접 이웃거리·z 로 어느 jaw 가 무엇을 쳤는지 분리.
                tcp = self.tcp_meas(e)
                cp = self.obj_pos(cube, e)
                nbrs = [(float(np.linalg.norm(tcp[:2] - self.obj_pos(c, e)[:2])), c)
                        for c in self.remaining[e] if c != cube]
                nd, nc = min(nbrs, default=(9.9, "none"))
                log(f"[CLIP] env{e} {cube}: drift={drift * 1000:.0f}mm "
                    f"tcp_xy=({tcp[0]:.3f},{tcp[1]:.3f}) cube_xy=({cp[0]:.3f},{cp[1]:.3f}) "
                    f"d_tcp_cube_xy={float(np.linalg.norm(tcp[:2] - cp[:2])) * 1000:.0f}mm "
                    f"nearest_other={nc}@{nd * 1000:.0f}mm tcp_z={tcp[2]:.3f} cube_z={cp[2]:.3f}")
                self._replan_cube(e, cube, f"하강 중 큐브 밀림 {drift * 1000:.0f}mm")
                return self.q_cmd[e], wp.grip
        if holding and wp.tag in ("lift", "transport"):
            dist = float(np.linalg.norm(self.obj_pos(cube, e) - self.tcp_meas(e)))
            if dist > args.drop_tol and not self.drop_logged[e]:
                # 쥔 큐브가 멀어짐 = drop 징후. **여기서 그리퍼를 열면 확실히 떨군다** —
                # 사용자 규칙 "집었으면 그릇에 놓을 때까지 안 연다". 열지·중단하지 않고 운반
                # 계속 → 최종 placement 실패 시에만 재시도(재접근에서 비로소 open). 1회만 기록.
                self._event(e, "drop", cube, wp.tag, f"{dist * 1000:.0f}mm")
                self.drop_logged[e] = True

        # ----- Cartesian 보간: 사다리꼴 속도 프로파일(가속→순항→감속), 위치·pitch 동시 -----
        # 등속 선형(텔레포트 느낌·급출발=큐브 밀침)이 아니라 양끝 0속도로 가·감속 → 자연스럽고
        # 부드러운 도착(arm 추종↑ = 정밀단계 대기↓). A9(cubic spline) 변형. roll/yaw 직접 세팅.
        cp = self.cur_pose[e]
        remaining = math.dist((cp.x, cp.y, cp.z), tuple(wp.pos))
        dt = 1.0 / 30.0
        v_decel = math.sqrt(max(2.0 * args.accel * remaining, 0.0))  # 목표서 0 되게 감속
        v_cmd = max(min(wp.speed, self.seg_v[e] + args.accel * dt, v_decel), args.min_speed)
        self.seg_v[e] = v_cmd
        step = v_cmd * dt
        frac = 1.0 if (remaining <= step or remaining < 1e-9) else step / remaining
        cp.x += (wp.pos[0] - cp.x) * frac
        cp.y += (wp.pos[1] - cp.y) * frac
        cp.z += (wp.pos[2] - cp.z) * frac
        cp.pitch += (wp.pitch - cp.pitch) * frac  # pitch 점진 재배향(anti-slam)
        cp.roll = wp.roll  # roll 직접 — 접근부터 grasp roll(±90) 고정
        cp.yaw = wp.yaw    # yaw 직접(fold 대칭, 스윙 없음)
        at_end = remaining <= step
        if at_end:
            self.seg_v[e] = 0.0  # segment 종료 → 다음 segment 0 부터 재가속

        q = self._ik(e, (cp.x, cp.y, cp.z), cp.pitch, cp.yaw, cp.roll)
        if q is not None:
            self.q_cmd[e] = q
        # IK 실패 시 직전 q_cmd 유지 (보간이 다음 step 더 가까운 점을 줌)

        # ----- 전이 -----
        reached = at_end and self._converged(e, tol)
        if reached and wp.tag == "slide":
            # slide 만: 관절 수렴(tol)만으론 TCP 가 수 mm 못 미친 채 advance → close 가 큐브를
            # jaw 가장자리서 놓침(Cube2형 miss). 실측 TCP 가 목표 reach_tol 안일 때만 도달.
            # (descend 는 offset 점이라 sub-cm 불필요 — 게이트 빼서 대기 단축)
            reached = float(np.linalg.norm(self.tcp_meas(e) - np.asarray(wp.pos, dtype=float))) \
                < args.reach_tol
        if reached:
            self.dwell[e] += 1
        if (reached and self.dwell[e] >= wp.settle) or timeout:
            # grasp 직후 lift 진입 전 큐브 z0 기록 (파지 검증 기준) + 기하 진단
            if wp.tag == "grasp":
                self.grasp_z0[e] = float(self.obj_pos(cube, e)[2])
                self._log_grasp(e, cube)
            self.dwell[e] = 0
            self.wp_steps[e] = 0
            self.seg_v[e] = 0.0  # WP 전환 → 다음 segment 0 부터 가속
            # lift 종료 = 파지 검증. 단 **쥐고 있으면(TCP 근처) 살짝만 올라와도 놓지 않는다**
            # (사용자: 집었으면 유지). 큐브가 안 오르고 TCP 에서도 멀면 = 빈 grasp → 재시도.
            if wp.tag == "lift" and not timeout:
                rose = float(self.obj_pos(cube, e)[2]) >= self.grasp_z0[e] + args.lift_check
                d_tcp = float(np.linalg.norm(self.obj_pos(cube, e) - self.tcp_meas(e)))
                if not rose and d_tcp > args.drop_tol:
                    self._replan_cube(e, cube, "파지 실패(빈 grasp)")
                    return self.q_cmd[e], wp.grip
            self.idx[e] += 1
            if self.idx[e] >= len(self.plan[e]):
                if self.is_nudge[e]:
                    # nudge 완료 = 재배치(실패 아님). round 증가 없이 재선택 → 옮긴 위치서 재시도.
                    self.is_nudge[e] = False
                    self.cur_cube[e] = None
                    self.plan[e] = []
                    self.idx[e] = self.dwell[e] = self.wp_steps[e] = 0
                else:
                    placed = self._placed(cube, e)
                    self._end_cube(e, placed=placed)
        return self.q_cmd[e], wp.grip

    # ---- 판정 -----------------------------------------------------------

    def _placed(self, cube: str, e: int) -> bool:
        p = self.obj_pos(cube, e)
        bowl_xy = self.obj_pos(BOWL_NAME, e)[:2]
        in_xy = float(np.linalg.norm(p[:2] - bowl_xy)) < BOWL_SUCCESS_RADIUS
        z_rel = float(p[2]) - DESK_TOP_Z
        in_z = BOWL_HEIGHT_RANGE[0] <= z_rel <= BOWL_HEIGHT_RANGE[1] + 0.10
        return in_xy and in_z

    # ---- 배치 액션 ------------------------------------------------------

    def _act_all(self, q_list: list[list[float]], grip_targets: list[float]) -> None:
        """전체 env 의 (q_cmd + 중력 처짐 적분 보상) 을 배치 action 으로 1회 step."""
        q_cmd = torch.tensor(q_list, device=self.device)                  # [N,5]
        q_now = self.robot.data.joint_pos[:, :5]                          # [N,5]
        self.q_bias = torch.clamp(
            self.q_bias + self.BIAS_KI * (q_cmd - q_now),
            -self.BIAS_MAX, self.BIAS_MAX)
        grip = torch.tensor(grip_targets, device=self.device) - GRIPPER_ACTION_OFFSET
        action = torch.cat([q_cmd + self.q_bias, grip.unsqueeze(-1)], dim=-1)  # [N,6]
        self.env.step(action)

    def _tick(self) -> None:
        """1 step: 스냅샷 → per-env executor → 배치 step."""
        self._snapshot()
        qs, grips = [], []
        for e in range(self.num_envs):
            q, g = self._step_env(e)
            qs.append(q)
            grips.append(g)
        self._act_all(qs, grips)

    # ---- 메인 루프 (headless 배치) --------------------------------------

    def run(self) -> int:
        """전 env 완료 또는 step cap 까지 실행. 반환 = 사용한 step 수."""
        steps = 0
        while not all(o == Outcome.DONE for o in self.outcome):
            if not simulation_app.is_running() or steps >= args.max_total_steps:
                break
            self._tick()
            steps += 1
        if steps >= args.max_total_steps:
            log(f"[SM] step cap {args.max_total_steps} 도달 — 미완료 env 존재")
        self._snapshot()
        self._summary()
        log(f"[SM] 전 env 완료까지 steps={steps} (~{steps / 30.0:.1f}s @30Hz, "
            f"병렬이라 가장 느린 env 기준)")
        return steps

    # ---- 결과 리포트 / taxonomy ----------------------------------------

    def _report(self, e: int) -> None:
        n_ok = sum(self._placed(c, e) for c in self.cubes)
        log(f"[SM] env{e} RESULT: {n_ok}/{len(self.cubes)} cubes in bowl "
            f"(events={len(self.events[e])})")

    def _summary(self) -> None:
        total_ok = sum(sum(self._placed(c, e) for c in self.cubes)
                       for e in range(self.num_envs))
        total = self.num_envs * len(self.cubes)
        clean = sum(1 for e in range(self.num_envs) if not self.events[e])
        retries = sum(len([ev for ev in self.events[e] if ev["kind"] == "retry"])
                      for e in range(self.num_envs))
        log(f"[SM] TOTAL: {total_ok}/{total} cubes in bowl "
            f"({100.0 * total_ok / max(total, 1):.1f}%) across {self.num_envs} envs | "
            f"first-attempt-clean {clean}/{self.num_envs} "
            f"({100.0 * clean / max(self.num_envs, 1):.1f}%) | retry events {retries}")

    def dump_taxonomy(self, path: str, spawn: dict | None = None) -> None:
        """per-env outcome taxonomy JSON 저장 (실패 분석·spawn 재현용)."""
        import re
        from collections import Counter
        envs = []
        reason_hist: Counter = Counter()
        for e in range(self.num_envs):
            placed = {c: bool(self._placed(c, e)) for c in self.cubes}
            for ev in self.events[e]:
                # 숫자(mm) 제거해 카테고리로 집계
                cat = re.sub(r"\d+", "", ev.get("detail", "")).strip() or ev["kind"]
                reason_hist[cat] += 1
            rec = {
                "env": e,
                "placed": sum(placed.values()),
                "per_cube": placed,
                "clean": not self.events[e],
                "events": self.events[e],
                "cubes": {c: self.obj_pos(c, e).tolist() for c in self.cubes},
                "bowl": self.obj_pos(BOWL_NAME, e).tolist(),
            }
            if spawn is not None:
                rec["spawn"] = {k: v[e][:7] for k, v in spawn.items()}  # pos+quat (재현용)
            envs.append(rec)
        total = self.num_envs * len(self.cubes)
        total_ok = sum(r["placed"] for r in envs)
        out = {
            "num_envs": self.num_envs,
            "active_objects": len(self.cubes),
            "seed": args.seed,
            "success_pct": 100.0 * total_ok / max(total, 1),
            "clean_pct": 100.0 * sum(r["clean"] for r in envs) / max(self.num_envs, 1),
            "reason_histogram": dict(reason_hist),
            "envs": envs,
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        log(f"[SM] taxonomy → {path}")


# ---------------------------------------------------------------------------
# spawn 캡처/복원 (R 키·replay 용)
# ---------------------------------------------------------------------------


def _capture_spawn(sm: SO101PickPlace) -> dict:
    """현재 객체(cube×4 + bowl) pose 를 **env-local**(pos-env_origin, quat) 로 캡처.

    env-local 이라 R(동일 env)·replay(다른 env 수)에서 각 env_origin 더해 복원하면 어디서든
    같은 배치를 재현한다.
    """
    origins = sm.scene.env_origins.detach().cpu().numpy()  # [N,3]
    out = {}
    for name in sm.cubes + [BOWL_NAME]:
        d = sm.scene[name].data
        pos = d.root_pos_w[:, :3].detach().cpu().numpy() - origins  # env-local
        quat = d.root_quat_w.detach().cpu().numpy()
        out[name] = np.concatenate([pos, quat], axis=1).tolist()  # [N,7]
    return out


def _restore_spawn(sm: SO101PickPlace, spawn: dict) -> None:
    """env-local pose 를 각 env_origin 더해 world 로 write (DR 우회). repo 표준 API 사용."""
    origins = sm.scene.env_origins  # [N,3] torch
    zero_vel = torch.zeros((sm.num_envs, 6), device=sm.device)
    for name, state in spawn.items():
        obj = sm.scene[name]
        t = torch.tensor(state, device=sm.device, dtype=torch.float32)  # [M,7]
        if t.shape[0] != sm.num_envs:  # replay: 1개 → 전 env 복제
            t = t[:1].repeat(sm.num_envs, 1)
        pos = t[:, :3] + origins
        pose = torch.cat([pos, t[:, 3:7]], dim=-1)  # [N,7] world
        obj.write_root_pose_to_sim(pose)
        obj.write_root_velocity_to_sim(zero_vel)
    sm.scene.write_data_to_sim()


# ---------------------------------------------------------------------------
# GUI 키보드 (carb input — teleop_se3_agent.py GuiKeyboard 패턴)
# ---------------------------------------------------------------------------


class GuiKeyboard:
    """carb input 구독으로 키 콜백 등록. R/N 같은 단발 키용."""

    def __init__(self) -> None:
        import carb
        import omni.appwindow
        self._callbacks: dict[str, callable] = {}
        self._input = carb.input.acquire_input_interface()
        self._kbd = omni.appwindow.get_default_app_window().get_keyboard()
        self._carb = carb
        self._sub = self._input.subscribe_to_keyboard_events(self._kbd, self._on_event)

    def add_callback(self, key: str, func) -> None:
        self._callbacks[key.upper()] = func

    def _on_event(self, event, *_args) -> None:
        if event.type == self._carb.input.KeyboardEventType.KEY_PRESS:
            cb = self._callbacks.get(event.input.name.upper())
            if cb is not None:
                cb()

    def close(self) -> None:
        if getattr(self, "_sub", None) is not None:
            self._input.unsubscribe_to_keyboard_events(self._kbd, self._sub)
            self._sub = None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def run_spawn_check(sm) -> None:
    """DR 적용 후 초기 spawn 의 도달가능성·뭉침을 env 전체로 점검 후 종료 (컨트롤러 미실행).

    사용자 검증용 — ① 닿을 수 없는 곳에 스폰되는가(_graspable=False 비율) ② 뭉쳐서만
    스폰되는가(큐브 간 거리·펼침). 컨트롤러를 돌리지 않으므로 displacement 없는 **순수 초기
    분포**를 본다(런타임 build_plan None 과 분리).
    """
    import statistics as _st

    for _ in range(args.settle_steps):
        if not simulation_app.is_running():
            return
        sm._tick()  # home 자세 유지하며 큐브 정착(z 분산 spawn 낙하 포함)
    sm._snapshot()
    spawn = _capture_spawn(sm)  # env-local [N,7] — 도달불가 env replay 용

    n_un = 0
    total = 0
    per_cube = {c: 0 for c in sm.cubes}
    minseps, maxpairs = [], []
    bad = []  # 도달불가 큐브 가진 env (single-env replay·영상용)
    for e in range(sm.num_envs):
        xy = [sm.obj_pos(c, e)[:2] for c in sm.cubes]
        pd = [float(np.linalg.norm(xy[i] - xy[j]))
              for i in range(len(xy)) for j in range(i + 1, len(xy))]
        if pd:
            minseps.append(min(pd))
            maxpairs.append(max(pd))
        un = []
        for c in sm.cubes:
            total += 1
            if not sm._graspable(e, c):
                n_un += 1
                per_cube[c] += 1
                un.append(c)
        if un:
            bad.append({
                "env": e, "placed": 0, "clean": False,
                "events": [{"kind": "unreachable", "cube": c, "wp": "spawn",
                            "detail": "graspable=False"} for c in un],
                "per_cube": {c: False for c in sm.cubes},
                "cubes": {c: sm.obj_pos(c, e).tolist() for c in sm.cubes},
                "bowl": sm.obj_pos(BOWL_NAME, e).tolist(),
                "spawn": {name: list(spawn[name][e][:7]) for name in spawn},
            })

    bad_path = "outputs/unreachable_spawns.json"
    with open(bad_path, "w") as f:
        json.dump({"num_envs": len(bad), "active_objects": len(sm.cubes),
                   "seed": args.seed, "envs": bad}, f, indent=2)

    log("=" * 72)
    log(f"[SPAWNCHK] envs={sm.num_envs} cubes/env={len(sm.cubes)} cube_sep={args.cube_sep} "
        f"scatter_far={args.scatter_far} scatter_z={args.scatter_z}")
    log(f"[SPAWNCHK] 초기 spawn 도달불가(_graspable=False): {n_un}/{total} "
        f"= {100.0 * n_un / max(total, 1):.1f}%   per_cube={per_cube}")
    if minseps:
        tight = 100.0 * sum(1 for v in maxpairs if v < 0.08) / len(maxpairs)
        log(f"[SPAWNCHK] 뭉침: min_pair_sep mean={_st.mean(minseps):.3f}(floor≈cube_sep) "
            f"max_pair mean={_st.mean(maxpairs):.3f}(클수록 펼쳐짐) 전부8cm내={tight:.1f}%")
    log(f"[SPAWNCHK] 도달불가 env {len(bad)}개 spawn → {bad_path} "
        f"(--replay_spawn {bad_path} --replay_env_idx 0,1,.. 로 single-env 재현)")
    log("=" * 72)


def _make_env(env_cfg):
    if args.video:
        env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
        os.makedirs("docs", exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env, video_folder="docs", name_prefix=args.video_name,
            step_trigger=lambda step: step == 0,
            video_length=args.video_length, disable_logger=True)
        return env, env.unwrapped
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    return env, env


def main() -> None:
    log("[SM] main entered.")
    env_cfg                = PickCubeEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed           = args.seed
    _apply_dr(env_cfg)
    # 컨트롤러가 종료를 관리 — RL termination 에 의한 reset 차단
    env_cfg.episode_length_s       = 1.0e6
    env_cfg.terminations.success   = None
    env_cfg.terminations.cube_lost = None
    env_cfg.viewer.eye    = args.view_eye
    env_cfg.viewer.lookat = args.view_lookat
    # 그리퍼 닫힘/열림 slew 가속 (인스턴스 override — 공유 RL cfg 불변)
    mv = dict(env_cfg.actions.arm.max_velocity)
    mv["gripper"] = args.gripper_speed
    env_cfg.actions.arm.max_velocity = mv
    # 2048-env physx capacity 여유 (인스턴스에만 override — 공유 cfg 불변)
    if args.num_envs >= 256:
        px = env_cfg.sim.physx
        px.gpu_total_aggregate_pairs_capacity = max(
            px.gpu_total_aggregate_pairs_capacity, 1024 * 1024 * 4)
        px.gpu_found_lost_aggregate_pairs_capacity = max(
            px.gpu_found_lost_aggregate_pairs_capacity, 1024 * 1024 * 16)
        px.gpu_max_rigid_patch_count = max(px.gpu_max_rigid_patch_count, 32 * 2 ** 16)
    log("[SM] env_cfg built — calling gym.make.")

    env, base = _make_env(env_cfg)
    log("[SM] env created.")
    env.reset()
    log("[SM] reset done — DR applied.")

    if args.calibrate:
        run_calibration(base)
        env.close()
        return

    sm = SO101PickPlace(env, CUBE_NAMES[: args.active_objects])

    if args.check_spawns:
        run_spawn_check(sm)
        env.close()
        return

    # replay_spawn: taxonomy 의 첫 실패 env spawn pose 로 덮어쓰고 정착 → GUI 1:1 재현
    if args.replay_spawn:
        with open(args.replay_spawn) as f:
            data = json.load(f)
        envs = data["envs"]
        n_act = data.get("active_objects", args.active_objects)
        if args.replay_env_idx >= 0:
            pick = envs[args.replay_env_idx % len(envs)]
        else:
            pick = next((r for r in envs
                         if not r.get("clean", True) or r["placed"] < n_act), envs[0])
        log(f"[SM] replay env{pick['env']} (placed={pick['placed']}/{n_act}, "
            f"clean={pick.get('clean')})")
        src = pick.get("spawn") or {**pick["cubes"], BOWL_NAME: pick["bowl"]}

        def _to_state(st):
            if len(st) >= 7:  # pos+quat
                return list(st[:7]) + [0.0] * 6
            return [st[0], st[1], st[2], 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        _restore_spawn(sm, {name: [_to_state(st)] for name, st in src.items()})
        for _ in range(args.settle_steps):
            sm._tick()
        sm.reset_all()

    if args.headless or args.video:
        # ----- 배치 모드: 완주 → taxonomy → 종료 -----
        spawn = _capture_spawn(sm)  # 초기(에피소드 시작) 객체 pose — 실패 env 재현용
        sm.run()
        tax = args.taxonomy or f"outputs/sm_scale_{args.num_envs}_seed{args.seed}.json"
        sm.dump_taxonomy(tax, spawn=spawn)
        env.close()
        return

    # ----- GUI 관전 모드: R=동일 셋업 / N=새 시드 -----
    state = {"reset": None}  # 'same' | 'new' | None

    kbd = None
    try:
        kbd = GuiKeyboard()
        kbd.add_callback("R", lambda: state.update(reset="same"))
        kbd.add_callback("N", lambda: state.update(reset="new"))
        log("[SM] GUI 관전 모드 — R: 동일 셋업 재시작 / N: 새 시드 재시작 / 창 닫기: 종료")
    except Exception as exc:  # noqa: BLE001
        log(f"[SM] 키보드 구독 실패(무시): {exc}")

    spawn = _capture_spawn(sm)
    while simulation_app.is_running():
        if state["reset"] is not None:
            mode = state["reset"]
            state["reset"] = None
            env.reset()  # DR 재샘플 (RNG advance)
            if mode == "same":
                _restore_spawn(sm, spawn)
                log("[SM] R — 동일 셋업 재시작")
            else:
                spawn = _capture_spawn(sm)
                log("[SM] N — 새 시드 재시작")
            sm.reset_all()
            for _ in range(args.settle_steps):
                sm._tick()
        if all(o == Outcome.DONE for o in sm.outcome):
            # 완료 → home 유지하며 키 대기
            sm._snapshot()
            sm._act_all([list(sm.HOME_Q)] * sm.num_envs,
                        [sm.HOME_GRIP] * sm.num_envs)
        else:
            sm._tick()

    if kbd is not None:
        kbd.close()
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("[SM] EXCEPTION:\n" + traceback.format_exc())
        raise
    finally:
        simulation_app.close()
