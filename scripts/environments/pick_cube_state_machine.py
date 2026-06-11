"""SO-101 pick-and-place rule-based state machine (cube_desk 씬) — 해석적 IK 판.

v1(해석적 IK + 단순 FSM 초안) → v2(Cartesian 직선 + 재시도 FSM 초안) 를 거쳐,
본 v3 는 둘을 실제 Isaac Lab 환경(`SimToReal-SO101-PickCube-v0`)에 연결한 메인 코드다.
Lula/DiffIK 등 수치 솔버 없이 **닫힌 해(closed-form) IK + joint position action** 만 사용한다.

설계
----
- 기구학: URDF(so_arm101.urdf) origin 체인을 pan 회전 평면의 2-link(+wrist 체인)로
  환원한 해석적 FK/IK. 관절 역할 분해:
    q1 (shoulder_pan)  : 방위각 (pan 축은 base -z → 부호 반전)
    q2,q3 (lift/elbow) : pitch 평면 2-link IK
    q4 (wrist_flex)    : 툴 피치 보정 (top-down = -90°, 도달 불가 시 점진 완화)
    q5 (wrist_roll)    : 손가락 닫힘축을 큐브 yaw 에 정렬 (90° 대칭 접기)
- action: PickCubeEnvCfg 의 6-dim SlewLimitedJointPositionAction.
  desired = raw*1.0 + default_offset (arm 0, gripper 0.20) 이므로
  arm 은 절대 관절각 그대로, gripper 는 target-0.20 을 raw 로 보낸다.
  속도 가감속은 slew limit(arm 5.0 / gripper 2.5 rad/s)이 보장한다.
- FSM: per-env 상태 배열로 num_envs 병렬 관리, 매 step env.step([N,6]) 1회.
  SAFE_Z 횡이동 / latch+drift 감지 / lift 후 파지 검증 / 재시도(MAX 3) /
  max_phase_steps timeout 안전장치.
- grasp = side-approach: 큐브 중심이 아니라 닫힘축의 base 쪽으로 side_offset 만큼
  비킨 지점에 수직 하강(찌르기 원천 차단) 후, 수평 SLIDE 로 큐브를 손가락 사이에
  넣고 닫는다. 비킴 방향이 base 쪽인 이유: fixed finger 가 닫힘축 base-반대쪽에
  있어 반대로 비키면 slide 중 fixed 가 선두로 큐브 면을 밀고 다닌다 (실측).
  중력 처짐(PD stiffness 17.8 의 정적 오차 ~0.14rad)은 적분 보상(q_bias)으로 제거.

검증 결과 (2026-06-11): 고정 spawn 4/4, DR full 2 env 6/8 — 실패는 reach 경계 spawn.

실행:
    OMNI_KIT_ACCEPT_EULA=YES uv run --group isaac python \\
        scripts/environments/pick_cube_state_machine.py \\
        --num_envs 1 --active_objects 4 --headless

기구학 캘리브레이션(1회 진단 — FK 예측 vs 시뮬 실측):
    OMNI_KIT_ACCEPT_EULA=YES uv run --group isaac python \\
        scripts/environments/pick_cube_state_machine.py --calibrate --headless
"""

from __future__ import annotations

import argparse
import faulthandler
import math
import os
import sys
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
    파일 append + fsync(크래시 시 buffer 유실 방지) + stderr 출력."""
    with open(_LOG_PATH, "a") as f:
        f.write(msg + "\n")
        f.flush()
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

parser = argparse.ArgumentParser(description="SO-101 pick-and-place state machine (cube_desk, 해석적 IK)")
parser.add_argument("--task", default="SimToReal-SO101-PickCube-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--active_objects", type=int, default=4, choices=[1, 2, 3, 4])
parser.add_argument("--object_radius_scale", type=float, default=1.0,
                    help="큐브 scatter DR 강도 (0=고정 spawn, 1=전체 workspace)")
parser.add_argument("--container_angle_scale", type=float, default=1.0,
                    help="그릇 arc DR 강도 (0=고정, 1=기본 각도범위)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--calibrate", action="store_true",
                    help="기구학 진단 모드: FK 예측 vs 시뮬 실측 비교 후 종료 (FSM 미실행)")
# FSM 전이·정착 파라미터
parser.add_argument("--joint_tol", type=float, default=0.09,
                    help="관절 수렴 판정 max|q_goal-q_now| (rad). 거친 이동 단계용")
parser.add_argument("--fine_joint_tol", type=float, default=0.025,
                    help="정밀 단계(DESCEND/LOWER) 관절 수렴 판정 (rad)")
parser.add_argument("--max_phase_steps", type=int, default=240,
                    help="한 단계에서 수렴 못해도 넘어가는 step 상한 (30 Hz 기준 8초)")
parser.add_argument("--grasp_dwell", type=int, default=10, help="그리퍼 닫힘 정착 step (30 Hz)")
parser.add_argument("--release_dwell", type=int, default=12,
                    help="그리퍼 열림 후 정지 step (0.4s — 움직이면 jaw 가 큐브를 퍼올림)")
parser.add_argument("--settle_steps", type=int, default=8, help="reset 후 큐브 정착 대기 step")
parser.add_argument("--max_retry", type=int, default=3, help="큐브당 grasp 재시도 횟수")
# 높이/오프셋 (m)
parser.add_argument("--safe_height", type=float, default=0.12,
                    help="책상 윗면 기준 횡이동 안전 고도 (그릇 테두리·큐브 위). "
                         "높일수록 top-down reach 반경이 급감하므로 최소한으로")
parser.add_argument("--grasp_z_offset", type=float, default=0.005,
                    help="grasp 시 큐브 중심 기준 TCP z 오프셋. +5mm가 검증값 — "
                         "더 깊게 내리면 reach 가장자리에서 실행 미달로 회귀")
parser.add_argument("--place_height", type=float, default=0.085,
                    help="그릇 중심 기준 release 시 TCP 높이")
parser.add_argument("--drift_tol", type=float, default=0.015,
                    help="latch 후 하강 중 큐브 xy 이탈 허용 (m). 초과 시 재접근")
parser.add_argument("--lift_check", type=float, default=0.03,
                    help="파지 검증: lift 후 큐브 최소 상승량 (m)")
# 그리퍼 명령 (joint target, rad)
parser.add_argument("--gripper_open", type=float, default=0.65,
                    help="30mm 큐브용 열림 joint target (rad). 더 좁히면 큐브가 "
                         "손가락 사이로 못 들어오고, 더 벌리면 회전형 jaw 끝이 "
                         "TCP 아래로 처져 하강이 일찍 막힘")
parser.add_argument("--gripper_open_large", type=float, default=0.85,
                    help="40mm 큐브용 열림 joint target (rad)")
parser.add_argument("--grasp_z_offset_large", type=float, default=0.005,
                    help="40mm 큐브용 grasp z 오프셋 (m)")
parser.add_argument("--side_offset", type=float, default=0.035,
                    help="side-approach 횡오프셋 (m): 큐브 중심에서 닫힘축 방향으로 "
                         "이만큼 비켜 수직 하강(찌르기 원천 차단) 후 수평 slide 로 "
                         "큐브를 손가락 사이에 넣고 닫는다")
parser.add_argument("--slide_speed", type=float, default=0.10,
                    help="SLIDE 수평 진입 속도 (m/s)")
parser.add_argument("--cube_clear", type=float, default=0.05,
                    help="비킴 지점·슬라이드 경로와 다른 큐브 사이 최소 거리 (m)")
parser.add_argument("--bowl_clear", type=float, default=0.12,
                    help="비킨 하강 지점이 그릇 중심에서 이만큼 못 떨어지면 wrist "
                         "roll 90° 대안 grasp 으로 비킴 방향을 돌린다 (그릇 끼임 방지)")
parser.add_argument("--slide_stop", type=float, default=0.010,
                    help="SLIDE 종점의 큐브 중심 잔여 거리 (m). 나머지는 close 시 "
                         "jaw 스윕이 마무리")
parser.add_argument("--pregrasp_height", type=float, default=0.04,
                    help="최종 하강 전 큐브 위 hover 높이 (m). 여기서 dwell 하며 "
                         "중력 처짐 보상을 수렴시켜 손가락 수직을 확보")
parser.add_argument("--pregrasp_dwell", type=int, default=5,
                    help="pre-grasp hover 정착 step (bias 적분 수렴용)")
parser.add_argument("--descend_speed", type=float, default=0.15,
                    help="DESCEND/LOWER Cartesian 수직 하강 속도 (m/s). joint 공간 "
                         "보간은 TCP 가 호를 그려 큐브를 찌르므로 z 를 점진 하강시켜 "
                         "수직 직선 경로를 강제")
parser.add_argument("--lift_speed", type=float, default=0.40,
                    help="LIFT 수직 상승 속도 (m/s)")
parser.add_argument("--transport_speed", type=float, default=0.50,
                    help="TRANSPORT 수평 운반 속도 (m/s). Cartesian 직선 ramp")
parser.add_argument("--lower_speed", type=float, default=0.25,
                    help="LOWER 하강 속도 (m/s)")
parser.add_argument("--gripper_close", type=float, default=-0.05)
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
# Part A — 해석적 기구학
# ---------------------------------------------------------------------------


class SO101Kinematics:
    """SO-101 5축 닫힌 해 FK/IK (robot base_link frame 기준).

    URDF(assets/robots/urdf/so_arm101.urdf) origin 체인을 base frame 에서 전개한 결과:
      · shoulder_pan 축: base (PAN_X, 0) 위치의 -z 축 → +q1 명령 = world yaw -q1
      · lift/elbow/wrist_flex 축: 모두 pan 회전 평면의 같은 pitch 축(+y 방향)
        → +q 회전이 평면각(atan2(z, r))을 감소시키는 동일 부호 체계
      · zero pose TCP = base (0.391, 0.000, 0.227) — pan 평면 위에 정확히 위치
        (lift 의 lateral -0.0183 이 wrist_roll origin 의 +0.0181 로 상쇄)
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

    def fk_tcp(self, q: list[float] | torch.Tensor) -> tuple[float, float, float]:
        """관절각 → TCP(gripper_frame) 위치, base frame. 검증·INIT 동기화용."""
        q1, q2, q3, q4, q5 = (float(v) for v in q[:5])
        th1 = self.TH1_0 - q2
        th2 = th1 + (self.TH2_0 - self.TH1_0) - q3
        th3 = th2 + (self.TH3_0 - self.TH2_0) - q4
        pr = (self.LIFT_R + self.L1 * math.cos(th1) + self.L2 * math.cos(th2)
              + self.L3 * math.cos(th3))
        pz = (self.LIFT_Z + self.L1 * math.sin(th1) + self.L2 * math.sin(th2)
              + self.L3 * math.sin(th3))
        # q5 회전에 의한 TCP 의 roll 축 주위 변위 (q5=0 기준 이미 L3/TH3_0 에 포함된
        # -ROLL_RHO 평면 성분을 빼고 회전 성분으로 대체)
        lat = self.ROLL_RHO * math.sin(q5)
        # 평면(접근축 수직) 성분 보정: -ρ → -ρ·cos(q5). th3 수직(-90°) 가정으로 z 가 아닌
        # 평면 radial 방향 — 오차 ≤2mm 라 fk 검증용으로는 무시.
        x = self.PAN_X + pr * math.cos(q1) - lat * math.sin(q1)
        y = -(pr * math.sin(q1) + lat * math.cos(q1))
        # 주의: pan 부호 반전(-q1) 전개 — base yaw = -q1 이므로
        #   radial 단위벡터 = (cos q1, -sin q1), lateral(+left) = (sin q1, cos q1)? 부호는
        #   calibrate 모드 실측으로 검증한다.
        z = pz
        return (x, y, z)

    def ik(self, tcp: tuple[float, float, float], grasp_yaw: float,
           pitch: float = -math.pi / 2,
           q_ref: list[float] | None = None,
           roll_offset: float = 0.0) -> list[float] | None:
        """TCP 목표(base frame) + 손가락 닫힘축 yaw → 관절각 5개. 도달 불가 시 None.

        pitch: 툴 접근 피치(wrist→TCP 평면각). -π/2 = 수직 top-down.
        roll_offset: q5 에 더하는 고정 회전 (rad). 큐브 90° 대칭을 이용해 닫힘축을
        ±90° 돌린 대안 grasp 자세 — 그릇 등 장애물을 피해 비킴 방향을 바꿀 때 사용.
        """
        x, y, z = tcp
        dx, dy = x - self.PAN_X, y
        r = math.hypot(dx, dy)
        if r < 1e-6:
            return None
        q1 = -math.atan2(dy, dx)  # pan 축 = base -z → 부호 반전

        # q5: 손가락 닫힘축을 큐브 yaw 에 정렬 (90° 대칭 → ±45° 접기).
        # world yaw → pan 프레임: 닫힘축 world 방위 = -q1 + (q5 의 기여). 부호·영점은
        # 대칭 접기 덕에 ±45° 안에서 자기수정되므로 단순형으로 둔다.
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
        rel = math.acos(c_rel)  # 상완→전완 평면각 차의 크기

        base_ang = math.atan2(wz, wr)
        # 두 가지 elbow 해 (전완이 상완 대비 위/아래) 중 limit 만족 해를 모으고,
        # q_ref(현재 자세)가 있으면 가까운 해를 채택 — 가지 비일관 swing 방지.
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
                 roll_offset: float = 0.0,
                 ) -> tuple[list[float], float] | None:
        """top-down 우선, 도달 불가 시 pitch 를 점진 완화하며 첫 해와 채택 pitch 반환.

        5-DOF position 우선·orientation best-effort 규약 (AGENTS.md).
        pitch_max -30°: SO-101 reach 가장자리(그릇 등 r>0.33)는 비스듬해야만 닿는다.
        """
        n_steps = 13
        for i in range(n_steps):
            pitch = pitch_min + (pitch_max - pitch_min) * i / (n_steps - 1)
            q = self.ik(tcp, grasp_yaw, pitch=pitch, q_ref=q_ref,
                        roll_offset=roll_offset)
            if q is not None:
                return q, pitch
        return None

    @staticmethod
    def _fold_45(a: float) -> float:
        return (a + math.pi / 4) % (math.pi / 2) - math.pi / 4


# ---------------------------------------------------------------------------
# Domain Randomization — events 만 강도 조정 (franka SM 과 동일 패턴)
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

    bowl = getattr(env_cfg.events, "randomize_bowl", None)
    a = float(args.container_angle_scale)
    if bowl is not None and a != 1.0:
        lo, hi = bowl.params["angle_range_deg"]
        bowl.params["angle_range_deg"] = (lo * a, hi * a)


# ---------------------------------------------------------------------------
# Part B — 캘리브레이션 모드 (--calibrate)
# ---------------------------------------------------------------------------


def _quat_to_yaw(quat: torch.Tensor) -> float:
    """wxyz quat → world yaw (rad)."""
    w, x, y, z = (float(v) for v in quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# USD root body frame ↔ URDF base_link frame 정합 (캘리브레이션 8자세 실측 fit, 잔차<0.3mm):
#   URDF = rot(+90°) · (USD_root_local - BASE_XY_OFFSET),  z - BASE_Z_OFFSET
# so101_follower.usd 의 root body 가 URDF base_link 대비 yaw -90° 회전 + 원점 시프트 상태.
BASE_XY_OFFSET = (0.0204, 0.0157)
BASE_Z_OFFSET = 0.0325
BASE_YAW_OFFSET = math.pi / 2  # USD root frame yaw → URDF base frame yaw 가산값


def _world_to_base(p_w: torch.Tensor, robot, e: int) -> tuple[float, float, float]:
    """world 좌표 → URDF base_link frame.

    1) root quat(yaw) 역회전으로 USD root body local 로,
    2) 실측 fit 상수로 URDF base_link frame 으로 변환.
    """
    root_p = robot.data.root_pos_w[e, :3]
    yaw = _quat_to_yaw(robot.data.root_quat_w[e])
    d = (p_w - root_p).detach().cpu().tolist()
    c, s = math.cos(-yaw), math.sin(-yaw)
    bx = c * d[0] - s * d[1]
    by = s * d[0] + c * d[1]
    # rot(+90°): (x, y) → (-y, x)
    return (-(by - BASE_XY_OFFSET[1]), bx - BASE_XY_OFFSET[0], d[2] - BASE_Z_OFFSET)


def run_calibration(env) -> None:
    """zero pose + 단일 joint 스텝 응답으로 FK 모델(상수·부호) 검증. 1회 진단용.

    출력: 예측 TCP(base frame) vs 실측(gripper body + URDF grasp offset) 오차.
    오차 > 5mm 면 SO101Kinematics 상수를 재보정할 것.
    """
    from isaaclab.utils.math import quat_apply

    kin = SO101Kinematics()
    robot = env.scene["robot"]
    names = list(robot.data.body_names)
    g_idx = names.index("gripper")
    grasp_off = torch.tensor([-0.0079, -0.000218121, -0.0981274], device=env.device)

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
    log(f"[CALIB] zero 평면각: th1_0={math.degrees(kin.TH1_0):.2f}° "
        f"th2_0={math.degrees(kin.TH2_0):.2f}° th3_0={math.degrees(kin.TH3_0):.2f}°")

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

    # --- gripper sweep: open 각별 TCP 하강 한계 실측 (40mm grasp 딜레마 진단) --
    # 빈 책상 좌표로 TCP 를 책상 표면(z≈0)까지 top-down 명령하고 실제로 어디서
    # 멈추는지 측정 — 회전형 jaw 끝이 책상에 닿는 높이가 open 각별 하강 한계.
    log("[CALIB] gripper sweep — open 각별 TCP 하강 한계 (책상 표면 목표)")
    desk_z_base = DESK_TOP_Z + 0.004 - 0.6749 - BASE_Z_OFFSET  # mat 윗면 ≈ base z
    probe = (0.26, 0.06, desk_z_base + 0.002)  # 큐브 없는 빈 좌표 (base frame)
    for open_cmd in (0.2, 0.5, 0.65, 0.85, 1.1):
        q = kin.ik(probe, 0.0)
        if q is None:
            log(f"[CALIB] open={open_cmd:.2f}  probe IK 실패 — 좌표 조정 필요")
            continue
        act = torch.zeros((env.num_envs, 6), device=env.device)
        act[0, :5] = torch.tensor(q, device=env.device)
        act[0, 5] = open_cmd - GRIPPER_ACTION_OFFSET
        # 적분 보상 없이 순수 명령 — 한계는 충돌이 결정하므로 90 step 정착
        for _ in range(90):
            if not simulation_app.is_running():
                return
            env.step(act)
        tcp = measured_tcp_base()
        clearance = tcp[2] - desk_z_base
        log(f"[CALIB] open={open_cmd:.2f}  TCP_z={tcp[2]:+.4f}  "
            f"책상 위 잔여 clearance={clearance * 1000:.1f}mm")

    # --- tool 접근축 기울기: pitch 명령별 접근축 실제 평면각 ------------------
    # ik(pitch) 는 wrist→TCP 벡터각 기준 — 손가락 접근축과는 고정 오프셋(δ_tool)
    # 차이가 있다. top-down(-90°) 명령 자세에서 접근축 후보(gripper body ±x/±z,
    # TCP→손끝)의 실제 기울기를 측정해 δ_tool 을 확정한다.
    log("[CALIB] tool 접근축 — pitch 명령별 gripper 축 world 방향")
    g_idx3 = names.index("gripper")
    for pitch_cmd in (-90.0, -60.0):
        q = kin.ik((probe[0], probe[1], probe[2] + 0.04), 0.0,
                   pitch=math.radians(pitch_cmd))
        if q is None:
            log(f"[CALIB] pitch={pitch_cmd:.0f}° IK 불가 — skip")
            continue
        act = torch.zeros((env.num_envs, 6), device=env.device)
        act[0, :5] = torch.tensor(q, device=env.device)
        act[0, 5] = 0.5 - GRIPPER_ACTION_OFFSET
        for _ in range(60):
            if not simulation_app.is_running():
                return
            env.step(act)
        gq = robot.data.body_quat_w[0, g_idx3]
        line = f"[CALIB] pitch_cmd={pitch_cmd:.0f}°"
        for label, vec in (("x_g", [1.0, 0.0, 0.0]), ("y_g", [0.0, 1.0, 0.0]),
                           ("z_g", [0.0, 0.0, 1.0])):
            v = quat_apply(gq.unsqueeze(0),
                           torch.tensor([vec], device=env.device)).squeeze(0)
            vx, vy, vz = (float(c) for c in v)
            elev = math.degrees(math.atan2(vz, math.hypot(vx, vy)))
            line += f"  {label}: elev={elev:+.1f}°"
        log(line)

    # --- roll 영점: q5 별 손가락 닫힘축 world yaw 실측 ----------------------
    # 닫힘축 = gripper frame x축 (jaw 회전축 -y_g 에 수직). top-down probe 자세에서
    # q5 를 바꿔가며 x_g 의 수평 yaw 를 측정 → ik() q5 식의 영점(δ5)·부호 검증.
    log("[CALIB] roll 영점 — q5 별 닫힘축 world yaw (base frame)")
    g_idx2 = names.index("gripper")
    root_yaw = _quat_to_yaw(robot.data.root_quat_w[0])
    for q5_cmd in (-0.6, -0.3, 0.0, 0.3, 0.6):
        q = kin.ik((probe[0], probe[1], probe[2] + 0.05), 0.0)
        if q is None:
            continue
        q[4] = q5_cmd
        act = torch.zeros((env.num_envs, 6), device=env.device)
        act[0, :5] = torch.tensor(q, device=env.device)
        act[0, 5] = 0.5 - GRIPPER_ACTION_OFFSET
        for _ in range(60):
            if not simulation_app.is_running():
                return
            env.step(act)
        gq = robot.data.body_quat_w[0, g_idx2]
        x_g = quat_apply(gq.unsqueeze(0),
                         torch.tensor([[1.0, 0.0, 0.0]], device=env.device)).squeeze(0)
        # world yaw → base frame yaw (root yaw + USD↔URDF 정합 보정)
        yaw_w = math.atan2(float(x_g[1]), float(x_g[0]))
        yaw_b = yaw_w - root_yaw + BASE_YAW_OFFSET
        yaw_b = (yaw_b + math.pi) % (2 * math.pi) - math.pi
        log(f"[CALIB] q5={q5_cmd:+.2f}  닫힘축 yaw_base={math.degrees(yaw_b):+.1f}°  "
            f"(x_g world=({float(x_g[0]):+.3f},{float(x_g[1]):+.3f},{float(x_g[2]):+.3f}))")


# ---------------------------------------------------------------------------
# Part C — State machine
# ---------------------------------------------------------------------------


class Phase(IntEnum):
    SETTLE        = 0   # reset 후 큐브 정착 대기
    APPROACH      = 1   # SAFE_Z 고도로 큐브 상공 횡이동
    PRE_GRASP     = 2   # 큐브 옆 비킨 지점 위 hover — 처짐 보상 수렴 대기 (+latch)
    DESCEND       = 3   # 비킨 지점에서 grasp 높이로 수직 하강 (큐브와 무충돌)
    SLIDE         = 4   # 닫힘축 따라 수평 진입 — 큐브가 손가락 사이로
    GRASP_DWELL   = 5   # 그리퍼 닫힘 정착
    DRAG          = 6   # inner/outer-reach 보정: 낮게 쥔 채 liftable 반경으로 끌기
    LIFT          = 7   # SAFE_Z 로 상승 + 파지 검증
    TRANSPORT     = 8   # SAFE_Z 유지하며 그릇 상공 횡이동
    LOWER         = 9   # 그릇 안 release 높이로 하강
    RELEASE_DWELL = 10  # 그리퍼 열림 정착
    RETREAT       = 11  # SAFE_Z 로 후퇴 → 다음 큐브
    HOME_FINAL    = 12  # 홈 자세 복귀 후 완료
    DONE          = 13


# 이동 단계: max_phase_steps timeout 적용 대상
_MOVE_PHASES = frozenset({
    Phase.APPROACH, Phase.PRE_GRASP, Phase.DESCEND, Phase.SLIDE, Phase.DRAG,
    Phase.LIFT, Phase.TRANSPORT, Phase.LOWER, Phase.RETREAT,
})
# 정밀 수렴(fine_joint_tol) 단계
_FINE_PHASES = frozenset({Phase.PRE_GRASP, Phase.DESCEND, Phase.SLIDE, Phase.LOWER})


class SO101PickPlaceSM:
    """num_envs 병렬 pick-and-place 컨트롤러 (해석적 IK + joint action).

    매 step 모든 env 의 관절 목표를 계산해 ``env.step([N, 6])`` 를 1회 호출한다.
    """

    HOME_Q = [0.0, -0.3, 0.6, 0.6, 0.0]  # 작업영역 위로 접은 대기 자세

    def __init__(self, env, active_cubes: list[str]) -> None:
        self.env      = env
        self.scene    = env.unwrapped.scene
        self.device   = env.unwrapped.device
        self.num_envs = env.unwrapped.num_envs
        self.robot    = self.scene["robot"]
        self.kin      = SO101Kinematics()

        # per-env 큐브 집합: 전체(리포트용) / 미처리 / 현재 대상 / 큐브별 시도 라운드
        self.all_cubes : list[str] = list(active_cubes)
        self.remaining : list[list[str]] = [list(active_cubes) for _ in range(self.num_envs)]
        self.cur_cube  : list[str | None] = [None] * self.num_envs
        self.rounds    : list[dict[str, int]] = [
            {c: 0 for c in active_cubes} for _ in range(self.num_envs)]

        # per-env 상태
        self.phase       : list[Phase] = [Phase.SETTLE] * self.num_envs
        self.n_placed    : list[int]   = [0]   * self.num_envs
        self.dwell_count : list[int]   = [0]   * self.num_envs
        self.phase_steps : list[int]   = [0]   * self.num_envs
        self.retries     : list[int]   = [0]   * self.num_envs
        # latch: (xy world, yaw, z world, pitch) — DESCEND drift 기준 + 목표·자세 고정.
        # z·pitch 까지 고정하는 이유: 하강 중 큐브를 누르면 큐브 z 가 내려가는데
        # 실시간 추적하면 목표도 따라 내려가 더 누르는 악순환 + IK pitch 폴백이
        # 바뀌며 자세가 급변(swing)한다. (이전 SM 확정 함정 — descend tilt 잠금)
        self.latched     : list[tuple | None] = [None] * self.num_envs
        self.grasp_z0    : list[float] = [0.0] * self.num_envs
        # 현재 관절 목표 (IK 실패 시 유지용)
        self.q_cmd       : list[list[float]] = [list(self.HOME_Q) for _ in range(self.num_envs)]
        # 중력 처짐 보상 적분기: PD(stiffness 17.8) 정적 오차(lift ~0.14rad)를
        # action 에 가산해 제거. 매 step bias += KI*(q_cmd - q_now), 클립 ±BIAS_MAX.
        self.q_bias      : list[list[float]] = [[0.0] * 5 for _ in range(self.num_envs)]
        # 최근 IK 가 채택한 접근 pitch (진단용)
        self.last_pitch  : list[float] = [-math.pi / 2] * self.num_envs
        # DESCEND/LOWER 수직 경로 보간용 현재 z 명령 (None = 미시작)
        self.z_ramp      : list[float | None] = [None] * self.num_envs
        # SLIDE/TRANSPORT 수평 ramp 진행 거리 (m) + TRANSPORT 시작점
        self.slide_s     : list[float] = [0.0] * self.num_envs
        # PRE_GRASP 에서 1회 확정하는 (roll_offset, 비킴 dx, dy) — step 간 토글 방지
        self.side_pick   : list[tuple[float, float, float] | None] = [None] * self.num_envs
        # LOWER 재배향 ramp 의 시작 pitch (진입 시점의 실제 pitch — 그릇은 top-down
        # 한계 밖이라 -90° 가정 시작은 즉시 IK 실패)
        self.rot_pitch0  : list[float | None] = [None] * self.num_envs
        self.move_from   : list[tuple[float, float]] = [(0.0, 0.0)] * self.num_envs

    BIAS_KI  = 0.06
    BIAS_MAX = 0.35

    # --- 위치 쿼리 -------------------------------------------------------

    def obj_pos(self, name: str, e: int) -> torch.Tensor:
        return self.scene[name].data.root_pos_w[e, :3].clone()

    def _tcp_meas_w(self, e: int) -> torch.Tensor:
        """gripper body + grasp offset → TCP world 실측 (진단용)."""
        from isaaclab.utils.math import quat_apply

        names = list(self.robot.data.body_names)
        g_idx = names.index("gripper")
        off = torch.tensor([-0.0079, -0.000218121, -0.0981274], device=self.device)
        gp = self.robot.data.body_pos_w[e, g_idx]
        gq = self.robot.data.body_quat_w[e, g_idx]
        return gp + quat_apply(gq.unsqueeze(0), off.unsqueeze(0)).squeeze(0)

    def obj_yaw(self, name: str, e: int) -> float:
        return _quat_to_yaw(self.scene[name].data.root_quat_w[e])

    def _to_base(self, world_xyz: torch.Tensor, e: int) -> tuple[float, float, float]:
        """world → URDF base_link frame (root yaw 역회전 + z 원점 보정)."""
        return _world_to_base(world_xyz, self.robot, e)

    def _safe_z_w(self, e: int) -> float:
        return DESK_TOP_Z + args.safe_height

    # --- 조건 체크 -------------------------------------------------------

    def _converged(self, e: int, tol: float) -> bool:
        q_now = self.robot.data.joint_pos[e, :5].detach().cpu().tolist()
        return max(abs(g - n) for g, n in zip(self.q_cmd[e], q_now)) < tol

    def _grasped(self, cube: str, e: int) -> bool:
        return self.obj_pos(cube, e)[2].item() > self.grasp_z0[e] + args.lift_check

    def _placed(self, cube: str, e: int) -> bool:
        p       = self.obj_pos(cube, e)
        bowl_xy = self.obj_pos(BOWL_NAME, e)[:2]
        in_xy   = torch.linalg.norm(p[:2] - bowl_xy).item() < BOWL_SUCCESS_RADIUS
        z_rel   = p[2].item() - DESK_TOP_Z
        in_z    = BOWL_HEIGHT_RANGE[0] <= z_rel <= BOWL_HEIGHT_RANGE[1] + 0.10
        return in_xy and in_z

    # --- 큐브 크기별 파라미터 ----------------------------------------------

    @staticmethod
    def _open_cmd(cube: str) -> float:
        return (args.gripper_open_large
                if CUBE_SIZES.get(cube, 0.030) >= 0.040 else args.gripper_open)

    @staticmethod
    def _zoff(cube: str) -> float:
        return (args.grasp_z_offset_large
                if CUBE_SIZES.get(cube, 0.030) >= 0.040 else args.grasp_z_offset)

    # --- IK 래퍼 ---------------------------------------------------------

    def _solve(self, e: int, target_w: tuple[float, float, float], yaw_w: float,
               roll_offset: float = 0.0) -> bool:
        """world 목표 → base 변환 → IK. 성공 시 q_cmd 갱신, 실패 시 False (q_cmd 유지)."""
        t = torch.tensor(target_w, device=self.device)
        yaw_b = yaw_w - _quat_to_yaw(self.robot.data.root_quat_w[e]) + BASE_YAW_OFFSET
        out = self.kin.ik_reach(self._to_base(t, e), yaw_b, q_ref=self.q_cmd[e],
                                roll_offset=roll_offset)
        if out is None:
            return False
        self.q_cmd[e], self.last_pitch[e] = out
        return True

    def _solve_fixed_pitch(self, e: int, target_w: tuple[float, float, float],
                           yaw_w: float, pitch: float,
                           roll_offset: float = 0.0) -> bool:
        """latch 된 pitch 로만 IK (폴백 스캔 없음) — DESCEND 자세 급변 방지."""
        t = torch.tensor(target_w, device=self.device)
        yaw_b = yaw_w - _quat_to_yaw(self.robot.data.root_quat_w[e]) + BASE_YAW_OFFSET
        q = self.kin.ik(self._to_base(t, e), yaw_b, pitch=pitch, q_ref=self.q_cmd[e],
                        roll_offset=roll_offset)
        if q is None:
            return False
        self.q_cmd[e], self.last_pitch[e] = q, pitch
        return True

    # --- grasp 후보·클리어런스·우선순위 헬퍼 --------------------------------

    def _grasp_candidates(self, e: int, cube: str) -> list[tuple]:
        """4방향(roll 0/±90/180°) side-approach 후보를 클리어런스와 함께 반환.

        각 후보의 비킨 하강 지점·슬라이드 경로가 그릇/다른 큐브와 충돌하지 않는지
        평가한다. 반환: [(clear:bool, margin:float, roll_off, dx, dy), ...]
        (0° = base쪽 비킴이 첫 후보 — 우선 선호)
        """
        p = self.obj_pos(cube, e)
        yaw = self.obj_yaw(cube, e)
        root_yaw = _quat_to_yaw(self.robot.data.root_quat_w[e])
        # 방위각: pan 축 기준 (PRE_GRASP 의 q1 과 동일 정의)
        tb = self._to_base(p, e)
        q1 = -math.atan2(tb[1], tb[0] - SO101Kinematics.PAN_X)
        q5_base = SO101Kinematics._fold_45((yaw - root_yaw + BASE_YAW_OFFSET) + q1)
        yaw_fixed_w = (q5_base - q1) - BASE_YAW_OFFSET + root_yaw
        bowl_xy = self.obj_pos(BOWL_NAME, e)[:2]
        others = [c for c in self.remaining[e] if c != cube]
        cands = []
        for roll_off in (0.0, math.pi / 2, -math.pi / 2, math.pi):
            ax_yaw = yaw_fixed_w + roll_off
            ddx, ddy = -math.cos(ax_yaw), -math.sin(ax_yaw)  # fixed 반대쪽 비킴
            # 체크 지점: 비킨 하강 지점 + 슬라이드 경로 중간점
            pts = [(p[0].item() + f * args.side_offset * ddx,
                    p[1].item() + f * args.side_offset * ddy) for f in (1.0, 0.5)]
            margin = float("inf")
            for cx, cy in pts:
                margin = min(margin,
                             math.hypot(cx - bowl_xy[0].item(),
                                        cy - bowl_xy[1].item()) - args.bowl_clear)
                for oc in others:
                    op = self.obj_pos(oc, e)
                    margin = min(margin,
                                 math.hypot(cx - op[0].item(), cy - op[1].item())
                                 - args.cube_clear)
            cands.append((margin >= 0.0, margin, roll_off, ddx, ddy))
        return cands

    def _place_xy(self, e: int) -> tuple[float, float]:
        """운반·하강 목표 xy — 그릇 중심을 base 쪽으로 살짝 당겨 reach 마진 확보.
        (성공 판정 반경 6cm 이므로 2cm 안쪽이어도 in-bowl)"""
        b = self.obj_pos(BOWL_NAME, e)
        root_xy = self.robot.data.root_pos_w[e, :2]
        vx, vy = root_xy[0].item() - b[0].item(), root_xy[1].item() - b[1].item()
        d = max(math.hypot(vx, vy), 1e-6)
        return (b[0].item() + 0.02 * vx / d, b[1].item() + 0.02 * vy / d)

    def _liftable(self, e: int, xy_w, z_w: float | None = None) -> bool:
        """해당 world xy 에서 safe_z 까지 들어올릴 IK 해가 존재하는가."""
        z = self._safe_z_w(e) if z_w is None else z_w
        t = torch.tensor([float(xy_w[0]), float(xy_w[1]), z], device=self.device)
        return self.kin.ik_reach(self._to_base(t, e), 0.0) is not None

    def _select_next_cube(self, e: int) -> str | None:
        """남은 큐브 중 다음 대상 선택: 장애물 클리어 > liftable > 근접 순.

        막힌 큐브는 자연히 뒤로 가고, 앞 큐브가 치워지면 클리어해진다.
        라운드 상한(2회) 초과 큐브는 후순위로만 선택.
        """
        if all(self.rounds[e][c] >= 2 for c in self.remaining[e]):
            return None  # 전 큐브 라운드 소진 — 무한 재시도 방지
        base_xy = self.robot.data.root_pos_w[e, :2]
        best = None
        for c in self.remaining[e]:
            p = self.obj_pos(c, e)
            dist = torch.linalg.norm(p[:2] - base_xy).item()
            clear = any(k[0] for k in self._grasp_candidates(e, c))
            liftable = self._liftable(e, (p[0].item(), p[1].item()))
            over_round = self.rounds[e][c] >= 2
            # 정렬키: (라운드 초과 아님, 클리어, liftable, -거리) 큰 것이 우선
            key = (not over_round, clear, liftable, -dist)
            if best is None or key > best[0]:
                best = (key, c)
        return best[1] if best is not None else None

    def _finish_cube(self, e: int, done: bool) -> None:
        """현재 큐브 처리 종료. done=False 면 라운드 +1 하고 remaining 에 유지
        (다른 큐브 처리 후 재시도) — 단 2라운드 초과면 완전 포기."""
        cube = self.cur_cube[e]
        if cube is not None:
            if done:
                self.remaining[e].remove(cube)
            else:
                self.rounds[e][cube] += 1
                if self.rounds[e][cube] >= 2 and len(self.remaining[e]) == 1:
                    log(f"[SM] env{e} {cube}: 라운드 소진 — 완전 포기")
                    self.remaining[e].remove(cube)
        self.cur_cube[e]     = None
        self.phase_steps[e]  = 0
        self.retries[e]      = 0
        self.latched[e]      = None
        self.z_ramp[e]       = None
        self.slide_s[e]      = 0.0
        self.side_pick[e]    = None
        self.rot_pitch0[e]   = None
        if not self.remaining[e]:
            self.dwell_count[e] = 0
            self.phase[e]       = Phase.HOME_FINAL
        else:
            self.phase[e] = Phase.APPROACH

    def _advance_cube(self, e: int) -> None:
        """(구) 다음 큐브로 — 실패 종료 의미로 사용되던 호출부 호환."""
        self._finish_cube(e, done=False)

    def _retry_or_skip(self, e: int, cube: str, reason: str) -> None:
        self.retries[e] += 1
        if self.retries[e] > args.max_retry:
            log(f"[SM] env{e} {cube}: {reason} — 재시도 소진, 뒤로 미룸")
            self._finish_cube(e, done=False)
            return
        log(f"[SM] env{e} {cube}: {reason} — retry {self.retries[e]}/{args.max_retry}")
        self.phase_steps[e] = 0
        self.latched[e]     = None
        self.z_ramp[e]      = None
        self.slide_s[e]     = 0.0
        self.side_pick[e]   = None
        self.phase[e]       = Phase.APPROACH

    # --- per-env 액션 계산 -------------------------------------------------

    def _compute_action(self, e: int) -> tuple[list[float], float]:
        """env e 의 (q_cmd[5], gripper_target) 반환 + phase 전이."""
        ph = self.phase[e]

        if ph == Phase.DONE:
            return self.HOME_Q, args.gripper_open

        # ----- SETTLE / HOME_FINAL: 홈 자세 유지 -----
        if ph in (Phase.SETTLE, Phase.HOME_FINAL):
            self.dwell_count[e] += 1
            if self.dwell_count[e] >= args.settle_steps:
                self.dwell_count[e] = 0
                if ph == Phase.SETTLE:
                    self.phase[e] = Phase.APPROACH
                else:
                    self.phase[e] = Phase.DONE
                    self._report(e)
            return list(self.HOME_Q), args.gripper_open

        if self.cur_cube[e] is None:
            self.cur_cube[e] = self._select_next_cube(e)
            if self.cur_cube[e] is None:
                self.dwell_count[e] = 0
                self.phase[e] = Phase.HOME_FINAL
                return list(self.HOME_Q), args.gripper_open
            log(f"[SM] env{e}: next cube = {self.cur_cube[e]} "
                f"(remaining {self.remaining[e]})")
        cube = self.cur_cube[e]

        if ph in _MOVE_PHASES:
            self.phase_steps[e] += 1
        timeout = self.phase_steps[e] >= args.max_phase_steps
        tol = args.fine_joint_tol if ph in _FINE_PHASES else args.joint_tol

        # ----- APPROACH: 큐브 상공 SAFE_Z 횡이동 -----
        if ph == Phase.APPROACH:
            p = self.obj_pos(cube, e)
            target = (p[0].item(), p[1].item(), self._safe_z_w(e))
            yaw = self.obj_yaw(cube, e)
            if not self._solve(e, target, yaw):
                self._retry_or_skip(e, cube, f"IK 실패(approach {target})")
                return self.q_cmd[e], self._open_cmd(cube)
            if self._converged(e, tol) or timeout:
                self.dwell_count[e] = 0
                self.phase_steps[e] = 0
                self.phase[e]       = Phase.PRE_GRASP
            return self.q_cmd[e], self._open_cmd(cube)

        # ----- PRE_GRASP: 큐브 옆 비킨 지점 위 hover — bias 수렴 후 latch -----
        if ph == Phase.PRE_GRASP:
            p = self.obj_pos(cube, e)
            yaw = self.obj_yaw(cube, e)
            # 비킴 방향 후보 선택 (side-approach + 그릇 회피).
            # fixed finger = 닫힘축 +방향(실측) → 비킴은 항상 닫힘축 −방향(jaw 선두).
            # 큐브는 90° 대칭이라 wrist roll 을 0/±90/180° 돌린 grasp 이 모두 동등 —
            # 닫힘축 자체를 돌려 비킨 하강 지점이 그릇 테두리(중심 0.12m)를 피하는
            # 첫 후보를 채택한다 (0°=base쪽 비킴 우선).
            # 후보 선택은 _grasp_candidates(그릇+다른 큐브 클리어런스)로 1회 확정 —
            # q_cmd 자기참조·경계 토글 진동 방지 (retry 시 재평가).
            if self.side_pick[e] is None:
                cands = self._grasp_candidates(e, cube)
                pick = next((k for k in cands if k[0]), None)  # 클리어 첫 후보
                if pick is None:
                    pick = max(cands, key=lambda k: k[1])      # 전부 막히면 최대 margin
                clear, margin, roll_off, ddx, ddy = pick
                self.side_pick[e] = (roll_off, ddx, ddy)
                if abs(roll_off) > 1e-9 or not clear:
                    log(f"[SM] env{e} {cube}: 장애물 회피 — roll "
                        f"{math.degrees(roll_off):+.0f}° (clear={clear} "
                        f"margin={margin * 1000:.0f}mm)")
            roll_off, dx, dy = self.side_pick[e]
            sx = p[0].item() + args.side_offset * dx
            sy = p[1].item() + args.side_offset * dy
            target = (sx, sy, p[2].item() + args.pregrasp_height)
            if not self._solve(e, target, yaw, roll_offset=roll_off):
                self._retry_or_skip(e, cube, f"IK 실패(pregrasp {target})")
                return self.q_cmd[e], self._open_cmd(cube)
            if self._converged(e, tol) or timeout:
                self.dwell_count[e] += 1
                if self.dwell_count[e] >= args.pregrasp_dwell:
                    # latch: (비킨 지점 xy, yaw, 큐브 z, pitch, 원본 큐브 xy,
                    #         비킴 방향, roll_offset)
                    self.latched[e]     = (
                        torch.tensor([sx, sy], device=self.device), yaw,
                        p[2].item(), self.last_pitch[e], p[:2].clone(), (dx, dy),
                        roll_off)
                    self.dwell_count[e] = 0
                    self.phase_steps[e] = 0
                    self.phase[e]       = Phase.DESCEND
            return self.q_cmd[e], self._open_cmd(cube)

        # ----- DESCEND: 비킨 지점에서 grasp 높이로 수직 하강 (큐브와 무충돌) -----
        if ph == Phase.DESCEND:
            p = self.obj_pos(cube, e)
            # 비켜 내려가므로 큐브 밀림은 없어야 — 감지되면 즉시 재접근 (이상 상황)
            disp = p[:2] - self.latched[e][4]
            if torch.linalg.norm(disp).item() > args.drift_tol:
                self._retry_or_skip(
                    e, cube,
                    f"하강 중 큐브 밀림 (변위 {torch.linalg.norm(disp).item() * 1000:.0f}mm)")
                return self.q_cmd[e], self._open_cmd(cube)
            lx, ly = self.latched[e][0].tolist()
            final_z = self.latched[e][2] + self._zoff(cube)
            # 수직 직선 경로: z 를 descend_speed 로 점진 하강 후 bias 잔차 수렴 대기
            if self.z_ramp[e] is None:
                self.z_ramp[e] = self.latched[e][2] + args.pregrasp_height
            self.z_ramp[e] = max(final_z, self.z_ramp[e] - args.descend_speed / 30.0)
            target = (lx, ly, self.z_ramp[e])
            if not self._solve_fixed_pitch(e, target, self.latched[e][1],
                                           self.latched[e][3],
                                           roll_offset=self.latched[e][6]):
                self._retry_or_skip(e, cube, f"IK 실패(descend {target})")
                return self.q_cmd[e], self._open_cmd(cube)
            tcp_z_err = self._tcp_meas_w(e)[2].item() - final_z
            ramp_done = self.z_ramp[e] <= final_z + 1e-6
            if ramp_done:
                self.dwell_count[e] += 1
            settle_timeout = self.dwell_count[e] >= 12
            if (ramp_done and tcp_z_err < 0.003) or settle_timeout or timeout:
                self.z_ramp[e]      = None
                self.slide_s[e]     = 0.0
                self.dwell_count[e] = 0
                self.phase_steps[e] = 0
                self.phase[e]       = Phase.SLIDE
            return self.q_cmd[e], self._open_cmd(cube)

        # ----- SLIDE: 닫힘축 따라 수평 진입 — 큐브가 손가락 사이로 -----
        if ph == Phase.SLIDE:
            p = self.obj_pos(cube, e)
            # 큐브가 크게 밀려나면(진입 실패) 재접근
            disp = p[:2] - self.latched[e][4]
            if torch.linalg.norm(disp).item() > 0.025:
                self._retry_or_skip(
                    e, cube,
                    f"slide 중 큐브 밀림 (변위 {torch.linalg.norm(disp).item() * 1000:.0f}mm)")
                return self.q_cmd[e], self._open_cmd(cube)
            sx, sy = self.latched[e][0].tolist()
            dx, dy = self.latched[e][5]
            final_z = self.latched[e][2] + self._zoff(cube)
            slide_len = max(0.0, args.side_offset - args.slide_stop)
            self.slide_s[e] = min(slide_len,
                                  self.slide_s[e] + args.slide_speed / 30.0)
            target = (sx - self.slide_s[e] * dx, sy - self.slide_s[e] * dy, final_z)
            if not self._solve_fixed_pitch(e, target, self.latched[e][1],
                                           self.latched[e][3],
                                           roll_offset=self.latched[e][6]):
                self._retry_or_skip(e, cube, f"IK 실패(slide {target})")
                return self.q_cmd[e], self._open_cmd(cube)
            slide_done = self.slide_s[e] >= slide_len - 1e-6
            if slide_done:
                self.dwell_count[e] += 1
            if (slide_done and self._converged(e, tol)) \
                    or self.dwell_count[e] >= 8 or timeout:
                self.grasp_z0[e]    = p[2].item()
                self.dwell_count[e] = 0
                self.phase_steps[e] = 0
                self.phase[e]       = Phase.GRASP_DWELL
                tcp = self._tcp_meas_w(e).detach().cpu().tolist()
                err_xy = math.hypot(tcp[0] - p[0].item(), tcp[1] - p[1].item())
                log(f"[SM] env{e} {cube}: grasp 진입 — TCP=({tcp[0]:.4f},{tcp[1]:.4f},{tcp[2]:.4f}) "
                    f"cube=({p[0].item():.4f},{p[1].item():.4f},{p[2].item():.4f}) "
                    f"err_xy={err_xy * 1000:.1f}mm err_z={(tcp[2] - p[2].item()) * 1000:.1f}mm "
                    f"pitch={math.degrees(self.last_pitch[e]):.0f}° timeout={timeout}")
            return self.q_cmd[e], self._open_cmd(cube)

        # ----- GRASP_DWELL: 위치 유지 + 그리퍼 닫힘 정착 -----
        if ph == Phase.GRASP_DWELL:
            self.dwell_count[e] += 1
            if self.dwell_count[e] >= args.grasp_dwell:
                self.dwell_count[e] = 0
                self.phase_steps[e] = 0
                lx, ly = self.latched[e][0].tolist()
                if self._liftable(e, (lx, ly)):
                    self.phase[e] = Phase.LIFT
                else:
                    # inner/outer-reach: 이 자리에선 safe_z 로 못 든다 — 끌어오기
                    self.slide_s[e] = 0.0
                    self.phase[e]   = Phase.DRAG
                    log(f"[SM] env{e} {cube}: liftable 아님 — DRAG 시작")
            return self.q_cmd[e], args.gripper_close

        # ----- DRAG: 낮게 쥔 채 liftable 반경으로 radial 끌기 -----
        if ph == Phase.DRAG:
            lx, ly = self.latched[e][0].tolist()
            root_xy = self.robot.data.root_pos_w[e, :2]
            rx, ry = lx - root_xy[0].item(), ly - root_xy[1].item()
            r_cur = math.hypot(rx, ry)
            # 목표 반경 0.20(중앙 영역) 방향으로 — 가까우면 밀고 멀면 당김
            sgn = 1.0 if r_cur < 0.20 else -1.0
            ux, uy = rx / max(r_cur, 1e-6), ry / max(r_cur, 1e-6)
            self.slide_s[e] += args.slide_speed / 30.0
            step_d = min(self.slide_s[e], 0.10)  # 끌기 한계 10cm
            nx, ny = lx + sgn * step_d * ux, ly + sgn * step_d * uy
            drag_z = self.latched[e][2] + self._zoff(cube) + 0.02  # 살짝 띄워 마찰 감소
            if not self._solve_fixed_pitch(e, (nx, ny, drag_z), self.latched[e][1],
                                           self.latched[e][3],
                                           roll_offset=self.latched[e][6]):
                # 끌기 경로 IK 실패 — 현 위치에서 그냥 lift 시도
                self.phase_steps[e] = 0
                self.phase[e]       = Phase.LIFT
                return self.q_cmd[e], args.gripper_close
            if self._liftable(e, (nx, ny)) or step_d >= 0.10 or timeout:
                # latch xy 를 끌어온 위치로 갱신 후 lift
                self.latched[e] = (torch.tensor([nx, ny], device=self.device),
                                   *self.latched[e][1:])
                self.slide_s[e]     = 0.0
                self.phase_steps[e] = 0
                self.phase[e]       = Phase.LIFT
                log(f"[SM] env{e} {cube}: DRAG 완료 — {step_d * 1000:.0f}mm 이동")
            return self.q_cmd[e], args.gripper_close

        # ----- LIFT: SAFE_Z 로 z-ramp 상승 + 파지 검증 -----
        if ph == Phase.LIFT:
            # 상승 중 이탈 조기 감지 — 빈손 운반(게이트 영구 미충족) 방지
            if (self.z_ramp[e] is not None
                    and torch.linalg.norm(
                        self.obj_pos(cube, e) - self._tcp_meas_w(e)).item() > 0.055):
                self._retry_or_skip(e, cube, "상승 중 큐브 이탈")
                return self.q_cmd[e], self._open_cmd(cube)
            lx, ly = self.latched[e][0].tolist()
            safe_z = self._safe_z_w(e)
            if self.z_ramp[e] is None:
                self.z_ramp[e] = self.latched[e][2] + self._zoff(cube)
            self.z_ramp[e] = min(safe_z, self.z_ramp[e] + args.lift_speed / 30.0)
            self._solve(e, (lx, ly, self.z_ramp[e]), self.latched[e][1],
                        roll_offset=self.latched[e][6])
            ramp_done = self.z_ramp[e] >= safe_z - 1e-6
            if (ramp_done and self._converged(e, tol)) or timeout:
                self.z_ramp[e] = None
                if not self._grasped(cube, e):
                    self._retry_or_skip(e, cube, "파지 실패(큐브 미상승)")
                    return self.q_cmd[e], self._open_cmd(cube)
                # TRANSPORT 수평 ramp 시작점 = 현재 위치
                self.move_from[e]   = (lx, ly)
                self.slide_s[e]     = 0.0
                self.phase_steps[e] = 0
                self.phase[e]       = Phase.TRANSPORT
            return self.q_cmd[e], args.gripper_close

        # ----- TRANSPORT: 그릇 상공으로 Cartesian 직선 ramp 횡이동 -----
        if ph == Phase.TRANSPORT:
            # drop 감지: 운반 중 큐브가 손에서 이탈(마찰 DR 하한 env) → 다시 집으러
            tcp_now = self._tcp_meas_w(e)
            cube_p = self.obj_pos(cube, e)
            if torch.linalg.norm(cube_p - tcp_now).item() > 0.055:
                log(f"[SM] env{e} {cube}: 운반 중 drop 감지 — 재시도")
                self._finish_cube(e, done=False)
                return self.q_cmd[e], args.gripper_open
            bx, by = self._place_xy(e)
            # 낙하점 보정: 쥔 큐브가 TCP 에서 닫힘축 방향 2~3cm 오프셋 — '큐브'가
            # 그릇 중심 위에 오도록 목표를 반대로 이동 (release 는 자세 불변이라
            # 여기서 맞춰 두면 그대로 유효)
            off_x = cube_p[0].item() - tcp_now[0].item()
            off_y = cube_p[1].item() - tcp_now[1].item()
            off_n = math.hypot(off_x, off_y)
            if off_n > 0.04:
                off_x, off_y = off_x * 0.04 / off_n, off_y * 0.04 / off_n
            bx, by = bx - off_x, by - off_y
            fx, fy = self.move_from[e]
            dist = math.hypot(bx - fx, by - fy)
            # pitch·roll 을 운반 진행률에 맞춰 동시 점진 보간 — ik_reach 가 그릇
            # 앞에서 pitch 를 단번에 완화(-90→-40°)하면 slew 풀속도 재배향으로
            # 팔이 위로 휘둘렸다 그릇에 떨어진다(슬램덩크). roll 90°(release 자세,
            # 사용자 지정)도 운반 중에 함께 돌린다.
            if self.rot_pitch0[e] is None:
                pitch_start = self.last_pitch[e]
                goal = self.kin.ik_reach(
                    self._to_base(torch.tensor([bx, by, self._safe_z_w(e)],
                                               device=self.device), e), 0.0)
                pitch_goal = goal[1] if goal is not None else pitch_start
                q5_now = self.q_cmd[e][4]
                roll_sign = 1.0 if (q5_now + math.pi / 2
                                    <= SO101Kinematics.JOINT_LIMITS[4][1]) else -1.0
                self.rot_pitch0[e] = (pitch_start, pitch_goal, roll_sign)
            pitch_start, pitch_goal, roll_sign = self.rot_pitch0[e]
            self.slide_s[e] = min(dist, self.slide_s[e] + args.transport_speed / 30.0)
            frac = 1.0 if dist < 1e-6 else self.slide_s[e] / dist
            pitch_now = pitch_start + (pitch_goal - pitch_start) * frac
            roll_now = roll_sign * (math.pi / 2) * frac
            target = (fx + (bx - fx) * frac,
                      fy + (by - fy) * frac, self._safe_z_w(e))
            if not self._solve_fixed_pitch(e, target, 0.0, pitch_now,
                                           roll_offset=roll_now):
                # 보간 자세 일시 불가 — 이전 자세 유지하며 ramp 계속
                if self.phase_steps[e] % 60 == 1:
                    log(f"[SM] env{e}: TRANSPORT IK 일시 실패 (frac={frac:.2f})")
                if timeout:
                    log(f"[SM] env{e}: 그릇 IK 실패 — 작업 중단")
                    self._advance_cube(e)
                return self.q_cmd[e], args.gripper_close
            ramp_done = self.slide_s[e] >= dist - 1e-6
            if (ramp_done and self._converged(e, tol)) or timeout:
                self.rot_pitch0[e]  = None
                self.slide_s[e]     = 0.0
                self.dwell_count[e] = 0
                self.phase_steps[e] = 0
                self.phase[e]       = Phase.LOWER
            return self.q_cmd[e], args.gripper_close

        # ----- LOWER(ROTATE): 자세 불변, wrist roll 만 +90° ramp 후 release -----
        # 사용자 지정: FK/IK 재계산 금지 — 다른 joint 는 그대로 두고 q5 만 돌린다.
        # 닫힘축이 접근축 주위로 90° 돌아 jaw 가 옆으로 열림 → 퍼올림 방지.
        # 하강도 하지 않는다 — 안전 고도에서 그대로 떨굼.
        if ph == Phase.LOWER:
            # 회전은 TRANSPORT 에서 이미 완료 — 여기선 정착 확인만 (짧은 dwell)
            self.dwell_count[e] += 1
            rot_t = 1.0
            if (self.dwell_count[e] >= 5 and self._converged(e, tol)) or timeout:
                self.rot_pitch0[e]  = None
                self.dwell_count[e] = 0
                self.phase_steps[e] = 0
                self.phase[e]       = Phase.RELEASE_DWELL
            return self.q_cmd[e], args.gripper_close

        # ----- RELEASE_DWELL: 그리퍼 열림 정착 -----
        if ph == Phase.RELEASE_DWELL:
            self.dwell_count[e] += 1
            if self.dwell_count[e] >= args.release_dwell:
                self.dwell_count[e] = 0
                self.n_placed[e]   += 1
                self.phase_steps[e] = 0
                self.phase[e]       = Phase.RETREAT
            return self.q_cmd[e], self._open_cmd(cube)

        # ----- RETREAT: 그릇 위 z-ramp 상승 — ramp 종료 즉시 다음 큐브 -----
        # (수렴 대기 생략: 다음 APPROACH 가 SAFE_Z 상공 목표라 자연스럽게 이어받음)
        if ph == Phase.RETREAT:
            b = self.obj_pos(BOWL_NAME, e)
            safe_z = self._safe_z_w(e)
            if self.z_ramp[e] is None:
                # 시작점 = 실측 TCP z — 고정 가정(place_height)은 release 자세가
                # 더 높을 때 '내려갔다 올라오는' 명령이 되어 그릇을 친다
                self.z_ramp[e] = self._tcp_meas_w(e)[2].item()
            self.z_ramp[e] = min(safe_z, self.z_ramp[e] + args.lift_speed / 30.0)
            self._solve(e, (b[0].item(), b[1].item(), self.z_ramp[e]), 0.0)
            if self.z_ramp[e] >= safe_z - 1e-6 or timeout:
                self._finish_cube(e, done=True)
            return self.q_cmd[e], self._open_cmd(cube)

        return list(self.HOME_Q), args.gripper_open

    # --- 배치 액션 + 메인 루프 ---------------------------------------------

    def _act_all(self, q_list: list[list[float]], grip_targets: list[float]) -> None:
        action = torch.zeros((self.num_envs, 6), device=self.device)
        for e in range(self.num_envs):
            # 적분 보상 갱신 (중력 처짐 제거)
            q_now = self.robot.data.joint_pos[e, :5].detach().cpu().tolist()
            for j in range(5):
                b = self.q_bias[e][j] + self.BIAS_KI * (q_list[e][j] - q_now[j])
                self.q_bias[e][j] = max(-self.BIAS_MAX, min(self.BIAS_MAX, b))
            action[e, :5] = torch.tensor(
                [q + b for q, b in zip(q_list[e], self.q_bias[e])], device=self.device)
            action[e, 5]  = grip_targets[e] - GRIPPER_ACTION_OFFSET
        self.env.step(action)

    def run(self) -> None:
        while not all(p == Phase.DONE for p in self.phase):
            if not simulation_app.is_running():
                break
            qs, grips = [], []
            for e in range(self.num_envs):
                q, g = self._compute_action(e)
                qs.append(q)
                grips.append(g)
            self._act_all(qs, grips)
        # 전체 env 합산 요약
        total_ok = sum(
            sum(self._placed(c, e) for c in self.all_cubes)
            for e in range(self.num_envs))
        total = self.num_envs * len(self.all_cubes)
        log(f"[SM] TOTAL: {total_ok}/{total} cubes in bowl across {self.num_envs} envs "
            f"({100.0 * total_ok / max(total, 1):.0f}%).")

    # --- 결과 리포트 -----------------------------------------------------

    def _report(self, e: int) -> None:
        cubes = self.all_cubes
        n_ok  = sum(self._placed(c, e) for c in cubes)
        for c in cubes:
            p    = self.obj_pos(c, e)
            dist = torch.linalg.norm(p[:2] - self.obj_pos(BOWL_NAME, e)[:2]).item()
            log(f"[SM] env{e} {c}: dist_bowl={dist:.3f}m z={p[2].item():.3f} "
                f"placed={self._placed(c, e)}")
        log(f"[SM] env{e} RESULT: {n_ok}/{len(cubes)} cubes in bowl.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    log("[SM] main entered.")
    env_cfg                = PickCubeEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed           = args.seed
    _apply_dr(env_cfg)
    # SM 도중 RL termination 에 의한 reset 차단 (rule-based 실행이라 종료는 SM 이 관리)
    env_cfg.episode_length_s          = 600.0
    env_cfg.terminations.success      = None
    env_cfg.terminations.cube_lost    = None
    env_cfg.viewer.eye    = args.view_eye
    env_cfg.viewer.lookat = args.view_lookat
    log("[SM] env_cfg built — calling gym.make.")

    if args.video:
        env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
        os.makedirs("docs", exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder="docs",
            name_prefix=args.video_name,
            step_trigger=lambda step: step == 0,
            video_length=args.video_length,
            disable_logger=True,
        )
    else:
        env = gym.make(args.task, cfg=env_cfg).unwrapped
    log("[SM] env created.")
    env.reset()
    log("[SM] reset done — DR applied.")

    if args.calibrate:
        run_calibration(env.unwrapped if args.video else env)
        env.close()
        return

    sm = SO101PickPlaceSM(env, CUBE_NAMES[: args.active_objects])
    sm.run()

    if not args.headless and not args.video:
        while simulation_app.is_running():
            sm._act_all([list(sm.HOME_Q)] * sm.num_envs,
                        [args.gripper_open] * sm.num_envs)

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
