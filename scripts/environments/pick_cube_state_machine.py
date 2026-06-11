"""PickCube rule-based state machine expert.

RL 전에 cube_task 씬이 물리적으로 pick-and-place 가능한지 증명하기 위한
scripted controller다. 큐브/그리퍼를 순간이동하지 않고, gripper tip 작업점을
목표 위치에 맞춘 뒤 joint-position action으로 실행한다.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import math
import multiprocessing
import os
import re
from pathlib import Path
import shutil
import sys
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from isaaclab.app import AppLauncher

# C 레벨 크래시(access violation 등)의 Python traceback 을 파일로 덤프.
os.makedirs("outputs", exist_ok=True)
_FH_FILE = open(os.path.abspath("outputs/pick_cube_sm_faulthandler.txt"), "w")
faulthandler.enable(file=_FH_FILE)


if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)


parser = argparse.ArgumentParser(description="PickCube rule-based state machine expert")
parser.add_argument("--task", default="SimToReal-SO101-PickCube-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--active_objects", type=int, default=1, choices=[1, 2, 3, 4])
# 기본값 1.0 = full domain randomization (teleop 수집과 동일 조건). 0.0 으로 주면
# DR 을 끈 fixed-spawn 이 되는데, 이는 평가 hacking 이므로 deliverable 에서는 쓰지 않는다.
parser.add_argument("--object_radius_scale", type=float, default=1.0)
parser.add_argument("--container_angle_scale", type=float, default=1.0)
parser.add_argument("--container_radius_scale", type=float, default=1.0)
# 모션 단계(approach/descend/lift/transport/place/retreat)의 *_steps 는 이제 상한(cap)이다.
# 단계가 성공(목표 도달/그리퍼 전이 완료)하면 _phase 가 early-exit 하므로 5 rad/s 에서 보통
# 훨씬 일찍 끝난다. settle/hold 류 대기는 사용자 요청대로 최소화했다.
parser.add_argument("--settle_steps", type=int, default=12, help="reset 후 물리 안정화(큐브 정지) 대기")
parser.add_argument("--approach_steps", type=int, default=120)
parser.add_argument("--descend_steps", type=int, default=160)
parser.add_argument("--close_steps", type=int, default=20)
parser.add_argument(
    "--grasp_settle_steps",
    type=int,
    default=5,
    help="그리퍼가 닫힌 뒤 grip 이 큐브에 안정적으로 물릴 때까지의 추가 dwell(step).",
)
parser.add_argument("--lift_steps", type=int, default=120)
parser.add_argument("--transport_steps", type=int, default=160)
parser.add_argument("--place_steps", type=int, default=120)
parser.add_argument("--open_steps", type=int, default=4)
parser.add_argument("--final_settle_steps", type=int, default=2)
parser.add_argument("--retreat_steps", type=int, default=100)
parser.add_argument("--idle_home_steps", type=int, default=30)
parser.add_argument("--command_settle_steps", type=int, default=0)
parser.add_argument("--max_grasp_attempts", type=int, default=3)
parser.add_argument("--approach_height", type=float, default=0.10)
parser.add_argument(
    "--lift_height",
    type=float,
    default=0.08,
    help="잡은 뒤 들어올리는 높이(m). 책상·주변 큐브(4cm)만 넘기면 충분 — 0.15 는 '번쩍' 과도 상승.",
)
parser.add_argument("--transport_height", type=float, default=0.12)
parser.add_argument(
    "--place_height",
    type=float,
    default=0.12,
    help=(
        "그릇 root 기준 release 높이(m). transport_height 와 같게 두면 place_descend 가 사실상 "
        "생략돼 이송 직후 바로 떨군다(그릇 테두리 ~5cm 위 ~7cm에서 낙하 — 그릇 내부가 미끄러워 "
        "큐브가 바닥 중앙으로 모이고, 낮게 넣으면 그리퍼가 그릇을 건드린다)."
    ),
)
parser.add_argument(
    "--stack_place_height_increment",
    type=float,
    default=0.025,
    help="Additional place height per cube already inside the bowl, reducing gripper-pile collisions.",
)
parser.add_argument("--grasp_z_offset", type=float, default=0.005, help="(legacy) cube center 기준 grasp z 오프셋. floor grasp 미사용 시 fallback.")
parser.add_argument(
    "--grasp_pick_offset",
    type=float,
    default=-0.005,
    help=(
        "Descend/grasp grasp-point 목표 z = 큐브 중심 + 이 값(m). 사용자 텔레옵 시연 사진 기준"
        "(두 finger 가 큐브 측면을 바닥 근처까지 깊게 감쌈) 중심 -5mm. 과거 음수 실험(v4 -8mm "
        "80%)은 중간tilt 기조에서의 결과 — 저tilt(수직) ladder 와 조합은 별개로 재검증한다. "
        "얕은 그립(hold 평균 18mm)이 40mm 큐브 실패(v18 Cube3 25%)·운반 낙하의 직접 원인."
    ),
)
parser.add_argument(
    "--grasp_lateral_offset",
    type=float,
    default=0.0,
    help=(
        "Descend/grasp 목표를 그리퍼 개방축(고정 finger→이동 jaw)을 따라 옆으로 미는 거리(m). 양수=이동 jaw "
        "쪽, 음수=고정 finger 쪽. 큐브(2.5cm)가 두 jaw 사이 gap 에 들어오게 해 고정 finger 가 큐브를 "
        "위에서 찌르는 것을 막는다. 큐브 half(0.0125) 부근에서 튜닝."
    ),
)
parser.add_argument(
    "--descend_tolerance",
    type=float,
    default=0.014,
    help="Descend 단계 early-exit 허용 오차(m). 일반 tolerance 보다 빡빡하게 해 grasp point 를 큐브까지 충분히 내린다.",
)
parser.add_argument(
    "--grasp_misalign_gate",
    type=float,
    default=0.02,
    help=(
        "닫기 직전 grasp point↔큐브(현 위치) 오차가 이 값(m)을 넘으면 닫지 않고 그 attempt 를 "
        "버린다(마지막 attempt 제외). 어긋난 채 닫으면 헛닫기 + 닫는 손가락이 큐브를 밀어 다음 "
        "시도를 악화시키는 악순환만 남는다(v10 4큐브: 실패 attempt 의 닫기 시점 오차 평균 2.1cm, "
        "시도마다 큐브 2~7cm 표류 → 책상 밖/reach 밖 이탈). 0 이면 비활성."
    ),
)
parser.add_argument(
    "--ik_ori_weight",
    type=float,
    default=0.06,
    help="Isaac-frame DLS refine 에서 top-down(접근축 수직) orientation task 가중치. 0 이면 위치만.",
)
parser.add_argument("--target_tolerance", type=float, default=0.014)
parser.add_argument("--ik_damping", type=float, default=0.05)
parser.add_argument("--ik_gain", type=float, default=0.85)
parser.add_argument("--max_joint_delta", type=float, default=0.075)
parser.add_argument(
    "--enable_jacobian_refine",
    action="store_true",
    help="Enable final local Jacobian refinement after random-FK waypoint selection. Off by default; current cube grasp is more stable without it.",
)
parser.add_argument(
    "--disable_jacobian_refine",
    action="store_true",
    help=argparse.SUPPRESS,
)
parser.add_argument(
    "--max_arm_step_delta",
    type=float,
    default=0.16667,
    help="Max per-step arm joint command change in radians (0.16667 ~= 5.0 rad/s at 30 Hz)",
)
parser.add_argument(
    "--max_gripper_step_delta",
    type=float,
    default=0.02,
    help=(
        "GRASP(close-on-cube) 단계 전용 그리퍼 per-step 한계(rad). 큐브를 무는 close 는 천천히 해야 "
        "(0.02≈0.6 rad/s) 큐브가 손가락 사이에 안착해 마찰로 잡힌다 — 빠르게 닫으면 큐브를 쳐낸다. "
        "그 외 단계(open/이동)는 --gripper_travel_step_delta 로 빠르게 움직인다."
    ),
)
parser.add_argument(
    "--gripper_travel_step_delta",
    type=float,
    default=0.06,
    help=(
        "GRASP 외 단계(open/descend 부분개방/release)의 그리퍼 per-step 한계(rad, ≈1.8 rad/s). "
        "과거엔 전 구간 0.005 로 묶여 open 1.0 rad 에 6.7초가 걸려 팔이 단계 사이에 정지해 보였다."
    ),
)
parser.add_argument(
    "--grasp_arm_step_delta",
    type=float,
    default=0.0,
    help=(
        "Descend/grasp 단계 전용 팔 per-step 한계(rad). 0=비활성(--max_arm_step_delta 사용). >0 이면 grasp "
        "단계에서만 팔을 더 천천히 움직여(예: 0.05≈1.5 rad/s) 큐브를 쳐내지 않고 정밀 접근한다. 접근/이송은 "
        "여전히 --max_arm_step_delta(5 rad/s)로 빠르게. (5 rad/s 는 상한 cap 이라 더 느리게 움직여도 무방.)"
    ),
)
parser.add_argument("--fk_samples", type=int, default=5000)
parser.add_argument(
    "--continuity_weight",
    type=float,
    default=0.05,
    help=(
        "random-FK 후보 점수 = dist + 이 값*|q-q_now|. 0.015(과거 기본)는 너무 낮아 FK 가 global 샘플로 "
        "큰 자세 점프를 골라 5 rad/s 팔이 큐브를 쳐내 날렸다(full-DR 평균 1.4/4). 0.05 면 도달 유지+가까운 "
        "자세 선호로 매끄러운 궤적 → flinging 거의 소멸(full-DR 평균 3.0/4, all-4 0%→40%)."
    ),
)
parser.add_argument(
    "--grasp_tilt_weight",
    type=float,
    default=0.0,
    help=(
        "Descend/grasp FK 후보 선택 시, 이동 jaw 가 바닥(CUBE_DESK_TOP_Z)까지 + 고정 finger 아래로 "
        "내려가는(=tilt 강한) pose 를 선호하는 점수 가중치. 0=off(거리·연속성만). SO-101 그리퍼는 "
        "고정 finger 가 길어 약한 tilt 면 이동 jaw 가 큐브 위에 남아 grasp 실패 — 이 항이 검증된 강tilt "
        "grasp(이동 jaw 가 큐브 바닥까지) 를 결정적으로 선택하게 한다."
    ),
)
parser.add_argument(
    "--grasp_config_mode",
    choices=["random_fk", "deterministic"],
    default="random_fk",
    help=(
        "joint waypoint 계산 방식. random_fk(기본)=기존 random-FK 5000샘플(grasp 자세 '추첨', "
        "단일 grasp ~67%%). deterministic=in-sim 가상 FK 위 finite-difference DLS — 난수 없이 "
        "동일 큐브 pose → 동일 q_goal. grasp 단계는 tilt ladder(접근축 후보)를 결정적 순서로 "
        "시도하고 scoop 점수(이동 jaw 가 바닥까지+고정 finger 아래)로 선택, 큐브 yaw 에 그리퍼 "
        "개방축을 정렬한다. 수렴 실패 시 random_fk 로 자동 폴백. --controller_mode joint_fk 전용."
    ),
)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument(
    "--num_episodes",
    type=int,
    default=1,
    help=(
        "신뢰성 sweep: >1 이면 한 Isaac 세션에서 env.reset()+SM 을 N회 반복하고(매 reset 마다 DR "
        "재추첨) all-4 성공률·per-cube 성공률을 집계해 --output_json 에 기록한다. dataset 기록은 무시한다."
    ),
)
parser.add_argument("--gripper_open", type=float, default=1.0)
parser.add_argument("--gripper_closed", type=float, default=0.0)
parser.add_argument(
    "--descend_gripper",
    type=float,
    default=0.5,
    help=(
        "Descend 동안 그리퍼 개방 정도(0=닫힘,1=완전개방). 완전개방이면 벌어진 손가락이 책상에 닿아 "
        "grasp point 가 큐브 위 ~1.6cm 에서 멈춘다(top-down). 덜 벌리면 손가락이 더 수직이라 큐브까지 "
        "내려가 잡을 수 있다. 큐브(2.5cm)가 들어갈 만큼은 벌려야 한다."
    ),
)
parser.add_argument("--control_point", choices=["jaw_offset", "midpoint"], default="jaw_offset")
parser.add_argument(
    "--controller_mode",
    choices=["joint_fk", "diff_ik", "rmpflow", "lula_ik", "ikpy"],
    default="ikpy",
    help=(
        "ikpy(기본)=ikpy IK(경로3, orientation 제약 가능); joint_fk=random-FK joint targets; "
        "diff_ik/rmpflow=Isaac Lab task-space; lula_ik=Lula LulaKinematicsSolver IK(경로2). "
        "lula_ik/ikpy 는 rmpflow 와 같은 phase driver 슬롯을 쓴다(매 스텝 live target 재독)."
    ),
)
parser.add_argument(
    "--disable_topdown",
    action="store_true",
    help="Disable top-down(palm-down) orientation constraint everywhere (drop 도 자유 자세).",
)
parser.add_argument(
    "--topdown_pick",
    action="store_true",
    help=(
        "Pick(approach/descend/grasp/lift) 에도 strict top-down 을 강제한다. 기본 off — 이 5-DOF 팔은 "
        "수직 자세로는 책상 위 큐브에 손이 닿지 않아(jaw 가 ~3cm 위에서 멈춤) grasp 가 실패한다. "
        "기본은 자연 tilt 로 위에서 내려 집고, drop 만 palm-down(level) 으로 한다."
    ),
)
parser.add_argument(
    "--ikpy_orientation_mode",
    choices=["Z", "Y", "X", "all"],
    default="Z",
    help="ikpy orientation_mode for the top-down constraint. Z=approach축(local Z)만 world −Z 로 정렬.",
)
parser.add_argument(
    "--ik_gripper_closed",
    type=float,
    default=0.0,
    help="Closed gripper joint target used by --controller_mode diff_ik.",
)
parser.add_argument(
    "--diff_ik_step_size",
    type=float,
    default=0.012,
    help="Max per-step relative Cartesian command in meters for --controller_mode diff_ik.",
)
parser.add_argument(
    "--rmpflow_internal_rollout_steps",
    type=int,
    default=90,
    help="Internal RMPFlow frames rolled out to produce a phase joint target before external slew limiting.",
)
parser.add_argument(
    "--disable_rmpflow_jacobian_refine",
    action="store_true",
    help="Disable USD Jacobian grasp-point refinement in the second half of RMPFlow phases.",
)
parser.add_argument(
    "--object_order",
    choices=["raster", "name", "near_bowl_first", "far_bowl_first", "hard_first", "far_base_first"],
    default="raster",
    help=(
        "Object execution order. raster(기본)=top 카메라 기준 좌상단→우하단(위 y큰 행부터, 행 안에서 "
        "왼 x작은 것부터); name=Cube1~4 이름순; near/far_bowl_first=y 정렬; hard_first=큰/먼 큐브 우선; "
        "far_base_first=로봇 base 에서 먼(reach 어려운) 큐브 먼저(그릇 비었을 때 처리해 기존 큐브 안 침)."
    ),
)
parser.add_argument(
    "--raster_row_band",
    type=float,
    default=0.06,
    help="raster 정렬에서 같은 '행'으로 묶는 y 밴드 폭(m). 이 폭 안의 큐브는 x(왼→오)로 정렬한다.",
)
parser.add_argument(
    "--object_cycles",
    type=int,
    default=1,
    help="Repeat the object order this many times, skipping cubes already inside the bowl.",
)
parser.add_argument(
    "--bowl_place_offset_radius",
    type=float,
    default=0.022,
    help="XY radius for per-cube bowl placement offsets in meters.",
)
parser.add_argument("--disable_dynamic_gripper_effort", action="store_true")
parser.add_argument("--min_gripper_effort", type=float, default=0.5)
parser.add_argument(
    "--carry_min_gripper_effort",
    type=float,
    default=0.5,
    help="Minimum gripper effort during lift/transport/place closed-gripper phases.",
)
parser.add_argument("--output_json", type=Path, default=Path("outputs/pick_cube_state_machine.json"))
parser.add_argument("--dataset_dir", type=Path, default=None, help="Optional LeRobot v3 episode output directory")
parser.add_argument("--expert_dataset_pt", type=Path, default=None, help="Optional raw rl_state/action expert dataset (.pt)")
parser.add_argument("--record_seconds", type=float, default=30.0, help="Seconds to record when --dataset_dir is set")
parser.add_argument(
    "--episode_length_s",
    type=float,
    default=None,
    help="Optional simulation episode timeout override. Defaults to a conservative multi-cube estimate.",
)
parser.add_argument("--overwrite_dataset", action="store_true", help="Replace --dataset_dir if it already exists")
parser.add_argument("--no_videos", action="store_true", help="Skip camera videos in the LeRobot dataset")
parser.add_argument("--warmup_steps", type=int, default=5, help="Render warmup steps before recording starts")
parser.add_argument("--gui", action="store_true", help="Open the Isaac GUI instead of forced headless mode")
parser.add_argument(
    "--review_video_dir",
    type=Path,
    default=None,
    help=(
        "에피소드별 리뷰 영상(mp4) 출력 디렉터리. 별도 뷰어 카메라(근거리 사선 부감)로 매 step "
        "캡처해 30fps 실시간 영상을 만든다. sweep(--num_episodes>1) 에선 epNN_{ok,fail}.mp4 로 저장."
    ),
)
parser.add_argument("--review_cam_eye", type=str, default="2.45,-0.40,1.10", help="뷰어 카메라 world 위치 x,y,z")
parser.add_argument("--review_cam_target", type=str, default="1.78,-0.37,0.72", help="뷰어 카메라 look-at 지점 x,y,z")
parser.add_argument("--review_cam_focal", type=float, default=18.0, help="뷰어 카메라 focal length (mm)")
parser.add_argument(
    "--review_pose_check",
    action="store_true",
    help="뷰어 카메라 정지 프레임 PNG(pose_check.png)만 저장하고 종료 — 구도 확인용.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = not args.gui
if args.livestream < 0:
    args.livestream = 0
args.enable_cameras = (args.dataset_dir is not None and not args.no_videos) or args.review_video_dir is not None

if args.num_envs != 1:
    raise ValueError("pick_cube_state_machine.py currently supports --num_envs=1 only.")

if args.review_pose_check and args.review_video_dir is None:
    raise ValueError("--review_pose_check 는 --review_video_dir 와 함께 사용한다.")

if args.grasp_config_mode == "deterministic" and args.controller_mode != "joint_fk":
    raise ValueError(
        "--grasp_config_mode deterministic 은 --controller_mode joint_fk 에서만 동작한다 "
        "(다른 모드는 phase driver 가 waypoint 를 계산)."
    )

# 커스텀 인자(task/num_envs/dataset_dir 등)를 통째로 넘기면 AppLauncher → carb
# 설정 경로로 전달되어 Windows에서 _prepare_ui access violation 위험.
# AppLauncher가 실제로 사용하는 키만 필터링해서 전달한다.
_LAUNCHER_KEYS = {
    "headless", "livestream", "enable_cameras", "experience", "device", "cpu",
    "disable_fabric", "offscreen_render", "kit_args",
}
_launcher_args = {k: v for k, v in vars(args).items() if k in _LAUNCHER_KEYS}
launcher = AppLauncher(_launcher_args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.controllers import DifferentialIKControllerCfg  # noqa: E402
from isaaclab.envs.mdp.actions import (  # noqa: E402
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
import isaaclab.sim as sim_utils  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.robot_motion.lula")
enable_extension("isaacsim.robot_motion.motion_generation")

from isaacsim.robot_motion.motion_generation.lula.motion_policies import RmpFlow  # noqa: E402
from isaacsim.robot_motion.motion_generation.lula.kinematics import LulaKinematicsSolver  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_inv  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402

if (args.dataset_dir is not None and not args.no_videos) or args.review_video_dir is not None:
    import imageio.v2 as imageio  # noqa: E402

import sim_to_real  # noqa: E402,F401
from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import (  # noqa: E402
    BOWL_HEIGHT_RANGE,
    BOWL_NAME,
    BOWL_SUCCESS_RADIUS,
    add_pick_cube_cameras,
    CUBE_NAMES,
    SO101_JOINT_ORDER,
    apply_curriculum,
)
from sim_to_real.tasks.pick_cube import mdp as task_mdp  # noqa: E402
from sim_to_real.tasks.pick_pen.pick_pen_env_cfg import (  # noqa: E402
    _look_at_quat_world,
    _pinhole_camera_cfg,
)
from sim_to_real.utils.gripper_effort import dynamic_reset_gripper_effort_limit_sim  # noqa: E402


ARM_DOF = 5
# 큐브 데스크 상판 z. robot base z=0.6749=desk_top-0.0301 → desk_top≈0.705 (펜 데스크의
# 0.76 과 다르다). bowl/cube z 는 reset 후 randomize 로 흔들리므로, 절대 상수 대신 가능하면
# live root_pos_w 를 쓴다. 이 상수는 fallback·진단용.
CUBE_DESK_TOP_Z = 0.705
# (legacy fallback) 구 균일 2.5cm 큐브의 half-height. 현재 씬은 30/40mm 2종이라
# 아래 CUBE_HALF_HEIGHTS 를 우선 사용한다. surface = cube_center_z - half.
CUBE_HALF_Z = 0.0125
# author_pick_cube_scene.py::CUBE_SCALES 와 동기 — Cube1/2=30mm, Cube3/4=40mm.
CUBE_HALF_HEIGHTS = {
    "Cube1": 0.015,
    "Cube2": 0.015,
    "Cube3": 0.020,
    "Cube4": 0.020,
}
# 40mm 큐브는 기본 개방(1.0 rad)/descend 개방(0.5)으로는 폭이 빠듯하다. pick 관련
# 단계에서만 큐브별로 더 벌린다(release/idle 은 args.gripper_open 유지). joint 상한 1.745.
PICK_GRIPPER_OPEN_BY_CUBE = {"Cube3": 1.4, "Cube4": 1.4}
DESCEND_GRIPPER_BY_CUBE = {"Cube3": 0.8, "Cube4": 0.8}
JAW_GRASP_OFFSET = (-0.021, -0.070, 0.020)
CONTROL_POINT_NAME = "gripper_jaw_midpoint"
FPS = 30
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
IMAGE_CHANNELS = 3
CAMERA_KEYS = ("top", "wrist")
CAMERA_SCENE_NAMES = {
    "top": "top_camera",
    "wrist": "wrist_camera",
}
BOWL_PLACE_OFFSET_DIRECTIONS = (
    (-1.0, -1.0),
    (1.0, -1.0),
    (-1.0, 1.0),
    (1.0, 1.0),
)
CUBE_TASK_NAME = "pick up the cube and place it in the bowl"
JOINT_FEATURE_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]
GRIPPER_LEROBOT_SCALE = 31.75
RMPFLOW_DIR = Path("assets/robots/rmpflow").resolve()
RMPFLOW_URDF_PATH = Path("assets/robots/urdf/so_arm101.urdf").resolve()
RMPFLOW_DESCRIPTOR_PATH = RMPFLOW_DIR / "so101_robot_description.yaml"
RMPFLOW_CONFIG_PATH = RMPFLOW_DIR / "so101_rmpflow_config.yaml"
GRIPPER_FRAME_OFFSET = (-0.0079, -0.000218121, -0.0981274)
# Lula solves FK in the URDF-local SO-101 frame, while the Isaac Lab USD
# articulation is authored under the cube_desk scene transform.  These values
# are the least-squares base pose that maps Lula gripper_frame_link FK to the
# USD gripper_frame_w diagnostics from successful direct-FSM poses.
RMPFLOW_BASE_POS_USD = (1.81791970, -0.58952723, 0.70832908)
RMPFLOW_BASE_QUAT_USD = (0.71116823, -0.00950808, 0.01529776, 0.70279110)
RMPFLOW_GRIPPER_FRAME_TARGET_OFFSET = (-0.078, 0.010, -0.002)

# Top-down(palm-down) end-effector orientation in world frame.
# ikpy FK 캘리브레이션(성공 grasp config)에서 gripper_frame_link 의 local Z 축이 이미
# 거의 world −Z(아래) 를 가리킨다(dot≈0.989). 따라서 top-down pick 과 level drop 은
# "local Z 축 → world −Z" 한 축만 제약하면 된다(5-DOF 로 도달 가능, 자연 해와 일치).
# 회전행렬 열 = [X, Y, Z]. orientation_mode="Z" 는 Z 열만 사용한다(roll 자유).
TOPDOWN_R_WORLD = (
    (1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, -1.0),
)


class PickCubeFSMState(str, Enum):
    IDLE = "IDLE"
    OPEN_GRIPPER = "OPEN_GRIPPER"
    MOVE_TO_PRE_PICK = "MOVE_TO_PRE_PICK"
    ORIENT_WRIST = "ORIENT_WRIST"
    DESCEND = "DESCEND"
    GRASP = "GRASP"
    LIFT = "LIFT"
    MOVE_TO_PRE_PLACE = "MOVE_TO_PRE_PLACE"
    PLACE_DESCEND = "PLACE_DESCEND"
    RELEASE = "RELEASE"
    MARK_DONE = "MARK_DONE"
    ALL_DONE = "ALL_DONE"


PICK_CUBE_FSM_SEQUENCE = tuple(state.value for state in PickCubeFSMState)


@configclass
class PickCubeDiffIkActionsCfg:
    """State-machine-only action surface: 3D task-space IK + binary gripper."""

    arm: DifferentialInverseKinematicsActionCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=SO101_JOINT_ORDER[:ARM_DOF],
        body_name="jaw",
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=JAW_GRASP_OFFSET,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        scale=1.0,
        controller=DifferentialIKControllerCfg(
            command_type="position",
            use_relative_mode=True,
            ik_method="dls",
            ik_params={"lambda_val": 0.04},
        ),
    )
    gripper: BinaryJointPositionActionCfg = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper"],
        open_command_expr={"gripper": args.gripper_open},
        close_command_expr={"gripper": args.ik_gripper_closed},
    )


@dataclass
class ImageStats:
    count: int = 0
    channel_min: np.ndarray = field(default_factory=lambda: np.full(3, 1.0, dtype=np.float64))
    channel_max: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    channel_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    channel_sumsq: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def update(self, image_u8: np.ndarray) -> None:
        values = image_u8.astype(np.float64) / 255.0
        flat = values.reshape(-1, 3)
        self.count += flat.shape[0]
        self.channel_min = np.minimum(self.channel_min, flat.min(axis=0))
        self.channel_max = np.maximum(self.channel_max, flat.max(axis=0))
        self.channel_sum += flat.sum(axis=0)
        self.channel_sumsq += np.square(flat).sum(axis=0)

    def to_json(self) -> dict[str, Any]:
        if self.count <= 0:
            mean = np.full(3, 0.5, dtype=np.float64)
            std = np.zeros(3, dtype=np.float64)
            min_v = np.zeros(3, dtype=np.float64)
            max_v = np.ones(3, dtype=np.float64)
        else:
            mean = self.channel_sum / self.count
            var = np.maximum(self.channel_sumsq / self.count - np.square(mean), 0.0)
            std = np.sqrt(var)
            min_v = self.channel_min
            max_v = self.channel_max

        def nested(values: np.ndarray) -> list[list[list[float]]]:
            return [[[float(v)]] for v in values.tolist()]

        return {
            "min": nested(min_v),
            "max": nested(max_v),
            "mean": nested(mean),
            "std": nested(std),
            "count": [int(self.count)],
            "q01": nested(min_v),
            "q10": nested(mean),
            "q50": nested(mean),
            "q90": nested(mean),
            "q99": nested(max_v),
        }


class LeRobotV3EpisodeRecorder:
    """Single-episode LeRobot v3 writer for scripted state-machine traces."""

    def __init__(self, root: Path, *, seconds: float, overwrite: bool, videos: bool) -> None:
        self.root = root.resolve()
        self.max_frames = max(1, int(round(seconds * FPS)))
        self.videos = videos
        self.rows: list[dict[str, Any]] = []
        self.image_stats = {cam: ImageStats() for cam in CAMERA_KEYS}
        self.writers: dict[str, Any] = {}
        self._prepare_output_dir(overwrite)
        if self.videos:
            self._open_video_writers()

    @property
    def frame_count(self) -> int:
        return len(self.rows)

    @property
    def done(self) -> bool:
        return self.frame_count >= self.max_frames

    def record(self, env, action: torch.Tensor) -> None:
        if self.done:
            return
        frame_idx = self.frame_count
        self.rows.append(
            {
                "action": _action_to_record(env, action).tolist(),
                "observation.state": _read_joint_state(env).tolist(),
                "timestamp": frame_idx / FPS,
                "frame_index": frame_idx,
                "episode_index": 0,
                "index": frame_idx,
                "task_index": 0,
            }
        )
        if self.videos:
            for cam, image in _capture_images(env).items():
                self.writers[cam].append_data(image)
                self.image_stats[cam].update(image)

    def finalize(self, *, task_name: str, run_result: dict[str, Any]) -> dict[str, Any]:
        self.close()
        if not self.rows:
            raise RuntimeError("No frames recorded for LeRobot dataset")
        self._write_data_parquet()
        self._write_tasks(task_name)
        self._write_episodes(task_name)
        self._write_info(task_name)
        self._write_stats()
        meta = {
            "task_id": "TA.CUBE.STATE_MACHINE.DATASET",
            "status": "passed" if run_result.get("status") == "passed" else "failed",
            "output_dir": str(self.root),
            "frames": self.frame_count,
            "seconds": self.frame_count / FPS,
            "fps": FPS,
            "videos": self.videos,
            "state_machine_status": run_result.get("status"),
            "placed_and_released": run_result.get("placed_and_released"),
        }
        (self.root / "meta" / "state_machine_result.json").write_text(
            json.dumps({"dataset": meta, "run": run_result}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return meta

    def close(self) -> None:
        for writer in self.writers.values():
            try:
                writer.close()
            except Exception:
                pass
        self.writers = {}

    def _prepare_output_dir(self, overwrite: bool) -> None:
        if self.root.exists():
            if not overwrite:
                raise FileExistsError(f"dataset_dir already exists: {self.root}")
            unsafe_targets = {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve()}
            if self.root in unsafe_targets or len(self.root.parts) < 4:
                raise ValueError(f"Refusing to remove unsafe dataset_dir: {self.root}")
            shutil.rmtree(self.root)

        (self.root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        (self.root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
        if self.videos:
            for cam in CAMERA_KEYS:
                (self.root / "videos" / f"observation.images.{cam}" / "chunk-000").mkdir(parents=True, exist_ok=True)

    def _open_video_writers(self) -> None:
        for cam in CAMERA_KEYS:
            path = self.root / "videos" / f"observation.images.{cam}" / "chunk-000" / "file-000.mp4"
            self.writers[cam] = imageio.get_writer(
                path,
                fps=FPS,
                codec="libx264",
                quality=8,
                macro_block_size=1,
                ffmpeg_params=["-pix_fmt", "yuv420p"],
            )

    def _write_data_parquet(self) -> None:
        fsl6 = pa.list_(pa.float32(), 6)
        table = pa.table(
            {
                "action": pa.array([r["action"] for r in self.rows], type=fsl6),
                "observation.state": pa.array([r["observation.state"] for r in self.rows], type=fsl6),
                "timestamp": pa.array([r["timestamp"] for r in self.rows], type=pa.float32()),
                "frame_index": pa.array([r["frame_index"] for r in self.rows], type=pa.int64()),
                "episode_index": pa.array([r["episode_index"] for r in self.rows], type=pa.int64()),
                "index": pa.array([r["index"] for r in self.rows], type=pa.int64()),
                "task_index": pa.array([r["task_index"] for r in self.rows], type=pa.int64()),
            }
        )
        pq.write_table(table, self.root / "data" / "chunk-000" / "file-000.parquet")

    def _write_tasks(self, task_name: str) -> None:
        table = pa.table({"task_index": [0], "__index_level_0__": [task_name]})
        pq.write_table(table, self.root / "meta" / "tasks.parquet")

    def _write_episodes(self, task_name: str) -> None:
        length = self.frame_count
        meta: dict[str, Any] = {
            "episode_index": 0,
            "tasks": [task_name],
            "length": length,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": 0,
            "dataset_to_index": length,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        if self.videos:
            for cam in CAMERA_KEYS:
                meta[f"videos/observation.images.{cam}/chunk_index"] = 0
                meta[f"videos/observation.images.{cam}/file_index"] = 0
                meta[f"videos/observation.images.{cam}/from_timestamp"] = 0.0
                meta[f"videos/observation.images.{cam}/to_timestamp"] = length / FPS

        arrays: dict[str, Any] = {
            "episode_index": pa.array([meta["episode_index"]], type=pa.int64()),
            "tasks": pa.array([meta["tasks"]], type=pa.list_(pa.string())),
            "length": pa.array([meta["length"]], type=pa.int64()),
            "data/chunk_index": pa.array([meta["data/chunk_index"]], type=pa.int64()),
            "data/file_index": pa.array([meta["data/file_index"]], type=pa.int64()),
            "dataset_from_index": pa.array([meta["dataset_from_index"]], type=pa.int64()),
            "dataset_to_index": pa.array([meta["dataset_to_index"]], type=pa.int64()),
            "meta/episodes/chunk_index": pa.array([meta["meta/episodes/chunk_index"]], type=pa.int64()),
            "meta/episodes/file_index": pa.array([meta["meta/episodes/file_index"]], type=pa.int64()),
        }
        if self.videos:
            for cam in CAMERA_KEYS:
                for name in ("chunk_index", "file_index"):
                    key = f"videos/observation.images.{cam}/{name}"
                    arrays[key] = pa.array([meta[key]], type=pa.int64())
                for name in ("from_timestamp", "to_timestamp"):
                    key = f"videos/observation.images.{cam}/{name}"
                    arrays[key] = pa.array([meta[key]], type=pa.float64())
        pq.write_table(pa.table(arrays), self.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    def _write_info(self, task_name: str) -> None:
        video_info = {
            "video.height": IMAGE_HEIGHT,
            "video.width": IMAGE_WIDTH,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": FPS,
            "video.channels": IMAGE_CHANNELS,
            "has_audio": False,
        }
        features: dict[str, Any] = {
            "action": {"dtype": "float32", "names": JOINT_FEATURE_NAMES, "shape": [6]},
            "observation.state": {"dtype": "float32", "names": JOINT_FEATURE_NAMES, "shape": [6]},
        }
        if self.videos:
            for cam in CAMERA_KEYS:
                features[f"observation.images.{cam}"] = {
                    "dtype": "video",
                    "shape": [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS],
                    "names": ["height", "width", "channels"],
                    "info": video_info,
                }
        features.update(
            {
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
            }
        )
        info = {
            "codebase_version": "v3.0",
            "robot_type": "so_follower",
            "total_episodes": 1,
            "total_frames": self.frame_count,
            "total_tasks": 1,
            "chunks_size": 1000,
            "data_files_size_in_mb": 100,
            "video_files_size_in_mb": 200,
            "fps": FPS,
            "splits": {"train": "0:1"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": features,
            "task": task_name,
        }
        (self.root / "meta" / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_stats(self) -> None:
        stats = {
            "action": _numeric_stats([r["action"] for r in self.rows]),
            "observation.state": _numeric_stats([r["observation.state"] for r in self.rows]),
            "timestamp": _numeric_stats([r["timestamp"] for r in self.rows]),
            "frame_index": _numeric_stats([r["frame_index"] for r in self.rows]),
            "episode_index": _numeric_stats([r["episode_index"] for r in self.rows]),
            "index": _numeric_stats([r["index"] for r in self.rows]),
            "task_index": _numeric_stats([r["task_index"] for r in self.rows]),
        }
        if self.videos:
            for cam in CAMERA_KEYS:
                stats[f"observation.images.{cam}"] = self.image_stats[cam].to_json()
        (self.root / "meta" / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


class ExpertTrajectoryRecorder:
    """Step-pre expert pairs for BC warm-start.

    LeRobot dataset rows intentionally store post-step observations. BC needs
    state_t -> action_t, so this recorder is called immediately before
    env.step(action).
    """

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.obs: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.phases: list[str] = []

    def record(self, env, action: torch.Tensor, phase: str) -> None:
        obs = task_mdp.rl_state(
            env.unwrapped,
            object_names=CUBE_NAMES,
            container_name=BOWL_NAME,
        )
        self.obs.append(obs[0].detach().cpu().to(torch.float32))
        self.actions.append(action[0, :6].detach().cpu().to(torch.float32))
        self.phases.append(phase)

    def finalize(self, *, run_result: dict[str, Any]) -> dict[str, Any]:
        if self.obs:
            obs = torch.stack(self.obs, dim=0)
            actions = torch.stack(self.actions, dim=0)
        else:
            obs = torch.empty((0, 43), dtype=torch.float32)
            actions = torch.empty((0, 6), dtype=torch.float32)

        meta = {
            "task_id": "TA.CUBE.STATE_MACHINE.EXPERT",
            "status": "passed" if run_result.get("placed_and_released") else "failed",
            "task": run_result.get("task"),
            "frames": int(obs.shape[0]),
            "active_objects": run_result.get("active_objects"),
            "object_radius_scale": run_result.get("object_radius_scale"),
            "container_angle_scale": run_result.get("container_angle_scale"),
            "container_radius_scale": run_result.get("container_radius_scale"),
            "placed_and_released": run_result.get("placed_and_released"),
            "final_inside": run_result.get("final_inside"),
            "controller": run_result.get("controller"),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "obs": obs,
                "actions": actions,
                "phases": self.phases,
                "meta": meta,
            },
            self.path,
        )
        return {
            "task_id": "TA.CUBE.STATE_MACHINE.EXPERT",
            "status": meta["status"],
            "path": str(self.path),
            "frames": int(obs.shape[0]),
            "obs_shape": list(obs.shape),
            "action_shape": list(actions.shape),
        }


class SO101RmpFlowJointTarget:
    """Thin RMPFlow wrapper that returns SO-101 arm joint targets.

    Isaac Lab's built-in RMPFlowAction always passes an orientation target.
    SO-101 has only 5 arm DOFs, so for this pick-place controller we use the
    lower-level RmpFlow API directly and send position-only targets.
    """

    def __init__(self, env, device: str) -> None:
        self.env = env
        self.device = device
        self.robot = env.unwrapped.scene["robot"]
        physics_dt = sim_utils.SimulationContext.instance().get_physics_dt()
        self._frame_dt = float(getattr(env.unwrapped, "step_dt", physics_dt))
        self._rmpflow = RmpFlow(
            robot_description_path=str(RMPFLOW_DESCRIPTOR_PATH),
            urdf_path=str(RMPFLOW_URDF_PATH),
            rmpflow_config_path=str(RMPFLOW_CONFIG_PATH),
            end_effector_frame_name="gripper_frame_link",
            maximum_substep_size=physics_dt / 5.0,
        )
        self._kinematics = self._rmpflow.get_kinematics_solver()
        self.active_dof_names = list(self._rmpflow.get_active_joints())

    def reset(self) -> None:
        self._rmpflow.reset()
        self._sync_base_pose()

    def _sync_base_pose(self) -> None:
        base_pos = np.asarray(RMPFLOW_BASE_POS_USD, dtype=np.float32)
        base_quat = np.asarray(RMPFLOW_BASE_QUAT_USD, dtype=np.float32)
        self._rmpflow.set_robot_base_pose(robot_position=base_pos, robot_orientation=base_quat)
        self._kinematics.set_robot_base_pose(robot_position=base_pos, robot_orientation=base_quat)

    def compute(
        self, target_w: torch.Tensor, target_R_w: np.ndarray | None = None
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # rmpflow 경로는 position-only. target_R_w 는 무시.
        self._sync_base_pose()
        target_usd = target_w.detach().cpu().reshape(3).numpy()
        target_usd = target_usd + np.asarray(RMPFLOW_GRIPPER_FRAME_TARGET_OFFSET, dtype=np.float32)
        warm_start = self.robot.data.joint_pos[0, :ARM_DOF].detach().cpu().numpy().astype(np.float64)
        ik_q, ik_success = self._kinematics.compute_inverse_kinematics(
            "gripper_frame_link",
            target_usd.astype(np.float64),
            target_orientation=None,
            warm_start=warm_start,
            position_tolerance=0.005,
        )
        if ik_success:
            self._rmpflow.set_cspace_target(np.asarray(ik_q, dtype=np.float64))
        self._rmpflow.set_end_effector_target(target_position=target_usd, target_orientation=None)
        current_q = self.robot.data.joint_pos[0, :ARM_DOF].detach().cpu().numpy().astype(np.float64)
        current_v = self.robot.data.joint_vel[0, :ARM_DOF].detach().cpu().numpy().astype(np.float64)
        q_next = current_q
        v_next = current_v
        for _ in range(max(1, args.rmpflow_internal_rollout_steps)):
            q_next, v_next = self._rmpflow.compute_joint_targets(
                q_next,
                v_next,
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                self._frame_dt,
            )
        q = torch.as_tensor(q_next[:ARM_DOF], device=self.device, dtype=torch.float32)
        return q, {
            "rmpflow_active_dof_names": self.active_dof_names,
            "rmpflow_joint_target": _round_list(q),
            "rmpflow_frame_name": "gripper_frame_link",
            "rmpflow_position_only": True,
            "rmpflow_target_usd": [round(float(v), 5) for v in target_usd.tolist()],
            "rmpflow_ik_success": bool(ik_success),
            "rmpflow_ik_cspace_target": [round(float(v), 5) for v in np.asarray(ik_q).tolist()],
            "rmpflow_base_pos_usd": [round(float(v), 5) for v in RMPFLOW_BASE_POS_USD],
            "rmpflow_base_quat_usd": [round(float(v), 5) for v in RMPFLOW_BASE_QUAT_USD],
            "rmpflow_frame_dt": round(float(self._frame_dt), 6),
            "rmpflow_internal_rollout_steps": int(args.rmpflow_internal_rollout_steps),
        }


def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """wxyz quaternion → 3x3 회전행렬 (base→world 변환용)."""
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class SO101LulaIkJointTarget:
    """경로 2 — Lula ``LulaKinematicsSolver`` position-only IK (RmpFlow 미경유).

    RmpFlow driver 와 동일한 ``compute(target_w) -> (q, plan)`` 인터페이스. base pose 와
    grasp-frame offset 은 RmpFlow 가 쓰는 검증 상수(``RMPFLOW_BASE_POS/QUAT_USD``,
    ``RMPFLOW_GRIPPER_FRAME_TARGET_OFFSET``)를 그대로 재사용한다.

    Lula IK 는 local 솔버라 warm_start 에서 먼 target 은 한 번에 못 푼다 → 현재 EE 에서
    target 까지 ``max_step`` 간격으로 보간하며 warm-start 를 체이닝한다.
    """

    def __init__(self, env, device: str, *, max_step: float = 0.04, tolerance: float = 0.005) -> None:
        self.env = env
        self.device = device
        self.robot = env.unwrapped.scene["robot"]
        self._max_step = float(max_step)
        self._tol = float(tolerance)
        self._kin = LulaKinematicsSolver(
            robot_description_path=str(RMPFLOW_DESCRIPTOR_PATH),
            urdf_path=str(RMPFLOW_URDF_PATH),
        )
        self._sync_base_pose()

    def reset(self) -> None:
        self._sync_base_pose()

    def _sync_base_pose(self) -> None:
        self._kin.set_robot_base_pose(
            np.asarray(RMPFLOW_BASE_POS_USD, dtype=np.float32),
            np.asarray(RMPFLOW_BASE_QUAT_USD, dtype=np.float32),
        )

    def compute(
        self, target_w: torch.Tensor, target_R_w: np.ndarray | None = None
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # lula 경로는 position-only(5-DOF full orientation 미보장). target_R_w 는 무시.
        self._sync_base_pose()
        target_usd = target_w.detach().cpu().reshape(3).numpy().astype(np.float64) + np.asarray(
            RMPFLOW_GRIPPER_FRAME_TARGET_OFFSET, dtype=np.float64
        )
        q = self.robot.data.joint_pos[0, :ARM_DOF].detach().cpu().numpy().astype(np.float64)
        start, _rot = self._kin.compute_forward_kinematics("gripper_frame_link", q)
        start = np.asarray(start, dtype=np.float64)
        dist = float(np.linalg.norm(target_usd - start))
        n = max(1, int(math.ceil(dist / max(self._max_step, 1e-3))))
        ok = False
        for i in range(1, n + 1):
            sub = start + (target_usd - start) * (float(i) / n)
            # compute_inverse_kinematics 는 (joint_positions: np.array, success: bool) 직접 반환.
            q_sol, ok = self._kin.compute_inverse_kinematics(
                "gripper_frame_link",
                sub,
                target_orientation=None,
                warm_start=q,
                position_tolerance=self._tol,
            )
            if q_sol is not None:
                q = np.asarray(q_sol, dtype=np.float64)[:ARM_DOF]
        ach, _r = self._kin.compute_forward_kinematics("gripper_frame_link", q)
        err = float(np.linalg.norm(np.asarray(ach, dtype=np.float64) - target_usd))
        q_t = torch.as_tensor(q[:ARM_DOF], device=self.device, dtype=torch.float32)
        return q_t, {
            "lula_ik_joint_target": _round_list(q_t),
            "lula_ik_success": bool(ok) and err <= self._tol * 3.0,
            "lula_ik_error_m": round(err, 5),
            "lula_ik_target_usd": [round(float(v), 5) for v in target_usd.tolist()],
            "lula_ik_frame_name": "gripper_frame_link",
        }


class SO101IkpyJointTarget:
    """경로 3 — ikpy ``Chain`` position-only IK (URDF 만).

    ikpy 는 base pose setter 가 없어, RmpFlow 와 동일한 base pose 상수로 world→base 변환
    후 풀고 다시 arm joint 만 반환한다. scipy least_squares 라 먼 target 도 견고.
    """

    def __init__(self, env, device: str, *, tolerance: float = 0.005) -> None:
        from ikpy.chain import Chain  # lazy: ikpy 는 순수 python

        self.env = env
        self.device = device
        self.robot = env.unwrapped.scene["robot"]
        self._tol = float(tolerance)
        self._mask = [False] + [True] * ARM_DOF + [False]
        self._chain = Chain.from_urdf_file(
            str(RMPFLOW_URDF_PATH), base_elements=["base_link"], active_links_mask=self._mask
        )
        self._n = len(self._chain.links)
        self._base_pos = np.asarray(RMPFLOW_BASE_POS_USD, dtype=np.float64)
        self._base_R = _quat_wxyz_to_matrix(np.asarray(RMPFLOW_BASE_QUAT_USD, dtype=np.float64))
        # arm joint 한계 (URDF 순서) — seed clamp 용.
        self._lo = np.array([-1.91986, -1.74533, -1.69, -1.65806, -2.74385], dtype=np.float64)
        self._hi = np.array([1.91986, 1.74533, 1.69, 1.65806, 2.84121], dtype=np.float64)

    def reset(self) -> None:  # ikpy 는 상태 없음
        pass

    def _full_q(self, q_arm: np.ndarray) -> np.ndarray:
        full = np.zeros(self._n, dtype=np.float64)
        full[1 : 1 + ARM_DOF] = q_arm
        return full

    def compute(
        self, target_w: torch.Tensor, target_R_w: np.ndarray | None = None
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        target_usd = target_w.detach().cpu().reshape(3).numpy().astype(np.float64) + np.asarray(
            RMPFLOW_GRIPPER_FRAME_TARGET_OFFSET, dtype=np.float64
        )
        target_base = self._base_R.T @ (target_usd - self._base_pos)
        warm = self.robot.data.joint_pos[0, :ARM_DOF].detach().cpu().numpy().astype(np.float64)
        # least_squares 는 seed 가 bound 를 부동소수점만큼 넘으면 거부 → 안쪽 clamp.
        warm = np.minimum(np.maximum(warm, self._lo + 1e-4), self._hi - 1e-4)
        if target_R_w is not None:
            # world 목표 회전 → base frame. ikpy 는 single-axis mode(X/Y/Z)에서
            # target_orientation 으로 "그 축의 방향 3-벡터"를 기대한다(3x3 아님). "all" 만 3x3.
            target_R_base = self._base_R.T @ np.asarray(target_R_w, dtype=np.float64)
            mode = args.ikpy_orientation_mode
            if mode == "all":
                ori_arg = target_R_base
            else:
                ori_arg = target_R_base[:, {"X": 0, "Y": 1, "Z": 2}[mode]]
            sol = self._chain.inverse_kinematics(
                target_position=target_base,
                target_orientation=ori_arg,
                orientation_mode=mode,
                initial_position=self._full_q(warm),
            )
        else:
            sol = self._chain.inverse_kinematics(
                target_position=target_base, orientation_mode=None, initial_position=self._full_q(warm)
            )
        q = np.minimum(np.maximum(np.asarray(sol[1 : 1 + ARM_DOF], dtype=np.float64), self._lo), self._hi)
        T_ach = self._chain.forward_kinematics(self._full_q(q))
        ach = np.asarray(T_ach[:3, 3], dtype=np.float64)
        err = float(np.linalg.norm(ach - target_base))
        # 달성한 approach 축(local Z) 의 world 방향 — top-down 진단용.
        z_axis_world = self._base_R @ np.asarray(T_ach[:3, 2], dtype=np.float64)
        q_t = torch.as_tensor(q, device=self.device, dtype=torch.float32)
        return q_t, {
            "ikpy_joint_target": _round_list(q_t),
            "ikpy_success": err <= self._tol,
            "ikpy_error_m": round(err, 5),
            "ikpy_target_usd": [round(float(v), 5) for v in target_usd.tolist()],
            "ikpy_frame_name": "gripper_frame_link",
            "ikpy_topdown": target_R_w is not None,
            "ikpy_approach_z_world": [round(float(v), 4) for v in z_axis_world.tolist()],
            "ikpy_approach_down_dot": round(float(z_axis_world @ np.array([0.0, 0.0, -1.0])), 4),
        }


def _make_cartesian_driver(env, device: str):
    """controller_mode 에 맞는 phase driver 를 만든다. joint_fk/diff_ik → None."""
    mode = args.controller_mode
    if mode == "rmpflow":
        return SO101RmpFlowJointTarget(env, device)
    if mode == "lula_ik":
        return SO101LulaIkJointTarget(env, device)
    if mode == "ikpy":
        return SO101IkpyJointTarget(env, device)
    return None


def _round_list(values: torch.Tensor, digits: int = 5) -> list[float]:
    return [round(float(v), digits) for v in values.detach().cpu().flatten().tolist()]


def _append_fsm_event(
    fsm_trace: list[dict[str, Any]],
    state: PickCubeFSMState,
    *,
    cube_name: str | None = None,
    attempt: int | None = None,
    next_state: PickCubeFSMState | None = None,
    reason: str | None = None,
    **fields: Any,
) -> None:
    event: dict[str, Any] = {
        "index": len(fsm_trace),
        "state": state.value,
    }
    if cube_name is not None:
        event["cube"] = cube_name
    if attempt is not None:
        event["attempt"] = attempt
    if next_state is not None:
        event["next_state"] = next_state.value
    if reason is not None:
        event["reason"] = reason
    event.update(fields)
    fsm_trace.append(event)


def _to_lerobot_units(values_rad: np.ndarray) -> np.ndarray:
    """Convert Isaac joint radians to the real LeRobot SO-101 convention."""

    values = np.asarray(values_rad, dtype=np.float32).copy()
    values[:ARM_DOF] = values[:ARM_DOF] * (180.0 / math.pi)
    values[5] = values[5] * GRIPPER_LEROBOT_SCALE
    return values.astype(np.float32)


def _read_joint_state(env) -> np.ndarray:
    robot = env.unwrapped.scene["robot"]
    return _to_lerobot_units(robot.data.joint_pos[0, :6].detach().cpu().numpy())


def _action_to_record(env, action_tensor: torch.Tensor) -> np.ndarray:
    if args.controller_mode == "diff_ik" or action_tensor.shape[-1] != 6:
        robot = env.unwrapped.scene["robot"]
        return _to_lerobot_units(robot.data.joint_pos[0, :6].detach().cpu().numpy())
    action = action_tensor[0, :6].detach().cpu().numpy()
    return _to_lerobot_units(action)


def _camera_rgb_u8(cam) -> np.ndarray:
    rgb = cam.data.output["rgb"][0].detach().cpu().numpy()
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        if np.issubdtype(rgb.dtype, np.floating):
            rgb = np.clip(rgb, 0.0, 1.0) * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb)


# 리뷰 영상 writer (열려 있으면 _step_env 가 매 step 뷰어 카메라 프레임을 append).
_REVIEW_WRITER: Any | None = None

# sim step 카운터 — 에피소드 소요시간(초 = steps / FPS) 측정용.
_STEP_COUNTER: int = 0


def _review_capture(env) -> None:
    if _REVIEW_WRITER is None:
        return
    cam = env.unwrapped.scene["review_camera"]
    _REVIEW_WRITER.append_data(_camera_rgb_u8(cam))


def _capture_images(env) -> dict[str, np.ndarray]:
    images: dict[str, np.ndarray] = {}
    for key in CAMERA_KEYS:
        cam = env.unwrapped.scene[CAMERA_SCENE_NAMES[key]]
        rgb = cam.data.output["rgb"][0].detach().cpu().numpy()
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        if rgb.dtype != np.uint8:
            if np.issubdtype(rgb.dtype, np.floating):
                rgb = np.clip(rgb, 0.0, 1.0) * 255.0
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        image = np.ascontiguousarray(rgb)
        expected_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
        if image.shape != expected_shape:
            raise ValueError(f"{key} image shape {image.shape}, expected {expected_shape}")
        images[key] = image
    return images


def _numeric_stats(array_like: list[Any] | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(array_like)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[0] == 0:
        width = arr.shape[1] if arr.ndim > 1 else 1
        zeros = [0.0 for _ in range(width)]
        return {k: zeros for k in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")} | {
            "count": [0]
        }
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [int(arr.shape[0])],
        "q01": np.quantile(arr, 0.01, axis=0).tolist(),
        "q10": np.quantile(arr, 0.10, axis=0).tolist(),
        "q50": np.quantile(arr, 0.50, axis=0).tolist(),
        "q90": np.quantile(arr, 0.90, axis=0).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).tolist(),
    }


def _body_pos(robot, body_name: str) -> torch.Tensor:
    body_id = robot.data.body_names.index(body_name)
    return robot.data.body_pos_w[:, body_id, :]


def _body_quat(robot, body_name: str) -> torch.Tensor:
    body_id = robot.data.body_names.index(body_name)
    return robot.data.body_quat_w[:, body_id, :]


def _quat_apply_wxyz(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply wxyz quaternion to a vector."""

    q_xyz = quat[:, 1:4]
    q_w = quat[:, 0:1]
    t = 2.0 * torch.cross(q_xyz, vec, dim=-1)
    return vec + q_w * t + torch.cross(q_xyz, t, dim=-1)


# 그리퍼 finger collision 메시 bbox(메시 로컬, m) + URDF collision origin(xyz, rpy).
# STL 측정값. "gripper"=고정 finger(wrist_roll_follower), "jaw"=모터 jaw(moving_jaw_so101_v1).
# finger tip 의 world 위치를 측정해 고정 finger 가 큐브 위를 찌르는지/수평 마진을 진단한다.
_FINGER_GEOM = {
    "jaw": dict(
        lo=(-0.0123, -0.082, -0.024),
        hi=(0.01, 0.01, 0.024),
        oxyz=(0.0, 0.0, 0.0189),
        orpy=(0.0, 0.0, 0.0),
    ),
    "gripper": dict(
        lo=(-0.0352, -0.0242, -0.0001),
        hi=(0.03, 0.0278, 0.1054),
        oxyz=(0.0, -0.000218214, 0.000949706),
        orpy=(-3.14159, 0.0, 0.0),
    ),
}


def _rpy_matrix(rpy: tuple[float, float, float], device, dtype) -> torch.Tensor:
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rx = torch.tensor([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], device=device, dtype=dtype)
    ry = torch.tensor([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], device=device, dtype=dtype)
    rz = torch.tensor([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], device=device, dtype=dtype)
    return rz @ ry @ rx


def _finger_world_aabb(robot, body_name: str) -> dict[str, list[float]]:
    """finger collision 메시의 8개 bbox 코너를 world 로 변환해 AABB(min/max) 반환."""
    g = _FINGER_GEOM[body_name]
    dev = robot.data.joint_pos.device
    dtype = torch.float32
    R_coll = _rpy_matrix(g["orpy"], dev, dtype)
    oxyz = torch.tensor(g["oxyz"], device=dev, dtype=dtype)
    lo = g["lo"]
    hi = g["hi"]
    corners = []
    for cx in (lo[0], hi[0]):
        for cy in (lo[1], hi[1]):
            for cz in (lo[2], hi[2]):
                corners.append([cx, cy, cz])
    p_mesh = torch.tensor(corners, device=dev, dtype=dtype)  # (8,3)
    p_link = (R_coll @ p_mesh.T).T + oxyz  # (8,3)
    body_pos = _body_pos(robot, body_name)  # (1,3)
    body_quat = _body_quat(robot, body_name)  # (1,4)
    p_world = body_pos + _quat_apply_wxyz(body_quat.expand(8, 4), p_link)  # (8,3)
    wmin = p_world.min(dim=0).values
    wmax = p_world.max(dim=0).values
    return {"min": _round_list(wmin), "max": _round_list(wmax)}


# finger collision 메시의 link-frame 코너 8개를 캐시(매 FK 샘플마다 재계산 회피).
_FINGER_LINK_CORNERS: dict[str, torch.Tensor] = {}


def _finger_link_corners(body_name: str, device, dtype) -> torch.Tensor:
    key = f"{body_name}:{device}"
    cached = _FINGER_LINK_CORNERS.get(key)
    if cached is None:
        g = _FINGER_GEOM[body_name]
        R_coll = _rpy_matrix(g["orpy"], device, dtype)
        oxyz = torch.tensor(g["oxyz"], device=device, dtype=dtype)
        lo, hi = g["lo"], g["hi"]
        corners = [
            [cx, cy, cz]
            for cx in (lo[0], hi[0])
            for cy in (lo[1], hi[1])
            for cz in (lo[2], hi[2])
        ]
        p_mesh = torch.tensor(corners, device=device, dtype=dtype)  # (8,3)
        cached = (R_coll @ p_mesh.T).T + oxyz  # (8,3) link-frame
        _FINGER_LINK_CORNERS[key] = cached
    return cached


def _finger_world_corners(robot, body_name: str) -> torch.Tensor:
    """finger collision 메시 AABB 8코너의 world 좌표 (8,3)."""
    dev = robot.data.joint_pos.device
    p_link = _finger_link_corners(body_name, dev, torch.float32)
    body_pos = _body_pos(robot, body_name)  # (1,3)
    body_quat = _body_quat(robot, body_name)  # (1,4)
    return body_pos + _quat_apply_wxyz(body_quat.expand(8, 4), p_link)  # (8,3)


def _finger_min_z(robot, body_name: str) -> float:
    """finger collision 메시 AABB 의 world z 최저점(빠른 경로). grasp tilt 점수용."""
    return float(_finger_world_corners(robot, body_name)[:, 2].min().item())


def _jacobian_row(robot, body_name: str) -> torch.Tensor:
    """Return body Jacobian row, shape (1, 6, ARM_DOF).

    PhysX omits the fixed root body from get_jacobians(), so body index N maps
    to Jacobian row N-1 for this fixed-base articulation.
    """

    body_id = robot.data.body_names.index(body_name)
    row = body_id - 1
    if row < 0:
        raise ValueError(f"Body {body_name!r} has no movable-body Jacobian row")
    jac = robot.root_physx_view.get_jacobians()
    return jac[:, row, :6, :ARM_DOF]


def _grasp_point_pos(robot) -> torch.Tensor:
    if args.control_point == "midpoint":
        return 0.5 * (_body_pos(robot, "gripper") + _body_pos(robot, "jaw"))
    offset = torch.tensor(JAW_GRASP_OFFSET, device=robot.data.joint_pos.device, dtype=torch.float32).reshape(1, 3)
    offset_w = _quat_apply_wxyz(_body_quat(robot, "jaw"), offset)
    return _body_pos(robot, "jaw") + offset_w


def _diagnostic_pose(env) -> dict[str, Any]:
    scene = env.unwrapped.scene
    robot = scene["robot"]
    gripper = _body_pos(robot, "gripper")[0]
    jaw = _body_pos(robot, "jaw")[0]
    gripper_frame_offset = torch.tensor(
        GRIPPER_FRAME_OFFSET,
        device=robot.data.joint_pos.device,
        dtype=torch.float32,
    ).reshape(1, 3)
    gripper_frame_w = _body_pos(robot, "gripper") + _quat_apply_wxyz(_body_quat(robot, "gripper"), gripper_frame_offset)
    # 실제 접근축(jaw→grasp point)의 world 방향과 down(−Z) 정렬도 — top-down 검증용.
    approach_w = _approach_axis_world(robot)
    down_dot = float((approach_w @ torch.tensor([0.0, 0.0, -1.0], device=approach_w.device, dtype=approach_w.dtype)).item())
    return {
        "gripper_w": _round_list(gripper),
        "jaw_w": _round_list(jaw),
        "gripper_frame_w": _round_list(gripper_frame_w[0]),
        "gripper_jaw_midpoint_w": _round_list(0.5 * (gripper + jaw)),
        "jaw_approach_axis_w": _round_list(approach_w),
        "jaw_approach_down_dot": round(down_dot, 4),
        "finger_moving_aabb_w": _finger_world_aabb(robot, "jaw"),
        "finger_fixed_aabb_w": _finger_world_aabb(robot, "gripper"),
        "cube_w": {
            name: _round_list(scene[name].data.root_pos_w[0])
            for name in CUBE_NAMES[: args.active_objects]
        },
        "bowl_w": _round_list(scene[BOWL_NAME].data.root_pos_w[0]),
        "bowl_quat_w": _round_list(scene[BOWL_NAME].data.root_quat_w[0]),
    }


def _bowl_tipped(env, max_tilt_deg: float = 45.0) -> bool:
    """그릇이 기울어졌는지(엎힘 감지). 그릇 local z축의 world z 성분으로 판정."""
    quat = env.unwrapped.scene[BOWL_NAME].data.root_quat_w[0:1]
    z_local = torch.tensor([[0.0, 0.0, 1.0]], device=quat.device, dtype=quat.dtype)
    z_w = _quat_apply_wxyz(quat, z_local)[0]
    return float(z_w[2].item()) < math.cos(math.radians(max_tilt_deg))


def _grasp_point_jacobian(robot) -> torch.Tensor:
    if args.control_point == "midpoint":
        return 0.5 * (_jacobian_row(robot, "gripper")[:, :3, :] + _jacobian_row(robot, "jaw")[:, :3, :])

    body_jac = _jacobian_row(robot, "jaw")
    linear = body_jac[:, :3, :]
    angular = body_jac[:, 3:6, :]
    offset = torch.tensor(JAW_GRASP_OFFSET, device=robot.data.joint_pos.device, dtype=torch.float32).reshape(1, 3)
    offset_w = _quat_apply_wxyz(_body_quat(robot, "jaw"), offset)
    point_terms = []
    for joint_id in range(ARM_DOF):
        point_terms.append(linear[:, :, joint_id] + torch.cross(angular[:, :, joint_id], offset_w, dim=-1))
    return torch.stack(point_terms, dim=-1)


def _step_env(env, action: torch.Tensor, *, min_gripper_effort: float | None = None):
    if not args.disable_dynamic_gripper_effort and getattr(env.unwrapped.cfg, "dynamic_reset_gripper_effort_limit", False):
        dynamic_reset_gripper_effort_limit_sim(
            env.unwrapped,
            "so101leader",
            min_effort=args.min_gripper_effort if min_gripper_effort is None else min_gripper_effort,
        )
    global _STEP_COUNTER
    step_out = env.step(action)
    _STEP_COUNTER += 1
    _review_capture(env)
    return step_out


def _arm_limits(robot, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    limits = robot.data.soft_joint_pos_limits[0, :ARM_DOF].to(device=device)
    lower = limits[:, 0]
    upper = limits[:, 1]
    # 일부 USD가 넓은 soft limit을 제공하지 못하는 경우를 위한 보수적 fallback.
    lower = torch.where(torch.isfinite(lower), lower, torch.full_like(lower, -3.14))
    upper = torch.where(torch.isfinite(upper), upper, torch.full_like(upper, 3.14))
    return lower, upper


# 단계별 팔 step 한계 override(grasp 단계만 느리게). None → args.max_arm_step_delta 사용.
_ARM_STEP_OVERRIDE: float | None = None


def _eff_arm_step() -> float:
    return abs(_ARM_STEP_OVERRIDE if _ARM_STEP_OVERRIDE is not None else args.max_arm_step_delta)


# 그리퍼 step 한계 override. 기본(None) = travel(빠름). GRASP(close-on-cube) 단계에서만
# _phase 가 args.max_gripper_step_delta(느림) 로 설정한다.
_GRIPPER_STEP_OVERRIDE: float | None = None


def _eff_gripper_step() -> float:
    return abs(
        _GRIPPER_STEP_OVERRIDE if _GRIPPER_STEP_OVERRIDE is not None else args.gripper_travel_step_delta
    )


def _joint_position_action(
    robot,
    command: torch.Tensor,
    arm_target: torch.Tensor,
    gripper_target: float,
    device: str,
) -> torch.Tensor:
    """Build a slew-limited joint-position action.

    FK waypoints can be far apart. Sending the final target directly makes the
    implicit PD drive snap toward it, so the commanded target itself is moved at
    a bounded per-step rate.
    """

    arm_target = arm_target[:ARM_DOF].to(device=device, dtype=torch.float32)
    lower, upper = _arm_limits(robot, device)

    arm_step = _eff_arm_step()
    arm_delta = torch.clamp(
        arm_target - command[:ARM_DOF],
        -arm_step,
        arm_step,
    )
    gripper_step = _eff_gripper_step()
    gripper_delta = torch.clamp(
        torch.tensor(float(gripper_target), device=device, dtype=torch.float32) - command[5],
        -gripper_step,
        gripper_step,
    )

    command[:ARM_DOF] = torch.minimum(torch.maximum(command[:ARM_DOF] + arm_delta, lower), upper)
    command[5] = command[5] + gripper_delta
    return command.reshape(1, 6).clone()


def _slew_limited_step_count(command: torch.Tensor, q_goal: torch.Tensor, gripper_target: float) -> int:
    arm_delta = float(torch.max(torch.abs(q_goal[:ARM_DOF] - command[:ARM_DOF])).item())
    gripper_delta = abs(float(gripper_target) - float(command[5].item()))
    arm_steps = math.ceil(arm_delta / max(_eff_arm_step(), 1e-6))
    gripper_steps = math.ceil(gripper_delta / max(_eff_gripper_step(), 1e-6))
    return max(arm_steps, gripper_steps) + max(0, args.command_settle_steps)


def _gripper_transition_step_count(current: float, target: float, step: float | None = None) -> int:
    eff = abs(step) if step is not None else _eff_gripper_step()
    return math.ceil(abs(float(target) - float(current)) / max(eff, 1e-6))


def _grasp_phase_steps(command: torch.Tensor, target: float) -> int:
    # GRASP는 "닫기 + 대기" 상태다. slew limit 때문에 열린 상태에서 바로 80 step만
    # 주면 lift 중에야 완전히 닫힐 수 있어, 필요한 닫기 시간과 settle을 보장한다.
    # close-on-cube 는 느린 속도(max_gripper_step_delta)로 계산한다.
    close_steps = _gripper_transition_step_count(
        float(command[5].item()), target, step=args.max_gripper_step_delta
    )
    return max(1, args.close_steps, close_steps + max(0, args.grasp_settle_steps))


def _approach_axis_world(robot) -> torch.Tensor:
    """jaw 원점 → grasp point 방향(접근축)의 world 단위벡터. top-down 이면 ≈ (0,0,-1)."""
    a_local = torch.tensor(JAW_GRASP_OFFSET, device=robot.data.joint_pos.device, dtype=torch.float32)
    a_local = (a_local / torch.linalg.norm(a_local)).reshape(1, 3)
    a_w = _quat_apply_wxyz(_body_quat(robot, "jaw"), a_local)[0]
    return a_w / torch.linalg.norm(a_w)


def _ik_action(
    robot,
    target_grasp_point_w: torch.Tensor,
    gripper_target: float,
    device: str,
    *,
    damping: float,
    gain: float,
    max_joint_delta: float,
    topdown: bool = False,
    ori_weight: float = 0.0,
) -> tuple[torch.Tensor, float]:
    """Isaac-frame damped-least-squares IK on the *actual* grasp point.

    ``topdown`` 이면 접근축(jaw→grasp point)을 world −Z 로 정렬하는 orientation task 를
    위치 task 와 함께 푼다(5-DOF: 위치 3 + 접근축 정렬 2). URDF/offset 불일치가 없어 grasp
    point 가 큐브에 정확히 맞고 수직으로 강하한다."""
    grasp_point = _grasp_point_pos(robot)
    error = (target_grasp_point_w - grasp_point)[0]
    j_pos = _grasp_point_jacobian(robot)[0]  # (3, 5)

    if topdown and ori_weight > 0.0 and args.control_point != "midpoint":
        jaw_jac = _jacobian_row(robot, "jaw")[0]  # (6, 5)
        j_ang = jaw_jac[3:6, :]  # (3, 5)
        a_w = _approach_axis_world(robot)  # (3,)
        a_des = torch.tensor([0.0, 0.0, -1.0], device=device, dtype=a_w.dtype)
        e_ori = a_des - a_w
        # d(a_w)/dq = -[a_w]_x @ J_ang  (body-fixed 단위벡터의 회전).
        ax, ay, az = a_w[0], a_w[1], a_w[2]
        skew_a = torch.stack([
            torch.stack([torch.zeros_like(ax), -az, ay]),
            torch.stack([az, torch.zeros_like(ax), -ax]),
            torch.stack([-ay, ax, torch.zeros_like(ax)]),
        ])
        j_ori = -(skew_a @ j_ang)  # (3, 5)
        j = torch.cat([j_pos, ori_weight * j_ori], dim=0)  # (6, 5)
        err = torch.cat([error, ori_weight * e_ori], dim=0)  # (6,)
        eye = torch.eye(6, device=device, dtype=j.dtype)
    else:
        j = j_pos
        err = error
        eye = torch.eye(3, device=device, dtype=j.dtype)

    lhs = j @ j.transpose(0, 1) + (damping * damping) * eye
    rhs = torch.linalg.solve(lhs, err.unsqueeze(-1)).squeeze(-1)
    dq = j.transpose(0, 1) @ rhs
    dq = torch.clamp(gain * dq, -max_joint_delta, max_joint_delta)

    lower, upper = _arm_limits(robot, device)
    q = robot.data.joint_pos[0, :ARM_DOF]
    q_target = torch.minimum(torch.maximum(q + dq, lower), upper)

    action = torch.zeros((1, 6), device=device, dtype=torch.float32)
    action[0, :ARM_DOF] = q_target
    action[0, 5] = float(gripper_target)
    return action, float(torch.linalg.norm(error).item())


# ── 결정적 waypoint/grasp 솔버 (--grasp_config_mode deterministic) ──────────────
# random-FK 5000샘플의 분산(단일 grasp ~67%)을 제거한다. in-sim 가상 FK(joint state
# 임시 기록 → scene.update → body pose 읽기, random-FK 와 동일 패턴) 위에서
# finite-difference Gauss-Newton(DLS) 을 수렴시킨다 — 난수 없음, 동일 입력 → 동일 출력.
# URDF/base 정합 오차도 없다(실제 USD articulation 의 FK 를 그대로 쓰므로).
# 전부 양수(base 반대쪽 기울임). 음수(base 쪽)는 이동 jaw 기하가 비대칭이라 scoop 이
# 항상 나쁘고(tilt_pen 0.026~0.062 측정) 닫는 중 스윙 사고의 원인이라 제거했다.
# 순서는 중간 tilt 우선 — 30/40mm 큐브는 극단 tilt(65~75°) 없이도 물린다(구 2.5cm 시절의
# 강tilt 선호를 완화. 극단 tilt 는 reach 가장자리에서 실행오차·그릇 충돌 원인).
# 순서 = *저tilt(수직) 우선* — 사용자 텔레옵 시연 사진(docs/pics/cube_desk/그립_1~3)이
# 기준: 거의 수직(tilt 0~15°) 접근 + 두 finger 가 큐브 양 측면을 바닥 근처까지 깊게 감싸
# 큐브가 패드 사이 중앙에 안착한다. 과거 "수직 불가(jaw 가 3cm 위에서 멈춤)" 진단은 옛
# DiffIK 경로의 자세 선택 문제였다. 저tilt 가 막히는 spawn 은 ladder 후순위(중·강tilt)와
# _FAILED_TILT_DEG 블랙리스트가 처리한다. 후보 제거는 금지 — 후보 전멸 시 random_fk
# 폴백(거친 스윙)이 4배로 늘며 4큐브 mean 2.625→1.625 회귀(v13 실측).
_DET_TILT_LADDER_DEG = (15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0)
_DET_FD_EPS = 1.0e-3
_DET_STEP_CLAMP = 0.25
_DET_DAMPING_SQ = 0.03 ** 2
_DET_W_TILT = 0.05   # 접근축(tilt) residual 가중 — 위치(미터 단위)가 항상 우선
_DET_W_ROLL = 0.08   # 개방축(큐브 yaw 정렬) residual 가중 — 0.03 은 너무 약해 모서리(마름모) 잡기 빈발
# 이송(level_axis) 단계의 palm-down 가중. _DET_W_TILT(0.05)를 그대로 쓰면 위치 목표에
# 항상 양보돼 grasp 때의 tilt 그대로 운반한다(사용자 v15 ep0 영상 관찰: "그릇 위로 가져갈
# 때 손목이 수평이 안 됨"). 위치 우선은 유지하되 자세가 실제로 반영될 만큼 올린다.
_DET_W_LEVEL = 0.2
_DET_SCOOP_WEIGHT = 0.25
_DET_CONTINUITY_WEIGHT = 0.02
_DET_ITERS_STAGE1 = 10   # grasp: 위치+tilt 수렴
_DET_ITERS_STAGE2 = 14   # grasp: +개방축 정렬 수렴
_DET_ITERS_POSONLY = 30  # 비grasp waypoint: 위치만

_ATTEMPT_RE = re.compile(r"\.attempt(\d+)\.")


def _attempt_variant_from_phase_name(name: str) -> int:
    m = _ATTEMPT_RE.search(name)
    return int(m.group(1)) - 1 if m else 0


# descend 가 고른 tilt(deg). grasp(닫기) 단계는 이 tilt 로 잠가 재계산한다 — 자유 재계산이
# 반대쪽 tilt 로 갈아타면 닫는 도중 팔이 스윙하며 큐브를 쳐낸다(ep9 진단).
_LAST_DESCEND_TILT_DEG: float | None = None
# descend 가 고른 그리퍼 개방축(world xy 단위벡터). tilt 와 같은 이유로 닫기/보정 단계에
# 잠근다 — 큐브는 90° 대칭이라 개방축 후보가 4개인데, 닫기 단계가 자세 변화 후 *다른*
# 후보를 고르면 wrist_roll 이 ±90~180° 회전하며 닫는 도중 손목이 돌아 그랩이 비틀리거나
# 큐브를 쳐낸다(사용자 영상 관찰: "집을 때 손목 반바퀴 회전").
_LAST_DESCEND_OPEN_AXIS: list[float] | None = None
# 현재 큐브에서 실행이 계통적으로 실패한 tilt 집합. 결정적 솔버는 같은 큐브 pose 에 같은
# tilt 를 재선택하므로(v12 ep16: 3 attempts ferr 4.31cm 가 mm 단위까지 동일) 실패 tilt 를
# 제외해야 재시도가 실제로 다른 시도가 된다. 큐브 시작 시 클리어, descend 실행 미달
# (ferr > 2×descend_tolerance)·skip_close 발동 시 추가.
_FAILED_TILT_DEG: set[float] = set()


def _cube_yaw_w(env, cube_name: str) -> float:
    q = env.unwrapped.scene[cube_name].data.root_quat_w[0]  # wxyz
    w, x, y, z = (float(v) for v in q.tolist())
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _deterministic_solve_joint_target(
    env,
    target_grasp_point_w: torch.Tensor,
    gripper_target: float,
    device: str,
    *,
    grasp_mode: bool,
    pos_gate: float,
    cube_name: str | None = None,
    variant: int = 0,
    floor_z: float = CUBE_DESK_TOP_Z,
    tilt_ladder: tuple[float, ...] | None = None,
    open_axis_override: torch.Tensor | None = None,
    level_axis: bool = False,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    """결정적 joint waypoint 솔버. 수렴 실패(gate 초과) 시 (None, diag) — 호출자가 random-FK 폴백.

    grasp_mode: tilt ladder 의 각 후보 접근축으로 [위치(3) + tilt축(3·약)] DLS → 큐브 yaw 에
    가장 가까운 개방축 후보를 골라 [+개방축(3·약)] 으로 정밀화. 후보 채점 = 위치오차 +
    scoop 패널티(이동 jaw 가 바닥까지·고정 finger 아래 — 검증된 강tilt grasp 조건) + 연속성.
    variant(=attempt-1) 가 ladder 를 회전시켜 재시도마다 다른 tilt 를 우선 탐색한다.
    """
    scene = env.unwrapped.scene
    robot = scene["robot"]
    env_ids = torch.tensor([0], device=device, dtype=torch.long)
    saved_q = robot.data.joint_pos[:, :6].clone()
    saved_v = robot.data.joint_vel[:, :6].clone()
    current_arm = saved_q[0, :ARM_DOF].clone()
    lower, upper = _arm_limits(robot, device)
    lo = lower + 1.0e-3
    hi = upper - 1.0e-3
    target = target_grasp_point_w.to(device=device, dtype=torch.float32).reshape(3)
    zero_vel = torch.zeros((1, 6), device=device)

    def write_q(q_arm: torch.Tensor) -> None:
        q = torch.zeros((1, 6), device=device)
        q[0, :ARM_DOF] = q_arm
        q[0, 5] = float(gripper_target)
        robot.write_joint_state_to_sim(q, zero_vel, env_ids=env_ids)
        scene.update(0.0)

    def residual(
        q_arm: torch.Tensor,
        a_des: torch.Tensor | None,
        o_des: torch.Tensor | None,
        a_weight: float = _DET_W_TILT,
    ) -> torch.Tensor:
        write_q(q_arm)
        parts = [target - _grasp_point_pos(robot)[0]]
        if a_des is not None:
            parts.append(a_weight * (a_des - _approach_axis_world(robot)))
        if o_des is not None:
            parts.append(_DET_W_ROLL * (o_des - _gripper_open_axis_h(robot)))
        return torch.cat(parts)

    def solve(
        q0: torch.Tensor,
        a_des: torch.Tensor | None,
        o_des: torch.Tensor | None,
        iters: int,
        a_weight: float = _DET_W_TILT,
    ) -> torch.Tensor:
        q = torch.minimum(torch.maximum(q0.clone(), lo), hi)
        r = residual(q, a_des, o_des, a_weight)
        eye = torch.eye(ARM_DOF, device=device, dtype=torch.float32)
        for _ in range(iters):
            jac = torch.zeros((r.shape[0], ARM_DOF), device=device, dtype=torch.float32)
            for j in range(ARM_DOF):
                qp = q.clone()
                stepped = min(float(q[j]) + _DET_FD_EPS, float(hi[j]))
                if stepped - float(q[j]) < 1.0e-9:  # 상한에 붙음 → 반대 방향 FD
                    stepped = float(q[j]) - _DET_FD_EPS
                qp[j] = stepped
                step = stepped - float(q[j])
                jac[:, j] = (residual(qp, a_des, o_des, a_weight) - r) / step
            jtj = jac.T @ jac + _DET_DAMPING_SQ * eye
            dq = -torch.linalg.solve(jtj, jac.T @ r)
            dq = torch.clamp(dq, -_DET_STEP_CLAMP, _DET_STEP_CLAMP)
            q = torch.minimum(torch.maximum(q + dq, lo), hi)
            r = residual(q, a_des, o_des, a_weight)
            if (
                float(torch.linalg.norm(r[:3]).item()) < 1.5e-3
                and float(torch.max(torch.abs(dq)).item()) < 1.0e-4
            ):
                break
        return q

    diag: dict[str, Any] = {
        "det_mode": "grasp" if grasp_mode else "waypoint",
        "det_pos_gate": round(float(pos_gate), 5),
    }
    best: tuple[float, torch.Tensor, torch.Tensor, dict[str, Any]] | None = None  # (score, q, gp, info)

    if not grasp_mode:
        # 이송·place 단계는 손목을 지면과 평행(palm-down, 접근축 ↓)으로 유지한 채 이동
        # — 큐브를 떨굴 때 수평 자세가 되도록(사용자 요청). 가중 _DET_W_LEVEL 은 위치
        # 우선이되 자세가 실제 반영될 만큼(0.05 는 항상 양보돼 tilt 채로 운반했음).
        a_level = (
            torch.tensor([0.0, 0.0, -1.0], device=device, dtype=torch.float32)
            if level_axis
            else None
        )
        q = solve(current_arm, a_level, None, _DET_ITERS_POSONLY, a_weight=_DET_W_LEVEL)
        write_q(q)
        gp = _grasp_point_pos(robot)[0].clone()
        pos_err = float(torch.linalg.norm(target - gp).item())
        if pos_err <= pos_gate:
            best = (pos_err, q, gp, {"det_pos_err_m": round(pos_err, 5)})
    else:
        base_xy = robot.data.root_pos_w[0, :2]
        r_h = target[:2] - base_xy
        n = float(torch.linalg.norm(r_h).item())
        r_hat = (
            torch.tensor([1.0, 0.0], device=device, dtype=torch.float32)
            if n < 1.0e-6
            else (r_h / n).to(dtype=torch.float32)
        )
        if tilt_ladder is not None:
            ladder = list(tilt_ladder)
        else:
            # 이 큐브에서 실행이 계통 실패한 tilt 는 제외 — 결정적이라 재선택 = 동일 실패.
            # 전부 실패면 원본 유지(전무보다 재시도가 낫다).
            ladder = [t for t in _DET_TILT_LADDER_DEG if t not in _FAILED_TILT_DEG]
            if not ladder:
                ladder = list(_DET_TILT_LADDER_DEG)
            k = max(0, int(variant)) % len(ladder)
            ladder = ladder[k:] + ladder[:k]
        yaw = _cube_yaw_w(env, cube_name) if cube_name is not None else None
        tried = 0
        for tilt_deg in ladder:
            tried += 1
            t = math.radians(abs(tilt_deg))
            sgn = 1.0 if tilt_deg >= 0.0 else -1.0
            a_des = torch.tensor(
                [
                    sgn * math.sin(t) * float(r_hat[0]),
                    sgn * math.sin(t) * float(r_hat[1]),
                    -math.cos(t),
                ],
                device=device,
                dtype=torch.float32,
            )
            a_des = a_des / torch.linalg.norm(a_des)
            q1 = solve(current_arm, a_des, None, _DET_ITERS_STAGE1)
            o_des: torch.Tensor | None = None
            if open_axis_override is not None:
                # 닫기/보정 단계 — descend 가 고른 개방축 그대로(재선택 시 wrist_roll
                # 90~180° 점프 → 닫는 중 손목 회전으로 큐브를 쳐냄).
                o_des = open_axis_override.to(device=device, dtype=torch.float32)
            elif yaw is not None:
                write_q(q1)
                o_now = _gripper_open_axis_h(robot)
                best_dot = 0.0
                for kk in range(4):
                    ang = yaw + kk * (math.pi / 2.0)
                    cand = torch.tensor(
                        [math.cos(ang), math.sin(ang), 0.0], device=device, dtype=torch.float32
                    )
                    d = float(torch.dot(cand, o_now).item())
                    if abs(d) > abs(best_dot):
                        best_dot = d
                        o_des = cand if d >= 0.0 else -cand
            q2 = solve(q1, a_des, o_des, _DET_ITERS_STAGE2)
            write_q(q2)
            gp = _grasp_point_pos(robot)[0].clone()
            pos_err = float(torch.linalg.norm(target - gp).item())
            # 수평 오차 방향분해는 *진단 전용*이다(score 미반영). v8 에서 ⊥성분 ×2.0 을
            # score 에 넣자 ladder 선택이 왜곡됐다 — ① 저tilt(15°) 후보가 err_perp 작다는
            # 이유로 승리해 scoop 부족으로 못 물고(ep2/8/10) ② descend_fix 재계산 순위가
            # 뒤집혀 다른 tilt 로 스윙, 큐브 0.57m 비산(ep4/12/13) → 90→65% 회귀 실측.
            # 계획 단계 err_perp 는 원래 1~3mm 로 작다 — 모서리 그립의 진짜 원인은 실행
            # 오차이며, 그 보정은 descend_fix(실행 후, tilt 잠금)가 담당한다.
            err_h_vec = (target - gp)[:2]
            pos_err_h = float(torch.linalg.norm(err_h_vec).item())
            jaw_z = _finger_min_z(robot, "jaw")
            fix_z = _finger_min_z(robot, "gripper")
            # 책상 *극단* 침투 필터: 가상 FK 엔 접촉 해소가 없어 좋은 scoop 계획도 손끝이
            # 책상면보다 수 mm 아래로 들어가는 게 정상이다(실행 시 물리가 받침). 다만
            # 15mm 이상 박히는 계획은 실행 시 팔이 책상에 걸려 멈춘다(v4 ep8: 27cm 미달).
            if min(jaw_z, fix_z) < floor_z - 0.015:
                continue
            # 스트래들 검증: 두 finger 의 하단(tip)이 개방축 투영에서 큐브 중심을 *사이에
            # 두고*, 적어도 한쪽은 큐브 윗면 아래로 내려가야 한다(측면 삽입). 한쪽 tip 이
            # 큐브 윗면 위에 얹히는 계획은 닫아도 못 물고 윗면을 누르며 떨기만 한다
            # (사용자 v19 ep0 영상 — docs/pics/cube_desk/시뮬_그립_3.png. 사용자 시연
            # 그립_1~3 과의 결정적 차이가 이 '양옆 감싸기'다). 실패 시 다음 tilt 후보로.
            if o_des is not None and cube_name is not None:
                cube_c = scene[cube_name].data.root_pos_w[0]
                cube_top_z = float(cube_c[2].item()) + _cube_half_z(cube_name)
                proj_c = float(torch.dot(cube_c[:2], o_des[:2]).item())
                tip_sides: list[float] = []
                tip_zs: list[float] = []
                for fbody in ("jaw", "gripper"):
                    pw = _finger_world_corners(robot, fbody)
                    zmin = float(pw[:, 2].min().item())
                    low = pw[pw[:, 2] <= zmin + 0.012]
                    proj = float((low[:, :2] @ o_des[:2]).mean().item())
                    tip_sides.append(proj - proj_c)
                    tip_zs.append(zmin)
                straddles = tip_sides[0] * tip_sides[1] < 0.0
                inserted = min(tip_zs) <= cube_top_z - 0.004
                if not (straddles and inserted):
                    continue
            tilt_pen = max(0.0, jaw_z - floor_z) + max(0.0, jaw_z - fix_z)
            continuity = float(torch.linalg.norm(q2 - current_arm).item())
            # 개방축↔큐브 면 정렬 잔차(rad). 크면 모서리(마름모)를 집는다 — 점수에 반영해
            # 정렬 나쁜 후보를 다음 ladder 후보에 밀리게 한다.
            roll_err = 0.0
            if o_des is not None:
                o_now = _gripper_open_axis_h(robot)
                roll_err = math.acos(max(-1.0, min(1.0, abs(float(torch.dot(o_now, o_des).item())))))
            if o_des is not None:
                err_along = abs(float(torch.dot(err_h_vec, o_des[:2]).item()))
                err_perp = math.sqrt(max(0.0, pos_err_h * pos_err_h - err_along * err_along))
            else:
                err_along = err_perp = pos_err_h * 0.7071
            score = (
                pos_err
                + _DET_SCOOP_WEIGHT * tilt_pen
                + _DET_CONTINUITY_WEIGHT * continuity
                + 0.05 * roll_err
            )
            info = {
                "det_tilt_deg": tilt_deg,
                "det_pos_err_m": round(pos_err, 5),
                "det_pos_err_h_m": round(pos_err_h, 5),
                "det_err_perp_m": round(err_perp, 5),
                "det_tilt_pen": round(tilt_pen, 5),
                "det_roll_err_deg": round(math.degrees(roll_err), 1),
                "det_jaw_min_z": round(jaw_z, 5),
                "det_fix_min_z": round(fix_z, 5),
                "det_open_axis": (
                    [round(float(v), 5) for v in o_des.tolist()] if o_des is not None else None
                ),
            }
            # 수평 오차는 채점·gate 어디에도 넣지 않는다 — hard 7mm gate 는 후보 전멸로
            # 90→65% 회귀(v7), soft ⊥×2.0 가중도 ladder 왜곡으로 65% 회귀(v8). 실행 후
            # 수평 오차가 크면 descend_fix(tilt 잠금) 가 보정한다.
            if pos_err <= pos_gate and (best is None or score < best[0]):
                best = (score, q2, gp, info)
                # 위치 정합 + scoop + 면 정렬 모두 충분한 해 → 남은 후보 생략(결정적 순서라 재현 동일)
                if pos_err <= pos_gate * 0.5 and tilt_pen <= 0.005 and roll_err <= math.radians(15.0):
                    break
        diag["det_candidates_tried"] = tried
        if yaw is not None:
            diag["det_cube_yaw_deg"] = round(math.degrees(yaw), 2)

    robot.write_joint_state_to_sim(saved_q, saved_v, env_ids=env_ids)
    scene.update(0.0)

    if best is None:
        diag["det_fallback"] = True
        return None, diag
    _score, q_best, gp_best, info = best
    diag.update(info)
    diag["det_fallback"] = False
    # 기존 random-FK plan 키와 동일 의미의 필드(트레이스 소비자 호환).
    diag["planned_error_m"] = info["det_pos_err_m"]
    diag["planned_grasp_point_w"] = _round_list(gp_best)
    diag["planned_joint_target"] = _round_list(q_best)
    return q_best, diag


def _fk_solve_joint_target(
    env,
    target_grasp_point_w: torch.Tensor,
    gripper_target: float,
    device: str,
    *,
    samples: int,
    continuity_weight: float,
    seed_offset: int,
    grasp_tilt_weight: float = 0.0,
    floor_z: float = CUBE_DESK_TOP_Z,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Random-FK waypoint solver.

    현재 scene에서 robot joint state만 잠깐 써서 FK 후보를 평가한 뒤 원래 상태로
    되돌린다. cube/bowl pose는 쓰지 않으며, 실제 rollout은 반환된 joint target을
    action으로 추종한다.
    """

    scene = env.unwrapped.scene
    robot = scene["robot"]
    env_ids = torch.tensor([0], device=device, dtype=torch.long)
    saved_q = robot.data.joint_pos[:, :6].clone()
    saved_v = robot.data.joint_vel[:, :6].clone()
    current_arm = saved_q[0, :ARM_DOF].clone()
    lower, upper = _arm_limits(robot, device)
    target = target_grasp_point_w.to(device=device, dtype=torch.float32).reshape(3)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed + seed_offset))

    best_score = float("inf")
    best_dist = float("inf")
    best_q = current_arm.clone()
    best_grasp_point = _grasp_point_pos(robot)[0].clone()
    best_tilt_pen = 0.0
    zero_vel = torch.zeros((1, 6), device=device)
    use_tilt = grasp_tilt_weight > 0.0

    def evaluate(q_arm: torch.Tensor) -> None:
        nonlocal best_score, best_dist, best_q, best_grasp_point, best_tilt_pen
        q = torch.zeros((1, 6), device=device)
        q[0, :ARM_DOF] = q_arm
        q[0, 5] = float(gripper_target)
        robot.write_joint_state_to_sim(q, zero_vel, env_ids=env_ids)
        scene.update(0.0)
        grasp_point = _grasp_point_pos(robot)[0].clone()
        dist = float(torch.linalg.norm(grasp_point - target).item())
        continuity = float(torch.linalg.norm(q_arm - current_arm).item())
        score = dist + continuity_weight * continuity
        tilt_pen = 0.0
        if use_tilt:
            # 이동 jaw 가 (1) 바닥까지 내려오고 (2) 고정 finger 아래로 가는 pose 를 선호.
            # SO-101 은 고정 finger 가 길어, 약tilt 면 이동 jaw 가 큐브 위에 남아 grasp 실패한다.
            jaw_z = _finger_min_z(robot, "jaw")
            fix_z = _finger_min_z(robot, "gripper")
            tilt_pen = max(0.0, jaw_z - floor_z) + max(0.0, jaw_z - fix_z)
            score = score + grasp_tilt_weight * tilt_pen
        if score < best_score:
            best_score = score
            best_dist = dist
            best_q = q_arm.clone()
            best_grasp_point = grasp_point.clone()
            best_tilt_pen = tilt_pen

    # 현재 근처 후보와 전역 후보를 섞는다. 전역 후보가 테이블 근처 자세를 찾고,
    # 근처 후보가 불필요한 큰 관절 점프를 줄인다.
    evaluate(current_arm)
    local_count = min(max(samples // 3, 1), samples)
    global_count = max(samples - local_count, 1)
    for _ in range(local_count):
        noise = torch.randn((ARM_DOF,), generator=gen, device=device) * 0.35
        evaluate(torch.minimum(torch.maximum(current_arm + noise, lower), upper))
    for _ in range(global_count):
        evaluate(lower + (upper - lower) * torch.rand((ARM_DOF,), generator=gen, device=device))

    robot.write_joint_state_to_sim(saved_q, saved_v, env_ids=env_ids)
    scene.update(0.0)
    return best_q, {
        "planned_error_m": round(best_dist, 5),
        "planned_score": round(best_score, 5),
        "planned_tilt_pen": round(best_tilt_pen, 5),
        "planned_grasp_point_w": _round_list(best_grasp_point),
        "planned_joint_target": _round_list(best_q),
    }


def _cube_inside_bowl(env, cube_name: str, radius: float) -> bool:
    inside = task_mdp.object_inside_container(
        env.unwrapped,
        object_cfg=SceneEntityCfg(cube_name),
        container_cfg=SceneEntityCfg(BOWL_NAME),
        radius=radius,
        height_range=BOWL_HEIGHT_RANGE,
    )
    return bool(inside[0].item())


def _cube_half_z(cube_name: str) -> float:
    return CUBE_HALF_HEIGHTS.get(cube_name, CUBE_HALF_Z)


def _pick_gripper_open(cube_name: str) -> float:
    return max(float(args.gripper_open), PICK_GRIPPER_OPEN_BY_CUBE.get(cube_name, 0.0))


def _pick_descend_gripper(cube_name: str) -> float:
    return max(float(args.descend_gripper), DESCEND_GRIPPER_BY_CUBE.get(cube_name, 0.0))


# 큐브가 이 영역을 벗어나면 더 시도하지 않는다(밀려나 reach 밖/책상 밖 — 재시도 낭비 방지).
# scatter 범위(x[1.66,2.04], y[-0.46,-0.345]) + 6cm 마진.
_CUBE_RANGE_X = (1.60, 2.10)
_CUBE_RANGE_Y = (-0.52, -0.285)


def _cube_out_of_range(env, cube_name: str) -> str | None:
    """큐브가 회수 불가 영역이면 사유 문자열, 아니면 None."""
    pos = env.unwrapped.scene[cube_name].data.root_pos_w[0]
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    if z < CUBE_DESK_TOP_Z - 0.05:
        return "fell_off_desk"
    if not (_CUBE_RANGE_X[0] <= x <= _CUBE_RANGE_X[1] and _CUBE_RANGE_Y[0] <= y <= _CUBE_RANGE_Y[1]):
        return "pushed_out_of_reach"
    return None


def _cube_lifted(env, cube_name: str, min_lift: float = 0.05) -> bool:
    cube = env.unwrapped.scene[cube_name]
    # 책상에서 쉬는 큐브 중심 z ≈ CUBE_DESK_TOP_Z + half(크기별). 그보다 min_lift 이상 올라가면 잡힌 것.
    # min_lift 는 lift 명령 높이(--lift_height 0.08)보다 충분히 낮아야 한다 — 0.08 동일값은
    # 마진 0 이라, 쥔 깊이·PD 처짐·렌더링 부하로 큐브가 +6~7cm 에 머물면 *성공한 grasp 을*
    # false 판정해 놓고 재시도하다 떨군다(v9 영상 run: 실패 13ep 중 10ep 이 hold 1~2cm 로
    # 쥔 채 +5.3~7.4cm 에서 게이트 미달 = 90→35% 폭락 주범). 0.05 = 3cm 마진이면서 책상 위
    # 큐브 최고 높이(40mm)보다 위라 미파지 오탐 없음.
    rest_center_z = CUBE_DESK_TOP_Z + _cube_half_z(cube_name)
    return bool((cube.data.root_pos_w[0, 2] > rest_center_z + min_lift).item())


def _bowl_top_z(env) -> float:
    """현재(reset/settle 이후) bowl root z. 절대 상수 대신 placement 기준으로 쓴다."""
    return float(env.unwrapped.scene[BOWL_NAME].data.root_pos_w[0, 2].item())


def _placed_and_released(env, cube_names: list[str], radius: float) -> bool:
    robot = env.unwrapped.scene["robot"]
    gripper_open = bool((robot.data.joint_pos[0, 5] > 0.60).item())
    if not gripper_open:
        return False
    return all(_cube_inside_bowl(env, name, radius) for name in cube_names)


def _phase(
    env,
    device: str,
    name: str,
    target_fn: Callable[[], torch.Tensor],
    gripper_target: float,
    steps: int,
    trace: list[dict[str, Any]],
    command: torch.Tensor,
    recorder: LeRobotV3EpisodeRecorder | None,
    expert_recorder: ExpertTrajectoryRecorder | None = None,
    rmpflow_driver: SO101RmpFlowJointTarget | None = None,
    *,
    tolerance: float,
    target_R_fn: Callable[[], Any] | None = None,
    success_fn: Callable[[], bool] | None = None,
    min_steps: int = 1,
    grasp_cube_name: str | None = None,
) -> dict[str, Any]:
    """단계 실행. ``steps`` 는 상한이며, 목표 도달(또는 ``success_fn``) 시 즉시 종료(early-exit)한다.

    ``target_R_fn`` 가 주어지고 cartesian driver(ikpy/lula/rmpflow)를 쓰면 매 스텝 그 world
    회전을 orientation 목표로 넘긴다(top-down/level). orientation 활성 시 position-only Jacobian
    refine 은 끈다(orientation 을 망가뜨리지 않도록).
    """
    robot = env.unwrapped.scene["robot"]
    min_error = float("inf")
    final_error = float("inf")
    final_error_h = float("inf")
    reached_step: int | None = None
    done_seen = False
    early_exit_step: int | None = None
    orientation_active = target_R_fn is not None and rmpflow_driver is not None

    def _cur_R() -> Any | None:
        return None if target_R_fn is None else np.asarray(target_R_fn(), dtype=np.float64)

    target = target_fn().to(device=device, dtype=torch.float32).reshape(1, 3)
    # descend/grasp 단계에서만 grasp tilt 점수항을 켜고(이동 jaw 를 큐브 바닥까지 내리는 pose 선호),
    # 팔 속도를 늦추며(큐브 쳐냄 방지), arm 이 tilted q_goal 에 도달할 때까지 완주(early-exit 로 tilt 잘림 방지).
    is_grasp_phase = (".descend" in name) or (".grasp" in name)
    global _ARM_STEP_OVERRIDE, _GRIPPER_STEP_OVERRIDE
    _prev_arm_override = _ARM_STEP_OVERRIDE
    _prev_gripper_override = _GRIPPER_STEP_OVERRIDE
    if is_grasp_phase and args.grasp_arm_step_delta > 0.0:
        _ARM_STEP_OVERRIDE = abs(args.grasp_arm_step_delta)
    # close-on-cube(.grasp)만 느린 그리퍼 — 나머지 단계는 travel 속도로 빠르게.
    if ".grasp" in name:
        _GRIPPER_STEP_OVERRIDE = abs(args.max_gripper_step_delta)
    # tilt 는 descend 에서 완성해야 한다(arm 이 tilted q_goal 에 도달할 때까지 완주). grasp(닫기)
    # 단계까지 settle 을 강제하면 닫는 동안 arm 이 재계산된 자세로 움직여 큐브를 밀어내 grip 이 풀린다.
    require_arm_settled = (".descend" in name) and rmpflow_driver is None
    if rmpflow_driver is None:
        global _LAST_DESCEND_TILT_DEG, _LAST_DESCEND_OPEN_AXIS
        q_goal = None
        det_diag: dict[str, Any] = {}
        is_close_phase = ".grasp" in name
        # descend_fix(착좌 보정)도 잠근다 — 자유 재계산이 다른 tilt 를 고르면 이미 내려간
        # 팔이 그 자세로 스윙하며 큐브를 쳐낸다(v8 ep4: 35°→25° 갈아타기로 0.57m 비산).
        is_fix_phase = ".descend_fix" in name
        if args.grasp_config_mode == "deterministic":
            det_gate = (args.descend_tolerance if is_grasp_phase else args.target_tolerance) * 0.8
            # 닫기/보정 단계는 descend 가 고른 tilt·개방축으로 잠근다(tilt 갈아타기 스윙 +
            # 개방축 재선택의 wrist_roll 90~180° 회전 둘 다 닫는 중 큐브를 쳐낸다).
            locked_ladder: tuple[float, ...] | None = None
            locked_axis: torch.Tensor | None = None
            if is_close_phase or is_fix_phase:
                if _LAST_DESCEND_TILT_DEG is not None:
                    locked_ladder = (_LAST_DESCEND_TILT_DEG,)
                if _LAST_DESCEND_OPEN_AXIS is not None:
                    locked_axis = torch.tensor(
                        _LAST_DESCEND_OPEN_AXIS, device=device, dtype=torch.float32
                    )
            q_goal, det_diag = _deterministic_solve_joint_target(
                env,
                target,
                gripper_target,
                device,
                grasp_mode=is_grasp_phase,
                pos_gate=det_gate,
                cube_name=grasp_cube_name,
                variant=_attempt_variant_from_phase_name(name),
                floor_z=CUBE_DESK_TOP_Z,
                tilt_ladder=locked_ladder,
                open_axis_override=locked_axis,
                # 이송·place·그릇이탈 단계는 손목 수평(palm-down) 유지 — 떨굴 때 수평
                # 자세 + 그릇 상공 통과 시 낮게 늘어진 손가락이 그릇을 긁지 않게.
                level_axis=any(
                    token in name
                    for token in (
                        ".transport_via", ".move_to_pre_place", ".place_descend",
                        ".bowl_departure_via",
                    )
                ),
            )
            # descend_fix 는 잠긴 tilt 재계산이라 갱신 불필요(원 descend 의 선택을 보존).
            if ".descend" in name and not is_fix_phase and q_goal is not None:
                _LAST_DESCEND_TILT_DEG = det_diag.get("det_tilt_deg")
                _LAST_DESCEND_OPEN_AXIS = det_diag.get("det_open_axis")
        if q_goal is not None:
            plan = det_diag
        elif args.grasp_config_mode == "deterministic" and (is_close_phase or is_fix_phase):
            # 닫기/보정 단계에서 잠긴 tilt 가 gate 를 못 넘으면 random-FK 로 갈아타지 않고
            # (전혀 다른 자세로 스윙할 위험) 현재 자세를 유지한다(보정 포기·그리퍼만 진행).
            q_goal = command[:ARM_DOF].clone()
            plan = {**det_diag, "det_grasp_hold_pose": True}
        else:
            # random_fk 모드 또는 deterministic gate 실패 폴백.
            q_goal, plan = _fk_solve_joint_target(
                env,
                target,
                gripper_target,
                device,
                samples=args.fk_samples,
                continuity_weight=args.continuity_weight,
                seed_offset=len(trace) * 997,
                grasp_tilt_weight=(args.grasp_tilt_weight if is_grasp_phase else 0.0),
                floor_z=CUBE_DESK_TOP_Z,
            )
            if det_diag:
                plan.update(det_diag)
    else:
        q_goal, plan = rmpflow_driver.compute(target[0], _cur_R())

    requested_steps = int(steps)
    if rmpflow_driver is None:
        actual_steps = max(1, requested_steps, _slew_limited_step_count(command, q_goal, gripper_target))
    else:
        actual_steps = max(1, requested_steps)
    # stall 감지: 단계가 목표 미달이면 cap 까지 헛돌며 팔이 멈춘 듯 보인다(예: descend
    # 160step=5.3s — 사용자 영상 관찰 "중간중간 멈춰 있음"). 명령이 수렴했는데 오차
    # 개선이 없으면 잔여 step 을 버리고 다음 단계로 넘어간다(성공 판정과 별개 — 실패
    # 단계의 *대기 시간*만 제거).
    stall_count = 0
    stall_best_err = float("inf")
    stall_prev_cmd = command[:ARM_DOF].clone()
    carry_phase = any(token in name for token in (".lift", ".transport", ".place", ".move_to_pre_place", ".place_descend"))
    phase_min_effort = args.carry_min_gripper_effort if carry_phase and gripper_target <= args.gripper_closed else None
    refine_steps = 0
    for step in range(actual_steps):
        err = float(torch.linalg.norm(_grasp_point_pos(robot)[0] - target[0]).item())
        if orientation_active:
            # top-down 강제: Isaac-frame DLS(위치+접근축) 를 매 스텝 적용해 실제 grasp point 를
            # 큐브에 맞추고 접근축을 수직으로 정렬한다(ikpy/URDF offset 불일치 회피).
            use_refine = not args.disable_jacobian_refine
        else:
            use_refine = (
                (args.enable_jacobian_refine or (rmpflow_driver is not None and not args.disable_rmpflow_jacobian_refine))
                and not args.disable_jacobian_refine
                and (step >= actual_steps // 2 or err <= max(0.08, args.target_tolerance * 3.0))
            )
            # det descend 미세보정: 팔이 계획 자세(q_goal)에 도착했는데도 grasp point 가
            # tolerance 밖이면(PD 처짐·접촉) 마지막 cm 를 폐루프 DLS 로 밀어넣는다.
            if (
                not use_refine
                and args.grasp_config_mode == "deterministic"
                and rmpflow_driver is None
                and ".descend" in name
                and not args.disable_jacobian_refine
                # refine 상한: 폐루프가 수렴 못 하면 미세 진동만 만든다("집기 전 부들부들"
                # — 사용자 v17 ep0 영상, ep8 의 78step 폭주). 25step(~0.8s) 안에 못 잡으면
                # 포기하고 닫기 게이트(skip_close)에 맡긴다.
                and refine_steps < 25
            ):
                arm_at_goal = float(
                    torch.max(torch.abs(command[:ARM_DOF] - q_goal[:ARM_DOF])).item()
                ) <= 2.0e-3
                use_refine = arm_at_goal and err > tolerance
        arm_target = q_goal
        if rmpflow_driver is not None:
            target = target_fn().to(device=device, dtype=torch.float32).reshape(1, 3)
            arm_target, plan = rmpflow_driver.compute(target[0], _cur_R())
            if use_refine:
                refine_action, _ = _ik_action(
                    robot,
                    target,
                    gripper_target,
                    device,
                    damping=args.ik_damping,
                    gain=args.ik_gain,
                    max_joint_delta=args.max_joint_delta,
                    topdown=orientation_active,
                    ori_weight=args.ik_ori_weight,
                )
                arm_target = refine_action[0, :ARM_DOF]
                refine_steps += 1
                plan["rmpflow_jacobian_refine"] = True
        elif use_refine:
            refine_action, _ = _ik_action(
                robot,
                target,
                gripper_target,
                device,
                damping=args.ik_damping,
                gain=args.ik_gain,
                max_joint_delta=args.max_joint_delta,
            )
            arm_target = refine_action[0, :ARM_DOF]
            refine_steps += 1
        action = _joint_position_action(robot, command, arm_target, gripper_target, device)
        if expert_recorder is not None:
            expert_recorder.record(env, action, name)
        step_out = _step_env(env, action, min_gripper_effort=phase_min_effort)
        if len(step_out) == 5:
            _obs, _rew, terminated, truncated, _infos = step_out
            dones = terminated | truncated
        else:
            _obs, _rew, dones, _infos = step_out
        if recorder is not None:
            recorder.record(env, action)
        # 스텝 직후 위치로 오차 갱신(early-exit 판정용).
        err_vec = _grasp_point_pos(robot)[0] - target[0]
        err_now = float(torch.linalg.norm(err_vec).item())
        min_error = min(min_error, err, err_now)
        final_error = err_now
        final_error_h = float(torch.linalg.norm(err_vec[:2]).item())
        if bool(dones[0].item()):
            done_seen = True
        if err_now <= tolerance and reached_step is None:
            reached_step = step + 1
        # ── early-exit: 단계가 성공하면 남은 step 을 소진하지 않고 바로 다음 단계로 ──
        if step + 1 >= max(1, min_steps):
            gripper_reached = abs(float(command[5].item()) - float(gripper_target)) <= 1.0e-3
            # grasp 단계는 arm 이 tilted q_goal 에 도달해야 tilt 가 실제 실행된다(제어점만 닿으면
            # 자세가 덜 기운 채 끊김). 그 외 단계는 위치 도달이면 충분.
            if require_arm_settled:
                # 미세보정(refine)이 이미 발동했다면 q_goal 도달 후 폐루프로 넘어간 것이므로
                # tilt 는 실행 완료 — command↔q_goal 비교 대신 settled 로 본다.
                arm_settled = refine_steps > 0 or float(
                    torch.max(torch.abs(command[:ARM_DOF] - q_goal[:ARM_DOF])).item()
                ) <= 2.0e-3
            else:
                arm_settled = True
            if success_fn is not None:
                done_phase = bool(success_fn())
            else:
                done_phase = (err_now <= tolerance) and gripper_reached and arm_settled
            if done_phase:
                early_exit_step = step + 1
                break
            # stall 감지 exit: 그리퍼 전이가 끝났는데 오차가 1mm 이상 개선되지 않는
            # step 이 0.5s(15step) 누적되면 — 물리적으로 더 갈 수 없는 단계다. 잔여
            # step 을 소진하지 않는다. 명령 수렴 여부는 조건에서 뺀다: refine 폐루프가
            # command 를 계속 미세 갱신하면 수렴 조건이 영영 거짓이라 stall 이 무력해져
            # cap 까지 헛돈다(v16 실측: descend cap 도달 34건). 슬루 이동 중에는 오차가
            # 줄어들므로 이 카운터는 자연히 리셋된다.
            # 슬루급(>0.01rad) 이동 중에는 리셋 — 보간 호에서 grasp point 오차가 일시
            # 정체해도 끊지 않는다. refine 의 미세 갱신(<0.01rad)은 이동으로 안 친다.
            cmd_moving = (
                float(torch.max(torch.abs(command[:ARM_DOF] - stall_prev_cmd)).item()) > 0.01
            )
            stall_prev_cmd = command[:ARM_DOF].clone()
            if gripper_reached and not cmd_moving:
                if err_now < stall_best_err - 1.0e-3:
                    stall_best_err = err_now
                    stall_count = 0
                else:
                    stall_count += 1
                    if stall_count >= 15:
                        early_exit_step = step + 1
                        break
            else:
                stall_count = 0

    stat = {
        "phase": name,
        "steps": int(early_exit_step if early_exit_step is not None else actual_steps),
        "requested_steps": requested_steps,
        "max_steps": int(actual_steps),
        "reached_step": reached_step,
        "early_exit_step": early_exit_step,
        "min_error_m": round(min_error, 5),
        "final_error_m": round(final_error, 5),
        "final_error_h_m": round(final_error_h, 5),
        "done_seen": done_seen,
        "target_grasp_point_w": _round_list(target[0]),
        "grasp_point_w": _round_list(_grasp_point_pos(robot)[0]),
        "joint_pos": _round_list(robot.data.joint_pos[0, :6]),
        "jacobian_refine_steps": refine_steps,
        **_diagnostic_pose(env),
        **plan,
    }
    trace.append(stat)
    _ARM_STEP_OVERRIDE = _prev_arm_override
    _GRIPPER_STEP_OVERRIDE = _prev_gripper_override
    return stat


def _hold_joint_target(
    env,
    target: torch.Tensor,
    gripper_target: float,
    steps: int,
    command: torch.Tensor,
    recorder: LeRobotV3EpisodeRecorder | None,
    expert_recorder: ExpertTrajectoryRecorder | None = None,
    phase: str = "hold",
) -> None:
    robot = env.unwrapped.scene["robot"]
    device = str(target.device)
    for _ in range(max(1, steps)):
        action = _joint_position_action(robot, command, target[:ARM_DOF], gripper_target, device)
        if expert_recorder is not None:
            expert_recorder.record(env, action, phase)
        _step_env(env, action)
        if recorder is not None:
            recorder.record(env, action)


def _idle_home_arm_target(robot, device: str) -> torch.Tensor:
    default_joint_pos = getattr(robot.data, "default_joint_pos", None)
    if default_joint_pos is not None:
        return default_joint_pos[0, :ARM_DOF].to(device=device, dtype=torch.float32).clone()
    return torch.zeros(ARM_DOF, device=device, dtype=torch.float32)


def _move_to_idle_home(
    env,
    device: str,
    command: torch.Tensor,
    trace: list[dict[str, Any]],
    recorder: LeRobotV3EpisodeRecorder | None,
    expert_recorder: ExpertTrajectoryRecorder | None,
    *,
    phase: str,
) -> None:
    robot = env.unwrapped.scene["robot"]
    home_arm = _idle_home_arm_target(robot, device)
    _hold_joint_target(
        env,
        home_arm,
        args.gripper_open,
        args.idle_home_steps,
        command,
        recorder,
        expert_recorder,
        phase=phase,
    )
    trace.append({
        "phase": phase,
        "state": PickCubeFSMState.IDLE.value,
        "steps": args.idle_home_steps,
        "grasp_point_w": _round_list(_grasp_point_pos(robot)[0]),
        "joint_pos": _round_list(robot.data.joint_pos[0, :6]),
        **_diagnostic_pose(env),
    })


def _target_from_cube(env, cube_name: str, dz: float) -> Callable[[], torch.Tensor]:
    def target() -> torch.Tensor:
        cube = env.unwrapped.scene[cube_name]
        pos = cube.data.root_pos_w[0].clone()
        pos[2] += dz
        return pos

    return target


def _gripper_open_axis_h(robot) -> torch.Tensor:
    """그리퍼 개방/폐쇄 축의 *수평* 단위벡터(world). 고정 finger(gripper) → 이동 jaw 방향.

    SO-101 그리퍼는 한쪽 고정 + 한쪽 모터 jaw 구조라, 큐브 중심에 수직으로 내리꽂으면 고정
    finger 가 큐브를 위에서 찌른다. 이 축을 따라 grasp 목표를 옆으로 밀어 큐브가 두 jaw 사이
    gap 에 오게 한다."""
    d = (_body_pos(robot, "jaw")[0] - _body_pos(robot, "gripper")[0]).clone()
    d[2] = 0.0
    n = torch.linalg.norm(d)
    if float(n) < 1e-6:
        return torch.tensor([1.0, 0.0, 0.0], device=d.device, dtype=d.dtype)
    return d / n


def _target_pick(env, cube_name: str) -> Callable[[], torch.Tensor]:
    """Descend/grasp 목표: 큐브 xy 위, z = 큐브 중심 + grasp_pick_offset, 그리고 그리퍼 개방축을
    따라 grasp_lateral_offset 만큼 옆으로 민 위치.

    고정 finger 가 큐브를 찌르지 않고 큐브가 두 jaw 의 gap 에 들어오도록 한다(사용자 지적). 큐브
    중심 기준 상대값이라 reach 안에서 안정적이다."""

    def target() -> torch.Tensor:
        robot = env.unwrapped.scene["robot"]
        cube = env.unwrapped.scene[cube_name]
        pos = cube.data.root_pos_w[0].clone()
        lateral = float(args.grasp_lateral_offset)
        if abs(lateral) > 1e-6:
            axis = _gripper_open_axis_h(robot).to(device=pos.device, dtype=pos.dtype)
            pos[:2] = pos[:2] + lateral * axis[:2]
        pos[2] = pos[2] + float(args.grasp_pick_offset)
        return pos

    return target


def _topdown_R_fn() -> Callable[[], Any] | None:
    """Drop/place 단계용 top-down(palm-down) 목표 회전 공급자. disable_topdown 시 None."""
    if args.disable_topdown:
        return None
    R = np.asarray(TOPDOWN_R_WORLD, dtype=np.float64)
    return lambda: R


def _pick_R_fn() -> Callable[[], Any] | None:
    """Pick 단계 목표 회전. 기본은 None(자연 tilt — top-down 으론 책상 큐브에 reach 불가).
    --topdown_pick 시에만 strict top-down 을 강제한다."""
    if args.topdown_pick:
        return _topdown_R_fn()
    return None


def _target_from_bowl(env, dz: float, xy_offset: torch.Tensor | None = None) -> Callable[[], torch.Tensor]:
    def target() -> torch.Tensor:
        bowl = env.unwrapped.scene[BOWL_NAME]
        pos = bowl.data.root_pos_w[0].clone()
        if xy_offset is not None:
            pos[:2] += xy_offset.to(device=pos.device, dtype=pos.dtype)
        # live bowl z 기준(과거 DESK_TOP_Z=0.76 펜 데스크값은 큐브 데스크에서 ~4.5cm 과대였다).
        pos[2] = pos[2] + dz
        return pos

    return target


def _fixed_target(target_w: torch.Tensor) -> Callable[[], torch.Tensor]:
    target = target_w.clone()

    def get_target() -> torch.Tensor:
        return target.clone()

    return get_target


def _bowl_place_offset(device: str, placement_index: int) -> torch.Tensor:
    radius = max(0.0, float(args.bowl_place_offset_radius))
    direction = BOWL_PLACE_OFFSET_DIRECTIONS[placement_index % len(BOWL_PLACE_OFFSET_DIRECTIONS)]
    scale = radius / math.sqrt(2.0)
    return torch.tensor([direction[0] * scale, direction[1] * scale], device=device, dtype=torch.float32)


def _ordered_active_names(env, active_names: list[str]) -> list[str]:
    if args.object_order == "name":
        return list(active_names)
    if args.object_order == "hard_first":
        hard_order = ["Cube4", "Cube1", "Cube2", "Cube3"]
        active = set(active_names)
        return [name for name in hard_order if name in active] + [name for name in active_names if name not in hard_order]

    scene = env.unwrapped.scene
    names = list(active_names)
    if args.object_order == "far_base_first":
        # 로봇 base 에서 먼(reach 어려운) 큐브부터 집는다. 그릇이 비고 다른 큐브가 다 놓인
        # 상태에서 가장 까다로운 큐브를 먼저 처리해, 이후 가까운 큐브 approach 가 기존 큐브를
        # 치지 않게 한다.
        base_xy = scene["robot"].data.root_pos_w[0, :2]
        names.sort(
            key=lambda name: float(((scene[name].data.root_pos_w[0, :2] - base_xy) ** 2).sum().item()),
            reverse=True,
        )
        return names
    if args.object_order == "raster":
        # top 카메라 기준 좌상단 → 우하단(사람이 글 읽듯) 순서.
        #   · "위"  = y 큰 쪽(그릇 쪽, 이미지 상단)  → 행은 y 내림차순
        #   · "왼쪽" = x 작은 쪽(이미지 왼쪽)         → 같은 행 안에서 x 오름차순
        # 흩어진 큐브를 raster_row_band(m) 폭으로 행으로 묶어 행 우선 정렬한다.
        xy = {name: scene[name].data.root_pos_w[0, :2].detach().cpu() for name in names}
        y_top = max(float(v[1].item()) for v in xy.values())
        band = max(1e-4, float(args.raster_row_band))
        def _raster_key(name: str) -> tuple[int, float]:
            x = float(xy[name][0].item())
            y = float(xy[name][1].item())
            row = int((y_top - y) // band)  # 0 = 가장 위(그릇 쪽) 행
            return (row, x)
        names.sort(key=_raster_key)
        return names

    reverse = args.object_order == "near_bowl_first"
    # 현재 reset pose 기준으로 bowl에 가까운 y(덜 음수) 큐브부터 집으면,
    # 아래쪽 큐브를 지나가며 중앙 큐브를 밀어내는 일이 줄어든다.
    names.sort(key=lambda name: float(scene[name].data.root_pos_w[0, 1].item()), reverse=reverse)
    return names


def _ik_position_action(env, target_grasp_point_w: torch.Tensor, gripper_command: float, device: str) -> torch.Tensor:
    robot = env.unwrapped.scene["robot"]
    target_w = target_grasp_point_w.to(device=device, dtype=torch.float32).reshape(1, 3)
    target_b = quat_apply(
        quat_inv(robot.data.root_quat_w),
        target_w - robot.data.root_pos_w,
    )
    current_w = _grasp_point_pos(robot).to(device=device, dtype=torch.float32)
    current_b = quat_apply(
        quat_inv(robot.data.root_quat_w),
        current_w - robot.data.root_pos_w,
    )
    delta_b = target_b - current_b
    max_step = max(1.0e-6, float(args.diff_ik_step_size))
    delta_norm = torch.linalg.vector_norm(delta_b, dim=-1, keepdim=True).clamp_min(1.0e-6)
    delta_b = delta_b * torch.clamp(max_step / delta_norm, max=1.0)
    action = torch.zeros((1, 4), device=device, dtype=torch.float32)
    action[:, :3] = delta_b
    action[:, 3] = float(gripper_command)
    return action


def _ik_phase(
    env,
    device: str,
    name: str,
    target_fn: Callable[[], torch.Tensor],
    gripper_command: float,
    steps: int,
    trace: list[dict[str, Any]],
    recorder: LeRobotV3EpisodeRecorder | None,
    *,
    tolerance: float,
) -> dict[str, Any]:
    robot = env.unwrapped.scene["robot"]
    min_error = float("inf")
    final_error = float("inf")
    reached_step: int | None = None
    done_seen = False
    actual_steps = max(1, int(steps))
    carry_phase = any(token in name for token in (".lift", ".transport", ".place", ".move_to_pre_place", ".place_descend"))
    phase_min_effort = args.carry_min_gripper_effort if carry_phase and gripper_command < 0.0 else None
    for step in range(actual_steps):
        target = target_fn().to(device=device, dtype=torch.float32).reshape(3)
        action = _ik_position_action(env, target, gripper_command, device)
        err = float(torch.linalg.norm(_grasp_point_pos(robot)[0] - target).item())
        step_out = _step_env(env, action, min_gripper_effort=phase_min_effort)
        if len(step_out) == 5:
            _obs, _rew, terminated, truncated, _infos = step_out
            dones = terminated | truncated
        else:
            _obs, _rew, dones, _infos = step_out
        if recorder is not None:
            recorder.record(env, action)
        min_error = min(min_error, err)
        final_error = err
        if bool(dones[0].item()):
            done_seen = True
        if err <= tolerance and reached_step is None:
            reached_step = step + 1

    target = target_fn().to(device=device, dtype=torch.float32).reshape(3)
    stat = {
        "phase": name,
        "steps": int(actual_steps),
        "requested_steps": int(steps),
        "reached_step": reached_step,
        "min_error_m": round(min_error, 5),
        "final_error_m": round(final_error, 5),
        "done_seen": done_seen,
        "target_grasp_point_w": _round_list(target),
        "grasp_point_w": _round_list(_grasp_point_pos(robot)[0]),
        "joint_pos": _round_list(robot.data.joint_pos[0, :6]),
        **_diagnostic_pose(env),
    }
    trace.append(stat)
    return stat


def _run_diff_ik_state_machine(
    env,
    device: str,
    active_names: list[str],
    recorder: LeRobotV3EpisodeRecorder | None = None,
) -> dict[str, Any]:
    scene = env.unwrapped.scene
    robot = scene["robot"]
    trace: list[dict[str, Any]] = []
    bowl_radius = BOWL_SUCCESS_RADIUS * max(0.1, args.container_radius_scale)

    current_target = lambda: _grasp_point_pos(robot)[0].clone()
    for _ in range(args.settle_steps):
        action = _ik_position_action(env, current_target(), args.gripper_open, device)
        _step_env(env, action)
        if recorder is not None:
            recorder.record(env, action)

    operation_order_base = _ordered_active_names(env, active_names)
    operation_order = operation_order_base * max(1, args.object_cycles)
    trace.append({
        "phase": "operation_order",
        "base_object_order": operation_order_base,
        "object_order": operation_order,
        "order_mode": args.object_order,
        "object_cycles": args.object_cycles,
        "bowl_place_offset_radius": args.bowl_place_offset_radius,
    })

    for placement_index, cube_name in enumerate(operation_order):
        if _cube_inside_bowl(env, cube_name, bowl_radius):
            trace.append({
                "phase": f"{cube_name.lower()}.skip_inside",
                "cycle_index": placement_index // max(1, len(operation_order_base)),
                "inside_bowl": True,
                "cube_w": _round_list(scene[cube_name].data.root_pos_w[0]),
            })
            continue
        cube_start = scene[cube_name].data.root_pos_w[0].clone()
        phase_prefix = cube_name.lower()
        bowl_offset_xy = _bowl_place_offset(device, placement_index)
        grasped = False
        for attempt in range(1, max(1, args.max_grasp_attempts) + 1):
            attempt_prefix = f"{phase_prefix}.attempt{attempt}"
            _ik_phase(
                env,
                device,
                f"{attempt_prefix}.approach",
                _target_from_cube(env, cube_name, args.approach_height),
                args.gripper_open,
                args.approach_steps,
                trace,
                recorder,
                tolerance=args.target_tolerance,
            )
            _ik_phase(
                env,
                device,
                f"{attempt_prefix}.descend",
                _target_from_cube(env, cube_name, args.grasp_z_offset),
                args.gripper_open,
                args.descend_steps,
                trace,
                recorder,
                tolerance=args.target_tolerance,
            )
            grasp_target = _target_from_cube(env, cube_name, args.grasp_z_offset)().clone()
            _ik_phase(
                env,
                device,
                f"{attempt_prefix}.close",
                lambda target=grasp_target: target,
                -1.0,
                args.close_steps,
                trace,
                recorder,
                tolerance=args.target_tolerance,
            )
            _ik_phase(
                env,
                device,
                f"{attempt_prefix}.lift",
                lambda target=grasp_target: target
                + torch.tensor([0.0, 0.0, args.lift_height], device=device, dtype=torch.float32),
                -1.0,
                args.lift_steps,
                trace,
                recorder,
                tolerance=args.target_tolerance,
            )
            grasped = _cube_lifted(env, cube_name)
            # seated 진단: grasp_feasibility 의 held 판정식(grasp point ↔ cube 거리).
            grasp_hold_dist = float(
                torch.linalg.norm(
                    scene[cube_name].data.root_pos_w[0] - _grasp_point_pos(robot)[0]
                ).item()
            )
            trace.append({
                "phase": f"{attempt_prefix}.lift_check",
                "grasped": grasped,
                "grasp_hold_dist_m": round(grasp_hold_dist, 5),
                "cube_w": _round_list(scene[cube_name].data.root_pos_w[0]),
                "joint_pos": _round_list(robot.data.joint_pos[0, :6]),
                **_diagnostic_pose(env),
            })
            if grasped:
                break

            _ik_phase(
                env,
                device,
                f"{attempt_prefix}.retry_open",
                lambda: _grasp_point_pos(robot)[0].clone(),
                args.gripper_open,
                max(args.open_steps, args.command_settle_steps // 2),
                trace,
                recorder,
                tolerance=args.target_tolerance,
            )

        if not grasped:
            cube_end = scene[cube_name].data.root_pos_w[0].clone()
            trace.append({
                "phase": f"{phase_prefix}.result",
                "cube_start_w": _round_list(cube_start),
                "cube_end_w": _round_list(cube_end),
                "inside_bowl": _cube_inside_bowl(env, cube_name, bowl_radius),
                "grasped": False,
            })
            continue

        _ik_phase(
            env,
            device,
            f"{phase_prefix}.transport",
            _target_from_bowl(env, args.transport_height, bowl_offset_xy),
            -1.0,
            args.transport_steps,
            trace,
            recorder,
            tolerance=args.target_tolerance,
        )
        _ik_phase(
            env,
            device,
            f"{phase_prefix}.place",
            _target_from_bowl(env, args.place_height, bowl_offset_xy),
            -1.0,
            args.place_steps,
            trace,
            recorder,
            tolerance=args.target_tolerance,
        )
        release_target = _grasp_point_pos(robot)[0].clone()
        _ik_phase(
            env,
            device,
            f"{phase_prefix}.open",
            lambda target=release_target: target,
            args.gripper_open,
            args.open_steps,
            trace,
            recorder,
            tolerance=args.target_tolerance,
        )
        _ik_phase(
            env,
            device,
            f"{phase_prefix}.final_settle",
            lambda target=release_target: target,
            args.gripper_open,
            args.final_settle_steps,
            trace,
            recorder,
            tolerance=args.target_tolerance,
        )
        release_lift_target = release_target.clone()
        release_lift_target[2] = _bowl_top_z(env) + args.transport_height
        _ik_phase(
            env,
            device,
            f"{phase_prefix}.release_lift",
            _fixed_target(release_lift_target),
            args.gripper_open,
            args.retreat_steps,
            trace,
            recorder,
            tolerance=args.target_tolerance,
        )
        _ik_phase(
            env,
            device,
            f"{phase_prefix}.retreat",
            _target_from_bowl(env, args.transport_height, bowl_offset_xy),
            args.gripper_open,
            args.retreat_steps,
            trace,
            recorder,
            tolerance=args.target_tolerance,
        )

        cube_end = scene[cube_name].data.root_pos_w[0].clone()
        trace.append({
            "phase": f"{phase_prefix}.result",
            "cube_start_w": _round_list(cube_start),
            "cube_end_w": _round_list(cube_end),
            "inside_bowl": _cube_inside_bowl(env, cube_name, bowl_radius),
        })

    final_inside = {name: _cube_inside_bowl(env, name, bowl_radius) for name in active_names}
    return {
        "trace": trace,
        "final_inside": final_inside,
        "placed_and_released": _placed_and_released(env, active_names, bowl_radius),
        "final_gripper": round(float(robot.data.joint_pos[0, 5].item()), 5),
        "final_grasp_point_w": _round_list(_grasp_point_pos(robot)[0]),
        "final_joint_pos": _round_list(robot.data.joint_pos[0, :6]),
        "bowl_w": _round_list(scene[BOWL_NAME].data.root_pos_w[0]),
        "cube_w": {name: _round_list(scene[name].data.root_pos_w[0]) for name in active_names},
    }


def _run_state_machine(
    env,
    device: str,
    active_names: list[str],
    recorder: LeRobotV3EpisodeRecorder | None = None,
    expert_recorder: ExpertTrajectoryRecorder | None = None,
) -> dict[str, Any]:
    scene = env.unwrapped.scene
    robot = scene["robot"]
    trace: list[dict[str, Any]] = []
    fsm_trace: list[dict[str, Any]] = []
    bowl_radius = BOWL_SUCCESS_RADIUS * max(0.1, args.container_radius_scale)
    command = robot.data.joint_pos[0, :6].clone()
    rmpflow_driver = _make_cartesian_driver(env, device)
    if rmpflow_driver is not None:
        rmpflow_driver.reset()

    for _ in range(args.settle_steps):
        zero_action = _joint_position_action(robot, command, torch.zeros(ARM_DOF, device=device), args.gripper_open, device)
        if expert_recorder is not None:
            expert_recorder.record(env, zero_action, "settle")
        _step_env(env, zero_action)
        if recorder is not None:
            recorder.record(env, zero_action)

    operation_order_base = _ordered_active_names(env, active_names)
    operation_order = operation_order_base * max(1, args.object_cycles)
    trace.append({
        "phase": "operation_order",
        "base_object_order": operation_order_base,
        "object_order": operation_order,
        "order_mode": args.object_order,
        "object_cycles": args.object_cycles,
        "bowl_place_offset_radius": args.bowl_place_offset_radius,
        "state_machine_sequence": list(PICK_CUBE_FSM_SEQUENCE),
    })

    for placement_index, cube_name in enumerate(operation_order):
        _append_fsm_event(
            fsm_trace,
            PickCubeFSMState.IDLE,
            cube_name=cube_name,
            next_state=PickCubeFSMState.OPEN_GRIPPER,
            reason="select_next_object",
            placement_index=placement_index,
        )
        if _cube_inside_bowl(env, cube_name, bowl_radius):
            _append_fsm_event(
                fsm_trace,
                PickCubeFSMState.MARK_DONE,
                cube_name=cube_name,
                next_state=PickCubeFSMState.IDLE,
                reason="already_inside_bowl",
                done=True,
                cube_w=_round_list(scene[cube_name].data.root_pos_w[0]),
            )
            trace.append({
                "phase": f"{cube_name.lower()}.skip_inside",
                "cycle_index": placement_index // max(1, len(operation_order_base)),
                "inside_bowl": True,
                "cube_w": _round_list(scene[cube_name].data.root_pos_w[0]),
            })
            continue
        cube_start = scene[cube_name].data.root_pos_w[0].clone()
        phase_prefix = cube_name.lower()
        bowl_offset_xy = _bowl_place_offset(device, placement_index)
        # 40mm 큐브(Cube3/4)는 pick 단계 개방을 더 크게(아래 *_BY_CUBE 상수).
        pick_open = _pick_gripper_open(cube_name)
        pick_descend_gripper = _pick_descend_gripper(cube_name)
        grasped = False
        _FAILED_TILT_DEG.clear()  # tilt 블랙리스트는 큐브 단위(위치 의존이라 이월 무의미)
        for attempt in range(1, max(1, args.max_grasp_attempts) + 1):
            attempt_prefix = f"{phase_prefix}.attempt{attempt}"
            # 큐브가 밀려나 회수 불가(책상 낙하/reach 밖)면 남은 재시도를 버린다(fail-fast).
            oor_reason = _cube_out_of_range(env, cube_name)
            if oor_reason is not None:
                _append_fsm_event(
                    fsm_trace,
                    PickCubeFSMState.MARK_DONE,
                    cube_name=cube_name,
                    attempt=attempt,
                    next_state=PickCubeFSMState.IDLE,
                    reason=f"cube_out_of_range:{oor_reason}",
                    done=False,
                    cube_w=_round_list(scene[cube_name].data.root_pos_w[0]),
                )
                trace.append({
                    "phase": f"{attempt_prefix}.abort_out_of_range",
                    "reason": oor_reason,
                    "cube_w": _round_list(scene[cube_name].data.root_pos_w[0]),
                })
                break
            cube_attempt_start = scene[cube_name].data.root_pos_w[0].clone()
            _append_fsm_event(
                fsm_trace,
                PickCubeFSMState.OPEN_GRIPPER,
                cube_name=cube_name,
                attempt=attempt,
                next_state=PickCubeFSMState.MOVE_TO_PRE_PICK,
                reason="command_open_before_pick",
            )
            open_hold = robot.data.joint_pos[0, :ARM_DOF].clone()
            _hold_joint_target(
                env,
                open_hold,
                pick_open,
                args.open_steps,
                command,
                recorder,
                expert_recorder,
                phase=f"{attempt_prefix}.open_gripper",
            )
            trace.append({
                "phase": f"{attempt_prefix}.open_gripper",
                "state": PickCubeFSMState.OPEN_GRIPPER.value,
                "steps": args.open_steps,
                "grasp_point_w": _round_list(_grasp_point_pos(robot)[0]),
                "joint_pos": _round_list(robot.data.joint_pos[0, :6]),
            })
            _append_fsm_event(
                fsm_trace,
                PickCubeFSMState.MOVE_TO_PRE_PICK,
                cube_name=cube_name,
                attempt=attempt,
                next_state=PickCubeFSMState.ORIENT_WRIST,
                reason="move_above_object_safe_height",
            )
            # 그릇 이탈 경유점(transport_via 의 역방향 대칭): 직전 release 로 grasp point 가
            # 그릇 상공에 있을 때 다음 큐브로 바로 joint-space 보간하면 호가 가라앉으며
            # 팔이 그릇을 쳐 담긴 큐브를 쏟는다(사용자 v15 ep0 영상: 3번째 release 후 4번째
            # 큐브로 이동 중 그릇 타격). 그릇 근처에서 출발하면 중간 지점 상공을 먼저 거친다.
            gp_depart = _grasp_point_pos(robot)[0]
            bowl_xy_dep = scene[BOWL_NAME].data.root_pos_w[0, :2]
            cube_xy_dep = scene[cube_name].data.root_pos_w[0, :2]
            if (
                float(torch.linalg.norm(gp_depart[:2] - bowl_xy_dep).item()) < 0.15
                and float(torch.linalg.norm(gp_depart[:2] - cube_xy_dep).item()) > 0.18
            ):
                via_dep = gp_depart.clone()
                via_dep[:2] = 0.5 * (gp_depart[:2] + cube_xy_dep)
                via_dep[2] = _bowl_top_z(env) + args.transport_height + 0.02
                _phase(
                    env,
                    device,
                    f"{attempt_prefix}.bowl_departure_via",
                    _fixed_target(via_dep),
                    pick_open,
                    max(30, args.approach_steps // 2),
                    trace,
                    command,
                    recorder,
                    expert_recorder,
                    rmpflow_driver,
                    tolerance=args.target_tolerance * 2.0,
                    target_R_fn=_topdown_R_fn(),
                )
            # 접근: 큐브 바로 위(approach_height) 로 top-down(palm-down) 자세로 이동.
            _phase(
                env,
                device,
                f"{attempt_prefix}.move_to_pre_pick",
                _target_from_cube(env, cube_name, args.approach_height),
                pick_open,
                args.approach_steps,
                trace,
                command,
                recorder,
                expert_recorder,
                rmpflow_driver,
                tolerance=args.target_tolerance,
                target_R_fn=_pick_R_fn(),
            )
            _append_fsm_event(
                fsm_trace,
                PickCubeFSMState.ORIENT_WRIST,
                cube_name=cube_name,
                attempt=attempt,
                next_state=PickCubeFSMState.DESCEND,
                reason="topdown_palm_down_enforced",
            )
            _append_fsm_event(
                fsm_trace,
                PickCubeFSMState.DESCEND,
                cube_name=cube_name,
                attempt=attempt,
                next_state=PickCubeFSMState.GRASP,
                reason="descend_vertical_to_floor",
            )
            # 강하 목표를 *진입 시점에 스냅샷*해 고정한다(큐브를 살짝 건드려도 추격 루프에
            # 빠지지 않게). 큐브는 approach 동안 이미 안정화됐다.
            grasp_target = _target_pick(env, cube_name)().clone()
            # 강하: 동일 xy, grasp point 를 큐브 하부(≈바닥) 까지 top-down 으로 수직 강하.
            descend_stat = _phase(
                env,
                device,
                f"{attempt_prefix}.descend",
                _fixed_target(grasp_target),
                pick_descend_gripper,
                args.descend_steps,
                trace,
                command,
                recorder,
                expert_recorder,
                rmpflow_driver,
                tolerance=args.descend_tolerance,
                target_R_fn=_pick_R_fn(),
                grasp_cube_name=cube_name,
            )
            # 착좌 게이트(deterministic): descend 가 목표를 충분히 못 맞췄으면 헛닫지 않고,
            # 큐브 *현재* 위치(밀렸을 수 있음) 기준으로 짧은 보정 descend 를 1회 더 한다.
            if (
                args.grasp_config_mode == "deterministic"
                and rmpflow_driver is None
                and (
                    float(descend_stat.get("final_error_m", 0.0)) > args.descend_tolerance * 1.5
                    # 수평으로 7mm 이상 빗나간 채 닫으면 꼭지점을 집는다 — 보정 후 닫기.
                    or float(descend_stat.get("final_error_h_m", 0.0)) > 0.007
                )
            ):
                grasp_target = _target_pick(env, cube_name)().clone()
                _phase(
                    env,
                    device,
                    f"{attempt_prefix}.descend_fix",
                    _fixed_target(grasp_target),
                    pick_descend_gripper,
                    max(30, args.descend_steps // 2),
                    trace,
                    command,
                    recorder,
                    expert_recorder,
                    rmpflow_driver,
                    tolerance=args.descend_tolerance,
                    target_R_fn=_pick_R_fn(),
                    grasp_cube_name=cube_name,
                )
            # 닫기 전 정렬 게이트: descend(+fix) 가 끝났는데도 grasp point 가 큐브 현 위치
            # 기준 gate 초과로 어긋나 있으면 닫지 않는다 — 어긋난 채 닫으면 못 물 뿐 아니라
            # 닫는 손가락이 큐브를 밀어 다음 attempt 의 조건을 망친다. 닫지 않으면 큐브가
            # 제자리라 재시도가 깨끗하다. 마지막 attempt 는 marginal 이라도 닫는다.
            if args.grasp_misalign_gate > 0.0 and attempt < max(1, args.max_grasp_attempts):
                pre_close_err = float(
                    torch.linalg.norm(
                        _grasp_point_pos(robot)[0] - _target_pick(env, cube_name)()
                    ).item()
                )
                if pre_close_err > args.grasp_misalign_gate:
                    # 이 tilt 의 실행은 계통 실패(결정적 — 재선택하면 동일 미달). 블랙리스트에
                    # 넣어 다음 attempt 가 다른 tilt 를 시도하게 한다.
                    if _LAST_DESCEND_TILT_DEG is not None:
                        _FAILED_TILT_DEG.add(float(_LAST_DESCEND_TILT_DEG))
                    _append_fsm_event(
                        fsm_trace,
                        PickCubeFSMState.GRASP,
                        cube_name=cube_name,
                        attempt=attempt,
                        next_state=PickCubeFSMState.OPEN_GRIPPER,
                        reason="skip_close_misaligned",
                        pre_close_err_m=round(pre_close_err, 5),
                    )
                    trace.append({
                        "phase": f"{attempt_prefix}.skip_close_misaligned",
                        "pre_close_err_m": round(pre_close_err, 5),
                        "grasp_point_w": _round_list(_grasp_point_pos(robot)[0]),
                        "cube_w": _round_list(scene[cube_name].data.root_pos_w[0]),
                    })
                    continue
            _append_fsm_event(
                fsm_trace,
                PickCubeFSMState.GRASP,
                cube_name=cube_name,
                attempt=attempt,
                next_state=PickCubeFSMState.LIFT,
                reason="close_gripper_top_down",
            )
            # 잡기: 닫는 동안 top-down DLS 를 유지(고정 grasp_target)해 grasp point 를 큐브에 계속
            # 눌러붙인 채 그리퍼를 닫는다. 검증된 20260605 close 메커니즘(닫는 중 IK 지속)과 동일.
            grasp_min_steps = _gripper_transition_step_count(
                float(command[5].item()), args.gripper_closed, step=args.max_gripper_step_delta
            ) + max(0, args.grasp_settle_steps)
            _phase(
                env,
                device,
                f"{attempt_prefix}.grasp",
                lambda target=grasp_target: target,
                args.gripper_closed,
                _grasp_phase_steps(command, args.gripper_closed),
                trace,
                command,
                recorder,
                expert_recorder,
                rmpflow_driver,
                tolerance=args.target_tolerance,
                target_R_fn=_pick_R_fn(),
                min_steps=grasp_min_steps,
                grasp_cube_name=cube_name,
            )
            _append_fsm_event(
                fsm_trace,
                PickCubeFSMState.LIFT,
                cube_name=cube_name,
                attempt=attempt,
                next_state=PickCubeFSMState.MOVE_TO_PRE_PLACE,
                reason="lift_to_safe_height",
            )
            _phase(
                env,
                device,
                f"{attempt_prefix}.lift",
                lambda target=grasp_target: target + torch.tensor(
                    [0.0, 0.0, args.lift_height], device=device, dtype=torch.float32
                ),
                args.gripper_closed,
                args.lift_steps,
                trace,
                command,
                recorder,
                expert_recorder,
                rmpflow_driver,
                tolerance=args.target_tolerance,
                target_R_fn=_pick_R_fn(),
            )
            grasped = _cube_lifted(env, cube_name)
            # seated 진단: grasp_feasibility 의 held 판정식(grasp point ↔ cube 거리).
            grasp_hold_dist = float(
                torch.linalg.norm(
                    scene[cube_name].data.root_pos_w[0] - _grasp_point_pos(robot)[0]
                ).item()
            )
            # 변위 진단: 잘 집으면 큐브가 거의 안 움직인다 — 밀침 정량화(영상 관찰 검증용).
            cube_moved = float(
                torch.linalg.norm(
                    scene[cube_name].data.root_pos_w[0, :2] - cube_attempt_start[:2]
                ).item()
            )
            trace.append({
                "phase": f"{attempt_prefix}.lift_check",
                "grasped": grasped,
                "grasp_hold_dist_m": round(grasp_hold_dist, 5),
                "cube_moved_m": round(cube_moved, 5),
                "cube_w": _round_list(scene[cube_name].data.root_pos_w[0]),
                "joint_pos": _round_list(robot.data.joint_pos[0, :6]),
                **_diagnostic_pose(env),
            })
            if grasped:
                _append_fsm_event(
                    fsm_trace,
                    PickCubeFSMState.LIFT,
                    cube_name=cube_name,
                    attempt=attempt,
                    next_state=PickCubeFSMState.MOVE_TO_PRE_PLACE,
                    reason="lift_check_passed",
                    grasped=True,
                    cube_w=_round_list(scene[cube_name].data.root_pos_w[0]),
                )
                break

            # 닫았는데 빈손 = 이 tilt 의 grasp 자세가 이 큐브 위치에서 물 수 없다는 뜻
            # (결정적 재선택 = 동일 빈손, v12 ep13: 15° 3연발). 다음 attempt 는 다른 tilt 로.
            if _LAST_DESCEND_TILT_DEG is not None:
                _FAILED_TILT_DEG.add(float(_LAST_DESCEND_TILT_DEG))
            _append_fsm_event(
                fsm_trace,
                PickCubeFSMState.LIFT,
                cube_name=cube_name,
                attempt=attempt,
                next_state=PickCubeFSMState.OPEN_GRIPPER,
                reason="lift_check_failed_retry",
                grasped=False,
                cube_w=_round_list(scene[cube_name].data.root_pos_w[0]),
            )

            # 재시도: 바닥을 쓸지 않도록 현재 위치에서 *수직으로* 들어올린 뒤(그리퍼 동시 개방)
            # 다음 attempt 의 move_to_pre_pick 이 위에서 다시 수직 강하한다.
            retry_up_target = _grasp_point_pos(robot)[0].clone()
            cube_now_z = float(scene[cube_name].data.root_pos_w[0, 2].item())
            retry_up_target[2] = cube_now_z + args.approach_height
            _phase(
                env,
                device,
                f"{attempt_prefix}.retry_lift",
                _fixed_target(retry_up_target),
                pick_open,
                args.lift_steps,
                trace,
                command,
                recorder,
                expert_recorder,
                rmpflow_driver,
                tolerance=args.target_tolerance,
                target_R_fn=_pick_R_fn(),
            )

        if not grasped:
            _move_to_idle_home(
                env,
                device,
                command,
                trace,
                recorder,
                expert_recorder,
                phase=f"{phase_prefix}.idle_home_after_failed_pick",
            )
            cube_end = scene[cube_name].data.root_pos_w[0].clone()
            _append_fsm_event(
                fsm_trace,
                PickCubeFSMState.MARK_DONE,
                cube_name=cube_name,
                next_state=PickCubeFSMState.IDLE,
                reason="max_grasp_attempts_exhausted",
                done=False,
                cube_w=_round_list(cube_end),
            )
            trace.append({
                "phase": f"{phase_prefix}.result",
                "cube_start_w": _round_list(cube_start),
                "cube_end_w": _round_list(cube_end),
                "inside_bowl": _cube_inside_bowl(env, cube_name, bowl_radius),
                "grasped": False,
            })
            continue

        _append_fsm_event(
            fsm_trace,
            PickCubeFSMState.MOVE_TO_PRE_PLACE,
            cube_name=cube_name,
            next_state=PickCubeFSMState.PLACE_DESCEND,
            reason="move_above_bowl_safe_height",
        )
        # release 높이는 *미리* 확정한다(stack 보정 포함). 과거엔 transport_via(+0.14) →
        # pre_place(+0.12) → place_descend(+0.12+0.025×stack) 로 단계별 z 가 갈지자라,
        # 그릇 위에서 "내려갔다 멈칫, 올라갔다 멈칫, 다시 내려와 떨굼" 하는 버벅임이
        # 보였다(사용자 v15 ep0 영상 관찰). 이제 경유점부터 release 높이로 단조 접근한다.
        stack_level = sum(1 for name in active_names if _cube_inside_bowl(env, name, bowl_radius))
        place_height = args.place_height + args.stack_place_height_increment * stack_level
        # 이송 경유점: joint-space 보간 경로가 호를 그리며 가라앉아 스윙 중 그릇을 칠 수
        # 있다(ep4 영상에서 그릇 엎음 확인). 멀리 옮길 때는 중간 지점 상공을 먼저 거친다.
        gp_now = _grasp_point_pos(robot)[0]
        bowl_xy_now = scene[BOWL_NAME].data.root_pos_w[0, :2]
        if float(torch.linalg.norm(gp_now[:2] - bowl_xy_now).item()) > 0.18:
            via = gp_now.clone()
            via[:2] = 0.5 * (gp_now[:2] + bowl_xy_now)
            via[2] = _bowl_top_z(env) + place_height + 0.02
            _phase(
                env,
                device,
                f"{phase_prefix}.transport_via",
                _fixed_target(via),
                args.gripper_closed,
                max(30, args.transport_steps // 2),
                trace,
                command,
                recorder,
                expert_recorder,
                rmpflow_driver,
                tolerance=args.target_tolerance * 2.0,
                target_R_fn=_topdown_R_fn(),
            )
        # 수송: 그릇 위 release 지점(stack 보정 높이)으로 한 번에 — top-down(palm-down)
        # 자세 유지 직선 이동. 별도 place_descend 단계는 제거(같은 높이로 또 solve 하며
        # 멈칫하던 원인). release 는 이 지점에서 바로 한다.
        _phase(
            env,
            device,
            f"{phase_prefix}.move_to_pre_place",
            _target_from_bowl(env, place_height, bowl_offset_xy),
            args.gripper_closed,
            args.transport_steps,
            trace,
            command,
            recorder,
            expert_recorder,
            rmpflow_driver,
            tolerance=args.target_tolerance,
            target_R_fn=_topdown_R_fn(),
        )
        _append_fsm_event(
            fsm_trace,
            PickCubeFSMState.PLACE_DESCEND,
            cube_name=cube_name,
            next_state=PickCubeFSMState.RELEASE,
            reason="release_at_pre_place_height",
        )
        trace[-1]["stack_level_before_place"] = stack_level
        trace[-1]["place_height"] = round(float(place_height), 5)
        # release 도달 게이트: pre_place 가 stall/미달로 끊겨도 무조건 release 하면
        # 그릇 코앞 책상에 떨군다(사용자 v17 ep0 영상: 1번째 큐브). 그릇 중심에서
        # 수평 4cm 초과로 어긋나 있으면 같은 목표로 1회 재이동 후 release 한다.
        place_target_fn = _target_from_bowl(env, place_height, bowl_offset_xy)
        place_err_h = float(
            torch.linalg.norm((_grasp_point_pos(robot)[0] - place_target_fn())[:2]).item()
        )
        if place_err_h > 0.04:
            _phase(
                env,
                device,
                f"{phase_prefix}.place_fix",
                place_target_fn,
                args.gripper_closed,
                max(30, args.transport_steps // 2),
                trace,
                command,
                recorder,
                expert_recorder,
                rmpflow_driver,
                tolerance=args.target_tolerance,
                target_R_fn=_topdown_R_fn(),
            )
        # release 전 정지 대기: 슬루 감속 진동 중에 열면 큐브가 관성으로 '던져진다'
        # (사용자 v17 ep0 영상: 3·4번째 큐브 반동 release, 4번째 빗나감). 팔이 잠잠해질
        # 때까지만 짧게 기다린다(최대 10 step ≈ 0.33s — 보통 2~4 step 에 정지).
        settle_hold = robot.data.joint_pos[0, :ARM_DOF].clone()
        for _ in range(15):
            if float(torch.max(torch.abs(robot.data.joint_vel[0, :ARM_DOF])).item()) < 0.25:
                break
            _hold_joint_target(
                env, settle_hold, args.gripper_closed, 1, command, recorder, expert_recorder,
                phase=f"{phase_prefix}.pre_release_settle",
            )
        # 열 때는 마지막 joint target을 유지한다. 위치 IK가 그릇 안 큐브를 다시
        # 추적하며 건드리지 않도록, release 동안 관절 목표를 고정한다.
        joint_hold = robot.data.joint_pos[0, :ARM_DOF].clone()
        _append_fsm_event(
            fsm_trace,
            PickCubeFSMState.RELEASE,
            cube_name=cube_name,
            next_state=PickCubeFSMState.MARK_DONE,
            reason="open_gripper_and_wait",
        )
        _hold_joint_target(
            env,
            joint_hold,
            args.gripper_open,
            args.open_steps,
            command,
            recorder,
            expert_recorder,
            phase=f"{phase_prefix}.release",
        )
        trace.append({
            "phase": f"{phase_prefix}.release",
            "state": PickCubeFSMState.RELEASE.value,
            "steps": args.open_steps,
            "grasp_point_w": _round_list(_grasp_point_pos(robot)[0]),
            "joint_pos": _round_list(robot.data.joint_pos[0, :6]),
        })
        _hold_joint_target(
            env,
            joint_hold,
            args.gripper_open,
            args.final_settle_steps,
            command,
            recorder,
            expert_recorder,
            phase=f"{phase_prefix}.final_settle",
        )
        # release 후 그릇 안 큐브를 건드리지 않게 *수직 위로만* 빠져나온다(top-down 유지).
        # max(): stack 보정으로 release 지점이 transport_height 보다 높을 때 아래로
        # 내려가는 역행(버벅임)을 막는다.
        release_lift_target = _grasp_point_pos(robot)[0].clone()
        release_lift_target[2] = _bowl_top_z(env) + max(
            args.transport_height, place_height + 0.03
        )
        _phase(
            env,
            device,
            f"{phase_prefix}.release_lift",
            _fixed_target(release_lift_target),
            args.gripper_open,
            args.retreat_steps,
            trace,
            command,
            recorder,
            expert_recorder,
            rmpflow_driver,
            tolerance=args.target_tolerance,
            target_R_fn=_topdown_R_fn(),
        )
        # idle_home 복귀는 제거(성공 경로). retreat 직후 바로 다음 큐브 approach 로 이어져
        # 동선·시간을 줄인다. 실패 경로에서만 idle_home 으로 안전 복귀한다.

        cube_end = scene[cube_name].data.root_pos_w[0].clone()
        inside_bowl = _cube_inside_bowl(env, cube_name, bowl_radius)
        _append_fsm_event(
            fsm_trace,
            PickCubeFSMState.MARK_DONE,
            cube_name=cube_name,
            next_state=PickCubeFSMState.IDLE,
            reason="object_cycle_complete",
            done=inside_bowl,
            cube_w=_round_list(cube_end),
        )
        trace.append({
            "phase": f"{phase_prefix}.result",
            "cube_start_w": _round_list(cube_start),
            "cube_end_w": _round_list(cube_end),
            "inside_bowl": inside_bowl,
        })

    final_inside = {
        name: _cube_inside_bowl(env, name, bowl_radius)
        for name in active_names
    }
    _append_fsm_event(
        fsm_trace,
        PickCubeFSMState.ALL_DONE,
        reason="operation_order_exhausted",
        final_inside=final_inside,
    )
    return {
        "trace": trace,
        "fsm_trace": fsm_trace,
        "fsm_state_sequence": list(PICK_CUBE_FSM_SEQUENCE),
        "final_inside": final_inside,
        "placed_and_released": _placed_and_released(env, active_names, bowl_radius),
        "final_gripper": round(float(robot.data.joint_pos[0, 5].item()), 5),
        "final_grasp_point_w": _round_list(_grasp_point_pos(robot)[0]),
        "final_joint_pos": _round_list(robot.data.joint_pos[0, :6]),
        "bowl_w": _round_list(scene[BOWL_NAME].data.root_pos_w[0]),
        "cube_w": {
            name: _round_list(scene[name].data.root_pos_w[0])
            for name in active_names
        },
    }


def main() -> None:
    env = None
    recorder: LeRobotV3EpisodeRecorder | None = None
    expert_recorder: ExpertTrajectoryRecorder | None = None
    try:
        device: str = args.device
        env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
        env_cfg.seed = args.seed
        apply_curriculum(
            env_cfg,
            active_objects=args.active_objects,
            object_radius_scale=args.object_radius_scale,
            container_angle_scale=args.container_angle_scale,
            container_radius_scale=args.container_radius_scale,
        )
        # State-machine 검증 중에는 중간 성공 termination으로 자동 reset되지 않게
        # 끄고, 마지막에 gripper release까지 포함해 직접 판정한다.
        env_cfg.terminations.success = None
        if args.controller_mode == "diff_ik":
            if args.expert_dataset_pt is not None:
                raise ValueError("--expert_dataset_pt is only supported with --controller_mode joint_fk")
            env_cfg.actions = PickCubeDiffIkActionsCfg()
        min_close_steps = math.ceil(
            abs(args.gripper_open - args.gripper_closed) / max(abs(args.max_gripper_step_delta), 1e-6)
        )
        effective_close_steps = max(
            args.close_steps,
            min_close_steps + max(0, args.grasp_settle_steps),
        )
        total_steps = (
            args.settle_steps
            + args.active_objects
            * max(1, args.object_cycles)
            * max(1, args.max_grasp_attempts)
            * (
                args.open_steps
                + args.approach_steps
                + args.descend_steps
                + effective_close_steps
                + args.lift_steps
                + args.command_settle_steps * 4
            )
            + args.active_objects
            * max(1, args.object_cycles)
            * (
                args.transport_steps
                + args.place_steps
                + args.open_steps
                + args.final_settle_steps
                + args.retreat_steps  # release_lift 수직 이탈
                + args.retreat_steps
                + args.idle_home_steps
                + args.command_settle_steps * 2
            )
            + 120
        )
        estimated_episode_length_s = max(
            total_steps * env_cfg.sim.dt * env_cfg.decimation + 5.0,
            args.record_seconds + 30.0,
            # 4-cube scripted rollouts can legitimately run for several
            # minutes because command slew limiting is part of the proof.
            150.0 * args.active_objects * max(1, args.max_grasp_attempts),
            180.0,
        )
        if args.episode_length_s is not None:
            estimated_episode_length_s = max(estimated_episode_length_s, args.episode_length_s)
        env_cfg.episode_length_s = max(
            env_cfg.episode_length_s,
            estimated_episode_length_s,
        )
        if args.dataset_dir is not None and not args.no_videos:
            add_pick_cube_cameras(env_cfg.scene)
        if args.review_video_dir is not None:
            # 리뷰용 뷰어 카메라(근거리 사선 부감) 주입 — InteractiveScene 이
            # scene_cfg.__dict__ 를 순회하므로 동적 속성이 센서로 등록된다.
            review_eye = tuple(float(v) for v in args.review_cam_eye.split(","))
            review_target = tuple(float(v) for v in args.review_cam_target.split(","))
            setattr(
                env_cfg.scene,
                "review_camera",
                _pinhole_camera_cfg(
                    "{ENV_REGEX_NS}/ReviewCamera",
                    review_eye,
                    _look_at_quat_world(review_eye, review_target),
                    args.review_cam_focal,
                    focus_distance=0.8,
                    clipping_range=(0.05, 6.0),
                ),
            )
            args.review_video_dir.mkdir(parents=True, exist_ok=True)

        env = gym.make(args.task, cfg=env_cfg)
        env.reset()
        if args.dataset_dir is not None:
            recorder = LeRobotV3EpisodeRecorder(
                args.dataset_dir,
                seconds=args.record_seconds,
                overwrite=args.overwrite_dataset,
                videos=not args.no_videos,
            )
        if args.expert_dataset_pt is not None:
            expert_recorder = ExpertTrajectoryRecorder(args.expert_dataset_pt)
        if args.controller_mode == "diff_ik":
            robot = env.unwrapped.scene["robot"]
            zero_action = _ik_position_action(env, _grasp_point_pos(robot)[0], args.gripper_open, device)
        else:
            zero_action = torch.zeros((1, 6), device=device)
            zero_action[0, 5] = args.gripper_open
        for _ in range(max(0, args.warmup_steps)):
            _step_env(env, zero_action)
        if args.review_pose_check:
            # 구도 확인: 렌더 안정화 후 정지 프레임만 저장하고 종료.
            for _ in range(10):
                _step_env(env, zero_action)
            frame = _camera_rgb_u8(env.unwrapped.scene["review_camera"])
            pose_path = args.review_video_dir / "pose_check.png"
            imageio.imwrite(str(pose_path), frame)
            print(f"[review] pose check saved: {pose_path}", flush=True)
            env.close()
            env = None
            return
        active_names = CUBE_NAMES[: args.active_objects]
        if args.num_episodes > 1:
            # 신뢰성 sweep: 한 세션에서 N회 reset+SM. dataset/expert 기록은 비활성.
            episodes: list[dict[str, Any]] = []
            for ep in range(args.num_episodes):
                if ep > 0:
                    env.reset()
                    for _ in range(max(0, args.warmup_steps)):
                        _step_env(env, zero_action)
                # reach 검증: SM 실행 전 spawn(초기) 큐브 xy 를 기록한다. 실패 큐브의 spawn 위치를
                # 모으면 어느 영역이 SO-101 reach 한계 너머인지(스폰 범위 조정 근거) 알 수 있다.
                _scene = env.unwrapped.scene
                spawn_xy = {
                    n: [round(float(c), 4) for c in _scene[n].data.root_pos_w[0, :2].tolist()]
                    for n in active_names
                }
                global _REVIEW_WRITER, _STEP_COUNTER
                review_path: Path | None = None
                if args.review_video_dir is not None:
                    review_path = args.review_video_dir / f"ep{ep:02d}.mp4"
                    _REVIEW_WRITER = imageio.get_writer(str(review_path), fps=FPS)
                ep_steps_before = _STEP_COUNTER
                try:
                    ep_result = _run_state_machine(env, device, active_names, None, None)
                finally:
                    if _REVIEW_WRITER is not None:
                        _REVIEW_WRITER.close()
                        _REVIEW_WRITER = None
                ep_duration_s = round((_STEP_COUNTER - ep_steps_before) / float(FPS), 2)
                inside = ep_result.get("final_inside", {})
                n_inside = int(sum(1 for v in inside.values() if v))
                if review_path is not None and review_path.exists():
                    suffix = "ok" if n_inside >= len(active_names) else "fail"
                    final_path = review_path.with_name(f"ep{ep:02d}_{suffix}.mp4")
                    review_path.rename(final_path)
                    print(f"[review] saved {final_path}", flush=True)
                # 실패 에피소드는 grasp 관련 phase 의 압축 진단을 남긴다(원인 추적용 —
                # 전체 trace 는 sweep 에서 버리므로 여기서만 추출).
                fail_diag: list[dict[str, Any]] | None = None
                if n_inside < len(active_names):
                    fail_diag = []
                    _diag_keys = (
                        "phase", "grasped", "grasp_hold_dist_m", "inside_bowl",
                        "det_tilt_deg", "det_pos_err_m", "det_pos_err_h_m", "det_err_perp_m",
                        "det_tilt_pen", "det_roll_err_deg",
                        "det_fallback", "det_grasp_hold_pose", "jacobian_refine_steps",
                        "cube_moved_m", "pre_close_err_m", "reason",
                        "det_jaw_min_z", "det_fix_min_z",
                        "planned_error_m", "final_error_m", "early_exit_step", "cube_w",
                    )
                    for entry in ep_result.get("trace", []):
                        ph = entry.get("phase", "")
                        if ph.endswith(
                            (
                                ".descend", ".descend_fix", ".grasp", ".lift_check", ".result",
                                ".skip_close_misaligned", ".abort_out_of_range", ".place_fix",
                            )
                        ):
                            fail_diag.append({k: entry[k] for k in _diag_keys if k in entry})
                episodes.append({
                    "episode": ep,
                    "spawn_xy": spawn_xy,
                    "duration_s": ep_duration_s,
                    "bowl_tipped": _bowl_tipped(env),
                    "final_inside": inside,
                    "n_inside": n_inside,
                    "all_placed": bool(ep_result.get("placed_and_released", False)),
                    "bowl_w": ep_result.get("bowl_w"),
                    "cube_w": ep_result.get("cube_w"),
                    "fail_diag": fail_diag,
                })
                print(f"[sweep] ep {ep}: n_inside={n_inside} inside={inside} spawn={spawn_xy}", flush=True)
            n_eps = len(episodes)
            all4 = int(sum(1 for e in episodes if e["n_inside"] >= args.active_objects))
            per_cube = {
                name: round(sum(1 for e in episodes if e["final_inside"].get(name)) / n_eps, 3)
                for name in active_names
            }
            sweep_payload = {
                "task_id": "TA.CUBE.STATE_MACHINE",
                "task": args.task,
                "status": "passed" if all4 == n_eps else "partial",
                "sweep": True,
                "num_episodes": n_eps,
                "active_objects": args.active_objects,
                "object_radius_scale": args.object_radius_scale,
                "container_angle_scale": args.container_angle_scale,
                "controller_mode": args.controller_mode,
                "grasp_config_mode": args.grasp_config_mode,
                "grasp_pick_offset": args.grasp_pick_offset,
                "grasp_lateral_offset": args.grasp_lateral_offset,
                "grasp_tilt_weight": args.grasp_tilt_weight,
                "object_cycles": args.object_cycles,
                "all4_success_rate": round(all4 / n_eps, 3),
                "mean_inside": round(sum(e["n_inside"] for e in episodes) / n_eps, 3),
                "per_cube_success_rate": per_cube,
                "episodes": episodes,
            }
            env.close()
            env = None
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(sweep_payload, indent=2, ensure_ascii=False), encoding="utf-8")
            print(json.dumps(sweep_payload, indent=2, ensure_ascii=False))
            return
        if args.controller_mode == "diff_ik":
            result = _run_diff_ik_state_machine(env, device, active_names, recorder)
        else:
            result = _run_state_machine(env, device, active_names, recorder, expert_recorder)
        passed = bool(result["placed_and_released"])
        common_controller = {
            "mode": args.controller_mode,
            "end_effector": "jaw + quat(jaw) * (-0.021, -0.070, 0.020)"
            if args.control_point == "jaw_offset"
            else CONTROL_POINT_NAME,
            "control_point": args.control_point,
            "command_settle_steps": args.command_settle_steps,
            "close_steps": args.close_steps,
            "grasp_settle_steps": args.grasp_settle_steps,
            "retreat_steps": args.retreat_steps,
            "idle_home_steps": args.idle_home_steps,
            "max_grasp_attempts": args.max_grasp_attempts,
            "stack_place_height_increment": args.stack_place_height_increment,
            "object_order": args.object_order,
            "object_cycles": args.object_cycles,
            "bowl_place_offset_radius": args.bowl_place_offset_radius,
            "gripper_open": args.gripper_open,
            "dynamic_gripper_effort": not args.disable_dynamic_gripper_effort,
            "min_gripper_effort": args.min_gripper_effort,
            "carry_min_gripper_effort": args.carry_min_gripper_effort,
            "fsm_state_sequence": list(PICK_CUBE_FSM_SEQUENCE),
        }
        if args.controller_mode == "diff_ik":
            controller_payload = {
                **common_controller,
                "type": "differential_ik_relative_position_binary_gripper",
                "body_name": "jaw",
                "body_offset": list(JAW_GRASP_OFFSET),
                "relative_mode": True,
                "max_cartesian_step_m": args.diff_ik_step_size,
                "ik_method": "dls",
                "ik_lambda": 0.04,
                "gripper_closed": args.ik_gripper_closed,
                "close_action_command": -1.0,
                "open_action_command": args.gripper_open,
            }
        elif args.controller_mode == "rmpflow":
            controller_payload = {
                **common_controller,
                "type": "rmpflow_position_only_ik_guided_joint_target",
                "urdf_file": str(RMPFLOW_URDF_PATH),
                "collision_file": str(RMPFLOW_DESCRIPTOR_PATH),
                "config_file": str(RMPFLOW_CONFIG_PATH),
                "frame_name": "gripper_frame_link",
                "target_orientation": None,
                "cspace_attractor": "Lula IK solution for the current position-only target",
                "execution_action": "slew_limited_joint_position",
                "internal_rollout_steps": args.rmpflow_internal_rollout_steps,
                "jacobian_refine": not args.disable_rmpflow_jacobian_refine,
                "max_arm_step_delta_rad": args.max_arm_step_delta,
                "max_gripper_step_delta_rad": args.max_gripper_step_delta,
                "gripper_closed": args.ik_gripper_closed,
            }
        else:
            controller_payload = {
                **common_controller,
                "type": (
                    "deterministic_fd_dls_waypoint_joint_position"
                    if args.grasp_config_mode == "deterministic"
                    else "random_fk_waypoint_joint_position"
                ),
                "grasp_config_mode": args.grasp_config_mode,
                "det_tilt_ladder_deg": list(_DET_TILT_LADDER_DEG),
                "fk_samples": args.fk_samples,
                "continuity_weight": args.continuity_weight,
                "grasp_tilt_weight": args.grasp_tilt_weight,
                "jacobian_refine": args.enable_jacobian_refine and not args.disable_jacobian_refine,
                "ik_damping": args.ik_damping,
                "ik_gain": args.ik_gain,
                "max_joint_delta": args.max_joint_delta,
                "max_arm_step_delta_rad": args.max_arm_step_delta,
                "max_gripper_step_delta_rad": args.max_gripper_step_delta,
                "gripper_closed": args.gripper_closed,
            }
        payload = {
            "task_id": "TA.CUBE.STATE_MACHINE",
            "task": args.task,
            "status": "passed" if passed else "failed",
            "active_objects": args.active_objects,
            "object_radius_scale": args.object_radius_scale,
            "container_angle_scale": args.container_angle_scale,
            "container_radius_scale": args.container_radius_scale,
            "controller": controller_payload,
            **result,
        }
        if recorder is not None:
            payload["dataset"] = recorder.finalize(task_name=CUBE_TASK_NAME, run_result=payload)
            recorder = None
        if expert_recorder is not None:
            payload["expert_dataset"] = expert_recorder.finalize(run_result=payload)
            expert_recorder = None
        env.close()
        env = None
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        if not passed:
            sys.exit(1)
    except Exception as exc:
        payload = {
            "task_id": "TA.CUBE.STATE_MACHINE",
            "task": args.task,
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        sys.exit(1)
    finally:
        if recorder is not None:
            recorder.close()
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        simulation_app.close()


if __name__ == "__main__":
    main()
