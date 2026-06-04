"""Pen Pick-and-Place task configuration — pure Isaac Lab 2.3.2."""

from __future__ import annotations

import math

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from sim_to_real.assets.scenes.pen_desk import PEN_DESK_CFG, PEN_DESK_USD_PATH, ROBOT_USD_PATH
from sim_to_real.utils.constant import PEN_CUP_NAME, PEN_NAMES
from sim_to_real.utils.domain_randomization import (
    randomize_object_in_ellipse,
    randomize_object_on_arc,
)

from . import mdp as task_mdp


# World-frame (x, y) of the pen cup at scene authoring time.
# PEN_CUP_LOCAL=(0, 0.40) + SCENE_OFFSET=(2.2, -0.57) = (2.2, -0.17)
PEN_CUP_CENTER_XY: tuple[float, float] = (2.2, -0.17)
PEN_CUP_SUCCESS_RADIUS: float = 0.05

# SO-101 joint order (North Star contract — must not change)
SO101_JOINT_ORDER: list[str] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Robot base position: SCENE_OFFSET(2.2, -0.57) + scene-local robot offset(0, -0.04).
#
# z 정합: 책상 상판(DeskTop) 윗면은 world z=0.92 (SCENE_OFFSET.z 0.92 + center
# -0.02 + half-thickness 0.02). 그러나 so101_follower.usd 의 articulation root
# 원점(z=0)은 베이스 바닥이 아니다 — 베이스 최하단 지오메트리가 local z=+0.0301
# 에 있다(USD bbox 측정). 따라서 원점을 0.92 에 두면 팔 전체가 ~3 cm 떠버린다
# (사용자가 보고한 "로봇이 책상 위에 떠 있는" 현상). 베이스 판이 상판에 닿도록
# 원점을 내린다:  robot_z = desk_top(0.92) - base_min_z(0.0301) ≈ 0.889.
_ROBOT_POS = (2.2, -0.61, 0.889)
# Identity rotation; articulation USD already faces the desk objects.
_ROBOT_ROT = (0.0, 0.0, 0.0, 1.0)  # (w, x, y, z)


def _yaw_quat(degrees: float) -> tuple[float, float, float, float]:
    half = math.radians(degrees) * 0.5
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _dot3(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross3(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _normalize3(v: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(_dot3(v, v))
    if norm < 1e-9:
        raise ValueError(f"Cannot normalize near-zero vector: {v!r}")
    return (v[0] / norm, v[1] / norm, v[2] / norm)


def _quat_from_matrix(
    m: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
) -> tuple[float, float, float, float]:
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return (
            0.25 * s,
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s,
        )
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        return (
            (m[2][1] - m[1][2]) / s,
            0.25 * s,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
        )
    if m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        return (
            (m[0][2] - m[2][0]) / s,
            (m[0][1] + m[1][0]) / s,
            0.25 * s,
            (m[1][2] + m[2][1]) / s,
        )
    s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
    return (
        (m[1][0] - m[0][1]) / s,
        (m[0][2] + m[2][0]) / s,
        (m[1][2] + m[2][1]) / s,
        0.25 * s,
    )


def _look_at_quat_world(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[float, float, float, float]:
    """Quaternion for Isaac Lab camera world convention: forward +X, up +Z."""

    forward = _normalize3((target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]))
    up_hint = _normalize3(up)
    up_axis_raw = (
        up_hint[0] - _dot3(up_hint, forward) * forward[0],
        up_hint[1] - _dot3(up_hint, forward) * forward[1],
        up_hint[2] - _dot3(up_hint, forward) * forward[2],
    )
    if _dot3(up_axis_raw, up_axis_raw) < 1e-9:
        up_axis_raw = (0.0, 1.0, 0.0)
    up_axis = _normalize3(up_axis_raw)
    right_axis = _cross3(up_axis, forward)
    matrix = (
        (forward[0], right_axis[0], up_axis[0]),
        (forward[1], right_axis[1], up_axis[1]),
        (forward[2], right_axis[2], up_axis[2]),
    )
    return _quat_from_matrix(matrix)


_PEN_INIT_STATES = {
    "PenWhite": ((2.05, -0.35, 0.9347), _yaw_quat(25.0)),
    "PenGray": ((2.35, -0.35, 0.9347), _yaw_quat(-30.0)),
    "PenBlack": ((2.25, -0.31, 0.9347), _yaw_quat(60.0)),
    "PenBlue": ((2.15, -0.31, 0.9347), _yaw_quat(-10.0)),
}
_PEN_CUP_INIT_STATE = ((2.2, -0.17, 0.926), _yaw_quat(0.0))

# ---------------------------------------------------------------------------
# 카메라 리그 상수 — North Star 계약: observation.images.{top,front,wrist}
#   · 모두 640×480 (W×H) RGB, update_period=0.0 (render_interval 마다 갱신)
#   · 포즈/FOV 는 실제 데이터셋 프레임(outputs/ta3_camera_refs/{top,front,wrist}
#     _t*.png)을 기준으로 튜닝. docs/pics 사무실 사진은 물리 배치 맥락일 뿐 —
#     특히 top 카메라는 사무실 사진보다 더 높게 물리 조정되었으므로 사진 포즈가
#     아니라 top 비디오 구도(로봇 베이스가 프레임 하단, 매트/컵이 위로 넓게)에
#     맞춘다.
#   · world frame 절대 좌표(convention="world", forward +X / up +Z). num_envs=1
#     smoke 기준. 멀티-env(TC.2)에서는 env-relative 좌표로 전환 필요.
# ---------------------------------------------------------------------------

# top: 로봇 뒤(-y)·높은 곳에서 내려보는 급경사 oblique. 로봇 베이스가 하단,
# 펜/컵/매트가 위로 펼쳐진다. 사무실 사진보다 높게 올린 실제 top 비디오에 맞춤.
_TOP_CAMERA_POS = (2.2, -1.12, 1.88)
_TOP_CAMERA_TARGET = (2.14, -0.15, 0.92)
_TOP_CAMERA_FOCAL = 16.0

# front: 로봇 전면(+y 를 바라봄)에 근접 장착. 베이스 바로 앞, 책상에서 ~8 cm
# 높이의 낮은 시점에서 작업 영역을 가로질러 본다. 펜이 전경, 컵이 중앙 —
# observation.images.front 와 정합. (기존 (1.46,0.16,0.99) 는 컵 너머 측면에서
# 거꾸로 보던 분리형 카메라라 잘못됨 → 로봇 전면 장착으로 교정.)
_FRONT_CAMERA_POS = (2.48, -0.56, 0.965)
_FRONT_CAMERA_TARGET = (2.16, -0.03, 0.955)
_FRONT_CAMERA_FOCAL = 14.0

# wrist: gripper 링크에 강결합되어 팔을 따라 움직인다. gripper 접근축(jaw/그립
# 지점 방향)을 따라 매트를 근접·광각으로 내려본다 — observation.images.wrist.
# gripper-local 축은 rest 자세에서 world 로 localX->+Z, localY->+X, localZ->+Y.
# gripper->jaw(접근/손가락 축) 방향을 gripper-local 좌표로 표현:
_WRIST_CAM_LOCAL_POS = (-0.040, 0.030, -0.120)
# gripper parent 회전을 역산해, rest 자세에서 camera world pos≈(2.149,-0.213,1.116)
# 기준 mat/cup 방향 world target≈(2.10,-0.13,0.93)을 바라보도록 한 local rot.
_WRIST_CAM_LOCAL_ROT = (0.2280, 0.0630, 0.9365, 0.2588)
_WRIST_CAMERA_FOCAL = 10.0


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


@configclass
class PickPenSceneCfg(InteractiveSceneCfg):
    """Scene: pen desk + SO-101 follower + 4 pens + cup."""

    # shared world assets (not per-env)
    ground_plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
    )

    # pen desk USD (contains desk, mat, and all rigid objects)
    scene: AssetBaseCfg = PEN_DESK_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    # SO-101 follower articulation
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD_PATH,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=_ROBOT_POS,
            rot=_ROBOT_ROT,
            joint_pos={j: 0.0 for j in SO101_JOINT_ORDER},
        ),
        actuators={
            # Feetech STS3215 근사: 7.4V~12V variants 기준 약 1.4~2.9 Nm.
            # 시뮬 hold 안정성을 위해 3.0 Nm 상한, 속도 한계 5.5 rad/s.
            "arm_joints": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_pan", "shoulder_lift", "elbow_flex",
                                  "wrist_flex", "wrist_roll"],
                effort_limit_sim=3.0,
                velocity_limit_sim=5.5,
                stiffness=400.0,
                damping=80.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper"],
                effort_limit_sim=1.5,
                velocity_limit_sim=6.0,
                stiffness=300.0,
                damping=60.0,
            ),
        },
    )

    # Rigid objects inside the scene USD (spawn=None → wrap existing prims)
    PenWhite: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/PenWhite",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_PEN_INIT_STATES["PenWhite"][0],
            rot=_PEN_INIT_STATES["PenWhite"][1],
        ),
    )
    PenGray: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/PenGray",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_PEN_INIT_STATES["PenGray"][0],
            rot=_PEN_INIT_STATES["PenGray"][1],
        ),
    )
    PenBlack: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/PenBlack",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_PEN_INIT_STATES["PenBlack"][0],
            rot=_PEN_INIT_STATES["PenBlack"][1],
        ),
    )
    PenBlue: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/PenBlue",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_PEN_INIT_STATES["PenBlue"][0],
            rot=_PEN_INIT_STATES["PenBlue"][1],
        ),
    )
    PenCup: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/PenCup",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_PEN_CUP_INIT_STATE[0],
            rot=_PEN_CUP_INIT_STATE[1],
        ),
    )

    # ------------------------------------------------------------------
    # 카메라는 기본 씬에 두지 않는다 — env_smoke 가 --enable_cameras 없이 돌도록.
    # 카메라 smoke/롤아웃은 gym.make() 전에 add_pick_pen_cameras(scene) 로 주입.
    # (InteractiveScene 이 scene_cfg.__dict__ 를 순회하므로 동적 주입이 센서로
    #  등록됨 — Isaac Lab 2.3.2 interactive_scene._add_entities_from_cfg 확인.)
    # ------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 카메라 리그 (선택 주입) — observation.images.{top,front,wrist}
# ---------------------------------------------------------------------------


def _pinhole_camera_cfg(
    prim_path: str,
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float],
    focal_length: float,
    *,
    focus_distance: float,
    clipping_range: tuple[float, float],
) -> TiledCameraCfg:
    """640×480 RGB TiledCamera. offset 은 prim_path 부모 프레임 기준."""

    return TiledCameraCfg(
        prim_path=prim_path,
        offset=TiledCameraCfg.OffsetCfg(pos=pos, rot=rot, convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length,
            focus_distance=focus_distance,
            horizontal_aperture=20.955,
            clipping_range=clipping_range,
        ),
        width=640,
        height=480,
        update_period=0.0,
    )


def make_pick_pen_camera_cfgs(
    *,
    top_pos: tuple[float, float, float] | None = None,
    top_target: tuple[float, float, float] | None = None,
    top_focal: float | None = None,
    front_pos: tuple[float, float, float] | None = None,
    front_target: tuple[float, float, float] | None = None,
    front_focal: float | None = None,
    wrist_local_pos: tuple[float, float, float] | None = None,
    wrist_local_rot: tuple[float, float, float, float] | None = None,
    wrist_focal: float | None = None,
) -> dict[str, TiledCameraCfg]:
    """top/front/wrist 카메라 cfg 3개를 반환.

    기본 PickPenSceneCfg 밖에 두어 기본 env 가 --enable_cameras 없이 돌게 한다.
    gym.make() 전에 add_pick_pen_cameras() 로 scene cfg 에 주입해서 쓴다.
    각 카메라는 480×640 RGB.
    """

    top_pos = _TOP_CAMERA_POS if top_pos is None else top_pos
    top_target = _TOP_CAMERA_TARGET if top_target is None else top_target
    top_focal = _TOP_CAMERA_FOCAL if top_focal is None else top_focal
    front_pos = _FRONT_CAMERA_POS if front_pos is None else front_pos
    front_target = _FRONT_CAMERA_TARGET if front_target is None else front_target
    front_focal = _FRONT_CAMERA_FOCAL if front_focal is None else front_focal
    wrist_local_pos = _WRIST_CAM_LOCAL_POS if wrist_local_pos is None else wrist_local_pos
    wrist_local_rot = _WRIST_CAM_LOCAL_ROT if wrist_local_rot is None else wrist_local_rot
    wrist_focal = _WRIST_CAMERA_FOCAL if wrist_focal is None else wrist_focal

    top = _pinhole_camera_cfg(
        "{ENV_REGEX_NS}/TopCamera",
        top_pos,
        _look_at_quat_world(top_pos, top_target),
        top_focal,
        focus_distance=1.3,
        clipping_range=(0.1, 6.0),
    )
    front = _pinhole_camera_cfg(
        "{ENV_REGEX_NS}/FrontCamera",
        front_pos,
        _look_at_quat_world(front_pos, front_target),
        front_focal,
        focus_distance=0.6,
        clipping_range=(0.05, 6.0),
    )
    # wrist: gripper 링크의 자식 prim → 팔을 따라 이동. pos/rot 은 gripper-local
    # 프레임 기준. 정확한 화각은 GPU 렌더로 최종 검증 필요(Codex).
    wrist = _pinhole_camera_cfg(
        "{ENV_REGEX_NS}/Robot/gripper/WristCamera",
        wrist_local_pos,
        wrist_local_rot,
        wrist_focal,
        focus_distance=0.2,
        clipping_range=(0.02, 3.0),
    )
    return {"top_camera": top, "front_camera": front, "wrist_camera": wrist}


def add_pick_pen_cameras(
    scene_cfg: PickPenSceneCfg,
    *,
    top_pos: tuple[float, float, float] | None = None,
    top_target: tuple[float, float, float] | None = None,
    top_focal: float | None = None,
    front_pos: tuple[float, float, float] | None = None,
    front_target: tuple[float, float, float] | None = None,
    front_focal: float | None = None,
    wrist_local_pos: tuple[float, float, float] | None = None,
    wrist_local_rot: tuple[float, float, float, float] | None = None,
    wrist_focal: float | None = None,
) -> PickPenSceneCfg:
    """카메라 리그를 scene cfg 인스턴스에 in-place 주입하고 반환.

    InteractiveScene 이 scene_cfg.__dict__ 를 순회하므로 여기서 추가한 속성이
    gym.make() 시 센서로 등록된다. 멀티-env 시 world 좌표 → env-relative 전환
    필요(TC.2).
    """

    for name, cam_cfg in make_pick_pen_camera_cfgs(
        top_pos=top_pos,
        top_target=top_target,
        top_focal=top_focal,
        front_pos=front_pos,
        front_target=front_target,
        front_focal=front_focal,
        wrist_local_pos=wrist_local_pos,
        wrist_local_rot=wrist_local_rot,
        wrist_focal=wrist_focal,
    ).items():
        setattr(scene_cfg, name, cam_cfg)
    return scene_cfg


# ---------------------------------------------------------------------------
# Actions  (6-dim joint position, North Star order)
# ---------------------------------------------------------------------------


@configclass
class PickPenActionsCfg:
    """6-dim joint position action matching North Star joint order."""

    arm: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=SO101_JOINT_ORDER,
        scale=1.0,
        use_default_offset=True,
    )


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


@configclass
class PickPenObservationsCfg:
    """Observations: policy (6-dim joint pos) + subtask signals + rl_policy (privileged RL state).

    Groups:
      policy      — 6-dim joint pos (North Star contract, immutable).
      subtask_terms — per-pen placement signals.
      rl_policy   — TB.3 privileged state (37-dim) for RL actor/critic.
                    Does NOT contain the policy group; use obs_groups in train.py
                    to map both policy and critic to rl_policy.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """6-dim joint position in North Star order."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=SO101_JOINT_ORDER,
                )
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SubtaskCfg(ObsGroup):
        """Per-pen placement signals (pen-in-cup, gripper open check)."""

        place_white = ObsTerm(
            func=task_mdp.pen_in_cup,
            params={"object_cfg": SceneEntityCfg("PenWhite"), "cup_center_xy": PEN_CUP_CENTER_XY},
        )
        place_gray = ObsTerm(
            func=task_mdp.pen_in_cup,
            params={"object_cfg": SceneEntityCfg("PenGray"), "cup_center_xy": PEN_CUP_CENTER_XY},
        )
        place_black = ObsTerm(
            func=task_mdp.pen_in_cup,
            params={"object_cfg": SceneEntityCfg("PenBlack"), "cup_center_xy": PEN_CUP_CENTER_XY},
        )
        place_blue = ObsTerm(
            func=task_mdp.pen_in_cup,
            params={"object_cfg": SceneEntityCfg("PenBlue"), "cup_center_xy": PEN_CUP_CENTER_XY},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RlPolicyCfg(ObsGroup):
        """TB.3 privileged state for RL training (37-dim, concatenated).

        Includes joint pos, gripper body pos, all pen/cup positions relative to env
        origin, gripper→pen relative vectors, and gripper open fraction.
        No FrameTransformer dependency — resolves gripper body by name.
        """

        rl_state_obs = ObsTerm(
            func=task_mdp.rl_state,
            params={
                "pen_names": PEN_NAMES,
                "cup_name": PEN_CUP_NAME,
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()
    rl_policy: RlPolicyCfg = RlPolicyCfg()


# ---------------------------------------------------------------------------
# Rewards — Phase B 단계형 보상
# ---------------------------------------------------------------------------


@configclass
class PickPenRewardsCfg:
    """단계형 보상 — reach → grasp → lift → transport → insert → release."""

    # Stage 1: EE → 가장 가까운 미배치 펜 접근 (밀집)
    reach_pen = RewTerm(
        func=task_mdp.reach_reward,
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["gripper"]),
            "cup_center_xy": PEN_CUP_CENTER_XY,
        },
    )

    # Stage 2: 그리퍼 닫힘 + 펜 근접 (sparse bonus, 미배치 펜 한정)
    grasp_pen = RewTerm(
        func=task_mdp.grasp_bonus,
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["gripper"]),
            "cup_center_xy": PEN_CUP_CENTER_XY,
        },
    )

    # Stage 2.5: 닫힌 그리퍼 + 들린 펜 + 컵 방향 운반 (밀집 도우미)
    carry_pen = RewTerm(
        func=task_mdp.carry_pen,
        weight=4.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["gripper"]),
            "cup_center_xy": PEN_CUP_CENTER_XY,
        },
    )

    # Stage 3: 펜을 책상에서 들어올린 높이 (밀집)
    lift_pen = RewTerm(
        func=task_mdp.lift_reward,
        weight=2.0,
    )

    # Stage 4: 들어올린 펜의 XY → 컵 접근 (밀집)
    transport_pen = RewTerm(
        func=task_mdp.transport_reward,
        weight=8.0,
        params={"cup_center_xy": PEN_CUP_CENTER_XY},
    )

    # Stage 4.5: 컵 XY 근처에서 컵 안 높이로 낮추기 (밀집)
    place_height_pen = RewTerm(
        func=task_mdp.place_height_reward,
        weight=6.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["gripper"]),
            "cup_center_xy": PEN_CUP_CENTER_XY,
        },
    )

    # Stage 5: 컵 안 삽입 — 그리퍼 조건 없음 (밀집, 펜 수 비례)
    insert_pen = RewTerm(
        func=task_mdp.insert_reward,
        weight=25.0,
        params={"cup_center_xy": PEN_CUP_CENTER_XY},
    )

    # Stage 6: 컵 안 + 그리퍼 열림 완료 (밀집, 배치된 펜 수)
    release_pen = RewTerm(
        func=task_mdp.release_bonus,
        weight=10.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "cup_center_xy": PEN_CUP_CENTER_XY,
        },
    )

    # 전체 성공 보너스 — 4개 펜 전부 배치 완료
    task_success = RewTerm(
        func=task_mdp.task_success_bonus,
        weight=100.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "cup_center_xy": PEN_CUP_CENTER_XY,
        },
    )

    # 행동률·관절 속도 페널티
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


@configclass
class PickPenTerminationsCfg:
    """Episode ends on timeout or when all pens are in the cup."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=task_mdp.task_done,
        params={
            "pens_cfg": [SceneEntityCfg(name) for name in PEN_NAMES],
            "cup_center_xy": PEN_CUP_CENTER_XY,
            "require_rest_pose": False,  # rest-pose check is TA.1 territory
        },
    )


# ---------------------------------------------------------------------------
# Events (domain randomisation)
# ---------------------------------------------------------------------------


@configclass
class PickPenEventCfg:
    """Reset and randomisation events."""

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # Pens scatter inside a small ellipse around their authored positions
    randomize_pen_white = randomize_object_in_ellipse("PenWhite", 0.05, 0.02, (-10.0, 10.0))
    randomize_pen_gray = randomize_object_in_ellipse("PenGray", 0.05, 0.02, (-10.0, 10.0))
    randomize_pen_black = randomize_object_in_ellipse("PenBlack", 0.05, 0.02, (-10.0, 10.0))
    randomize_pen_blue = randomize_object_in_ellipse("PenBlue", 0.05, 0.02, (-10.0, 10.0))

    # Cup swings along a forward-facing ±20° arc
    randomize_pen_cup = randomize_object_on_arc(PEN_CUP_NAME, radius=0.44, angle_range_deg=(-20.0, 20.0))


# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------


@configclass
class PickPenEnvCfg(ManagerBasedRLEnvCfg):
    """Pen Pick-and-Place environment — pure Isaac Lab 2.3.2 ManagerBased."""

    scene: PickPenSceneCfg = PickPenSceneCfg(num_envs=1, env_spacing=2.5)
    observations: PickPenObservationsCfg = PickPenObservationsCfg()
    actions: PickPenActionsCfg = PickPenActionsCfg()
    rewards: PickPenRewardsCfg = PickPenRewardsCfg()
    terminations: PickPenTerminationsCfg = PickPenTerminationsCfg()
    events: PickPenEventCfg = PickPenEventCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # Physics: 120 Hz simulation, 30 Hz policy (decimation=4)
        self.sim.dt = 1.0 / 120.0
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 30.0
        # GPU pipeline
        self.sim.physx.enable_external_forces_every_iteration = True
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        # 2048+ env PPO에서 aggregate pair가 18k를 넘는다. 64k로 여유 확보.
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024


# ---------------------------------------------------------------------------
# 커리큘럼 적용 헬퍼 — gym.make() 이전에 env_cfg 에 in-place 적용
# ---------------------------------------------------------------------------

# 보상 항에서 펜 목록을 오버라이드할 term 이름 목록
_PEN_REWARD_TERMS = (
    "reach_pen",
    "grasp_pen",
    "carry_pen",
    "lift_pen",
    "transport_pen",
    "place_height_pen",
    "insert_pen",
    "release_pen",
    "task_success",
)
_CUP_RADIUS_REWARD_TERMS = (
    "reach_pen",
    "grasp_pen",
    "carry_pen",
    "place_height_pen",
    "insert_pen",
    "release_pen",
    "task_success",
)


def apply_curriculum(
    env_cfg: PickPenEnvCfg,
    *,
    active_pens: int = 4,
    pen_radius_scale: float = 1.0,
    cup_angle_scale: float = 1.0,
    cup_radius_scale: float = 1.0,
    grasp_assist: bool = False,
    grasp_assist_distance: float = 0.075,
    grasp_assist_offset_x: float = 0.0,
    grasp_assist_offset_y: float = 0.0,
    grasp_assist_offset_z: float = 0.0,
    place_assist_distance: float = 0.0,
) -> None:
    """커리큘럼 파라미터를 env_cfg 에 in-place 적용.

    Args:
        active_pens: 학습에 사용할 펜 수 (1~4). PEN_NAMES 앞에서부터 선택.
        pen_radius_scale: randomize_pen_* 의 x/y_radius 곱셈 배율.
        cup_angle_scale: randomize_pen_cup 의 angle_range_deg 곱셈 배율 (0 기준).
        cup_radius_scale: 컵 안 판정 반경 배율. 기본 1.0 = 0.05m.
        grasp_assist: 닫힌 그리퍼 근처 펜을 따라오게 하는 TB.3 학습 보조 event.
        grasp_assist_distance: assist attach 거리.
        grasp_assist_offset_{x,y,z}: gripper body 기준 world-frame pen center offset.
        place_assist_distance: 컵 근방 도달 시 컵 중심으로 스냅하는 거리. 0이면 비활성.
    """
    active_pens = max(1, min(4, active_pens))
    active_names = PEN_NAMES[:active_pens]
    active_cfgs = [SceneEntityCfg(n) for n in active_names]
    cup_radius = PEN_CUP_SUCCESS_RADIUS * max(0.1, cup_radius_scale)

    # 보상 term 에 활성 펜 목록 주입
    for term_name in _PEN_REWARD_TERMS:
        term = getattr(env_cfg.rewards, term_name, None)
        if term is not None:
            term.params["pen_cfgs"] = active_cfgs
    for term_name in _CUP_RADIUS_REWARD_TERMS:
        term = getattr(env_cfg.rewards, term_name, None)
        if term is not None:
            term.params["cup_radius"] = cup_radius

    # 종료 조건에 활성 펜 목록 주입
    env_cfg.terminations.success.params["pens_cfg"] = active_cfgs
    env_cfg.terminations.success.params["radius"] = cup_radius

    # ellipse 반경 스케일링 — randomize_pen_{white,gray,black,blue}
    if pen_radius_scale != 1.0:
        for pen_name in PEN_NAMES:
            attr_name = "randomize_pen_" + pen_name[3:].lower()  # PenWhite → white
            term = getattr(env_cfg.events, attr_name, None)
            if term is not None:
                p = term.params
                p["x_radius"] = p["x_radius"] * pen_radius_scale
                p["y_radius"] = p["y_radius"] * pen_radius_scale

    # 컵 각도 범위 스케일링 (0° 대칭 기준)
    if cup_angle_scale != 1.0:
        cup_term = env_cfg.events.randomize_pen_cup
        if cup_term is not None:
            lo, hi = cup_term.params["angle_range_deg"]
            cup_term.params["angle_range_deg"] = (lo * cup_angle_scale, hi * cup_angle_scale)

    if grasp_assist:
        env_cfg.events.soft_grasp_assist = EventTerm(
            func=task_mdp.soft_grasp_assist,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            params={
                "robot_cfg": SceneEntityCfg("robot", body_names=["gripper"]),
                "pen_cfgs": active_cfgs,
                "cup_center_xy": PEN_CUP_CENTER_XY,
                "cup_radius": cup_radius,
                "attach_distance": grasp_assist_distance,
                "place_distance": place_assist_distance,
                "offset": (grasp_assist_offset_x, grasp_assist_offset_y, grasp_assist_offset_z),
            },
        )
    elif hasattr(env_cfg.events, "soft_grasp_assist"):
        env_cfg.events.soft_grasp_assist = None
