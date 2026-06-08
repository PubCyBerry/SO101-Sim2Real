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
SO101_JOINT_TARGET_MAX_VELOCITY: dict[str, float] = {
    "shoulder_pan": 5.00,
    "shoulder_lift": 5.00,
    "elbow_flex": 5.00,
    "wrist_flex": 5.00,
    "wrist_roll": 5.00,
    # 그리퍼도 5.0 rad/s 상한(사용자 지시). 이건 *상한*이지 항상 이 속도로 닫는 게
    # 아니다 — teleop 은 leader 입력 속도를, RL/SM 은 명령 속도를 따른다. (과거 SM 이
    # cap 5.0 에서 큐브를 snap-튕긴 건 명령 속도를 너무 키운 탓이라, 접촉 시 명령 속도
    # 자체를 줄이는 쪽으로 푼다.)
    "gripper": 5.00,
}
"""Processed joint-position target speed cap in rad/s (sim time).

팔 joint = 5.0 rad/s 상한(사용자 요청, teleop·RL·state-machine 공용). 그리퍼 = 1.0 rad/s
(grasp 접촉 안정). actuator ``velocity_limit_sim`` 은 ≥5 rad/s 헤드룸을 유지해야 명령 속도를
실제로 추종한다. 주의: 1.0 rad/s 로 학습된 RL checkpoint 는 팔 변경으로 dynamics 가 바뀌어
재학습이 필요하다."""

# Robot base position: SCENE_OFFSET(2.2, -0.57) + scene-local robot offset(0, -0.04).
#
# z 정합: 책상 상판(DeskTop) 윗면은 world z=0.76 (SCENE_OFFSET.z 0.76 + center
# -0.02 + half-thickness 0.02). 그러나 so101_follower.usd 의 articulation root
# 원점(z=0)은 베이스 바닥이 아니다 — 베이스 최하단 지오메트리가 local z=+0.0301
# 에 있다(USD bbox 측정). 따라서 원점을 0.76 에 두면 팔 전체가 ~3 cm 떠버린다
# (사용자가 보고한 "로봇이 책상 위에 떠 있는" 현상). 베이스 판이 상판에 닿도록
# 원점을 내린다:  robot_z = desk_top(0.76) - base_min_z(0.0301) ≈ 0.7299.
_ROBOT_POS = (2.2, -0.61, 0.7299)
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
    "PenWhite": ((2.05, -0.35, 0.7747), _yaw_quat(25.0)),
    "PenGray": ((2.35, -0.35, 0.7747), _yaw_quat(-30.0)),
    "PenBlack": ((2.25, -0.31, 0.7747), _yaw_quat(60.0)),
    "PenBlue": ((2.15, -0.31, 0.7747), _yaw_quat(-10.0)),
}
_PEN_CUP_INIT_STATE = ((2.2, -0.17, 0.766), _yaw_quat(0.0))

# ---------------------------------------------------------------------------
# 카메라 리그 상수 — North Star 계약: observation.images.{top,wrist,front}
#   · 모두 640×480 (W×H) RGB, update_period=1/30
#   · 포즈/FOV 는 실제 데이터셋 프레임을 기준으로 튜닝.
#   · top 은 world frame 절대 좌표, wrist 는 gripper 링크 자식 prim 의 local offset.
#     num_envs=1 smoke 기준.
# ---------------------------------------------------------------------------

# top: 로봇 뒤(-y)·높은 곳에서 내려보는 급경사 oblique. 로봇 베이스가 하단,
# 펜/컵/매트가 위로 펼쳐진다.
_TOP_CAMERA_POS = (2.2, -1.12, 1.72)
_TOP_CAMERA_TARGET = (2.14, -0.15, 0.76)
_TOP_CAMERA_FOCAL = 16.0

# wrist: gripper 위/옆에 강결합된 카메라. gripper 움직임을 그대로 따라간다.
# gripper-local 축은 rest 자세에서 대략 localX->+Z, localY->-X, localZ->-Y.
_WRIST_CAM_LOCAL_POS = (0.035, 0.035, -0.075)
# gripper parent 회전을 역산해, rest 자세에서 camera world pos≈(2.149,-0.213,0.956)
# 기준 mat/cup 방향 world target≈(2.10,-0.13,0.77)을 바라보도록 한 local rot.
_WRIST_CAM_LOCAL_ROT = (0.2280, 0.0630, 0.9365, 0.2588)
_WRIST_CAMERA_FOCAL = 10.0

# front: 책상 정면에서 작업공간을 바라보는 카메라.
# 기본값은 --tune_cameras 로 튜닝 후 이 상수에 업데이트할 것.
_FRONT_CAMERA_POS = (2.14, 0.65, 1.10)
_FRONT_CAMERA_TARGET = (2.14, -0.15, 0.80)
_FRONT_CAMERA_FOCAL = 18.0


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
                # leisaac SO101_FOLLOWER_CFG 검증값(enabled_self_collisions + solver 4/4).
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=_ROBOT_POS,
            rot=_ROBOT_ROT,
            joint_pos={j: 0.0 for j in SO101_JOINT_ORDER},
        ),
        actuators={
            # leisaac SO101_FOLLOWER_CFG 검증값 이식 (ref_repos/leisaac 의
            # assets/robots/lerobot.py). Feetech STS3215 를 낮은 stiffness(soft PD)
            # + 높은 effort 상한으로 모델링한다. 그리퍼가 큐브/펜에 막혀도 클램프
            # 토크가 최대 10 Nm 까지 올라가 grasp 가 유지된다(이전 1.5 Nm 상한은
            # stiffness 300 에서 ~0.3° 만에 포화돼 들어올릴 때 미끄러짐).
            "arm_joints": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_pan", "shoulder_lift", "elbow_flex",
                                  "wrist_flex", "wrist_roll"],
                effort_limit_sim=10.0,
                velocity_limit_sim=10.0,
                stiffness=17.8,
                damping=0.6,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper"],
                effort_limit_sim=10.0,
                velocity_limit_sim=10.0,
                stiffness=17.8,
                damping=0.6,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
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
# 카메라 리그 (선택 주입) — observation.images.{top,wrist,front}
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
        update_period=1.0 / 30.0,
        update_latest_camera_pose=True,
    )


def make_pick_pen_camera_cfgs(
    *,
    top_pos: tuple[float, float, float] | None = None,
    top_target: tuple[float, float, float] | None = None,
    top_focal: float | None = None,
    wrist_local_pos: tuple[float, float, float] | None = None,
    wrist_local_rot: tuple[float, float, float, float] | None = None,
    wrist_focal: float | None = None,
    front_pos: tuple[float, float, float] | None = None,
    front_target: tuple[float, float, float] | None = None,
    front_focal: float | None = None,
) -> dict[str, TiledCameraCfg]:
    """top/wrist/front 카메라 cfg 3개를 반환.

    기본 PickPenSceneCfg 밖에 두어 기본 env 가 --enable_cameras 없이 돌게 한다.
    gym.make() 전에 add_pick_pen_cameras() 로 scene cfg 에 주입해서 쓴다.
    각 카메라는 480×640 RGB.
    """

    top_pos = _TOP_CAMERA_POS if top_pos is None else top_pos
    top_target = _TOP_CAMERA_TARGET if top_target is None else top_target
    top_focal = _TOP_CAMERA_FOCAL if top_focal is None else top_focal
    wrist_local_pos = _WRIST_CAM_LOCAL_POS if wrist_local_pos is None else wrist_local_pos
    wrist_local_rot = _WRIST_CAM_LOCAL_ROT if wrist_local_rot is None else wrist_local_rot
    wrist_focal = _WRIST_CAMERA_FOCAL if wrist_focal is None else wrist_focal
    front_pos = _FRONT_CAMERA_POS if front_pos is None else front_pos
    front_target = _FRONT_CAMERA_TARGET if front_target is None else front_target
    front_focal = _FRONT_CAMERA_FOCAL if front_focal is None else front_focal

    top = _pinhole_camera_cfg(
        "{ENV_REGEX_NS}/TopCamera",
        top_pos,
        _look_at_quat_world(top_pos, top_target),
        top_focal,
        focus_distance=1.3,
        clipping_range=(0.1, 6.0),
    )
    # wrist: robot 링크의 자식 prim → gripper 회전을 따라 이동/회전한다.
    # pos/rot 은 gripper local frame 기준. 정확한 화각은 GUI 렌더로 튜닝한다.
    wrist = _pinhole_camera_cfg(
        "{ENV_REGEX_NS}/Robot/gripper/WristCamera",
        wrist_local_pos,
        wrist_local_rot,
        wrist_focal,
        focus_distance=0.2,
        clipping_range=(0.02, 3.0),
    )
    front = _pinhole_camera_cfg(
        "{ENV_REGEX_NS}/FrontCamera",
        front_pos,
        _look_at_quat_world(front_pos, front_target),
        front_focal,
        focus_distance=1.0,
        clipping_range=(0.1, 6.0),
    )
    return {"top_camera": top, "wrist_camera": wrist, "front_camera": front}


def add_pick_pen_cameras(
    scene_cfg: PickPenSceneCfg,
    *,
    top_pos: tuple[float, float, float] | None = None,
    top_target: tuple[float, float, float] | None = None,
    top_focal: float | None = None,
    wrist_local_pos: tuple[float, float, float] | None = None,
    wrist_local_rot: tuple[float, float, float, float] | None = None,
    wrist_focal: float | None = None,
    front_pos: tuple[float, float, float] | None = None,
    front_target: tuple[float, float, float] | None = None,
    front_focal: float | None = None,
) -> PickPenSceneCfg:
    """top/wrist/front 카메라 리그를 scene cfg 인스턴스에 in-place 주입하고 반환.

    InteractiveScene 이 scene_cfg.__dict__ 를 순회하므로 여기서 추가한 속성이
    gym.make() 시 센서로 등록된다.
    """

    for name, cam_cfg in make_pick_pen_camera_cfgs(
        top_pos=top_pos,
        top_target=top_target,
        top_focal=top_focal,
        wrist_local_pos=wrist_local_pos,
        wrist_local_rot=wrist_local_rot,
        wrist_focal=wrist_focal,
        front_pos=front_pos,
        front_target=front_target,
        front_focal=front_focal,
    ).items():
        setattr(scene_cfg, name, cam_cfg)
    return scene_cfg


# ---------------------------------------------------------------------------
# Actions  (6-dim joint position, North Star order)
# ---------------------------------------------------------------------------


@configclass
class PickPenActionsCfg:
    """6-dim joint position action matching North Star joint order."""

    arm: task_mdp.SlewLimitedJointPositionActionCfg = task_mdp.SlewLimitedJointPositionActionCfg(
        asset_name="robot",
        joint_names=SO101_JOINT_ORDER,
        scale=1.0,
        use_default_offset=True,
        max_velocity=SO101_JOINT_TARGET_MAX_VELOCITY,
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
      rl_policy   — TB.3 privileged state (43-dim) for RL actor/critic.
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
        """TB.3 privileged state for RL training (43-dim, concatenated).

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
            "cup_cfg": SceneEntityCfg(PEN_CUP_NAME),
        },
    )

    # Stage 2: 그리퍼 닫힘 + 펜 근접 (sparse bonus, 미배치 펜 한정)
    grasp_pen = RewTerm(
        func=task_mdp.grasp_bonus,
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["gripper"]),
            "cup_center_xy": PEN_CUP_CENTER_XY,
            "cup_cfg": SceneEntityCfg(PEN_CUP_NAME),
        },
    )

    # Stage 2.5: 닫힌 그리퍼 + 들린 펜 + 컵 방향 운반 (밀집 도우미)
    carry_pen = RewTerm(
        func=task_mdp.carry_pen,
        weight=4.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["gripper"]),
            "cup_center_xy": PEN_CUP_CENTER_XY,
            "cup_cfg": SceneEntityCfg(PEN_CUP_NAME),
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
        params={
            "cup_center_xy": PEN_CUP_CENTER_XY,
            "cup_cfg": SceneEntityCfg(PEN_CUP_NAME),
        },
    )

    # Stage 4.5: 컵 XY 근처에서 컵 안 높이로 낮추기 (밀집)
    place_height_pen = RewTerm(
        func=task_mdp.place_height_reward,
        weight=6.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["gripper"]),
            "cup_center_xy": PEN_CUP_CENTER_XY,
            "cup_cfg": SceneEntityCfg(PEN_CUP_NAME),
        },
    )

    # Stage 5: 컵 안 삽입 — 그리퍼 조건 없음 (밀집, 펜 수 비례)
    insert_pen = RewTerm(
        func=task_mdp.insert_reward,
        weight=25.0,
        params={
            "cup_center_xy": PEN_CUP_CENTER_XY,
            "cup_cfg": SceneEntityCfg(PEN_CUP_NAME),
        },
    )

    # Stage 6: 컵 안 + 그리퍼 열림 완료 (밀집, 배치된 펜 수)
    release_pen = RewTerm(
        func=task_mdp.release_bonus,
        weight=10.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "cup_center_xy": PEN_CUP_CENTER_XY,
            "cup_cfg": SceneEntityCfg(PEN_CUP_NAME),
        },
    )

    # 전체 성공 보너스 — 4개 펜 전부 배치 완료
    task_success = RewTerm(
        func=task_mdp.task_success_bonus,
        weight=100.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "cup_center_xy": PEN_CUP_CENTER_XY,
            "cup_cfg": SceneEntityCfg(PEN_CUP_NAME),
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
            "cup_cfg": SceneEntityCfg(PEN_CUP_NAME),
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
    dynamic_reset_gripper_effort_limit: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        # Physics: 120 Hz simulation, 30 Hz policy (decimation=4)
        self.sim.dt = 1.0 / 120.0
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 30.0
        # GPU pipeline
        self.sim.physx.enable_external_forces_every_iteration = True
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        # 4096 env PPO에서 aggregate pair가 134k 근처까지 올라간다. 256k로 여유 확보.
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 256 * 1024


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
) -> None:
    """커리큘럼 파라미터를 env_cfg 에 in-place 적용.

    Args:
        active_pens: 학습에 사용할 펜 수 (1~4). PEN_NAMES 앞에서부터 선택.
        pen_radius_scale: randomize_pen_* 의 x/y_radius 곱셈 배율.
        cup_angle_scale: randomize_pen_cup 의 angle_range_deg 곱셈 배율 (0 기준).
        cup_radius_scale: 컵 안 판정 반경 배율. 기본 1.0 = 0.05m.
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
