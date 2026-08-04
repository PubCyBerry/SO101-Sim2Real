"""공통 Pick-and-Place 관측 함수.

세 계열:
- *완료 판정* (`object_grasped`, `object_in_container`) — 관리자 기반 관측 그룹과
  success termination 에서 사용. 그리퍼 상태를 포함하므로 "배치 완료" 는 릴리즈까지 의미함.
- *기하 헬퍼* (`object_inside_container`, `object_lifted`, `ee_near_object`,
  `object_above_container_xy`) — 오라클 자동 채점기에서 각 서브페이즈 독립 패스/페일 추적.
- *센서 관측* (`ee_frame_state`, `image_raw`) — Workshop ``mdp/obs.py`` 이식. EE pose(FrameTransformer)
  와 카메라 이미지를 관측 그룹에 노출(policy 6-dim joint 계약과 별개, privileged/visual 그룹용).
"""

from __future__ import annotations

import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv, ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

from ._geometry import CONTAINER_DEFAULT_CENTER_XY, DESK_TOP_Z


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def _get_gripper_joint_index(robot_cfg: SceneEntityCfg) -> int:
    """robot_cfg.joint_names 에서 'gripper' 찾기. 없으면 -1."""
    if hasattr(robot_cfg, "joint_names") and robot_cfg.joint_names:
        try:
            return list(robot_cfg.joint_names).index("gripper")
        except ValueError:
            pass
    return -1


# ---------------------------------------------------------------------------
# 완료 판정 함수
# ---------------------------------------------------------------------------


def object_grasped(
    env: ManagerBasedRLEnv | DirectRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    diff_threshold: float = 0.05,
    grasp_threshold: float = 0.60,
) -> torch.Tensor:
    """물체가 EE jaw 에 의해 파지됐는지 확인한다."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    obj_pos = obj.data.root_pos_w
    jaw_pos = ee_frame.data.target_pos_w[:, 1, :]
    pos_diff = torch.linalg.vector_norm(obj_pos - jaw_pos, dim=1)
    gripper_idx = _get_gripper_joint_index(robot_cfg)
    return torch.logical_and(pos_diff < diff_threshold, robot.data.joint_pos[:, gripper_idx] < grasp_threshold)


def object_in_container(
    env: ManagerBasedRLEnv | DirectRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    container_cfg: SceneEntityCfg | None = None,
    container_center_xy: tuple[float, float] = CONTAINER_DEFAULT_CENTER_XY,
    radius: float = 0.05,
    height_range: tuple[float, float] = (0.005, 0.18),
    grasp_threshold: float = 0.60,
) -> torch.Tensor:
    """물체가 컨테이너 풋프린트 안에 있고 그리퍼가 열려 있으면 True."""
    robot: Articulation = env.scene[robot_cfg.name]
    inside = object_inside_container(
        env,
        object_cfg=object_cfg,
        container_cfg=container_cfg,
        container_center_xy=container_center_xy,
        radius=radius,
        height_range=height_range,
    )
    gripper_idx = _get_gripper_joint_index(robot_cfg)
    gripper_open = robot.data.joint_pos[:, gripper_idx] > grasp_threshold
    return torch.logical_and(inside, gripper_open)


# ---------------------------------------------------------------------------
# 기하 헬퍼 함수 (그리퍼 조건 없음)
# ---------------------------------------------------------------------------


def object_inside_container(
    env: ManagerBasedRLEnv | DirectRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    container_cfg: SceneEntityCfg | None = None,
    container_center_xy: tuple[float, float] = CONTAINER_DEFAULT_CENTER_XY,
    radius: float = 0.05,
    height_range: tuple[float, float] = (0.005, 0.18),
) -> torch.Tensor:
    """물체 위치가 컨테이너 풋프린트 안에 있는지 (그리퍼 조건 없음)."""
    obj: RigidObject = env.scene[object_cfg.name]
    obj_pos = obj.data.root_pos_w - env.scene.env_origins

    if container_cfg is not None:
        container = env.scene[container_cfg.name]
        container_pos = container.data.root_pos_w - env.scene.env_origins
        cx, cy, cz = container_pos[:, 0], container_pos[:, 1], container_pos[:, 2]
    else:
        cx = torch.full((env.num_envs,), container_center_xy[0], device=env.device)
        cy = torch.full((env.num_envs,), container_center_xy[1], device=env.device)
        cz = torch.full((env.num_envs,), DESK_TOP_Z, device=env.device)

    inside_xy = torch.hypot(obj_pos[:, 0] - cx, obj_pos[:, 1] - cy) < radius
    above_floor = obj_pos[:, 2] > (cz + height_range[0])
    below_lip = obj_pos[:, 2] < (cz + height_range[1])
    return torch.logical_and(inside_xy, torch.logical_and(above_floor, below_lip))


def object_above_container_xy(
    env: ManagerBasedRLEnv | DirectRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    container_cfg: SceneEntityCfg | None = None,
    container_center_xy: tuple[float, float] = CONTAINER_DEFAULT_CENTER_XY,
    radius: float = 0.05,
) -> torch.Tensor:
    """물체 XY 위치가 컨테이너 반경 이내 (높이 무관)."""
    obj: RigidObject = env.scene[object_cfg.name]
    obj_pos = obj.data.root_pos_w - env.scene.env_origins

    if container_cfg is not None:
        container = env.scene[container_cfg.name]
        container_pos = container.data.root_pos_w - env.scene.env_origins
        cx, cy = container_pos[:, 0], container_pos[:, 1]
    else:
        cx = torch.full((env.num_envs,), container_center_xy[0], device=env.device)
        cy = torch.full((env.num_envs,), container_center_xy[1], device=env.device)

    return torch.hypot(obj_pos[:, 0] - cx, obj_pos[:, 1] - cy) < radius


def object_lifted(
    env: ManagerBasedRLEnv | DirectRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    height_threshold: float = 0.05,
) -> torch.Tensor:
    """물체가 책상 상판 기준 ``height_threshold`` (m) 이상 들려 있으면 True."""
    obj: RigidObject = env.scene[object_cfg.name]
    obj_z = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return obj_z > (DESK_TOP_Z + height_threshold)


def ee_near_object(
    env: ManagerBasedRLEnv | DirectRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    distance_threshold: float = 0.05,
) -> torch.Tensor:
    """EE jaw 가 ``distance_threshold`` (m) 이내로 물체에 접근해 있으면 True."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    jaw_pos = ee_frame.data.target_pos_w[:, 1, :]
    obj_pos = obj.data.root_pos_w
    return torch.linalg.vector_norm(obj_pos - jaw_pos, dim=1) < distance_threshold


# ---------------------------------------------------------------------------
# 센서 관측 (Workshop mdp/obs.py 이식)
# ---------------------------------------------------------------------------


def ee_frame_state(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """End-effector frame pose 를 로봇 root frame 기준으로 반환 (7D: pos3 + quat4 wxyz).

    Workshop ``mdp/obs.py:ee_frame_state`` 이식. FrameTransformer(``ee_frame``)의 첫 target
    (index 0 = gripper)의 world pose 를 robot root 로 subtract 한다. 우리 ``ee_frame`` 은
    target 2개(gripper=0, jaw=1)라 여기선 gripper(0)만 쓴다. policy 6-dim joint 계약과 별개
    (privileged/subtask 관측 용).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    robot_root_pos, robot_root_quat = robot.data.root_pos_w, robot.data.root_quat_w
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos, ee_frame_quat = ee_frame.data.target_pos_w[:, 0, :], ee_frame.data.target_quat_w[:, 0, :]
    ee_frame_pos_robot, ee_frame_quat_robot = math_utils.subtract_frame_transforms(
        robot_root_pos, robot_root_quat, ee_frame_pos, ee_frame_quat
    )
    return torch.cat([ee_frame_pos_robot, ee_frame_quat_robot], dim=1)


def image_raw(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
) -> torch.Tensor:
    """카메라 sensor 의 raw 출력(``data.output[data_type]``)을 clone 해 반환.

    Workshop ``mdp/obs.py:image_raw`` 이식. 정규화 없이 원본 텐서를 그대로 노출한다
    (instance segmentation 등 비-RGB 채널용). RGB 정규화판은 isaaclab stock ``mdp.image``.
    """
    sensor = env.scene[sensor_cfg.name]
    return sensor.data.output[data_type].clone()
