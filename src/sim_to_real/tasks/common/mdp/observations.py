"""공통 Pick-and-Place 관측 함수.

두 계열:
- *완료 판정* (`object_grasped`, `object_in_container`) — 관리자 기반 관측 그룹과
  success termination 에서 사용. 그리퍼 상태를 포함하므로 "배치 완료" 는 릴리즈까지 의미함.
- *기하 헬퍼* (`object_inside_container`, `object_lifted`, `ee_near_object`,
  `object_above_container_xy`) — 오라클 자동 채점기에서 각 서브페이즈 독립 패스/페일 추적.
"""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

from ._geometry import CONTAINER_DEFAULT_CENTER_XY, DESK_TOP_Z


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
    return torch.logical_and(pos_diff < diff_threshold, robot.data.joint_pos[:, -1] < grasp_threshold)


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
    gripper_open = robot.data.joint_pos[:, -1] > grasp_threshold
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
