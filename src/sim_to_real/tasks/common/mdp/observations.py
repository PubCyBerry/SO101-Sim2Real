"""공통 Pick-and-Place 관측 함수.

세 계열:
- *완료 판정* (`object_grasped`, `object_in_container`) — 관리자 기반 관측 그룹과
  success termination 에서 사용. 그리퍼 상태를 포함하므로 "배치 완료" 는 릴리즈까지 의미함.
- *기하 헬퍼* (`object_inside_container`, `object_lifted`, `ee_near_object`,
  `object_above_container_xy`) — 오라클 자동 채점기에서 각 서브페이즈 독립 패스/페일 추적.
- *RL 상태* (`rl_state`) — 학습 전용 특권 full-state.
"""

from __future__ import annotations

from typing import Sequence

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

from ._geometry import (
    CONTAINER_DEFAULT_CENTER_XY,
    DESK_TOP_Z,
    JAW_GRASP_OFFSET,
    _quat_apply_wxyz,
    _yaw_from_quat_wxyz,
)


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------


def _get_action_term(env: ManagerBasedRLEnv | DirectRLEnv, term_name: str):
    """ActionManager에서 term을 방어적으로 찾는다."""
    action_manager = getattr(env, "action_manager", None)
    if action_manager is None:
        return None

    get_term = getattr(action_manager, "get_term", None)
    if callable(get_term):
        try:
            return get_term(term_name)
        except Exception:
            pass

    for attr_name in ("_terms", "_action_terms"):
        terms = getattr(action_manager, attr_name, None)
        if isinstance(terms, dict) and term_name in terms:
            return terms[term_name]

    return None


def _current_action_target(
    env: ManagerBasedRLEnv | DirectRLEnv,
    joint_pos: torch.Tensor,
    term_name: str,
) -> torch.Tensor:
    """현재 action term target을 반환하고, 접근 실패 시 joint_pos로 대체한다."""
    term = _get_action_term(env, term_name)
    if term is None:
        return joint_pos

    for attr_name in ("processed_actions", "_processed_actions", "_limited_actions"):
        value = getattr(term, attr_name, None)
        if isinstance(value, torch.Tensor) and value.shape == joint_pos.shape:
            return value

    return joint_pos


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


# ---------------------------------------------------------------------------
# RL 특권 상태 (TB.3) — FrameTransformer 의존 없음
# ---------------------------------------------------------------------------


def rl_state(
    env: ManagerBasedRLEnv | DirectRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_names: Sequence[str] = (),
    container_name: str = "container",
    gripper_body_name: str = "gripper",
    action_term_name: str = "arm",
    include_velocities: bool = False,
    include_orientation: bool = False,
    include_container_orientation: bool = False,
    object_half_extents: Sequence[float] | None = None,
    num_active: int | None = None,
) -> torch.Tensor:
    """RL 학습용 특권 상태 벡터.

    기본 shape: (num_envs, D) where D = 6+6+3+N*3+3+N*3+1.

    구성 (모든 위치는 env-origin 상대):
      [0:6]      robot joint positions (rad)
      [6:12]     processed joint target (rad)
      [12:15]    jaw-offset grasp point position (m)
      [15:15+N*3] object positions, N×3 (m)
      [-3-N*3:-3] container position (m)
      [-3-N*3 이전] grasp point→object relative vectors, N×3 (m)  — 정확한 오프셋은 N에 따라 달라짐
      [-1]       gripper joint pos normalised [0,1]

    include_velocities=True: +N*3+6+3 (joint_vel + ee_lin_vel + obj_lin_vel)
    include_orientation=True: +N*2+N+4+3 (obj yaw sin/cos + half-extent + ee quat + grasp→container)
    include_container_orientation=True: +4 (container quat wxyz)
    num_active: 이 수보다 뒤 인덱스 obj 관련 블록을 0 마스킹.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    origins = env.scene.env_origins  # (N, 3)

    n_objects = len(object_names)
    n_act = n_objects if num_active is None else max(0, min(n_objects, int(num_active)))
    obj_active = [1.0 if i < n_act else 0.0 for i in range(n_objects)]

    joint_pos = robot.data.joint_pos
    joint_target = _current_action_target(env, joint_pos, action_term_name)

    body_names: list[str] = robot.data.body_names
    if "jaw" in body_names:
        jaw_idx = body_names.index("jaw")
        offset = torch.tensor(JAW_GRASP_OFFSET, device=env.device, dtype=robot.data.body_pos_w.dtype)
        offset = offset.unsqueeze(0).expand(env.num_envs, -1)
        grasp_pos = robot.data.body_pos_w[:, jaw_idx, :] + _quat_apply_wxyz(
            robot.data.body_quat_w[:, jaw_idx, :], offset
        )
        gripper_pos = grasp_pos - origins
    else:
        try:
            gripper_idx = body_names.index(gripper_body_name)
        except ValueError:
            gripper_idx = next(
                (i for i, n in enumerate(body_names) if gripper_body_name in n), 0
            )
        gripper_pos = robot.data.body_pos_w[:, gripper_idx, :] - origins

    obj_parts: list[torch.Tensor] = []
    for i, name in enumerate(object_names):
        obj: RigidObject = env.scene[name]
        obj_parts.append(obj.data.root_pos_w - origins)
    obj_pos = torch.cat([p * obj_active[i] for i, p in enumerate(obj_parts)], dim=-1) if obj_parts else torch.zeros(env.num_envs, 0, device=env.device)

    container: RigidObject = env.scene[container_name]
    container_pos = container.data.root_pos_w - origins

    rel_parts: list[torch.Tensor] = []
    for i in range(n_objects):
        rel_parts.append((obj_parts[i] - gripper_pos) * obj_active[i])
    rel_pos = torch.cat(rel_parts, dim=-1) if rel_parts else torch.zeros(env.num_envs, 0, device=env.device)

    gripper_open = joint_pos[:, -1:].clamp(0.0, 1.0)

    parts = [joint_pos, joint_target, gripper_pos, obj_pos, container_pos, rel_pos, gripper_open]

    if include_velocities:
        ee_body_idx = body_names.index("jaw") if "jaw" in body_names else gripper_idx
        joint_vel = robot.data.joint_vel
        ee_vel = robot.data.body_lin_vel_w[:, ee_body_idx, :]
        obj_vel = torch.cat(
            [env.scene[n].data.root_lin_vel_w * obj_active[i] for i, n in enumerate(object_names)],
            dim=-1,
        ) if object_names else torch.zeros(env.num_envs, 0, device=env.device)
        parts += [joint_vel, ee_vel, obj_vel]

    if include_orientation:
        yaw_parts: list[torch.Tensor] = []
        for i, name in enumerate(object_names):
            yaw = _yaw_from_quat_wxyz(env.scene[name].data.root_quat_w)
            sc = torch.stack([torch.sin(yaw), torch.cos(yaw)], dim=-1) * obj_active[i]
            yaw_parts.append(sc)
        obj_yaw = torch.cat(yaw_parts, dim=-1) if yaw_parts else torch.zeros(env.num_envs, 0, device=env.device)

        if object_half_extents is not None:
            he_vals = [float(object_half_extents[i]) * obj_active[i] for i in range(n_objects)]
            he = torch.tensor(he_vals, device=env.device, dtype=torch.float32).unsqueeze(0).expand(env.num_envs, -1)
        else:
            he = torch.zeros(env.num_envs, n_objects, device=env.device)

        if "jaw" in body_names:
            ee_idx = body_names.index("jaw")
        else:
            ee_idx = next((i for i, n in enumerate(body_names) if gripper_body_name in n), 0)
        ee_quat = robot.data.body_quat_w[:, ee_idx, :]
        grasp_to_container = container_pos - gripper_pos

        parts += [obj_yaw, he, ee_quat, grasp_to_container]

    if include_container_orientation:
        container_quat = container.data.root_quat_w
        parts.append(container_quat)

    return torch.cat(parts, dim=-1).float()
