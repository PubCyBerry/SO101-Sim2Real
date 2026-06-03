"""PickPen 전용 event terms."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from sim_to_real.utils.constant import PEN_NAMES

_DESK_TOP_Z: float = 0.92


def _make_pen_cfgs(pen_cfgs: list[SceneEntityCfg] | None) -> list[SceneEntityCfg]:
    if pen_cfgs is None:
        return [SceneEntityCfg(n) for n in PEN_NAMES]
    return pen_cfgs


def _get_gripper_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    robot_cfg.resolve(env.scene)
    robot: Articulation = env.scene[robot_cfg.name]
    ids = robot_cfg.body_ids
    if isinstance(ids, int):
        return robot.data.body_pos_w[:, ids, :]
    if isinstance(ids, slice):
        return robot.data.body_pos_w[:, ids, :][:, 0, :]
    return robot.data.body_pos_w[:, ids[0], :]


def _pen_inside_cup_mask(
    env: ManagerBasedRLEnv,
    pen_pos: torch.Tensor,
    cup_center_xy: tuple[float, float],
    radius: float,
    height_range: tuple[float, float],
) -> torch.Tensor:
    local_pos = pen_pos - env.scene.env_origins
    cx = torch.full((env.num_envs,), cup_center_xy[0], device=env.device)
    cy = torch.full((env.num_envs,), cup_center_xy[1], device=env.device)
    inside_xy = torch.hypot(local_pos[:, 0] - cx, local_pos[:, 1] - cy) < radius
    above = local_pos[:, 2] > (_DESK_TOP_Z + height_range[0])
    below = local_pos[:, 2] < (_DESK_TOP_Z + height_range[1])
    return inside_xy & above & below


def soft_grasp_assist(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["gripper"]),
    pen_cfgs: list[SceneEntityCfg] | None = None,
    cup_center_xy: tuple[float, float] = (2.2, -0.17),
    cup_radius: float = 0.05,
    cup_height_range: tuple[float, float] = (0.005, 0.18),
    attach_distance: float = 0.075,
    place_distance: float = 0.0,
    place_height: float = 0.07,
    close_threshold: float = 0.50,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """닫힌 그리퍼 근처의 가장 가까운 활성 펜을 그리퍼에 부드럽게 붙인다.

    TB.3 state-based 전문가 학습 전용 보조 event다. 기본 env에는 주입하지 않고
    train/eval curriculum에서 명시적으로 켰을 때만 매 step 실행한다.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)
    if len(env_ids) == 0:
        return

    cfgs = _make_pen_cfgs(pen_cfgs)
    robot_cfg.resolve(env.scene)
    robot: Articulation = env.scene[robot_cfg.name]
    closed = robot.data.joint_pos[env_ids, -1] < close_threshold
    if not bool(closed.any().item()):
        return

    gripper_pos = _get_gripper_pos(env, robot_cfg)[env_ids]
    offset_tensor = torch.tensor(offset, device=env.device, dtype=gripper_pos.dtype)

    per_pen_dist: list[torch.Tensor] = []
    per_pen_valid: list[torch.Tensor] = []
    for cfg in cfgs:
        pen: RigidObject = env.scene[cfg.name]
        pen_pos_all = pen.data.root_pos_w
        placed = _pen_inside_cup_mask(
            env,
            pen_pos_all,
            cup_center_xy,
            cup_radius,
            cup_height_range,
        )[env_ids]
        dist = torch.linalg.vector_norm(pen_pos_all[env_ids] - gripper_pos, dim=1)
        valid = closed & ~placed & (dist < attach_distance)
        per_pen_dist.append(torch.where(valid, dist, torch.full_like(dist, float("inf"))))
        per_pen_valid.append(valid)

    if not per_pen_dist:
        return
    dist_stack = torch.stack(per_pen_dist, dim=0)
    nearest_dist, nearest_index = dist_stack.min(dim=0)
    has_target = torch.isfinite(nearest_dist)
    if not bool(has_target.any().item()):
        return

    target_pos = gripper_pos + offset_tensor
    zero_vel = torch.zeros((len(env_ids), 6), device=env.device, dtype=target_pos.dtype)

    for pen_index, cfg in enumerate(cfgs):
        mask = has_target & (nearest_index == pen_index) & per_pen_valid[pen_index]
        if not bool(mask.any().item()):
            continue
        pen: RigidObject = env.scene[cfg.name]
        selected_env_ids = env_ids[mask]
        selected_target = target_pos[mask].clone()
        if place_distance > 0.0:
            origins = env.scene.env_origins[selected_env_ids]
            cup_xy = torch.tensor(cup_center_xy, device=env.device, dtype=selected_target.dtype)
            xy_dist = torch.linalg.vector_norm((selected_target - origins)[:, :2] - cup_xy, dim=1)
            place_mask = xy_dist < place_distance
            if bool(place_mask.any().item()):
                selected_target[place_mask, 0] = origins[place_mask, 0] + cup_center_xy[0]
                selected_target[place_mask, 1] = origins[place_mask, 1] + cup_center_xy[1]
                selected_target[place_mask, 2] = origins[place_mask, 2] + _DESK_TOP_Z + place_height
        pose = pen.data.root_state_w[selected_env_ids, :7].clone()
        pose[:, :3] = selected_target
        pen.write_root_pose_to_sim(pose, env_ids=selected_env_ids)
        pen.write_root_velocity_to_sim(zero_vel[mask], env_ids=selected_env_ids)
