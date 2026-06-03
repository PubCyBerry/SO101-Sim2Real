"""Pen Pick-and-Place subtask observation terms.

Two families live here:

- *Outcome* terms (``pen_grasped``, ``pen_in_cup``) — used by the manager-based
  observation group and the success termination. They include gripper state
  so a "placed" result implies the robot also released the pen.
- *Geometry-only* helpers (``pen_inside_cup``, ``pen_lifted``, ``ee_near_pen``,
  ``pen_above_cup_xy``) — used by the oracle's auto-scoring tracker so each
  sub-phase has an independent pass/fail signal even when the gripper
  command is wrong.
"""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer


PEN_CUP_DEFAULT_CENTER_XY: tuple[float, float] = (2.2, -0.17)
DESK_TOP_Z: float = 0.92


def pen_grasped(
    env: ManagerBasedRLEnv | DirectRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("PenWhite"),
    diff_threshold: float = 0.05,
    grasp_threshold: float = 0.60,
) -> torch.Tensor:
    """Check if an object(pen) is grasped by the specified robot."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    pen_pos = object.data.root_pos_w
    jaw_pos = ee_frame.data.target_pos_w[:, 1, :]
    pos_diff = torch.linalg.vector_norm(pen_pos - jaw_pos, dim=1)
    return torch.logical_and(pos_diff < diff_threshold, robot.data.joint_pos[:, -1] < grasp_threshold)


def pen_in_cup(
    env: ManagerBasedRLEnv | DirectRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("PenWhite"),
    cup_cfg: SceneEntityCfg | None = None,
    cup_center_xy: tuple[float, float] = PEN_CUP_DEFAULT_CENTER_XY,
    radius: float = 0.05,
    height_range: tuple[float, float] = (0.005, 0.18),
    grasp_threshold: float = 0.60,
) -> torch.Tensor:
    """Pen sits inside the cup footprint AND the gripper has released."""
    robot: Articulation = env.scene[robot_cfg.name]
    inside = pen_inside_cup(
        env,
        object_cfg=object_cfg,
        cup_cfg=cup_cfg,
        cup_center_xy=cup_center_xy,
        radius=radius,
        height_range=height_range,
    )
    gripper_open = robot.data.joint_pos[:, -1] > grasp_threshold
    return torch.logical_and(inside, gripper_open)


# ---------------------------------------------------------------------------
# Geometry-only helpers (used by the oracle auto-scorer)
# ---------------------------------------------------------------------------


def pen_inside_cup(
    env: ManagerBasedRLEnv | DirectRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("PenWhite"),
    cup_cfg: SceneEntityCfg | None = None,
    cup_center_xy: tuple[float, float] = PEN_CUP_DEFAULT_CENTER_XY,
    radius: float = 0.05,
    height_range: tuple[float, float] = (0.005, 0.18),
) -> torch.Tensor:
    """Pen position is inside the cup footprint (no gripper condition)."""
    pen: RigidObject = env.scene[object_cfg.name]
    pen_pos = pen.data.root_pos_w - env.scene.env_origins

    if cup_cfg is not None:
        cup = env.scene[cup_cfg.name]
        cup_pos = cup.data.root_pos_w - env.scene.env_origins
        cx, cy, cz = cup_pos[:, 0], cup_pos[:, 1], cup_pos[:, 2]
    else:
        cx = torch.full((env.num_envs,), cup_center_xy[0], device=env.device)
        cy = torch.full((env.num_envs,), cup_center_xy[1], device=env.device)
        cz = torch.full((env.num_envs,), DESK_TOP_Z, device=env.device)

    inside_xy = torch.hypot(pen_pos[:, 0] - cx, pen_pos[:, 1] - cy) < radius
    above_floor = pen_pos[:, 2] > (cz + height_range[0])
    below_lip = pen_pos[:, 2] < (cz + height_range[1])
    return torch.logical_and(inside_xy, torch.logical_and(above_floor, below_lip))


def pen_above_cup_xy(
    env: ManagerBasedRLEnv | DirectRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("PenWhite"),
    cup_cfg: SceneEntityCfg | None = None,
    cup_center_xy: tuple[float, float] = PEN_CUP_DEFAULT_CENTER_XY,
    radius: float = 0.05,
) -> torch.Tensor:
    """Pen XY position is within the cup radius, regardless of height."""
    pen: RigidObject = env.scene[object_cfg.name]
    pen_pos = pen.data.root_pos_w - env.scene.env_origins

    if cup_cfg is not None:
        cup = env.scene[cup_cfg.name]
        cup_pos = cup.data.root_pos_w - env.scene.env_origins
        cx, cy = cup_pos[:, 0], cup_pos[:, 1]
    else:
        cx = torch.full((env.num_envs,), cup_center_xy[0], device=env.device)
        cy = torch.full((env.num_envs,), cup_center_xy[1], device=env.device)

    return torch.hypot(pen_pos[:, 0] - cx, pen_pos[:, 1] - cy) < radius


def pen_lifted(
    env: ManagerBasedRLEnv | DirectRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("PenWhite"),
    height_threshold: float = 0.05,
) -> torch.Tensor:
    """Pen is at least ``height_threshold`` (m) above the desk surface."""
    pen: RigidObject = env.scene[object_cfg.name]
    pen_z = pen.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return pen_z > (DESK_TOP_Z + height_threshold)


def ee_near_pen(
    env: ManagerBasedRLEnv | DirectRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("PenWhite"),
    distance_threshold: float = 0.05,
) -> torch.Tensor:
    """EE jaw position is within ``distance_threshold`` (m) of the pen."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    pen: RigidObject = env.scene[object_cfg.name]
    jaw_pos = ee_frame.data.target_pos_w[:, 1, :]
    pen_pos = pen.data.root_pos_w
    return torch.linalg.vector_norm(pen_pos - jaw_pos, dim=1) < distance_threshold
