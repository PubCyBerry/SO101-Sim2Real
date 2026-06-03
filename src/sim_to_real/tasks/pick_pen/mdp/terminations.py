"""Pen Pick-and-Place success termination."""

from __future__ import annotations

import math

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


# Threshold for "at rest pose": all joints within ±15° of zero
_REST_THRESHOLD_RAD: float = 15.0 * math.pi / 180.0
_DESK_TOP_Z: float = 0.92


def _is_at_rest_pose(joint_pos: torch.Tensor) -> torch.Tensor:
    """All joints within _REST_THRESHOLD_RAD of zero (radians)."""
    return (joint_pos.abs() < _REST_THRESHOLD_RAD).all(dim=-1)


def task_done(
    env: ManagerBasedRLEnv | DirectRLEnv,
    pens_cfg: list[SceneEntityCfg],
    cup_center_xy: tuple[float, float] = (2.2, -0.17),
    radius: float = 0.05,
    height_range: tuple[float, float] = (0.005, 0.18),
    require_rest_pose: bool = True,
) -> torch.Tensor:
    """All listed pens are inside the cup footprint and the arm is at rest."""
    done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    cx = torch.full((env.num_envs,), cup_center_xy[0], device=env.device)
    cy = torch.full((env.num_envs,), cup_center_xy[1], device=env.device)

    for pen_cfg in pens_cfg:
        pen: RigidObject = env.scene[pen_cfg.name]
        pen_pos = pen.data.root_pos_w - env.scene.env_origins
        inside_xy = torch.hypot(pen_pos[:, 0] - cx, pen_pos[:, 1] - cy) < radius
        above_floor = pen_pos[:, 2] > (_DESK_TOP_Z + height_range[0])
        below_lip = pen_pos[:, 2] < (_DESK_TOP_Z + height_range[1])
        done = torch.logical_and(done, torch.logical_and(inside_xy, torch.logical_and(above_floor, below_lip)))

    if require_rest_pose:
        robot: Articulation = env.scene["robot"]
        done = torch.logical_and(done, _is_at_rest_pose(robot.data.joint_pos))

    return done
