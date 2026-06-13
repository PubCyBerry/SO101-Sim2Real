"""공통 Pick-and-Place 종료 조건."""

from __future__ import annotations

import math

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from ._geometry import DESK_TOP_Z
from .rewards import _container_xy

_REST_THRESHOLD_RAD: float = 15.0 * math.pi / 180.0


def _is_at_rest_pose(joint_pos: torch.Tensor) -> torch.Tensor:
    """모든 joint 가 ±15° 이내이면 True."""
    return (joint_pos.abs() < _REST_THRESHOLD_RAD).all(dim=-1)


def task_done(
    env: ManagerBasedRLEnv | DirectRLEnv,
    objects_cfg: list[SceneEntityCfg],
    container_center_xy: tuple[float, float] = (2.2, -0.17),
    container_cfg: SceneEntityCfg | None = None,
    radius: float = 0.05,
    height_range: tuple[float, float] = (0.005, 0.18),
    require_rest_pose: bool = True,
) -> torch.Tensor:
    """모든 물체가 컨테이너 안에 있고 (require_rest_pose 시) 팔이 rest pose 이면 True."""
    done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    cx, cy = _container_xy(env, container_center_xy, container_cfg)

    for obj_cfg in objects_cfg:
        obj: RigidObject = env.scene[obj_cfg.name]
        obj_pos = obj.data.root_pos_w - env.scene.env_origins
        inside_xy = torch.hypot(obj_pos[:, 0] - cx, obj_pos[:, 1] - cy) < radius
        above_floor = obj_pos[:, 2] > (DESK_TOP_Z + height_range[0])
        below_lip = obj_pos[:, 2] < (DESK_TOP_Z + height_range[1])
        done = torch.logical_and(
            done,
            torch.logical_and(inside_xy, torch.logical_and(above_floor, below_lip)),
        )

    if require_rest_pose:
        robot: Articulation = env.scene["robot"]
        done = torch.logical_and(done, _is_at_rest_pose(robot.data.joint_pos))

    return done
