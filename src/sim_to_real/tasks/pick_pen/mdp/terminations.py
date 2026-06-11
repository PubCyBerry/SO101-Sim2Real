"""Pen Pick-and-Place success termination."""

from __future__ import annotations

import math

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


# Threshold for "at rest pose": all joints within ±15° of zero
_REST_THRESHOLD_RAD: float = 15.0 * math.pi / 180.0
_DESK_TOP_Z: float = 0.76


def _is_at_rest_pose(joint_pos: torch.Tensor) -> torch.Tensor:
    """All joints within _REST_THRESHOLD_RAD of zero (radians)."""
    return (joint_pos.abs() < _REST_THRESHOLD_RAD).all(dim=-1)


def _cup_xy(
    env: ManagerBasedRLEnv | DirectRLEnv,
    cup_center_xy: tuple[float, float],
    cup_cfg: SceneEntityCfg | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cup/bowl xy in env-local frame."""
    if cup_cfg is not None:
        cup: RigidObject = env.scene[cup_cfg.name]
        local = cup.data.root_pos_w - env.scene.env_origins
        return local[:, 0], local[:, 1]

    cx = torch.full((env.num_envs,), cup_center_xy[0], device=env.device)
    cy = torch.full((env.num_envs,), cup_center_xy[1], device=env.device)
    return cx, cy


def task_done(
    env: ManagerBasedRLEnv | DirectRLEnv,
    pens_cfg: list[SceneEntityCfg],
    cup_center_xy: tuple[float, float] = (2.2, -0.17),
    cup_cfg: SceneEntityCfg | None = None,
    radius: float = 0.05,
    height_range: tuple[float, float] = (0.005, 0.18),
    require_rest_pose: bool = True,
) -> torch.Tensor:
    """All listed pens are inside the cup footprint and the arm is at rest."""
    done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    cx, cy = _cup_xy(env, cup_center_xy, cup_cfg)

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


def cube_lost(
    env: ManagerBasedRLEnv | DirectRLEnv,
    pens_cfg: list[SceneEntityCfg],
    fall_z: float = 0.10,
) -> torch.Tensor:
    """활성 큐브 중 하나라도 책상보다 ``fall_z`` 아래로 추락하면 True (회복 불가).

    잘못된 grasp 로 큐브를 책상 밖/아래로 쳐내 영영 도달 불가가 된 상태를 빠르게
    종료해 학습 낭비를 막고(나머지 step 이 무의미), early termination 으로 '그 큐브
    가치 0' 을 critic 에 전파해 애초에 안 쳐내도록 압력을 준다. xy 멀리 밀침은
    cube_predisturb 패널티가 억제하고, 책상 끝을 넘으면 결국 z 로 잡힌다.

    비활성 큐브(지면 아래 z=-1.0 로 치워둔 것)는 pens_cfg(active 만)에서 제외되므로
    오탐하지 않는다 — apply_curriculum 이 active_cfgs 를 주입한다.
    """
    lost = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for pen_cfg in pens_cfg:
        pen: RigidObject = env.scene[pen_cfg.name]
        pen_z = pen.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
        lost = torch.logical_or(lost, pen_z < (_DESK_TOP_Z - fall_z))
    return lost
