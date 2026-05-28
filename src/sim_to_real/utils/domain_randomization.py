"""Domain randomization helpers for sim_to_real tasks.

Mirrors the public surface of ``leisaac.utils.domain_randomization`` (each
helper returns an :class:`isaaclab.managers.EventTermCfg` to drop into
``domain_randomization(env_cfg, random_options=[...])``) but adds non-rectangular
sampling shapes that the leisaac ``randomize_object_uniform`` cannot express:

- :func:`randomize_object_in_ellipse` — pen xy uniformly inside an axis-aligned
  ellipse centered at the authored default pose. Use when the desired cluster
  is wider along one axis than the other (e.g. pens scattered side-by-side on
  the mat).
- :func:`randomize_object_on_arc` — pen-cup xy along a forward-facing arc whose
  0° point is the authored default pose. Use when an object should swing
  left/right around the robot at a fixed radius.

The actual sampling code (``_randomize_*_fn``) follows the leisaac
``leisaac/enhance/envs/mdp/events.py`` patterns: receive ``env``, ``env_ids``,
and ``asset_cfg``; read the authored ``default_root_state``; write through
``RigidObject.write_root_pose_to_sim`` / ``write_root_velocity_to_sim``.
"""

from __future__ import annotations

import math

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg


# ---------------------------------------------------------------------------
# mdp functions (called by the event manager on reset)
# ---------------------------------------------------------------------------


def _randomize_object_in_ellipse_fn(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    x_radius: float,
    y_radius: float,
    yaw_range_deg: tuple[float, float],
) -> None:
    """Place the asset's xy uniformly inside an axis-aligned ellipse.

    Uses polar sampling with ``sqrt(u)`` radius so the distribution is uniform
    over the ellipse *area* (naive ``r ~ U[0,1]`` would bias toward the
    center).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    default = asset.data.default_root_state[env_ids].clone()

    n = len(env_ids)
    device = env.device

    r = torch.sqrt(torch.rand(n, device=device))
    theta = torch.rand(n, device=device) * (2.0 * math.pi)
    dx = x_radius * r * torch.cos(theta)
    dy = y_radius * r * torch.sin(theta)

    new_x = default[:, 0] + dx
    new_y = default[:, 1] + dy
    new_z = default[:, 2]

    min_yaw, max_yaw = yaw_range_deg
    if max_yaw - min_yaw > 0.0:
        yaw_delta = (torch.rand(n, device=device) * (max_yaw - min_yaw) + min_yaw) * (math.pi / 180.0)
        zero = torch.zeros(n, device=device)
        yaw_quat = math_utils.quat_from_euler_xyz(zero, zero, yaw_delta)
        new_quat = math_utils.quat_mul(default[:, 3:7], yaw_quat)
    else:
        new_quat = default[:, 3:7]

    positions = torch.stack([new_x, new_y, new_z], dim=-1) + env.scene.env_origins[env_ids]
    pose = torch.cat([positions, new_quat], dim=-1)

    asset.write_root_pose_to_sim(pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=env_ids)


def _randomize_object_on_arc_fn(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    radius: float,
    angle_range_deg: tuple[float, float],
) -> None:
    """Place the asset's xy on an arc whose 0° point is the authored default.

    Arc center is implicitly ``(default_x, default_y - radius)`` — i.e. the
    point ``radius`` meters behind the asset along +y (toward the robot).
    Positive angle rotates toward +x (robot's right), negative toward -x.

    Geometry::

        +y  forward
         ^
         |   * default (0°)
         |  /
         | /  radius
         |/
         o ----> +x  (positive angle direction)
        robot
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    default = asset.data.default_root_state[env_ids].clone()

    n = len(env_ids)
    device = env.device

    min_deg, max_deg = angle_range_deg
    angles_rad = (torch.rand(n, device=device) * (max_deg - min_deg) + min_deg) * (math.pi / 180.0)

    center_x = default[:, 0]
    center_y = default[:, 1] - radius

    new_x = center_x + radius * torch.sin(angles_rad)
    new_y = center_y + radius * torch.cos(angles_rad)
    new_z = default[:, 2]

    positions = torch.stack([new_x, new_y, new_z], dim=-1) + env.scene.env_origins[env_ids]
    pose = torch.cat([positions, default[:, 3:7]], dim=-1)

    asset.write_root_pose_to_sim(pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=env_ids)


# ---------------------------------------------------------------------------
# EventTermCfg wrappers (called from task __post_init__)
# ---------------------------------------------------------------------------


def randomize_object_in_ellipse(
    name: str,
    x_radius: float,
    y_radius: float,
    yaw_range_deg: tuple[float, float] = (0.0, 0.0),
) -> EventTerm:
    """Reset event that places ``name``'s xy uniformly inside an ellipse.

    Args:
        name: prim name registered via ``parse_usd_and_create_subassets``.
        x_radius: half-width of the ellipse (along world / scene-local +x).
        y_radius: half-depth of the ellipse (along world / scene-local +y).
        yaw_range_deg: extra yaw jitter (deg) applied on top of the authored
            orientation. ``(0, 0)`` keeps the authored yaw exactly.
    """
    return EventTerm(
        func=_randomize_object_in_ellipse_fn,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name),
            "x_radius": float(x_radius),
            "y_radius": float(y_radius),
            "yaw_range_deg": yaw_range_deg,
        },
    )


def randomize_object_on_arc(
    name: str,
    radius: float,
    angle_range_deg: tuple[float, float],
) -> EventTerm:
    """Reset event that places ``name``'s xy on a forward-facing arc.

    Args:
        name: prim name registered via ``parse_usd_and_create_subassets``.
        radius: arc radius (meters). Set to the authored distance from the
            robot base to the default object pose (e.g. PenCup default is
            0.22 m ahead of the robot, so ``radius=0.22``).
        angle_range_deg: ``(min, max)`` arc angle in degrees. ``(0, 0)`` is
            forward (+y in scene-local). Positive rotates toward +x (right),
            negative toward -x (left).
    """
    return EventTerm(
        func=_randomize_object_on_arc_fn,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name),
            "radius": float(radius),
            "angle_range_deg": angle_range_deg,
        },
    )
