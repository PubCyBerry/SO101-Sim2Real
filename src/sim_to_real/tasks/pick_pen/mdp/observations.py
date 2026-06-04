"""Pen Pick-and-Place subtask observation terms.

Three families live here:

- *Outcome* terms (``pen_grasped``, ``pen_in_cup``) — used by the manager-based
  observation group and the success termination. They include gripper state
  so a "placed" result implies the robot also released the pen.
- *Geometry-only* helpers (``pen_inside_cup``, ``pen_lifted``, ``ee_near_pen``,
  ``pen_above_cup_xy``) — used by the oracle's auto-scoring tracker so each
  sub-phase has an independent pass/fail signal even when the gripper
  command is wrong.
- *RL state* (``rl_state``) — privileged full state for RL training: joint pos,
  slew-limited action target, jaw-offset grasp point position, all pen/cup positions,
  relative vectors. Does NOT depend on FrameTransformer or ee_frame.
"""

from __future__ import annotations

from typing import Sequence

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer


PEN_CUP_DEFAULT_CENTER_XY: tuple[float, float] = (2.2, -0.17)
DESK_TOP_Z: float = 0.76
JAW_GRASP_OFFSET: tuple[float, float, float] = (-0.021, -0.070, 0.020)


def _quat_apply_wxyz(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """wxyz quaternion으로 vec를 회전한다."""
    qw = quat[:, 0:1]
    qv = quat[:, 1:4]
    uv = torch.cross(qv, vec, dim=-1)
    uuv = torch.cross(qv, uv, dim=-1)
    return vec + 2.0 * (qw * uv + uuv)


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
# RL privileged state (TB.3) — no FrameTransformer dependency
# ---------------------------------------------------------------------------


def rl_state(
    env: ManagerBasedRLEnv | DirectRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pen_names: Sequence[str] = ("PenWhite", "PenGray", "PenBlack", "PenBlue"),
    cup_name: str = "PenCup",
    gripper_body_name: str = "gripper",
    action_term_name: str = "arm",
) -> torch.Tensor:
    """Privileged state vector for RL training.

    Shape: (num_envs, D) where D = 6 + 6 + 3 + 4*3 + 3 + 4*3 + 1 = 43.

    Breakdown (all positions are env-origin-relative):
      [0:6]   robot joint positions in SO-101 order (rad)
      [6:12]  current processed joint target in SO-101 order (rad)
      [12:15] jaw-offset grasp point position (m)
      [15:27] pen positions, 4×3 in pen_names order (m)
      [27:30] cup position (m)
      [30:42] grasp point→pen relative vectors, 4×3 (m)
      [42]    gripper joint pos normalised to [0,1] (open fraction)
    """
    robot: Articulation = env.scene[robot_cfg.name]
    origins = env.scene.env_origins  # (N, 3)

    # --- joint positions (6) ---
    joint_pos = robot.data.joint_pos  # (N, num_joints)

    # --- current slew-limited action target (6) ---
    joint_target = _current_action_target(env, joint_pos, action_term_name)

    # --- grasp point position relative to env origin (3) ---
    body_names: list[str] = robot.data.body_names
    if "jaw" in body_names:
        jaw_idx = body_names.index("jaw")
        offset = torch.tensor(JAW_GRASP_OFFSET, device=env.device, dtype=robot.data.body_pos_w.dtype)
        offset = offset.unsqueeze(0).expand(env.num_envs, -1)
        grasp_pos = robot.data.body_pos_w[:, jaw_idx, :] + _quat_apply_wxyz(robot.data.body_quat_w[:, jaw_idx, :], offset)
        gripper_pos = grasp_pos - origins
    else:
        try:
            gripper_idx = body_names.index(gripper_body_name)
        except ValueError:
            # 폴백: "gripper" 포함 첫 번째 바디
            gripper_idx = next(
                (i for i, n in enumerate(body_names) if gripper_body_name in n), 0
            )
        gripper_pos = robot.data.body_pos_w[:, gripper_idx, :] - origins  # (N, 3)

    # --- pen positions relative to env origin (4×3) ---
    pen_parts: list[torch.Tensor] = []
    for name in pen_names:
        pen: RigidObject = env.scene[name]
        pen_parts.append(pen.data.root_pos_w - origins)  # (N, 3)
    pen_pos = torch.cat(pen_parts, dim=-1)  # (N, 12)

    # --- cup position relative to env origin (3) ---
    cup: RigidObject = env.scene[cup_name]
    cup_pos = cup.data.root_pos_w - origins  # (N, 3)

    # --- grasp point → pen relative vectors (4×3) ---
    rel_parts: list[torch.Tensor] = []
    for i in range(len(pen_names)):
        pen_p = pen_parts[i]  # (N, 3)
        rel_parts.append(pen_p - gripper_pos)
    rel_pos = torch.cat(rel_parts, dim=-1)  # (N, 12)

    # --- gripper open fraction normalised to [0, 1] (1) ---
    # gripper joint is the last joint; use simple min-max normalisation
    # full-open ≈ 1.0 rad, full-closed ≈ 0.0 rad (Feetech STS3215 limits)
    gripper_open = joint_pos[:, -1:].clamp(0.0, 1.0)  # (N, 1)

    state = torch.cat([joint_pos, joint_target, gripper_pos, pen_pos, cup_pos, rel_pos, gripper_open], dim=-1)
    return state.float()
