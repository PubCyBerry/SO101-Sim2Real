"""Gripper effort helpers matching leisaac's teleop/replay contact behavior."""

from __future__ import annotations

import torch


def dynamic_reset_gripper_effort_limit_sim(
    env,
    teleop_device: str = "so101leader",
    *,
    min_effort: float = 0.5,
    max_effort: float = 10.0,
    mass_scale: float = 0.15,
    update_threshold: float = 0.05,
) -> None:
    """Adapt gripper effort to the nearest rigid object's mass.

    leisaac keeps the SO-101 actuator profile at a high maximum effort, but its
    teleop/replay loop lowers the gripper effort each step based on the nearest
    object's mass. This prevents the gripper from over-driving small objects
    into convex-decomposition collision hulls while keeping enough force to lift.
    """

    arms = []
    if "bi-so101leader" in teleop_device:
        for name in ("left_arm", "right_arm"):
            if name in env.scene.articulations:
                arms.append(env.scene.articulations[name])
    elif "robot" in env.scene.articulations:
        arms.append(env.scene["robot"])

    for arm in arms:
        _write_gripper_effort_limit(env, arm, min_effort, max_effort, mass_scale, update_threshold)


def _write_gripper_effort_limit(
    env,
    arm,
    min_effort: float,
    max_effort: float,
    mass_scale: float,
    update_threshold: float,
) -> None:
    object_positions = []
    object_masses = []
    for obj in env.scene._rigid_objects.values():
        object_positions.append(obj.data.body_link_pos_w[:, 0])
        object_masses.append(obj.data.default_mass)

    if not object_positions:
        return

    gripper_pos = arm.data.body_link_pos_w[:, -1]
    object_positions_t = torch.stack(object_positions)
    object_masses_t = torch.stack(object_masses)
    distances = torch.linalg.norm(object_positions_t - gripper_pos.unsqueeze(0), dim=2)
    _, min_indices = torch.min(distances, dim=0)

    target_masses = object_masses_t[min_indices.cpu(), 0, 0].to(arm._data.joint_effort_limits.device)
    target_limits = torch.clamp(target_masses / mass_scale, min=min_effort, max=max_effort)

    current_limits = arm._data.joint_effort_limits[:, -1]
    need_update = torch.abs(target_limits - current_limits) > update_threshold
    if torch.any(need_update):
        new_limits = current_limits.clone()
        new_limits[need_update] = target_limits[need_update]
        arm.write_joint_effort_limit_to_sim(limits=new_limits, joint_ids=[5 for _ in range(env.num_envs)])
