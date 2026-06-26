"""Cube Pick-and-Place 전용 관측 함수 — 레퍼런스 정합 저차원 상태(ref_state)."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat

from sim_to_real.tasks.common.mdp._geometry import JAW_GRASP_OFFSET, _quat_apply_wxyz


def _euler3(quat: torch.Tensor) -> torch.Tensor:
    """wxyz quaternion → euler XYZ(roll,pitch,yaw) (N,3), rad (-π,π]."""
    r, p, y = euler_xyz_from_quat(quat)
    return torch.stack([r, p, y], dim=-1)


def ref_state(
    env: ManagerBasedRLEnv | DirectRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_name: str = "Cube1",
    container_name: str = "Bowl",
    gripper_body_name: str = "gripper",
) -> torch.Tensor:
    """레퍼런스(ref_repos/pick_and_place, IsaacLab Lift-Cube-Place) 정합 저차원 상태 (54-dim).

    구성 (순서대로 concat, 위치는 env-origin 상대; obs_normalization 이 상수 offset 흡수):
      joint_pos   6   (5축 + 그리퍼)
      joint_vel   6
      tcp pose    6   (grasp point pos 3 + euler RPY 3)
      tcp vel     6   (lin 3 + ang 3, world frame)
      cube pose   6   (pos 3 + euler RPY 3)
      cube vel    6   (lin 3 + ang 3)
      bowl pose   6   (pos 3 + euler RPY 3)
      bowl vel    6   (lin 3 + ang 3)
      last_action 6   (raw 정책 출력 — env.action_manager.action)

    TCP = jaw grasp point(JAW_GRASP_OFFSET 적용). 단일 큐브(cube_name) 전용.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    origins = env.scene.env_origins

    joint_pos = robot.data.joint_pos
    joint_vel = robot.data.joint_vel

    body_names: list[str] = robot.data.body_names
    if "jaw" in body_names:
        ee_idx = body_names.index("jaw")
        off = torch.tensor(JAW_GRASP_OFFSET, device=env.device, dtype=robot.data.body_pos_w.dtype)
        off = off.unsqueeze(0).expand(env.num_envs, -1)
        ee_quat = robot.data.body_quat_w[:, ee_idx, :]
        tcp_pos = robot.data.body_pos_w[:, ee_idx, :] + _quat_apply_wxyz(ee_quat, off) - origins
    else:
        ee_idx = next((i for i, n in enumerate(body_names) if gripper_body_name in n), 0)
        ee_quat = robot.data.body_quat_w[:, ee_idx, :]
        tcp_pos = robot.data.body_pos_w[:, ee_idx, :] - origins
    tcp_pose = torch.cat([tcp_pos, _euler3(ee_quat)], dim=-1)
    tcp_vel = torch.cat(
        [robot.data.body_lin_vel_w[:, ee_idx, :], robot.data.body_ang_vel_w[:, ee_idx, :]], dim=-1
    )

    cube: RigidObject = env.scene[cube_name]
    cube_pose = torch.cat([cube.data.root_pos_w - origins, _euler3(cube.data.root_quat_w)], dim=-1)
    cube_vel = torch.cat([cube.data.root_lin_vel_w, cube.data.root_ang_vel_w], dim=-1)

    bowl: RigidObject = env.scene[container_name]
    bowl_pose = torch.cat([bowl.data.root_pos_w - origins, _euler3(bowl.data.root_quat_w)], dim=-1)
    bowl_vel = torch.cat([bowl.data.root_lin_vel_w, bowl.data.root_ang_vel_w], dim=-1)

    last_action = env.action_manager.action

    return torch.cat(
        [joint_pos, joint_vel, tcp_pose, tcp_vel,
         cube_pose, cube_vel, bowl_pose, bowl_vel, last_action],
        dim=-1,
    ).float()
