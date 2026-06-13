"""Cube Pick-and-Place 전용 관측 함수."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from sim_to_real.tasks.common.mdp._geometry import JAW_GRASP_OFFSET, _quat_apply_wxyz, _yaw_from_quat_wxyz


def grasp_focus_state(
    env: ManagerBasedRLEnv | DirectRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_name: str = "Cube1",
    gripper_body_name: str = "gripper",
) -> torch.Tensor:
    """RND novelty 전용 grasp 부분공간 상태 (~27-dim).

    전체 rl_state(83)는 reach 완성 후에도 EE 미세이동 등 grasp 무관 novelty 를 카운트해
    RND 탐색이 분산된다. 이 함수는 **grasp 와 직결된 차원만** 모아 RND novelty 를 정렬·닫기·
    들기 탐색에 집중시킨다: joint pos/vel + grasp point + gripper open + 활성 큐브 pos/vel +
    grasp→cube 상대 + cube yaw sin/cos. (단일 큐브 스테이지 = cube_name 1개 기준.)
    """
    robot: Articulation = env.scene[robot_cfg.name]
    origins = env.scene.env_origins

    joint_pos = robot.data.joint_pos          # (N, 6)
    joint_vel = robot.data.joint_vel          # (N, 6)

    body_names: list[str] = robot.data.body_names
    if "jaw" in body_names:
        jaw_idx = body_names.index("jaw")
        off = torch.tensor(JAW_GRASP_OFFSET, device=env.device, dtype=robot.data.body_pos_w.dtype)
        off = off.unsqueeze(0).expand(env.num_envs, -1)
        grasp_pos_w = robot.data.body_pos_w[:, jaw_idx, :] + _quat_apply_wxyz(robot.data.body_quat_w[:, jaw_idx, :], off)
        ee_idx = jaw_idx
    else:
        ee_idx = next((i for i, n in enumerate(body_names) if gripper_body_name in n), 0)
        grasp_pos_w = robot.data.body_pos_w[:, ee_idx, :]
    grasp_pos = grasp_pos_w - origins         # (N, 3)
    ee_vel = robot.data.body_lin_vel_w[:, ee_idx, :]  # (N, 3)
    gripper_open = joint_pos[:, -1:].clamp(0.0, 1.0)  # (N, 1)

    cube: RigidObject = env.scene[cube_name]
    cube_pos = cube.data.root_pos_w - origins  # (N, 3)
    cube_vel = cube.data.root_lin_vel_w        # (N, 3)
    rel = cube_pos - grasp_pos                 # (N, 3)
    yaw = _yaw_from_quat_wxyz(cube.data.root_quat_w)
    yaw_sc = torch.stack([torch.sin(yaw), torch.cos(yaw)], dim=-1)  # (N, 2)

    state = torch.cat(
        [joint_pos, joint_vel, grasp_pos, ee_vel, gripper_open, cube_pos, cube_vel, rel, yaw_sc],
        dim=-1,
    )  # 6+6+3+3+1+3+3+3+2 = 30
    return state.float()
