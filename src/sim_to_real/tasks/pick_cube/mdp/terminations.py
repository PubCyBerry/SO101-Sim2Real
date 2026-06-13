"""Cube Pick-and-Place 전용 종료 조건."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from sim_to_real.tasks.common.mdp._geometry import DESK_TOP_Z
from sim_to_real.tasks.common.mdp.rewards import _object_inside_container_mask
from sim_to_real.tasks.pick_cube.mdp.rewards import _over_bowl_grasped_mask


def cube_lost(
    env: ManagerBasedRLEnv | DirectRLEnv,
    objects_cfg: list[SceneEntityCfg],
    fall_z: float = 0.10,
) -> torch.Tensor:
    """활성 큐브 중 하나라도 책상보다 ``fall_z`` 아래로 추락하면 True (회복 불가).

    잘못된 grasp 로 큐브를 책상 밖/아래로 쳐내 영영 도달 불가가 된 상태를 빠르게
    종료해 학습 낭비를 막고(나머지 step 이 무의미), early termination 으로 '그 큐브
    가치 0' 을 critic 에 전파해 애초에 안 쳐내도록 압력을 준다. xy 멀리 밀침은
    cube_predisturb 패널티가 억제하고, 책상 끝을 넘으면 결국 z 로 잡힌다.

    비활성 큐브(지면 아래 z=-1.0 로 치워둔 것)는 objects_cfg(active 만)에서 제외되므로
    오탐하지 않는다 — apply_curriculum 이 active_cfgs 를 주입한다.
    """
    lost = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for cube_cfg in objects_cfg:
        cube: RigidObject = env.scene[cube_cfg.name]
        cube_z = cube.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
        lost = torch.logical_or(lost, cube_z < (DESK_TOP_Z - fall_z))
    return lost


def over_bowl_grasped(
    env: ManagerBasedRLEnv | DirectRLEnv,
    objects_cfg: list[SceneEntityCfg],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["gripper"]),
    container_center_xy: tuple[float, float] = (2.2, -0.17),
    container_cfg: SceneEntityCfg | None = None,
    over_bowl_xy: float = 0.10,
    lift_min: float = 0.02,
    grasp_dist: float = 0.07,
    close_threshold: float = 0.50,
) -> torch.Tensor:
    """skill1(acquire+transport) 종료 — 큐브를 그릇 위에서 grasp 한 채 들고 있으면 True.

    도달 즉시 종료 → dense 보상이 hover trap 으로 이어질 수 없다(skill chaining 의 핵심).
    skill2 의 reset 분포(=이 종료 상태)와 동일 판정. reward(over_bowl_grasped_bonus)와 공유.
    """
    return _over_bowl_grasped_mask(
        env, objects_cfg, robot_cfg, container_center_xy, container_cfg,
        over_bowl_xy, lift_min, grasp_dist, close_threshold,
    )


def cube_placed_open(
    env: ManagerBasedRLEnv | DirectRLEnv,
    objects_cfg: list[SceneEntityCfg],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    container_center_xy: tuple[float, float] = (2.2, -0.17),
    container_cfg: SceneEntityCfg | None = None,
    radius: float = 0.05,
    height_range: tuple[float, float] = (0.005, 0.12),
    open_threshold: float = 0.60,
) -> torch.Tensor:
    """skill2(place+release) 종료 — 모든 활성 큐브가 그릇 안 AND 그리퍼 열림이면 True.

    open 강제 = '닫은 채 그릇 안' 으로 success 회피 → release 행동을 반드시 학습(VLA 품질).
    """
    robot: Articulation = env.scene["robot"]
    done = robot.data.joint_pos[:, -1] > open_threshold  # gripper open
    for obj_cfg in objects_cfg:
        obj: RigidObject = env.scene[obj_cfg.name]
        inside = _object_inside_container_mask(
            env, obj.data.root_pos_w, container_center_xy, radius, height_range, container_cfg
        )
        done = torch.logical_and(done, inside)
    return done
