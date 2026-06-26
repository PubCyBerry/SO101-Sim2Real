"""Cube Pick-and-Place 전용 종료 조건 (추론/데이터 substrate).

VLA-only 리팩토링: skill-chaining RL 종료(over_bowl_grasped·cube_placed_open)는 제거.
eval success 는 공용 ``task_done`` (common.mdp), 실패 컷은 아래 ``cube_lost`` 만 사용.
"""

from __future__ import annotations

import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from sim_to_real.tasks.common.mdp._geometry import DESK_TOP_Z


def cube_lost(
    env: ManagerBasedRLEnv | DirectRLEnv,
    objects_cfg: list[SceneEntityCfg],
    fall_z: float = 0.10,
) -> torch.Tensor:
    """활성 큐브 중 하나라도 책상보다 ``fall_z`` 아래로 추락하면 True (회복 불가).

    잘못된 grasp 로 큐브를 책상 밖/아래로 쳐내 영영 도달 불가가 된 에피소드를 빠르게
    종료해 eval 낭비를 막는다. 비활성 큐브(지면 아래 z=-1.0 로 치워둔 것)는
    objects_cfg(active 만)에서 제외되므로 오탐하지 않는다.
    """
    lost = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for cube_cfg in objects_cfg:
        cube: RigidObject = env.scene[cube_cfg.name]
        cube_z = cube.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
        lost = torch.logical_or(lost, cube_z < (DESK_TOP_Z - fall_z))
    return lost
