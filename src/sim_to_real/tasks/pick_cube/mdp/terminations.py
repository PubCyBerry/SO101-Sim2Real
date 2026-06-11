"""Cube Pick-and-Place 전용 종료 조건."""

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
