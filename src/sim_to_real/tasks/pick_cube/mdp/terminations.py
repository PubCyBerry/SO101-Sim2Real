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
from sim_to_real.tasks.common.mdp.terminations import task_done


def task_done_confirmed(
    env: ManagerBasedRLEnv | DirectRLEnv,
    objects_cfg: list[SceneEntityCfg],
    container_center_xy: tuple[float, float] = (2.2, -0.17),
    container_cfg: SceneEntityCfg | None = None,
    radius: float = 0.05,
    height_range: tuple[float, float] = (0.005, 0.18),
    require_rest_pose: bool = False,
    confirm_steps: int = 15,
) -> torch.Tensor:
    """디바운스 성공 종료 — ``task_done`` 이 ``confirm_steps`` 연속 성립할 때만 True.

    leisaac Workshop ``vial_placed_on_rack_termination`` 의 confirm-counter 아이디어를
    우리 기하 성공 판정(``task_done``, 모든 큐브가 그릇 반경 안)에 씌운 것. 한 프레임
    떨림으로 큐브가 순간 반경에 들어왔다 나가는 가짜 성공(bowl 내부 미끄러짐 등)을 걸러
    eval 성공률을 안정화한다. 카운터는 env 인스턴스에 저장(``env._pick_success_counter``)해
    함수 속성 전역상태 충돌을 피한다.

    파라미터는 ``task_done`` 과 동일 + ``confirm_steps``. eval 변형에서 사용.
    """
    done_now = task_done(
        env,
        objects_cfg=objects_cfg,
        container_center_xy=container_center_xy,
        container_cfg=container_cfg,
        radius=radius,
        height_range=height_range,
        require_rest_pose=require_rest_pose,
    )

    if not hasattr(env, "_pick_success_counter"):
        env._pick_success_counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    # 에피소드 시작 시 카운터 리셋
    env._pick_success_counter[env.episode_length_buf <= 1] = 0

    env._pick_success_counter = torch.where(
        done_now,
        env._pick_success_counter + 1,
        torch.zeros_like(env._pick_success_counter),
    )
    return env._pick_success_counter >= confirm_steps


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
