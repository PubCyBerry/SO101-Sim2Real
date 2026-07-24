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


def _returned_home(
    env: ManagerBasedRLEnv | DirectRLEnv,
    pos_tol: float,
    moved_tol: float,
    hold_steps: int,
) -> torch.Tensor:
    """"이동 후 init 자세 복귀 + hold_steps 연속 정지" per-env 판정 (stateful, step당 1회 갱신).

    ``returned_home_after_motion`` 과 ``placed_and_returned`` 가 공유한다. TerminationManager
    가 두 term 을 같은 step 에 각각 호출하므로 ``common_step_counter`` 캐시로 카운터 중복
    증가를 막는다. q_init = ``default_joint_pos`` (드라이버가 init_state 로 설정 → 단일 소스).
    """
    step = int(env.common_step_counter)
    if getattr(env, "_ret_home_step", None) == step:
        return env._ret_home_done

    robot = env.scene["robot"]
    dist = torch.amax(torch.abs(robot.data.joint_pos - robot.data.default_joint_pos), dim=1)

    if not hasattr(env, "_ret_home_moved"):
        env._ret_home_moved = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._ret_home_hold = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    # 에피소드 시작 시 상태 리셋 (task_done_confirmed 와 동일 패턴)
    fresh = env.episode_length_buf <= 1
    env._ret_home_moved[fresh] = False
    env._ret_home_hold[fresh] = 0

    env._ret_home_moved = env._ret_home_moved | (dist > moved_tol)  # "움직였다" 래치
    at_home = dist < pos_tol
    env._ret_home_hold = torch.where(
        env._ret_home_moved & at_home,
        env._ret_home_hold + 1,
        torch.zeros_like(env._ret_home_hold),
    )
    env._ret_home_done = env._ret_home_hold >= hold_steps
    env._ret_home_step = step
    return env._ret_home_done


def returned_home_after_motion(
    env: ManagerBasedRLEnv | DirectRLEnv,
    pos_tol: float = 0.05,
    moved_tol: float = 0.15,
    hold_steps: int = 30,
) -> torch.Tensor:
    """datagen 에피소드 종료 — 이동을 경험한 뒤 init 자세(pos_tol rad)로 복귀해
    ``hold_steps``(30 step = 1 s @30 Hz) 연속 정지하면 True.

    pre-roll(이동 전 정지 구간)에서는 moved 래치가 False 라 발화하지 않고, 궤적 중간에
    init 근방을 스쳐가도 1 s 연속 정지가 아니면 발화하지 않는다. cuRobo 궤적은 retreat
    phase 가 init 으로 cspace 복귀하므로 back-padding(init hold)이 곧 post-hold 가 된다.
    """
    return _returned_home(env, pos_tol, moved_tol, hold_steps)


def placed_and_returned(
    env: ManagerBasedRLEnv | DirectRLEnv,
    cube_cfg: SceneEntityCfg,
    bowl_cfg: SceneEntityCfg,
    bowl_tol: float = 0.06,
    pos_tol: float = 0.05,
    moved_tol: float = 0.15,
    hold_steps: int = 30,
) -> torch.Tensor:
    """``returned_home_after_motion`` AND 큐브가 그릇 중심 xy ``bowl_tol`` 이내.

    record 모드에서 term 이름 "success" 로 등록 — stock RecorderManager 의
    ``record_pre_reset`` 이 auto-reset 순간 이 값을 읽어 episode success attr 로 기록한다.
    """
    returned = _returned_home(env, pos_tol, moved_tol, hold_steps)
    cube: RigidObject = env.scene[cube_cfg.name]
    bowl: RigidObject = env.scene[bowl_cfg.name]
    placed = torch.linalg.norm(cube.data.root_pos_w[:, :2] - bowl.data.root_pos_w[:, :2], dim=1) < bowl_tol
    return returned & placed


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
