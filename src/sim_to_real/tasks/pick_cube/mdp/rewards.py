"""Cube Pick-and-Place 전용 보상 함수 — 레퍼런스(ref_repos/pick_and_place) 정합.

성공이 확인된 IsaacLab Lift-Cube-Place 레시피의 6개 보상항 중 dense 4항을 SO-101+그릇
환경에 이식한다(action_rate/joint_vel 페널티는 isaaclab.envs.mdp 의 표준 함수를 cfg 에서
직접 쓴다). 레퍼런스의 평면 마커 target_region → 우리 그릇(BOWL)으로 매핑하고, 높이는
DESK_TOP_Z(책상면) 기준으로 변환한다(레퍼런스는 책상=z0 절대좌표). 레퍼런스가 단일 객체
레시피이므로 active 큐브에 대해 per-cube 합산한다(active_objects=1 이면 레퍼런스와 수치 동일).
"""

from __future__ import annotations

import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from sim_to_real.tasks.common.mdp._geometry import DESK_TOP_Z
from sim_to_real.tasks.common.mdp.rewards import (
    _container_xy,
    _get_gripper_pos,
    _make_object_cfgs,
    _object_pos_w,
)


def _container_pos_local(
    env: ManagerBasedRLEnv,
    container_center_xy: tuple[float, float],
    container_cfg: SceneEntityCfg | None,
) -> torch.Tensor:
    """컨테이너(레퍼런스 target_region ≈ 우리 그릇) 3D env-local 위치, shape (num_envs, 3).

    container_cfg 가 있으면 그릇 실제 root pos(env-local), 없으면 center_xy + 책상면 z.
    """
    cx, cy = _container_xy(env, container_center_xy, container_cfg)
    if container_cfg is not None:
        container: RigidObject = env.scene[container_cfg.name]
        cz = container.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    else:
        cz = torch.full((env.num_envs,), DESK_TOP_Z, device=env.device)
    return torch.stack([cx, cy, cz], dim=-1)


def reaching_object_ref(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    object_cfgs: list[SceneEntityCfg] | None = None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["gripper"]),
) -> torch.Tensor:
    """[레퍼런스 reaching_object] EE→큐브 접근 tanh-커널: ``1 - tanh(|cube-ee|/std)``.

    active 큐브 합산(active=1 이면 레퍼런스와 동일). weight 1.0.
    """
    cfgs = _make_object_cfgs(object_cfgs)
    ee = _get_gripper_pos(env, robot_cfg)
    total = torch.zeros(env.num_envs, device=env.device)
    for cfg in cfgs:
        d = torch.linalg.vector_norm(_object_pos_w(env, cfg) - ee, dim=1)
        total = total + (1.0 - torch.tanh(d / std))
    return total


def lifting_object_dist_limit_ref(
    env: ManagerBasedRLEnv,
    minimal_height: float = 0.04,
    minimal_dist: float = 0.05,
    object_cfgs: list[SceneEntityCfg] | None = None,
    container_center_xy: tuple[float, float] = (2.2, -0.17),
    container_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """[레퍼런스 object_is_lifted_dist_limit] 그릇에서 멀고(3D dist≥minimal_dist) 들렸을 때
    (책상면 위 높이 h>minimal_height) 높이 비례 보상 ``clamp((h-minimal_height)/0.05, 0, 1)``.

    dist-게이트가 "그릇 근처서 들고 캠핑"을 차단하고 height-cap 이 상한을 둬 camp-free.
    들려면 큐브를 실제로 잡아야 하므로 암묵적 grasp 신호로도 작동. active 큐브 합산. weight 30.0.
    """
    cfgs = _make_object_cfgs(object_cfgs)
    target = _container_pos_local(env, container_center_xy, container_cfg)
    total = torch.zeros(env.num_envs, device=env.device)
    for cfg in cfgs:
        local = _object_pos_w(env, cfg) - env.scene.env_origins
        dist = torch.linalg.vector_norm(local - target, dim=1)
        h = local[:, 2] - DESK_TOP_Z
        cond = (dist >= minimal_dist) & (h > minimal_height)
        height_reward = torch.clamp((h - minimal_height) / 0.05, min=0.0, max=1.0)
        total = total + cond.float() * height_reward
    return total


def object_target_region_distance_ref(
    env: ManagerBasedRLEnv,
    std: float = 0.3,
    object_cfgs: list[SceneEntityCfg] | None = None,
    container_center_xy: tuple[float, float] = (2.2, -0.17),
    container_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """[레퍼런스 object_target_region_tracking] 큐브→그릇 접근 tanh-커널(3D): ``1 - tanh(|cube-그릇|/std)``.

    큐브를 그릇 **안**(3D center) 으로 끌어 "그릇에 넣기" 를 학습한다. active 큐브 합산. weight 16.0.
    """
    cfgs = _make_object_cfgs(object_cfgs)
    target = _container_pos_local(env, container_center_xy, container_cfg)
    total = torch.zeros(env.num_envs, device=env.device)
    for cfg in cfgs:
        local = _object_pos_w(env, cfg) - env.scene.env_origins
        dist = torch.linalg.vector_norm(local - target, dim=1)
        total = total + (1.0 - torch.tanh(dist / std))
    return total


def object_lowering_ref(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    minimal_dist: float = 0.05,
    object_cfgs: list[SceneEntityCfg] | None = None,
    container_center_xy: tuple[float, float] = (2.2, -0.17),
    container_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """[레퍼런스 object_lowering] 그릇 근처(3D dist<minimal_dist)에서 큐브가 **내려가는 동안만**
    ``1000·clamp(Δ하강,0)·(1-tanh(dist/std))``.

    delta(직전 높이−현재 높이) 보상이라 정지 시 0 → camp-free. 이전 높이는
    ``env._ref_prev_cube_z`` (num_envs, K) 버퍼에 보관·매 step 갱신(함수 자체 lazy init).
    reset 직후 상승 점프는 clamp(min=0) 으로 흡수. active 큐브 합산. weight 7.0.
    """
    cfgs = _make_object_cfgs(object_cfgs)
    target = _container_pos_local(env, container_center_xy, container_cfg)
    cur_z = torch.stack(
        [(_object_pos_w(env, cfg) - env.scene.env_origins)[:, 2] for cfg in cfgs], dim=1
    )  # (num_envs, K)
    prev = getattr(env, "_ref_prev_cube_z", None)
    if prev is None or prev.shape != cur_z.shape:
        env._ref_prev_cube_z = cur_z.detach().clone()
        return torch.zeros(env.num_envs, device=env.device)

    total = torch.zeros(env.num_envs, device=env.device)
    for i, cfg in enumerate(cfgs):
        local = _object_pos_w(env, cfg) - env.scene.env_origins
        dist = torch.linalg.vector_norm(local - target, dim=1)
        height_delta = prev[:, i] - cur_z[:, i]
        total = total + (
            1000.0
            * (dist < minimal_dist).float()
            * torch.clamp(height_delta, min=0.0)
            * (1.0 - torch.tanh(dist / std))
        )
    env._ref_prev_cube_z = cur_z.detach().clone()
    return total
