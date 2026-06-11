"""Cube Pick-and-Place 전용 보상 함수."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from sim_to_real.tasks.common.mdp._geometry import DESK_TOP_Z
from sim_to_real.tasks.common.mdp.rewards import _container_xy, _make_object_cfgs, _object_pos_w


def over_bowl_drop_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_cfgs: list[SceneEntityCfg] | None = None,
    bowl_center_xy: tuple[float, float] = (2.2, -0.17),
    bowl_cfg: SceneEntityCfg | None = None,
    xy_range: float = 0.10,
    lift_min: float = 0.02,
    open_threshold: float = 0.60,
    close_ref: float = 0.20,
) -> torch.Tensor:
    """들린 큐브가 그릇 XY 위에 있을 때 **그리퍼를 여는 행동**을 dense 보상 — [0, num_cubes].

    carry/transport/place_height 는 '잡은 채' 보상하므로 정책이 그릇 위에서 큐브를
    들고만 있는 local optimum(release valley)에 빠진다. 이 term 은 inside 게이트 없이
    '그릇 위 + 들림 + 그리퍼 open_frac' 에 연속 gradient 를 주어, 그릇 위에서 손을 펴
    큐브를 떨어뜨리는 행동으로 부드럽게 유도한다(바닥까지 안 내려도 됨 — 그릇 내부가
    미끄러워 위에서 떨궈도 중앙으로 정착). open_frac = (gripper-close_ref)/(open_thr-close_ref).
    """
    cfgs = _make_object_cfgs(cube_cfgs)
    robot: Articulation = env.scene[robot_cfg.name]
    gripper = robot.data.joint_pos[:, -1]
    open_frac = ((gripper - close_ref) / max(open_threshold - close_ref, 1e-6)).clamp(0.0, 1.0)

    cx, cy = _container_xy(env, bowl_center_xy, bowl_cfg)
    total = torch.zeros(env.num_envs, device=env.device)
    for cfg in cfgs:
        cube_pos = _object_pos_w(env, cfg)
        local = cube_pos - env.scene.env_origins
        lifted = local[:, 2] > (DESK_TOP_Z + lift_min)
        xy_dist = torch.hypot(local[:, 0] - cx, local[:, 1] - cy)
        xy_rew = torch.clamp(1.0 - xy_dist / max(xy_range, 1e-6), 0.0, 1.0)
        total = total + lifted.float() * xy_rew * open_frac
    return total


# ---------------------------------------------------------------------------
# 큐브 변위 패널티 — 잡기 전 큐브를 쳐서 밀어내는 것 억제 (정밀 grasp proxy)
# ---------------------------------------------------------------------------


def cube_predisturb_penalty(
    env: ManagerBasedRLEnv,
    cube_cfgs: list[SceneEntityCfg] | None = None,
    lift_min: float = 0.02,
) -> torch.Tensor:
    """잡기 전(책상 위·안 들린) 큐브가 reset 초기 xy 에서 밀려난 거리 합(+값).

    제대로 정밀 접근하면 그리퍼/팔이 큐브를 안 쳐 변위≈0. 거칠게 접근해 큐브를 치면
    변위↑ → 패널티. 들어올린 큐브(lifted)는 의도된 이동이라 제외(들기 전 책상 위
    수평 변위만 본다). "큐브를 최소로 움직이며 감싸 잡기"를 유도 = 정밀 grasp 의 proxy.
    RewTerm weight 는 음수. ``env._cube_init_xy``(PickCubeEnv 가 reset 직후 저장,
    CUBE_NAMES 순서 (N, n_cubes, 2)) 가 없으면 0 — getattr 가드.
    """
    init_xy = getattr(env, "_cube_init_xy", None)
    if init_xy is None:
        return torch.zeros(env.num_envs, device=env.device)

    cfgs = _make_object_cfgs(cube_cfgs)
    total = torch.zeros(env.num_envs, device=env.device)
    for i, cfg in enumerate(cfgs):
        cube_pos = _object_pos_w(env, cfg)
        local = cube_pos - env.scene.env_origins
        lifted = local[:, 2] > (DESK_TOP_Z + lift_min)
        disp = torch.linalg.vector_norm(local[:, :2] - init_xy[:, i, :], dim=-1)
        total = total + (~lifted).float() * disp
    return total


# ---------------------------------------------------------------------------
# Place 단계 PBRS — potential-based reward shaping (유지 보상 누적 제거 → hover 차단)
# ---------------------------------------------------------------------------


def _place_potential(
    env: ManagerBasedRLEnv,
    cfgs: list[SceneEntityCfg],
    bowl_center_xy: tuple[float, float],
    bowl_cfg: SceneEntityCfg | None,
    bowl_radius: float,
    bowl_height_range: tuple[float, float],
    xy_range: float,
) -> torch.Tensor:
    """Place 진행 potential Φ(s) ∈ [0, num_cubes].

    큐브별: 그릇 안이면 1.0, 아니면 (0.3·xy근접 + 0.2·z하강) 의 부분 진행도.
    "그릇에 얼마나 다가갔나"를 단조로 나타내는 척도 — 절대 상태 함수라 PBRS 의
    Φ 로 쓰면 유지 시 보상 0(telescoping), 진행 시에만 +.
    """
    cx, cy = _container_xy(env, bowl_center_xy, bowl_cfg)
    z_min, z_max = bowl_height_range
    total = torch.zeros(env.num_envs, device=env.device)
    for cfg in cfgs:
        cube_pos = _object_pos_w(env, cfg)
        local = cube_pos - env.scene.env_origins
        inside = _cube_inside_bowl_mask(env, cube_pos, bowl_center_xy, bowl_radius, bowl_height_range, bowl_cfg)
        xy_dist = torch.hypot(local[:, 0] - cx, local[:, 1] - cy)
        xy_prog = torch.clamp(1.0 - xy_dist / max(xy_range, 1e-6), 0.0, 1.0)
        z_prog = torch.clamp(
            (local[:, 2] - DESK_TOP_Z - z_min) / max(z_max - z_min, 1e-6), 0.0, 1.0
        )
        phi = inside.float() * 1.0 + (~inside).float() * (0.3 * xy_prog + 0.2 * z_prog)
        total = total + phi
    return total


def _cube_inside_bowl_mask(
    env: ManagerBasedRLEnv,
    cube_pos: torch.Tensor,
    bowl_center_xy: tuple[float, float],
    radius: float,
    height_range: tuple[float, float],
    bowl_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """큐브가 그릇 안에 있는지 여부, shape (num_envs,) bool."""
    local_pos = cube_pos - env.scene.env_origins
    cx, cy = _container_xy(env, bowl_center_xy, bowl_cfg)
    inside_xy = torch.hypot(local_pos[:, 0] - cx, local_pos[:, 1] - cy) < radius
    above = local_pos[:, 2] > (DESK_TOP_Z + height_range[0])
    below = local_pos[:, 2] < (DESK_TOP_Z + height_range[1])
    return inside_xy & above & below


def place_pbrs_reward(
    env: ManagerBasedRLEnv,
    cube_cfgs: list[SceneEntityCfg] | None = None,
    bowl_center_xy: tuple[float, float] = (2.2, -0.17),
    bowl_cfg: SceneEntityCfg | None = None,
    bowl_radius: float = 0.05,
    bowl_height_range: tuple[float, float] = (0.005, 0.12),
    xy_range: float = 0.40,
    gamma: float = 0.997,
) -> torch.Tensor:
    """Potential-based reward shaping: ``r = γ·Φ(s_t) − Φ(s_{t-1})`` (Ng 1999).

    transport/place_height/insert 같은 dense '유지' 보상을 대체한다. 큐브가 그릇으로
    **진행할 때만** +, 같은 자리를 유지하면 ≈(γ−1)Φ<0(미세 손실) → hover 가 value 상
    이득이 안 됨(누적은 telescoping = γ^T Φ_T − Φ_0 라 유지로 안 늘어남). optimal policy
    불변(이론 보장). grasp 단계엔 적용 안 함(검증된 dense 보상 보존).

    이전 step Φ 는 PickCubeEnv 가 ``_place_potential_prev`` (N,) 로 보관·이 함수가
    매 step 갱신(side-effect). reset 직후엔 env 가 0 으로 초기화(첫 step jump 최소).
    버퍼 없으면 0 반환(다른 task 안전).
    """
    prev = getattr(env, "_place_potential_prev", None)
    if prev is None:
        return torch.zeros(env.num_envs, device=env.device)
    cfgs = _make_object_cfgs(cube_cfgs)
    phi_now = _place_potential(env, cfgs, bowl_center_xy, bowl_cfg, bowl_radius, bowl_height_range, xy_range)
    shaped = gamma * phi_now - prev
    env._place_potential_prev = phi_now.detach()
    return shaped


# ---------------------------------------------------------------------------
# 그릇 교란 패널티 — 운반/place 중 그릇을 밀치거나 엎는 것 억제
# ---------------------------------------------------------------------------


def bowl_disturb_penalty(
    env: ManagerBasedRLEnv,
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("Bowl"),
    disp_coef: float = 4.0,
) -> torch.Tensor:
    """그릇을 초기 pose 에서 밀친(xy 변위)·엎은(tilt) 정도에 비례한 패널티 신호(+값).

    PickCubeEnv 가 reset 직후 저장한 ``env._bowl_init_quat`` / ``env._bowl_init_xy`` 를
    기준으로, 현재 그릇 up-vector 의 기울기(1-cosθ ∈ [0,2])와 env-local xy 변위(m)를
    ``tilt + disp_coef*disp`` 로 합산한다. RewTerm weight 는 음수로 둔다. 버퍼가 없는
    env(다른 task)면 0 — getattr 가드. tilt 가 주신호(엎힘), disp 는 보조(밀림).
    """
    init_quat = getattr(env, "_bowl_init_quat", None)
    init_xy = getattr(env, "_bowl_init_xy", None)
    if init_quat is None or init_xy is None:
        return torch.zeros(env.num_envs, device=env.device)

    bowl: RigidObject = env.scene[bowl_cfg.name]

    def _up(q: torch.Tensor) -> torch.Tensor:
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        return torch.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1.0 - 2 * (x * x + y * y)], dim=-1)

    up_now = _up(bowl.data.root_quat_w)
    up_init = _up(init_quat)
    cos_ang = (up_now * up_init).sum(dim=-1).clamp(-1.0, 1.0)
    tilt = 1.0 - cos_ang  # [0,2]: 0=그대로, 1=90°, 2=완전 엎힘

    cur_xy = bowl.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    disp = torch.linalg.vector_norm(cur_xy - init_xy, dim=-1)
    return tilt + disp_coef * disp
