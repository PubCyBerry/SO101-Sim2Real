"""Cube Pick-and-Place 전용 보상 함수."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from sim_to_real.tasks.common.mdp._geometry import DESK_TOP_Z
from sim_to_real.tasks.common.mdp.rewards import (
    _container_xy,
    _get_gripper_pos,
    _make_object_cfgs,
    _object_pos_w,
)


def over_bowl_drop_pbrs_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfgs: list[SceneEntityCfg] | None = None,
    container_center_xy: tuple[float, float] = (2.2, -0.17),
    container_cfg: SceneEntityCfg | None = None,
    xy_range: float = 0.12,
    lift_min: float = 0.02,
    open_threshold: float = 0.60,
    close_ref: float = 0.35,
    gamma: float = 0.997,
) -> torch.Tensor:
    """PBRS: r = γΦ(s_t) − Φ(s_{t-1}) (Ng 1999) — over_bowl + 열기 유도.

    Φ(s) 설계: 들린 큐브가 그릇 XY 위 = 0.6 + 0.2·open_frac + 0.2·(1-z_norm)
               들렸으나 그릇 밖 = 0.3·xy_prog + 0.1·open_frac
               안 들림 = 0
    carry/guided_lift 와 직접 경쟁하지 않고(유지=≈(γ-1)Φ 미세 손실), 진행할 때만 +.
    이전 step Φ 는 PickCubeEnv._over_bowl_drop_potential_prev 가 보관(매 step 갱신).
    버퍼 없으면 0 반환(다른 task 안전).
    """
    prev = getattr(env, "_over_bowl_drop_potential_prev", None)
    if prev is None:
        return torch.zeros(env.num_envs, device=env.device)

    cfgs = _make_object_cfgs(object_cfgs)
    robot: Articulation = env.scene[robot_cfg.name]
    gripper = robot.data.joint_pos[:, -1]
    open_frac = ((gripper - close_ref) / max(open_threshold - close_ref, 1e-6)).clamp(0.0, 1.0)
    cx, cy = _container_xy(env, container_center_xy, container_cfg)

    total = torch.zeros(env.num_envs, device=env.device)
    for cfg in cfgs:
        cube_pos = _object_pos_w(env, cfg)
        local = cube_pos - env.scene.env_origins
        lifted = local[:, 2] > (DESK_TOP_Z + lift_min)
        xy_dist = torch.hypot(local[:, 0] - cx, local[:, 1] - cy)
        xy_prog = torch.clamp(1.0 - xy_dist / max(xy_range, 1e-6), 0.0, 1.0)
        z_norm = torch.clamp((local[:, 2] - DESK_TOP_Z - lift_min) / 0.15, 0.0, 1.0)
        over_bowl = lifted & (xy_dist < xy_range)
        phi = (over_bowl.float() * (0.6 + 0.2 * open_frac + 0.2 * (1.0 - z_norm)) +
               (~over_bowl).float() * lifted.float() * (0.3 * xy_prog + 0.1 * open_frac))
        total = total + phi

    shaped = gamma * total - prev
    env._over_bowl_drop_potential_prev = total.detach()
    return shaped


def over_bowl_drop_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfgs: list[SceneEntityCfg] | None = None,
    container_center_xy: tuple[float, float] = (2.2, -0.17),
    container_cfg: SceneEntityCfg | None = None,
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
    cfgs = _make_object_cfgs(object_cfgs)
    robot: Articulation = env.scene[robot_cfg.name]
    gripper = robot.data.joint_pos[:, -1]
    open_frac = ((gripper - close_ref) / max(open_threshold - close_ref, 1e-6)).clamp(0.0, 1.0)

    cx, cy = _container_xy(env, container_center_xy, container_cfg)
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
# Skill-1(acquire+transport) terminal — 그릇 위에서 큐브를 grasp 한 채 들고 있음
# ---------------------------------------------------------------------------


def _over_bowl_grasped_mask(
    env: ManagerBasedRLEnv,
    cfgs: list[SceneEntityCfg],
    robot_cfg: SceneEntityCfg,
    container_center_xy: tuple[float, float],
    container_cfg: SceneEntityCfg | None,
    over_bowl_xy: float,
    lift_min: float,
    grasp_dist: float,
    close_threshold: float,
) -> torch.Tensor:
    """활성 큐브 중 하나라도 '그릇 위 + 들림 + grasp(닫힘 & grasp point 근접)' 이면 True.

    skill chaining 의 handoff 상태(skill1 종료 = skill2 시작) 판정. (num_envs,) bool.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    gripper_closed = robot.data.joint_pos[:, -1] < close_threshold
    ee = _get_gripper_pos(env, robot_cfg)
    cx, cy = _container_xy(env, container_center_xy, container_cfg)
    mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for cfg in cfgs:
        obj = _object_pos_w(env, cfg)
        local = obj - env.scene.env_origins
        lifted = local[:, 2] > (DESK_TOP_Z + lift_min)
        xy_dist = torch.hypot(local[:, 0] - cx, local[:, 1] - cy)
        dist_ee = torch.linalg.vector_norm(obj - ee, dim=1)
        grasped = gripper_closed & (dist_ee < grasp_dist)
        mask = mask | (lifted & (xy_dist < over_bowl_xy) & grasped)
    return mask


def over_bowl_grasped_bonus(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["gripper"]),
    object_cfgs: list[SceneEntityCfg] | None = None,
    container_center_xy: tuple[float, float] = (2.2, -0.17),
    container_cfg: SceneEntityCfg | None = None,
    over_bowl_xy: float = 0.10,
    lift_min: float = 0.02,
    grasp_dist: float = 0.07,
    close_threshold: float = 0.50,
) -> torch.Tensor:
    """skill1 terminal 보너스 — over-bowl-grasped 상태에서 1.0, 아니면 0.0.

    종료(terminations.over_bowl_grasped)와 동일 조건이라, 도달 step 에 1회 지급되는
    terminal 보너스로 동작한다(episode 즉시 종료 → hover 누적 불가). weight 는 cfg 에서.
    """
    cfgs = _make_object_cfgs(object_cfgs)
    return _over_bowl_grasped_mask(
        env, cfgs, robot_cfg, container_center_xy, container_cfg,
        over_bowl_xy, lift_min, grasp_dist, close_threshold,
    ).float()


# ---------------------------------------------------------------------------
# 큐브 변위 패널티 — 잡기 전 큐브를 쳐서 밀어내는 것 억제 (정밀 grasp proxy)
# ---------------------------------------------------------------------------


def cube_predisturb_penalty(
    env: ManagerBasedRLEnv,
    object_cfgs: list[SceneEntityCfg] | None = None,
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

    cfgs = _make_object_cfgs(object_cfgs)
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
    container_center_xy: tuple[float, float],
    container_cfg: SceneEntityCfg | None,
    container_radius: float,
    container_height_range: tuple[float, float],
    xy_range: float,
) -> torch.Tensor:
    """Place 진행 potential Φ(s) ∈ [0, num_cubes].

    큐브별: 그릇 안이면 1.0, 아니면 (0.3·xy근접 + 0.2·z하강) 의 부분 진행도.
    "그릇에 얼마나 다가갔나"를 단조로 나타내는 척도 — 절대 상태 함수라 PBRS 의
    Φ 로 쓰면 유지 시 보상 0(telescoping), 진행 시에만 +.
    """
    cx, cy = _container_xy(env, container_center_xy, container_cfg)
    z_min, z_max = container_height_range
    total = torch.zeros(env.num_envs, device=env.device)
    for cfg in cfgs:
        cube_pos = _object_pos_w(env, cfg)
        local = cube_pos - env.scene.env_origins
        inside = _cube_inside_bowl_mask(env, cube_pos, container_center_xy, container_radius, container_height_range, container_cfg)
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
    container_center_xy: tuple[float, float],
    radius: float,
    height_range: tuple[float, float],
    container_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """큐브가 그릇 안에 있는지 여부, shape (num_envs,) bool."""
    local_pos = cube_pos - env.scene.env_origins
    cx, cy = _container_xy(env, container_center_xy, container_cfg)
    inside_xy = torch.hypot(local_pos[:, 0] - cx, local_pos[:, 1] - cy) < radius
    above = local_pos[:, 2] > (DESK_TOP_Z + height_range[0])
    below = local_pos[:, 2] < (DESK_TOP_Z + height_range[1])
    return inside_xy & above & below


def _task_progress_potential(
    env: ManagerBasedRLEnv,
    cfgs: list[SceneEntityCfg],
    robot_cfg: SceneEntityCfg,
    container_center_xy: tuple[float, float],
    container_cfg: SceneEntityCfg | None,
    container_radius: float,
    container_height_range: tuple[float, float],
    reach_range: float,
    grasp_dist: float,
    close_threshold: float,
    lift_min: float,
    lift_ref: float,
    transport_range: float,
) -> torch.Tensor:
    """전체 task 진행 potential Φ(s) ∈ [0, num_cubes] — reach→grasp→lift→transport→place 단조.

    큐브별(미배치):
      inside_bowl : 1.0
      lifted      : 0.4 + 0.2·lift_prog + 0.3·bowl_xy_prog (≤0.9)  ← grasp 무관(놓기 후 낙하도 유지)
      grasped     : 0.35 (closed+near, 책상 위)
      그 외       : 0.2·reach_prog (EE→큐브 접근)
    PBRS 로 쓰면 grasp 시 0.2→0.35 점프(+강화), 들기/운반 진행 +, 정지=0(hover 차단),
    놓기(그릇 위 들림→그릇 안) 0.9→1.0(+). dense '유지' 보상 없이 grasp 를 강화하면서
    hover 를 원천 차단(carry/lift/transport dense 의 hover income 문제 해결, T28)."""
    from sim_to_real.tasks.common.mdp.rewards import (
        _get_gripper_pos, _object_pos_w, _container_xy, _object_inside_container_mask,
    )
    from sim_to_real.tasks.common.mdp._geometry import DESK_TOP_Z as _DTZ

    robot_cfg.resolve(env.scene)
    robot = env.scene[robot_cfg.name]
    gripper_joint = robot.data.joint_pos[:, -1]
    closed = gripper_joint < close_threshold
    ee = _get_gripper_pos(env, robot_cfg)
    cx, cy = _container_xy(env, container_center_xy, container_cfg)
    total = torch.zeros(env.num_envs, device=env.device)
    for cfg in cfgs:
        obj = _object_pos_w(env, cfg)
        local = obj - env.scene.env_origins
        dist = torch.linalg.vector_norm(obj - ee, dim=1)
        inside = _object_inside_container_mask(
            env, obj, container_center_xy, container_radius, container_height_range, container_cfg
        )
        lifted = local[:, 2] > (_DTZ + lift_min)
        grasped = closed & (dist < grasp_dist)
        reach_prog = torch.clamp(1.0 - dist / max(reach_range, 1e-6), 0.0, 1.0)
        lift_prog = torch.clamp((local[:, 2] - _DTZ) / max(lift_ref, 1e-6), 0.0, 1.0)
        bowl_xy = torch.hypot(local[:, 0] - cx, local[:, 1] - cy)
        bowl_prog = torch.clamp(1.0 - bowl_xy / max(transport_range, 1e-6), 0.0, 1.0)
        phi_lifted = 0.4 + 0.2 * lift_prog + 0.3 * bowl_prog
        phi = torch.where(
            inside, torch.ones_like(reach_prog),
            torch.where(lifted, phi_lifted,
                        torch.where(grasped, torch.full_like(reach_prog, 0.35), 0.2 * reach_prog)),
        )
        total = total + phi
    return total


def task_progress_pbrs_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["gripper"]),
    object_cfgs: list[SceneEntityCfg] | None = None,
    container_center_xy: tuple[float, float] = (2.2, -0.17),
    container_cfg: SceneEntityCfg | None = None,
    container_radius: float = 0.06,
    container_height_range: tuple[float, float] = (0.005, 0.12),
    reach_range: float = 0.20,
    grasp_dist: float = 0.06,
    close_threshold: float = 0.50,
    lift_min: float = 0.02,
    lift_ref: float = 0.10,
    transport_range: float = 0.30,
    gamma: float = 0.997,
) -> torch.Tensor:
    """전체 task 진행 PBRS: ``r = γ·Φ(s_t) − Φ(s_{t-1})`` (Ng 1999).

    reach/grasp/lift/transport/place 의 dense 단계 보상을 **하나의 progress PBRS 로 대체**.
    grasp 를 강화(Φ 점프)하면서도 정지(hover)는 보상 0 — dense 유지 보상의 hover income
    문제(T28)와 grasp 미강화(v16~v19) 를 동시에 해결. **전용 버퍼**
    ``_task_progress_potential_prev`` 사용(reset 시 0). place_pbrs 의 _place_potential_prev 와
    분리 — 같은 버퍼면 RewardManager 가 두 함수 다 호출하며 telescoping 이 깨진다."""
    prev = getattr(env, "_task_progress_potential_prev", None)
    if prev is None:
        return torch.zeros(env.num_envs, device=env.device)
    cfgs = _make_object_cfgs(object_cfgs)
    phi_now = _task_progress_potential(
        env, cfgs, robot_cfg, container_center_xy, container_cfg, container_radius,
        container_height_range, reach_range, grasp_dist, close_threshold, lift_min, lift_ref, transport_range,
    )
    shaped = gamma * phi_now - prev
    env._task_progress_potential_prev = phi_now.detach()
    return shaped


def place_pbrs_reward(
    env: ManagerBasedRLEnv,
    object_cfgs: list[SceneEntityCfg] | None = None,
    container_center_xy: tuple[float, float] = (2.2, -0.17),
    container_cfg: SceneEntityCfg | None = None,
    container_radius: float = 0.05,
    container_height_range: tuple[float, float] = (0.005, 0.12),
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
    cfgs = _make_object_cfgs(object_cfgs)
    phi_now = _place_potential(env, cfgs, container_center_xy, container_cfg, container_radius, container_height_range, xy_range)
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
