"""공통 단계형 보상 함수 — reach → grasp → lift → transport → place → release.

모든 공개 함수는 shape (num_envs,) 의 유한 float Tensor를 반환한다.
contact sensor 없이 robot body_pos_w 와 RigidObject root_pos_w 만 사용한다.
"""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from ._geometry import DESK_TOP_Z, JAW_GRASP_OFFSET, _quat_apply_wxyz


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------


def _get_gripper_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """jaw-offset grasp point world pos, shape (num_envs, 3)."""
    robot_cfg.resolve(env.scene)
    robot: Articulation = env.scene[robot_cfg.name]
    body_names: list[str] = robot.data.body_names
    if "jaw" in body_names:
        jaw_idx = body_names.index("jaw")
        offset = torch.tensor(JAW_GRASP_OFFSET, device=env.device, dtype=robot.data.body_pos_w.dtype)
        offset = offset.unsqueeze(0).expand(env.num_envs, -1)
        return robot.data.body_pos_w[:, jaw_idx, :] + _quat_apply_wxyz(
            robot.data.body_quat_w[:, jaw_idx, :], offset
        )
    ids = robot_cfg.body_ids
    if isinstance(ids, int):
        return robot.data.body_pos_w[:, ids, :]
    if isinstance(ids, slice):
        return robot.data.body_pos_w[:, ids, :][:, 0, :]
    return robot.data.body_pos_w[:, ids[0], :]


def _object_pos_w(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """물체 world pos, shape (num_envs, 3)."""
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_pos_w


def _container_xy(
    env: ManagerBasedRLEnv,
    container_center_xy: tuple[float, float],
    container_cfg: SceneEntityCfg | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """컨테이너 xy 를 env-local frame으로 반환한다."""
    if container_cfg is not None:
        container: RigidObject = env.scene[container_cfg.name]
        local = container.data.root_pos_w - env.scene.env_origins
        return local[:, 0], local[:, 1]
    cx = torch.full((env.num_envs,), container_center_xy[0], device=env.device)
    cy = torch.full((env.num_envs,), container_center_xy[1], device=env.device)
    return cx, cy


def _object_inside_container_mask(
    env: ManagerBasedRLEnv,
    object_pos: torch.Tensor,
    container_center_xy: tuple[float, float],
    radius: float,
    height_range: tuple[float, float],
    container_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """물체가 컨테이너 안에 있는지 여부, shape (num_envs,) bool."""
    local_pos = object_pos - env.scene.env_origins
    cx, cy = _container_xy(env, container_center_xy, container_cfg)
    inside_xy = torch.hypot(local_pos[:, 0] - cx, local_pos[:, 1] - cy) < radius
    above = local_pos[:, 2] > (DESK_TOP_Z + height_range[0])
    below = local_pos[:, 2] < (DESK_TOP_Z + height_range[1])
    return inside_xy & above & below


def _make_object_cfgs(object_cfgs: list[SceneEntityCfg] | None) -> list[SceneEntityCfg]:
    """object_cfgs 를 검증해 반환한다. None 전달은 허용하지 않는다.

    호출부(env_cfg RewTerm params)에서 반드시 명시 주입할 것.
    AGENTS.md: "reward/rl_state 기본값 의존 금지".
    """
    if object_cfgs is None:
        raise ValueError(
            "object_cfgs must be explicitly provided; "
            "implicit fallback to a task-specific default is not allowed. "
            "Pass SceneEntityCfg list via RewTerm params."
        )
    return object_cfgs


# ---------------------------------------------------------------------------
# Stage 1 — reach: EE → 가장 가까운 미배치 물체
# ---------------------------------------------------------------------------


def reach_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["gripper"]),
    object_cfgs: list[SceneEntityCfg] | None = None,
    container_center_xy: tuple[float, float] = (2.2, -0.17),
    container_cfg: SceneEntityCfg | None = None,
    container_radius: float = 0.05,
    container_height_range: tuple[float, float] = (0.005, 0.18),
    reach_range: float = 0.30,
) -> torch.Tensor:
    """EE와 가장 가까운 미배치 물체의 거리 기반 밀집 보상 — [0, 1].

    reach_range 이내: 선형 증가. 모든 물체 배치 완료 환경은 1.0.
    """
    cfgs = _make_object_cfgs(object_cfgs)
    ee_pos = _get_gripper_pos(env, robot_cfg)

    per_obj_dists = []
    all_placed = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    for cfg in cfgs:
        obj_pos = _object_pos_w(env, cfg)
        dist = torch.linalg.vector_norm(obj_pos - ee_pos, dim=1)
        placed = _object_inside_container_mask(
            env, obj_pos, container_center_xy, container_radius, container_height_range, container_cfg
        )
        masked = torch.where(placed, torch.full_like(dist, reach_range), dist)
        per_obj_dists.append(masked)
        all_placed = all_placed & placed

    min_dist = torch.stack(per_obj_dists, dim=0).min(dim=0).values
    reward = torch.clamp(1.0 - min_dist / reach_range, 0.0, 1.0)
    return torch.where(all_placed, torch.ones_like(reward), reward)


# ---------------------------------------------------------------------------
# Stage 2 — grasp: 그리퍼 닫힘 + 물체 근접 (미배치 한정)
# ---------------------------------------------------------------------------


def pregrasp_bonus(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["gripper"]),
    object_cfgs: list[SceneEntityCfg] | None = None,
    container_center_xy: tuple[float, float] = (2.2, -0.17),
    container_cfg: SceneEntityCfg | None = None,
    container_radius: float = 0.05,
    container_height_range: tuple[float, float] = (0.005, 0.18),
    diff_threshold: float = 0.08,
    close_threshold: float = 0.50,
) -> torch.Tensor:
    """EE 근접 + 그리퍼 닫힘 보상 (grasp_bonus 이전 탐색 장벽 낮춤)."""
    cfgs = _make_object_cfgs(object_cfgs)
    robot_cfg.resolve(env.scene)
    robot: Articulation = env.scene[robot_cfg.name]
    gripper_closed = robot.data.joint_pos[:, -1] < close_threshold
    ee_pos = _get_gripper_pos(env, robot_cfg)

    total = torch.zeros(env.num_envs, device=env.device)
    for cfg in cfgs:
        obj_pos = _object_pos_w(env, cfg)
        dist = torch.linalg.vector_norm(obj_pos - ee_pos, dim=1)
        placed = _object_inside_container_mask(
            env, obj_pos, container_center_xy, container_radius, container_height_range, container_cfg
        )
        total = total + ((dist < diff_threshold) & gripper_closed & ~placed).float()
    return total


def guided_lift_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["gripper"]),
    object_cfgs: list[SceneEntityCfg] | None = None,
    container_center_xy: tuple[float, float] = (2.2, -0.17),
    container_cfg: SceneEntityCfg | None = None,
    container_radius: float = 0.05,
    container_height_range: tuple[float, float] = (0.005, 0.18),
    diff_threshold: float = 0.10,
    close_threshold: float = 0.50,
    lift_start: float = 0.015,
    lift_height: float = 0.060,
) -> torch.Tensor:
    """pregrasp 상태에서 물체가 책상에서 떨어지는 초기 lift를 연속 보상한다."""
    cfgs = _make_object_cfgs(object_cfgs)
    robot_cfg.resolve(env.scene)
    robot: Articulation = env.scene[robot_cfg.name]
    gripper_closed = robot.data.joint_pos[:, -1] < close_threshold
    ee_pos = _get_gripper_pos(env, robot_cfg)
    span = max(lift_height - lift_start, 1e-6)

    total = torch.zeros(env.num_envs, device=env.device)
    for cfg in cfgs:
        obj_pos = _object_pos_w(env, cfg)
        local_z = obj_pos[:, 2] - env.scene.env_origins[:, 2]
        height_rew = torch.clamp((local_z - DESK_TOP_Z - lift_start) / span, 0.0, 1.0)
        dist = torch.linalg.vector_norm(obj_pos - ee_pos, dim=1)
        placed = _object_inside_container_mask(
            env, obj_pos, container_center_xy, container_radius, container_height_range, container_cfg
        )
        active = (dist < diff_threshold) & gripper_closed & ~placed
        total = total + active.float() * height_rew
    return total


def grasp_bonus(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["gripper"]),
    object_cfgs: list[SceneEntityCfg] | None = None,
    container_center_xy: tuple[float, float] = (2.2, -0.17),
    container_cfg: SceneEntityCfg | None = None,
    container_radius: float = 0.05,
    container_height_range: tuple[float, float] = (0.005, 0.18),
    diff_threshold: float = 0.07,
    close_threshold: float = 0.50,
    lift_min: float = 0.02,
) -> torch.Tensor:
    """그리퍼 닫힘 AND EE 근접 AND 물체 살짝 들린 상태 (미배치만) — [0, num_objects]."""
    cfgs = _make_object_cfgs(object_cfgs)
    robot_cfg.resolve(env.scene)
    robot: Articulation = env.scene[robot_cfg.name]
    gripper_closed = robot.data.joint_pos[:, -1] < close_threshold
    ee_pos = _get_gripper_pos(env, robot_cfg)

    total = torch.zeros(env.num_envs, device=env.device)
    for cfg in cfgs:
        obj_pos = _object_pos_w(env, cfg)
        obj_local_z = obj_pos[:, 2] - env.scene.env_origins[:, 2]
        lifted = obj_local_z > (DESK_TOP_Z + lift_min)
        dist = torch.linalg.vector_norm(obj_pos - ee_pos, dim=1)
        placed = _object_inside_container_mask(
            env, obj_pos, container_center_xy, container_radius, container_height_range, container_cfg
        )
        near = dist < diff_threshold
        total = total + (near & gripper_closed & lifted & ~placed).float()
    return total


# ---------------------------------------------------------------------------
# Stage 1.5 — grasp align: 열린 그리퍼를 물체에 정밀 3D 정렬
# ---------------------------------------------------------------------------


def grasp_align_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["gripper"]),
    object_cfgs: list[SceneEntityCfg] | None = None,
    container_center_xy: tuple[float, float] = (2.2, -0.17),
    container_cfg: SceneEntityCfg | None = None,
    container_radius: float = 0.05,
    container_height_range: tuple[float, float] = (0.005, 0.18),
    align_xy: float = 0.05,
    align_z: float = 0.06,
    open_target: float = 0.70,
) -> torch.Tensor:
    """열린 그리퍼 grasp point 를 미배치 물체에 정밀 정렬하는 밀집 보상 — [0, num_objects]."""
    cfgs = _make_object_cfgs(object_cfgs)
    robot_cfg.resolve(env.scene)
    robot: Articulation = env.scene[robot_cfg.name]
    ee_pos = _get_gripper_pos(env, robot_cfg)
    open_frac = torch.clamp(robot.data.joint_pos[:, -1] / max(open_target, 1e-6), 0.0, 1.0)

    total = torch.zeros(env.num_envs, device=env.device)
    for cfg in cfgs:
        obj_pos = _object_pos_w(env, cfg)
        delta = obj_pos - ee_pos
        xy_dist = torch.linalg.vector_norm(delta[:, :2], dim=1)
        z_dist = torch.abs(delta[:, 2])
        xy_rew = torch.clamp(1.0 - xy_dist / max(align_xy, 1e-6), 0.0, 1.0)
        z_rew = torch.clamp(1.0 - z_dist / max(align_z, 1e-6), 0.0, 1.0)
        placed = _object_inside_container_mask(
            env, obj_pos, container_center_xy, container_radius, container_height_range, container_cfg
        )
        total = total + (~placed).float() * xy_rew * z_rew * open_frac
    return total


# ---------------------------------------------------------------------------
# Stage 1.7 — grasp close: 정렬된 채 닫기
# ---------------------------------------------------------------------------


def grasp_close_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["gripper"]),
    object_cfgs: list[SceneEntityCfg] | None = None,
    container_center_xy: tuple[float, float] = (2.2, -0.17),
    container_cfg: SceneEntityCfg | None = None,
    container_radius: float = 0.05,
    container_height_range: tuple[float, float] = (0.005, 0.18),
    align_xy: float = 0.05,
    align_z: float = 0.06,
    open_target: float = 0.70,
) -> torch.Tensor:
    """grasp point 가 물체에 정밀 정렬된 채 그리퍼를 닫는 행동을 보상 — [0, num_objects]."""
    cfgs = _make_object_cfgs(object_cfgs)
    robot_cfg.resolve(env.scene)
    robot: Articulation = env.scene[robot_cfg.name]
    ee_pos = _get_gripper_pos(env, robot_cfg)
    closed_frac = torch.clamp(1.0 - robot.data.joint_pos[:, -1] / max(open_target, 1e-6), 0.0, 1.0)

    total = torch.zeros(env.num_envs, device=env.device)
    for cfg in cfgs:
        obj_pos = _object_pos_w(env, cfg)
        delta = obj_pos - ee_pos
        xy_rew = torch.clamp(1.0 - torch.linalg.vector_norm(delta[:, :2], dim=1) / max(align_xy, 1e-6), 0.0, 1.0)
        z_rew = torch.clamp(1.0 - torch.abs(delta[:, 2]) / max(align_z, 1e-6), 0.0, 1.0)
        placed = _object_inside_container_mask(
            env, obj_pos, container_center_xy, container_radius, container_height_range, container_cfg
        )
        total = total + (~placed).float() * xy_rew * z_rew * closed_frac
    return total


# ---------------------------------------------------------------------------
# Stage 1.8 — grasp contact: 양 손가락이 같은 물체에 접촉
# ---------------------------------------------------------------------------


def grasp_contact_reward(
    env: ManagerBasedRLEnv,
    object_cfgs: list[SceneEntityCfg] | None = None,
    container_center_xy: tuple[float, float] = (2.2, -0.17),
    container_cfg: SceneEntityCfg | None = None,
    container_radius: float = 0.05,
    container_height_range: tuple[float, float] = (0.005, 0.18),
    jaw_sensor: str = "contact_jaw",
    gripper_sensor: str = "contact_gripper",
    force_threshold: float = 0.1,
) -> torch.Tensor:
    """양 손가락(jaw·gripper)이 동일 물체에 접촉하면 보상 — [0, num_objects]."""
    cfgs = _make_object_cfgs(object_cfgs)
    zero = torch.zeros(env.num_envs, device=env.device)
    try:
        jaw_fm = env.scene.sensors[jaw_sensor].data.force_matrix_w
        grip_fm = env.scene.sensors[gripper_sensor].data.force_matrix_w
    except (KeyError, AttributeError):
        return zero
    if jaw_fm is None or grip_fm is None:
        return zero

    total = zero.clone()
    for i, cfg in enumerate(cfgs):
        if i >= jaw_fm.shape[2] or i >= grip_fm.shape[2]:
            break
        jaw_mag = torch.linalg.vector_norm(jaw_fm[:, 0, i, :], dim=-1)
        grip_mag = torch.linalg.vector_norm(grip_fm[:, 0, i, :], dim=-1)
        both = (jaw_mag > force_threshold) & (grip_mag > force_threshold)
        obj_pos = _object_pos_w(env, cfg)
        placed = _object_inside_container_mask(
            env, obj_pos, container_center_xy, container_radius, container_height_range, container_cfg
        )
        total = total + (both & ~placed).float()
    return total


# ---------------------------------------------------------------------------
# Stage 2.5 — carry: 닫힌 그리퍼 + 들린 물체 + 컨테이너 방향 진행
# ---------------------------------------------------------------------------


def carry_object(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["gripper"]),
    object_cfgs: list[SceneEntityCfg] | None = None,
    container_center_xy: tuple[float, float] = (2.2, -0.17),
    container_cfg: SceneEntityCfg | None = None,
    container_radius: float = 0.05,
    container_height_range: tuple[float, float] = (0.005, 0.18),
    lift_min: float = 0.02,
    carry_range: float = 0.40,
    diff_threshold: float = 0.10,
    close_threshold: float = 0.50,
) -> torch.Tensor:
    """닫힌 그리퍼 + 들린 물체 + 컨테이너 방향 XY 진행 밀집 보상 — [0, num_objects]."""
    cfgs = _make_object_cfgs(object_cfgs)
    robot_cfg.resolve(env.scene)
    robot: Articulation = env.scene[robot_cfg.name]
    gripper_closed = robot.data.joint_pos[:, -1] < close_threshold
    ee_pos = _get_gripper_pos(env, robot_cfg)

    cx, cy = _container_xy(env, container_center_xy, container_cfg)
    total = torch.zeros(env.num_envs, device=env.device)

    for cfg in cfgs:
        obj_pos = _object_pos_w(env, cfg)
        local = obj_pos - env.scene.env_origins
        obj_local_z = local[:, 2]
        lifted = obj_local_z > (DESK_TOP_Z + lift_min)
        dist_ee = torch.linalg.vector_norm(obj_pos - ee_pos, dim=1)
        near = dist_ee < diff_threshold
        placed = _object_inside_container_mask(
            env, obj_pos, container_center_xy, container_radius, container_height_range, container_cfg
        )
        xy_dist = torch.hypot(local[:, 0] - cx, local[:, 1] - cy)
        xy_rew = torch.clamp(1.0 - xy_dist / max(carry_range, 1e-6), 0.0, 1.0)
        carrying = gripper_closed & near & lifted & ~placed
        total = total + carrying.float() * xy_rew

    return total


# ---------------------------------------------------------------------------
# Stage 3 — lift: 물체를 책상에서 들어올린 높이
# ---------------------------------------------------------------------------


def lift_reward(
    env: ManagerBasedRLEnv,
    object_cfgs: list[SceneEntityCfg] | None = None,
    lift_height: float = 0.05,
    max_height: float = 0.20,
) -> torch.Tensor:
    """들어올린 물체 높이 기반 밀집 보상 (합산) — [0, num_objects]."""
    cfgs = _make_object_cfgs(object_cfgs)
    span = max(max_height - lift_height, 1e-6)
    total = torch.zeros(env.num_envs, device=env.device)

    for cfg in cfgs:
        obj_pos = _object_pos_w(env, cfg)
        obj_z = obj_pos[:, 2] - env.scene.env_origins[:, 2]
        h = torch.clamp(obj_z - DESK_TOP_Z - lift_height, 0.0, span)
        total = total + h / span
    return total


# ---------------------------------------------------------------------------
# Stage 4 — transport: 들어올린 물체의 XY 거리 → 컨테이너
# ---------------------------------------------------------------------------


def transport_reward(
    env: ManagerBasedRLEnv,
    object_cfgs: list[SceneEntityCfg] | None = None,
    container_center_xy: tuple[float, float] = (2.2, -0.17),
    container_cfg: SceneEntityCfg | None = None,
    lift_height: float = 0.05,
    transport_range: float = 0.40,
) -> torch.Tensor:
    """들어올린 물체의 XY-컨테이너 거리 기반 밀집 보상 (합산) — [0, num_objects]."""
    cfgs = _make_object_cfgs(object_cfgs)
    cx, cy = _container_xy(env, container_center_xy, container_cfg)
    total = torch.zeros(env.num_envs, device=env.device)

    for cfg in cfgs:
        obj_pos = _object_pos_w(env, cfg)
        local = obj_pos - env.scene.env_origins
        lifted = local[:, 2] > (DESK_TOP_Z + lift_height)
        xy_dist = torch.hypot(local[:, 0] - cx, local[:, 1] - cy)
        xy_rew = torch.clamp(1.0 - xy_dist / max(transport_range, 1e-6), 0.0, 1.0)
        total = total + lifted.float() * xy_rew
    return total


# ---------------------------------------------------------------------------
# Stage 4.5 — place height: 컨테이너 XY 근처에서 컨테이너 안 높이로 낮추기
# ---------------------------------------------------------------------------


def place_height_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["gripper"]),
    object_cfgs: list[SceneEntityCfg] | None = None,
    container_center_xy: tuple[float, float] = (2.2, -0.17),
    container_cfg: SceneEntityCfg | None = None,
    container_radius: float = 0.05,
    container_height_range: tuple[float, float] = (0.005, 0.18),
    target_height: float = 0.07,
    xy_range: float = 0.18,
    z_range: float = 0.16,
    lift_min: float = 0.02,
    diff_threshold: float = 0.12,
    close_threshold: float = 0.50,
    require_carry: bool = True,
) -> torch.Tensor:
    """운반 중인 물체를 컨테이너 XY 근처에서 컨테이너 안 높이로 낮추는 dense reward."""
    cfgs = _make_object_cfgs(object_cfgs)
    robot_cfg.resolve(env.scene)
    robot: Articulation = env.scene[robot_cfg.name]
    gripper_closed = robot.data.joint_pos[:, -1] < close_threshold
    ee_pos = _get_gripper_pos(env, robot_cfg)

    cx, cy = _container_xy(env, container_center_xy, container_cfg)
    total = torch.zeros(env.num_envs, device=env.device)

    for cfg in cfgs:
        obj_pos = _object_pos_w(env, cfg)
        local = obj_pos - env.scene.env_origins
        obj_local_z = local[:, 2]
        lifted = obj_local_z > (DESK_TOP_Z + lift_min)
        dist_ee = torch.linalg.vector_norm(obj_pos - ee_pos, dim=1)
        near = dist_ee < diff_threshold
        xy_dist = torch.hypot(local[:, 0] - cx, local[:, 1] - cy)
        xy_rew = torch.clamp(1.0 - xy_dist / max(xy_range, 1e-6), 0.0, 1.0)
        target_z = DESK_TOP_Z + target_height
        z_rew = torch.clamp(1.0 - torch.abs(obj_local_z - target_z) / max(z_range, 1e-6), 0.0, 1.0)
        carrying = gripper_closed & near & lifted
        active = carrying if require_carry else lifted
        total = total + active.float() * xy_rew * z_rew

    return total


# ---------------------------------------------------------------------------
# Stage 5 — insert: 컨테이너 안 삽입
# ---------------------------------------------------------------------------


def insert_reward(
    env: ManagerBasedRLEnv,
    object_cfgs: list[SceneEntityCfg] | None = None,
    container_center_xy: tuple[float, float] = (2.2, -0.17),
    container_cfg: SceneEntityCfg | None = None,
    container_radius: float = 0.05,
    container_height_range: tuple[float, float] = (0.005, 0.18),
) -> torch.Tensor:
    """컨테이너 안에 삽입된 물체 수 (그리퍼 조건 없음) — [0, num_objects]."""
    cfgs = _make_object_cfgs(object_cfgs)
    total = torch.zeros(env.num_envs, device=env.device)

    for cfg in cfgs:
        obj_pos = _object_pos_w(env, cfg)
        inside = _object_inside_container_mask(
            env, obj_pos, container_center_xy, container_radius, container_height_range, container_cfg
        )
        total = total + inside.float()
    return total


# ---------------------------------------------------------------------------
# Stage 6 — release: 컨테이너 안 + 그리퍼 열림
# ---------------------------------------------------------------------------


def release_bonus(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfgs: list[SceneEntityCfg] | None = None,
    container_center_xy: tuple[float, float] = (2.2, -0.17),
    container_cfg: SceneEntityCfg | None = None,
    container_radius: float = 0.05,
    container_height_range: tuple[float, float] = (0.005, 0.18),
    open_threshold: float = 0.60,
) -> torch.Tensor:
    """컨테이너 안 + 그리퍼 열림 조건 만족 물체 수 — [0, num_objects]."""
    cfgs = _make_object_cfgs(object_cfgs)
    robot: Articulation = env.scene[robot_cfg.name]
    gripper_open = robot.data.joint_pos[:, -1] > open_threshold
    total = torch.zeros(env.num_envs, device=env.device)

    for cfg in cfgs:
        obj_pos = _object_pos_w(env, cfg)
        inside = _object_inside_container_mask(
            env, obj_pos, container_center_xy, container_radius, container_height_range, container_cfg
        )
        total = total + (inside & gripper_open).float()
    return total


# ---------------------------------------------------------------------------
# Place 단계 PBRS — potential-based reward shaping
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
    """Place 진행 potential Φ(s) ∈ [0, num_objects]."""
    cx, cy = _container_xy(env, container_center_xy, container_cfg)
    z_min, z_max = container_height_range
    total = torch.zeros(env.num_envs, device=env.device)
    for cfg in cfgs:
        obj_pos = _object_pos_w(env, cfg)
        local = obj_pos - env.scene.env_origins
        inside = _object_inside_container_mask(
            env, obj_pos, container_center_xy, container_radius, container_height_range, container_cfg
        )
        xy_dist = torch.hypot(local[:, 0] - cx, local[:, 1] - cy)
        xy_prog = torch.clamp(1.0 - xy_dist / max(xy_range, 1e-6), 0.0, 1.0)
        z_prog = torch.clamp(
            (local[:, 2] - DESK_TOP_Z - z_min) / max(z_max - z_min, 1e-6), 0.0, 1.0
        )
        phi = inside.float() * 1.0 + (~inside).float() * (0.3 * xy_prog + 0.2 * z_prog)
        total = total + phi
    return total


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
    """Potential-based reward shaping: r = γ·Φ(s_t) − Φ(s_{t-1}) (Ng 1999).

    이전 step Φ 는 PickCubeEnv 가 ``_place_potential_prev`` (N,) 로 보관.
    버퍼 없으면 0 반환(다른 task 안전).
    """
    prev = getattr(env, "_place_potential_prev", None)
    if prev is None:
        return torch.zeros(env.num_envs, device=env.device)
    cfgs = _make_object_cfgs(object_cfgs)
    phi_now = _place_potential(
        env, cfgs, container_center_xy, container_cfg, container_radius, container_height_range, xy_range
    )
    shaped = gamma * phi_now - prev
    env._place_potential_prev = phi_now.detach()
    return shaped


# ---------------------------------------------------------------------------
# 전체 성공 보너스
# ---------------------------------------------------------------------------


def task_success_bonus(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfgs: list[SceneEntityCfg] | None = None,
    container_center_xy: tuple[float, float] = (2.2, -0.17),
    container_cfg: SceneEntityCfg | None = None,
    container_radius: float = 0.05,
    container_height_range: tuple[float, float] = (0.005, 0.18),
    open_threshold: float = 0.60,
    require_open: bool = True,
) -> torch.Tensor:
    """모든 대상이 배치 완료되면 1.0, 미완료 시 0.0."""
    cfgs = _make_object_cfgs(object_cfgs)
    robot: Articulation = env.scene[robot_cfg.name]
    gripper_open = robot.data.joint_pos[:, -1] > open_threshold
    all_placed = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    for cfg in cfgs:
        obj_pos = _object_pos_w(env, cfg)
        inside = _object_inside_container_mask(
            env, obj_pos, container_center_xy, container_radius, container_height_range, container_cfg
        )
        all_placed = all_placed & inside

    if require_open:
        all_placed = all_placed & gripper_open

    return all_placed.float()


# ---------------------------------------------------------------------------
# 시간 형상화 보상
# ---------------------------------------------------------------------------


def _all_placed_mask(
    env: ManagerBasedRLEnv,
    object_cfgs: list[SceneEntityCfg] | None,
    container_center_xy: tuple[float, float],
    container_cfg: SceneEntityCfg | None,
    container_radius: float,
    container_height_range: tuple[float, float],
) -> torch.Tensor:
    """모든 대상 물체가 컨테이너 안에 있으면 True, shape (num_envs,) bool."""
    cfgs = _make_object_cfgs(object_cfgs)
    all_placed = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for cfg in cfgs:
        obj_pos = _object_pos_w(env, cfg)
        inside = _object_inside_container_mask(
            env, obj_pos, container_center_xy, container_radius, container_height_range, container_cfg
        )
        all_placed = all_placed & inside
    return all_placed


def time_penalty(
    env: ManagerBasedRLEnv,
    object_cfgs: list[SceneEntityCfg] | None = None,
    container_center_xy: tuple[float, float] = (2.2, -0.17),
    container_cfg: SceneEntityCfg | None = None,
    container_radius: float = 0.05,
    container_height_range: tuple[float, float] = (0.005, 0.18),
) -> torch.Tensor:
    """과제 미완료 동안 매 step 1.0, 완료 시 0.0 (양수 반환, weight 음수로 패널티)."""
    all_placed = _all_placed_mask(
        env, object_cfgs, container_center_xy, container_cfg, container_radius, container_height_range
    )
    return torch.where(
        all_placed,
        torch.zeros(env.num_envs, device=env.device),
        torch.ones(env.num_envs, device=env.device),
    )


def early_finish_bonus(
    env: ManagerBasedRLEnv,
    object_cfgs: list[SceneEntityCfg] | None = None,
    container_center_xy: tuple[float, float] = (2.2, -0.17),
    container_cfg: SceneEntityCfg | None = None,
    container_radius: float = 0.05,
    container_height_range: tuple[float, float] = (0.005, 0.18),
    scale: float = 1.0,
) -> torch.Tensor:
    """전부 배치된 env 에 한해 남은 시간 비율 보너스, 아니면 0.0."""
    all_placed = _all_placed_mask(
        env, object_cfgs, container_center_xy, container_cfg, container_radius, container_height_range
    )
    max_len = float(max(int(env.max_episode_length), 1))
    remaining = 1.0 - env.episode_length_buf.float() / max_len
    remaining = torch.clamp(remaining, 0.0, 1.0) * float(scale)
    return torch.where(all_placed, remaining, torch.zeros(env.num_envs, device=env.device))
