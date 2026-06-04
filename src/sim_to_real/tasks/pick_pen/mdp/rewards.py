"""단계형 보상 함수 — reach → grasp → lift → transport → insert → release.

모든 공개 함수는 shape (num_envs,) 의 유한 float Tensor를 반환한다.
contact sensor 없이 robot body_pos_w 와 RigidObject root_pos_w 만 사용한다.
"""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from sim_to_real.utils.constant import PEN_NAMES

# world frame 책상 상판 z — PickPenSceneCfg 와 동기화 유지
_DESK_TOP_Z: float = 0.76
_JAW_GRASP_OFFSET: tuple[float, float, float] = (-0.021, -0.070, 0.020)


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------


def _quat_apply_wxyz(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """wxyz quaternion으로 vec를 회전한다."""
    qw = quat[:, 0:1]
    qv = quat[:, 1:4]
    uv = torch.cross(qv, vec, dim=-1)
    uuv = torch.cross(qv, uv, dim=-1)
    return vec + 2.0 * (qw * uv + uuv)


def _get_gripper_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """state machine과 같은 jaw-offset grasp point world pos, shape (num_envs, 3).

    body_names=["gripper"] 로 SceneEntityCfg 를 전달해야 한다.
    resolve() 는 멱등 호출 — 첫 스텝에 body_ids 를 캐시한다.
    """
    robot_cfg.resolve(env.scene)
    robot: Articulation = env.scene[robot_cfg.name]
    body_names: list[str] = robot.data.body_names
    if "jaw" in body_names:
        jaw_idx = body_names.index("jaw")
        offset = torch.tensor(_JAW_GRASP_OFFSET, device=env.device, dtype=robot.data.body_pos_w.dtype)
        offset = offset.unsqueeze(0).expand(env.num_envs, -1)
        return robot.data.body_pos_w[:, jaw_idx, :] + _quat_apply_wxyz(robot.data.body_quat_w[:, jaw_idx, :], offset)

    ids = robot_cfg.body_ids
    if isinstance(ids, int):
        return robot.data.body_pos_w[:, ids, :]
    if isinstance(ids, slice):
        return robot.data.body_pos_w[:, ids, :][:, 0, :]
    return robot.data.body_pos_w[:, ids[0], :]


def _pen_pos_w(env: ManagerBasedRLEnv, pen_cfg: SceneEntityCfg) -> torch.Tensor:
    """펜 world pos, shape (num_envs, 3)."""
    pen: RigidObject = env.scene[pen_cfg.name]
    return pen.data.root_pos_w


def _cup_xy(
    env: ManagerBasedRLEnv,
    cup_center_xy: tuple[float, float],
    cup_cfg: SceneEntityCfg | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """컵/그릇 xy를 env-local frame으로 반환한다."""
    if cup_cfg is not None:
        cup: RigidObject = env.scene[cup_cfg.name]
        local = cup.data.root_pos_w - env.scene.env_origins
        return local[:, 0], local[:, 1]

    cx = torch.full((env.num_envs,), cup_center_xy[0], device=env.device)
    cy = torch.full((env.num_envs,), cup_center_xy[1], device=env.device)
    return cx, cy


def _pen_inside_cup_mask(
    env: ManagerBasedRLEnv,
    pen_pos: torch.Tensor,
    cup_center_xy: tuple[float, float],
    radius: float,
    height_range: tuple[float, float],
    cup_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """펜이 컵 안에 있는지 여부, shape (num_envs,) bool."""
    local_pos = pen_pos - env.scene.env_origins
    cx, cy = _cup_xy(env, cup_center_xy, cup_cfg)
    inside_xy = torch.hypot(local_pos[:, 0] - cx, local_pos[:, 1] - cy) < radius
    above = local_pos[:, 2] > (_DESK_TOP_Z + height_range[0])
    below = local_pos[:, 2] < (_DESK_TOP_Z + height_range[1])
    return inside_xy & above & below


def _make_pen_cfgs(pen_cfgs: list[SceneEntityCfg] | None) -> list[SceneEntityCfg]:
    if pen_cfgs is None:
        return [SceneEntityCfg(n) for n in PEN_NAMES]
    return pen_cfgs


# ---------------------------------------------------------------------------
# Stage 1 — reach: EE → 가장 가까운 미배치 펜
# ---------------------------------------------------------------------------


def reach_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["gripper"]),
    pen_cfgs: list[SceneEntityCfg] | None = None,
    cup_center_xy: tuple[float, float] = (2.2, -0.17),
    cup_cfg: SceneEntityCfg | None = None,
    cup_radius: float = 0.05,
    cup_height_range: tuple[float, float] = (0.005, 0.18),
    reach_range: float = 0.30,
) -> torch.Tensor:
    """EE와 가장 가까운 미배치 펜의 거리 기반 밀집 보상 — [0, 1].

    reach_range 이내: 선형 증가. 모든 펜 배치 완료 환경은 1.0.
    """
    cfgs = _make_pen_cfgs(pen_cfgs)
    ee_pos = _get_gripper_pos(env, robot_cfg)  # (N, 3)

    per_pen_dists = []
    all_placed = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    for cfg in cfgs:
        pen_pos = _pen_pos_w(env, cfg)
        dist = torch.linalg.vector_norm(pen_pos - ee_pos, dim=1)
        placed = _pen_inside_cup_mask(env, pen_pos, cup_center_xy, cup_radius, cup_height_range, cup_cfg)
        # 배치된 펜은 reach_range 로 마스킹 (탐색 대상 제외)
        masked = torch.where(placed, torch.full_like(dist, reach_range), dist)
        per_pen_dists.append(masked)
        all_placed = all_placed & placed

    min_dist = torch.stack(per_pen_dists, dim=0).min(dim=0).values  # (N,)
    reward = torch.clamp(1.0 - min_dist / reach_range, 0.0, 1.0)
    # 이미 모두 배치됐으면 최대 보상
    return torch.where(all_placed, torch.ones_like(reward), reward)


# ---------------------------------------------------------------------------
# Stage 2 — grasp: 그리퍼 닫힘 + 펜 근접 (미배치 펜 한정)
# ---------------------------------------------------------------------------


def pregrasp_bonus(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["gripper"]),
    pen_cfgs: list[SceneEntityCfg] | None = None,
    cup_center_xy: tuple[float, float] = (2.2, -0.17),
    cup_cfg: SceneEntityCfg | None = None,
    cup_radius: float = 0.05,
    cup_height_range: tuple[float, float] = (0.005, 0.18),
    diff_threshold: float = 0.08,
    close_threshold: float = 0.50,
) -> torch.Tensor:
    """EE 근접 + 그리퍼 닫힘 보상.

    lifted 조건이 있는 grasp_bonus 이전 단계의 탐색 장벽을 낮춘다.
    """
    cfgs = _make_pen_cfgs(pen_cfgs)
    robot_cfg.resolve(env.scene)
    robot: Articulation = env.scene[robot_cfg.name]
    gripper_closed = robot.data.joint_pos[:, -1] < close_threshold
    ee_pos = _get_gripper_pos(env, robot_cfg)

    total = torch.zeros(env.num_envs, device=env.device)
    for cfg in cfgs:
        pen_pos = _pen_pos_w(env, cfg)
        dist = torch.linalg.vector_norm(pen_pos - ee_pos, dim=1)
        placed = _pen_inside_cup_mask(env, pen_pos, cup_center_xy, cup_radius, cup_height_range, cup_cfg)
        total = total + ((dist < diff_threshold) & gripper_closed & ~placed).float()
    return total


def grasp_bonus(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["gripper"]),
    pen_cfgs: list[SceneEntityCfg] | None = None,
    cup_center_xy: tuple[float, float] = (2.2, -0.17),
    cup_cfg: SceneEntityCfg | None = None,
    cup_radius: float = 0.05,
    cup_height_range: tuple[float, float] = (0.005, 0.18),
    diff_threshold: float = 0.07,
    close_threshold: float = 0.50,
    lift_min: float = 0.02,
) -> torch.Tensor:
    """그리퍼 닫힘 AND EE 근접 AND 펜 살짝 들린 상태 (미배치만).

    책상 위 정적 파지는 보상하지 않음 — 실제 픽업(lift_min 이상 상승)이어야 함.
    합산 — [0, 4].
    """
    cfgs = _make_pen_cfgs(pen_cfgs)
    robot_cfg.resolve(env.scene)
    robot: Articulation = env.scene[robot_cfg.name]
    gripper_closed = robot.data.joint_pos[:, -1] < close_threshold  # (N,)
    ee_pos = _get_gripper_pos(env, robot_cfg)  # (N, 3)

    total = torch.zeros(env.num_envs, device=env.device)
    for cfg in cfgs:
        pen_pos = _pen_pos_w(env, cfg)
        # 환경 원점 기준 local z 로 리프트 확인
        pen_local_z = pen_pos[:, 2] - env.scene.env_origins[:, 2]
        lifted = pen_local_z > (_DESK_TOP_Z + lift_min)
        dist = torch.linalg.vector_norm(pen_pos - ee_pos, dim=1)
        placed = _pen_inside_cup_mask(env, pen_pos, cup_center_xy, cup_radius, cup_height_range, cup_cfg)
        near = dist < diff_threshold
        total = total + (near & gripper_closed & lifted & ~placed).float()
    return total


# ---------------------------------------------------------------------------
# Stage 2.5 — carry: 닫힌 그리퍼 + 들린 펜 + 컵 방향 진행 (밀집 도우미)
# ---------------------------------------------------------------------------


def carry_pen(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["gripper"]),
    pen_cfgs: list[SceneEntityCfg] | None = None,
    cup_center_xy: tuple[float, float] = (2.2, -0.17),
    cup_cfg: SceneEntityCfg | None = None,
    cup_radius: float = 0.05,
    cup_height_range: tuple[float, float] = (0.005, 0.18),
    lift_min: float = 0.02,
    carry_range: float = 0.40,
    diff_threshold: float = 0.10,
    close_threshold: float = 0.50,
) -> torch.Tensor:
    """닫힌 그리퍼 + 들린 펜 + 컵 방향 XY 진행 밀집 보상.

    grasp_bonus 와 달리 연속값(XY 진행도)을 반환 — [0, num_pens].
    책상 위에서는 lifted=False 라 0이 되어 false-grasp 촉진을 차단한다.
    """
    cfgs = _make_pen_cfgs(pen_cfgs)
    robot_cfg.resolve(env.scene)
    robot: Articulation = env.scene[robot_cfg.name]
    gripper_closed = robot.data.joint_pos[:, -1] < close_threshold
    ee_pos = _get_gripper_pos(env, robot_cfg)

    cx, cy = _cup_xy(env, cup_center_xy, cup_cfg)
    total = torch.zeros(env.num_envs, device=env.device)

    for cfg in cfgs:
        pen_pos = _pen_pos_w(env, cfg)
        local = pen_pos - env.scene.env_origins
        pen_local_z = local[:, 2]

        lifted = pen_local_z > (_DESK_TOP_Z + lift_min)
        dist_ee = torch.linalg.vector_norm(pen_pos - ee_pos, dim=1)
        near = dist_ee < diff_threshold
        placed = _pen_inside_cup_mask(env, pen_pos, cup_center_xy, cup_radius, cup_height_range, cup_cfg)

        xy_dist = torch.hypot(local[:, 0] - cx, local[:, 1] - cy)
        xy_rew = torch.clamp(1.0 - xy_dist / max(carry_range, 1e-6), 0.0, 1.0)

        carrying = gripper_closed & near & lifted & ~placed
        total = total + carrying.float() * xy_rew

    return total


# ---------------------------------------------------------------------------
# Stage 3 — lift: 펜을 책상에서 들어올린 높이
# ---------------------------------------------------------------------------


def lift_reward(
    env: ManagerBasedRLEnv,
    pen_cfgs: list[SceneEntityCfg] | None = None,
    lift_height: float = 0.05,
    max_height: float = 0.20,
) -> torch.Tensor:
    """들어올린 펜 높이 기반 밀집 보상 (모든 펜 합산) — [0, num_pens].

    lift_height ~ max_height 구간에서 펜 당 [0, 1] 선형 증가.
    """
    cfgs = _make_pen_cfgs(pen_cfgs)
    span = max(max_height - lift_height, 1e-6)
    total = torch.zeros(env.num_envs, device=env.device)

    for cfg in cfgs:
        pen_pos = _pen_pos_w(env, cfg)
        pen_z = pen_pos[:, 2] - env.scene.env_origins[:, 2]
        h = torch.clamp(pen_z - _DESK_TOP_Z - lift_height, 0.0, span)
        total = total + h / span
    return total


# ---------------------------------------------------------------------------
# Stage 4 — transport: 들어올린 펜의 XY 거리 → 컵
# ---------------------------------------------------------------------------


def transport_reward(
    env: ManagerBasedRLEnv,
    pen_cfgs: list[SceneEntityCfg] | None = None,
    cup_center_xy: tuple[float, float] = (2.2, -0.17),
    cup_cfg: SceneEntityCfg | None = None,
    lift_height: float = 0.05,
    transport_range: float = 0.40,
) -> torch.Tensor:
    """들어올린 펜의 XY-컵 거리 기반 밀집 보상 (합산) — [0, num_pens].

    들어올린 펜만 계산. cup 반경 이내면 펜 당 1.0.
    """
    cfgs = _make_pen_cfgs(pen_cfgs)
    cx, cy = _cup_xy(env, cup_center_xy, cup_cfg)
    total = torch.zeros(env.num_envs, device=env.device)

    for cfg in cfgs:
        pen_pos = _pen_pos_w(env, cfg)
        local = pen_pos - env.scene.env_origins
        lifted = local[:, 2] > (_DESK_TOP_Z + lift_height)
        xy_dist = torch.hypot(local[:, 0] - cx, local[:, 1] - cy)
        xy_rew = torch.clamp(1.0 - xy_dist / max(transport_range, 1e-6), 0.0, 1.0)
        total = total + lifted.float() * xy_rew
    return total


# ---------------------------------------------------------------------------
# Stage 4.5 — place height: 컵 XY 근처에서 컵 안 높이로 낮추기
# ---------------------------------------------------------------------------


def place_height_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["gripper"]),
    pen_cfgs: list[SceneEntityCfg] | None = None,
    cup_center_xy: tuple[float, float] = (2.2, -0.17),
    cup_cfg: SceneEntityCfg | None = None,
    cup_radius: float = 0.05,
    cup_height_range: tuple[float, float] = (0.005, 0.18),
    target_height: float = 0.07,
    xy_range: float = 0.18,
    z_range: float = 0.16,
    lift_min: float = 0.02,
    diff_threshold: float = 0.12,
    close_threshold: float = 0.50,
    require_carry: bool = True,
) -> torch.Tensor:
    """운반 중인 펜을 컵 XY 근처에서 컵 안 높이로 낮추는 dense reward.

    transport_reward 는 XY 접근만 보상하므로, 정책이 컵 위에서 계속 높게 들고
    있는 collapse가 생길 수 있다. 이 term은 닫힌 그리퍼가 실제로 펜을 들고
    있을 때만 컵 중심 XY와 목표 z를 동시에 맞추도록 보상한다.
    """
    cfgs = _make_pen_cfgs(pen_cfgs)
    robot_cfg.resolve(env.scene)
    robot: Articulation = env.scene[robot_cfg.name]
    gripper_closed = robot.data.joint_pos[:, -1] < close_threshold
    ee_pos = _get_gripper_pos(env, robot_cfg)

    cx, cy = _cup_xy(env, cup_center_xy, cup_cfg)
    total = torch.zeros(env.num_envs, device=env.device)

    for cfg in cfgs:
        pen_pos = _pen_pos_w(env, cfg)
        local = pen_pos - env.scene.env_origins
        pen_local_z = local[:, 2]
        lifted = pen_local_z > (_DESK_TOP_Z + lift_min)
        dist_ee = torch.linalg.vector_norm(pen_pos - ee_pos, dim=1)
        near = dist_ee < diff_threshold
        xy_dist = torch.hypot(local[:, 0] - cx, local[:, 1] - cy)
        xy_rew = torch.clamp(1.0 - xy_dist / max(xy_range, 1e-6), 0.0, 1.0)

        target_z = _DESK_TOP_Z + target_height
        z_rew = torch.clamp(1.0 - torch.abs(pen_local_z - target_z) / max(z_range, 1e-6), 0.0, 1.0)

        carrying = gripper_closed & near & lifted
        active = carrying if require_carry else lifted
        total = total + active.float() * xy_rew * z_rew

    return total


# ---------------------------------------------------------------------------
# Stage 5 — insert: 컵 안 삽입 (그리퍼 조건 없음)
# ---------------------------------------------------------------------------


def insert_reward(
    env: ManagerBasedRLEnv,
    pen_cfgs: list[SceneEntityCfg] | None = None,
    cup_center_xy: tuple[float, float] = (2.2, -0.17),
    cup_cfg: SceneEntityCfg | None = None,
    cup_radius: float = 0.05,
    cup_height_range: tuple[float, float] = (0.005, 0.18),
) -> torch.Tensor:
    """컵 안에 삽입된 펜 수 (그리퍼 조건 없음) — [0, num_pens]."""
    cfgs = _make_pen_cfgs(pen_cfgs)
    total = torch.zeros(env.num_envs, device=env.device)

    for cfg in cfgs:
        pen_pos = _pen_pos_w(env, cfg)
        inside = _pen_inside_cup_mask(env, pen_pos, cup_center_xy, cup_radius, cup_height_range, cup_cfg)
        total = total + inside.float()
    return total


# ---------------------------------------------------------------------------
# Stage 6 — release: 컵 안 + 그리퍼 열림 (= pen_in_cup 달성 펜 수)
# ---------------------------------------------------------------------------


def release_bonus(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pen_cfgs: list[SceneEntityCfg] | None = None,
    cup_center_xy: tuple[float, float] = (2.2, -0.17),
    cup_cfg: SceneEntityCfg | None = None,
    cup_radius: float = 0.05,
    cup_height_range: tuple[float, float] = (0.005, 0.18),
    open_threshold: float = 0.60,
) -> torch.Tensor:
    """컵 안 + 그리퍼 열림 조건 만족 펜 수 — [0, num_pens]."""
    cfgs = _make_pen_cfgs(pen_cfgs)
    robot: Articulation = env.scene[robot_cfg.name]
    gripper_open = robot.data.joint_pos[:, -1] > open_threshold  # (N,)
    total = torch.zeros(env.num_envs, device=env.device)

    for cfg in cfgs:
        pen_pos = _pen_pos_w(env, cfg)
        inside = _pen_inside_cup_mask(env, pen_pos, cup_center_xy, cup_radius, cup_height_range, cup_cfg)
        total = total + (inside & gripper_open).float()
    return total


# ---------------------------------------------------------------------------
# 전체 성공 보너스 — 4개 펜 전부 배치 완료
# ---------------------------------------------------------------------------


def task_success_bonus(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pen_cfgs: list[SceneEntityCfg] | None = None,
    cup_center_xy: tuple[float, float] = (2.2, -0.17),
    cup_cfg: SceneEntityCfg | None = None,
    cup_radius: float = 0.05,
    cup_height_range: tuple[float, float] = (0.005, 0.18),
    open_threshold: float = 0.60,
    require_open: bool = True,
) -> torch.Tensor:
    """모든 대상이 배치 완료되면 1.0, 미완료 시 0.0."""
    cfgs = _make_pen_cfgs(pen_cfgs)
    robot: Articulation = env.scene[robot_cfg.name]
    gripper_open = robot.data.joint_pos[:, -1] > open_threshold
    all_placed = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    for cfg in cfgs:
        pen_pos = _pen_pos_w(env, cfg)
        inside = _pen_inside_cup_mask(env, pen_pos, cup_center_xy, cup_radius, cup_height_range, cup_cfg)
        all_placed = all_placed & inside

    if require_open:
        all_placed = all_placed & gripper_open

    return all_placed.float()
