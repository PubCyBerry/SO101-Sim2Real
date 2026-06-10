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


def guided_lift_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["gripper"]),
    pen_cfgs: list[SceneEntityCfg] | None = None,
    cup_center_xy: tuple[float, float] = (2.2, -0.17),
    cup_cfg: SceneEntityCfg | None = None,
    cup_radius: float = 0.05,
    cup_height_range: tuple[float, float] = (0.005, 0.18),
    diff_threshold: float = 0.10,
    close_threshold: float = 0.50,
    lift_start: float = 0.015,
    lift_height: float = 0.060,
) -> torch.Tensor:
    """pregrasp 상태에서 물체가 책상에서 떨어지는 초기 lift를 연속 보상한다."""
    cfgs = _make_pen_cfgs(pen_cfgs)
    robot_cfg.resolve(env.scene)
    robot: Articulation = env.scene[robot_cfg.name]
    gripper_closed = robot.data.joint_pos[:, -1] < close_threshold
    ee_pos = _get_gripper_pos(env, robot_cfg)
    span = max(lift_height - lift_start, 1e-6)

    total = torch.zeros(env.num_envs, device=env.device)
    for cfg in cfgs:
        pen_pos = _pen_pos_w(env, cfg)
        local_z = pen_pos[:, 2] - env.scene.env_origins[:, 2]
        height_rew = torch.clamp((local_z - _DESK_TOP_Z - lift_start) / span, 0.0, 1.0)
        dist = torch.linalg.vector_norm(pen_pos - ee_pos, dim=1)
        placed = _pen_inside_cup_mask(env, pen_pos, cup_center_xy, cup_radius, cup_height_range, cup_cfg)
        active = (dist < diff_threshold) & gripper_closed & ~placed
        total = total + active.float() * height_rew
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
# Stage 1.5 — grasp align: 열린 그리퍼를 큐브에 정밀 3D 정렬 (탐색 valley 메움)
# ---------------------------------------------------------------------------


def grasp_align_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["gripper"]),
    pen_cfgs: list[SceneEntityCfg] | None = None,
    cup_center_xy: tuple[float, float] = (2.2, -0.17),
    cup_cfg: SceneEntityCfg | None = None,
    cup_radius: float = 0.05,
    cup_height_range: tuple[float, float] = (0.005, 0.18),
    align_xy: float = 0.05,
    align_z: float = 0.06,
    open_target: float = 0.70,
) -> torch.Tensor:
    """열린 그리퍼 grasp point 를 미배치 큐브에 정밀 정렬하는 밀집 보상 — [0, num_pens].

    reach_reward(0.30m 범위)는 거칠어 최종 cm 단위 접근을 형상화하지 못한다. 이 term 은
    grasp point 가 큐브에 ① XY 정밀 정렬 × ② Z 정밀 정렬 × ③ 그리퍼가 큐브를 받아들일
    만큼 열림(open_frac) 의 곱을 보상한다. "열린 채 정확히 위치" = 감싸기 직전 자세.

    설계 의도:
      - open_frac 항이 "닫은 채 옆에서 근접"(camping)을 보상에서 제외 → pregrasp local
        optimum 회피. 닫는 행동은 align 보상을 깎으므로, 닫기→들기 전이는 guided_lift/
        carry(상위 weight) + 부트스트랩 value 전파가 견인한다.
      - lift 게이트 없음 → grasp_bonus 의 chicken-egg(들려야 grasp 보상) 와 분리.
    """
    cfgs = _make_pen_cfgs(pen_cfgs)
    robot_cfg.resolve(env.scene)
    robot: Articulation = env.scene[robot_cfg.name]
    ee_pos = _get_gripper_pos(env, robot_cfg)  # (N, 3)
    # gripper open fraction [0,1] (rl_state 와 동일 규약: 마지막 joint, full-open≈1)
    open_frac = torch.clamp(robot.data.joint_pos[:, -1] / max(open_target, 1e-6), 0.0, 1.0)

    total = torch.zeros(env.num_envs, device=env.device)
    for cfg in cfgs:
        pen_pos = _pen_pos_w(env, cfg)
        delta = pen_pos - ee_pos
        xy_dist = torch.linalg.vector_norm(delta[:, :2], dim=1)
        z_dist = torch.abs(delta[:, 2])
        xy_rew = torch.clamp(1.0 - xy_dist / max(align_xy, 1e-6), 0.0, 1.0)
        z_rew = torch.clamp(1.0 - z_dist / max(align_z, 1e-6), 0.0, 1.0)
        placed = _pen_inside_cup_mask(env, pen_pos, cup_center_xy, cup_radius, cup_height_range, cup_cfg)
        total = total + (~placed).float() * xy_rew * z_rew * open_frac
    return total


# ---------------------------------------------------------------------------
# Stage 1.7 — grasp close: 정렬된 채 닫기 (align→lift valley 메움)
# ---------------------------------------------------------------------------


def grasp_close_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["gripper"]),
    pen_cfgs: list[SceneEntityCfg] | None = None,
    cup_center_xy: tuple[float, float] = (2.2, -0.17),
    cup_cfg: SceneEntityCfg | None = None,
    cup_radius: float = 0.05,
    cup_height_range: tuple[float, float] = (0.005, 0.18),
    align_xy: float = 0.05,
    align_z: float = 0.06,
    open_target: float = 0.70,
) -> torch.Tensor:
    """grasp point 가 큐브에 정밀 정렬된 채 그리퍼를 닫는 행동을 보상 — [0, num_pens].

    grasp_align_reward(열림+정렬)의 거울. closed_frac(닫힐수록 1)을 곱해 "정렬된 채
    닫기"를 보상한다. lift 게이트 없음 → align→close→lift 사이 보상 valley 제거:
    그리퍼가 열림→닫힘으로 가며 align(open_frac)은 줄지만 이 항(closed_frac)은 늘어
    연속 그래디언트가 된다. camping 차단: 큐브를 밀쳐 정렬이 깨지면 0.
    """
    cfgs = _make_pen_cfgs(pen_cfgs)
    robot_cfg.resolve(env.scene)
    robot: Articulation = env.scene[robot_cfg.name]
    ee_pos = _get_gripper_pos(env, robot_cfg)
    # 닫힘 정도 [0,1]: open_target(열림)→0, 그 이하로 닫을수록 1
    closed_frac = torch.clamp(1.0 - robot.data.joint_pos[:, -1] / max(open_target, 1e-6), 0.0, 1.0)

    total = torch.zeros(env.num_envs, device=env.device)
    for cfg in cfgs:
        pen_pos = _pen_pos_w(env, cfg)
        delta = pen_pos - ee_pos
        xy_rew = torch.clamp(1.0 - torch.linalg.vector_norm(delta[:, :2], dim=1) / max(align_xy, 1e-6), 0.0, 1.0)
        z_rew = torch.clamp(1.0 - torch.abs(delta[:, 2]) / max(align_z, 1e-6), 0.0, 1.0)
        placed = _pen_inside_cup_mask(env, pen_pos, cup_center_xy, cup_radius, cup_height_range, cup_cfg)
        total = total + (~placed).float() * xy_rew * z_rew * closed_frac
    return total


# ---------------------------------------------------------------------------
# Stage 1.8 — grasp contact: 양 손가락이 같은 큐브에 접촉 (ContactSensor, 직접 grasp 신호)
# ---------------------------------------------------------------------------


def grasp_contact_reward(
    env: ManagerBasedRLEnv,
    pen_cfgs: list[SceneEntityCfg] | None = None,
    cup_center_xy: tuple[float, float] = (2.2, -0.17),
    cup_cfg: SceneEntityCfg | None = None,
    cup_radius: float = 0.05,
    cup_height_range: tuple[float, float] = (0.005, 0.18),
    jaw_sensor: str = "contact_jaw",
    gripper_sensor: str = "contact_gripper",
    force_threshold: float = 0.1,
) -> torch.Tensor:
    """양 손가락(jaw·gripper)이 동일 큐브에 접촉하면 보상 — [0, num_pens].

    ContactSensor force_matrix_w(손가락↔큐브 접촉력)로 "실제 envelop grasp"를 직접 감지.
    기하 proxy(거리)보다 직접적 — 손가락 사이에 큐브가 끼었는지 물리 접촉으로 판정.
    필터 순서 = CUBE_NAMES = pen_cfgs(활성 큐브, 첫 N개) 순서와 동일하므로 index i 매칭.
    센서 데이터 미준비(None) 시 0 반환(첫 step 가드).
    """
    cfgs = _make_pen_cfgs(pen_cfgs)
    zero = torch.zeros(env.num_envs, device=env.device)
    try:
        jaw_fm = env.scene.sensors[jaw_sensor].data.force_matrix_w   # (N, 1, M, 3)
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
        pen_pos = _pen_pos_w(env, cfg)
        placed = _pen_inside_cup_mask(env, pen_pos, cup_center_xy, cup_radius, cup_height_range, cup_cfg)
        total = total + (both & ~placed).float()
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
# Stage 5.5 — over-bowl drop: 그릇 위에서 그리퍼 열기 유도 (release valley 메움)
# ---------------------------------------------------------------------------


def over_bowl_drop_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pen_cfgs: list[SceneEntityCfg] | None = None,
    cup_center_xy: tuple[float, float] = (2.2, -0.17),
    cup_cfg: SceneEntityCfg | None = None,
    xy_range: float = 0.10,
    lift_min: float = 0.02,
    open_threshold: float = 0.60,
    close_ref: float = 0.20,
) -> torch.Tensor:
    """들린 큐브가 그릇 XY 위에 있을 때 **그리퍼를 여는 행동**을 dense 보상 — [0, num_pens].

    carry/transport/place_height 는 '잡은 채' 보상하므로 정책이 그릇 위에서 큐브를
    들고만 있는 local optimum(release valley)에 빠진다. 이 term 은 inside 게이트 없이
    '그릇 위 + 들림 + 그리퍼 open_frac' 에 연속 gradient 를 주어, 그릇 위에서 손을 펴
    큐브를 떨어뜨리는 행동으로 부드럽게 유도한다(바닥까지 안 내려도 됨 — 그릇 내부가
    미끄러워 위에서 떨궈도 중앙으로 정착). open_frac = (gripper-close_ref)/(open_thr-close_ref).
    """
    cfgs = _make_pen_cfgs(pen_cfgs)
    robot: Articulation = env.scene[robot_cfg.name]
    gripper = robot.data.joint_pos[:, -1]
    open_frac = ((gripper - close_ref) / max(open_threshold - close_ref, 1e-6)).clamp(0.0, 1.0)

    cx, cy = _cup_xy(env, cup_center_xy, cup_cfg)
    total = torch.zeros(env.num_envs, device=env.device)
    for cfg in cfgs:
        pen_pos = _pen_pos_w(env, cfg)
        local = pen_pos - env.scene.env_origins
        lifted = local[:, 2] > (_DESK_TOP_Z + lift_min)
        xy_dist = torch.hypot(local[:, 0] - cx, local[:, 1] - cy)
        xy_rew = torch.clamp(1.0 - xy_dist / max(xy_range, 1e-6), 0.0, 1.0)
        total = total + lifted.float() * xy_rew * open_frac
    return total


# ---------------------------------------------------------------------------
# 큐브 변위 패널티 — 잡기 전 큐브를 쳐서 밀어내는 것 억제 (정밀 grasp proxy)
# ---------------------------------------------------------------------------


def cube_predisturb_penalty(
    env: ManagerBasedRLEnv,
    pen_cfgs: list[SceneEntityCfg] | None = None,
    lift_min: float = 0.02,
) -> torch.Tensor:
    """잡기 전(책상 위·안 들린) 큐브가 reset 초기 xy 에서 밀려난 거리 합(+값).

    제대로 정밀 접근하면 그리퍼/팔이 큐브를 안 쳐 변위≈0. 거칠게 접근해 큐브를 치면
    변위↑ → 패널티. 들어올린 큐브(lifted)는 의도된 이동이라 제외(들기 전 책상 위
    수평 변위만 본다). "큐브를 최소로 움직이며 감싸 잡기"를 유도 = 정밀 grasp 의 proxy.
    RewTerm weight 는 음수. ``env._cube_init_xy``(PickCubeEnv 가 reset 직후 저장,
    CUBE_NAMES 순서 (N, n_pens, 2)) 가 없으면 0 — getattr 가드.
    """
    init_xy = getattr(env, "_cube_init_xy", None)
    if init_xy is None:
        return torch.zeros(env.num_envs, device=env.device)

    cfgs = _make_pen_cfgs(pen_cfgs)
    total = torch.zeros(env.num_envs, device=env.device)
    for i, cfg in enumerate(cfgs):
        pen_pos = _pen_pos_w(env, cfg)
        local = pen_pos - env.scene.env_origins
        lifted = local[:, 2] > (_DESK_TOP_Z + lift_min)
        disp = torch.linalg.vector_norm(local[:, :2] - init_xy[:, i, :], dim=-1)
        total = total + (~lifted).float() * disp
    return total


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


# ---------------------------------------------------------------------------
# 속도 보상 — 느린 정책에 페널티, 빠른 성공에 보너스
#   reward hacking 아님: grasp/배치 판정·성공 반경은 손대지 않고 "시간"만 형상화.
# ---------------------------------------------------------------------------


def _all_placed_mask(
    env: ManagerBasedRLEnv,
    pen_cfgs: list[SceneEntityCfg] | None,
    cup_center_xy: tuple[float, float],
    cup_cfg: SceneEntityCfg | None,
    cup_radius: float,
    cup_height_range: tuple[float, float],
) -> torch.Tensor:
    """모든 대상 큐브가 그릇 안에 있으면 True, shape (num_envs,) bool.

    task_done / task_success_bonus 와 동일한 _pen_inside_cup_mask 기준을 쓴다
    (그리퍼 open 조건은 보지 않음 — 시간 형상화는 배치 완료 시점 기준).
    """
    cfgs = _make_pen_cfgs(pen_cfgs)
    all_placed = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for cfg in cfgs:
        pen_pos = _pen_pos_w(env, cfg)
        inside = _pen_inside_cup_mask(env, pen_pos, cup_center_xy, cup_radius, cup_height_range, cup_cfg)
        all_placed = all_placed & inside
    return all_placed


def time_penalty(
    env: ManagerBasedRLEnv,
    pen_cfgs: list[SceneEntityCfg] | None = None,
    cup_center_xy: tuple[float, float] = (2.2, -0.17),
    cup_cfg: SceneEntityCfg | None = None,
    cup_radius: float = 0.05,
    cup_height_range: tuple[float, float] = (0.005, 0.18),
) -> torch.Tensor:
    """과제 미완료 동안 매 control step 마다 1.0, 완료 시 0.0 반환.

    "경과 step 1회" 를 나타내는 양수 값을 반환하고, RewTerm weight 를 음수(예:
    -0.02)로 줘 실제 보상은 음수(페널티)가 되게 한다. 큐브가 전부 배치되면 0 이
    되어, 빠르게 성공할수록 누적 페널티가 작아진다. (weight 와 부호가 곱해져
    최종 보상이 정해지므로, 여기서 음수를 반환하면 weight 음수와 만나 페널티가
    아닌 보상이 되는 부호 버그가 생긴다 — 반드시 양수 반환.)
    """
    all_placed = _all_placed_mask(env, pen_cfgs, cup_center_xy, cup_cfg, cup_radius, cup_height_range)
    return torch.where(
        all_placed,
        torch.zeros(env.num_envs, device=env.device),
        torch.ones(env.num_envs, device=env.device),
    )


def early_finish_bonus(
    env: ManagerBasedRLEnv,
    pen_cfgs: list[SceneEntityCfg] | None = None,
    cup_center_xy: tuple[float, float] = (2.2, -0.17),
    cup_cfg: SceneEntityCfg | None = None,
    cup_radius: float = 0.05,
    cup_height_range: tuple[float, float] = (0.005, 0.18),
    scale: float = 1.0,
) -> torch.Tensor:
    """전부 배치된 env 에 한해 남은 시간 비율 보너스, 아니면 0.0.

    bonus = scale * (1 - episode_length_buf / max_episode_length), 범위 [0, scale].
    매 step 적용되므로 일찍 완료해 그 상태를 유지할수록 누적 보너스가 커진다 →
    빠르고 정확한(배치 유지) 정책이 선택된다. RewTerm weight 로 절대 크기를 조정.
    """
    all_placed = _all_placed_mask(env, pen_cfgs, cup_center_xy, cup_cfg, cup_radius, cup_height_range)
    max_len = float(max(int(env.max_episode_length), 1))
    remaining = 1.0 - env.episode_length_buf.float() / max_len
    remaining = torch.clamp(remaining, 0.0, 1.0) * float(scale)
    return torch.where(all_placed, remaining, torch.zeros(env.num_envs, device=env.device))
