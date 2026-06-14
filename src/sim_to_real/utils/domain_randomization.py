"""Domain randomization helpers for sim_to_real tasks.

Mirrors the public surface of ``leisaac.utils.domain_randomization`` (each
helper returns an :class:`isaaclab.managers.EventTermCfg` to drop into
``domain_randomization(env_cfg, random_options=[...])``) but adds non-rectangular
sampling shapes that the leisaac ``randomize_object_uniform`` cannot express:

- :func:`randomize_object_in_ellipse` — pen xy uniformly inside an axis-aligned
  ellipse centered at the authored default pose. Use when the desired cluster
  is wider along one axis than the other (e.g. pens scattered side-by-side on
  the mat).
- :func:`randomize_object_on_arc` — pen-cup xy along a forward-facing arc whose
  0° point is the authored default pose. Use when an object should swing
  left/right around the robot at a fixed radius.

The actual sampling code (``_randomize_*_fn``) follows the leisaac
``leisaac/enhance/envs/mdp/events.py`` patterns: receive ``env``, ``env_ids``,
and ``asset_cfg``; read the authored ``default_root_state``; write through
``RigidObject.write_root_pose_to_sim`` / ``write_root_velocity_to_sim``.
"""

from __future__ import annotations

import math

import isaaclab.envs.mdp as mdp
import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg


# ---------------------------------------------------------------------------
# mdp functions (called by the event manager on reset)
# ---------------------------------------------------------------------------


def _randomize_object_in_ellipse_fn(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    x_radius: float,
    y_radius: float,
    yaw_range_deg: tuple[float, float],
) -> None:
    """Place the asset's xy uniformly inside an axis-aligned ellipse.

    Uses polar sampling with ``sqrt(u)`` radius so the distribution is uniform
    over the ellipse *area* (naive ``r ~ U[0,1]`` would bias toward the
    center).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    default = asset.data.default_root_state[env_ids].clone()

    n = len(env_ids)
    device = env.device

    r = torch.sqrt(torch.rand(n, device=device))
    theta = torch.rand(n, device=device) * (2.0 * math.pi)
    dx = x_radius * r * torch.cos(theta)
    dy = y_radius * r * torch.sin(theta)

    new_x = default[:, 0] + dx
    new_y = default[:, 1] + dy
    new_z = default[:, 2]

    min_yaw, max_yaw = yaw_range_deg
    if max_yaw - min_yaw > 0.0:
        yaw_delta = (torch.rand(n, device=device) * (max_yaw - min_yaw) + min_yaw) * (math.pi / 180.0)
        zero = torch.zeros(n, device=device)
        yaw_quat = math_utils.quat_from_euler_xyz(zero, zero, yaw_delta)
        new_quat = math_utils.quat_mul(default[:, 3:7], yaw_quat)
    else:
        new_quat = default[:, 3:7]

    positions = torch.stack([new_x, new_y, new_z], dim=-1) + env.scene.env_origins[env_ids]
    pose = torch.cat([positions, new_quat], dim=-1)

    asset.write_root_pose_to_sim(pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=env_ids)


def _randomize_object_on_arc_fn(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    radius: float,
    angle_range_deg: tuple[float, float],
) -> None:
    """Place the asset's xy on an arc whose 0° point is the authored default.

    Arc center is implicitly ``(default_x, default_y - radius)`` — i.e. the
    point ``radius`` meters behind the asset along +y (toward the robot).
    Positive angle rotates toward +x (robot's right), negative toward -x.

    Geometry::

        +y  forward
         ^
         |   * default (0°)
         |  /
         | /  radius
         |/
         o ----> +x  (positive angle direction)
        robot
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    default = asset.data.default_root_state[env_ids].clone()

    n = len(env_ids)
    device = env.device

    min_deg, max_deg = angle_range_deg
    angles_rad = (torch.rand(n, device=device) * (max_deg - min_deg) + min_deg) * (math.pi / 180.0)

    center_x = default[:, 0]
    center_y = default[:, 1] - radius

    new_x = center_x + radius * torch.sin(angles_rad)
    new_y = center_y + radius * torch.cos(angles_rad)
    new_z = default[:, 2]

    positions = torch.stack([new_x, new_y, new_z], dim=-1) + env.scene.env_origins[env_ids]
    pose = torch.cat([positions, default[:, 3:7]], dim=-1)

    asset.write_root_pose_to_sim(pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=env_ids)


# ---------------------------------------------------------------------------
# EventTermCfg wrappers (called from task __post_init__)
# ---------------------------------------------------------------------------


def randomize_object_in_ellipse(
    name: str,
    x_radius: float,
    y_radius: float,
    yaw_range_deg: tuple[float, float] = (0.0, 0.0),
) -> EventTerm:
    """Reset event that places ``name``'s xy uniformly inside an ellipse.

    Args:
        name: prim name registered via ``parse_usd_and_create_subassets``.
        x_radius: half-width of the ellipse (along world / scene-local +x).
        y_radius: half-depth of the ellipse (along world / scene-local +y).
        yaw_range_deg: extra yaw jitter (deg) applied on top of the authored
            orientation. ``(0, 0)`` keeps the authored yaw exactly.
    """
    return EventTerm(
        func=_randomize_object_in_ellipse_fn,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name),
            "x_radius": float(x_radius),
            "y_radius": float(y_radius),
            "yaw_range_deg": yaw_range_deg,
        },
    )


def _randomize_cubes_scattered_fn(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    cube_cfgs: list[SceneEntityCfg],
    bowl_cfg: SceneEntityCfg,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    yaw_range_deg: tuple[float, float],
    min_cube_sep: float,
    min_bowl_sep: float,
    max_attempts: int,
    num_active: int | None = None,
    min_base_sep: float = 0.0,
    robot_cfg: SceneEntityCfg | None = None,
    z_range: tuple[float, float] = (0.0, 0.0),
    full_orient: bool = False,
    volume_inset: float = 0.0,
) -> None:
    """큐브들을 workspace 내에서 무작위로 배치한다.

    순차 rejection sampling: 큐브를 하나씩 놓으면서 이미 놓인 큐브·그릇과의
    최소 거리 조건이 깨지는 후보를 버린다.  max_attempts 내 조건을 충족 못하면
    해당 env 는 default_root_state 위치로 fallback 한다.

    ``volume_inset`` 으로 x/y 범위를 안쪽으로 줄여 **큐브 볼륨(footprint)이 사각형
    안**에 들어오게 한다(중심이 아니라 부피 기준). 보통 max 큐브의 face 대각 절반
    ((size/2)·√2) 을 준다.

    ``full_orient=True`` 면 yaw 만이 아니라 **full 6D orientation(uniform SO(3))** 을
    샘플하고, ``z_range`` 로 살짝 띄워 떨궈 random face 로 안착하게 한다. (face 가 매
    reset 마다 달라짐 — sim2real 다양성.)

    bowl 의 DR 은 이 함수 실행 전(reset_scene → randomize_bowl 순서)에 완료되지
    않으므로 bowl 기준점으로는 default_root_state 를 사용하고, min_bowl_sep 에
    bowl arc 이동량(≈0.05 m)을 흡수할 여유를 두어야 한다.
    """
    n = len(env_ids)
    if n == 0:
        return

    device = env.device
    # 볼륨이 사각형 안에 들어오도록 중심 샘플 범위를 inset 만큼 축소
    x_lo, x_hi = x_range[0] + volume_inset, x_range[1] - volume_inset
    y_lo, y_hi = y_range[0] + volume_inset, y_range[1] - volume_inset
    yaw_lo = yaw_range_deg[0] * math.pi / 180.0
    yaw_hi = yaw_range_deg[1] * math.pi / 180.0

    bowl_asset: RigidObject = env.scene[bowl_cfg.name]
    bowl_default_xy = bowl_asset.data.default_root_state[env_ids, :2]  # (n, 2) env-local

    # robot base 최소 이격: base 발치(inner-reach)는 안전고도 접근 IK 가 없어
    # 어떤 컨트롤러도 수행 불가 — spawn 자체를 막는다.
    base_xy = None
    if min_base_sep > 0.0 and robot_cfg is not None:
        base_xy = env.scene[robot_cfg.name].data.default_root_state[env_ids, :2]
    min_base_sep_sq = min_base_sep ** 2

    # 누적 배치 xy — 각 큐브를 놓을 때 이전 큐브들과의 거리 확인에 사용
    placed_xy: list[torch.Tensor] = []

    min_bowl_sep_sq = min_bowl_sep ** 2
    min_cube_sep_sq = min_cube_sep ** 2

    n_active = len(cube_cfgs) if num_active is None else max(1, min(len(cube_cfgs), int(num_active)))

    for cube_idx, cube_cfg in enumerate(cube_cfgs):
        asset: RigidObject = env.scene[cube_cfg.name]
        default = asset.data.default_root_state[env_ids].clone()  # (n, 13)

        # 비활성 큐브(stage 에서 안 쓰는 큐브): 지면 아래로 치워 비활성화
        # (작업공간·시야·물리 간섭 제거). 떨어져 사라지며 obs 는 num_active 로 0 마스킹됨.
        if cube_idx >= n_active:
            park = default[:, :3].clone()
            park[:, 2] = -1.0  # 지면(z=0) 아래로 → 낙하해 작업공간 이탈
            park_pos = park + env.scene.env_origins[env_ids]
            pose = torch.cat([park_pos, default[:, 3:7]], dim=-1)
            asset.write_root_pose_to_sim(pose, env_ids=env_ids)
            asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=env_ids)
            continue

        # 최종 위치. 조건 충족 못한 env 는 default 좌표로 유지
        final_x = default[:, 0].clone()
        final_y = default[:, 1].clone()
        placed = torch.zeros(n, dtype=torch.bool, device=device)

        for _ in range(max_attempts):
            unplaced_mask = ~placed
            if not unplaced_mask.any():
                break
            idx = unplaced_mask.nonzero(as_tuple=True)[0]  # (m,) 미배치 env 인덱스

            cand_x = torch.rand(len(idx), device=device) * (x_hi - x_lo) + x_lo
            cand_y = torch.rand(len(idx), device=device) * (y_hi - y_lo) + y_lo

            # 그릇 최소 거리 확인
            bxy = bowl_default_xy[idx]
            ok = (cand_x - bxy[:, 0]).pow(2) + (cand_y - bxy[:, 1]).pow(2) >= min_bowl_sep_sq

            # robot base 최소 거리 확인 (inner-reach spawn 금지)
            if base_xy is not None:
                rxy = base_xy[idx]
                ok = ok & (
                    (cand_x - rxy[:, 0]).pow(2) + (cand_y - rxy[:, 1]).pow(2)
                    >= min_base_sep_sq
                )

            # 이미 배치된 큐브들과의 최소 거리 확인
            for prev in placed_xy:
                pxy = prev[idx]
                ok = ok & (
                    (cand_x - pxy[:, 0]).pow(2) + (cand_y - pxy[:, 1]).pow(2) >= min_cube_sep_sq
                )

            accept = idx[ok]
            final_x[accept] = cand_x[ok]
            final_y[accept] = cand_y[ok]
            placed[accept] = True

        placed_xy.append(torch.stack([final_x, final_y], dim=-1))

        # orientation: full_orient 면 **이산 stable-face + random yaw**.
        #   평평한 매트 위 큐브는 6면 중 하나로만 안착 → 6 (roll,pitch) 중 택1 후 yaw 균등.
        #   처음부터 안착 자세라 tumble drift=0(볼륨 in-rect 보장)·z 띄움 불요·면 다양.
        #   uniform SO(3)+낙하 는 drift 로 사각형 이탈(9% OOB)해 폐기.
        if full_orient:
            faces = torch.tensor(
                [[0.0, 0.0], [math.pi, 0.0], [math.pi / 2, 0.0],
                 [-math.pi / 2, 0.0], [0.0, math.pi / 2], [0.0, -math.pi / 2]],
                device=device,
            )
            pick = torch.randint(0, 6, (n,), device=device)
            rp = faces[pick]
            yaw = torch.rand(n, device=device) * (2.0 * math.pi)
            new_quat = math_utils.quat_from_euler_xyz(rp[:, 0], rp[:, 1], yaw)
        else:
            yaw_delta = torch.rand(n, device=device) * (yaw_hi - yaw_lo) + yaw_lo
            zero = torch.zeros(n, device=device)
            yaw_quat = math_utils.quat_from_euler_xyz(zero, zero, yaw_delta)
            new_quat = math_utils.quat_mul(default[:, 3:7], yaw_quat)

        # z 분산(쌓임 유발): default z 위로 [z_lo, z_hi] 띄워 spawn → 낙하·적재
        z_lo, z_hi = z_range
        final_z = default[:, 2]
        if z_hi > z_lo:
            final_z = final_z + (torch.rand(n, device=device) * (z_hi - z_lo) + z_lo)
        positions = torch.stack([final_x, final_y, final_z], dim=-1) + env.scene.env_origins[env_ids]
        pose = torch.cat([positions, new_quat], dim=-1)
        asset.write_root_pose_to_sim(pose, env_ids=env_ids)
        asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=env_ids)


def randomize_object_on_arc(
    name: str,
    radius: float,
    angle_range_deg: tuple[float, float],
) -> EventTerm:
    """Reset event that places ``name``'s xy on a forward-facing arc.

    Args:
        name: prim name registered via ``parse_usd_and_create_subassets``.
        radius: arc radius (meters). Set to the authored distance from the
            robot base to the default object pose (e.g. PenCup default is
            0.22 m ahead of the robot, so ``radius=0.22``).
        angle_range_deg: ``(min, max)`` arc angle in degrees. ``(0, 0)`` is
            forward (+y in scene-local). Positive rotates toward +x (right),
            negative toward -x (left).
    """
    return EventTerm(
        func=_randomize_object_on_arc_fn,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name),
            "radius": float(radius),
            "angle_range_deg": angle_range_deg,
        },
    )


def randomize_cubes_scattered(
    cube_names: list[str],
    bowl_name: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    yaw_range_deg: tuple[float, float] = (-30.0, 30.0),
    min_cube_sep: float = 0.10,
    min_bowl_sep: float = 0.18,
    max_attempts: int = 50,
    num_active: int | None = None,
    min_base_sep: float = 0.0,
    robot_name: str = "robot",
    z_range: tuple[float, float] = (0.0, 0.0),
    full_orient: bool = False,
    volume_inset: float = 0.0,
) -> EventTerm:
    """큐브 N개를 workspace 내에서 완전 무작위로 배치하는 reset event.

    num_active 지정 시 앞 num_active 개만 workspace 에 배치하고, 나머지는 지면 아래로
    치워 비활성화한다(커리큘럼의 active_objects 와 연동).

    Args:
        cube_names: 배치할 큐브 prim 이름 목록. 순서대로 처리한다.
        bowl_name: 그릇 prim 이름 (거리 제약 기준).
        x_range: workspace x 범위 (world/env-local).
        y_range: workspace y 범위 (world/env-local).
        yaw_range_deg: 각 큐브에 적용할 yaw jitter 범위 (도).
        min_cube_sep: 큐브 간 최소 중심 거리 (m).
        min_bowl_sep: 큐브-그릇 간 최소 중심 거리 (m). bowl DR 이동량(≈0.05 m)을
            흡수할 여유를 포함해야 한다.
        max_attempts: env 당 rejection sampling 최대 시도 횟수. 초과 시 해당
            env 의 큐브는 default_root_state 위치로 fallback.
        min_base_sep: 큐브-robot base 간 최소 중심 거리 (m). 0 이면 비활성.
            base 발치(inner-reach)는 안전고도 접근 IK 가 없어 수행 불가 —
            SO-101 cube_desk 실측 한계 r≈0.13 에 여유를 더해 0.135 권장.
        robot_name: base 이격 기준 articulation 이름.
    """
    return EventTerm(
        func=_randomize_cubes_scattered_fn,
        mode="reset",
        params={
            "cube_cfgs": [SceneEntityCfg(n) for n in cube_names],
            "bowl_cfg": SceneEntityCfg(bowl_name),
            "x_range": x_range,
            "y_range": y_range,
            "yaw_range_deg": yaw_range_deg,
            "min_cube_sep": float(min_cube_sep),
            "min_bowl_sep": float(min_bowl_sep),
            "max_attempts": int(max_attempts),
            "num_active": num_active,
            "min_base_sep": float(min_base_sep),
            "robot_cfg": SceneEntityCfg(robot_name) if min_base_sep > 0.0 else None,
            "z_range": z_range,
            "full_orient": bool(full_orient),
            "volume_inset": float(volume_inset),
        },
    )


# ---------------------------------------------------------------------------
# 물리 DR — Isaac Lab stock term(ManagerTermBase) 래퍼.
#   sim2real 일반화를 위해 큐브의 마찰/질량을 env 별로 무작위화한다.
#   mode="startup": env 초기화 시 1회 샘플 → env 간 물리 다양성(표준 패턴).
#   grasp weld/유지력 추가가 아니므로 reward hacking 이 아니다.
# ---------------------------------------------------------------------------


def randomize_object_material(
    name: str,
    *,
    static_friction_range: tuple[float, float] = (1.4, 2.0),
    dynamic_friction_range: tuple[float, float] = (1.2, 1.7),
    restitution_range: tuple[float, float] = (0.0, 0.0),
    num_buckets: int = 64,
) -> EventTerm:
    """``name`` rigid body 의 마찰/반발 계수를 무작위 material bucket 으로 할당.

    기본 범위는 큐브 authored 값(static 1.8 / dynamic 1.5) 중심의 ±대역.
    make_consistent=True 로 dynamic ≤ static 을 강제한다.
    """
    return EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(name),
            "static_friction_range": static_friction_range,
            "dynamic_friction_range": dynamic_friction_range,
            "restitution_range": restitution_range,
            "num_buckets": int(num_buckets),
            "make_consistent": True,
        },
    )


def randomize_object_mass(
    name: str,
    *,
    mass_range: tuple[float, float] = (0.9, 1.1),
    operation: str = "scale",
) -> EventTerm:
    """``name`` rigid body 의 질량을 무작위화 (기본 ±10% scale).

    grasp 안정 한계(actuator 10Nm) 내로 유지하기 위해 좁은 범위만 쓴다.
    """
    return EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(name),
            "mass_distribution_params": mass_range,
            "operation": operation,
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
