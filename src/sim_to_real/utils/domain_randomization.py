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
from isaaclab.managers import ManagerTermBase
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


def _bell_halfwidth(y: torch.Tensor, ys: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
    """종 모양 x 반너비 |x|<=w(y) 의 piecewise-linear 보간.

    y<=ys[0] → ws[0], y>=ys[-1] → ws[-1], 그 사이는 선형. ys 는 오름차순.
    """
    w = torch.full_like(y, float(ws[-1]))
    w = torch.where(y <= ys[0], ws[0], w)
    for i in range(len(ys) - 1):
        m = (y > ys[i]) & (y <= ys[i + 1])
        frac = (y - ys[i]) / (ys[i + 1] - ys[i])
        w = torch.where(m, ws[i] + frac * (ws[i + 1] - ws[i]), w)
    return w


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
    cube_sizes: list[float] | None = None,
    cube_sep_margin: float = 0.005,
    x_halfwidth_by_y: list[tuple[float, float]] | None = None,
    x_exclude_box: tuple[float, float, float, float] | None = None,
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

    # 종 모양(bell) x 반너비 프로파일 — |x|<=w(y). grasp 가능범위(좌우대칭)를 사각형 대신 종으로.
    bell_ys = bell_ws = None
    if x_halfwidth_by_y:
        bp = sorted(x_halfwidth_by_y)
        bell_ys = torch.tensor([p[0] for p in bp], device=device, dtype=torch.float32)
        bell_ws = torch.tensor([p[1] for p in bp], device=device, dtype=torch.float32)

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

    # 큐브 크기 대응: footprint 반경 r=s·√2/2 (임의 yaw 코너 최악)로 per-pair·per-bowl 최소
    # 이격을 동적 계산. cube_sizes 미지정(legacy)=스칼라 min_cube_sep/min_bowl_sep 그대로.
    #   큐브쌍: r_i + r_j + margin (40mm쌍 ≈0.057+margin → 기존 0.060 재현).
    #   그릇  : min_bowl_sep(40mm 정합값) + (r_i − r_40). 50mm 면 +0.007.
    radii = None
    if cube_sizes is not None:
        radii = [float(s) * (2.0 ** 0.5) / 2.0 for s in cube_sizes]
        R_REF40 = 0.040 * (2.0 ** 0.5) / 2.0

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

        # 이 큐브의 그릇 이격 (크기 대응: 큰 큐브일수록 더 멀리)
        cur_bowl_sep_sq = min_bowl_sep_sq
        if radii is not None:
            cur_bowl_sep_sq = (min_bowl_sep + (radii[cube_idx] - R_REF40)) ** 2

        for _ in range(max_attempts):
            unplaced_mask = ~placed
            if not unplaced_mask.any():
                break
            idx = unplaced_mask.nonzero(as_tuple=True)[0]  # (m,) 미배치 env 인덱스

            cand_x = torch.rand(len(idx), device=device) * (x_hi - x_lo) + x_lo
            cand_y = torch.rand(len(idx), device=device) * (y_hi - y_lo) + y_lo

            # 종 모양 x 반너비 제한: |x| <= w(y) (grasp 가능범위, 좌우대칭 종모양)
            if bell_ys is not None:
                ok0 = cand_x.abs() <= _bell_halfwidth(cand_y, bell_ys, bell_ws)
            else:
                ok0 = torch.ones(len(idx), dtype=torch.bool, device=device)

            # 로봇암 주변 제외 박스(사각형) — 이 박스 안 후보 배제.
            if x_exclude_box is not None:
                ex0, ex1, ey0, ey1 = x_exclude_box
                in_box = (cand_x >= ex0) & (cand_x <= ex1) & (cand_y >= ey0) & (cand_y <= ey1)
                ok0 = ok0 & ~in_box

            # 그릇 최소 거리 확인 (큐브 크기 대응 이격)
            bxy = bowl_default_xy[idx]
            ok = ok0 & ((cand_x - bxy[:, 0]).pow(2) + (cand_y - bxy[:, 1]).pow(2) >= cur_bowl_sep_sq)

            # robot base 최소 거리 확인 (inner-reach spawn 금지)
            if base_xy is not None:
                rxy = base_xy[idx]
                ok = ok & (
                    (cand_x - rxy[:, 0]).pow(2) + (cand_y - rxy[:, 1]).pow(2)
                    >= min_base_sep_sq
                )

            # 이미 배치된 큐브들과의 최소 거리 확인 (per-pair 크기 대응 이격)
            for prev_idx, prev in enumerate(placed_xy):
                pair_sep_sq = min_cube_sep_sq
                if radii is not None:
                    pair_sep_sq = (radii[cube_idx] + radii[prev_idx] + cube_sep_margin) ** 2
                pxy = prev[idx]
                ok = ok & (
                    (cand_x - pxy[:, 0]).pow(2) + (cand_y - pxy[:, 1]).pow(2) >= pair_sep_sq
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
    cube_sizes: list[float] | None = None,
    cube_sep_margin: float = 0.005,
    x_halfwidth_by_y: list[tuple[float, float]] | None = None,
    x_exclude_box: tuple[float, float, float, float] | None = None,
) -> EventTerm:
    """큐브 N개를 workspace 내에서 완전 무작위로 배치하는 reset event.

    ``x_halfwidth_by_y`` 지정 시 x_range 사각형 대신 **좌우대칭 종 모양**(|x|<=w(y),
    (y,halfwidth) breakpoint 선형보간)으로 스폰 영역을 제한한다 — grasp 가능범위 정합.
    ``x_exclude_box=(x0,x1,y0,y1)`` 지정 시 그 사각형(로봇암 주변) 안은 배제한다.

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
            "cube_sizes": ([float(s) for s in cube_sizes] if cube_sizes is not None else None),
            "cube_sep_margin": float(cube_sep_margin),
            "x_halfwidth_by_y": (
                [(float(a), float(b)) for a, b in x_halfwidth_by_y]
                if x_halfwidth_by_y is not None else None
            ),
            "x_exclude_box": (
                tuple(float(v) for v in x_exclude_box) if x_exclude_box is not None else None
            ),
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


# ---------------------------------------------------------------------------
# 시각 도메인 랜덤화 (sim2real) — NVIDIA Workshop resets.py 참고.
# cuRobo oracle 은 큐브 world pose 만 쓰므로 시각 변화는 grasp 기하에 무영향(C-안전).
# 라이트는 글로벌 /World/Light·/World/KeyLight(per-env 아님) → reset 당 전 env 공통값.
# 카메라 focal 은 per-env prim. 모든 USD set 은 IsValid 가드(속성 없으면 no-op, 무크래시).
# ---------------------------------------------------------------------------


def _randomize_lights_fn(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    dome_prim_path: str,
    key_prim_path: str,
    dome_intensity_range: tuple[float, float],
    key_intensity_range: tuple[float, float],
    warmth_range: tuple[float, float],
) -> None:
    """Dome/Key 라이트 intensity + 색온도(warmth) 무작위화. reset 당 1 샘플(전 env 공통)."""
    from isaaclab.sim import get_current_stage
    from pxr import Gf, Sdf

    if env_ids is None or len(env_ids) == 0:
        return
    stage = get_current_stage()

    def _u(lo: float, hi: float) -> float:
        return float(torch.empty(1).uniform_(float(lo), float(hi)).item())

    dome_i = _u(*dome_intensity_range)
    key_i = _u(*key_intensity_range)
    t = _u(*warmth_range)  # 0=cool(푸른) ~ 1=warm(주황)
    cool = (0.82, 0.90, 1.0)
    warm = (1.0, 0.88, 0.72)
    rgb = tuple(cool[i] * (1.0 - t) + warm[i] * t for i in range(3))

    with Sdf.ChangeBlock():
        for path, inten in ((dome_prim_path, dome_i), (key_prim_path, key_i)):
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            ia = prim.GetAttribute("inputs:intensity")
            if ia.IsValid():
                ia.Set(float(inten))
            ca = prim.GetAttribute("inputs:color")
            if ca.IsValid():
                ca.Set(Gf.Vec3f(float(rgb[0]), float(rgb[1]), float(rgb[2])))


def randomize_lights(
    dome_prim_path: str = "/World/Light",
    key_prim_path: str = "/World/KeyLight",
    *,
    dome_intensity_range: tuple[float, float] = (1400.0, 2700.0),
    key_intensity_range: tuple[float, float] = (1100.0, 2400.0),
    warmth_range: tuple[float, float] = (0.0, 1.0),
) -> EventTerm:
    """라이트(밝기·색온도) 무작위화 EventTerm(reset). sim2real 시각 다양성."""
    return EventTerm(
        func=_randomize_lights_fn,
        mode="reset",
        params={
            "dome_prim_path": dome_prim_path,
            "key_prim_path": key_prim_path,
            "dome_intensity_range": dome_intensity_range,
            "key_intensity_range": key_intensity_range,
            "warmth_range": warmth_range,
        },
    )


def _randomize_camera_focal_fn(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    camera_prim_globs: list[str],
    focal_range: tuple[float, float],
) -> None:
    """카메라 focalLength 무작위화(FOV 변화). 카메라 없으면 no-op.

    ★per-env 독립 샘플(robot color 와 동일 수정): prim(=env clone)마다 새로 뽑고
    env_ids 에 든 env 만 갱신 — 옛 구현은 glob 당 1 샘플이라 전 env 동일 FOV."""
    import re

    import isaaclab.sim as sim_utils
    from pxr import Sdf

    if env_ids is None or len(env_ids) == 0:
        return
    reset_envs = {int(i) for i in env_ids.tolist()}
    with Sdf.ChangeBlock():
        for glob in camera_prim_globs:
            for prim in sim_utils.find_matching_prims(glob):
                if not prim.IsValid():
                    continue
                m = re.search(r"env_(\d+)", str(prim.GetPath()))
                if m is not None and int(m.group(1)) not in reset_envs:
                    continue
                fa = prim.GetAttribute("focalLength")
                if fa.IsValid():
                    fa.Set(float(torch.empty(1).uniform_(
                        float(focal_range[0]), float(focal_range[1])).item()))


def randomize_camera_focal(
    *,
    focal_range: tuple[float, float] = (14.0, 22.0),
    camera_prim_globs: list[str] | None = None,
) -> EventTerm:
    """top/wrist/front 카메라 focalLength 무작위화 EventTerm(reset). 카메라 리그 없으면 no-op.

    기본 focal_range 는 보정된 nominal 18mm(`pick_cube_env_cfg._*_CAMERA_FOCAL`)를 18±2 로
    straddle 한다 → 학습 데이터 focal 분포가 배포(eval) focal 18 을 중심에 포함(sim2real 정합).
    """
    if camera_prim_globs is None:
        camera_prim_globs = [
            "/World/envs/env_.*/TopCamera",
            "/World/envs/env_.*/Robot/gripper/WristCamera",
            "/World/envs/env_.*/Robot/shoulder/FrontCamera",
        ]
    return EventTerm(
        func=_randomize_camera_focal_fn,
        mode="reset",
        params={"camera_prim_globs": camera_prim_globs, "focal_range": focal_range},
    )


# ---------------------------------------------------------------------------
# 로봇 외형 색 DR (sim2real) — env 당 팔레트 색 1개로 로봇 전체 재색칠.
#
# ★2026-07-20 재작성: 옛 구현(USD shader diffuse 직접 Set)은 Isaac Fabric(flatcache) 렌더에
# 반영 안 돼 multi-env 서 전 로봇이 author 색(보라)으로 렌더됐다(실측). 스톡 randomize_visual_color
# 처럼 Replicator(create_batch.material + modify.attribute)로 OmniPBR 을 새로 만들어야 Fabric 에
# 반영된다. body 는 링크에 강하게 바인딩돼 mesh 재바인딩으론 안 바뀌어 → robot **root** 에
# strongerThanDescendants 로 바인딩해 전 서브트리를 한 색으로 override. servo/metal 개별 고정은
# 이 방식에선 불가(로봇 통짜 한 색). ★scene.replicate_physics=False 필수(-DR scene 이 끔).
# 상세 = _RandomizeRobotColor docstring. 팔레트 = 실 SO-101 offerings 색 + 기본 보라.
# ---------------------------------------------------------------------------


# 실 SO-101 색 팔레트 (Workshop ROBOT_COLORS + 우리 기본 보라). root 바인딩이라 로봇 전체를
# 이 중 한 색으로 칠한다(servo/metal 개별 고정은 없음 — Fabric 반영엔 root override 가 필요).
ROBOT_PLASTIC_COLORS: dict[str, tuple[float, float, float]] = {
    "purple": (0.40, 0.03, 0.75),   # 우리 기본 (patch_robot_colors PLASTIC_PURPLE)
    "orange": (0.876, 0.317, 0.132),
    "teal": (0.0, 0.8, 0.502),
    "white": (0.95, 0.95, 0.95),
    "black": (0.08, 0.08, 0.08),
}


class _RandomizeRobotColor(ManagerTermBase):
    """로봇 plastic 바디를 **env 당 팔레트 색 1개**로 재색칠 (Replicator = Fabric-aware).

    ★왜 USD-edit 이 아니라 Replicator 인가 (2026-07-20 실측): Isaac Sim 은 Fabric(flatcache)
    로 렌더하므로 런타임 USD material 편집(shader diffuse 직접 Set·새 material 바인딩 모두)이
    **렌더에 반영되지 않는다**. 그 결과 multi-env 서 전 로봇이 author 색(보라)으로 렌더됐다.
    스톡 ``randomize_visual_color`` 과 동일하게 replicator ``create_batch.material`` +
    ``modify.attribute`` 로 OmniPBR 을 새로 바인딩·수정해야 Fabric 에 반영된다. 스톡은
    per-mesh 랜덤(무지개)이지만 여기선 **env 당 색 1개**를 그 env 의 전 plastic mesh 에 균일 적용.

    ★``scene.replicate_physics=False`` 필수: replication(Fabric) 이 켜지면 전 env 가 env_0 의
    material 로 렌더돼 per-env 색이 무시된다(전 로봇 동색). -DR scene 이 이를 False 로 둔다.

    servo/metal(``material_sts3215`` 등)은 재색칠 대상 material 을 바인딩한 mesh 만 고르는
    방식으로 자동 제외(색 고정).

    ★mode=``prestartup`` 인 이유(reset 불가, 2026-07-20 실측): __init__ 이 robot subtree 를
    de-instance(구조 변경)해 physx tensor view 를 무효화한다. prestartup 은 ``scene.update``
    **전에** apply 되며 그 Replicator op 이 view 를 리프레시 → 정상. reset 모드면 __call__ 이
    scene.update 후로 밀려 view 무효 상태로 ``get_dof_velocities`` 크래시("Simulation view
    invalidated"). 즉 **런타임 Replicator 재색칠은 리셋마다 못 한다** → env 당 색은 런 내내 고정.
    N/R·에피소드 리셋으로 재추첨하려면 Fabric 직접 write(구조 변경 없는) 경로 필요(후속 과제).
    다양성: env 수만큼 서로 다른 팔레트 색(≤팔레트 크기) + 새 ``--seed`` 로 재추첨.
    """

    def __init__(self, cfg: EventTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        import re

        from isaacsim.core.utils.extensions import enable_extension
        import isaaclab.sim as sim_utils
        from pxr import Usd, UsdShade

        if env.cfg.scene.replicate_physics:
            raise RuntimeError(
                "randomize_robot_color 는 scene.replicate_physics=False 필요 — Fabric replication "
                "이 켜지면 per-env robot material 이 무시돼 전 로봇이 같은 색으로 렌더된다.")

        enable_extension("omni.replicator.core")
        import omni.replicator.core as rep

        glob = cfg.params.get("robot_prim_glob", "/World/envs/env_.*/Robot")
        names = cfg.params.get("color_names") or list(ROBOT_PLASTIC_COLORS.keys())
        self._palette = [ROBOT_PLASTIC_COLORS[n] for n in names]
        self._rep = rep

        # per-env robot **root** 에 OmniPBR 1개를 strongerThanDescendants 로 바인딩 → 로봇 전
        # 서브트리(body 포함) 를 한 색으로 override. mesh 단위 재바인딩은 body 처럼 링크에 강하게
        # 바인딩된 원본 material 을 못 이겨 body 가 안 바뀌었다(실측: 관절만 바뀜). root override 가
        # 전 하위 binding 을 이긴다. OmniPBR 생성은 스톡 randomize_visual_color 와 동일 = Fabric 반영.
        # env 인덱스 자연 정렬(lexicographic 는 env_10 < env_2 라 ≥10 env 서 self._mats[i]↔env i 깨짐).
        def _env_idx(prim):
            m = re.search(r"env_(\d+)", str(prim.GetPath()))
            return int(m.group(1)) if m else 0
        roots = sorted(sim_utils.find_matching_prims(glob), key=_env_idx)
        for root in roots:
            for prim in Usd.PrimRange(root):
                if prim.IsInstanceable():
                    prim.SetInstanceable(False)
        self._mats = (rep.functional.create_batch.material(
            mdl="OmniPBR.mdl", bind_prims=roots, count=len(roots), project_uvw=True)
            if roots else None)
        stage = env.sim.stage
        for root, mat in zip(roots, self._mats or []):
            mat_prim = mat if isinstance(mat, Usd.Prim) else stage.GetPrimAtPath(str(mat))
            if mat_prim and mat_prim.IsValid():
                UsdShade.MaterialBindingAPI(root).Bind(
                    UsdShade.Material(mat_prim),
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants)

    def __call__(self, env: ManagerBasedRLEnv, env_ids, robot_prim_glob=None, color_names=None):
        import numpy as np

        if self._mats is None:
            return
        # reset 이벤트: **리셋된 env 만** 재색칠(에피소드 시작마다 새 색, 에피소드 내 고정).
        # self._mats 는 env 인덱스 순(자연 정렬)이라 self._mats[i] = env i 의 material.
        # 다른 env 를 건드리면 진행 중 에피소드의 로봇색이 바뀌므로 env_ids 로만 국한한다.
        if env_ids is None:                                       # 전 env 대상 호출(방어)
            ids = list(range(len(self._mats)))
        elif torch.is_tensor(env_ids):
            ids = env_ids.detach().cpu().tolist()
        else:
            ids = list(env_ids)
        ids = [i for i in ids if 0 <= i < len(self._mats)]
        if not ids:
            return
        mats = [self._mats[i] for i in ids]
        pick = torch.randint(0, len(self._palette), (len(mats),))
        cols = np.array([self._palette[int(pick[k])] for k in range(len(mats))], dtype=float)
        self._rep.functional.modify.attribute(mats, "diffuse_color_constant", cols)


def randomize_robot_color(
    robot_prim_glob: str = "/World/envs/env_.*/Robot",
    *,
    color_names: list[str] | None = None,
) -> EventTerm:
    """로봇 plastic 바디 색 per-env 무작위화 EventTerm. sim2real 외형 다양성.

    ★``scene.replicate_physics=False`` 필요(Fabric replication 이 per-env 색을 무시). -DR
    scene 이 이를 끈다. Replicator 로 OmniPBR 재바인딩 → Fabric 반영(USD-edit 은 무반영).
    mode=``prestartup`` → env 당 색 1개를 런 내내 고정(각 로봇 고유색). reset 모드는 physx
    view 무효화로 크래시(위 클래스 docstring 참조) → 리셋 재추첨 불가, 새 ``--seed`` 로 재추첨.

    Args:
        robot_prim_glob: per-env Robot 루트 prim glob (하위 mesh 트리를 순회).
        color_names: 사용할 ``ROBOT_PLASTIC_COLORS`` 키 목록. None 이면 전체 팔레트.
    """
    return EventTerm(
        func=_RandomizeRobotColor,
        mode="prestartup",
        params={"robot_prim_glob": robot_prim_glob, "color_names": color_names},
    )
