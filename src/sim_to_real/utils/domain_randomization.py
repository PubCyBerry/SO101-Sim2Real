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
from isaaclab.utils import configclass

from .cube_specs import mass_for_size


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


def _prim_env_index(prim) -> int:
    """``/World/envs/env_<i>/...`` prim → env 인덱스.

    prim 목록 정렬 키. lexicographic 정렬은 ``env_10 < env_2`` 라 10 env 이상에서
    prim[i] ↔ env i 대응이 깨진다(실측 버그) — 항상 이 키로 자연 정렬한다.
    """
    import re

    m = re.search(r"env_(\d+)", str(prim.GetPath()))
    return int(m.group(1)) if m else 0


# env 에 붙는 per-env 큐브 한 변(m) 기록: ``env.cube_size_m[name] -> (num_envs,) tensor``.
# 크기 DR 이 뽑은 실제 값을 소비자(스폰 z 보정 · cuRobo SM 의 grasp 조준/planner 요청)가
# 되읽는 유일한 통로다. 크기 DR 이 없는 env 에는 attribute 자체가 없다(소비자는 폴백).
CUBE_SIZE_ATTR = "cube_size_m"


def _randomize_cube_scale_fn(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
    sizes: list[float],
    base_sizes: list[float],
) -> None:
    """큐브 USD scale + 질량을 env 별로 **이산** 무작위화 (mode=``prestartup`` 전용).

    ``mdp.randomize_rigid_body_scale`` 대신 자체 구현인 이유:
      1. 스톡은 연속 range 만 받는다 — 우리는 5 mm 격자(25/30/35/40)의 이산 사다리다.
      2. 스톡은 ``xformOp:scale`` 이 없을 때 ``xformOpOrder`` 를 통째로
         ``[translate, orient, scale]`` 로 덮어쓴다. 씬의 큐브 order 는
         ``[translate, rotateZ]`` 라 authored rotateZ 가 order 에서 사라진다.
         여기서는 ``AddScaleOp()`` 로 **append** 만 한다.
      3. 스케일과 함께 질량을 ``mass_for_size`` 로 같이 옮겨야 grasp 물리가 일관된다
         (USD scale 은 mass attr 을 건드리지 않아 25 mm 큐브가 40 mm 무게로 남는다).

    ⚠ ``prestartup`` = 물리 파싱 **전** USD 편집. EventManager 가 이 모드에서
    ``scene.replicate_physics=True`` 를 금지한다(복제되면 env 간 속성이 공유됨).
    """
    import isaaclab.sim as sim_utils
    from isaaclab.sim import get_current_stage
    from pxr import Gf, UsdGeom, UsdPhysics

    if env.sim.is_playing():
        raise RuntimeError("큐브 크기 DR 은 시뮬 시작 전에만 가능(mode='prestartup')")
    get_current_stage()  # stage 준비 확인(없으면 여기서 실패해야 진단이 쉽다)
    num_envs = env.scene.num_envs
    choices = torch.tensor([float(s) for s in sizes], dtype=torch.float32)
    record: dict[str, torch.Tensor] = dict(getattr(env, CUBE_SIZE_ATTR, {}) or {})

    for asset_cfg, base in zip(asset_cfgs, base_sizes):
        prims = sorted(sim_utils.find_matching_prims(env.scene[asset_cfg.name].cfg.prim_path),
                       key=_prim_env_index)
        # prim ↔ env 대응이 어긋나면 크기와 조준 z 가 조용히 다른 env 를 가리킨다 → fail-fast.
        if len(prims) != num_envs:
            raise RuntimeError(
                f"{asset_cfg.name}: prim {len(prims)}개 ≠ env {num_envs}개 "
                f"({env.scene[asset_cfg.name].cfg.prim_path}) — 크기 DR 을 적용할 수 없다")
        picked = choices[torch.randint(0, len(choices), (num_envs,))]
        for i, prim in enumerate(prims):
            size = float(picked[i])
            factor = size / float(base)
            xform = UsdGeom.Xformable(prim)
            op = next((o for o in xform.GetOrderedXformOps()
                       if o.GetOpType() == UsdGeom.XformOp.TypeScale), None)
            if op is None:
                op = xform.AddScaleOp()  # order 뒤에 append — authored translate/rotateZ 보존
            op.Set(Gf.Vec3f(factor, factor, factor))
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(mass_for_size(size))
        record[asset_cfg.name] = picked.to(env.device)
        uniq = sorted({round(float(v), 4) for v in picked.tolist()})
        print(f"[cube-scale-dr] {asset_cfg.name}: base={base:.3f} sizes={uniq} "
              f"(n_env={num_envs})", flush=True)
    setattr(env, CUBE_SIZE_ATTR, record)


def randomize_cube_scale(
    cube_names: list[str],
    sizes: list[float],
    base_sizes: list[float],
) -> EventTerm:
    """큐브 크기(+질량) 이산 무작위화 EventTerm(``prestartup``).

    Args:
        cube_names: 스케일할 큐브 prim 이름.
        sizes: 뽑을 한 변 후보(m). 등확률.
        base_sizes: 각 큐브의 **authored** 한 변(m) — USD scale = size/base 라 필요하다.

    ⚠ env 당 크기는 런 내내 고정이다(USD 편집은 물리 파싱 전에만 가능 — 리셋마다
    재추첨 불가). 다양성은 **env 수 + 새 seed**로 얻는다. 크기 DR 을 쓰는 env 는
    ``scene.replicate_physics=False`` 여야 한다.
    """
    return EventTerm(
        func=_randomize_cube_scale_fn,
        mode="prestartup",
        params={
            "asset_cfgs": [SceneEntityCfg(n) for n in cube_names],
            "sizes": [float(s) for s in sizes],
            "base_sizes": [float(s) for s in base_sizes],
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
    base_sep_offset_xy: tuple[float, float] = (0.0, 0.0),
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

    bowl 기준점 = **실제(post-DR) 그릇 xy**. ``randomize_bowl`` 을 이 term 앞에 배치
    (EventCfg 선언순서 = EventManager 적용순서)하고, ``write_root_pose_to_sim`` 이
    ``root_pose_w`` 를 물리스텝 없이 즉시 갱신하므로 실제 그릇좌표를 읽어 rejection 한다.
    (옛 방식=default_root_state[nominal] → 그릇 arc 이동으로 ``min_bowl_sep`` 불변식이
    사후 파괴됨: cube-bowl 0.126<0.14 스폰 → transit 계획 실패. randomize_bowl 없는
    config 면 root_pos_w == nominal 이라 무해.)
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
    # 실제(post-DR) 그릇 xy(env-local) — randomize_bowl 이 앞서 적용됨(EventCfg 선언순서).
    # world(root_pos_w) − env_origin = env-local. randomize_bowl 없는 config 면 nominal 과 동일(무해).
    bowl_xy = bowl_asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]  # (n, 2)

    # robot base 최소 이격: base 발치(inner-reach)는 안전고도 접근 IK 가 없어
    # 어떤 컨트롤러도 수행 불가 — spawn 자체를 막는다.
    base_xy = None
    if min_base_sep > 0.0 and robot_cfg is not None:
        # ★min-reach 중심 = 로봇 root(마운트 원점) + pan축 offset(팔 실제 회전중심).
        # offset 은 spawn_area.PAN_AXIS_XY (단일소스, pick_cube_env_cfg 가 전달) — 마운트원점
        # 기준이면 -x 근접 corner 를 도달불가인데 통과시킴(sweep 판정 in_spawn_area 와 동일 중심).
        root_xy = env.scene[robot_cfg.name].data.default_root_state[env_ids, :2]
        base_xy = root_xy + torch.tensor(base_sep_offset_xy, device=device, dtype=root_xy.dtype)
    min_base_sep_sq = min_base_sep ** 2

    # 누적 배치 xy / footprint 반경 — 각 큐브를 놓을 때 이전 큐브들과의 거리 확인에 사용
    placed_xy: list[torch.Tensor] = []
    placed_r: list[torch.Tensor | None] = []

    min_bowl_sep_sq = min_bowl_sep ** 2
    min_cube_sep_sq = min_cube_sep ** 2

    # 큐브 크기 대응: footprint 반경 r=s·√2/2 (임의 yaw 코너 최악)로 per-pair·per-bowl 최소
    # 이격을 동적 계산. cube_sizes 미지정(legacy)=스칼라 min_cube_sep/min_bowl_sep 그대로.
    #   큐브쌍: r_i + r_j + margin (40mm쌍 ≈0.057+margin → 기존 0.060 재현).
    #   그릇  : min_bowl_sep(40mm 정합값) + (r_i − r_40). 50mm 면 +0.007.
    # ★크기 DR 이 켜지면 반경은 **env 마다 다르다**(`env.cube_size_m`) → nominal 대신 실제
    #   크기를 쓴다. authored 가 사다리 하한(25 mm)이 된 뒤로는 이게 필수다: nominal 기준이면
    #   40 mm 가 뽑힌 env 가 25 mm 이격으로 그릇에 10.6 mm 더 붙어 스폰된다(transit 계획 실패).
    radii = None
    if cube_sizes is not None:
        radii = [float(s) * (2.0 ** 0.5) / 2.0 for s in cube_sizes]
    R_REF40 = 0.040 * (2.0 ** 0.5) / 2.0
    dr_size_map = getattr(env, CUBE_SIZE_ATTR, None) or {}

    def _footprint_radius(cube_idx: int, name: str) -> torch.Tensor | None:
        """이 큐브의 per-env footprint 반경 (n,). 크기 정보가 전혀 없으면 None."""
        sizes = dr_size_map.get(name)
        if sizes is not None:
            return sizes[env_ids] * (2.0 ** 0.5) / 2.0
        if radii is not None:
            return torch.full((n,), radii[cube_idx], device=device, dtype=torch.float32)
        return None

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

        # 이 큐브의 그릇 이격 (크기 대응: 큰 큐브일수록 더 멀리). per-env 텐서 (n,).
        r_this = _footprint_radius(cube_idx, cube_cfg.name)
        if r_this is None:
            cur_bowl_sep_sq = torch.full((n,), min_bowl_sep_sq, device=device, dtype=torch.float32)
        else:
            cur_bowl_sep_sq = (min_bowl_sep + (r_this - R_REF40)).clamp_min(0.0) ** 2

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
            bxy = bowl_xy[idx]
            ok = ok0 & ((cand_x - bxy[:, 0]).pow(2) + (cand_y - bxy[:, 1]).pow(2)
                        >= cur_bowl_sep_sq[idx])

            # robot base 최소 거리 확인 (inner-reach spawn 금지)
            if base_xy is not None:
                rxy = base_xy[idx]
                ok = ok & (
                    (cand_x - rxy[:, 0]).pow(2) + (cand_y - rxy[:, 1]).pow(2)
                    >= min_base_sep_sq
                )

            # 이미 배치된 큐브들과의 최소 거리 확인 (per-pair·per-env 크기 대응 이격)
            for prev_idx, prev in enumerate(placed_xy):
                r_prev = placed_r[prev_idx]
                if r_this is None or r_prev is None:
                    pair_sep_sq = min_cube_sep_sq
                else:
                    pair_sep_sq = ((r_this + r_prev + cube_sep_margin) ** 2)[idx]
                pxy = prev[idx]
                ok = ok & (
                    (cand_x - pxy[:, 0]).pow(2) + (cand_y - pxy[:, 1]).pow(2) >= pair_sep_sq
                )

            accept = idx[ok]
            final_x[accept] = cand_x[ok]
            final_y[accept] = cand_y[ok]
            placed[accept] = True

        placed_xy.append(torch.stack([final_x, final_y], dim=-1))
        placed_r.append(r_this)

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
        # 크기 DR 보정: authored z 는 nominal 반높이 기준이라, 스케일된 env 는 그 차이만큼
        # 내려앉혀야 한다. 안 하면 작은 큐브가 (nominal−실제)/2 만큼 공중에서 떨어져
        # 튀면서 DR 이 정한 xy 를 벗어난다.
        dr_sizes = dr_size_map.get(cube_cfg.name)
        if dr_sizes is not None and cube_sizes is not None:
            final_z = final_z + 0.5 * (dr_sizes[env_ids] - float(cube_sizes[cube_idx]))
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
    base_sep_offset_xy: tuple[float, float] = (0.0, 0.0),
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
            "base_sep_offset_xy": tuple(float(v) for v in base_sep_offset_xy),
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
        roots = sorted(sim_utils.find_matching_prims(glob), key=_prim_env_index)
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


# ---------------------------------------------------------------------------
# 카메라 extrinsic DR (6-DoF pose) — episode bias(리셋) + frame-wise jitter(30 Hz)
#
# ■ 좌표계 계약 (transform 합성부와 함께 읽을 것)
#   Camera key        : top | wrist | front            (obs/dataset key)
#   Scene asset       : f"{key}_camera"                (TiledCamera)
#   Parent prim       : top   = /World/envs/env_i                  (정적 xform)
#                       wrist = /World/envs/env_i/Robot/gripper    (articulation link)
#                       front = /World/envs/env_i/Robot/shoulder   (articulation link)
#                       → ★셋 다 "부모 prim 기준 local xform" 갱신 하나로 동일 처리.
#                         top 만 부모가 정적일 뿐, front 도 wrist 처럼 링크에 붙어 있다.
#   Delta frame       : camera_local — nominal 카메라 축 기준
#   Camera convention : prim 에 저장된 local quat = **OpenGL/USD 카메라**(fwd −Z, up +Y, right +X).
#                       cfg 의 ``OffsetCfg.convention="world"`` 는 author 시점 표기일 뿐,
#                       런타임 prim 값과 다르다 → nominal 을 prim 에서 읽어 변환 왕복을 없앤다.
#                       그래서 delta 축 이름을 roll/pitch/yaw 가 아니라 **x/y/z** 로 쓴다:
#                         x = right (tilt), y = up (pan), z = backward(= −optical, roll)
#   Quaternion order  : wxyz (IsaacLab 전역 · XFormPrim.get/set_local_poses 동일)
#   Euler order       : math_utils.quat_from_euler_xyz → XYZ 규약, R = Rz(z)·Ry(y)·Rx(x)
#   Composition order : pos  = pos_nom + R(quat_nom) · Δt        (nominal 축으로 회전 후 더함)
#                       quat = quat_nom ⊗ Δq                     (nominal **오른쪽**에 delta)
#   Nominal source    : 최초 1회 ``cam._view.get_local_poses()`` — teleop ``--tune_cameras``
#                       오버라이드도 자동 반영. 이후 immutable(누적 금지의 기준점).
#   Update frequency  : env.step() 당 1회 = 30 Hz = 카메라 프레임율
#                       (sim 120 Hz / decimation 4 / render_interval 4 / update_period 1/30 →
#                        control·render·camera 가 같은 tick. per-camera frame counter 불필요)
#   Temporal model    : smooth_correlated(기본, exponential smoothing) | iid
#
# ■ 왜 step 최상단인가: ``ManagerBasedRLEnv.step()`` 은 decimation 루프 **안**에서 렌더하고,
#   mode="interval" EventTerm 은 그 렌더 **뒤**에 돈다 → interval term 으로 하면 1 프레임 늦는다.
#   PickCubeEnv.step() 최상단에서 write 하면 같은 step 의 render 가 새 pose 를 쓰고, obs manager
#   가 그 픽셀을 읽는다(RGB ↔ pose 동기, 지연 0).
#
# ■ Fabric 반영 (2026-07-30 실측 PASS): USD local xform write 는 세 카메라 모두 Fabric 렌더에
#   반영된다(정지 씬 픽셀 diff: noop 0.04 vs 0.20 m 이동 14~44/255). articulation 링크 자식
#   (wrist/front)도 동일. robot color DR 이 겪은 "USD-edit 무반영" 함정은 material 한정이었다.
#   단 RTX 시간축 누적(TAA/denoise) 때문에 큰 점프 직후 1 프레임에 잔상이 남는다(측정 restored
#   diff 1.2~2.8) → iid 모드가 비현실적으로 보이는 또 하나의 이유. 기본은 smooth_correlated.
# ---------------------------------------------------------------------------


CAMERA_DR_KEYS: tuple[str, ...] = ("top", "wrist", "front")
"""extrinsic DR 대상 카메라 key. scene asset 이름은 ``f"{key}_camera"``."""


@configclass
class CameraExtrinsicDRTermCfg:
    """카메라 1대의 extrinsic DR 범위. 모든 값은 **± 대칭 half-range**(비대칭 필요 없음).

    축 = nominal 카메라 로컬(OpenGL) 축: ``x=right`` · ``y=up`` · ``z=backward(−optical)``.
    회전도 같은 축 기준(x=tilt, y=pan, z=roll)이라 roll/pitch/yaw 로 부르지 않는다.

    - ``bias_*``   : 에피소드 내 고정(리셋마다 재추첨) — 카메라 재설치/캘리브 잔차 모델.
    - ``jitter_*`` : 매 카메라 프레임 갱신 — 미세 진동/pose 불확실성 모델.
    """

    enabled: bool = True
    bias_trans_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bias_rot_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    jitter_trans_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    jitter_rot_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    jitter_trans_alpha: float = 0.9
    """translation smoothing 계수. 1 에 가까울수록 느리게 변한다(=강한 시간 상관).

    ★``jitter_*`` 범위는 **hard clamp** 이고 실제로 실현되는 진폭은 그보다 훨씬 작다:
    uniform target 을 exponential smoothing 하면
    ``std(delta) = half_range/√3 · √((1−α)/(1+α))`` 이다. α=0.9 면 half_range 의 **0.229 배**
    (±3 mm → std 0.40 mm, 실측 0.39~0.41 mm 일치). 진폭을 키우려면 범위를 넓히기보다 α 를
    낮추는 게 직접적이다(대신 프레임간 변화가 커진다).
    """
    jitter_rot_alpha: float = 0.9


@configclass
class CameraExtrinsicDRCfg:
    """카메라 3종 extrinsic DR 설정. ``PickCubeDREnvCfg.camera_extrinsic_dr`` 로 주입.

    이 필드가 없거나 ``enabled=False`` 면 step 훅이 즉시 return → DR-off 변형 무영향.
    카메라가 씬에서 제거된 경로(SM state-only, ``remove_pick_cube_cameras``)도 자동 no-op.
    """

    enabled: bool = True
    temporal_mode: str = "smooth_correlated"
    """``smooth_correlated``(기본) | ``iid``.

    ``iid`` = 매 프레임 전 범위에서 독립 샘플 → smoothing 없음. 카메라가 순간이동하는 것처럼
    보이고 비현실적인 optical flow·플리커가 생긴다(RTX 시간축 누적 때문에 잔상까지 섞인다).
    frame-wise DR 논문 재현이나 강한 robustness ablation 전용 — 학습 기본값 아님.
    """
    use_episode_bias: bool = True
    """리셋마다 에피소드 고정 bias 를 추첨(per-frame 비용 0). 실기기 갭의 주범은 진동이 아니라
    설치·캘리브 오차라서 기본 on 이다. False 면 frame jitter 만 남는다."""
    initialize_random_offset: bool = True
    """True = 리셋 직후 첫 프레임부터 jitter 가 범위 안 임의값. False = nominal 에서 출발."""
    target_update_interval_frames: int = 1
    """새 jitter target 을 몇 카메라 프레임마다 뽑을지. 1 = 매 프레임(기본).
    2/4/8 로 늘리면 target 이 유지되는 동안 delta 가 그 target 으로 수렴해 더 느린 표류가 된다."""

    top: CameraExtrinsicDRTermCfg = CameraExtrinsicDRTermCfg(
        bias_trans_m=(0.015, 0.015, 0.015),
        bias_rot_deg=(1.5, 1.5, 1.5),
        jitter_trans_m=(0.003, 0.003, 0.003),
        jitter_rot_deg=(0.4, 0.4, 0.4),
        jitter_trans_alpha=0.90,
        jitter_rot_alpha=0.90,
    )
    """world 고정 oblique 부감(수직 top-down 아님, 하향 26.5°). 삼각대/클램프 재설치 오차를
    감당해야 하므로 bias 를 가장 크게 준다."""
    wrist: CameraExtrinsicDRTermCfg = CameraExtrinsicDRTermCfg(
        bias_trans_m=(0.003, 0.003, 0.003),
        bias_rot_deg=(1.0, 1.0, 1.0),
        jitter_trans_m=(0.001, 0.001, 0.001),
        jitter_rot_deg=(0.3, 0.3, 0.3),
        jitter_trans_alpha=0.95,
        jitter_rot_alpha=0.95,
    )
    """gripper 링크 자식(eye-in-hand). 로봇 모션 자체가 시야를 크게 흔들므로 범위를 가장 좁게,
    smoothing 을 가장 강하게. 나사 체결 mount 라 실제 설치 오차도 작다."""
    front: CameraExtrinsicDRTermCfg = CameraExtrinsicDRTermCfg(
        bias_trans_m=(0.005, 0.005, 0.005),
        bias_rot_deg=(1.0, 1.0, 1.0),
        jitter_trans_m=(0.0015, 0.0015, 0.0015),
        jitter_rot_deg=(0.3, 0.3, 0.3),
        jitter_trans_alpha=0.92,
        jitter_rot_alpha=0.92,
    )
    """★shoulder 링크 자식 — 고정 3인칭이 아니라 shoulder_pan 회전을 따라간다. pan 회전이
    이미 큰 시점 변화를 주므로 wrist 급으로 좁게 잡는다."""


class CameraExtrinsicDR:
    """카메라 3종의 6-DoF extrinsic DR 상태 + 매 프레임 pose write.

    상태는 ``(C, N, 6)`` 텐서 하나로 유지한다(C=카메라, N=env, ``[:3]``=translation m,
    ``[3:]``=rotation rad). 카메라 3개 루프만 Python 이고 env 방향은 전부 vectorized.

    ★누적 금지: pose 는 매 프레임 **nominal 에서 다시** 계산한다(현재 pose 에 delta 를 곱하지
    않는다). temporal state 는 bounded delta 값으로만 들고 있어 장시간 실행에도 발산하지 않는다.
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: CameraExtrinsicDRCfg):
        self.cfg = cfg
        self._env = env
        self._frame = 0
        self._cams: list = []
        self._keys: list[str] = []
        half_bias: list[list[float]] = []
        half_jit: list[list[float]] = []
        alpha: list[list[float]] = []
        pos_nom: list[torch.Tensor] = []
        quat_nom: list[torch.Tensor] = []

        for key in CAMERA_DR_KEYS:
            term: CameraExtrinsicDRTermCfg | None = getattr(cfg, key, None)
            if term is None or not term.enabled:
                continue
            cam = env.scene.sensors.get(f"{key}_camera")
            # 카메라가 없거나(SM state-only 경로) 아직 초기화 전이면 대상에서 제외.
            if cam is None or getattr(cam, "_view", None) is None:
                continue
            # nominal = prim 에 실제 저장된 local pose (OpenGL 카메라 규약, wxyz).
            p, q = cam._view.get_local_poses()
            pos_nom.append(p.clone().to(env.device))
            quat_nom.append(q.clone().to(env.device))
            self._cams.append(cam)
            self._keys.append(key)
            half_bias.append([*term.bias_trans_m, *(math.radians(d) for d in term.bias_rot_deg)])
            half_jit.append([*term.jitter_trans_m, *(math.radians(d) for d in term.jitter_rot_deg)])
            alpha.append([term.jitter_trans_alpha] * 3 + [term.jitter_rot_alpha] * 3)

        self.num_cameras = len(self._cams)
        if self.num_cameras == 0:
            return
        dev, n = env.device, env.num_envs
        self._pos_nom = torch.stack(pos_nom)                       # (C, N, 3)
        self._quat_nom = torch.stack(quat_nom)                     # (C, N, 4) wxyz
        self._half_bias = torch.tensor(half_bias, device=dev).unsqueeze(1)   # (C, 1, 6)
        self._half_jit = torch.tensor(half_jit, device=dev).unsqueeze(1)     # (C, 1, 6)
        self._alpha = torch.tensor(alpha, device=dev).unsqueeze(1)           # (C, 1, 6)
        self._bias = torch.zeros((self.num_cameras, n, 6), device=dev)
        self._delta = torch.zeros((self.num_cameras, n, 6), device=dev)
        self._target = torch.zeros((self.num_cameras, n, 6), device=dev)

    @property
    def keys(self) -> list[str]:
        """DR 이 실제로 적용되는 카메라 key 목록(씬에 없거나 disabled 는 제외됨)."""
        return list(self._keys)

    def _sample(self, half: torch.Tensor, num_envs: int) -> torch.Tensor:
        """``(C, num_envs, 6)`` uniform ``[-half, +half]``. half=0 인 축은 분기 없이 0."""
        u = torch.rand((self.num_cameras, num_envs, 6), device=half.device) * 2.0 - 1.0
        return u * half

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """리셋된 env 의 temporal state 만 초기화(+ episode bias 재추첨).

        리셋되지 않은 env 의 상태는 건드리지 않는다. 여기서 pose 를 즉시 write 하는 이유:
        ``_reset_idx`` 직후 ``num_rerenders_on_reset`` 재렌더가 새 bias 를 보게 하려는 것.
        """
        if self.num_cameras == 0:
            return
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._delta.device)
        elif not torch.is_tensor(env_ids):
            env_ids = torch.as_tensor(list(env_ids), device=self._delta.device)
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.to(self._delta.device)
        n = int(env_ids.numel())

        if self.cfg.use_episode_bias:
            self._bias[:, env_ids] = self._sample(self._half_bias, n)
        else:
            self._bias[:, env_ids] = 0.0
        fresh = self._sample(self._half_jit, n) if self.cfg.initialize_random_offset else \
            torch.zeros((self.num_cameras, n, 6), device=self._delta.device)
        self._delta[:, env_ids] = fresh
        self._target[:, env_ids] = fresh
        self._write(env_ids)

    def update(self) -> None:
        """카메라 프레임 1개분 jitter 갱신 + pose write. ``env.step()`` 최상단에서 호출."""
        if self.num_cameras == 0:
            return
        n = self._env.num_envs
        if self.cfg.temporal_mode == "iid":
            # 매 프레임 독립 샘플 — smoothing 없음(ablation 전용, 클래스 docstring 참조).
            self._delta = self._sample(self._half_jit, n)
        else:
            if self._frame % max(1, int(self.cfg.target_update_interval_frames)) == 0:
                self._target = self._sample(self._half_jit, n)
            self._delta = self._alpha * self._delta + (1.0 - self._alpha) * self._target
            # 범위 밖으로 나갈 수는 없지만(볼록 결합) 수치 오차 방어로 clamp.
            self._delta = torch.clamp(self._delta, -self._half_jit, self._half_jit)
        self._frame += 1
        self._write()

    def randomized_local_poses(self) -> tuple[torch.Tensor, torch.Tensor]:
        """현재 delta 를 nominal 에 합성한 local pose ``((C,N,3), (C,N,4) wxyz)``.

        ★nominal 에서 재계산 — 이전 프레임 pose 를 재사용하지 않는다(발산 방지).
        회전 delta 는 nominal 오른쪽에 곱한다(= 카메라 로컬 축 기준 회전).
        """
        delta = self._bias + self._delta                                   # (C, N, 6)
        pos = self._pos_nom + math_utils.quat_apply(self._quat_nom, delta[..., :3])
        dq = math_utils.quat_from_euler_xyz(delta[..., 3], delta[..., 4], delta[..., 5])
        quat = math_utils.quat_mul(self._quat_nom, dq)
        return pos, torch.nn.functional.normalize(quat, dim=-1)

    def _write(self, env_ids: torch.Tensor | None = None) -> None:
        """local xform 을 USD 로 write. env 방향은 batched, 카메라 3개만 Python 루프.

        ``Camera.set_world_poses`` 대신 ``_view.set_local_poses`` 를 쓰는 이유: 전자는 부모
        world transform 을 USD ``ComputeLocalToWorldTransform`` 으로 재계산해(비용 + Fabric
        stale 위험) local 을 역산한다. 우리가 원하는 건 애초에 local mount 갱신이다.
        (private ``_view`` 접근 — IsaacLab 2.3.2 에 local pose 공개 API 가 없다.)
        view index == env index 는 TiledCamera 관측이 이미 의존하는 규약이다.
        """
        from pxr import Sdf

        pos, quat = self.randomized_local_poses()
        if env_ids is not None:
            pos, quat = pos[:, env_ids], quat[:, env_ids]
        # USD write 는 CPU 값을 요구한다(set_local_poses 내부 tolist). 카메라마다 동기화하지
        # 않도록 한 번에 옮긴다.
        # ponytail: set_local_poses 는 prim 마다 Python xformOp Set 루프(translate·orient 각
        # 1회전) 라 프레임당 6×num_envs 회전이다. 16 env 실측 비용 = +2.8 ms/step(79.6→82.4,
        # +3.5%). 더 줄이려면 xformOp 두 개를 한 루프에서 쓰는 자체 writer 가 필요한데
        # isaacsim 의 attribute 타입 처리(Quatd/Quatf)를 복제해야 한다 — 그 값어치가 될 때만.
        pos_c, quat_c = pos.cpu(), quat.cpu()
        idx = None if env_ids is None else env_ids.tolist()
        with Sdf.ChangeBlock():
            for c, cam in enumerate(self._cams):
                cam._view.set_local_poses(pos_c[c], quat_c[c], indices=idx)


def _get_camera_extrinsic_dr(env: ManagerBasedRLEnv) -> CameraExtrinsicDR | None:
    """env 에 캐시된 DR 상태를 돌려주고, 없으면 만든다. cfg 없거나 off 면 None."""
    cfg: CameraExtrinsicDRCfg | None = getattr(env.cfg, "camera_extrinsic_dr", None)
    if cfg is None or not cfg.enabled:
        return None
    state: CameraExtrinsicDR | None = getattr(env, "_camera_extrinsic_dr", None)
    if state is not None:
        return state
    state = CameraExtrinsicDR(env, cfg)
    if state.num_cameras == 0:
        # 카메라 없음(SM state-only 경로) = 정상 no-op. 캐시하지 않는다 — sensor 초기화가
        # 아직 안 끝난 이른 호출일 수도 있어서 다음 프레임에 다시 시도한다(비용 = dict get 3회).
        # 단 씬에 카메라가 **있는데** 잡히지 않으면 조용히 DR-off 가 되므로 한 번 경고한다.
        if not getattr(env, "_camera_extrinsic_dr_warned", False) and any(
                f"{k}_camera" in env.scene.sensors for k in CAMERA_DR_KEYS):
            env._camera_extrinsic_dr_warned = True
            print("[camera-dr] ⚠ 카메라 asset 은 있는데 sensor view 가 아직 없음 → 이 프레임 no-op",
                  flush=True)
        return None
    env._camera_extrinsic_dr = state
    print(f"[camera-dr] extrinsic DR on: {state.keys} mode={cfg.temporal_mode} "
          f"episode_bias={cfg.use_episode_bias}", flush=True)
    return state


def update_camera_extrinsic_dr(env: ManagerBasedRLEnv) -> None:
    """카메라 extrinsic DR 프레임 갱신 — ``env.step()`` **최상단**(렌더 전)에서 호출.

    cfg 미주입·``enabled=False``·카메라 없는 경로에서는 즉시 return 한다.
    """
    state = _get_camera_extrinsic_dr(env)
    if state is not None:
        state.update()


def _reset_camera_extrinsic_dr_fn(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    state = _get_camera_extrinsic_dr(env)
    if state is not None:
        state.reset(env_ids)


def reset_camera_extrinsic_dr() -> EventTerm:
    """카메라 extrinsic DR temporal state 리셋 + episode bias 재추첨 EventTerm(reset).

    frame-wise 갱신은 EventTerm 이 아니라 ``PickCubeEnv.step()`` 훅이 한다
    (mode="interval" 은 렌더 뒤에 돌아서 1 프레임 늦다). 범위·모드는
    ``env_cfg.camera_extrinsic_dr``(:class:`CameraExtrinsicDRCfg`) 가 단일 소스다.
    """
    return EventTerm(func=_reset_camera_extrinsic_dr_fn, mode="reset", params={})
