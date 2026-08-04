"""공통 Pick-and-Place 종료 조건."""

from __future__ import annotations

import math

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from ._geometry import DESK_TOP_Z

_REST_THRESHOLD_RAD: float = 15.0 * math.pi / 180.0


def object_inside_container_frame(
    env: ManagerBasedRLEnv | DirectRLEnv,
    object_pos_w: torch.Tensor,
    container_cfg: SceneEntityCfg,
    radius: float,
    height_range: tuple[float, float],
    max_tilt_deg: float = 30.0,
) -> torch.Tensor:
    """물체가 컨테이너 **로컬 프레임 안**에 들어 있는가.

    world 축 기준 "xy 거리 + z 창" 판정의 구멍 두 개를 막는다:

    1. **rim 위를 통과시킨다** — 그릇 rim 상단은 원점 기준 약 0.080 m 다. z 상한이 그보다 크면
       큐브가 rim 에 얹혀 있거나 그 위에 떠 있어도 성공으로 센다.
    2. **그릇 자세를 안 본다** — 그릇이 들리거나 기울거나 뒤집혀도 원점 기준 높이만 맞으면
       통과한다. "그릇을 들어 큐브 위에 덮는" 형태가 여기로 샌다.

    그래서 물체 위치를 **그릇 로컬 프레임**으로 옮겨 radial·높이를 재고, 그릇이 서 있는지
    (로컬 +z 가 world +z 와 `max_tilt_deg` 이내)까지 함께 요구한다. 뒤집힌 그릇은 로컬 z 가
    음수가 되므로 자동으로 탈락한다.
    """
    container: RigidObject = env.scene[container_cfg.name]
    container_pos = container.data.root_pos_w
    container_quat = container.data.root_quat_w

    # world → container local (역회전). quat wxyz.
    offset = object_pos_w - container_pos
    w, x, y, z = container_quat.unbind(dim=-1)
    # R^T · v  (R = quat 회전행렬)
    row0 = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], dim=-1)
    row1 = torch.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], dim=-1)
    row2 = torch.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], dim=-1)
    local = torch.stack([(row0 * offset).sum(-1), (row1 * offset).sum(-1), (row2 * offset).sum(-1)], dim=-1)

    inside_radial = torch.hypot(local[:, 0], local[:, 1]) < radius
    inside_height = (local[:, 2] > height_range[0]) & (local[:, 2] < height_range[1])
    # 그릇 로컬 +z(= row2 의 3번째 성분 = R[2,2]) 가 world +z 와 이루는 각.
    upright = row2[:, 2] > math.cos(math.radians(max_tilt_deg))
    return inside_radial & inside_height & upright


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


def _is_at_rest_pose(joint_pos: torch.Tensor) -> torch.Tensor:
    """모든 joint 가 ±15° 이내이면 True."""
    return (joint_pos.abs() < _REST_THRESHOLD_RAD).all(dim=-1)


def task_done(
    env: ManagerBasedRLEnv | DirectRLEnv,
    objects_cfg: list[SceneEntityCfg],
    container_center_xy: tuple[float, float] = (2.2, -0.17),
    container_cfg: SceneEntityCfg | None = None,
    radius: float = 0.05,
    height_range: tuple[float, float] = (0.005, 0.18),
    require_rest_pose: bool = True,
) -> torch.Tensor:
    """모든 물체가 컨테이너 안에 있고 (require_rest_pose 시) 팔이 rest pose 이면 True."""
    done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    if container_cfg is not None:
        # 컨테이너를 아는 경우 = 로컬 프레임 containment(rim 위·뒤집힌 그릇 배제).
        for obj_cfg in objects_cfg:
            obj: RigidObject = env.scene[obj_cfg.name]
            done = torch.logical_and(done, object_inside_container_frame(
                env, obj.data.root_pos_w, container_cfg, radius, height_range))
    else:
        # 컨테이너 asset 이 없으면 상수 중심 기준 원기둥으로 폴백(pen 잔재 경로).
        cx, cy = _container_xy(env, container_center_xy, container_cfg)
        for obj_cfg in objects_cfg:
            obj = env.scene[obj_cfg.name]
            obj_pos = obj.data.root_pos_w - env.scene.env_origins
            inside_xy = torch.hypot(obj_pos[:, 0] - cx, obj_pos[:, 1] - cy) < radius
            above_floor = obj_pos[:, 2] > (DESK_TOP_Z + height_range[0])
            below_lip = obj_pos[:, 2] < (DESK_TOP_Z + height_range[1])
            done = torch.logical_and(
                done,
                torch.logical_and(inside_xy, torch.logical_and(above_floor, below_lip)),
            )

    if require_rest_pose:
        robot: Articulation = env.scene["robot"]
        done = torch.logical_and(done, _is_at_rest_pose(robot.data.joint_pos))

    return done
