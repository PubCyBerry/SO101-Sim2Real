"""공유 상수 및 순수 수학 헬퍼 — common MDP 내부 전용."""

from __future__ import annotations

import torch

# world frame 책상 상판 z — scene authoring 상수와 동기화 유지
DESK_TOP_Z: float = 0.76

# jaw body 기준 실제 접촉 중심까지의 로컬 오프셋 (m)
JAW_GRASP_OFFSET: tuple[float, float, float] = (-0.021, -0.070, 0.020)

# 컨테이너(컵/그릇) scene-local 기본 위치 (m). env_cfg 에서 명시 주입 시 무시.
CONTAINER_DEFAULT_CENTER_XY: tuple[float, float] = (2.2, -0.17)


def _quat_apply_wxyz(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """wxyz quaternion으로 vec를 회전한다."""
    qw = quat[:, 0:1]
    qv = quat[:, 1:4]
    uv = torch.cross(qv, vec, dim=-1)
    uuv = torch.cross(qv, uv, dim=-1)
    return vec + 2.0 * (qw * uv + uuv)


def _yaw_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """wxyz quaternion에서 yaw(z축 회전)를 라디안으로 추출한다. (N,) 반환."""
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return torch.atan2(siny_cosp, cosy_cosp)
