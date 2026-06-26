"""회전 변환 헬퍼 — leisaac ``utils/math_utils.py`` verbatim vendor.

``device_base.Device._convert_delta_from_frame`` 가 의존한다. isaaclab.utils.math 만 사용
(isaac-sim Docker 에서 가용).
"""

import isaaclab.utils.math as math_utils
import torch


def rotvec_to_euler(rotvec: torch.Tensor) -> torch.Tensor:
    """rotation vector(axis-angle) 배치를 Euler XYZ delta 로 변환."""
    # |rotvec| = 환경별 회전 크기 (shape: [N, 1])
    rotvec_norm = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    rotvec_norm_clamped = torch.clamp(rotvec_norm, min=1.0e-8)
    axis = rotvec / rotvec_norm_clamped

    # norm ~ 0 이면 axis 방향이 불명 → +X 로 폴백
    default_axis = torch.tensor([1.0, 0.0, 0.0], device=rotvec.device, dtype=axis.dtype).view(1, 3)
    axis = torch.where(rotvec_norm > 1.0e-8, axis, default_axis.repeat(rotvec.shape[0], 1))

    delta_quat = math_utils.quat_from_angle_axis(rotvec_norm.squeeze(-1), axis)
    delta_roll, delta_pitch, delta_yaw = math_utils.euler_xyz_from_quat(delta_quat)
    return torch.cat([delta_roll, delta_pitch, delta_yaw], dim=0)
