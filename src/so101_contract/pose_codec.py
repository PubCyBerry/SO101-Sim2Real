"""Phase 12 — format-neutral EEF pose codec (NumPy/Torch 공통).

세 pose format을 하나의 SE(3) 경로로 통일한다.

===========================  ====  =========================================
format                       dim   추가 계약
===========================  ====  =========================================
``xyz_rot6d_rows``           9     rotation matrix 첫 두 **row**
``xyz_quaternion_wxyz``      7     unit normalization, scalar-first,
                                   deterministic sign canonicalization
``xyz_rpy``                  6     radian, fixed-axis XYZ
                                   ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``,
                                   명시적 wrap ``[-π, π)``
===========================  ====  =========================================

위 dim은 gripper를 제외한 pose 차원이다. gripper 1D를 포함한 전체 state/action은
각각 10D/8D/7D다.

**모든 relative 변환은 rotation matrix/SE(3)를 거친다.** Rot6D·quaternion·RPY vector를
직접 빼거나 더하지 않는다.

.. code-block:: text

    T_relative[h] = inverse(T_state) @ T_action[h]
    T_absolute[h] = T_state @ T_relative[h]

입력 backend(NumPy/Torch), dtype, device를 보존하며 chunk의 모든 timestep은 동일한
기준 state pose를 공유한다.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

from .action_representation import PoseFormat, coerce_pose_format, pose_format_dims
from .array_backend import (
    Array,
    absolute as _abs,
    acos as _acos,
    any_true as _any,
    asin as _asin,
    atan2 as _atan2,
    check_array as _check_array,
    clip as _clip,
    concat as _concat,
    cos as _cos,
    epsilon as _eps,
    full_like as _full_like,
    movedim as _movedim,
    norm as _norm,
    replace_indices as _replace,
    require_same_backend as _same_backend,
    resolve_chunk_reference,
    sin as _sin,
    sqrt as _sqrt,
    stack as _stack,
    swap_last_two as _swap_last_two,
    take_indices as _take,
    validate_indices as _validate_indices,
    where as _where,
)
from .eef_relative_action import matrix_to_rot6d_rows, rot6d_rows_to_matrix

POSE_CODEC_VERSION = "so101_pose_codec_v2"

_TAU = 2.0 * math.pi


# --- quaternion --------------------------------------------------------------


def _quaternion_to_matrix(quaternion: Array) -> Array:
    """정규화된 ``wxyz`` quaternion ``(...,4)`` → rotation matrix ``(...,3,3)``."""
    w = quaternion[..., 0]
    x = quaternion[..., 1]
    y = quaternion[..., 2]
    z = quaternion[..., 3]
    row0 = _stack(
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        axis=-1,
    )
    row1 = _stack(
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        axis=-1,
    )
    row2 = _stack(
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        axis=-1,
    )
    return _stack([row0, row1, row2], axis=-2)


def _matrix_to_quaternion(rotation: Array) -> Array:
    """Rotation matrix ``(...,3,3)`` → canonical unit ``wxyz`` quaternion ``(...,4)``.

    수치적으로 안정한 4-branch 공식을 batch 전체에 대해 계산한 뒤 가장 큰 trace 항을
    고른다(0으로 나누는 branch를 실행하지 않기 위해 모든 후보를 clip한 값으로 계산).
    """
    m00 = rotation[..., 0, 0]
    m01 = rotation[..., 0, 1]
    m02 = rotation[..., 0, 2]
    m10 = rotation[..., 1, 0]
    m11 = rotation[..., 1, 1]
    m12 = rotation[..., 1, 2]
    m20 = rotation[..., 2, 0]
    m21 = rotation[..., 2, 1]
    m22 = rotation[..., 2, 2]

    floor = 10.0 * _eps(rotation)
    trace_w = 1.0 + m00 + m11 + m22
    trace_x = 1.0 + m00 - m11 - m22
    trace_y = 1.0 - m00 + m11 - m22
    trace_z = 1.0 - m00 - m11 + m22

    scale_w = _sqrt(_clip(trace_w, floor, None)) * 2.0
    scale_x = _sqrt(_clip(trace_x, floor, None)) * 2.0
    scale_y = _sqrt(_clip(trace_y, floor, None)) * 2.0
    scale_z = _sqrt(_clip(trace_z, floor, None)) * 2.0

    candidate_w = _stack(
        [0.25 * scale_w, (m21 - m12) / scale_w, (m02 - m20) / scale_w, (m10 - m01) / scale_w],
        axis=-1,
    )
    candidate_x = _stack(
        [(m21 - m12) / scale_x, 0.25 * scale_x, (m01 + m10) / scale_x, (m02 + m20) / scale_x],
        axis=-1,
    )
    candidate_y = _stack(
        [(m02 - m20) / scale_y, (m01 + m10) / scale_y, 0.25 * scale_y, (m12 + m21) / scale_y],
        axis=-1,
    )
    candidate_z = _stack(
        [(m10 - m01) / scale_z, (m02 + m20) / scale_z, (m12 + m21) / scale_z, 0.25 * scale_z],
        axis=-1,
    )

    use_w = (trace_w >= trace_x) & (trace_w >= trace_y) & (trace_w >= trace_z)
    use_x = (trace_x >= trace_y) & (trace_x >= trace_z)
    use_y = trace_y >= trace_z
    quaternion = _where(
        use_w[..., None],
        candidate_w,
        _where(
            use_x[..., None],
            candidate_x,
            _where(use_y[..., None], candidate_y, candidate_z),
        ),
    )
    return canonicalize_quaternion_wxyz(quaternion)


def canonicalize_quaternion_wxyz(quaternion: Array) -> Array:
    """Unit normalization + deterministic sign canonicalization.

    ``q``와 ``-q``는 같은 회전이므로 학습/통계의 불연속을 없애기 위해 부호를 고정한다.
    규칙은 ``w > 0`` 우선이며, ``w ≈ 0``이면 첫 번째 유의미한 성분이 양수가 되게 한다.
    """
    values = _check_array(quaternion, "quaternion", last_dim=4)
    norm = _norm(values, axis=-1, keepdims=True)
    threshold = max(1e-12, 10.0 * _eps(values))
    if _any(norm <= threshold):
        raise ValueError("quaternion norm is zero or numerically unstable")
    values = values / norm

    significant = max(1e-9, math.sqrt(_eps(values)))
    ones = _full_like(values[..., 0], 1.0)
    zeros = _full_like(values[..., 0], 0.0)
    sign = zeros
    for index in range(4):
        component = values[..., index]
        # 아직 부호가 정해지지 않은 원소만, 첫 유의미한 성분의 부호로 결정한다.
        candidate = _where(
            _abs(component) > significant,
            _where(component >= 0.0, ones, -ones),
            zeros,
        )
        sign = _where(_abs(sign) > 0.0, sign, candidate)
    # unit quaternion은 항상 |성분| >= 0.5 인 성분이 있으므로 fallback은 방어용이다.
    sign = _where(_abs(sign) > 0.0, sign, ones)
    return values * sign[..., None]


def canonicalize_quaternion_sequence(
    quaternions: Array,
    *,
    axis: int = -2,
    reference: Array | None = None,
) -> Array:
    """Sequence 연속성 helper.

    부호 정규화는 hemisphere를 고정하지만 회전이 ``w = 0`` 경계를 지나면 여전히
    ±180° 점프가 남는다. chunk 안에서 인접 quaternion의 내적이 음수면 부호를 뒤집어
    시간축 연속성을 만든다. ``reference``를 주면 첫 원소를 그 기준에 맞춘다.
    """
    values = _check_array(quaternions, "quaternions", last_dim=4)
    if values.ndim < 2:
        raise ValueError("quaternion sequence needs a time axis")
    moved = _movedim(values, axis, 0)
    if moved.shape[0] == 0:
        raise ValueError("quaternion sequence is empty")

    previous: Array | None = None
    if reference is not None:
        previous = _check_array(reference, "reference", last_dim=4)
        _same_backend(values, previous, left_name="quaternions", right_name="reference")

    aligned: list[Array] = []
    ones = _full_like(moved[0][..., 0], 1.0)
    for index in range(moved.shape[0]):
        current = moved[index]
        if previous is not None:
            dot = (previous * current).sum(-1)
            sign = _where(dot < 0.0, -ones, ones)
            current = current * sign[..., None]
        aligned.append(current)
        previous = current
    return _movedim(_stack(aligned, axis=0), 0, axis)


# --- roll/pitch/yaw ----------------------------------------------------------


def wrap_angles_to_pi(angles: Array) -> Array:
    """각도를 ``[-π, π)``로 명시적으로 wrap."""
    values = _check_array(angles, "angles")
    return (values + math.pi) % _TAU - math.pi


def _rpy_to_matrix(rpy: Array) -> Array:
    """Fixed-axis XYZ ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""
    roll = rpy[..., 0]
    pitch = rpy[..., 1]
    yaw = rpy[..., 2]
    cos_r, sin_r = _cos(roll), _sin(roll)
    cos_p, sin_p = _cos(pitch), _sin(pitch)
    cos_y, sin_y = _cos(yaw), _sin(yaw)
    row0 = _stack(
        [
            cos_y * cos_p,
            cos_y * sin_p * sin_r - sin_y * cos_r,
            cos_y * sin_p * cos_r + sin_y * sin_r,
        ],
        axis=-1,
    )
    row1 = _stack(
        [
            sin_y * cos_p,
            sin_y * sin_p * sin_r + cos_y * cos_r,
            sin_y * sin_p * cos_r - cos_y * sin_r,
        ],
        axis=-1,
    )
    row2 = _stack([-sin_p, cos_p * sin_r, cos_p * cos_r], axis=-1)
    return _stack([row0, row1, row2], axis=-2)


def _matrix_to_rpy(rotation: Array) -> Array:
    """Rotation matrix → fixed-axis XYZ ``[roll, pitch, yaw]``.

    Gimbal singularity(``|pitch| → π/2``)에서는 roll/yaw가 축퇴하므로 ``roll = 0``으로
    고정하고 남은 자유도를 yaw에 몰아 넣는다. 두 branch를 모두 계산한 뒤 선택하므로
    NaN gradient나 backend별 분기 차이가 생기지 않는다.
    """
    m00 = rotation[..., 0, 0]
    m01 = rotation[..., 0, 1]
    m10 = rotation[..., 1, 0]
    m11 = rotation[..., 1, 1]
    m20 = rotation[..., 2, 0]
    m21 = rotation[..., 2, 1]
    m22 = rotation[..., 2, 2]

    pitch = _asin(_clip(-m20, -1.0, 1.0))
    cos_pitch = _cos(pitch)
    tolerance = max(1e-9, math.sqrt(_eps(rotation)))
    regular = _abs(cos_pitch) > tolerance

    zeros = _full_like(pitch, 0.0)
    roll = _where(regular, _atan2(m21, m22), zeros)
    yaw = _where(regular, _atan2(m10, m00), _atan2(-m01, m11))
    return wrap_angles_to_pi(_stack([roll, pitch, yaw], axis=-1))


# --- pose encode/decode ------------------------------------------------------


def decode_pose(pose: Array, pose_format: PoseFormat | str) -> tuple[Array, Array]:
    """Pose vector ``(...,P)`` → ``(translation (...,3), rotation (...,3,3))``."""
    resolved = coerce_pose_format(pose_format)
    _, pose_dim = pose_format_dims(resolved)
    values = _check_array(pose, "pose", last_dim=pose_dim)
    translation = values[..., :3]
    rotation_values = values[..., 3:]
    if resolved is PoseFormat.XYZ_ROT6D_ROWS:
        rotation = rot6d_rows_to_matrix(rotation_values)
    elif resolved is PoseFormat.XYZ_QUATERNION_WXYZ:
        rotation = _quaternion_to_matrix(canonicalize_quaternion_wxyz(rotation_values))
    else:
        rotation = _rpy_to_matrix(rotation_values)
    return translation, rotation


def encode_pose(
    translation: Array,
    rotation: Array,
    pose_format: PoseFormat | str,
) -> Array:
    """``(translation, rotation matrix)`` → pose vector ``(...,P)``."""
    resolved = coerce_pose_format(pose_format)
    _check_array(translation, "translation", last_dim=3)
    _check_array(rotation, "rotation")
    if rotation.shape[-2:] != (3, 3):
        raise ValueError(f"rotation shape must end in (3,3), got {tuple(rotation.shape)}")
    _same_backend(translation, rotation, left_name="translation", right_name="rotation")

    if resolved is PoseFormat.XYZ_ROT6D_ROWS:
        rotation_values = matrix_to_rot6d_rows(rotation)
    elif resolved is PoseFormat.XYZ_QUATERNION_WXYZ:
        rotation_values = _matrix_to_quaternion(rotation)
    else:
        rotation_values = _matrix_to_rpy(rotation)
    return _concat([translation, rotation_values], axis=-1)


def canonicalize_pose(pose: Array, pose_format: PoseFormat | str) -> Array:
    """Format별 canonical 형태로 정규화(Rot6D 직교화, quaternion 부호/노름, RPY wrap)."""
    translation, rotation = decode_pose(pose, pose_format)
    return encode_pose(translation, rotation, pose_format)


def convert_pose_format(
    pose: Array,
    source_format: PoseFormat | str,
    target_format: PoseFormat | str,
) -> Array:
    """Pose format 간 변환. 항상 rotation matrix를 경유한다."""
    translation, rotation = decode_pose(pose, source_format)
    return encode_pose(translation, rotation, target_format)


def rotation_geodesic_angle(left: Array, right: Array) -> Array:
    """두 rotation matrix 사이 geodesic 각도(rad). residual/metric 보고용."""
    _check_array(left, "left")
    _check_array(right, "right")
    delta = _swap_last_two(left) @ right
    trace = delta[..., 0, 0] + delta[..., 1, 1] + delta[..., 2, 2]
    cosine = _clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return _acos(cosine)


# --- SE(3) absolute ↔ relative ----------------------------------------------


def absolute_pose_to_relative(
    state_pose: Array,
    action_pose: Array,
    pose_format: PoseFormat | str,
) -> Array:
    """``T_relative[h] = inverse(T_state) @ T_action[h]``. 모든 h가 같은 state를 기준으로 한다."""
    resolved = coerce_pose_format(pose_format)
    _, pose_dim = pose_format_dims(resolved)
    reference = resolve_chunk_reference(
        state_pose,
        action_pose,
        pose_dim,
        state_name="state_pose",
        action_name="action_pose",
    )

    state_translation, state_rotation = decode_pose(reference, resolved)
    action_translation, action_rotation = decode_pose(action_pose, resolved)

    state_rotation_inverse = _swap_last_two(state_rotation)[..., None, :, :]
    relative_rotation = state_rotation_inverse @ action_rotation
    translation_delta = action_translation - state_translation[..., None, :]
    relative_translation = (state_rotation_inverse @ translation_delta[..., None])[..., 0]
    return encode_pose(relative_translation, relative_rotation, resolved)


def relative_pose_to_absolute(
    state_pose: Array,
    relative_pose: Array,
    pose_format: PoseFormat | str,
) -> Array:
    """``T_absolute[h] = T_state @ T_relative[h]``."""
    resolved = coerce_pose_format(pose_format)
    _, pose_dim = pose_format_dims(resolved)
    reference = resolve_chunk_reference(
        state_pose,
        relative_pose,
        pose_dim,
        state_name="state_pose",
        action_name="relative_pose",
    )

    state_translation, state_rotation = decode_pose(reference, resolved)
    relative_translation, relative_rotation = decode_pose(relative_pose, resolved)

    state_rotation_chunk = state_rotation[..., None, :, :]
    absolute_rotation = state_rotation_chunk @ relative_rotation
    absolute_translation = state_translation[..., None, :] + (
        state_rotation_chunk @ relative_translation[..., None]
    )[..., 0]
    return encode_pose(absolute_translation, absolute_rotation, resolved)


# --- full feature vector (pose group + passthrough) --------------------------


def absolute_actions_to_relative(
    state: Array,
    actions: Array,
    pose_format: PoseFormat | str,
    *,
    state_pose_indices: Sequence[int] | None = None,
    action_pose_indices: Sequence[int] | None = None,
) -> Array:
    """Full feature action에서 pose group만 relative로 바꾸고 gripper는 passthrough."""
    resolved = coerce_pose_format(pose_format)
    _, pose_dim = pose_format_dims(resolved)
    _same_backend(state, actions, left_name="state", right_name="actions")
    default = tuple(range(pose_dim))
    state_indices = _validate_indices(
        default if state_pose_indices is None else state_pose_indices,
        feature_dim=state.shape[-1],
        expected_count=pose_dim,
        name="state_pose_indices",
    )
    action_indices = _validate_indices(
        default if action_pose_indices is None else action_pose_indices,
        feature_dim=actions.shape[-1],
        expected_count=pose_dim,
        name="action_pose_indices",
    )
    relative_pose = absolute_pose_to_relative(
        _take(state, state_indices),
        _take(actions, action_indices),
        resolved,
    )
    return _replace(actions, action_indices, relative_pose)


def relative_actions_to_absolute(
    state: Array,
    relative_actions: Array,
    pose_format: PoseFormat | str,
    *,
    state_pose_indices: Sequence[int] | None = None,
    action_pose_indices: Sequence[int] | None = None,
) -> Array:
    """Full feature action에서 pose group만 absolute로 복원하고 gripper는 passthrough."""
    resolved = coerce_pose_format(pose_format)
    _, pose_dim = pose_format_dims(resolved)
    _same_backend(state, relative_actions, left_name="state", right_name="relative_actions")
    default = tuple(range(pose_dim))
    state_indices = _validate_indices(
        default if state_pose_indices is None else state_pose_indices,
        feature_dim=state.shape[-1],
        expected_count=pose_dim,
        name="state_pose_indices",
    )
    action_indices = _validate_indices(
        default if action_pose_indices is None else action_pose_indices,
        feature_dim=relative_actions.shape[-1],
        expected_count=pose_dim,
        name="action_pose_indices",
    )
    absolute_pose = relative_pose_to_absolute(
        _take(state, state_indices),
        _take(relative_actions, action_indices),
        resolved,
    )
    return _replace(relative_actions, action_indices, absolute_pose)
