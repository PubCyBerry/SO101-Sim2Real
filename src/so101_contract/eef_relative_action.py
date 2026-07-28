"""EEF ``xyz + Rot6D(rows)`` absolute↔relative SE(3) 변환.

LeRobot v3 dataset에는 absolute EEF state/action을 유지하고, policy processor에서
다음 변환을 적용하기 위한 공통 수학 구현이다.

.. code-block:: text

    T_relative = inverse(T_state) @ T_action
    T_absolute = T_state @ T_relative

회전 표현은 Isaac-GR00T N1.7과 같은 rotation matrix 첫 두 **row**를 사용한다.
NumPy와 Torch 입력을 모두 지원하며 입력 backend, dtype, device를 보존한다.

지원 shape:

- state: ``(9,)`` 또는 ``(T_obs,9)`` + action ``(H,9)``
- state: ``(B,9)`` 또는 ``(B,T_obs,9)`` + action ``(B,H,9)``

state와 action rank가 같으면 state의 마지막 observation을 기준 pose로 사용한다.
chunk의 모든 timestep은 동일한 기준 pose를 공유한다.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
import torch

EEF_RELATIVE_ACTION_VERSION = "so101_eef_relative_se3_v1"
EEF_POSE_DIM = 9
DEFAULT_EEF_POSE_INDICES = tuple(range(EEF_POSE_DIM))

Array: TypeAlias = np.ndarray | torch.Tensor


def _numpy_float_array(values: np.ndarray, name: str, *, last_dim: int | None = None) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind != "f":
        raise TypeError(f"{name} must use a floating dtype, got {array.dtype}")
    if array.ndim == 0:
        raise ValueError(f"{name} must have at least one dimension, got {array.shape}")
    if last_dim is not None and array.shape[-1] != last_dim:
        raise ValueError(f"{name} shape must end in {last_dim}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def _torch_float_tensor(
    values: torch.Tensor,
    name: str,
    *,
    last_dim: int | None = None,
) -> torch.Tensor:
    if not torch.is_floating_point(values):
        raise TypeError(f"{name} must use a floating dtype, got {values.dtype}")
    if values.ndim == 0:
        raise ValueError(f"{name} must have at least one dimension, got {tuple(values.shape)}")
    if last_dim is not None and values.shape[-1] != last_dim:
        raise ValueError(f"{name} shape must end in {last_dim}, got {tuple(values.shape)}")
    if not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} contains NaN or infinity")
    return values


def _numpy_epsilon(dtype: np.dtype) -> float:
    return max(1e-12, float(np.finfo(dtype).eps) * 10.0)


def _torch_epsilon(dtype: torch.dtype) -> float:
    return max(1e-12, float(torch.finfo(dtype).eps) * 10.0)


def _rot6d_rows_to_matrix_numpy(rot6d: np.ndarray) -> np.ndarray:
    values = _numpy_float_array(rot6d, "rot6d", last_dim=6)
    rows = values.reshape(*values.shape[:-1], 2, 3)
    row0 = rows[..., 0, :]
    row1_raw = rows[..., 1, :]
    epsilon = _numpy_epsilon(values.dtype)

    row0_norm = np.linalg.norm(row0, axis=-1, keepdims=True)
    if np.any(row0_norm <= epsilon):
        raise ValueError("rot6d first row norm is zero or numerically unstable")
    row0 = row0 / row0_norm

    row1 = row1_raw - np.sum(row1_raw * row0, axis=-1, keepdims=True) * row0
    row1_norm = np.linalg.norm(row1, axis=-1, keepdims=True)
    if np.any(row1_norm <= epsilon):
        raise ValueError("rot6d rows are parallel or numerically unstable")
    row1 = row1 / row1_norm
    row2 = np.cross(row0, row1)
    return np.stack([row0, row1, row2], axis=-2).astype(values.dtype, copy=False)


def _rot6d_rows_to_matrix_torch(rot6d: torch.Tensor) -> torch.Tensor:
    values = _torch_float_tensor(rot6d, "rot6d", last_dim=6)
    rows = values.reshape(*values.shape[:-1], 2, 3)
    row0 = rows[..., 0, :]
    row1_raw = rows[..., 1, :]
    epsilon = _torch_epsilon(values.dtype)

    row0_norm = torch.linalg.vector_norm(row0, dim=-1, keepdim=True)
    if bool(torch.any(row0_norm <= epsilon)):
        raise ValueError("rot6d first row norm is zero or numerically unstable")
    row0 = row0 / row0_norm

    row1 = row1_raw - torch.sum(row1_raw * row0, dim=-1, keepdim=True) * row0
    row1_norm = torch.linalg.vector_norm(row1, dim=-1, keepdim=True)
    if bool(torch.any(row1_norm <= epsilon)):
        raise ValueError("rot6d rows are parallel or numerically unstable")
    row1 = row1 / row1_norm
    row2 = torch.linalg.cross(row0, row1, dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def rot6d_rows_to_matrix(rot6d: Array) -> Array:
    """Rot6D 첫 두 row ``(...,6)``를 rotation matrix ``(...,3,3)``로 복원."""
    if isinstance(rot6d, torch.Tensor):
        return _rot6d_rows_to_matrix_torch(rot6d)
    if isinstance(rot6d, np.ndarray):
        return _rot6d_rows_to_matrix_numpy(rot6d)
    raise TypeError(f"rot6d must be numpy.ndarray or torch.Tensor, got {type(rot6d).__name__}")


def matrix_to_rot6d_rows(rotation: Array) -> Array:
    """Rotation matrix ``(...,3,3)``의 첫 두 row를 ``(...,6)``으로 encode."""
    if isinstance(rotation, torch.Tensor):
        values = _torch_float_tensor(rotation, "rotation")
        if values.shape[-2:] != (3, 3):
            raise ValueError(
                f"rotation shape must end in (3,3), got {tuple(values.shape)}"
            )
        return values[..., :2, :].reshape(*values.shape[:-2], 6)
    if isinstance(rotation, np.ndarray):
        values = _numpy_float_array(rotation, "rotation")
        if values.shape[-2:] != (3, 3):
            raise ValueError(f"rotation shape must end in (3,3), got {values.shape}")
        return values[..., :2, :].reshape(*values.shape[:-2], 6)
    raise TypeError(
        f"rotation must be numpy.ndarray or torch.Tensor, got {type(rotation).__name__}"
    )


def _resolve_numpy_reference(state_pose: np.ndarray, action_pose: np.ndarray) -> np.ndarray:
    state = _numpy_float_array(state_pose, "state_pose", last_dim=EEF_POSE_DIM)
    actions = _numpy_float_array(action_pose, "action_pose", last_dim=EEF_POSE_DIM)
    if state.dtype != actions.dtype:
        raise TypeError(f"state/action dtype mismatch: {state.dtype} != {actions.dtype}")
    if actions.ndim < 2:
        raise ValueError(
            f"action_pose must contain a chunk dimension, got shape {actions.shape}"
        )

    if state.ndim == actions.ndim:
        if state.shape[-2] == 0:
            raise ValueError("state observation history is empty")
        reference = state[..., -1, :]
    elif state.ndim == actions.ndim - 1:
        reference = state
    else:
        raise ValueError(
            "state/action rank mismatch: expected state rank equal to action rank "
            f"or one less, got {state.ndim} and {actions.ndim}"
        )
    if reference.shape[:-1] != actions.shape[:-2]:
        raise ValueError(
            f"state/action batch shape mismatch: {reference.shape[:-1]} != {actions.shape[:-2]}"
        )
    return reference


def _resolve_torch_reference(
    state_pose: torch.Tensor,
    action_pose: torch.Tensor,
) -> torch.Tensor:
    state = _torch_float_tensor(state_pose, "state_pose", last_dim=EEF_POSE_DIM)
    actions = _torch_float_tensor(action_pose, "action_pose", last_dim=EEF_POSE_DIM)
    if state.dtype != actions.dtype:
        raise TypeError(f"state/action dtype mismatch: {state.dtype} != {actions.dtype}")
    if state.device != actions.device:
        raise ValueError(f"state/action device mismatch: {state.device} != {actions.device}")
    if actions.ndim < 2:
        raise ValueError(
            f"action_pose must contain a chunk dimension, got shape {tuple(actions.shape)}"
        )

    if state.ndim == actions.ndim:
        if state.shape[-2] == 0:
            raise ValueError("state observation history is empty")
        reference = state[..., -1, :]
    elif state.ndim == actions.ndim - 1:
        reference = state
    else:
        raise ValueError(
            "state/action rank mismatch: expected state rank equal to action rank "
            f"or one less, got {state.ndim} and {actions.ndim}"
        )
    if reference.shape[:-1] != actions.shape[:-2]:
        raise ValueError(
            "state/action batch shape mismatch: "
            f"{tuple(reference.shape[:-1])} != {tuple(actions.shape[:-2])}"
        )
    return reference


def _absolute_eef_to_relative_numpy(
    state_pose: np.ndarray,
    action_pose: np.ndarray,
) -> np.ndarray:
    actions = _numpy_float_array(action_pose, "action_pose", last_dim=EEF_POSE_DIM)
    reference = _resolve_numpy_reference(state_pose, actions)
    state_rotation = _rot6d_rows_to_matrix_numpy(reference[..., 3:])
    action_rotation = _rot6d_rows_to_matrix_numpy(actions[..., 3:])

    state_rotation_inverse = np.swapaxes(state_rotation, -1, -2)
    relative_rotation = (
        state_rotation_inverse[..., None, :, :] @ action_rotation
    )
    translation_delta = actions[..., :3] - reference[..., None, :3]
    relative_translation = (
        state_rotation_inverse[..., None, :, :] @ translation_delta[..., None]
    )[..., 0]
    relative_rot6d = matrix_to_rot6d_rows(relative_rotation)
    return np.concatenate([relative_translation, relative_rot6d], axis=-1).astype(
        actions.dtype,
        copy=False,
    )


def _absolute_eef_to_relative_torch(
    state_pose: torch.Tensor,
    action_pose: torch.Tensor,
) -> torch.Tensor:
    actions = _torch_float_tensor(action_pose, "action_pose", last_dim=EEF_POSE_DIM)
    reference = _resolve_torch_reference(state_pose, actions)
    state_rotation = _rot6d_rows_to_matrix_torch(reference[..., 3:])
    action_rotation = _rot6d_rows_to_matrix_torch(actions[..., 3:])

    state_rotation_inverse = state_rotation.transpose(-1, -2)
    relative_rotation = (
        state_rotation_inverse.unsqueeze(-3) @ action_rotation
    )
    translation_delta = actions[..., :3] - reference[..., None, :3]
    relative_translation = (
        state_rotation_inverse.unsqueeze(-3) @ translation_delta.unsqueeze(-1)
    ).squeeze(-1)
    relative_rot6d = matrix_to_rot6d_rows(relative_rotation)
    return torch.cat([relative_translation, relative_rot6d], dim=-1)


def absolute_eef_to_relative(state_pose: Array, action_pose: Array) -> Array:
    """Absolute EEF action chunk를 current-state-relative SE(3) pose로 변환."""
    if isinstance(state_pose, torch.Tensor) and isinstance(action_pose, torch.Tensor):
        return _absolute_eef_to_relative_torch(state_pose, action_pose)
    if isinstance(state_pose, np.ndarray) and isinstance(action_pose, np.ndarray):
        return _absolute_eef_to_relative_numpy(state_pose, action_pose)
    raise TypeError(
        "state_pose and action_pose must use the same numpy/torch backend, got "
        f"{type(state_pose).__name__} and {type(action_pose).__name__}"
    )


def _relative_eef_to_absolute_numpy(
    state_pose: np.ndarray,
    relative_pose: np.ndarray,
) -> np.ndarray:
    relative = _numpy_float_array(relative_pose, "relative_pose", last_dim=EEF_POSE_DIM)
    reference = _resolve_numpy_reference(state_pose, relative)
    state_rotation = _rot6d_rows_to_matrix_numpy(reference[..., 3:])
    relative_rotation = _rot6d_rows_to_matrix_numpy(relative[..., 3:])

    absolute_rotation = state_rotation[..., None, :, :] @ relative_rotation
    absolute_translation = reference[..., None, :3] + (
        state_rotation[..., None, :, :] @ relative[..., :3, None]
    )[..., 0]
    absolute_rot6d = matrix_to_rot6d_rows(absolute_rotation)
    return np.concatenate([absolute_translation, absolute_rot6d], axis=-1).astype(
        relative.dtype,
        copy=False,
    )


def _relative_eef_to_absolute_torch(
    state_pose: torch.Tensor,
    relative_pose: torch.Tensor,
) -> torch.Tensor:
    relative = _torch_float_tensor(relative_pose, "relative_pose", last_dim=EEF_POSE_DIM)
    reference = _resolve_torch_reference(state_pose, relative)
    state_rotation = _rot6d_rows_to_matrix_torch(reference[..., 3:])
    relative_rotation = _rot6d_rows_to_matrix_torch(relative[..., 3:])

    absolute_rotation = state_rotation.unsqueeze(-3) @ relative_rotation
    absolute_translation = reference[..., None, :3] + (
        state_rotation.unsqueeze(-3) @ relative[..., :3].unsqueeze(-1)
    ).squeeze(-1)
    absolute_rot6d = matrix_to_rot6d_rows(absolute_rotation)
    return torch.cat([absolute_translation, absolute_rot6d], dim=-1)


def relative_eef_to_absolute(state_pose: Array, relative_pose: Array) -> Array:
    """Current-state-relative EEF chunk를 absolute SE(3) target으로 복원."""
    if isinstance(state_pose, torch.Tensor) and isinstance(relative_pose, torch.Tensor):
        return _relative_eef_to_absolute_torch(state_pose, relative_pose)
    if isinstance(state_pose, np.ndarray) and isinstance(relative_pose, np.ndarray):
        return _relative_eef_to_absolute_numpy(state_pose, relative_pose)
    raise TypeError(
        "state_pose and relative_pose must use the same numpy/torch backend, got "
        f"{type(state_pose).__name__} and {type(relative_pose).__name__}"
    )


def _validate_pose_indices(
    indices: Sequence[int],
    *,
    feature_dim: int,
    name: str,
) -> tuple[int, ...]:
    resolved = tuple(int(index) for index in indices)
    if len(resolved) != EEF_POSE_DIM:
        raise ValueError(f"{name} must contain exactly {EEF_POSE_DIM} indices, got {resolved}")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{name} contains duplicate indices: {resolved}")
    if any(index < 0 or index >= feature_dim for index in resolved):
        raise ValueError(f"{name} is outside feature dimension {feature_dim}: {resolved}")
    return resolved


def _take_indices(values: Array, indices: tuple[int, ...]) -> Array:
    if isinstance(values, torch.Tensor):
        index = torch.tensor(indices, dtype=torch.long, device=values.device)
        return torch.index_select(values, dim=-1, index=index)
    return values[..., list(indices)]


def _replace_indices(values: Array, indices: tuple[int, ...], replacement: Array) -> Array:
    if isinstance(values, torch.Tensor):
        output = values.clone()
        index = torch.tensor(indices, dtype=torch.long, device=values.device)
        output.index_copy_(-1, index, replacement)
        return output
    output = values.copy()
    output[..., list(indices)] = replacement
    return output


def absolute_actions_to_relative(
    state: Array,
    actions: Array,
    *,
    state_pose_indices: Sequence[int] = DEFAULT_EEF_POSE_INDICES,
    action_pose_indices: Sequence[int] = DEFAULT_EEF_POSE_INDICES,
) -> Array:
    """Full feature action에서 EEF 9D만 relative로 바꾸고 나머지는 passthrough."""
    same_backend = (
        isinstance(state, torch.Tensor)
        and isinstance(actions, torch.Tensor)
        or isinstance(state, np.ndarray)
        and isinstance(actions, np.ndarray)
    )
    if not same_backend:
        raise TypeError(
            "state and actions must use the same numpy/torch backend, got "
            f"{type(state).__name__} and {type(actions).__name__}"
        )
    if not isinstance(state, (np.ndarray, torch.Tensor)):
        raise TypeError(f"state must be numpy.ndarray or torch.Tensor, got {type(state).__name__}")
    if state.ndim == 0 or actions.ndim == 0:
        raise ValueError("state/actions must have at least one dimension")
    state_indices = _validate_pose_indices(
        state_pose_indices,
        feature_dim=state.shape[-1],
        name="state_pose_indices",
    )
    action_indices = _validate_pose_indices(
        action_pose_indices,
        feature_dim=actions.shape[-1],
        name="action_pose_indices",
    )
    state_pose = _take_indices(state, state_indices)
    action_pose = _take_indices(actions, action_indices)
    relative_pose = absolute_eef_to_relative(state_pose, action_pose)
    return _replace_indices(actions, action_indices, relative_pose)


def relative_actions_to_absolute(
    state: Array,
    relative_actions: Array,
    *,
    state_pose_indices: Sequence[int] = DEFAULT_EEF_POSE_INDICES,
    action_pose_indices: Sequence[int] = DEFAULT_EEF_POSE_INDICES,
) -> Array:
    """Full feature action에서 EEF 9D만 absolute로 복원하고 나머지는 passthrough."""
    same_backend = (
        isinstance(state, torch.Tensor)
        and isinstance(relative_actions, torch.Tensor)
        or isinstance(state, np.ndarray)
        and isinstance(relative_actions, np.ndarray)
    )
    if not same_backend:
        raise TypeError(
            "state and relative_actions must use the same numpy/torch backend, got "
            f"{type(state).__name__} and {type(relative_actions).__name__}"
        )
    if not isinstance(state, (np.ndarray, torch.Tensor)):
        raise TypeError(f"state must be numpy.ndarray or torch.Tensor, got {type(state).__name__}")
    if state.ndim == 0 or relative_actions.ndim == 0:
        raise ValueError("state/relative_actions must have at least one dimension")
    state_indices = _validate_pose_indices(
        state_pose_indices,
        feature_dim=state.shape[-1],
        name="state_pose_indices",
    )
    action_indices = _validate_pose_indices(
        action_pose_indices,
        feature_dim=relative_actions.shape[-1],
        name="action_pose_indices",
    )
    state_pose = _take_indices(state, state_indices)
    relative_pose = _take_indices(relative_actions, action_indices)
    absolute_pose = relative_eef_to_absolute(state_pose, relative_pose)
    return _replace_indices(relative_actions, action_indices, absolute_pose)
