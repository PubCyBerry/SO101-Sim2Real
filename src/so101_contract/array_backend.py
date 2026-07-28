"""NumPy/Torch 공통 배열 연산 shim.

pose codec(EEF)과 joint topology 변환이 같은 shape/dtype/device 계약을 쓰도록,
두 backend에서 이름이나 시그니처가 다른 연산만 얇게 감싼다. backend별 구현을 통째로
복제하지 않기 위한 최소 레이어이며 로봇/표현 관련 의미는 담지 않는다.

공통 chunk 계약:

- ``state``: ``(...,D)`` 또는 observation history ``(...,T_obs,D)``
- ``actions``: ``(...,H,D)``
- chunk의 모든 timestep은 :func:`resolve_chunk_reference`가 고른 **하나의** state를 공유한다.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeAlias

import numpy as np
import torch

Array: TypeAlias = np.ndarray | torch.Tensor


def is_torch(value: Any) -> bool:
    return isinstance(value, torch.Tensor)


def concat(arrays: Sequence[Array], axis: int = -1) -> Array:
    if is_torch(arrays[0]):
        return torch.cat(list(arrays), dim=axis)
    return np.concatenate(list(arrays), axis=axis)


def stack(arrays: Sequence[Array], axis: int = -1) -> Array:
    if is_torch(arrays[0]):
        return torch.stack(list(arrays), dim=axis)
    return np.stack(list(arrays), axis=axis)


def norm(values: Array, *, axis: int = -1, keepdims: bool = True) -> Array:
    if is_torch(values):
        return torch.linalg.vector_norm(values, dim=axis, keepdim=keepdims)
    return np.linalg.norm(values, axis=axis, keepdims=keepdims)


def cross(left: Array, right: Array) -> Array:
    if is_torch(left):
        return torch.linalg.cross(left, right, dim=-1)
    return np.cross(left, right)


def atan2(numerator: Array, denominator: Array) -> Array:
    if is_torch(numerator):
        return torch.atan2(numerator, denominator)
    return np.arctan2(numerator, denominator)


def asin(values: Array) -> Array:
    return torch.asin(values) if is_torch(values) else np.arcsin(values)


def acos(values: Array) -> Array:
    return torch.acos(values) if is_torch(values) else np.arccos(values)


def sqrt(values: Array) -> Array:
    return torch.sqrt(values) if is_torch(values) else np.sqrt(values)


def absolute(values: Array) -> Array:
    return torch.abs(values) if is_torch(values) else np.abs(values)


def cos(values: Array) -> Array:
    return torch.cos(values) if is_torch(values) else np.cos(values)


def sin(values: Array) -> Array:
    return torch.sin(values) if is_torch(values) else np.sin(values)


def clip(values: Array, low: float | None, high: float | None) -> Array:
    if is_torch(values):
        return torch.clamp(values, min=low, max=high)
    return np.clip(values, low, high)


def where(condition: Array, when_true: Array, when_false: Array) -> Array:
    if is_torch(condition):
        return torch.where(condition, when_true, when_false)
    return np.where(condition, when_true, when_false)


def swap_last_two(values: Array) -> Array:
    if is_torch(values):
        return values.transpose(-1, -2)
    return np.swapaxes(values, -1, -2)


def movedim(values: Array, source: int, destination: int) -> Array:
    if is_torch(values):
        return torch.movedim(values, source, destination)
    return np.moveaxis(values, source, destination)


def full_like(values: Array, fill_value: float) -> Array:
    if is_torch(values):
        return torch.full_like(values, fill_value)
    return np.full_like(values, fill_value)


def as_like(values: Array, data: Sequence[float]) -> Array:
    """상수 vector를 ``values``의 dtype/device로 올린다."""
    if is_torch(values):
        return torch.as_tensor(list(data), dtype=values.dtype, device=values.device)
    return np.asarray(list(data), dtype=values.dtype)


def epsilon(values: Array) -> float:
    if is_torch(values):
        return float(torch.finfo(values.dtype).eps)
    return float(np.finfo(values.dtype).eps)


def any_true(values: Array) -> bool:
    if is_torch(values):
        return bool(torch.any(values))
    return bool(np.any(values))


def check_array(values: Any, name: str, *, last_dim: int | None = None) -> Array:
    """floating dtype, 최소 rank, 마지막 차원, finite 값을 검사."""
    if not isinstance(values, (np.ndarray, torch.Tensor)):
        raise TypeError(
            f"{name} must be numpy.ndarray or torch.Tensor, got {type(values).__name__}"
        )
    if is_torch(values):
        if not torch.is_floating_point(values):
            raise TypeError(f"{name} must use a floating dtype, got {values.dtype}")
        finite = bool(torch.isfinite(values).all())
    else:
        if values.dtype.kind != "f":
            raise TypeError(f"{name} must use a floating dtype, got {values.dtype}")
        finite = bool(np.all(np.isfinite(values)))
    if values.ndim == 0:
        raise ValueError(f"{name} must have at least one dimension")
    if last_dim is not None and values.shape[-1] != last_dim:
        raise ValueError(f"{name} shape must end in {last_dim}, got {tuple(values.shape)}")
    if not finite:
        raise ValueError(f"{name} contains NaN or infinity")
    return values


def require_same_backend(left: Any, right: Any, *, left_name: str, right_name: str) -> None:
    if is_torch(left) != is_torch(right):
        raise TypeError(
            f"{left_name} and {right_name} must use the same numpy/torch backend, got "
            f"{type(left).__name__} and {type(right).__name__}"
        )


def resolve_chunk_reference(
    state: Array,
    actions: Array,
    feature_dim: int,
    *,
    state_name: str = "state",
    action_name: str = "actions",
) -> Array:
    """Chunk 전체가 공유할 단일 기준 state를 고른다.

    state rank가 action rank와 같으면 observation history로 보고 마지막 관측을 쓴다.
    """
    resolved_state = check_array(state, state_name, last_dim=feature_dim)
    resolved_actions = check_array(actions, action_name, last_dim=feature_dim)
    require_same_backend(
        resolved_state,
        resolved_actions,
        left_name=state_name,
        right_name=action_name,
    )
    if resolved_state.dtype != resolved_actions.dtype:
        raise TypeError(
            f"{state_name}/{action_name} dtype mismatch: "
            f"{resolved_state.dtype} != {resolved_actions.dtype}"
        )
    if is_torch(resolved_state) and resolved_state.device != resolved_actions.device:
        raise ValueError(
            f"{state_name}/{action_name} device mismatch: "
            f"{resolved_state.device} != {resolved_actions.device}"
        )
    if resolved_actions.ndim < 2:
        raise ValueError(
            f"{action_name} must contain a chunk dimension, got shape "
            f"{tuple(resolved_actions.shape)}"
        )

    if resolved_state.ndim == resolved_actions.ndim:
        if resolved_state.shape[-2] == 0:
            raise ValueError(f"{state_name} observation history is empty")
        reference = resolved_state[..., -1, :]
    elif resolved_state.ndim == resolved_actions.ndim - 1:
        reference = resolved_state
    else:
        raise ValueError(
            f"{state_name}/{action_name} rank mismatch: expected state rank equal to action "
            f"rank or one less, got {resolved_state.ndim} and {resolved_actions.ndim}"
        )
    if tuple(reference.shape[:-1]) != tuple(resolved_actions.shape[:-2]):
        raise ValueError(
            f"{state_name}/{action_name} batch shape mismatch: "
            f"{tuple(reference.shape[:-1])} != {tuple(resolved_actions.shape[:-2])}"
        )
    return reference


def validate_indices(
    indices: Sequence[int],
    *,
    feature_dim: int,
    expected_count: int | None,
    name: str,
) -> tuple[int, ...]:
    resolved = tuple(int(index) for index in indices)
    if expected_count is not None and len(resolved) != expected_count:
        raise ValueError(f"{name} must contain exactly {expected_count} indices, got {resolved}")
    if not resolved:
        raise ValueError(f"{name} must not be empty")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{name} contains duplicate indices: {resolved}")
    if any(index < 0 or index >= feature_dim for index in resolved):
        raise ValueError(f"{name} is outside feature dimension {feature_dim}: {resolved}")
    return resolved


def take_indices(values: Array, indices: tuple[int, ...]) -> Array:
    if is_torch(values):
        index = torch.tensor(indices, dtype=torch.long, device=values.device)
        return torch.index_select(values, dim=-1, index=index)
    return values[..., list(indices)]


def replace_indices(values: Array, indices: tuple[int, ...], replacement: Array) -> Array:
    """``indices`` 위치만 교체한 새 배열. 입력을 in-place 수정하지 않는다."""
    if is_torch(values):
        output = values.clone()
        index = torch.tensor(indices, dtype=torch.long, device=values.device)
        output.index_copy_(-1, index, replacement)
        return output
    output = values.copy()
    output[..., list(indices)] = replacement
    return output
