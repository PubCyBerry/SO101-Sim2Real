#!/usr/bin/env python3
"""SO-101 EEF-relative SE(3) core의 Phase 1 계약 검증.

검증 범위:

- NumPy/Torch absolute→relative→absolute round-trip
- Rot6D row convention과 rigid-transform translation
- observation history의 마지막 state를 chunk 전체의 공통 기준으로 사용
- full 10D action에서 absolute gripper passthrough
- LeRobot v0.6.0 GR00T-N1.7 native EEF decoder와 수치 parity
- invalid shape/dtype/rotation fail-fast
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from so101_contract.eef_relative_action import (  # noqa: E402
    EEF_RELATIVE_ACTION_VERSION,
    absolute_actions_to_relative,
    absolute_eef_to_relative,
    matrix_to_rot6d_rows,
    relative_actions_to_absolute,
    relative_eef_to_absolute,
    rot6d_rows_to_matrix,
)
from so101_contract.eef_action_contract import (  # noqa: E402
    CANONICAL_ACTION_NAMES,
    ActionRepresentationConfig,
    resolve_eef_action_contract,
)


def _assert_close(
    actual,
    expected,
    *,
    message: str,
    atol: float,
) -> None:
    actual_array = (
        actual.detach().cpu().numpy() if isinstance(actual, torch.Tensor) else np.asarray(actual)
    )
    expected_array = (
        expected.detach().cpu().numpy()
        if isinstance(expected, torch.Tensor)
        else np.asarray(expected)
    )
    if not np.allclose(actual_array, expected_array, atol=atol, rtol=0.0):
        error = float(np.max(np.abs(actual_array - expected_array)))
        raise AssertionError(f"{message}: max_abs_error={error}")


def _random_rotations(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    matrices = rng.normal(size=(*shape, 3, 3))
    flat = matrices.reshape(-1, 3, 3)
    rotations = []
    for matrix in flat:
        q, _ = np.linalg.qr(matrix)
        if np.linalg.det(q) < 0.0:
            q[:, -1] *= -1.0
        rotations.append(q)
    return np.stack(rotations).reshape(*shape, 3, 3)


def _poses(
    translations: np.ndarray,
    rotations: np.ndarray,
    *,
    dtype: np.dtype,
) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(translations, dtype=dtype),
            np.asarray(matrix_to_rot6d_rows(rotations), dtype=dtype),
        ],
        axis=-1,
    )


def validate_known_transform() -> None:
    # state가 world z축으로 +90° 회전했을 때 local +x 10cm는 world +y 10cm다.
    state_rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    relative_rotation = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    state = _poses(np.asarray([1.0, 2.0, 3.0]), state_rotation, dtype=np.float64)
    expected_relative = _poses(
        np.asarray([[0.1, 0.0, 0.0]]),
        relative_rotation[None, ...],
        dtype=np.float64,
    )
    absolute_rotation = state_rotation @ relative_rotation
    absolute = _poses(
        np.asarray([[1.0, 2.1, 3.0]]),
        absolute_rotation[None, ...],
        dtype=np.float64,
    )

    actual_relative = absolute_eef_to_relative(state, absolute)
    _assert_close(
        actual_relative,
        expected_relative,
        message="known transform absolute→relative",
        atol=1e-12,
    )
    reconstructed = relative_eef_to_absolute(state, expected_relative)
    _assert_close(
        reconstructed,
        absolute,
        message="known transform relative→absolute",
        atol=1e-12,
    )


def validate_numpy_roundtrip() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260727)
    batch, history, horizon = 6, 3, 11
    state_rotations = _random_rotations(rng, (batch, history))
    action_rotations = _random_rotations(rng, (batch, horizon))
    state = _poses(
        rng.uniform(-0.5, 0.5, (batch, history, 3)),
        state_rotations,
        dtype=np.float64,
    )
    actions = _poses(
        rng.uniform(-0.5, 0.5, (batch, horizon, 3)),
        action_rotations,
        dtype=np.float64,
    )

    relative = absolute_eef_to_relative(state, actions)
    reconstructed = relative_eef_to_absolute(state, relative)
    _assert_close(
        reconstructed,
        actions,
        message="NumPy float64 round-trip",
        atol=1e-12,
    )

    # 첫 history가 아니라 마지막 history만 기준으로 쓴다는 계약을 고정한다.
    last_state_relative = absolute_eef_to_relative(state[:, -1], actions)
    _assert_close(
        relative,
        last_state_relative,
        message="last observation is shared chunk reference",
        atol=1e-12,
    )

    state32 = state.astype(np.float32)
    actions32 = actions.astype(np.float32)
    reconstructed32 = relative_eef_to_absolute(
        state32,
        absolute_eef_to_relative(state32, actions32),
    )
    _assert_close(
        reconstructed32,
        actions32,
        message="NumPy float32 round-trip",
        atol=3e-5,
    )
    return state, actions, relative


def validate_passthrough(state: np.ndarray, actions: np.ndarray) -> None:
    batch, horizon = actions.shape[:2]
    state_gripper = np.linspace(10.0, 60.0, batch, dtype=np.float64)[:, None, None]
    state_full = np.concatenate([state, np.broadcast_to(state_gripper, (*state.shape[:2], 1))], axis=-1)
    action_gripper = np.linspace(0.0, 100.0, batch * horizon, dtype=np.float64).reshape(
        batch,
        horizon,
        1,
    )
    actions_full = np.concatenate([actions, action_gripper], axis=-1)

    relative_full = absolute_actions_to_relative(state_full, actions_full)
    if not np.array_equal(relative_full[..., 9:], action_gripper):
        raise AssertionError("absolute gripper changed during absolute→relative")
    reconstructed = relative_actions_to_absolute(state_full, relative_full)
    _assert_close(
        reconstructed,
        actions_full,
        message="10D pose+gripper round-trip",
        atol=1e-12,
    )


def validate_torch_parity(
    state: np.ndarray,
    actions: np.ndarray,
    expected_relative: np.ndarray,
) -> None:
    state_tensor = torch.tensor(state, dtype=torch.float64)
    actions_tensor = torch.tensor(actions, dtype=torch.float64, requires_grad=True)
    relative_tensor = absolute_eef_to_relative(state_tensor, actions_tensor)
    _assert_close(
        relative_tensor,
        expected_relative,
        message="Torch/NumPy float64 parity",
        atol=1e-12,
    )
    reconstructed = relative_eef_to_absolute(state_tensor, relative_tensor)
    _assert_close(
        reconstructed,
        actions,
        message="Torch float64 round-trip",
        atol=1e-12,
    )

    reconstructed.square().mean().backward()
    if actions_tensor.grad is None or not bool(torch.isfinite(actions_tensor.grad).all()):
        raise AssertionError("Torch transform gradient is missing or non-finite")

    state32 = torch.tensor(state, dtype=torch.float32)
    actions32 = torch.tensor(actions, dtype=torch.float32)
    relative32 = absolute_eef_to_relative(state32, actions32)
    _assert_close(
        relative32,
        expected_relative.astype(np.float32),
        message="Torch/NumPy float32 parity",
        atol=3e-5,
    )

    if torch.cuda.is_available():
        state_cuda = state32.cuda()
        actions_cuda = actions32.cuda()
        relative_cuda = absolute_eef_to_relative(state_cuda, actions_cuda)
        _assert_close(
            relative_cuda,
            relative32,
            message="Torch CPU/CUDA float32 parity",
            atol=1e-5,
        )


def _load_lerobot_groot_utils():
    reference_root = Path(
        os.environ.get("LEROBOT_REFERENCE_REPO", REPO_ROOT / "ref_repos" / "lerobot")
    )
    source_path = (
        reference_root
        / "src"
        / "lerobot"
        / "policies"
        / "groot"
        / "utils.py"
    )
    if not source_path.is_file():
        raise FileNotFoundError(f"LeRobot GR00T reference source not found: {source_path}")
    spec = importlib.util.spec_from_file_location("_lerobot_v060_groot_utils", source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load LeRobot GR00T utils: {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_lerobot_groot_parity(
    state: np.ndarray,
    actions: np.ndarray,
    relative: np.ndarray,
) -> None:
    groot_utils = _load_lerobot_groot_utils()
    upstream = groot_utils.relative_eef_to_absolute(relative, state[:, -1])
    _assert_close(
        upstream,
        actions.astype(np.float32),
        message="LeRobot v0.6.0 GR00T native relative decoder parity",
        atol=3e-5,
    )


def validate_fail_fast() -> None:
    identity_pose = np.asarray(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        dtype=np.float32,
    )
    actions = np.broadcast_to(identity_pose, (2, 3, 9)).copy()
    state = np.broadcast_to(identity_pose, (2, 9)).copy()

    cases = [
        (
            "invalid pose dim",
            lambda: absolute_eef_to_relative(state, actions[..., :8]),
            ValueError,
        ),
        (
            "integer dtype",
            lambda: absolute_eef_to_relative(state.astype(np.int64), actions),
            TypeError,
        ),
        (
            "backend mismatch",
            lambda: absolute_eef_to_relative(torch.from_numpy(state), actions),
            TypeError,
        ),
        (
            "parallel rot6d rows",
            lambda: rot6d_rows_to_matrix(
                np.asarray([1.0, 0.0, 0.0, 2.0, 0.0, 0.0], dtype=np.float32)
            ),
            ValueError,
        ),
        (
            "batch mismatch",
            lambda: absolute_eef_to_relative(state[:1], actions),
            ValueError,
        ),
    ]
    for name, function, expected_error in cases:
        try:
            function()
        except expected_error:
            continue
        raise AssertionError(f"{name} did not raise {expected_error.__name__}")


def _write_contract_fixture(root: Path) -> None:
    (root / "meta").mkdir(parents=True)
    feature = {
        "dtype": "float32",
        "shape": [10],
        "names": list(CANONICAL_ACTION_NAMES),
    }
    info = {
        "codebase_version": "v3.0",
        "features": {
            "observation.state": feature,
            "action": feature,
        },
        "so101_eef_conversion": {
            "version": "so101_lerobot_abs_joint_to_abs_eef_v2",
            "source_domain": "sim",
            "base_frame": "base_link",
            "eef_frame": "tcp_grasp",
            "eef_kinematics_version": "so101_base_tcp_grasp_fk_v2",
            "rotation_representation": "rot6d",
            "rotation_format": "xyz+rot6d_rows",
            "gripper_format": "canonical_policy_feature_[0,100]",
            "keep_joints": False,
            "urdf_sha256": "a" * 64,
            "robot_yaml_sha256": "b" * 64,
        },
    }
    modality = {
        "state": {
            "eef_9d": {"start": 0, "end": 9},
            "gripper_position": {"start": 9, "end": 10},
        },
        "action": {
            "eef_9d": {"start": 0, "end": 9},
            "gripper_position": {"start": 9, "end": 10},
        },
    }
    (root / "meta" / "info.json").write_text(
        json.dumps(info, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "meta" / "modality.json").write_text(
        json.dumps(modality, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_metadata_contract() -> None:
    scratch = REPO_ROOT / "scratch"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="eef-contract-", dir=scratch) as directory:
        root = Path(directory)
        _write_contract_fixture(root)
        config = ActionRepresentationConfig(mode="eef_relative")
        resolved = resolve_eef_action_contract(root, config)
        if resolved.state_pose_indices != tuple(range(9)):
            raise AssertionError(f"state pose indices mismatch: {resolved.state_pose_indices}")
        if resolved.action_pose_indices != tuple(range(9)):
            raise AssertionError(f"action pose indices mismatch: {resolved.action_pose_indices}")
        if resolved.passthrough_action_indices != (9,):
            raise AssertionError(
                f"passthrough action indices mismatch: {resolved.passthrough_action_indices}"
            )
        if len(resolved.fingerprint) != 64:
            raise AssertionError(f"contract fingerprint is not SHA-256: {resolved.fingerprint}")
        if resolve_eef_action_contract(root, config).fingerprint != resolved.fingerprint:
            raise AssertionError("contract fingerprint is not deterministic")

        info_path = root / "meta" / "info.json"
        original_info = json.loads(info_path.read_text(encoding="utf-8"))
        invalid_cases = [
            (
                "RPY dataset",
                ("so101_eef_conversion", "rotation_representation"),
                "rpy",
            ),
            (
                "keep-joints dataset",
                ("so101_eef_conversion", "keep_joints"),
                True,
            ),
            (
                "wrong feature names",
                ("features", "action", "names"),
                ["wrong"] * 10,
            ),
            (
                "missing kinematics hash",
                ("so101_eef_conversion", "urdf_sha256"),
                "",
            ),
        ]
        for name, key_path, invalid_value in invalid_cases:
            info = json.loads(json.dumps(original_info))
            target = info
            for key in key_path[:-1]:
                target = target[key]
            target[key_path[-1]] = invalid_value
            info_path.write_text(json.dumps(info) + "\n", encoding="utf-8")
            try:
                resolve_eef_action_contract(root, config)
            except (KeyError, TypeError, ValueError):
                continue
            raise AssertionError(f"{name} metadata was accepted")


def main() -> None:
    validate_known_transform()
    state, actions, relative = validate_numpy_roundtrip()
    validate_passthrough(state, actions)
    validate_torch_parity(state, actions, relative)
    validate_lerobot_groot_parity(state, actions, relative)
    validate_fail_fast()
    validate_metadata_contract()
    print(f"PASS: SO-101 EEF-relative contract ({EEF_RELATIVE_ACTION_VERSION})")


if __name__ == "__main__":
    main()
