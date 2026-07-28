#!/usr/bin/env python3
"""Phase 12 — format-neutral pose codec 검증.

세 pose format(``xyz_rot6d_rows``·``xyz_quaternion_wxyz``·``xyz_rpy``)이 같은 SE(3)
수학을 쓰는지, 그리고 각 format 고유의 함정(quaternion 부호, RPY gimbal singularity와
angle wrap, float32/64 정밀도)을 실제로 처리하는지 offline으로 확인한다.

.. code-block:: bash

    python scripts/contract/validate_pose_codec.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from so101_contract.action_representation import (  # noqa: E402
    EEF_POSE_FORMATS,
    PoseFormat,
    pose_format_dims,
)
from so101_contract.pose_codec import (  # noqa: E402
    absolute_actions_to_relative,
    absolute_pose_to_relative,
    canonicalize_pose,
    canonicalize_quaternion_sequence,
    canonicalize_quaternion_wxyz,
    convert_pose_format,
    decode_pose,
    encode_pose,
    relative_actions_to_absolute,
    relative_pose_to_absolute,
    rotation_geodesic_angle,
    wrap_angles_to_pi,
)

# 문서 §17 권장 허용치.
TOLERANCE = {
    np.float64: 1e-9,
    np.float32: 3e-5,
}


def _random_rotations(count: int, seed: int) -> np.ndarray:
    """QR 분해로 균일한 SO(3) 표본을 만든다."""
    generator = np.random.default_rng(seed)
    matrices = np.empty((count, 3, 3), dtype=np.float64)
    for index in range(count):
        q, r = np.linalg.qr(generator.normal(size=(3, 3)))
        q = q * np.sign(np.diag(r))
        if np.linalg.det(q) < 0.0:
            q[:, 0] *= -1.0
        matrices[index] = q
    return matrices


def _rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    pose = np.asarray([0.0, 0.0, 0.0, roll, pitch, yaw], dtype=np.float64)
    return decode_pose(pose, PoseFormat.XYZ_RPY)[1]


def _max_rotation_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def check_encode_decode_round_trip() -> None:
    """모든 format에서 rotation matrix ↔ pose vector round-trip."""
    rotations = _random_rotations(256, seed=11)
    translations = np.random.default_rng(12).normal(scale=0.4, size=(256, 3))
    for pose_format in EEF_POSE_FORMATS:
        rotation_dim, pose_dim = pose_format_dims(pose_format)
        for dtype in (np.float64, np.float32):
            pose = encode_pose(
                translations.astype(dtype),
                rotations.astype(dtype),
                pose_format,
            )
            if pose.shape != (256, pose_dim) or pose.dtype != dtype:
                raise AssertionError(
                    f"{pose_format.value} encode shape/dtype mismatch: {pose.shape}/{pose.dtype}"
                )
            decoded_translation, decoded_rotation = decode_pose(pose, pose_format)
            tolerance = TOLERANCE[dtype]
            error = max(
                _max_rotation_error(decoded_rotation, rotations),
                _max_rotation_error(decoded_translation, translations),
            )
            if error > tolerance:
                raise AssertionError(
                    f"{pose_format.value}/{np.dtype(dtype).name} round-trip error "
                    f"{error:.3e} > {tolerance:.1e}"
                )
            if decoded_rotation.shape[-1] != 3 or pose.shape[-1] - 3 != rotation_dim:
                raise AssertionError(f"{pose_format.value} rotation dim mismatch")
    print("PASS: pose encode/decode round-trip (rot6d·wxyz·rpy, float32/float64)")


def check_cross_format_equivalence() -> None:
    """같은 pose를 서로 다른 format으로 옮겨도 SE(3)가 보존된다."""
    rotations = _random_rotations(128, seed=13)
    translations = np.random.default_rng(14).normal(scale=0.3, size=(128, 3))
    reference = encode_pose(translations, rotations, PoseFormat.XYZ_ROT6D_ROWS)
    for target in EEF_POSE_FORMATS:
        converted = convert_pose_format(reference, PoseFormat.XYZ_ROT6D_ROWS, target)
        back = convert_pose_format(converted, target, PoseFormat.XYZ_ROT6D_ROWS)
        error = _max_rotation_error(back, reference)
        if error > TOLERANCE[np.float64]:
            raise AssertionError(
                f"rot6d→{target.value}→rot6d conversion error {error:.3e}"
            )
    print("PASS: cross-format conversion preserves SE(3)")


def check_quaternion_sign_canonicalization() -> None:
    """``q``와 ``-q``가 같은 canonical quaternion으로 수렴하고 결정적이다."""
    rotations = _random_rotations(256, seed=15)
    quaternion = encode_pose(
        np.zeros((256, 3)),
        rotations,
        PoseFormat.XYZ_QUATERNION_WXYZ,
    )[:, 3:]
    if float(np.min(quaternion[:, 0])) < 0.0:
        raise AssertionError("canonical quaternion must keep w >= 0")

    flipped = canonicalize_quaternion_wxyz(-quaternion)
    if _max_rotation_error(flipped, quaternion) > TOLERANCE[np.float64]:
        raise AssertionError("canonicalization is not sign-invariant")

    scaled = canonicalize_quaternion_wxyz(quaternion * 7.5)
    if _max_rotation_error(scaled, quaternion) > TOLERANCE[np.float64]:
        raise AssertionError("canonicalization does not normalize to unit norm")

    # w ≈ 0 (180° 회전) 에서도 첫 유의미 성분 부호 규칙으로 결정적이어야 한다.
    boundary = np.asarray(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    canonical = canonicalize_quaternion_wxyz(boundary)
    expected = np.asarray(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    if _max_rotation_error(canonical, expected) > TOLERANCE[np.float64]:
        raise AssertionError(f"w≈0 quaternion sign rule is not deterministic: {canonical}")

    try:
        canonicalize_quaternion_wxyz(np.zeros((2, 4)))
    except ValueError:
        pass
    else:
        raise AssertionError("zero-norm quaternion must be rejected")
    print("PASS: quaternion unit norm and deterministic sign canonicalization")


def check_quaternion_sequence_continuity() -> None:
    """Chunk 시간축에서 ±q 점프가 제거된다."""
    angles = np.linspace(0.0, 2.4 * math.pi, 64)
    axis = np.asarray([0.3, -0.5, 0.81], dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    quaternions = np.stack(
        [
            np.concatenate(
                [[math.cos(angle / 2.0)], math.sin(angle / 2.0) * axis],
            )
            for angle in angles
        ]
    )
    canonical = canonicalize_quaternion_wxyz(quaternions)
    jumps = np.sum(np.sum(canonical[1:] * canonical[:-1], axis=-1) < 0.0)
    if jumps == 0:
        raise AssertionError("fixture does not cross the sign boundary; test is vacuous")

    continuous = canonicalize_quaternion_sequence(canonical, axis=-2)
    if np.any(np.sum(continuous[1:] * continuous[:-1], axis=-1) < 0.0):
        raise AssertionError("quaternion sequence still contains sign discontinuities")

    original = decode_pose(
        np.concatenate([np.zeros((64, 3)), quaternions], axis=-1),
        PoseFormat.XYZ_QUATERNION_WXYZ,
    )[1]
    aligned = decode_pose(
        np.concatenate([np.zeros((64, 3)), continuous], axis=-1),
        PoseFormat.XYZ_QUATERNION_WXYZ,
    )[1]
    if _max_rotation_error(aligned, original) > TOLERANCE[np.float64]:
        raise AssertionError("sequence alignment changed the underlying rotation")

    batched = canonicalize_quaternion_sequence(
        np.broadcast_to(canonical, (3, 64, 4)).copy(),
        axis=-2,
    )
    if batched.shape != (3, 64, 4):
        raise AssertionError(f"batched sequence helper shape mismatch: {batched.shape}")
    print(f"PASS: quaternion sequence continuity (removed {int(jumps)} sign flips)")


def check_rpy_wrap_and_gimbal() -> None:
    """RPY wrap 규칙과 gimbal singularity."""
    wrapped = wrap_angles_to_pi(
        np.asarray([3.0 * math.pi, -3.0 * math.pi, 0.5, -math.pi], dtype=np.float64)
    )
    if float(np.max(wrapped)) >= math.pi or float(np.min(wrapped)) < -math.pi:
        raise AssertionError(f"wrap_angles_to_pi left values outside [-pi, pi): {wrapped}")

    # yaw = 3π 와 yaw = π 는 같은 회전이고, encode 결과는 wrap된 값이다.
    unwrapped = np.asarray([0.1, -0.2, 0.3, 0.4, 0.2, 3.0 * math.pi], dtype=np.float64)
    canonical = canonicalize_pose(unwrapped, PoseFormat.XYZ_RPY)
    if float(np.max(np.abs(canonical[3:]))) > math.pi:
        raise AssertionError(f"canonical RPY is not wrapped: {canonical}")
    if _max_rotation_error(
        decode_pose(canonical, PoseFormat.XYZ_RPY)[1],
        decode_pose(unwrapped, PoseFormat.XYZ_RPY)[1],
    ) > TOLERANCE[np.float64]:
        raise AssertionError("RPY wrap changed the rotation")

    # gimbal lock: |pitch| = π/2 에서 roll/yaw가 축퇴해도 matrix round-trip은 유지된다.
    singular_cases = [
        (0.0, math.pi / 2.0, 0.0),
        (0.7, math.pi / 2.0, -1.1),
        (-0.4, -math.pi / 2.0, 2.0),
        (0.3, math.pi / 2.0 - 1e-9, 0.9),
        (0.3, -math.pi / 2.0 + 1e-9, -0.9),
    ]
    worst = 0.0
    for roll, pitch, yaw in singular_cases:
        rotation = _rotation_from_rpy(roll, pitch, yaw)
        encoded = encode_pose(np.zeros(3), rotation, PoseFormat.XYZ_RPY)
        decoded = decode_pose(encoded, PoseFormat.XYZ_RPY)[1]
        error = _max_rotation_error(decoded, rotation)
        worst = max(worst, error)
        if error > 1e-7:
            raise AssertionError(
                f"gimbal case rpy=({roll},{pitch},{yaw}) round-trip error {error:.3e}"
            )
        if abs(float(encoded[3])) > 1e-6 and abs(abs(pitch) - math.pi / 2.0) < 1e-12:
            raise AssertionError("exact gimbal case must collapse roll to zero")
    print(f"PASS: RPY wrap and gimbal singularity (max matrix error {worst:.3e})")


def check_relative_round_trip() -> None:
    """모든 format에서 absolute↔relative가 같은 SE(3)를 왕복한다."""
    generator = np.random.default_rng(21)
    state_rotation = _random_rotations(4, seed=22)
    action_rotation = _random_rotations(4 * 8, seed=23).reshape(4, 8, 3, 3)
    state_translation = generator.normal(scale=0.25, size=(4, 3))
    action_translation = generator.normal(scale=0.25, size=(4, 8, 3))

    for pose_format in EEF_POSE_FORMATS:
        for dtype in (np.float64, np.float32):
            state = encode_pose(
                state_translation.astype(dtype),
                state_rotation.astype(dtype),
                pose_format,
            )
            actions = encode_pose(
                action_translation.astype(dtype),
                action_rotation.astype(dtype),
                pose_format,
            )
            relative = absolute_pose_to_relative(state, actions, pose_format)
            restored = relative_pose_to_absolute(state, relative, pose_format)
            error = _max_rotation_error(
                decode_pose(restored, pose_format)[1],
                decode_pose(actions, pose_format)[1],
            )
            translation_error = _max_rotation_error(restored[..., :3], actions[..., :3])
            tolerance = TOLERANCE[dtype]
            if max(error, translation_error) > tolerance:
                raise AssertionError(
                    f"{pose_format.value}/{np.dtype(dtype).name} relative round-trip error "
                    f"{max(error, translation_error):.3e} > {tolerance:.1e}"
                )

            # identity state 기준에서는 relative == absolute 여야 한다.
            identity_state = encode_pose(
                np.zeros((4, 3), dtype=dtype),
                np.broadcast_to(np.eye(3, dtype=dtype), (4, 3, 3)).copy(),
                pose_format,
            )
            identity_relative = absolute_pose_to_relative(identity_state, actions, pose_format)
            identity_error = _max_rotation_error(
                decode_pose(identity_relative, pose_format)[1],
                decode_pose(actions, pose_format)[1],
            )
            if identity_error > tolerance:
                raise AssertionError(
                    f"{pose_format.value} identity-state relative mismatch {identity_error:.3e}"
                )
    print("PASS: absolute↔relative SE(3) round-trip in every pose format")


def check_shared_reference_and_rotation_math() -> None:
    """모든 horizon이 같은 state를 기준으로 하고, translation이 회전을 반영한다."""
    rotation = _rotation_from_rpy(0.0, 0.0, math.pi / 2.0)
    state = encode_pose(
        np.asarray([1.0, 2.0, 3.0]),
        rotation,
        PoseFormat.XYZ_ROT6D_ROWS,
    )
    actions = encode_pose(
        np.asarray([[1.0, 2.5, 3.0], [1.0, 3.0, 3.0]]),
        np.broadcast_to(rotation, (2, 3, 3)).copy(),
        PoseFormat.XYZ_ROT6D_ROWS,
    )
    relative = absolute_pose_to_relative(state, actions, PoseFormat.XYZ_ROT6D_ROWS)
    # world +y 이동은 state가 z로 90° 돌아 있으므로 body -x 가 아니라 body +x 여야 한다.
    expected = np.asarray([[0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
    if _max_rotation_error(relative[:, :3], expected) > 1e-9:
        raise AssertionError(
            f"relative translation ignores state rotation: {relative[:, :3]}"
        )
    if float(np.max(np.abs(relative[:, 3:] - np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])))) > 1e-9:
        raise AssertionError("same-rotation actions must yield identity relative rotation")

    # observation history가 있으면 마지막 state만 기준이 된다.
    history = np.stack([encode_pose(
        np.asarray([9.0, 9.0, 9.0]),
        np.eye(3),
        PoseFormat.XYZ_ROT6D_ROWS,
    ), state])
    from_history = absolute_pose_to_relative(history, actions, PoseFormat.XYZ_ROT6D_ROWS)
    if _max_rotation_error(from_history, relative) > 1e-12:
        raise AssertionError("observation history did not use the last state as reference")

    geodesic = rotation_geodesic_angle(
        np.eye(3),
        _rotation_from_rpy(0.0, 0.0, math.pi / 3.0),
    )
    if abs(float(geodesic) - math.pi / 3.0) > 1e-9:
        raise AssertionError(f"rotation_geodesic_angle is wrong: {geodesic}")
    print("PASS: single chunk reference, rotated translation, geodesic metric")


def check_passthrough_and_indices() -> None:
    """Full feature vector에서 pose group만 변환되고 gripper는 그대로 통과한다."""
    generator = np.random.default_rng(31)
    for pose_format in EEF_POSE_FORMATS:
        _, pose_dim = pose_format_dims(pose_format)
        rotation = _random_rotations(1, seed=32)[0]
        state = np.concatenate(
            [encode_pose(np.zeros(3), rotation, pose_format), [42.0]]
        )
        actions = np.concatenate(
            [
                encode_pose(
                    generator.normal(scale=0.1, size=(5, 3)),
                    _random_rotations(5, seed=33),
                    pose_format,
                ),
                generator.uniform(0.0, 100.0, size=(5, 1)),
            ],
            axis=-1,
        )
        relative = absolute_actions_to_relative(state, actions, pose_format)
        if not np.array_equal(relative[:, pose_dim:], actions[:, pose_dim:]):
            raise AssertionError(f"{pose_format.value} gripper passthrough was modified")
        restored = relative_actions_to_absolute(state, relative, pose_format)
        if _max_rotation_error(restored, actions) > TOLERANCE[np.float64]:
            raise AssertionError(f"{pose_format.value} full-feature round-trip failed")
        if actions.shape[-1] != pose_dim + 1:
            raise AssertionError("fixture dimension mismatch")

        try:
            absolute_actions_to_relative(
                state,
                actions,
                pose_format,
                action_pose_indices=tuple(range(pose_dim - 1)),
            )
        except ValueError:
            pass
        else:
            raise AssertionError("wrong pose index count must be rejected")
    print("PASS: gripper passthrough and pose index validation")


def check_torch_parity() -> None:
    """Torch와 NumPy 결과가 같고 dtype/device가 보존된다."""
    state_rotation = _random_rotations(2, seed=41)
    action_rotation = _random_rotations(2 * 6, seed=42).reshape(2, 6, 3, 3)
    state_translation = np.random.default_rng(43).normal(size=(2, 3))
    action_translation = np.random.default_rng(44).normal(size=(2, 6, 3))

    for pose_format in EEF_POSE_FORMATS:
        state = encode_pose(state_translation, state_rotation, pose_format)
        actions = encode_pose(action_translation, action_rotation, pose_format)
        numpy_relative = absolute_pose_to_relative(state, actions, pose_format)

        torch_relative = absolute_pose_to_relative(
            torch.from_numpy(state),
            torch.from_numpy(actions),
            pose_format,
        )
        if torch_relative.dtype is not torch.float64:
            raise AssertionError("torch path did not preserve float64")
        error = _max_rotation_error(torch_relative.numpy(), numpy_relative)
        if error > TOLERANCE[np.float64]:
            raise AssertionError(f"{pose_format.value} torch/numpy mismatch {error:.3e}")

        float32_relative = absolute_pose_to_relative(
            torch.from_numpy(state).float(),
            torch.from_numpy(actions).float(),
            pose_format,
        )
        if float32_relative.dtype is not torch.float32:
            raise AssertionError("torch float32 dtype was not preserved")
        error32 = _max_rotation_error(float32_relative.numpy(), numpy_relative)
        if error32 > TOLERANCE[np.float32]:
            raise AssertionError(
                f"{pose_format.value} torch float32 parity error {error32:.3e}"
            )

        restored = relative_pose_to_absolute(
            torch.from_numpy(state),
            torch_relative,
            pose_format,
        )
        if _max_rotation_error(restored.numpy(), actions) > TOLERANCE[np.float64]:
            raise AssertionError(f"{pose_format.value} torch round-trip failed")

    sequence = canonicalize_quaternion_sequence(
        torch.from_numpy(
            encode_pose(np.zeros((7, 3)), _random_rotations(7, seed=45), PoseFormat.XYZ_QUATERNION_WXYZ)[:, 3:]
        ),
        axis=-2,
    )
    if sequence.shape != (7, 4):
        raise AssertionError(f"torch sequence helper shape mismatch: {tuple(sequence.shape)}")
    print("PASS: torch/numpy parity, dtype preservation, float32 tolerance")


def check_invalid_inputs() -> None:
    """계약 위반은 추정 보정 없이 실패한다."""
    cases = [
        (
            "backend mismatch",
            lambda: absolute_pose_to_relative(
                np.zeros((1, 9)),
                torch.zeros(1, 2, 9),
                PoseFormat.XYZ_ROT6D_ROWS,
            ),
        ),
        (
            "wrong pose dim",
            lambda: decode_pose(np.zeros((2, 8)), PoseFormat.XYZ_ROT6D_ROWS),
        ),
        (
            "joint mode pose format",
            lambda: decode_pose(np.zeros((2, 9)), PoseFormat.NOT_APPLICABLE),
        ),
        (
            "non-finite pose",
            lambda: canonicalize_pose(
                np.asarray([0.0, 0.0, 0.0, np.nan, 0.0, 0.0]),
                PoseFormat.XYZ_RPY,
            ),
        ),
        (
            "missing chunk dim",
            lambda: absolute_pose_to_relative(
                np.zeros(6),
                np.zeros(6),
                PoseFormat.XYZ_RPY,
            ),
        ),
        (
            "degenerate rot6d",
            lambda: decode_pose(
                np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0]),
                PoseFormat.XYZ_ROT6D_ROWS,
            ),
        ),
        (
            "integer dtype",
            lambda: canonicalize_pose(np.zeros((2, 6), dtype=np.int64), PoseFormat.XYZ_RPY),
        ),
    ]
    for label, call in cases:
        try:
            call()
        except (TypeError, ValueError):
            continue
        raise AssertionError(f"invalid input was accepted: {label}")
    print(f"PASS: {len(cases)} invalid pose codec inputs rejected")


CHECKS = (
    check_encode_decode_round_trip,
    check_cross_format_equivalence,
    check_quaternion_sign_canonicalization,
    check_quaternion_sequence_continuity,
    check_rpy_wrap_and_gimbal,
    check_relative_round_trip,
    check_shared_reference_and_rotation_math,
    check_passthrough_and_indices,
    check_torch_parity,
    check_invalid_inputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for check in CHECKS:
        check()
    print(f"PASS: pose codec contract ({len(CHECKS)} checks)")


if __name__ == "__main__":
    main()
