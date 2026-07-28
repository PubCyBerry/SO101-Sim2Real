"""SO-101 absolute EEF target용 bounded sequential inverse kinematics.

학습 dataset 변환과 online platform adapter가 같은
``base_link → tcp_grasp`` FK를 사용하도록 :mod:`eef_kinematics` 위에 구현한다.
외부 IK runtime에 의존하지 않는 NumPy damped least-squares solver라 Windows real
client와 Linux sim client에서 동일하게 실행할 수 있다.

SO-101 arm은 5 DoF이므로 임의의 SE(3) target은 풀리지 않는다. 따라서 solver는
joint clamp로 실패를 숨기지 않고 position/orientation residual과 명시적 success
플래그를 반환한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .eef_kinematics import (
    ARM_JOINT_ORDER,
    SO101EndEffectorKinematics,
    decode_rotation_representation,
)

EEF_IK_VERSION = "so101_bounded_dls_ik_v1"


def load_arm_joint_limits(urdf_path: str | Path) -> np.ndarray:
    """URDF에서 canonical arm 순서의 lower/upper radian limit을 읽는다."""
    root = ET.parse(Path(urdf_path)).getroot()
    limits_by_name: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        if name not in ARM_JOINT_ORDER:
            continue
        limit = joint.find("limit")
        if limit is None or limit.get("lower") is None or limit.get("upper") is None:
            raise ValueError(f"URDF joint {name!r} has no finite lower/upper limit")
        lower = float(limit.get("lower"))
        upper = float(limit.get("upper"))
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError(f"invalid URDF joint limit for {name!r}: {lower}, {upper}")
        limits_by_name[str(name)] = (lower, upper)
    missing = [name for name in ARM_JOINT_ORDER if name not in limits_by_name]
    if missing:
        raise KeyError(f"URDF arm joint limits missing {missing}")
    return np.asarray([limits_by_name[name] for name in ARM_JOINT_ORDER], dtype=np.float64)


def rotation_log_vector(rotation: np.ndarray) -> np.ndarray:
    """SO(3) rotation matrix의 principal logarithm을 axis-angle vector로 반환."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"rotation must be finite (3,3), got {matrix.shape}")
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    skew_vector = np.asarray(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=np.float64,
    )
    if angle < 1e-8:
        return 0.5 * skew_vector
    if np.pi - angle < 1e-5:
        # pi 부근은 skew 성분이 0으로 사라지므로 R+I의 가장 안정적인 열을 축으로 쓴다.
        symmetric = matrix + np.eye(3, dtype=np.float64)
        column = symmetric[:, int(np.argmax(np.linalg.norm(symmetric, axis=0)))]
        norm = float(np.linalg.norm(column))
        if norm < 1e-10:
            raise ValueError("cannot resolve rotation axis near pi")
        axis = column / norm
        dominant = int(np.argmax(np.abs(axis)))
        if skew_vector[dominant] * axis[dominant] < 0.0:
            axis *= -1.0
        return axis * angle
    return skew_vector * (0.5 * angle / np.sin(angle))


@dataclass(frozen=True)
class IKConfig:
    """Bounded DLS solver 수치/수렴 계약."""

    position_weight: float = 1.0
    orientation_weight: float = 0.15
    damping: float = 1e-3
    finite_difference_eps: float = 1e-5
    max_iterations: int = 80
    position_tolerance_m: float = 5e-4
    orientation_tolerance_rad: float = 1e-2
    max_iteration_step_rad: float = 0.20
    line_search_steps: int = 6

    def __post_init__(self) -> None:
        positive = {
            "position_weight": self.position_weight,
            "orientation_weight": self.orientation_weight,
            "damping": self.damping,
            "finite_difference_eps": self.finite_difference_eps,
            "position_tolerance_m": self.position_tolerance_m,
            "orientation_tolerance_rad": self.orientation_tolerance_rad,
            "max_iteration_step_rad": self.max_iteration_step_rad,
        }
        invalid = {name: value for name, value in positive.items() if value <= 0}
        if invalid:
            raise ValueError(f"IKConfig values must be positive: {invalid}")
        if self.max_iterations <= 0 or self.line_search_steps <= 0:
            raise ValueError("max_iterations and line_search_steps must be positive")


@dataclass(frozen=True)
class IKResult:
    success: bool
    joint_radians: np.ndarray
    position_residual_m: float
    orientation_residual_rad: float
    iterations: int
    reason: str


@dataclass(frozen=True)
class SequentialIKResult:
    success: bool
    joint_radians: np.ndarray | None
    steps: tuple[IKResult, ...]
    failed_index: int | None
    reason: str


class SO101BoundedIK:
    """Numerical Jacobian과 URDF bounds를 쓰는 SO-101 5축 DLS IK."""

    def __init__(
        self,
        kinematics: SO101EndEffectorKinematics,
        joint_limits_rad: np.ndarray,
        *,
        config: IKConfig | None = None,
    ) -> None:
        limits = np.asarray(joint_limits_rad, dtype=np.float64)
        if limits.shape != (5, 2) or not np.all(np.isfinite(limits)):
            raise ValueError(f"joint_limits_rad must be finite (5,2), got {limits.shape}")
        if np.any(limits[:, 0] >= limits[:, 1]):
            raise ValueError("every IK joint lower limit must be smaller than upper limit")
        self.kinematics = kinematics
        self.joint_limits_rad = limits
        self.config = config or IKConfig()

    @classmethod
    def from_files(
        cls,
        urdf_path: str | Path,
        robot_yaml_path: str | Path,
        *,
        config: IKConfig | None = None,
    ) -> "SO101BoundedIK":
        return cls(
            SO101EndEffectorKinematics.from_files(urdf_path, robot_yaml_path),
            load_arm_joint_limits(urdf_path),
            config=config,
        )

    @staticmethod
    def _target_matrix(target_pose: np.ndarray, representation: str) -> np.ndarray:
        pose = np.asarray(target_pose, dtype=np.float64)
        rotation_dim = {"rot6d": 6, "rpy": 3, "wxyz": 4}.get(representation)
        if rotation_dim is None:
            raise ValueError(f"unsupported rotation representation: {representation!r}")
        if pose.shape != (3 + rotation_dim,) or not np.all(np.isfinite(pose)):
            raise ValueError(
                f"target_pose must be finite ({3 + rotation_dim},), got {pose.shape}"
            )
        target = np.eye(4, dtype=np.float64)
        target[:3, 3] = pose[:3]
        target[:3, :3] = decode_rotation_representation(
            pose[3:],
            representation,
        )
        return target

    @staticmethod
    def _pose_residual(current: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float, float]:
        position = target[:3, 3] - current[:3, 3]
        orientation = rotation_log_vector(target[:3, :3] @ current[:3, :3].T)
        return position, float(np.linalg.norm(position)), float(np.linalg.norm(orientation))

    def _jacobian(self, joints: np.ndarray, current: np.ndarray) -> np.ndarray:
        eps = self.config.finite_difference_eps
        jacobian = np.empty((6, 5), dtype=np.float64)
        for index in range(5):
            perturbed = joints.copy()
            direction = 1.0
            if perturbed[index] + eps > self.joint_limits_rad[index, 1]:
                direction = -1.0
            perturbed[index] += direction * eps
            next_pose = self.kinematics.forward_matrices(perturbed)
            jacobian[:3, index] = (
                next_pose[:3, 3] - current[:3, 3]
            ) / (direction * eps)
            jacobian[3:, index] = rotation_log_vector(
                next_pose[:3, :3] @ current[:3, :3].T
            ) / (direction * eps)
        return jacobian

    def solve(
        self,
        target_pose: np.ndarray,
        seed_joint_radians: np.ndarray,
        *,
        representation: str = "rot6d",
        step_lower_rad: np.ndarray | None = None,
        step_upper_rad: np.ndarray | None = None,
    ) -> IKResult:
        """한 pose를 푼다. 주어진 step bounds도 URDF bounds와 교집합으로 적용한다."""
        target = self._target_matrix(target_pose, representation)
        joints = np.asarray(seed_joint_radians, dtype=np.float64)
        if joints.shape != (5,) or not np.all(np.isfinite(joints)):
            raise ValueError(f"seed_joint_radians must be finite (5,), got {joints.shape}")

        lower = self.joint_limits_rad[:, 0].copy()
        upper = self.joint_limits_rad[:, 1].copy()
        if step_lower_rad is not None:
            candidate = np.asarray(step_lower_rad, dtype=np.float64)
            if candidate.shape != (5,) or not np.all(np.isfinite(candidate)):
                raise ValueError("step_lower_rad must be finite (5,)")
            lower = np.maximum(lower, candidate)
        if step_upper_rad is not None:
            candidate = np.asarray(step_upper_rad, dtype=np.float64)
            if candidate.shape != (5,) or not np.all(np.isfinite(candidate)):
                raise ValueError("step_upper_rad must be finite (5,)")
            upper = np.minimum(upper, candidate)
        if np.any(lower > upper):
            return IKResult(
                False,
                np.clip(joints, self.joint_limits_rad[:, 0], self.joint_limits_rad[:, 1]),
                float("inf"),
                float("inf"),
                0,
                "empty_joint_bound_intersection",
            )
        joints = np.clip(joints, lower, upper)

        weights = np.asarray(
            [self.config.position_weight] * 3 + [self.config.orientation_weight] * 3,
            dtype=np.float64,
        )
        best_joints = joints.copy()
        best_cost = float("inf")
        best_position = float("inf")
        best_orientation = float("inf")
        reason = "max_iterations"

        for iteration in range(self.config.max_iterations + 1):
            current = self.kinematics.forward_matrices(joints)
            position_vector, position_norm, orientation_norm = self._pose_residual(
                current,
                target,
            )
            residual = np.concatenate(
                [
                    position_vector,
                    rotation_log_vector(target[:3, :3] @ current[:3, :3].T),
                ]
            )
            weighted_residual = residual * weights
            cost = float(np.linalg.norm(weighted_residual))
            if cost < best_cost:
                best_cost = cost
                best_joints = joints.copy()
                best_position = position_norm
                best_orientation = orientation_norm
            if (
                position_norm <= self.config.position_tolerance_m
                and orientation_norm <= self.config.orientation_tolerance_rad
            ):
                return IKResult(
                    True,
                    joints.astype(np.float32),
                    position_norm,
                    orientation_norm,
                    iteration,
                    "converged",
                )
            if iteration == self.config.max_iterations:
                break

            jacobian = self._jacobian(joints, current)
            weighted_jacobian = jacobian * weights[:, None]
            system = (
                weighted_jacobian.T @ weighted_jacobian
                + (self.config.damping**2) * np.eye(5, dtype=np.float64)
            )
            try:
                delta = np.linalg.solve(
                    system,
                    weighted_jacobian.T @ weighted_residual,
                )
            except np.linalg.LinAlgError:
                reason = "singular_system"
                break
            if not np.all(np.isfinite(delta)):
                reason = "non_finite_update"
                break
            max_delta = float(np.max(np.abs(delta)))
            if max_delta > self.config.max_iteration_step_rad:
                delta *= self.config.max_iteration_step_rad / max_delta

            accepted = False
            for line_search_index in range(self.config.line_search_steps):
                scale = 0.5**line_search_index
                candidate = np.clip(joints + scale * delta, lower, upper)
                candidate_pose = self.kinematics.forward_matrices(candidate)
                candidate_position, _, _ = self._pose_residual(candidate_pose, target)
                candidate_orientation = rotation_log_vector(
                    target[:3, :3] @ candidate_pose[:3, :3].T
                )
                candidate_cost = float(
                    np.linalg.norm(
                        np.concatenate([candidate_position, candidate_orientation]) * weights
                    )
                )
                if candidate_cost < cost - 1e-12:
                    joints = candidate
                    accepted = True
                    break
            if not accepted:
                reason = "line_search_stalled"
                break

        return IKResult(
            False,
            best_joints.astype(np.float32),
            best_position,
            best_orientation,
            min(self.config.max_iterations, iteration),
            reason,
        )

    def solve_chunk(
        self,
        target_poses: np.ndarray,
        seed_joint_radians: np.ndarray,
        *,
        representation: str = "rot6d",
        max_joint_step_rad: float | np.ndarray | None = None,
    ) -> SequentialIKResult:
        """Chunk를 순서대로 풀며 h번째 해를 h+1 seed로 사용한다."""
        poses = np.asarray(target_poses, dtype=np.float64)
        expected_pose_dim = {"rot6d": 9, "rpy": 6, "wxyz": 7}.get(representation)
        if expected_pose_dim is None:
            raise ValueError(f"unsupported rotation representation: {representation!r}")
        if poses.ndim != 2 or poses.shape[1] != expected_pose_dim:
            raise ValueError(
                f"target_poses must be (H,{expected_pose_dim}), got {poses.shape}"
            )
        if poses.shape[0] == 0 or not np.all(np.isfinite(poses)):
            raise ValueError("target_poses must be non-empty and finite")
        seed = np.asarray(seed_joint_radians, dtype=np.float64)
        if seed.shape != (5,) or not np.all(np.isfinite(seed)):
            raise ValueError("seed_joint_radians must be finite (5,)")

        max_step: np.ndarray | None = None
        if max_joint_step_rad is not None:
            max_step = np.broadcast_to(
                np.asarray(max_joint_step_rad, dtype=np.float64),
                (5,),
            ).copy()
            if not np.all(np.isfinite(max_step)) or np.any(max_step <= 0):
                raise ValueError("max_joint_step_rad must be finite and positive")

        solutions: list[np.ndarray] = []
        results: list[IKResult] = []
        previous = seed.copy()
        for index, pose in enumerate(poses):
            lower = None if max_step is None else previous - max_step
            upper = None if max_step is None else previous + max_step
            result = self.solve(
                pose,
                previous,
                representation=representation,
                step_lower_rad=lower,
                step_upper_rad=upper,
            )
            results.append(result)
            if not result.success:
                return SequentialIKResult(
                    False,
                    None,
                    tuple(results),
                    index,
                    f"step_{index}:{result.reason}",
                )
            previous = np.asarray(result.joint_radians, dtype=np.float64)
            solutions.append(previous.copy())

        return SequentialIKResult(
            True,
            np.asarray(solutions, dtype=np.float32),
            tuple(results),
            None,
            "converged",
        )
