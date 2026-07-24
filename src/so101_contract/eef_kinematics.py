"""SO-101 ``base_link → tcp_grasp`` forward kinematics contract.

GR00T-N1.7 EEF 데이터셋 변환과 online observation 생성이 같은 구현을 쓰도록,
URDF 관절 체인과 cuRobo robot YAML의 ``tcp_grasp`` fixed transform을 읽어 순수
NumPy FK를 제공한다. Isaac Lab, cuRobo, Pinocchio 의존성은 없다.

좌표 계약:

- 입력: SO101 arm 5축 joint angle, radian, ``SO101_JOINT_ORDER[:5]`` 순서
- 출력: URDF ``base_link`` 기준 ``tcp_grasp`` absolute pose
- 회전 표현:
  - ``rot6d``: GR00T-N1.7 구현과 같은 rotation matrix 첫 두 **행(row)** flatten
  - ``rpy``: URDF fixed-axis roll/pitch/yaw, radian, ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``
  - ``wxyz``: unit quaternion, scalar-first, canonical hemisphere(``w >= 0``)

Real/sim joint feature를 radian으로 바꾸는 calibration은 이 모듈 밖에서 수행한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .feature_codec import SO101_JOINT_ORDER

ARM_JOINT_ORDER = SO101_JOINT_ORDER[:5]
EEF_KINEMATICS_VERSION = "so101_base_tcp_grasp_fk_v2"
ROTATION_REPRESENTATIONS = ("rot6d", "rpy", "wxyz")
ROTATION_REPRESENTATION_DIMS = {
    "rot6d": 6,
    "rpy": 3,
    "wxyz": 4,
}


def _parse_vector(text: str | None, length: int, *, default: tuple[float, ...]) -> np.ndarray:
    values = default if text is None else tuple(float(v) for v in text.split())
    if len(values) != length:
        raise ValueError(f"expected {length} values, got {values}")
    return np.asarray(values, dtype=np.float64)


def _rotation_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rotation_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _homogeneous(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _origin_transform(origin: ET.Element | None) -> np.ndarray:
    if origin is None:
        return np.eye(4, dtype=np.float64)
    xyz = _parse_vector(origin.get("xyz"), 3, default=(0.0, 0.0, 0.0))
    roll, pitch, yaw = _parse_vector(origin.get("rpy"), 3, default=(0.0, 0.0, 0.0))
    # URDF rpy = fixed-axis roll/pitch/yaw, 즉 Rz(yaw) @ Ry(pitch) @ Rx(roll).
    rotation = _rotation_z(yaw) @ _rotation_y(pitch) @ _rotation_x(roll)
    return _homogeneous(rotation, xyz)


def _quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm([qw, qx, qy, qz]))
    if norm < 1e-12:
        raise ValueError("tcp_grasp quaternion norm is zero")
    qw, qx, qy, qz = (np.asarray([qw, qx, qy, qz]) / norm).tolist()
    return np.asarray(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix → canonical unit quaternion ``[w,x,y,z]``."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = np.sqrt(max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])) * 2.0
        qw = (matrix[2, 1] - matrix[1, 2]) / scale
        qx = 0.25 * scale
        qy = (matrix[0, 1] + matrix[1, 0]) / scale
        qz = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = np.sqrt(max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])) * 2.0
        qw = (matrix[0, 2] - matrix[2, 0]) / scale
        qx = (matrix[0, 1] + matrix[1, 0]) / scale
        qy = 0.25 * scale
        qz = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = np.sqrt(max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])) * 2.0
        qw = (matrix[1, 0] - matrix[0, 1]) / scale
        qx = (matrix[0, 2] + matrix[2, 0]) / scale
        qy = (matrix[1, 2] + matrix[2, 1]) / scale
        qz = 0.25 * scale

    quaternion = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise ValueError("cannot encode rotation matrix as quaternion")
    quaternion /= norm

    # q와 -q가 같은 회전을 뜻하므로 통계/학습의 불연속을 줄이도록 hemisphere를 고정한다.
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    elif abs(quaternion[0]) < 1e-12:
        first_nonzero = next((value for value in quaternion[1:] if abs(value) >= 1e-12), 0.0)
        if first_nonzero < 0.0:
            quaternion *= -1.0
    return quaternion


def encode_rotation_matrices(rotation_matrices: np.ndarray, representation: str) -> np.ndarray:
    """Rotation matrix ``(...,3,3)``를 선택한 연속 벡터 표현으로 변환."""
    rotations = np.asarray(rotation_matrices, dtype=np.float64)
    if rotations.shape[-2:] != (3, 3):
        raise ValueError(f"rotation_matrices shape must end in (3,3), got {rotations.shape}")
    if not np.all(np.isfinite(rotations)):
        raise ValueError("rotation_matrices contains NaN or infinity")
    if representation not in ROTATION_REPRESENTATION_DIMS:
        raise ValueError(
            f"unknown rotation representation {representation!r}; "
            f"expected one of {ROTATION_REPRESENTATIONS}"
        )

    prefix = rotations.shape[:-2]
    if representation == "rot6d":
        return rotations[..., :2, :].reshape(*prefix, 6).astype(np.float32)

    if representation == "rpy":
        pitch = np.arcsin(np.clip(-rotations[..., 2, 0], -1.0, 1.0))
        regular = np.abs(np.cos(pitch)) > 1e-7
        roll = np.where(
            regular,
            np.arctan2(rotations[..., 2, 1], rotations[..., 2, 2]),
            0.0,
        )
        yaw = np.where(
            regular,
            np.arctan2(rotations[..., 1, 0], rotations[..., 0, 0]),
            np.arctan2(-rotations[..., 0, 1], rotations[..., 1, 1]),
        )
        return np.stack([roll, pitch, yaw], axis=-1).astype(np.float32)

    flat = rotations.reshape(-1, 3, 3)
    quaternions = np.stack([_matrix_to_quaternion_wxyz(matrix) for matrix in flat])
    return quaternions.reshape(*prefix, 4).astype(np.float32)


def decode_rotation_representation(values: np.ndarray, representation: str) -> np.ndarray:
    """선택한 회전 벡터 표현을 rotation matrix ``(...,3,3)``로 복원."""
    encoded = np.asarray(values, dtype=np.float64)
    expected_dim = ROTATION_REPRESENTATION_DIMS.get(representation)
    if expected_dim is None:
        raise ValueError(
            f"unknown rotation representation {representation!r}; "
            f"expected one of {ROTATION_REPRESENTATIONS}"
        )
    if encoded.ndim == 0 or encoded.shape[-1] != expected_dim:
        raise ValueError(
            f"{representation} rotation shape must end in {expected_dim}, got {encoded.shape}"
        )
    if not np.all(np.isfinite(encoded)):
        raise ValueError(f"{representation} rotation contains NaN or infinity")

    prefix = encoded.shape[:-1]
    if representation == "rot6d":
        rows = encoded.reshape(*prefix, 2, 3)
        row0 = rows[..., 0, :]
        row0_norm = np.linalg.norm(row0, axis=-1, keepdims=True)
        if np.any(row0_norm < 1e-12):
            raise ValueError("rot6d first row norm is zero")
        row0 = row0 / row0_norm
        row1 = rows[..., 1, :] - np.sum(rows[..., 1, :] * row0, axis=-1, keepdims=True) * row0
        row1_norm = np.linalg.norm(row1, axis=-1, keepdims=True)
        if np.any(row1_norm < 1e-12):
            raise ValueError("rot6d rows are parallel")
        row1 = row1 / row1_norm
        row2 = np.cross(row0, row1)
        return np.stack([row0, row1, row2], axis=-2)

    if representation == "rpy":
        roll, pitch, yaw = np.moveaxis(encoded, -1, 0)
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        rotations = np.empty((*prefix, 3, 3), dtype=np.float64)
        rotations[..., 0, 0] = cy * cp
        rotations[..., 0, 1] = cy * sp * sr - sy * cr
        rotations[..., 0, 2] = cy * sp * cr + sy * sr
        rotations[..., 1, 0] = sy * cp
        rotations[..., 1, 1] = sy * sp * sr + cy * cr
        rotations[..., 1, 2] = sy * sp * cr - cy * sr
        rotations[..., 2, 0] = -sp
        rotations[..., 2, 1] = cp * sr
        rotations[..., 2, 2] = cp * cr
        return rotations

    flat = encoded.reshape(-1, 4)
    rotations = np.stack([_quaternion_wxyz_to_matrix(quaternion) for quaternion in flat])
    return rotations.reshape(*prefix, 3, 3)


def _axis_angle_rotation_batch(axis: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """고정 local axis와 N개 angle → ``(N, 3, 3)`` rotation matrix."""
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        raise ValueError("revolute joint axis norm is zero")
    x, y, z = axis / norm
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    outer = np.outer([x, y, z], [x, y, z])
    cos = np.cos(angles)[:, None, None]
    sin = np.sin(angles)[:, None, None]
    return cos * np.eye(3)[None, :, :] + (1.0 - cos) * outer[None, :, :] + sin * skew[None, :, :]


@dataclass(frozen=True)
class _Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray


class SO101EndEffectorKinematics:
    """URDF arm chain과 robot YAML TCP를 사용하는 vectorized SO-101 FK."""

    def __init__(
        self,
        *,
        base_link: str,
        tcp_name: str,
        chain: tuple[_Joint, ...],
        parent_to_tcp: np.ndarray,
    ) -> None:
        self.base_link = base_link
        self.tcp_name = tcp_name
        self.chain = chain
        self.parent_to_tcp = np.asarray(parent_to_tcp, dtype=np.float64)

        movable = tuple(j.name for j in chain if j.joint_type in {"revolute", "continuous"})
        if movable != ARM_JOINT_ORDER:
            raise ValueError(
                f"URDF arm chain order mismatch: {movable}, expected {ARM_JOINT_ORDER}"
            )

    @classmethod
    def from_files(
        cls,
        urdf_path: str | Path,
        robot_yaml_path: str | Path,
        *,
        tcp_name: str = "tcp_grasp",
    ) -> "SO101EndEffectorKinematics":
        urdf_path = Path(urdf_path)
        robot_yaml_path = Path(robot_yaml_path)
        if not urdf_path.is_file():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        if not robot_yaml_path.is_file():
            raise FileNotFoundError(f"robot YAML not found: {robot_yaml_path}")

        import yaml  # PyYAML: LeRobot 공통 의존성, 실제 FK config 로드 시에만 필요.

        with robot_yaml_path.open(encoding="utf-8") as stream:
            robot_cfg = yaml.safe_load(stream)["kinematics"]
        base_link = str(robot_cfg["base_link"])
        tcp_cfg = robot_cfg["extra_links"][tcp_name]
        tcp_values = np.asarray(tcp_cfg["fixed_transform"], dtype=np.float64)
        if tcp_values.shape != (7,):
            raise ValueError(
                f"{tcp_name}.fixed_transform must be [xyz,qwxyz] length 7, got {tcp_values}"
            )
        tcp_parent = str(tcp_cfg["parent_link_name"])
        parent_to_tcp = _homogeneous(
            _quaternion_wxyz_to_matrix(tcp_values[3:]),
            tcp_values[:3],
        )

        root = ET.parse(urdf_path).getroot()
        joints_by_child: dict[str, _Joint] = {}
        for element in root.findall("joint"):
            name = str(element.get("name"))
            joint_type = str(element.get("type"))
            parent_node = element.find("parent")
            child_node = element.find("child")
            if parent_node is None or child_node is None:
                raise ValueError(f"URDF joint {name!r} has no parent/child")
            parent = str(parent_node.get("link"))
            child = str(child_node.get("link"))
            axis_node = element.find("axis")
            axis = _parse_vector(
                None if axis_node is None else axis_node.get("xyz"),
                3,
                default=(0.0, 0.0, 0.0) if joint_type == "fixed" else (1.0, 0.0, 0.0),
            )
            joints_by_child[child] = _Joint(
                name=name,
                joint_type=joint_type,
                parent=parent,
                child=child,
                origin=_origin_transform(element.find("origin")),
                axis=axis,
            )

        reversed_chain: list[_Joint] = []
        current = tcp_parent
        visited: set[str] = set()
        while current != base_link:
            if current in visited:
                raise ValueError(f"cycle in URDF chain at link {current!r}")
            visited.add(current)
            if current not in joints_by_child:
                raise ValueError(
                    f"cannot trace URDF chain from {tcp_parent!r} to {base_link!r}; "
                    f"missing parent joint for {current!r}"
                )
            joint = joints_by_child[current]
            reversed_chain.append(joint)
            current = joint.parent

        return cls(
            base_link=base_link,
            tcp_name=tcp_name,
            chain=tuple(reversed(reversed_chain)),
            parent_to_tcp=parent_to_tcp,
        )

    def forward_matrices(self, joint_radians: np.ndarray) -> np.ndarray:
        """Arm joint radian ``(...,5|6)`` → absolute TCP transform ``(...,4,4)``."""
        joints = np.asarray(joint_radians, dtype=np.float64)
        if joints.ndim == 0 or joints.shape[-1] not in {5, 6}:
            raise ValueError(f"joint_radians shape must end in 5 or 6, got {joints.shape}")
        if not np.all(np.isfinite(joints)):
            raise ValueError("joint_radians contains NaN or infinity")

        prefix = joints.shape[:-1]
        flat = joints[..., :5].reshape(-1, 5)
        batch = flat.shape[0]
        transforms = np.broadcast_to(np.eye(4), (batch, 4, 4)).copy()
        joint_indices = {name: i for i, name in enumerate(ARM_JOINT_ORDER)}

        for joint in self.chain:
            transforms = transforms @ joint.origin[None, :, :]
            if joint.joint_type == "fixed":
                continue
            if joint.joint_type not in {"revolute", "continuous"}:
                raise ValueError(f"unsupported joint type {joint.joint_type!r}: {joint.name}")
            rotations = _axis_angle_rotation_batch(
                joint.axis,
                flat[:, joint_indices[joint.name]],
            )
            motion = np.broadcast_to(np.eye(4), (batch, 4, 4)).copy()
            motion[:, :3, :3] = rotations
            transforms = transforms @ motion

        transforms = transforms @ self.parent_to_tcp[None, :, :]
        return transforms.reshape(*prefix, 4, 4)

    def forward_xyz_rotation(
        self,
        joint_radians: np.ndarray,
        representation: str = "rot6d",
    ) -> np.ndarray:
        """Arm joint radian → ``xyz + selected rotation`` absolute EEF pose."""
        transforms = self.forward_matrices(joint_radians)
        xyz = transforms[..., :3, 3]
        rotation = encode_rotation_matrices(transforms[..., :3, :3], representation)
        return np.concatenate([xyz, rotation], axis=-1).astype(np.float32)

    def forward_xyz_rot6d(self, joint_radians: np.ndarray) -> np.ndarray:
        """Arm joint radian → GR00T ``xyz + rotation 첫 두 행`` absolute EEF 9D."""
        return self.forward_xyz_rotation(joint_radians, "rot6d")

    def forward_xyz_rpy(self, joint_radians: np.ndarray) -> np.ndarray:
        """Arm joint radian → ``xyz + fixed-axis RPY radians`` absolute EEF 6D."""
        return self.forward_xyz_rotation(joint_radians, "rpy")

    def forward_xyz_wxyz(self, joint_radians: np.ndarray) -> np.ndarray:
        """Arm joint radian → ``xyz + canonical quaternion wxyz`` absolute EEF 7D."""
        return self.forward_xyz_rotation(joint_radians, "wxyz")
