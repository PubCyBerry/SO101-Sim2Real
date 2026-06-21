"""Canonical policy I/O 계약과 fail-closed 검증."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

CANONICAL_SCHEMA = "so101-canonical-v1"
JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
FEATURE_NAMES = tuple(f"{name}.pos" for name in JOINT_ORDER)
CAMERA_ORDER = ("top", "wrist", "front")


class ContractError(ValueError):
    """실행을 중단해야 하는 policy I/O 계약 불일치."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: Mapping[str, Any], hash_key: str) -> str:
    payload = {key: item for key, item in value.items() if key != hash_key}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PolicyIOContract:
    schema: str
    joint_order: tuple[str, ...]
    feature_names: tuple[str, ...]
    arm_unit: str
    gripper_unit: str
    action_mode: str
    fps: int
    cameras: tuple[str, ...]
    image_shape: tuple[int, int, int]
    image_dtype: str
    image_encoding: str
    contract_hash: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PolicyIOContract":
        expected_hash = content_hash(raw, "contract_hash")
        supplied_hash = raw.get("contract_hash")
        if supplied_hash and supplied_hash != expected_hash:
            raise ContractError(
                f"policy_io contract hash 불일치: supplied={supplied_hash}, expected={expected_hash}"
            )
        cameras = raw["cameras"]
        contract = cls(
            schema=str(raw["schema"]),
            joint_order=tuple(raw["joint_order"]),
            feature_names=tuple(raw["feature_names"]),
            arm_unit=str(raw["units"]["arm"]),
            gripper_unit=str(raw["units"]["gripper"]),
            action_mode=str(raw["action_mode"]),
            fps=int(raw["fps"]),
            cameras=tuple(cameras["order"]),
            image_shape=tuple(int(value) for value in cameras["shape"]),
            image_dtype=str(cameras["dtype"]),
            image_encoding=str(cameras["encoding"]),
            contract_hash=expected_hash,
        )
        contract.validate()
        return contract

    @classmethod
    def load(cls, path: str | Path) -> "PolicyIOContract":
        with Path(path).open("r", encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))

    def validate(self) -> None:
        checks = {
            "schema": (self.schema, CANONICAL_SCHEMA),
            "joint_order": (self.joint_order, JOINT_ORDER),
            "feature_names": (self.feature_names, FEATURE_NAMES),
            "arm_unit": (self.arm_unit, "urdf_radian"),
            "gripper_unit": (self.gripper_unit, "jaw_aperture_mm"),
            "action_mode": (self.action_mode, "absolute_position_target"),
            "fps": (self.fps, 30),
            "cameras": (self.cameras, CAMERA_ORDER),
            "image_shape": (self.image_shape, (480, 640, 3)),
            "image_dtype": (self.image_dtype, "uint8"),
            "image_encoding": (self.image_encoding, "rgb8"),
        }
        mismatches = [
            f"{name}: actual={actual!r}, expected={expected!r}"
            for name, (actual, expected) in checks.items()
            if actual != expected
        ]
        if mismatches:
            raise ContractError("canonical policy I/O 계약 불일치: " + "; ".join(mismatches))

    def assert_compatible(self, other: "PolicyIOContract") -> None:
        if self.contract_hash != other.contract_hash:
            raise ContractError(
                f"contract hash 불일치: runtime={self.contract_hash}, artifact={other.contract_hash}"
            )

    def validate_state(self, values: np.ndarray | list[float]) -> np.ndarray:
        result = np.asarray(values, dtype=np.float32)
        if result.shape != (6,):
            raise ContractError(f"canonical state/action shape은 (6,)이어야 한다: {result.shape}")
        if not np.all(np.isfinite(result)):
            raise ContractError("canonical state/action에 NaN/Inf가 있다")
        return np.ascontiguousarray(result)

    def validate_images(self, images: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        if tuple(images) != self.cameras:
            raise ContractError(
                f"camera 순서 불일치: actual={tuple(images)}, expected={self.cameras}"
            )
        validated: dict[str, np.ndarray] = {}
        for name in self.cameras:
            image = np.asarray(images[name])
            if image.shape != self.image_shape or str(image.dtype) != self.image_dtype:
                raise ContractError(
                    f"{name} image 계약 불일치: shape={image.shape}, dtype={image.dtype}, "
                    f"expected={self.image_shape}/{self.image_dtype}"
                )
            validated[name] = np.ascontiguousarray(image)
        return validated


def default_contract_dict() -> dict[str, Any]:
    raw: dict[str, Any] = {
        "schema": CANONICAL_SCHEMA,
        "joint_order": list(JOINT_ORDER),
        "feature_names": list(FEATURE_NAMES),
        "units": {"arm": "urdf_radian", "gripper": "jaw_aperture_mm"},
        "action_mode": "absolute_position_target",
        "fps": 30,
        "cameras": {
            "order": list(CAMERA_ORDER),
            "shape": [480, 640, 3],
            "dtype": "uint8",
            "encoding": "rgb8",
        },
    }
    raw["contract_hash"] = content_hash(raw, "contract_hash")
    return raw
