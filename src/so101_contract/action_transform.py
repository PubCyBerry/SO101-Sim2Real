"""Phase 13–14 — mode/format 중립 action target 변환.

4개 mode의 "absolute dataset action → model target" 및 그 역변환을 한 곳에 모은다.
stats 생성기와 LeRobot processor가 **같은 구현**을 공유하므로 두 경로의 수치가 갈라지지
않는다.

.. code-block:: text

    joint_absolute  encode = canonical wrap only          decode = canonical wrap only
    joint_relative  encode = topology-aware difference    decode = topology-aware add + wrap
    eef_absolute    encode = pose canonicalization        decode = pose canonicalization
    eef_relative    encode = inv(T_state) @ T_action      decode = T_state @ T_relative

공통 규칙:

- chunk의 모든 horizon이 **하나의** 기준 state를 공유한다.
- gripper 등 passthrough group은 어떤 mode에서도 변환하지 않는다.
- ``xyz_quaternion_wxyz``는 부호 정규화 후 chunk 시간축 연속성까지 적용하고,
  ``xyz_rpy``는 wrap을 적용한다. 이 정규화는 **stats 계산과 학습 target 생성 전에** 끝난다.
- ``encode``와 ``decode``가 같은 canonical form을 보장한다. 복원된 absolute EEF chunk도
  quaternion 부호/연속성, RPY wrap, Rot6D 직교 row 계약을 만족한다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any

from .action_representation import (
    ActionRepresentationMode,
    ActionRepresentationSpec,
    PoseFormat,
)
from .array_backend import (
    Array,
    replace_indices,
    take_indices,
    validate_indices,
)
from .joint_feature_codec import (
    build_joint_feature_contract,
    validate_joint_feature_contract,
)
from .joint_topology import (
    JointTopology,
    absolute_joint_actions_to_relative,
    canonicalize_joint_actions,
    relative_joint_actions_to_absolute,
)
from .pose_codec import (
    POSE_CODEC_VERSION,
    absolute_actions_to_relative,
    canonicalize_pose,
    canonicalize_quaternion_sequence,
    relative_actions_to_absolute,
)

ACTION_TRANSFORM_VERSION = "so101_action_transform_v2"


def _is_so101_canonical(joint_topology: dict[str, Any] | None) -> bool:
    """SO-101 canonical joint layout(전 arm joint radian)인지."""
    if not isinstance(joint_topology, dict):
        return False
    joints = joint_topology.get("joints")
    if not isinstance(joints, list) or not joints:
        return False
    return all(
        isinstance(joint, dict) and joint.get("unit") == "radian" for joint in joints
    )


@dataclass(frozen=True)
class ActionRepresentationTransform:
    """Resolved index/topology를 포함한 mode별 target 변환기.

    dataset metadata에서 resolve한 index와 topology만 담으므로, checkpoint에 직렬화해
    dataset 없이 복원할 수 있다.
    """

    spec: ActionRepresentationSpec
    state_indices: tuple[int, ...]
    action_indices: tuple[int, ...]
    passthrough_action_indices: tuple[int, ...]
    state_dim: int
    action_dim: int
    joint_topology: JointTopology | None = None
    joint_feature_contract: dict[str, Any] | None = None
    version: str = ACTION_TRANSFORM_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.spec, ActionRepresentationSpec):
            raise TypeError(f"spec must be ActionRepresentationSpec, got {type(self.spec).__name__}")
        if self.version != ACTION_TRANSFORM_VERSION:
            raise ValueError(
                f"action transform version mismatch: {self.version!r} != "
                f"{ACTION_TRANSFORM_VERSION!r}"
            )
        state_dim = int(self.state_dim)
        action_dim = int(self.action_dim)
        if state_dim <= 0 or action_dim <= 0:
            raise ValueError(f"state/action dim must be positive, got {state_dim}/{action_dim}")

        expected_count = self.spec.pose_dim if self.spec.is_eef else None
        if not self.spec.is_eef:
            if self.joint_topology is None:
                raise ValueError(
                    f"mode={self.spec.mode.value!r} requires an explicit joint topology"
                )
            expected_count = self.joint_topology.dim
        elif self.joint_topology is not None:
            raise ValueError("EEF modes must not carry a joint topology")

        object.__setattr__(
            self,
            "state_indices",
            validate_indices(
                self.state_indices,
                feature_dim=state_dim,
                expected_count=expected_count,
                name="state_indices",
            ),
        )
        object.__setattr__(
            self,
            "action_indices",
            validate_indices(
                self.action_indices,
                feature_dim=action_dim,
                expected_count=expected_count,
                name="action_indices",
            ),
        )
        object.__setattr__(
            self,
            "passthrough_action_indices",
            validate_indices(
                self.passthrough_action_indices,
                feature_dim=action_dim,
                expected_count=None,
                name="passthrough_action_indices",
            ),
        )
        if set(self.action_indices).intersection(self.passthrough_action_indices):
            raise ValueError("transform and passthrough action indices overlap")
        classified = set(self.action_indices) | set(self.passthrough_action_indices)
        if classified != set(range(action_dim)):
            raise ValueError(
                "action indices do not classify the complete action vector: "
                f"classified={sorted(classified)}, expected={list(range(action_dim))}"
            )
        object.__setattr__(self, "state_dim", state_dim)
        object.__setattr__(self, "action_dim", action_dim)

        # joint mode는 명시적 feature 계약(단위·gripper 의미·index)을 항상 들고 다닌다.
        # 단일 소스 builder로 만들고, 외부에서 들어온 payload는 검증 후 그대로 보존한다.
        if self.spec.is_eef:
            if self.joint_feature_contract is not None:
                raise ValueError("EEF transforms must not carry a joint feature contract")
        else:
            gripper_index = int(self.passthrough_action_indices[0])
            try:
                built = build_joint_feature_contract(
                    self.joint_topology.to_dict(),
                    gripper_index=gripper_index,
                    gripper_group=self.spec.passthrough_action_groups[0],
                )
            except ValueError:
                # SO-101 canonical joint layout(전 arm radian)이 아니면 계약을 만들지 않는다.
                # 이런 transform은 추론 startup에서 거부된다(계약 payload 필수).
                built = None
            if self.joint_feature_contract is None:
                object.__setattr__(self, "joint_feature_contract", built)
            elif built is None:
                raise ValueError(
                    "serialized joint feature contract is present but the topology is not "
                    "SO-101 canonical (all arm joints must be radian)"
                )
            else:
                validated = validate_joint_feature_contract(
                    self.joint_feature_contract,
                    joint_topology=self.joint_topology.to_dict(),
                    gripper_representation=self.spec.gripper_representation,
                    action_dim=action_dim,
                    source="serialized transform",
                )
                if validated != built:
                    raise ValueError(
                        "serialized joint feature contract disagrees with the resolved "
                        f"topology/indices: {validated} != {built}"
                    )

    # --- 내부 helper ---------------------------------------------------------

    def _validate_runtime(self, state: Array | None, actions: Array) -> None:
        if state is None:
            if self.requires_state_reference:
                raise ValueError(
                    f"mode={self.spec.mode.value!r} requires a reference state; absolute modes "
                    "may pass None"
                )
        elif state.shape[-1] != self.state_dim:
            raise ValueError(
                f"runtime state dim {state.shape[-1]} != contract {self.state_dim}"
            )
        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"runtime action dim {actions.shape[-1]} != contract {self.action_dim}"
            )
        if actions.ndim < 2:
            raise ValueError(
                f"action must contain a chunk dimension, got shape {tuple(actions.shape)}"
            )

    def _canonicalize_pose_group(self, actions: Array) -> Array:
        """Pose group을 format canonical 형태로 정규화(+ quaternion 시간축 연속성)."""
        pose = take_indices(actions, self.action_indices)
        pose = canonicalize_pose(pose, self.spec.pose_format)
        if self.spec.pose_format is PoseFormat.XYZ_QUATERNION_WXYZ and pose.ndim >= 2:
            quaternion = canonicalize_quaternion_sequence(pose[..., 3:7], axis=-2)
            pose = replace_indices(pose, (3, 4, 5, 6), quaternion)
        return replace_indices(actions, self.action_indices, pose)

    # --- 공개 API ------------------------------------------------------------

    def encode(self, state: Array | None, actions: Array) -> Array:
        """Absolute dataset action chunk → model target chunk.

        absolute mode는 기준 state가 필요 없으므로 ``state=None``을 허용한다.
        """
        self._validate_runtime(state, actions)
        mode = self.spec.mode
        if mode is ActionRepresentationMode.EEF_RELATIVE:
            relative = absolute_actions_to_relative(
                state,
                actions,
                self.spec.pose_format,
                state_pose_indices=self.state_indices,
                action_pose_indices=self.action_indices,
            )
            return self._canonicalize_pose_group(relative)
        if mode is ActionRepresentationMode.EEF_ABSOLUTE:
            return self._canonicalize_pose_group(actions)
        if mode is ActionRepresentationMode.JOINT_RELATIVE:
            return absolute_joint_actions_to_relative(
                state,
                actions,
                self.joint_topology,
                state_joint_indices=self.state_indices,
                action_joint_indices=self.action_indices,
            )
        return canonicalize_joint_actions(
            actions,
            self.joint_topology,
            action_joint_indices=self.action_indices,
        )

    def decode(self, state: Array | None, targets: Array) -> Array:
        """Model target chunk → absolute action chunk.

        absolute mode는 기준 state가 필요 없으므로 ``state=None``을 허용한다.
        """
        self._validate_runtime(state, targets)
        mode = self.spec.mode
        if mode is ActionRepresentationMode.EEF_RELATIVE:
            absolute = relative_actions_to_absolute(
                state,
                targets,
                self.spec.pose_format,
                state_pose_indices=self.state_indices,
                action_pose_indices=self.action_indices,
            )
            # 복원된 chunk도 encode와 같은 canonical form을 만족해야 한다:
            # quaternion 부호/시간축 연속성, RPY wrap, Rot6D 직교 row.
            return self._canonicalize_pose_group(absolute)
        if mode is ActionRepresentationMode.EEF_ABSOLUTE:
            return self._canonicalize_pose_group(targets)
        if mode is ActionRepresentationMode.JOINT_RELATIVE:
            return relative_joint_actions_to_absolute(
                state,
                targets,
                self.joint_topology,
                state_joint_indices=self.state_indices,
                action_joint_indices=self.action_indices,
            )
        return canonicalize_joint_actions(
            targets,
            self.joint_topology,
            action_joint_indices=self.action_indices,
        )

    @property
    def requires_state_reference(self) -> bool:
        """Relative mode만 prediction-time state cache가 필요하다."""
        return self.spec.is_relative

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "pose_codec_version": POSE_CODEC_VERSION,
            "action_representation": self.spec.to_dict(),
            "state_indices": list(self.state_indices),
            "action_indices": list(self.action_indices),
            "passthrough_action_indices": list(self.passthrough_action_indices),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "joint_topology": (
                self.joint_topology.to_dict() if self.joint_topology is not None else None
            ),
            "joint_feature_contract": self.joint_feature_contract,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ActionRepresentationTransform:
        if not isinstance(payload, dict):
            raise TypeError(f"transform payload must be an object, got {type(payload).__name__}")
        if payload.get("pose_codec_version") not in (None, POSE_CODEC_VERSION):
            raise ValueError(
                f"pose codec version mismatch: {payload.get('pose_codec_version')!r} != "
                f"{POSE_CODEC_VERSION!r}"
            )
        topology = payload.get("joint_topology")
        spec = ActionRepresentationSpec.from_dict(payload["action_representation"])
        contract = payload.get("joint_feature_contract")
        if not spec.is_eef and contract is None and _is_so101_canonical(topology):
            # 신규 v2 manifest는 명시적 계약을 반드시 담는다(기본값 합성 금지).
            raise ValueError(
                "serialized joint transform has no explicit joint feature contract; "
                "regenerate or migrate the checkpoint"
            )
        return cls(
            spec=spec,
            state_indices=tuple(payload["state_indices"]),
            action_indices=tuple(payload["action_indices"]),
            passthrough_action_indices=tuple(payload["passthrough_action_indices"]),
            state_dim=int(payload["state_dim"]),
            action_dim=int(payload["action_dim"]),
            joint_topology=JointTopology.from_dict(topology) if topology else None,
            joint_feature_contract=contract,
            version=payload.get("version", ACTION_TRANSFORM_VERSION),
        )

    def fingerprint(self) -> str:
        canonical = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


def make_action_transform(
    spec: ActionRepresentationSpec,
    *,
    state_indices: Sequence[int],
    action_indices: Sequence[int],
    passthrough_action_indices: Sequence[int],
    state_dim: int,
    action_dim: int,
    joint_topology: JointTopology | None = None,
    joint_feature_contract: dict[str, Any] | None = None,
) -> ActionRepresentationTransform:
    return ActionRepresentationTransform(
        spec=spec,
        state_indices=tuple(state_indices),
        action_indices=tuple(action_indices),
        passthrough_action_indices=tuple(passthrough_action_indices),
        state_dim=int(state_dim),
        action_dim=int(action_dim),
        joint_topology=joint_topology,
        joint_feature_contract=joint_feature_contract,
    )
