#!/usr/bin/env python3
"""Phase 14 — schema v2 dataset action contract 검증.

확인 항목:

- Rot6D/wxyz/RPY dataset metadata의 feature names·group·frame·format·dimension 엄격 resolve
- ``scripts/convert/joint_dataset_to_eef.py``가 실제로 쓰는 names/metadata와 대조
- joint dataset의 group/joint topology 선언(``so101_action_representation``)과 주입 경로
- dataset absolute-storage 원칙과 mode↔dataset space 정합
- 잘못된 metadata의 fail-fast와 contract fingerprint 무효화

.. code-block:: bash

    python scripts/contract/validate_action_dataset_contract.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

REPO_ROOT = Path(__file__).resolve().parents[2]

from so101_contract.action_dataset_contract import (  # noqa: E402
    ACTION_DATASET_CONTRACT_VERSION,
    DATASET_CONTRACT_BLOCK,
    EEF_ROTATION_NAMES,
    POSE_FORMAT_METADATA_STRING,
    ROTATION_REPRESENTATION_TO_POSE_FORMAT,
    eef_feature_names,
    resolve_action_contract_from_metadata,
    resolve_action_representation_contract,
)
from so101_contract.action_representation import (  # noqa: E402
    EEF_POSE_FORMATS,
    ActionRepresentationMode,
    ActionRepresentationSpec,
    PoseFormat,
)
from so101_contract.joint_topology import TAU, JointType  # noqa: E402

_FAKE_SHA256 = "a" * 64
_ARM_JOINT_NAMES = tuple(f"arm.joint_{index}" for index in range(5))
_JOINT_METADATA = {
    name: {"type": "revolute", "period": TAU, "lower": -math.pi, "upper": math.pi}
    for name in _ARM_JOINT_NAMES
}


def _eef_info(pose_format: PoseFormat) -> dict:
    names = list(eef_feature_names(pose_format))
    representation = next(
        key
        for key, value in ROTATION_REPRESENTATION_TO_POSE_FORMAT.items()
        if value is pose_format
    )
    feature = {"dtype": "float32", "shape": [len(names)], "names": names}
    return {
        "codebase_version": "v3.0",
        "features": {"observation.state": feature, "action": dict(feature)},
        "so101_eef_conversion": {
            "version": "so101_lerobot_abs_joint_to_abs_eef_v2",
            "source_domain": "sim",
            "base_frame": "base_link",
            "eef_frame": "tcp_grasp",
            "eef_kinematics_version": "so101_base_tcp_grasp_fk_v2",
            "rotation_representation": representation,
            "rotation_format": POSE_FORMAT_METADATA_STRING[pose_format],
            "gripper_format": "canonical_policy_feature_[0,100]",
            "keep_joints": False,
            "urdf_sha256": _FAKE_SHA256,
            "robot_yaml_sha256": _FAKE_SHA256,
        },
    }


def _eef_modality(pose_format: PoseFormat) -> dict:
    pose_dim = 3 + len(EEF_ROTATION_NAMES[pose_format])
    split = {
        f"eef_{pose_dim}d": {"start": 0, "end": pose_dim},
        "gripper_position": {"start": pose_dim, "end": pose_dim + 1},
    }
    return {"state": split, "action": split}


def _joint_info(*, declare_block: bool = True) -> dict:
    names = list(_ARM_JOINT_NAMES) + ["gripper.pos"]
    feature = {"dtype": "float32", "shape": [len(names)], "names": names}
    info = {
        "codebase_version": "v3.0",
        "features": {"observation.state": feature, "action": dict(feature)},
    }
    if declare_block:
        info[DATASET_CONTRACT_BLOCK] = {
            "version": ACTION_DATASET_CONTRACT_VERSION,
            "space": "joint",
            "storage_reference": "absolute",
            "groups": {
                "arm_joints": {"start": 0, "end": 5},
                "gripper_position": {"start": 5, "end": 6},
            },
            "joints": [
                {"name": name, **_JOINT_METADATA[name]} for name in _ARM_JOINT_NAMES
            ],
        }
    return info


def _eef_spec(pose_format: PoseFormat, *, relative: bool = True) -> ActionRepresentationSpec:
    return ActionRepresentationSpec(
        mode=(
            ActionRepresentationMode.EEF_RELATIVE
            if relative
            else ActionRepresentationMode.EEF_ABSOLUTE
        ),
        pose_format=pose_format,
    )


def check_converter_metadata_parity() -> None:
    """실제 converter 상수와 v2 계약이 같은 names/format을 쓰는지 대조."""
    module_path = REPO_ROOT / "scripts" / "convert" / "joint_dataset_to_eef.py"
    source = module_path.read_text(encoding="utf-8")
    # converter는 Isaac/lerobot 의존이 있어 import하지 않고 상수 리터럴만 대조한다.
    for pose_format, names in EEF_ROTATION_NAMES.items():
        for name in names:
            if f'"{name}"' not in source:
                raise AssertionError(
                    f"converter does not emit the rotation feature name {name!r} "
                    f"expected by {pose_format.value}"
                )
    for pose_format, metadata in POSE_FORMAT_METADATA_STRING.items():
        if f'"{metadata}"' not in source:
            raise AssertionError(
                f"converter rotation_format string {metadata!r} for {pose_format.value} is missing"
            )
    for name in ("tcp_grasp.x", "tcp_grasp.y", "tcp_grasp.z", "gripper.pos"):
        if f'"{name}"' not in source:
            raise AssertionError(f"converter does not emit the feature name {name!r}")
    if 'f"eef_{eef_dim}d"' not in source:
        raise AssertionError("converter no longer emits the eef_<dim>d modality group name")
    if importlib.util.find_spec("so101_contract") is None:  # pragma: no cover - 방어용
        raise AssertionError("so101_contract is not importable")
    print("PASS: converter feature names/metadata match the v2 contract (3 pose formats)")


def check_eef_contract_all_formats() -> None:
    """3개 pose format 모두 resolve되고 dimension/index가 정확하다."""
    expected_dims = {
        PoseFormat.XYZ_ROT6D_ROWS: (9, 10),
        PoseFormat.XYZ_QUATERNION_WXYZ: (7, 8),
        PoseFormat.XYZ_RPY: (6, 7),
    }
    fingerprints = set()
    for pose_format in EEF_POSE_FORMATS:
        pose_dim, action_dim = expected_dims[pose_format]
        for relative in (True, False):
            spec = _eef_spec(pose_format, relative=relative)
            contract = resolve_action_contract_from_metadata(
                _eef_info(pose_format),
                _eef_modality(pose_format),
                spec,
                info_sha256=_FAKE_SHA256,
                modality_sha256=_FAKE_SHA256,
            )
            if contract.action_dim != action_dim or contract.state_dim != action_dim:
                raise AssertionError(
                    f"{pose_format.value} dim mismatch: {contract.action_dim} != {action_dim}"
                )
            if contract.transform.action_indices != tuple(range(pose_dim)):
                raise AssertionError(f"{pose_format.value} pose indices mismatch")
            if contract.transform.passthrough_action_indices != (pose_dim,):
                raise AssertionError(f"{pose_format.value} gripper passthrough index mismatch")
            if contract.transform.joint_topology is not None:
                raise AssertionError("EEF contract must not carry a joint topology")
            if contract.base_frame != "base_link" or contract.eef_frame != "tcp_grasp":
                raise AssertionError("EEF frames were not resolved")
            if contract.urdf_sha256 != _FAKE_SHA256:
                raise AssertionError("kinematics hashes were not resolved")
            if contract.schema_version != ACTION_DATASET_CONTRACT_VERSION:
                raise AssertionError("dataset contract schema version mismatch")
            fingerprints.add(contract.fingerprint)
    if len(fingerprints) != 6:
        raise AssertionError(f"expected 6 distinct EEF contracts, got {len(fingerprints)}")
    print("PASS: EEF dataset contract resolves for rot6d/wxyz/rpy in absolute and relative modes")


def check_eef_metadata_fail_fast() -> None:
    """잘못된 EEF metadata는 추정 보정 없이 거부한다."""
    spec = _eef_spec(PoseFormat.XYZ_QUATERNION_WXYZ)

    def resolve(info=None, modality=None, target_spec=None):
        return resolve_action_contract_from_metadata(
            info if info is not None else _eef_info(PoseFormat.XYZ_QUATERNION_WXYZ),
            modality if modality is not None else _eef_modality(PoseFormat.XYZ_QUATERNION_WXYZ),
            target_spec or spec,
            info_sha256=_FAKE_SHA256,
            modality_sha256=_FAKE_SHA256,
        )

    def mutate(**changes):
        info = json.loads(json.dumps(_eef_info(PoseFormat.XYZ_QUATERNION_WXYZ)))
        for key, value in changes.items():
            target = info
            *parents, leaf = key.split(".")
            for parent in parents:
                target = target[parent]
            if value is None:
                target.pop(leaf, None)
            else:
                target[leaf] = value
        return info

    rot6d_names = list(eef_feature_names(PoseFormat.XYZ_ROT6D_ROWS))
    rejects = {
        "format mismatch with dataset": lambda: resolve(
            target_spec=_eef_spec(PoseFormat.XYZ_RPY)
        ),
        "missing conversion block": lambda: resolve(
            info=mutate(**{"so101_eef_conversion": None})
        ),
        "wrong rotation format string": lambda: resolve(
            info=mutate(**{"so101_eef_conversion.rotation_format": "xyz+quat"})
        ),
        "wrong frame": lambda: resolve(
            info=mutate(**{"so101_eef_conversion.eef_frame": "gripper_tip"})
        ),
        "keep_joints enabled": lambda: resolve(
            info=mutate(**{"so101_eef_conversion.keep_joints": True})
        ),
        "invalid urdf hash": lambda: resolve(
            info=mutate(**{"so101_eef_conversion.urdf_sha256": "deadbeef"})
        ),
        "unexpected feature names": lambda: resolve(
            info=mutate(
                **{
                    "features.observation.state": {
                        "dtype": "float32",
                        "shape": [8],
                        "names": ["x"] * 8,
                    }
                }
            )
        ),
        "rot6d names with wxyz config": lambda: resolve(
            info=mutate(
                **{
                    "features.observation.state": {
                        "dtype": "float32",
                        "shape": [10],
                        "names": rot6d_names,
                    },
                    "features.action": {
                        "dtype": "float32",
                        "shape": [10],
                        "names": rot6d_names,
                    },
                }
            )
        ),
        "missing modality groups": lambda: resolve(modality={"state": {}, "action": {}}),
        "group does not partition": lambda: resolve(
            modality={
                "state": {"eef_7d": {"start": 0, "end": 7}},
                "action": {"eef_7d": {"start": 0, "end": 7}},
            }
        ),
        "wrong pose group size": lambda: resolve(
            modality={
                "state": {
                    "eef_7d": {"start": 0, "end": 6},
                    "gripper_position": {"start": 6, "end": 8},
                },
                "action": {
                    "eef_7d": {"start": 0, "end": 6},
                    "gripper_position": {"start": 6, "end": 8},
                },
            }
        ),
        "wrong codebase version": lambda: resolve(info=mutate(codebase_version="v2.1")),
        "relative dataset storage": lambda: resolve(
            info=mutate(
                **{
                    DATASET_CONTRACT_BLOCK: {
                        "storage_reference": "relative",
                        "space": "eef",
                    }
                }
            )
        ),
        "declared space mismatch": lambda: resolve(
            info=mutate(**{DATASET_CONTRACT_BLOCK: {"space": "joint"}})
        ),
    }
    for label, call in rejects.items():
        try:
            call()
        except (KeyError, TypeError, ValueError):
            continue
        raise AssertionError(f"invalid EEF dataset metadata was accepted: {label}")
    print(f"PASS: {len(rejects)} invalid EEF dataset metadata cases rejected")


def check_joint_contract() -> None:
    """joint dataset의 group/topology 선언과 명시적 주입."""
    for mode in (
        ActionRepresentationMode.JOINT_ABSOLUTE,
        ActionRepresentationMode.JOINT_RELATIVE,
    ):
        spec = ActionRepresentationSpec(mode=mode)
        contract = resolve_action_contract_from_metadata(
            _joint_info(),
            None,
            spec,
            info_sha256=_FAKE_SHA256,
        )
        topology = contract.transform.joint_topology
        if topology is None or topology.dim != 5:
            raise AssertionError("joint topology was not resolved from the declared block")
        if topology.names != _ARM_JOINT_NAMES:
            raise AssertionError("joint names were not resolved from feature metadata")
        if any(joint.type is not JointType.REVOLUTE for joint in topology.joints):
            raise AssertionError("declared joint types were lost")
        if contract.transform.passthrough_action_indices != (5,):
            raise AssertionError("gripper passthrough index mismatch")
        if contract.eef_kinematics_version is not None or contract.base_frame is not None:
            raise AssertionError("joint contract must not carry EEF frames/kinematics")
        if contract.modality_sha256 is not None:
            raise AssertionError("joint contract recorded a modality hash it never read")

    # modality.json 없이 metadata만 주입하는 경로(Phase 15 이전 joint dataset 보강).
    spec = ActionRepresentationSpec(mode=ActionRepresentationMode.JOINT_RELATIVE)
    info = _joint_info(declare_block=False)
    info[DATASET_CONTRACT_BLOCK] = {
        "space": "joint",
        "storage_reference": "absolute",
        "groups": {
            "arm_joints": {"start": 0, "end": 5},
            "gripper_position": {"start": 5, "end": 6},
        },
    }
    injected = resolve_action_contract_from_metadata(
        info,
        None,
        spec,
        info_sha256=_FAKE_SHA256,
        joint_metadata=_JOINT_METADATA,
    )
    if injected.transform.joint_topology.dim != 5:
        raise AssertionError("explicitly injected joint metadata was not used")

    try:
        resolve_action_contract_from_metadata(
            info,
            None,
            spec,
            info_sha256=_FAKE_SHA256,
        )
    except KeyError:
        pass
    else:
        raise AssertionError("joint mode without topology metadata must fail fast")

    try:
        resolve_action_contract_from_metadata(
            _joint_info(declare_block=False),
            None,
            spec,
            info_sha256=_FAKE_SHA256,
            joint_metadata=_JOINT_METADATA,
        )
    except KeyError:
        pass
    else:
        raise AssertionError("joint mode without group metadata must fail fast")
    print("PASS: joint dataset contract (declared block, injected metadata, fail-fast)")


def check_fingerprint_invalidation() -> None:
    """계약 요소가 바뀌면 fingerprint가 달라진다."""
    spec = _eef_spec(PoseFormat.XYZ_ROT6D_ROWS)
    base = resolve_action_contract_from_metadata(
        _eef_info(PoseFormat.XYZ_ROT6D_ROWS),
        _eef_modality(PoseFormat.XYZ_ROT6D_ROWS),
        spec,
        info_sha256=_FAKE_SHA256,
        modality_sha256=_FAKE_SHA256,
    )
    same = resolve_action_contract_from_metadata(
        _eef_info(PoseFormat.XYZ_ROT6D_ROWS),
        _eef_modality(PoseFormat.XYZ_ROT6D_ROWS),
        spec,
        info_sha256=_FAKE_SHA256,
        modality_sha256=_FAKE_SHA256,
    )
    if base.fingerprint != same.fingerprint:
        raise AssertionError("identical metadata produced different fingerprints")

    changed_info = resolve_action_contract_from_metadata(
        _eef_info(PoseFormat.XYZ_ROT6D_ROWS),
        _eef_modality(PoseFormat.XYZ_ROT6D_ROWS),
        spec,
        info_sha256="b" * 64,
        modality_sha256=_FAKE_SHA256,
    )
    if changed_info.fingerprint == base.fingerprint:
        raise AssertionError("info.json checksum change must invalidate the fingerprint")

    absolute = resolve_action_contract_from_metadata(
        _eef_info(PoseFormat.XYZ_ROT6D_ROWS),
        _eef_modality(PoseFormat.XYZ_ROT6D_ROWS),
        _eef_spec(PoseFormat.XYZ_ROT6D_ROWS, relative=False),
        info_sha256=_FAKE_SHA256,
        modality_sha256=_FAKE_SHA256,
    )
    if absolute.fingerprint == base.fingerprint:
        raise AssertionError("mode change must invalidate the fingerprint")
    print("PASS: contract fingerprint reacts to metadata, mode and format changes")


def check_dataset_directory_resolution() -> None:
    """실제 디렉터리 layout에서 resolve."""
    with tempfile.TemporaryDirectory(prefix="so101-dataset-contract-") as directory:
        root = Path(directory)
        (root / "meta").mkdir()
        (root / "meta" / "info.json").write_text(
            json.dumps(_eef_info(PoseFormat.XYZ_RPY)),
            encoding="utf-8",
        )
        (root / "meta" / "modality.json").write_text(
            json.dumps(_eef_modality(PoseFormat.XYZ_RPY)),
            encoding="utf-8",
        )
        contract = resolve_action_representation_contract(root, _eef_spec(PoseFormat.XYZ_RPY))
        if contract.action_dim != 7 or contract.modality_sha256 is None:
            raise AssertionError("directory-based EEF resolution failed")
        groups = contract.feature_groups("action")
        if groups != {"eef_6d": [0, 6], "gripper_position": [6, 7]}:
            raise AssertionError(f"manifest feature groups mismatch: {groups}")

        joint_root = root / "joint"
        (joint_root / "meta").mkdir(parents=True)
        (joint_root / "meta" / "info.json").write_text(
            json.dumps(_joint_info()),
            encoding="utf-8",
        )
        joint_contract = resolve_action_representation_contract(
            joint_root,
            ActionRepresentationSpec(mode=ActionRepresentationMode.JOINT_RELATIVE),
        )
        if joint_contract.transform.joint_topology.dim != 5:
            raise AssertionError("directory-based joint resolution failed")

        missing = root / "missing"
        missing.mkdir()
        try:
            resolve_action_representation_contract(missing, _eef_spec(PoseFormat.XYZ_RPY))
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("missing metadata must fail fast")
    print("PASS: dataset directory resolution for EEF and joint datasets")


CHECKS = (
    check_converter_metadata_parity,
    check_eef_contract_all_formats,
    check_eef_metadata_fail_fast,
    check_joint_contract,
    check_fingerprint_invalidation,
    check_dataset_directory_resolution,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for check in CHECKS:
        check()
    print(f"PASS: schema v2 dataset action contract ({len(CHECKS)} checks)")


if __name__ == "__main__":
    main()
