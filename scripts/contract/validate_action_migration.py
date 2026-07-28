#!/usr/bin/env python3
"""Phase 16 — legacy checkpoint migration과 checkpoint 계약 resolve 검증.

확인 항목:

- manifest 없는 checkpoint: **정확한 opt-in flag 없이는 거부**, 있으면 joint_absolute로 migration
- 명시적 v1 EEF-relative(xyz_rot6d_rows) migration
- **원본 checkpoint byte 불변**(tree hash 동일)
- migrated checkpoint를 다시 로드하면 schema v2 processor pair가 붙고
  ``validate_checkpoint_manifest``가 통과
- local 디렉터리 vs offline Hub snapshot/cache resolve parity
- in-place/비어 있지 않은 output/지원하지 않는 legacy/모호한 입력 fail-fast
- CLI assertion 불일치는 시작 전에 실패

실제 ``lerobot-train``으로 만든 checkpoint를 쓴다(합성 label 아님). Hub 업로드는 하지 않고,
로컬 캐시 layout을 만들어 offline snapshot resolve만 검증한다.

.. code-block:: bash

    python scripts/contract/validate_action_migration.py --fixture-root scratch/fx16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lerobot.policies import make_pre_post_processors  # noqa: E402

from so101_contract.action_checkpoint_contract import (  # noqa: E402
    resolve_checkpoint_contract,
)
from so101_contract.action_manifest import (  # noqa: E402
    ACTION_REPRESENTATION_MANIFEST,
    LEGACY_JOINT_ABSOLUTE_OPT_IN,
)
from so101_contract.action_migration import (  # noqa: E402
    ACTION_MIGRATION_VERSION,
    checkpoint_directory_sha256,
    detect_source_schema_state,
    migrate_checkpoint,
    plan_migration,
)
from so101_contract.action_representation import (  # noqa: E402
    ActionRepresentationMode,
    PoseFormat,
)
from so101_contract.lerobot_v2_integration import (  # noqa: E402
    has_action_representation_steps,
    validate_checkpoint_manifest,
)

HORIZON = 4


def _train_checkpoint(fixture: Path, workspace: Path, mode: str, pose_format: str | None) -> Path:
    output = workspace / f"train_{mode}"
    command = [
        "lerobot-train",
        f"--dataset.repo_id=local/{fixture.name}",
        f"--dataset.root={fixture}",
        "--policy.type=act",
        "--policy.device=cpu",
        "--policy.push_to_hub=false",
        f"--policy.chunk_size={HORIZON}",
        "--policy.n_action_steps=2",
        "--policy.dim_model=32",
        "--policy.n_heads=4",
        "--policy.dim_feedforward=64",
        "--policy.n_encoder_layers=1",
        "--policy.n_decoder_layers=1",
        "--policy.use_vae=false",
        "--policy.vision_backbone=resnet18",
        f"--policy.action_representation.mode={mode}",
        f"--output_dir={output}",
        "--steps=1",
        "--save_freq=1",
        "--batch_size=2",
        "--num_workers=0",
        "--log_freq=1",
        "--wandb.enable=false",
    ]
    if pose_format:
        command.append(f"--policy.action_representation.pose_format={pose_format}")
    environment = dict(os.environ)
    environment.setdefault("HF_HUB_OFFLINE", "1")
    result = subprocess.run(command, env=environment, capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join((result.stdout + result.stderr).splitlines()[-30:])
        raise RuntimeError(f"lerobot-train failed:\n{tail}")
    checkpoints = sorted((output / "checkpoints").glob("*/pretrained_model"))
    if not checkpoints:
        raise AssertionError("training produced no checkpoint")
    return checkpoints[-1]


def _manifest_less_copy(source: Path, destination: Path) -> Path:
    """manifest만 제거한 legacy 사본(그 외 파일은 그대로)."""
    shutil.copytree(source, destination)
    (destination / ACTION_REPRESENTATION_MANIFEST).unlink()
    return destination


def _v1_manifest_copy(source: Path, destination: Path) -> Path:
    """v1 EEF-relative manifest를 가진 legacy 사본.

    v1 patch를 되돌려 실제 v1 학습을 재현할 수는 없으므로, v1 manifest schema를 그대로
    쓰는 사본을 만들어 **schema dispatch 경로**를 검증한다. 이 경로는 v1 manifest를 신뢰하지
    않고 dataset/stats에서 v2 계약을 다시 만든다.
    """
    shutil.copytree(source, destination)
    v1_manifest = {
        "schema_version": "so101_eef_checkpoint_manifest_v1",
        "contract_schema_version": "so101_eef_action_contract_v1",
        "transform_version": "so101_eef_relative_se3_v1",
        "representation": {
            "mode": "eef_relative",
            "pose_format": "xyz_rot6d_rows",
            "state_pose_group": "eef_9d",
            "action_pose_group": "eef_9d",
        },
        "resolved_contract": {"fingerprint": "c" * 64},
        "relative_stats": {"profile_id": f"sha256:{'d' * 64}", "content_sha256": "d" * 64},
    }
    (destination / ACTION_REPRESENTATION_MANIFEST).write_text(
        json.dumps(v1_manifest, indent=2),
        encoding="utf-8",
    )
    return destination


def _reload_and_validate(checkpoint: Path) -> dict:
    from lerobot.configs.policies import PreTrainedConfig

    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = "cpu"
    preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=str(checkpoint))
    if not has_action_representation_steps(preprocessor, postprocessor):
        raise AssertionError(f"migrated checkpoint has no v2 processor pair: {checkpoint}")
    return validate_checkpoint_manifest(checkpoint, config, preprocessor)


def check_manifest_less_migration(joint_checkpoint: Path, dataset: Path, workspace: Path) -> None:
    """opt-in flag가 없으면 거부, 있으면 joint_absolute로 migration."""
    legacy = _manifest_less_copy(joint_checkpoint, workspace / "legacy_joint")
    if detect_source_schema_state(legacy) != "manifest_absent":
        raise AssertionError("manifest-less checkpoint was not detected")
    before = checkpoint_directory_sha256(legacy)

    try:
        plan_migration(
            legacy,
            workspace / "out_no_flag",
            dataset_root=dataset,
            horizon=HORIZON,
        )
    except PermissionError as exc:
        if LEGACY_JOINT_ABSOLUTE_OPT_IN not in str(exc):
            raise AssertionError("refusal message must name the exact opt-in flag") from exc
    else:
        raise AssertionError("manifest-less migration without the opt-in flag was accepted")

    plan = plan_migration(
        legacy,
        workspace / "out_joint",
        dataset_root=dataset,
        horizon=HORIZON,
        allow_legacy_joint_absolute=True,
    )
    result = migrate_checkpoint(plan)
    manifest = result.manifest
    if manifest["mode"] != "joint_absolute" or manifest["pose_format"] != "not_applicable":
        raise AssertionError(f"migrated representation is wrong: {manifest['mode']}")
    legacy_section = manifest["legacy"]
    if legacy_section.get("allowed") is not True:
        raise AssertionError("legacy opt-in was not recorded")
    if legacy_section.get("flag") != LEGACY_JOINT_ABSOLUTE_OPT_IN:
        raise AssertionError("legacy flag name was not recorded")
    migration = manifest["migration"]
    for key in ("version", "source_schema_state", "source_tree_sha256", "opt_in_flag"):
        if key not in migration:
            raise AssertionError(f"migration section is missing {key!r}")
    if migration["version"] != ACTION_MIGRATION_VERSION:
        raise AssertionError("migration version was not recorded")
    if migration["source_tree_sha256"] != before["tree_sha256"]:
        raise AssertionError("recorded source identity does not match the source")
    if migration["source_schema_state"] != "manifest_absent":
        raise AssertionError("source schema state was not recorded")

    after = checkpoint_directory_sha256(legacy)
    if after["tree_sha256"] != before["tree_sha256"]:
        raise AssertionError("source checkpoint was modified during migration")
    if (legacy / ACTION_REPRESENTATION_MANIFEST).exists():
        raise AssertionError("migration wrote a manifest into the source checkpoint")

    reloaded = _reload_and_validate(result.output)
    if reloaded["manifest_sha256"] != manifest["manifest_sha256"]:
        raise AssertionError("reloaded manifest differs from the migrated manifest")
    print("PASS: manifest-less joint_absolute migration (flag required, source unchanged, reload OK)")


def check_v1_eef_migration(eef_checkpoint: Path, dataset: Path, workspace: Path) -> None:
    """명시적 v1 EEF-relative migration."""
    legacy = _v1_manifest_copy(eef_checkpoint, workspace / "legacy_eef")
    if detect_source_schema_state(legacy) != "v1_eef_relative":
        raise AssertionError("v1 EEF-relative manifest was not detected")
    before = checkpoint_directory_sha256(legacy)

    try:
        plan_migration(
            legacy,
            workspace / "out_eef_bad",
            dataset_root=dataset,
            horizon=HORIZON,
            allow_legacy_joint_absolute=True,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("joint opt-in flag was accepted for a v1 EEF checkpoint")

    try:
        plan_migration(
            legacy,
            workspace / "out_eef_bad2",
            dataset_root=dataset,
            horizon=HORIZON,
            pose_format="xyz_rpy",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("conflicting explicit pose_format was accepted")

    plan = plan_migration(
        legacy,
        workspace / "out_eef",
        dataset_root=dataset,
        horizon=HORIZON,
        mode="eef_relative",
        pose_format="xyz_rot6d_rows",
    )
    result = migrate_checkpoint(plan)
    manifest = result.manifest
    if manifest["mode"] != "eef_relative" or manifest["pose_format"] != "xyz_rot6d_rows":
        raise AssertionError("v1 migration produced the wrong representation")
    if manifest["legacy"]["allowed"] is not False:
        raise AssertionError("v1 EEF migration must not claim a legacy joint opt-in")
    if manifest["migration"]["source_schema_state"] != "v1_eef_relative":
        raise AssertionError("v1 source state was not recorded")
    if manifest["kinematics"] is None:
        raise AssertionError("EEF manifest must carry the kinematics contract")
    if checkpoint_directory_sha256(legacy)["tree_sha256"] != before["tree_sha256"]:
        raise AssertionError("v1 source checkpoint was modified")

    contract = resolve_checkpoint_contract(result.output)
    if contract.spec.mode is not ActionRepresentationMode.EEF_RELATIVE:
        raise AssertionError("resolved contract mode mismatch")
    if contract.spec.pose_format is not PoseFormat.XYZ_ROT6D_ROWS:
        raise AssertionError("resolved contract pose format mismatch")
    if not contract.requires_ik:
        raise AssertionError("EEF contract must route through IK")
    _reload_and_validate(result.output)
    print("PASS: explicit v1 eef_relative migration (source unchanged, reload OK)")


def check_migration_guards(joint_checkpoint: Path, dataset: Path, workspace: Path) -> None:
    """in-place/비어 있지 않은 output/모호한 입력 거부."""
    legacy = _manifest_less_copy(joint_checkpoint, workspace / "legacy_guard")
    rejects = {
        "in-place output": lambda: plan_migration(
            legacy,
            legacy,
            dataset_root=dataset,
            horizon=HORIZON,
            allow_legacy_joint_absolute=True,
        ),
        "output inside source": lambda: plan_migration(
            legacy,
            legacy / "migrated",
            dataset_root=dataset,
            horizon=HORIZON,
            allow_legacy_joint_absolute=True,
        ),
        "non joint_absolute opt-in": lambda: plan_migration(
            legacy,
            workspace / "guard_a",
            dataset_root=dataset,
            horizon=HORIZON,
            mode="eef_relative",
            allow_legacy_joint_absolute=True,
        ),
        "pose format on joint": lambda: plan_migration(
            legacy,
            workspace / "guard_b",
            dataset_root=dataset,
            horizon=HORIZON,
            pose_format="xyz_rpy",
            allow_legacy_joint_absolute=True,
        ),
        "missing dataset": lambda: plan_migration(
            legacy,
            workspace / "guard_c",
            dataset_root=workspace / "no_such_dataset",
            horizon=HORIZON,
            allow_legacy_joint_absolute=True,
        ),
        "non-positive horizon": lambda: plan_migration(
            legacy,
            workspace / "guard_d",
            dataset_root=dataset,
            horizon=0,
            allow_legacy_joint_absolute=True,
        ),
        "already v2 source": lambda: plan_migration(
            joint_checkpoint,
            workspace / "guard_e",
            dataset_root=dataset,
            horizon=HORIZON,
            allow_legacy_joint_absolute=True,
        ),
    }
    for label, call in rejects.items():
        try:
            call()
        except (FileNotFoundError, PermissionError, ValueError) as exc:
            del exc
            continue
        raise AssertionError(f"invalid migration input was accepted: {label}")

    # 비어 있지 않은 임의 디렉터리는 덮어쓰지 않는다.
    occupied = workspace / "occupied"
    occupied.mkdir()
    (occupied / "important.bin").write_bytes(b"do not delete")
    plan = plan_migration(
        legacy,
        occupied,
        dataset_root=dataset,
        horizon=HORIZON,
        allow_legacy_joint_absolute=True,
    )
    try:
        migrate_checkpoint(plan)
    except FileExistsError:
        pass
    else:
        raise AssertionError("migration overwrote a non-empty output directory")
    if not (occupied / "important.bin").is_file():
        raise AssertionError("migration deleted unrelated output content")
    try:
        migrate_checkpoint(plan, overwrite=True)
    except FileExistsError:
        pass
    else:
        raise AssertionError("--overwrite must still refuse a non-migrated directory")

    # 지원하지 않는 legacy manifest는 명확히 실패한다.
    unsupported = workspace / "legacy_unsupported"
    shutil.copytree(legacy, unsupported)
    (unsupported / ACTION_REPRESENTATION_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": "so101_eef_checkpoint_manifest_v1",
                "representation": {"mode": "eef_relative", "pose_format": "xyz_rpy"},
            }
        ),
        encoding="utf-8",
    )
    try:
        detect_source_schema_state(unsupported)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("unsupported legacy manifest was accepted")
    print(f"PASS: {len(rejects) + 3} migration guards (no silent overwrite, no format guessing)")


def check_factory_load_gate(joint_checkpoint: Path, workspace: Path) -> None:
    """서버/추론의 실제 강제 지점: patch된 factory가 manifest 없는 checkpoint를 거부한다.

    entrypoint hook은 정적 CHECKPOINT_PATH만 보므로, 동적으로 선택된 모델의 방어선은
    factory의 ``validate_checkpoint_manifest``다. 이 검사는 그 경로를 실제로 실행한다.
    """
    from lerobot.configs.policies import PreTrainedConfig

    legacy = _manifest_less_copy(joint_checkpoint, workspace / "factory_gate")
    config = PreTrainedConfig.from_pretrained(legacy)
    config.device = "cpu"
    # patch된 factory가 processor 구성 도중 manifest를 강제한다. 예외는 여기서 난다.
    try:
        make_pre_post_processors(config, pretrained_path=str(legacy))
    except FileNotFoundError as exc:
        if "migration" not in str(exc).lower():
            raise AssertionError(
                f"factory gate must point at the migration tool: {exc}"
            ) from exc
    else:
        raise AssertionError("factory accepted a checkpoint without a schema v2 manifest")

    # 정상 v2 checkpoint는 통과한다.
    manifest = _reload_and_validate(joint_checkpoint)
    if manifest["mode"] != "joint_absolute":
        raise AssertionError("valid v2 checkpoint did not pass the factory gate")
    print("PASS: patched factory rejects manifest-less checkpoints and accepts valid v2 ones")


def check_contract_resolution_parity(joint_checkpoint: Path, workspace: Path) -> None:
    """local 디렉터리 vs offline Hub snapshot resolve parity + CLI assertion."""
    local = resolve_checkpoint_contract(joint_checkpoint)

    # 업로드 없이 HF 캐시 layout만 만들어 offline snapshot resolve를 검증한다.
    cache = workspace / "hf_cache"
    repo_id = "so101-test/joint-absolute"
    revision = "e" * 40
    repo_dir = cache / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / revision
    blobs = repo_dir / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir(parents=True)
    payload = (joint_checkpoint / ACTION_REPRESENTATION_MANIFEST).read_bytes()
    blob_name = hashlib.sha1(payload).hexdigest()
    (blobs / blob_name).write_bytes(payload)
    (snapshot / ACTION_REPRESENTATION_MANIFEST).symlink_to(blobs / blob_name)
    refs = repo_dir / "refs"
    refs.mkdir(parents=True)
    (refs / "main").write_text(revision, encoding="utf-8")

    hub = resolve_checkpoint_contract(
        repo_id,
        revision=revision,
        local_files_only=True,
        cache_dir=cache,
    )

    if hub.spec != local.spec:
        raise AssertionError("local and Hub-snapshot contracts resolved different representations")
    if hub.manifest_sha256 != local.manifest_sha256:
        raise AssertionError("local and Hub-snapshot manifest hashes differ")
    if hub.action_dim != local.action_dim or hub.routing != local.routing:
        raise AssertionError("local and Hub-snapshot routing contracts differ")

    # CLI assertion: 생략은 수용, 불일치는 시작 전 실패.
    local.assert_cli()
    local.assert_cli(mode="joint_absolute", policy_type="act")
    mismatches = {
        "mode": lambda: local.assert_cli(mode="eef_relative"),
        "pose format": lambda: local.assert_cli(pose_format="xyz_rot6d_rows"),
        "policy family": lambda: local.assert_cli(policy_type="groot"),
    }
    for label, call in mismatches.items():
        try:
            call()
        except ValueError:
            continue
        raise AssertionError(f"CLI assertion mismatch was accepted: {label}")
    print("PASS: local vs offline Hub-snapshot parity and CLI assertions (no upload performed)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    scratch = ROOT / "scratch"
    scratch.mkdir(exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="migration-", dir=scratch))
    try:
        joint_dataset = (args.fixture_root / "joint").resolve()
        eef_dataset = (args.fixture_root / "xyz_rot6d_rows").resolve()
        joint_checkpoint = _train_checkpoint(joint_dataset, workspace, "joint_absolute", None)
        eef_checkpoint = _train_checkpoint(
            eef_dataset,
            workspace,
            "eef_relative",
            "xyz_rot6d_rows",
        )
        check_manifest_less_migration(joint_checkpoint, joint_dataset, workspace)
        check_v1_eef_migration(eef_checkpoint, eef_dataset, workspace)
        check_migration_guards(joint_checkpoint, joint_dataset, workspace)
        check_factory_load_gate(joint_checkpoint, workspace)
        check_contract_resolution_parity(joint_checkpoint, workspace)
        print("PASS: action representation migration and checkpoint contract (5 checks)")
        return 0
    finally:
        if not args.keep:
            shutil.rmtree(workspace, ignore_errors=True)
        else:
            print(json.dumps({"workspace": str(workspace)}))


if __name__ == "__main__":
    raise SystemExit(main())
