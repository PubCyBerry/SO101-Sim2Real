#!/usr/bin/env python3
"""Phase 17 — 3 policy × 8 representation = 24 조합 offline validation matrix (§25.2).

24개 조합 **각각**에 대해 §25.2의 13개 필수 검증을 실제 LeRobot policy/processor/
SyncInferenceEngine/PolicyServer로 실행하고, 결과를 machine-readable JSON으로 적는다.
대표 조합 결과를 다른 조합에 복제하지 않는다(조합별로 config/processor/manifest/stats/
checkpoint를 새로 만들고 다시 읽는다).

13개 named check (조합마다 전부 존재해야 하며, 누락은 전체 실패):

1.  ``config_serialization_roundtrip``
2.  ``manifest_generate_hash_load_tamper``
3.  ``dataset_metadata``
4.  ``stats_generate_and_checkpoint_persistence``
5.  ``one_batch_forward_backward``
6.  ``checkpoint_save_reload_parity``
7.  ``full_chunk_inference``
8.  ``sync_and_async_inference``
9.  ``processor_ordering``
10. ``platform_adapter_routing``
11. ``fk_ik_roundtrip_or_joint_command_path``
12. ``legacy_checkpoint_migration``
13. ``invalid_cli_checkpoint_fail_fast``

weight 폭증 회피 전략: ACT는 조합마다 dimension이 달라 실제 ``save_pretrained`` /
``from_pretrained``를 그대로 쓴다(작다). SmolVLA/GR00T는 padding capacity 덕분에 8개
representation이 같은 weight shape를 쓰므로 **weight 파일을 family당 한 번만** 저장하고
조합 디렉터리에는 **hardlink**로 dedup한다. config/processor/manifest/stats는 조합마다
따로 저장·재로드하므로 조합별 계약 reload를 축약하지 않는다. 임시 산출물은 종료 시 지운다.

.. code-block:: bash

    python scripts/contract/validate_action_representation_matrix.py \\
        --fixture-root scratch/p17-baseline \\
        --output scratch/p17-matrix/phase17_24combo.json
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy as _copy
import datetime as _dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import validate_action_representation_policies as p15  # noqa: E402
import validate_action_routing as p16r  # noqa: E402

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies import make_pre_post_processors  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402

from so101_contract.action_checkpoint_contract import (  # noqa: E402
    assert_checkpoint_representation,
    resolve_checkpoint_contract,
)
from so101_contract.action_dataset_contract import (  # noqa: E402
    resolve_action_representation_contract,
)
from so101_contract.action_manifest import (  # noqa: E402
    ACTION_REPRESENTATION_MANIFEST,
    LEGACY_JOINT_ABSOLUTE_OPT_IN,
    canonical_manifest_sha256,
    read_action_representation_manifest,
    validate_action_representation_manifest,
    write_action_representation_manifest,
)
from so101_contract.action_migration import (  # noqa: E402
    checkpoint_directory_sha256,
    detect_source_schema_state,
    migrate_checkpoint,
    plan_migration,
)
from so101_contract.action_representation import (  # noqa: E402
    ActionRepresentationMode,
    ActionRepresentationSpec,
    PoseFormat,
    combination_id,
)
from so101_contract.action_representation_stats import (  # noqa: E402
    ActionStatsSampling,
    calculate_action_representation_stats,
    empty_stats_artifact,
    read_action_stats_artifact,
    restore_stats_from_processor,
    select_stats_profile,
    serialize_stats_for_processor,
    upsert_stats_profile,
    write_action_stats_artifact,
)
from so101_contract.action_routing import make_router  # noqa: E402
from so101_contract.eef_relative_action import (  # noqa: E402
    absolute_actions_to_relative as v1_absolute_actions_to_relative,
)
from so101_contract.joint_feature_codec import sim_publish_command  # noqa: E402
from so101_contract.follower_calibration import (  # noqa: E402
    sim_radians_to_real_follower,
)
from so101_contract.lerobot_full_chunk import FullChunkPolicyRunner  # noqa: E402
from so101_contract.lerobot_v2_integration import (  # noqa: E402
    action_representation_encode_step,
    build_manifest_from_processor,
    has_action_representation_steps,
    load_pretrained_with_selective_reuse,
    processor_step_order,
    validate_checkpoint_manifest,
)
from so101_contract.pose_codec import (  # noqa: E402
    convert_pose_format,
    rotation_geodesic_angle,
)

SCHEMA_VERSION = "so101_action_representation_matrix_v1"
HORIZON = p15.HORIZON
IMAGE_KEY = p15.IMAGE_KEY
EXECUTION_HORIZON = 2
POLICY_FAMILIES = ("act", "smolvla", "groot")

#: §25.2의 13개 필수 check. 순서/이름 고정이며 누락은 전체 실패다.
CHECK_NAMES = (
    "config_serialization_roundtrip",
    "manifest_generate_hash_load_tamper",
    "dataset_metadata",
    "stats_generate_and_checkpoint_persistence",
    "one_batch_forward_backward",
    "checkpoint_save_reload_parity",
    "full_chunk_inference",
    "sync_and_async_inference",
    "processor_ordering",
    "platform_adapter_routing",
    "fk_ik_roundtrip_or_joint_command_path",
    "legacy_checkpoint_migration",
    "invalid_cli_checkpoint_fail_fast",
)

#: Phase 8 offline sweep 기반 허용치.
TOL_POSITION_M = 5.0e-3
TOL_ROTATION_RAD = 5.0e-2
TOL_V1_PARITY = 3.0e-5


# --- 결과 컨테이너 -------------------------------------------------------------


class CheckRecorder:
    """조합 하나의 13개 check 결과를 모은다."""

    def __init__(self, combination: str) -> None:
        self.combination = combination
        self.checks: dict[str, dict] = {}

    @contextmanager
    def run(self, name: str):
        if name not in CHECK_NAMES:
            raise KeyError(f"unknown check name: {name}")
        evidence: dict = {}
        started = time.perf_counter()
        try:
            yield evidence
        except Exception as exc:  # noqa: BLE001 - check별 실패를 기록하고 계속한다
            self.checks[name] = {
                "status": "fail",
                "evidence": evidence,
                "duration_s": round(time.perf_counter() - started, 4),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=12),
            }
            print(f"    FAIL {self.combination}/{name}: {type(exc).__name__}: {exc}")
        else:
            self.checks[name] = {
                "status": "pass",
                "evidence": evidence,
                "duration_s": round(time.perf_counter() - started, 4),
            }

    def missing(self) -> list[str]:
        return [name for name in CHECK_NAMES if name not in self.checks]

    def to_dict(self) -> dict:
        missing = self.missing()
        for name in missing:
            self.checks[name] = {
                "status": "fail",
                "evidence": {},
                "duration_s": 0.0,
                "error": "check did not run (missing check is a combination failure)",
            }
        ordered = {name: self.checks[name] for name in CHECK_NAMES}
        failed = [name for name, value in ordered.items() if value["status"] == "fail"]
        skipped = [name for name, value in ordered.items() if value["status"] == "skip"]
        return {
            "combination": self.combination,
            "status": "fail" if failed else "pass",
            "missing_checks": missing,
            "failed_checks": failed,
            "skipped_checks": skipped,
            "checks": ordered,
        }


def _expect_raises(callable_, exceptions, description: str):
    """`callable_`이 반드시 실패해야 하는 guard. 성공하면 AssertionError."""
    try:
        callable_()
    except exceptions as exc:
        return f"{type(exc).__name__}: {exc}"
    raise AssertionError(f"expected guard did not trigger: {description}")


# --- 조합 재료 ----------------------------------------------------------------


def _representation_config(spec):
    return p15._representation_config(spec)


def _dataset_features(contract) -> dict:
    return {
        ACTION: {
            "dtype": "float32",
            "shape": (contract.action_dim,),
            "names": list(contract.action_names),
        }
    }


def _sampling() -> ActionStatsSampling:
    return ActionStatsSampling(
        observation_delta_indices=(0,),
        action_delta_indices=tuple(range(HORIZON)),
    )


def _raw_observation(contract, state: np.ndarray, *, image_hw=(256, 256)) -> dict:
    """PolicyServer가 받는 raw observation(feature 이름 → 값).

    state는 fixture의 실제 값이어야 한다. 0 벡터는 rot6d/quaternion pose로 유효하지 않다.
    """
    values = np.asarray(state, dtype=np.float64).reshape(-1)
    if values.shape[0] != len(contract.action_names):
        raise ValueError(
            f"raw observation state dim mismatch: {values.shape[0]} != {len(contract.action_names)}"
        )
    raw = {name: float(values[index]) for index, name in enumerate(contract.action_names)}
    raw["top"] = np.zeros((*image_hw, 3), dtype=np.uint8)
    raw["task"] = "phase17 fixture"
    return raw


# --- checkpoint 저장/재로드 ----------------------------------------------------


def _link_weight_files(shared: Path, destination: Path) -> list[str]:
    """weight 파일만 hardlink(불가하면 symlink)로 dedup한다."""
    linked: list[str] = []
    for path in sorted(shared.iterdir()):
        if not path.is_file():
            continue
        if path.suffix not in (".safetensors",) and not path.name.endswith(".index.json"):
            continue
        target = destination / path.name
        if target.exists():
            target.unlink()
        try:
            os.link(path, target)
        except OSError:
            target.symlink_to(path)
        linked.append(path.name)
    if not linked:
        raise AssertionError(f"no weight files found to link from {shared}")
    return linked


def _save_combination_checkpoint(
    directory: Path,
    *,
    family: str,
    model,
    config,
    preprocessor,
    postprocessor,
    manifest: dict,
    shared_weights: dict[str, Path],
) -> dict:
    """조합 checkpoint 디렉터리를 만든다.

    ACT는 실제 ``save_pretrained``(작음). SmolVLA/GR00T는 weight를 family당 한 번만 쓰고
    조합 디렉터리에는 hardlink로 dedup하되 config/processor/manifest는 조합마다 새로 쓴다.
    """
    directory.mkdir(parents=True, exist_ok=True)
    info: dict = {"weight_strategy": "per_combination_save_pretrained"}
    if family == "act":
        model.save_pretrained(directory)
    else:
        shared = shared_weights.get(family)
        if shared is None:
            shared = directory.parent / f"_shared_weights_{family}"
            shared.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(shared)
            shared_weights[family] = shared
        info["weight_strategy"] = "shared_deduplicated_hardlink"
        info["shared_weights_dir"] = str(shared)
        info["linked_weight_files"] = _link_weight_files(shared, directory)
        config.save_pretrained(directory)
    preprocessor.save_pretrained(directory)
    postprocessor.save_pretrained(directory)
    write_action_representation_manifest(directory, manifest)
    info["files"] = sorted(path.name for path in directory.iterdir() if path.is_file())
    return info


def _reload_checkpoint(directory: Path, family: str, device: str, model):
    """조합 checkpoint를 실제로 다시 읽는다.

    config/processor/manifest는 항상 disk에서 재구성한다. weight는 ACT면 새 policy를
    from_pretrained로 만들고, SmolVLA/GR00T는 조합 디렉터리의 (dedup된) safetensors를
    strict하게 다시 로드해 in-memory state를 교체한다.
    """
    config = PreTrainedConfig.from_pretrained(directory)
    config.device = device
    preprocessor, postprocessor = make_pre_post_processors(
        config, pretrained_path=str(directory)
    )
    if family == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy

        reloaded = ACTPolicy.from_pretrained(directory)
        reloaded.to(device)
        reuse = None
    else:
        weight_file = directory / "model.safetensors"
        if not weight_file.is_file():
            index = directory / "model.safetensors.index.json"
            if not index.is_file():
                raise FileNotFoundError(f"no safetensors weights in {directory}")
            weight_file = index
        # strict=True. tying된 shared tensor는 safetensors 저장이 중복을 버리는 것이므로
        # 누락이 아니라 ``tied``로 분류된다(loader 계약).
        reuse = load_pretrained_with_selective_reuse(
            model, weight_file, device="cpu", strict=True
        )
        reloaded = model
    return config, preprocessor, postprocessor, reloaded, reuse


# --- policy 구성 ---------------------------------------------------------------


def _build_policy(family: str, spec, bundle, *, cached_model: dict, device: str):
    """Phase 15와 같은 실제 policy/processor를 만든다(모델 mock 없음)."""
    from lerobot.configs import FeatureType, PolicyFeature

    contract = bundle["contract"]
    dim = contract.action_dim
    image_shape = (3, 256, 256)
    config_features = {
        "input": {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(dim,)),
            IMAGE_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=image_shape),
        },
        "output": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(dim,))},
    }
    if family == "act":
        from lerobot.configs import NormalizationMode
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy

        config = ACTConfig(
            chunk_size=HORIZON,
            n_action_steps=EXECUTION_HORIZON,
            input_features=config_features["input"],
            output_features=config_features["output"],
            normalization_mapping={
                "STATE": NormalizationMode.MEAN_STD,
                "VISUAL": NormalizationMode.IDENTITY,
                "ACTION": NormalizationMode.MEAN_STD,
            },
            device=device,
            dim_model=32,
            n_heads=4,
            dim_feedforward=64,
            n_encoder_layers=1,
            n_decoder_layers=1,
            use_vae=False,
            dropout=0.0,
            vision_backbone="resnet18",
            pretrained_backbone_weights=None,
            action_representation=_representation_config(spec),
        )
        preprocessor, postprocessor = make_pre_post_processors(
            config,
            dataset_stats=bundle["dataset_stats"],
            dataset_meta=bundle["dataset_meta"],
        )
        model = ACTPolicy(config).to(device)
    elif family == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        config = PreTrainedConfig.from_pretrained(
            p15.SMOLVLA_MODEL_ID, local_files_only=True
        )
        config.input_features = config_features["input"]
        config.output_features = config_features["output"]
        config.chunk_size = HORIZON
        config.n_action_steps = EXECUTION_HORIZON
        config.device = device
        config.load_vlm_weights = False
        config.action_representation = _representation_config(spec)
        preprocessor, postprocessor = make_pre_post_processors(
            config,
            dataset_stats=bundle["dataset_stats"],
            dataset_meta=bundle["dataset_meta"],
        )
        if cached_model.get("smolvla") is None:
            cached_model["smolvla"] = SmolVLAPolicy(config).to(device)
        model = cached_model["smolvla"]
        model.config = config
    else:
        from lerobot.policies.groot.configuration_groot import GrootConfig
        from lerobot.policies.groot.modeling_groot import GrootPolicy

        config = GrootConfig(
            base_model_path=p15.GROOT_MODEL_ID,
            embodiment_tag="new_embodiment",
            chunk_size=HORIZON,
            n_action_steps=EXECUTION_HORIZON,
            input_features=config_features["input"],
            output_features=config_features["output"],
            device=device,
            use_bf16=device == "cuda",
            model_params_fp32=device != "cuda",
            tune_llm=False,
            tune_visual=False,
            tune_projector=True,
            tune_diffusion_model=True,
            tune_vlln=True,
            num_inference_timesteps=1,
            use_relative_actions=False,
            action_representation=_representation_config(spec),
        )
        preprocessor, postprocessor = make_pre_post_processors(
            config,
            dataset_stats=bundle["dataset_stats"],
            dataset_meta=bundle["dataset_meta"],
        )
        if cached_model.get("groot") is None:
            cached_model["groot"] = GrootPolicy(config).to(device)
        model = cached_model["groot"]
        model.config = config
    model.train()
    return config, model, preprocessor, postprocessor, image_shape


# --- 개별 check ---------------------------------------------------------------


def _check_config_roundtrip(evidence, directory: Path, config, spec) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(directory)
    reloaded = PreTrainedConfig.from_pretrained(directory)
    representation = reloaded.action_representation
    if representation.mode != spec.mode.value:
        raise AssertionError(
            f"config round-trip changed the mode: {representation.mode!r} != {spec.mode.value!r}"
        )
    expected_format = (
        None if spec.pose_format is PoseFormat.NOT_APPLICABLE else spec.pose_format.value
    )
    if (representation.pose_format or None) != expected_format:
        raise AssertionError(
            "config round-trip changed the pose format: "
            f"{representation.pose_format!r} != {expected_format!r}"
        )
    if not spec.is_eef and representation.pose_format not in (None, "", "not_applicable"):
        raise AssertionError(
            f"joint mode must not carry a pose format: {representation.pose_format!r}"
        )
    spec_payload = json.loads(json.dumps(spec.to_dict(), sort_keys=True))
    if ActionRepresentationSpec.from_dict(spec_payload) != spec:
        raise AssertionError("ActionRepresentationSpec JSON round-trip is not identity")
    evidence.update(
        {
            "config_file": str(directory / "config.json"),
            "mode": representation.mode,
            "pose_format": representation.pose_format,
            "spec_json_roundtrip": "identity",
        }
    )


def _check_manifest(evidence, directory: Path, config, preprocessor, model, spec, family, contract):
    manifest = build_manifest_from_processor(config, preprocessor, policy=model)
    validate_action_representation_manifest(
        manifest, expected_spec=spec, expected_policy_type=family
    )
    if manifest["resolved_contract_fingerprint"] != contract.fingerprint:
        raise AssertionError("manifest fingerprint does not match the dataset contract")

    scratch = directory / "_manifest_probe"
    scratch.mkdir(parents=True, exist_ok=True)
    write_action_representation_manifest(scratch, manifest)
    loaded = read_action_representation_manifest(scratch)
    if loaded is None:
        raise AssertionError("manifest was not readable after write")
    if loaded["manifest_sha256"] != manifest["manifest_sha256"]:
        raise AssertionError("manifest hash changed across write/load")
    payload = {k: v for k, v in loaded.items() if k != "manifest_sha256"}
    if canonical_manifest_sha256(payload) != loaded["manifest_sha256"]:
        raise AssertionError("manifest sha256 does not cover its own payload")

    tampered = _copy.deepcopy(loaded)
    tampered["action_horizon"] = int(tampered["action_horizon"]) + 1
    (scratch / ACTION_REPRESENTATION_MANIFEST).write_text(
        json.dumps(tampered, indent=2), encoding="utf-8"
    )
    tamper_error = _expect_raises(
        lambda: resolve_checkpoint_contract(scratch, expected_policy_type=family),
        (ValueError, KeyError, TypeError),
        "tampered manifest must be rejected",
    )
    shutil.rmtree(scratch, ignore_errors=True)
    evidence.update(
        {
            "manifest_sha256": manifest["manifest_sha256"],
            "hash_stable_across_write_load": True,
            "tamper_rejected_with": tamper_error,
        }
    )
    return manifest


def _check_dataset_metadata(evidence, fixture_root: Path, spec, contract) -> None:
    again = resolve_action_representation_contract(fixture_root, spec)
    if again.fingerprint != contract.fingerprint:
        raise AssertionError("dataset contract resolve is not deterministic")
    if len(contract.action_names) != contract.action_dim:
        raise AssertionError("action names do not cover the action dimension")
    if spec.is_eef:
        expected_dim = spec.pose_dim + 1
    else:
        expected_dim = 6
    if contract.action_dim != expected_dim:
        raise AssertionError(
            f"dataset action dim {contract.action_dim} != contract expectation {expected_dim}"
        )
    if spec.action_group not in contract.action_groups:
        raise AssertionError(f"dataset lacks transform group {spec.action_group!r}")

    # 같은 dataset에 맞지 않는 representation은 명시적으로 거부돼야 한다.
    if spec.is_eef:
        other = PoseFormat.XYZ_RPY if spec.pose_format is not PoseFormat.XYZ_RPY else PoseFormat.XYZ_ROT6D_ROWS
        wrong = ActionRepresentationSpec(mode=spec.mode, pose_format=other)
    else:
        wrong = ActionRepresentationSpec(
            mode=ActionRepresentationMode.EEF_ABSOLUTE,
            pose_format=PoseFormat.XYZ_ROT6D_ROWS,
        )
    mismatch = _expect_raises(
        lambda: resolve_action_representation_contract(fixture_root, wrong),
        (ValueError, KeyError, TypeError),
        "dataset must reject a representation it does not store",
    )
    evidence.update(
        {
            "dataset_root": str(fixture_root),
            "fingerprint": contract.fingerprint,
            "action_dim": contract.action_dim,
            "groups": {k: list(v) for k, v in contract.action_groups.items()},
            "mismatched_representation_rejected_with": mismatch,
        }
    )


def _check_stats(evidence, directory: Path, spec, contract, bundle, preprocessor) -> None:
    """조합별로 stats를 다시 계산하고, 파일과 checkpoint processor 양쪽에서 복원한다."""
    transform = contract.transform
    sampling = _sampling()
    result = calculate_action_representation_stats(
        bundle["episodes"],
        sampling,
        transform,
        dataset_fingerprint=contract.fingerprint,
    )
    artifact, changed = upsert_stats_profile(empty_stats_artifact(), result)
    if not changed:
        raise AssertionError("stats profile was not inserted into a fresh artifact")

    stats_root = directory / "_stats"
    stats_root.mkdir(parents=True, exist_ok=True)
    written = write_action_stats_artifact(stats_root, artifact)
    reloaded = read_action_stats_artifact(stats_root)
    profile_id, profile = select_stats_profile(
        reloaded, transform, sampling, dataset_fingerprint=contract.fingerprint
    )
    if profile_id != result.profile_id:
        raise AssertionError("stats profile id changed across write/read")
    if profile != result.profile:
        raise AssertionError("stats profile content changed across write/read")

    fixture_artifact = read_action_stats_artifact(bundle["fixture_root"])
    _, fixture_profile = select_stats_profile(
        fixture_artifact, transform, sampling, dataset_fingerprint=contract.fingerprint
    )
    if fixture_profile != result.profile:
        raise AssertionError(
            "regenerated stats disagree with the fixture profile stored in the dataset"
        )

    # checkpoint persistence/restore: processor에 실리는 형태로 직렬화했다가 복원한다.
    serialized = serialize_stats_for_processor(profile)
    restored_id, restored = restore_stats_from_processor(json.loads(json.dumps(serialized)))
    if restored_id != profile_id:
        raise AssertionError("processor stats restore lost the profile id")
    if restored != profile:
        raise AssertionError("processor stats restore changed the profile content")

    step = action_representation_encode_step(preprocessor)
    if step is None:
        raise AssertionError("preprocessor has no v2 encode step to carry stats")
    context_stats = dict(step.manifest_context.get("stats") or {})
    if context_stats.get("profile_id") != profile_id:
        raise AssertionError(
            "checkpoint processor stats do not match the regenerated profile: "
            f"{context_stats.get('profile_id')} != {profile_id}"
        )
    evidence.update(
        {
            "artifact_file": str(written),
            "profile_id": profile_id,
            "content_sha256": profile_id.removeprefix("sha256:"),
            "kind": profile["kind"],
            "horizon": sampling.horizon,
            "matches_fixture_profile": True,
            "processor_restore": "identity",
        }
    )
    shutil.rmtree(stats_root, ignore_errors=True)


def _check_forward_backward(evidence, model, preprocessor, bundle, image_shape, family, spec):
    batch = int(bundle["state"].shape[0])
    device = next(model.parameters()).device
    images = torch.zeros(batch, *image_shape, device=device)
    raw_batch = {
        OBS_STATE: bundle["state"].clone().to(device),
        IMAGE_KEY: images,
        ACTION: bundle["action"].clone().to(device),
        "action_is_pad": torch.zeros(batch, HORIZON, dtype=torch.bool, device=device),
        "task": ["phase17 fixture"] * batch,
    }
    processed = preprocessor(raw_batch)
    if tuple(processed[ACTION].shape[:2]) != (batch, HORIZON):
        raise AssertionError(f"processed action chunk shape mismatch: {processed[ACTION].shape}")
    model.train()
    loss, _ = model(processed)
    if not torch.isfinite(loss):
        raise AssertionError(f"{family}/{spec.stats_profile_kind} loss is not finite")
    model.zero_grad(set_to_none=True)
    loss.backward()
    finite_grads = sum(
        1
        for parameter in model.parameters()
        if parameter.grad is not None and torch.all(torch.isfinite(parameter.grad))
    )
    if finite_grads == 0:
        raise AssertionError("no finite gradients were produced")
    model.zero_grad(set_to_none=True)
    evidence.update(
        {
            "loss": float(loss.detach().cpu()),
            "finite_grad_tensors": finite_grads,
            "batch": batch,
            "horizon": HORIZON,
        }
    )
    return images


def _predict_chunk(model, preprocessor, observation) -> torch.Tensor:
    model.eval()
    with torch.inference_mode():
        processed = preprocessor(dict(observation))
        chunk = model.predict_action_chunk(processed)
    return chunk.detach().float().cpu().clone()


def _check_reload_parity(
    evidence,
    directory: Path,
    *,
    family,
    spec,
    model,
    config,
    preprocessor,
    postprocessor,
    manifest,
    observation,
    shared_weights,
    device,
):
    save_info = _save_combination_checkpoint(
        directory,
        family=family,
        model=model,
        config=config,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        manifest=manifest,
        shared_weights=shared_weights,
    )
    torch.manual_seed(0)
    before = _predict_chunk(model, preprocessor, observation)

    reloaded_config, reloaded_pre, reloaded_post, reloaded_model, reuse = _reload_checkpoint(
        directory, family, device, model
    )
    if not has_action_representation_steps(reloaded_pre, reloaded_post):
        raise AssertionError("reloaded checkpoint has no v2 processor pair")
    reloaded_manifest = validate_checkpoint_manifest(directory, reloaded_config, reloaded_pre)
    if reloaded_manifest["manifest_sha256"] != manifest["manifest_sha256"]:
        raise AssertionError("manifest identity changed across checkpoint reload")
    if reloaded_config.action_representation.mode != spec.mode.value:
        raise AssertionError("reloaded config lost the representation mode")
    if processor_step_order(reloaded_pre) != processor_step_order(preprocessor):
        raise AssertionError("reloaded preprocessor ordering differs")
    if processor_step_order(reloaded_post) != processor_step_order(postprocessor):
        raise AssertionError("reloaded postprocessor ordering differs")

    torch.manual_seed(0)
    after = _predict_chunk(reloaded_model, reloaded_pre, observation)
    if before.shape != after.shape:
        raise AssertionError(f"reload changed the chunk shape: {before.shape} != {after.shape}")
    delta = float(torch.max(torch.abs(before - after)))
    if not np.isfinite(delta) or delta > 1e-4:
        raise AssertionError(f"model output parity broke after reload: max|Δ|={delta}")
    evidence.update(
        {
            "checkpoint_dir": str(directory),
            "manifest_sha256": reloaded_manifest["manifest_sha256"],
            "max_abs_output_delta": delta,
            "weight_reload": save_info["weight_strategy"],
            "reused_parameter_count": None if reuse is None else len(reuse.reused),
            "reinitialized_layers": []
            if reuse is None
            else [key for key, _, _ in reuse.reinitialized],
            "tied_shared_tensor_keys": [] if reuse is None else list(reuse.tied),
            "files": save_info["files"],
        }
    )
    return reloaded_config, reloaded_pre, reloaded_post, reloaded_model


def _check_full_chunk(evidence, model, preprocessor, postprocessor, observation, contract):
    runner = FullChunkPolicyRunner(
        model, preprocessor, postprocessor, execution_horizon=EXECUTION_HORIZON
    )
    with torch.inference_mode():
        first = runner.next_action(dict(observation))
        second = runner.next_action(dict(observation))
    if tuple(first.shape) != (1, contract.action_dim):
        raise AssertionError(f"full-chunk action shape mismatch: {tuple(first.shape)}")
    metrics = runner.metrics
    if metrics.preprocessor_calls != 1 or metrics.postprocessor_calls != 1:
        raise AssertionError(f"full chunk was reprocessed per tick: {metrics}")
    if metrics.actions_dequeued != 2:
        raise AssertionError(f"execution horizon not honored: {metrics}")
    absolute = torch.cat([first, second]).detach().float().cpu().numpy()
    if not np.all(np.isfinite(absolute)):
        raise AssertionError("full-chunk absolute actions contain non-finite values")
    evidence.update(
        {
            "preprocessor_calls": metrics.preprocessor_calls,
            "postprocessor_calls": metrics.postprocessor_calls,
            "actions_dequeued": metrics.actions_dequeued,
            "action_shape": list(first.shape),
        }
    )
    runner.reset()
    return absolute


class _CountingPipeline:
    """실제 pipeline을 감싸 호출 횟수/입력 shape만 센다(동작은 그대로)."""

    def __init__(self, wrapped) -> None:
        self.wrapped = wrapped
        self.calls: list[tuple] = []

    def __call__(self, value):
        shape = tuple(value.shape) if hasattr(value, "shape") else None
        self.calls.append(shape)
        return self.wrapped(value)

    def __getattr__(self, name):
        return getattr(self.wrapped, name)


def _check_sync_and_async(
    evidence,
    *,
    directory: Path,
    family,
    contract,
    model,
    preprocessor,
    postprocessor,
    device,
    state,
):
    """실제 SyncInferenceEngine + 실제 PolicyServer async 경로."""
    from lerobot.rollout.inference.sync import SyncInferenceEngine

    counting_pre = _CountingPipeline(preprocessor)
    counting_post = _CountingPipeline(postprocessor)
    engine = SyncInferenceEngine(
        policy=model,
        preprocessor=counting_pre,
        postprocessor=counting_post,
        dataset_features=_dataset_features(contract),
        ordered_action_keys=list(contract.action_names),
        task="phase17 fixture",
        device=device,
        robot_type="so101_follower",
    )
    if engine._full_chunk_runner is None:
        raise AssertionError("sync engine did not activate the v2 full-chunk runner")
    engine.start()
    obs_frame = {
        OBS_STATE: np.asarray(state, dtype=np.float32).reshape(-1),
        IMAGE_KEY: np.zeros((256, 256, 3), dtype=np.uint8),
    }
    first = engine.get_action(dict(obs_frame))
    second = engine.get_action(dict(obs_frame))
    engine.stop()
    if first is None or second is None:
        raise AssertionError("sync engine returned no action")
    if len(counting_pre.calls) != 1 or len(counting_post.calls) != 1:
        raise AssertionError(
            "sync engine must postprocess the full chunk once per chunk: "
            f"pre={counting_pre.calls}, post={counting_post.calls}"
        )
    post_shape = counting_post.calls[0]
    if post_shape is None or len(post_shape) != 3 or post_shape[2] != contract.action_dim:
        raise AssertionError(f"sync postprocessor did not receive a full chunk: {post_shape}")
    if tuple(first.shape) != (contract.action_dim,):
        raise AssertionError(f"sync action shape mismatch: {tuple(first.shape)}")

    async_evidence = _run_async_server(
        directory=directory, family=family, contract=contract, device=device, state=state
    )
    evidence.update(
        {
            "sync": {
                "preprocessor_calls": len(counting_pre.calls),
                "postprocessor_calls": len(counting_post.calls),
                "full_chunk_shape": list(post_shape),
                "actions_dequeued": 2,
            },
            "async": async_evidence,
        }
    )


def _run_async_server(*, directory: Path, family, contract, device, state) -> dict:
    import pickle

    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation
    from lerobot.async_inference.policy_server import PolicyServer
    from lerobot.transport import services_pb2

    class _LocalContext:
        @staticmethod
        def peer() -> str:
            return "phase17-matrix"

    server = PolicyServer(
        PolicyServerConfig(
            host="127.0.0.1",
            port=8080,
            fps=30,
            inference_latency=0.0,
            obs_queue_timeout=1.0,
        )
    )
    context = _LocalContext()
    server.Ready(services_pb2.Empty(), context)
    observation_features = {
        OBS_STATE: {
            "dtype": "float32",
            "shape": (contract.action_dim,),
            "names": list(contract.action_names),
        },
        IMAGE_KEY: {"dtype": "image", "shape": (256, 256, 3), "names": None},
    }
    instructions = RemotePolicyConfig(
        policy_type=family,
        pretrained_name_or_path=str(directory),
        lerobot_features=observation_features,
        actions_per_chunk=EXECUTION_HORIZON,
        device=device,
        rename_map={},
    )
    server.SendPolicyInstructions(
        services_pb2.PolicySetup(data=pickle.dumps(instructions)), context
    )
    if not getattr(server, "_full_chunk_actions", False):
        raise AssertionError("async server did not activate the v2 full-chunk path")
    counting = _CountingPipeline(server.postprocessor)
    counting_pre = _CountingPipeline(server.preprocessor)
    server.postprocessor = counting
    server.preprocessor = counting_pre

    timestamp = 100.0
    actions = server._predict_action_chunk(
        TimedObservation(
            timestamp=timestamp,
            timestep=7,
            observation=_raw_observation(contract, state),
            must_go=True,
        )
    )
    if len(counting_pre.calls) != 1 or len(counting.calls) != 1:
        raise AssertionError(
            "async server must pre/postprocess exactly once per chunk: "
            f"pre={counting_pre.calls}, post={counting.calls}"
        )
    chunk_shape = counting.calls[0]
    if chunk_shape is None or len(chunk_shape) != 3 or chunk_shape[2] != contract.action_dim:
        raise AssertionError(f"async postprocessor did not receive a full chunk: {chunk_shape}")
    if chunk_shape[1] < EXECUTION_HORIZON:
        raise AssertionError(
            f"policy chunk shorter than the external slice: {chunk_shape[1]}"
        )
    if len(actions) != EXECUTION_HORIZON:
        raise AssertionError(
            f"async slice length mismatch: {len(actions)} != {EXECUTION_HORIZON}"
        )
    timesteps = [action.get_timestep() for action in actions]
    timestamps = [action.get_timestamp() for action in actions]
    if timesteps != [7, 8]:
        raise AssertionError(f"async timestep continuity broke: {timesteps}")
    dt = 1.0 / 30.0
    expected = [timestamp + index * dt for index in range(EXECUTION_HORIZON)]
    if not np.allclose(timestamps, expected, atol=1e-9):
        raise AssertionError(f"async timestamp continuity broke: {timestamps} != {expected}")
    values = np.stack(
        [action.get_action().detach().cpu().numpy().reshape(-1) for action in actions]
    )
    if values.shape != (EXECUTION_HORIZON, contract.action_dim):
        raise AssertionError(f"async action shape mismatch: {values.shape}")
    if not np.all(np.isfinite(values)):
        raise AssertionError("async actions contain non-finite values")
    del server
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return {
        "preprocessor_calls": len(counting_pre.calls),
        "postprocessor_calls": len(counting.calls),
        "full_chunk_shape": list(chunk_shape),
        "slice_len": len(actions),
        "timesteps": timesteps,
        "timestamp_delta_s": round(timestamps[1] - timestamps[0], 9),
    }


def _check_processor_ordering(
    evidence, preprocessor, postprocessor, bundle, spec, family, config, contract
):
    pre_order, post_order = p15._check_ordering(preprocessor, postprocessor)
    p15._check_encode_contract(preprocessor, bundle, spec)
    if family in ("smolvla", "groot"):
        expected = 32 if family == "smolvla" else 132
        max_state = getattr(config, "max_state_dim")
        max_action = getattr(config, "max_action_dim")
        if max_state != expected or max_action != expected:
            raise AssertionError(
                f"{family} padding capacity changed: {max_state}/{max_action} != {expected}"
            )
        if contract.action_dim > max_action:
            raise AssertionError("dataset action dim exceeds the padding capacity")
        capacity = {"max_state_dim": max_state, "max_action_dim": max_action}
    else:
        # ACT는 padding이 없고 dimension 종속 layer를 그대로 쓴다.
        state_shape = tuple(config.input_features[OBS_STATE].shape)
        action_shape = tuple(config.output_features[ACTION].shape)
        if state_shape != (contract.state_dim,) or action_shape != (contract.action_dim,):
            raise AssertionError(
                "ACT feature dimensions must match the dataset exactly: "
                f"state={state_shape}, action={action_shape}"
            )
        capacity = {
            "dimension_specific": True,
            "state_shape": list(state_shape),
            "action_shape": list(action_shape),
        }
    evidence.update({"pre_order": pre_order, "post_order": post_order, "capacity": capacity})
    if spec.is_eef and spec.pose_format is PoseFormat.XYZ_ROT6D_ROWS and spec.is_relative:
        evidence["v1_numeric_parity"] = _v1_parity(contract, bundle)


def _v1_parity(contract, bundle) -> dict:
    """v1 `eef_relative + rot6d` 수치 parity(v2 transform vs v1 SE(3) 구현)."""
    state = bundle["state"].to(torch.float64)
    action = bundle["action"].to(torch.float64)
    v2_values = contract.transform.encode(state, action).detach().cpu().numpy()
    v1_values = v1_absolute_actions_to_relative(state, action).detach().cpu().numpy()
    delta = float(np.max(np.abs(np.asarray(v2_values) - np.asarray(v1_values))))
    if not np.isfinite(delta) or delta > TOL_V1_PARITY:
        raise AssertionError(f"v1 rot6d numeric parity broke: max|delta|={delta}")
    return {"max_abs_delta": delta, "tolerance": TOL_V1_PARITY, "reference": "v1_eef_relative_action"}


def _check_routing(evidence, directory: Path, *, spec, family, absolute_chunk, policy_io):
    contract = resolve_checkpoint_contract(directory, expected_policy_type=family)
    counting = p16r._CountingPolicyIO(policy_io)
    router = make_router(contract, policy_io=counting)
    current = p16r._current_state()
    routed = router.route(np.asarray(absolute_chunk, dtype=np.float32), current, platform="sim")
    if not routed.success:
        raise AssertionError(f"routing failed: {routed.reason} (index={routed.failed_index})")
    expected_ik = 1 if spec.is_eef else 0
    if routed.ik_calls != expected_ik:
        raise AssertionError(
            f"mode={spec.mode.value} expects {expected_ik} IK call(s), got {routed.ik_calls}"
        )
    if counting.calls != expected_ik:
        raise AssertionError(
            f"platform adapter called {counting.calls} times, expected {expected_ik}"
        )
    if routed.canonical_joint_radians is None:
        raise AssertionError("routing produced no joint command")
    if routed.canonical_joint_radians.shape != (len(absolute_chunk), 6):
        raise AssertionError(
            f"routing changed horizon/dof: {routed.canonical_joint_radians.shape}"
        )
    if routed.platform_dtype != "float32":
        raise AssertionError(f"platform command dtype must be float32: {routed.platform_dtype}")
    evidence.update(
        {
            "route": list(routed.route),
            "ik_calls": routed.ik_calls,
            "platform": routed.platform,
            "horizon": routed.horizon,
            "second_decode": False,
        }
    )
    return router, routed


def _check_kinematics_or_joint_path(
    evidence, *, spec, router, routed, absolute_chunk, policy_io
):
    if spec.is_eef:
        poses = np.asarray(absolute_chunk, dtype=np.float64)[:, : spec.pose_dim]
        if spec.pose_format is not PoseFormat.XYZ_ROT6D_ROWS:
            poses = convert_pose_format(poses, spec.pose_format, PoseFormat.XYZ_ROT6D_ROWS)
        achieved = np.stack(
            [
                np.asarray(
                    policy_io.kinematics.forward_xyz_rot6d(joints[:5].astype(np.float64)),
                    dtype=np.float64,
                )
                for joints in routed.canonical_joint_radians
            ]
        )
        position_error = float(np.max(np.linalg.norm(achieved[:, :3] - poses[:, :3], axis=-1)))
        rotation_error = float(
            np.max(
                np.abs(
                    rotation_geodesic_angle(
                        _rot6d_matrix(achieved[:, 3:9]), _rot6d_matrix(poses[:, 3:9])
                    )
                )
            )
        )
        if position_error > TOL_POSITION_M or rotation_error > TOL_ROTATION_RAD:
            raise AssertionError(
                "FK/IK round-trip exceeded tolerance: "
                f"pos={position_error} rot={rotation_error}"
            )
        evidence.update(
            {
                "kind": "eef_fk_ik_roundtrip",
                "max_position_error_m": position_error,
                "max_rotation_error_rad": rotation_error,
                "tolerance": {"position_m": TOL_POSITION_M, "rotation_rad": TOL_ROTATION_RAD},
            }
        )
        return

    # joint mode: IK를 읽지 않고 sim radian passthrough / real 경계 정확히 1회.
    if routed.ik_calls != 0:
        raise AssertionError("joint mode must not call IK")
    canonical = routed.canonical_joint_radians
    published = np.stack([sim_publish_command(row) for row in canonical])
    if not np.allclose(published, canonical, atol=0.0, rtol=0.0):
        raise AssertionError("sim publish applied a second unit conversion")
    real_state = np.asarray(
        sim_radians_to_real_follower(p16r._current_state()), dtype=np.float32
    )
    real_routed = router.route(
        np.asarray(absolute_chunk, dtype=np.float32), real_state, platform="real_dry_run"
    )
    if not real_routed.success:
        raise AssertionError(f"real dry-run routing failed: {real_routed.reason}")
    if real_routed.ik_calls != 0:
        raise AssertionError("joint real route must not call IK")
    delta = float(
        np.max(np.abs(real_routed.canonical_joint_radians - canonical))
    )
    if not np.isfinite(delta) or delta > 1e-4:
        raise AssertionError(
            f"real boundary conversion is not exactly-once/invertible: max|Δ|={delta}"
        )
    evidence.update(
        {
            "kind": "joint_command_path",
            "ik_calls": 0,
            "sim_publish": "canonical_passthrough",
            "real_boundary_roundtrip_max_delta": delta,
        }
    )


def _rot6d_matrix(rows: np.ndarray) -> np.ndarray:
    from so101_contract.eef_relative_action import rot6d_rows_to_matrix

    return rot6d_rows_to_matrix(np.asarray(rows, dtype=np.float64))


def _copy_checkpoint_dedup(source: Path, destination: Path) -> Path:
    """weight를 hardlink로 복사(대형 checkpoint 폭증 방지)."""
    shutil.copytree(source, destination, copy_function=os.link)
    return destination


def _check_migration(evidence, directory: Path, *, spec, fixture_root: Path, workspace: Path):
    """지원되는 legacy만 positive migration, 나머지는 명시적 guard로 기록."""
    workspace.mkdir(parents=True, exist_ok=True)
    manifest_less = _copy_checkpoint_dedup(directory, workspace / "legacy_manifest_less")
    (manifest_less / ACTION_REPRESENTATION_MANIFEST).unlink()
    if detect_source_schema_state(manifest_less) != "manifest_absent":
        raise AssertionError("manifest-less legacy checkpoint was not detected")
    before = checkpoint_directory_sha256(manifest_less)

    guards: dict[str, str] = {}
    guards["missing_opt_in"] = _expect_raises(
        lambda: plan_migration(
            manifest_less,
            workspace / "out_no_opt_in",
            dataset_root=fixture_root,
            horizon=HORIZON,
        ),
        PermissionError,
        "manifest-less migration without the opt-in flag",
    )

    result_payload = None
    if spec.mode is ActionRepresentationMode.JOINT_ABSOLUTE:
        plan = plan_migration(
            manifest_less,
            workspace / "migrated_joint_absolute",
            dataset_root=fixture_root,
            horizon=HORIZON,
            allow_legacy_joint_absolute=True,
        )
        result = migrate_checkpoint(plan)
        migrated_manifest = read_action_representation_manifest(result.output)
        if migrated_manifest is None:
            raise AssertionError("migration produced no schema v2 manifest")
        if migrated_manifest["mode"] != ActionRepresentationMode.JOINT_ABSOLUTE.value:
            raise AssertionError("migration declared the wrong mode")
        if not (migrated_manifest.get("legacy") or {}).get("allowed"):
            raise AssertionError("legacy opt-in was not recorded in the manifest")
        contract = resolve_checkpoint_contract(result.output)
        if contract.spec.mode is not ActionRepresentationMode.JOINT_ABSOLUTE:
            raise AssertionError("migrated checkpoint does not resolve to joint_absolute")
        result_payload = {
            "kind": "manifest_less_joint_absolute",
            "output": str(result.output),
            "manifest_sha256": migrated_manifest["manifest_sha256"],
        }
        shutil.rmtree(result.output, ignore_errors=True)
    elif spec.mode is ActionRepresentationMode.EEF_RELATIVE and (
        spec.pose_format is PoseFormat.XYZ_ROT6D_ROWS
    ):
        v1_source = _copy_checkpoint_dedup(directory, workspace / "legacy_v1_eef")
        # hardlink dedup 사본이므로 반드시 link를 끊고 새 파일로 써야 원본이 오염되지 않는다.
        (v1_source / ACTION_REPRESENTATION_MANIFEST).unlink()
        (v1_source / ACTION_REPRESENTATION_MANIFEST).write_text(
            json.dumps(
                {
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
                    "relative_stats": {
                        "profile_id": f"sha256:{'d' * 64}",
                        "content_sha256": "d" * 64,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if detect_source_schema_state(v1_source) != "v1_eef_relative":
            raise AssertionError("v1 EEF-relative legacy checkpoint was not detected")
        plan = plan_migration(
            v1_source,
            workspace / "migrated_v1_eef",
            dataset_root=fixture_root,
            horizon=HORIZON,
            mode="eef_relative",
            pose_format="xyz_rot6d_rows",
        )
        result = migrate_checkpoint(plan)
        migrated_manifest = read_action_representation_manifest(result.output)
        if migrated_manifest is None or migrated_manifest["mode"] != "eef_relative":
            raise AssertionError("v1 migration did not produce an eef_relative v2 manifest")
        if (migrated_manifest.get("legacy") or {}).get("allowed"):
            raise AssertionError("v1 migration must not set the joint_absolute opt-in flag")
        guards["opt_in_on_v1_source"] = _expect_raises(
            lambda: plan_migration(
                v1_source,
                workspace / "out_v1_opt_in",
                dataset_root=fixture_root,
                horizon=HORIZON,
                allow_legacy_joint_absolute=True,
            ),
            ValueError,
            "joint_absolute opt-in on a v1 EEF source",
        )
        result_payload = {
            "kind": "v1_eef_relative_rot6d",
            "output": str(result.output),
            "manifest_sha256": migrated_manifest["manifest_sha256"],
        }
        shutil.rmtree(result.output, ignore_errors=True)
        shutil.rmtree(v1_source, ignore_errors=True)
    else:
        # 이 representation은 지원되는 legacy source가 없다. 추정 금지가 계약이므로
        # 명시적 거부 자체를 증거로 남긴다(silent skip 아님).
        guards["opt_in_cannot_declare_this_mode"] = _expect_raises(
            lambda: plan_migration(
                manifest_less,
                workspace / "out_wrong_mode",
                dataset_root=fixture_root,
                horizon=HORIZON,
                mode=spec.mode.value,
                pose_format=(
                    None if spec.pose_format is PoseFormat.NOT_APPLICABLE else spec.pose_format.value
                ),
                allow_legacy_joint_absolute=True,
            ),
            ValueError,
            f"legacy opt-in must not declare {spec.mode.value}",
        )
        result_payload = {
            "kind": "expected_guard",
            "reason": (
                "no supported legacy source declares "
                f"{spec.mode.value}"
                + (
                    ""
                    if spec.pose_format is PoseFormat.NOT_APPLICABLE
                    else f"+{spec.pose_format.value}"
                )
                + "; migration refuses to infer it"
            ),
        }

    after = checkpoint_directory_sha256(manifest_less)
    if after["tree_sha256"] != before["tree_sha256"]:
        raise AssertionError("migration mutated the legacy source checkpoint")
    shutil.rmtree(manifest_less, ignore_errors=True)
    evidence.update(
        {
            "migration": result_payload,
            "guards": guards,
            "source_bytes_unchanged": True,
        }
    )


def _check_fail_fast(evidence, directory: Path, *, spec, family, workspace: Path):
    guards: dict[str, str] = {}
    wrong_mode = (
        ActionRepresentationMode.EEF_ABSOLUTE
        if spec.mode is not ActionRepresentationMode.EEF_ABSOLUTE
        else ActionRepresentationMode.JOINT_ABSOLUTE
    )
    guards["cli_mode_mismatch"] = _expect_raises(
        lambda: assert_checkpoint_representation(
            directory,
            mode=wrong_mode.value,
            pose_format=None,
            policy_type=family,
        ),
        (ValueError, TypeError),
        "CLI mode assertion mismatch must refuse to start",
    )
    guards["cli_policy_type_mismatch"] = _expect_raises(
        lambda: assert_checkpoint_representation(
            directory,
            policy_type="act" if family != "act" else "smolvla",
        ),
        (ValueError, TypeError),
        "CLI policy type assertion mismatch must refuse to start",
    )
    if not spec.is_eef:
        guards["joint_mode_rejects_pose_format"] = _expect_raises(
            lambda: ActionRepresentationSpec(
                mode=spec.mode, pose_format=PoseFormat.XYZ_ROT6D_ROWS
            ),
            (ValueError, TypeError),
            "joint mode must reject a pose format",
        )
    else:
        other = (
            PoseFormat.XYZ_RPY
            if spec.pose_format is not PoseFormat.XYZ_RPY
            else PoseFormat.XYZ_ROT6D_ROWS
        )
        guards["cli_pose_format_mismatch"] = _expect_raises(
            lambda: assert_checkpoint_representation(
                directory,
                mode=spec.mode.value,
                pose_format=other.value,
                policy_type=family,
            ),
            (ValueError, TypeError),
            "CLI pose format assertion mismatch must refuse to start",
        )
    guards["ambiguous_absolute_mode_rejected"] = _expect_raises(
        lambda: ActionRepresentationSpec(mode="absolute"),
        (ValueError, TypeError, KeyError),
        "ambiguous mode=absolute must be refused",
    )
    empty = workspace / "no_manifest"
    empty.mkdir(parents=True, exist_ok=True)
    guards["missing_manifest_rejected"] = _expect_raises(
        lambda: resolve_checkpoint_contract(empty),
        (FileNotFoundError, ValueError),
        "checkpoint without a manifest must fail-fast",
    )
    shutil.rmtree(empty, ignore_errors=True)
    # 정상 assertion은 통과해야 한다(guard가 무조건 실패하는 게 아님).
    assert_checkpoint_representation(
        directory,
        mode=spec.mode.value,
        pose_format=(
            None if spec.pose_format is PoseFormat.NOT_APPLICABLE else spec.pose_format.value
        ),
        policy_type=family,
    )
    evidence.update({"guards": guards, "matching_assertion": "accepted"})


# --- 조합 실행 ----------------------------------------------------------------


def run_combination(
    family: str,
    spec,
    bundle: dict,
    *,
    workspace: Path,
    device: str,
    cached_model: dict,
    shared_weights: dict,
    policy_io,
) -> dict:
    combination = combination_id(family, spec)
    recorder = CheckRecorder(combination)
    contract = bundle["contract"]
    combo_dir = workspace / "checkpoints" / combination
    combo_dir.mkdir(parents=True, exist_ok=True)
    aux_dir = workspace / "aux" / combination
    aux_dir.mkdir(parents=True, exist_ok=True)

    config, model, preprocessor, postprocessor, image_shape = _build_policy(
        family, spec, bundle, cached_model=cached_model, device=device
    )

    with recorder.run("config_serialization_roundtrip") as evidence:
        _check_config_roundtrip(evidence, combo_dir, config, spec)

    manifest = None
    with recorder.run("manifest_generate_hash_load_tamper") as evidence:
        manifest = _check_manifest(
            evidence, combo_dir, config, preprocessor, model, spec, family, contract
        )

    with recorder.run("dataset_metadata") as evidence:
        _check_dataset_metadata(evidence, bundle["fixture_root"], spec, contract)

    with recorder.run("stats_generate_and_checkpoint_persistence") as evidence:
        _check_stats(evidence, combo_dir, spec, contract, bundle, preprocessor)

    with recorder.run("one_batch_forward_backward") as evidence:
        _check_forward_backward(
            evidence, model, preprocessor, bundle, image_shape, family, spec
        )

    with recorder.run("processor_ordering") as evidence:
        _check_processor_ordering(
            evidence, preprocessor, postprocessor, bundle, spec, family, config, contract
        )

    observation = {
        OBS_STATE: bundle["state"][:1].clone().to(device),
        IMAGE_KEY: torch.zeros(1, *image_shape, device=device),
        "task": ["phase17 fixture"],
    }

    reloaded = None
    if manifest is not None:
        with recorder.run("checkpoint_save_reload_parity") as evidence:
            reloaded = _check_reload_parity(
                evidence,
                combo_dir,
                family=family,
                spec=spec,
                model=model,
                config=config,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                manifest=manifest,
                observation=observation,
                shared_weights=shared_weights,
                device=device,
            )

    run_pre = preprocessor if reloaded is None else reloaded[1]
    run_post = postprocessor if reloaded is None else reloaded[2]
    run_model = model if reloaded is None else reloaded[3]

    absolute_chunk = None
    with recorder.run("full_chunk_inference") as evidence:
        absolute_chunk = _check_full_chunk(
            evidence, run_model, run_pre, run_post, observation, contract
        )

    with recorder.run("sync_and_async_inference") as evidence:
        _check_sync_and_async(
            evidence,
            directory=combo_dir,
            family=family,
            contract=contract,
            model=run_model,
            preprocessor=run_pre,
            postprocessor=run_post,
            device=device,
            state=bundle["state"][0].detach().cpu().numpy(),
        )

    # routing/FK-IK는 정책 출력이 아니라 **도달 가능한** absolute chunk로 검증한다.
    reachable_chunk = p16r._absolute_chunk(spec, policy_io)
    router = routed = None
    with recorder.run("platform_adapter_routing") as evidence:
        evidence["policy_chunk_finite"] = bool(
            absolute_chunk is None or np.all(np.isfinite(absolute_chunk))
        )
        router, routed = _check_routing(
            evidence,
            combo_dir,
            spec=spec,
            family=family,
            absolute_chunk=reachable_chunk,
            policy_io=policy_io,
        )

    with recorder.run("fk_ik_roundtrip_or_joint_command_path") as evidence:
        if router is None or routed is None:
            raise AssertionError("routing did not produce a router/result to verify")
        _check_kinematics_or_joint_path(
            evidence,
            spec=spec,
            router=router,
            routed=routed,
            absolute_chunk=reachable_chunk,
            policy_io=policy_io,
        )

    with recorder.run("legacy_checkpoint_migration") as evidence:
        _check_migration(
            evidence,
            combo_dir,
            spec=spec,
            fixture_root=bundle["fixture_root"],
            workspace=aux_dir / "migration",
        )

    with recorder.run("invalid_cli_checkpoint_fail_fast") as evidence:
        _check_fail_fast(
            evidence, combo_dir, spec=spec, family=family, workspace=aux_dir / "failfast"
        )

    # 조합 임시 산출물 정리(공유 weight는 마지막에 한 번 지운다).
    shutil.rmtree(combo_dir, ignore_errors=True)
    shutil.rmtree(aux_dir, ignore_errors=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return recorder.to_dict()


# --- 환경 메타데이터 -----------------------------------------------------------


def _coerce_bool(value: str) -> bool | None:
    lowered = value.strip().lower()
    if lowered in ("1", "true", "yes", "dirty"):
        return True
    if lowered in ("0", "false", "no", "clean"):
        return False
    return None


def _git_identity() -> dict:
    """Provenance는 host가 주입한 값을 우선한다.

    container 안에서는 git이 없거나 worktree 메타데이터가 보이지 않을 수 있다. 검출에
    실패했는데 ``dirty=false``로 단정하면 **실제 dirty worktree를 clean으로 오기록**하므로,
    그 경우 ``None``(unknown)을 남기고 detection source를 함께 적는다.
    """

    def _run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=False
            )
        except OSError:
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    injected_commit = os.environ.get("SO101_PROJECT_GIT_COMMIT", "").strip()
    injected_branch = os.environ.get("SO101_PROJECT_GIT_BRANCH", "").strip()
    injected_dirty = _coerce_bool(os.environ.get("SO101_PROJECT_GIT_DIRTY", ""))

    commit = injected_commit or (_run("rev-parse", "HEAD") or None)
    branch = injected_branch or (_run("rev-parse", "--abbrev-ref", "HEAD") or None)
    if injected_dirty is not None:
        dirty = injected_dirty
        dirty_source = "injected"
    else:
        porcelain = _run("status", "--porcelain")
        dirty = None if porcelain is None else bool(porcelain)
        dirty_source = "detected" if porcelain is not None else "unknown"
    return {
        "commit": commit,
        "branch": branch,
        "dirty": dirty,
        "provenance_source": {
            "commit": "injected" if injected_commit else ("detected" if commit else "unknown"),
            "branch": "injected" if injected_branch else ("detected" if branch else "unknown"),
            "dirty": dirty_source,
        },
    }


def _docker_image_id() -> str | None:
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.Id}}", "policy-server:0.6.0"],
            capture_output=True,
            text=True,
            check=False,
        )
        identity = result.stdout.strip()
    except OSError:
        identity = ""
    return identity or os.environ.get("SO101_DOCKER_IMAGE_ID") or None


def _lerobot_identity() -> dict:
    import importlib.metadata

    from so101_contract.eef_checkpoint_manifest import PINNED_LEROBOT_COMMIT

    return {
        "version": importlib.metadata.version("lerobot"),
        "commit": os.environ.get("LEROBOT_GIT_COMMIT", "").strip().lower()
        or PINNED_LEROBOT_COMMIT,
    }


def _device_identity(device: str) -> dict:
    payload = {"device": device, "torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        payload["gpu_name"] = torch.cuda.get_device_name(0)
        payload["gpu_capability"] = ".".join(str(v) for v in torch.cuda.get_device_capability(0))
    return payload


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    # NamedTemporaryFile은 0600으로 만든다. container(root)에서 쓴 artifact를 host
    # 사용자가 jq로 읽어야 하므로 0644로 바꾼 뒤 publish한다.
    temporary.chmod(0o644)
    temporary.replace(path)


# --- main ---------------------------------------------------------------------


def _load_bundle(fixture_root: Path, fixture_name: str, spec) -> dict:
    root = fixture_root / fixture_name
    bundle = p15._dataset_bundle(root, spec)
    contract = bundle["contract"]
    bundle["fixture_root"] = root
    bundle["episodes"] = p15.load_lerobot_v3_episodes(
        root,
        state_key=contract.state_key,
        action_key=contract.action_key,
        state_dim=contract.state_dim,
        action_dim=contract.action_dim,
    )
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--policies", default=",".join(POLICY_FAMILIES))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--representations",
        default="",
        help="빠른 개발용 필터(쉼표 구분 stats profile kind). completion 실행에서는 비워 둔다.",
    )
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    families = [name.strip() for name in args.policies.split(",") if name.strip()]
    unknown = [name for name in families if name not in POLICY_FAMILIES]
    if unknown:
        raise SystemExit(f"unknown policy families: {unknown}")
    representation_filter = {
        name.strip() for name in args.representations.split(",") if name.strip()
    }

    workspace = Path(
        args.work_dir
        or tempfile.mkdtemp(prefix="p17-matrix-", dir=str(ROOT / "scratch"))
    )
    workspace.mkdir(parents=True, exist_ok=True)
    policy_io = p16r._policy_io()

    bundles: dict[str, dict] = {}
    combinations: list[dict] = []
    cached_model: dict = {}
    shared_weights: dict = {}
    started = time.time()
    try:
        for family in families:
            for fixture_name, spec in p15._specs():
                if representation_filter and spec.stats_profile_kind not in representation_filter:
                    continue
                key = f"{fixture_name}:{spec.stats_profile_kind}"
                if key not in bundles:
                    bundles[key] = _load_bundle(args.fixture_root, fixture_name, spec)
                torch.manual_seed(args.seed)
                np.random.seed(args.seed)
                print(f"[matrix] {combination_id(family, spec)}")
                combinations.append(
                    run_combination(
                        family,
                        spec,
                        bundles[key],
                        workspace=workspace,
                        device=args.device,
                        cached_model=cached_model,
                        shared_weights=shared_weights,
                        policy_io=policy_io,
                    )
                )
    finally:
        if not args.keep_work:
            shutil.rmtree(workspace, ignore_errors=True)

    expected_total = len(families) * 8
    passed = [entry for entry in combinations if entry["status"] == "pass"]
    failed = [entry for entry in combinations if entry["status"] != "pass"]
    checks_total = sum(len(entry["checks"]) for entry in combinations)
    checks_passed = sum(
        1
        for entry in combinations
        for value in entry["checks"].values()
        if value["status"] == "pass"
    )
    checks_skipped = [
        {"combination": entry["combination"], "check": name, "reason": value.get("reason")}
        for entry in combinations
        for name, value in entry["checks"].items()
        if value["status"] == "skip"
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "duration_s": round(time.time() - started, 2),
        "git": _git_identity(),
        "docker_image_id": _docker_image_id(),
        "lerobot": _lerobot_identity(),
        "runtime": _device_identity(args.device),
        "seed": args.seed,
        "fixture_root": str(args.fixture_root),
        "required_checks": list(CHECK_NAMES),
        "totals": {
            "expected_combinations": expected_total,
            "ran_combinations": len(combinations),
            "passed_combinations": len(passed),
            "failed_combinations": len(failed),
            "expected_checks": expected_total * len(CHECK_NAMES),
            "ran_checks": checks_total,
            "passed_checks": checks_passed,
            "complete": len(combinations) == expected_total and not failed,
        },
        "failures": [
            {
                "combination": entry["combination"],
                "failed_checks": entry["failed_checks"],
                "missing_checks": entry["missing_checks"],
                "errors": {
                    name: entry["checks"][name].get("error")
                    for name in entry["failed_checks"]
                },
            }
            for entry in failed
        ],
        "skips": checks_skipped,
        "combinations": combinations,
    }
    _atomic_write_json(args.output, payload)

    print(
        f"\n{len(passed)}/{expected_total} combinations passed; "
        f"{checks_passed}/{expected_total * len(CHECK_NAMES)} required checks passed"
    )
    for entry in failed:
        print(f"  FAILED {entry['combination']}: {entry['failed_checks'] or entry['missing_checks']}")
    if failed or len(combinations) != expected_total:
        if len(combinations) != expected_total:
            print(
                f"  INCOMPLETE: ran {len(combinations)} of {expected_total} combinations "
                "(completion runs must cover all 24)"
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
