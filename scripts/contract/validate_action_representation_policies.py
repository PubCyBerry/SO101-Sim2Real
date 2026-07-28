#!/usr/bin/env python3
"""Phase 15 — 3 policy × 8 representation = 24 조합 통합 검증.

각 조합에서 다음을 실제 LeRobot factory/policy class로 확인한다(mock 없음).

1. dataset 계약 → stats profile → v2 context resolve
2. ``make_pre_post_processors``가 만든 pipeline의 **step ordering**
   (encode < normalize/pack, decode > unnormalize/decode, decode < device)
3. policy family capacity(SmolVLA 32 / GR00T 132)와 output slice
4. encode target이 ``ActionRepresentationTransform.encode``와 수치 일치
5. 실제 policy 1-batch forward/backward(finite loss/grad)
6. full-chunk runner가 chunk당 pre/post를 정확히 1회 호출
7. schema v2 manifest 생성·검증·processor fingerprint 일치

ACT는 dimension이 정확히 일치해야 하므로 조합마다 새 policy를 만든다. SmolVLA/GR00T는
padding capacity 덕분에 8개 representation이 같은 weight shape를 쓰므로 base weight는
한 번만 로드하고 재사용한다(3B를 24번 로드하지 않는다).

.. code-block:: bash

    python scripts/contract/validate_action_representation_policies.py \\
        --fixture-root scratch/fx_images --policies act,smolvla,groot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402
from lerobot.configs.action_representation import ActionRepresentationConfig  # noqa: E402
from lerobot.policies import make_pre_post_processors  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402

from so101_contract.action_dataset_contract import (  # noqa: E402
    resolve_action_representation_contract,
)
from so101_contract.action_manifest import (  # noqa: E402
    validate_action_representation_manifest,
)
from so101_contract.action_representation import (  # noqa: E402
    ActionRepresentationMode,
    ActionRepresentationSpec,
    PoseFormat,
    combination_id,
)
from so101_contract.action_representation_stats import (  # noqa: E402
    load_lerobot_v3_episodes,
)
from so101_contract.lerobot_full_chunk import FullChunkPolicyRunner  # noqa: E402
from so101_contract.lerobot_v2_integration import (  # noqa: E402
    build_manifest_from_processor,
    processor_step_order,
)

SMOLVLA_MODEL_ID = "lerobot/smolvla_base"
GROOT_MODEL_ID = "nvidia/GR00T-N1.7-3B"
HORIZON = 4
IMAGE_KEY = "observation.images.top"

_NORMALIZE_STEPS = ("normalizer_processor", "groot_n1_7_pack_inputs_v1")
_DECODE_STEPS = (
    "unnormalizer_processor",
    "groot_action_unpack_unnormalize_v2",
    "groot_n1_7_action_decode_v1",
)
_ENCODE_STEP = "so101_action_representation_encode_v2"
_DECODE_STEP = "so101_action_representation_decode_v2"
_DEVICE_STEP = "device_processor"


def _specs() -> list[tuple[str, ActionRepresentationSpec]]:
    """(fixture 이름, spec) 8개."""
    entries: list[tuple[str, ActionRepresentationSpec]] = [
        ("joint", ActionRepresentationSpec(mode=ActionRepresentationMode.JOINT_ABSOLUTE)),
        ("joint", ActionRepresentationSpec(mode=ActionRepresentationMode.JOINT_RELATIVE)),
    ]
    for pose_format in (
        PoseFormat.XYZ_ROT6D_ROWS,
        PoseFormat.XYZ_QUATERNION_WXYZ,
        PoseFormat.XYZ_RPY,
    ):
        for mode in (
            ActionRepresentationMode.EEF_ABSOLUTE,
            ActionRepresentationMode.EEF_RELATIVE,
        ):
            entries.append(
                (pose_format.value, ActionRepresentationSpec(mode=mode, pose_format=pose_format))
            )
    return entries


def _representation_config(spec: ActionRepresentationSpec) -> ActionRepresentationConfig:
    return ActionRepresentationConfig(
        mode=spec.mode.value,
        pose_format=None if spec.pose_format is PoseFormat.NOT_APPLICABLE else spec.pose_format.value,
        stats_file=spec.stats_file,
    )


def _numeric_stats(values: np.ndarray) -> dict[str, list[float]]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": values.mean(0).tolist(),
        "std": (values.std(0) + 1e-6).tolist(),
        "min": values.min(0).tolist(),
        "max": values.max(0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def _dataset_bundle(root: Path, spec: ActionRepresentationSpec) -> dict:
    contract = resolve_action_representation_contract(root, spec)
    episodes = load_lerobot_v3_episodes(
        root,
        state_key=contract.state_key,
        action_key=contract.action_key,
        state_dim=contract.state_dim,
        action_dim=contract.action_dim,
    )
    states = np.concatenate([episode.states for episode in episodes])
    actions = np.concatenate([episode.actions for episode in episodes])
    dataset_stats = {OBS_STATE: _numeric_stats(states), ACTION: _numeric_stats(actions)}
    dataset_meta = SimpleNamespace(
        root=root,
        repo_id=f"local/{root.name}",
        revision=None,
        features={
            OBS_STATE: {
                "dtype": "float32",
                "shape": (contract.state_dim,),
                "names": list(contract.state_names),
            },
            ACTION: {
                "dtype": "float32",
                "shape": (contract.action_dim,),
                "names": list(contract.action_names),
            },
            IMAGE_KEY: {"dtype": "video", "shape": (3, 256, 256), "names": None},
        },
        stats=dataset_stats,
    )
    episode = episodes[0]
    batch_state = torch.from_numpy(episode.states[:2].copy())
    batch_action = torch.from_numpy(
        np.stack([episode.actions[index : index + HORIZON] for index in range(2)])
    )
    return {
        "contract": contract,
        "dataset_stats": dataset_stats,
        "dataset_meta": dataset_meta,
        "state": batch_state,
        "action": batch_action,
    }


def _check_ordering(preprocessor, postprocessor) -> tuple[list, list]:
    pre_order = processor_step_order(preprocessor)
    post_order = processor_step_order(postprocessor)
    if _ENCODE_STEP not in pre_order:
        raise AssertionError(f"encode step missing from preprocessor: {pre_order}")
    if _DECODE_STEP not in post_order:
        raise AssertionError(f"decode step missing from postprocessor: {post_order}")

    encode_index = pre_order.index(_ENCODE_STEP)
    normalize_indices = [pre_order.index(name) for name in _NORMALIZE_STEPS if name in pre_order]
    if not normalize_indices:
        raise AssertionError(f"preprocessor has no normalize/pack step: {pre_order}")
    if encode_index > min(normalize_indices):
        raise AssertionError(
            f"encode step must precede normalization: {pre_order}"
        )

    decode_index = post_order.index(_DECODE_STEP)
    policy_decode = [post_order.index(name) for name in _DECODE_STEPS if name in post_order]
    if not policy_decode:
        raise AssertionError(f"postprocessor has no unnormalize/decode step: {post_order}")
    if decode_index < max(policy_decode):
        raise AssertionError(
            f"decode step must follow the policy unnormalize/decode step: {post_order}"
        )
    if _DEVICE_STEP in post_order and decode_index > post_order.index(_DEVICE_STEP):
        raise AssertionError(f"decode step must precede the device move: {post_order}")
    return pre_order, post_order


def _check_encode_contract(preprocessor, bundle, spec) -> None:
    """pipeline에 꽂힌 encode step이 transform 계약과 정확히 같은 target을 만든다."""
    from lerobot.processor.pipeline import TransitionKey

    from so101_contract.lerobot_v2_integration import action_representation_encode_step

    step = action_representation_encode_step(preprocessor)
    state = bundle["state"].clone()
    action = bundle["action"].clone()
    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: state},
        TransitionKey.ACTION: action,
    }
    produced = step(transition)[TransitionKey.ACTION]
    expected = step.transform.encode(state, action)
    if not torch.allclose(produced, expected, atol=1e-6):
        raise AssertionError(
            f"{spec.stats_profile_kind} pipeline encode disagrees with the transform contract"
        )
    if spec.is_relative and step.get_cached_state() is None:
        raise AssertionError("relative mode did not cache the prediction-time state")
    if not spec.is_relative and step.get_cached_state() is not None:
        raise AssertionError("absolute mode must not cache a state")
    step.reset()


def _build_act(config_features, spec, contract, dataset_stats, dataset_meta, image_shape):
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    config = ACTConfig(
        chunk_size=HORIZON,
        n_action_steps=2,
        input_features=config_features["input"],
        output_features=config_features["output"],
        normalization_mapping={
            "STATE": NormalizationMode.MEAN_STD,
            "VISUAL": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.MEAN_STD,
        },
        device="cpu",
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
        dataset_stats=dataset_stats,
        dataset_meta=dataset_meta,
    )
    return config, ACTPolicy(config), preprocessor, postprocessor


def _run_combination(
    policy_family: str,
    spec: ActionRepresentationSpec,
    bundle: dict,
    *,
    cached_model: dict,
    device: str,
) -> dict:
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

    if policy_family == "act":
        config, model, preprocessor, postprocessor = _build_act(
            config_features,
            spec,
            contract,
            bundle["dataset_stats"],
            bundle["dataset_meta"],
            image_shape,
        )
    elif policy_family == "smolvla":
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        config = PreTrainedConfig.from_pretrained(SMOLVLA_MODEL_ID, local_files_only=True)
        config.input_features = config_features["input"]
        config.output_features = config_features["output"]
        config.chunk_size = HORIZON
        config.n_action_steps = 2
        config.device = device
        config.load_vlm_weights = False
        config.action_representation = _representation_config(spec)
        preprocessor, postprocessor = make_pre_post_processors(
            config,
            dataset_stats=bundle["dataset_stats"],
            dataset_meta=bundle["dataset_meta"],
        )
        if cached_model.get("smolvla") is None:
            cached_model["smolvla"] = SmolVLAPolicy(config).to(config.device)
        model = cached_model["smolvla"]
        model.config = config
        model.train()
    else:
        from lerobot.policies.groot.configuration_groot import GrootConfig
        from lerobot.policies.groot.modeling_groot import GrootPolicy

        config = GrootConfig(
            base_model_path=GROOT_MODEL_ID,
            embodiment_tag="new_embodiment",
            chunk_size=HORIZON,
            n_action_steps=2,
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
            cached_model["groot"] = GrootPolicy(config).to(config.device)
        model = cached_model["groot"]
        model.config = config
        model.train()

    pre_order, post_order = _check_ordering(preprocessor, postprocessor)
    _check_encode_contract(preprocessor, bundle, spec)

    # capacity 검증
    if policy_family in ("smolvla", "groot"):
        max_state = getattr(config, "max_state_dim")
        max_action = getattr(config, "max_action_dim")
        expected_capacity = 32 if policy_family == "smolvla" else 132
        if max_state != expected_capacity or max_action != expected_capacity:
            raise AssertionError(
                f"{policy_family} capacity changed: {max_state}/{max_action} != {expected_capacity}"
            )
        if contract.action_dim > max_action:
            raise AssertionError("dataset action dim exceeds the padding capacity")

    batch = int(bundle["state"].shape[0])
    images = torch.zeros(batch, *image_shape)
    raw_batch = {
        OBS_STATE: bundle["state"].clone(),
        IMAGE_KEY: images,
        ACTION: bundle["action"].clone(),
        "action_is_pad": torch.zeros(batch, HORIZON, dtype=torch.bool),
        "task": ["schema v2 fixture"] * batch,
    }
    processed = preprocessor(raw_batch)
    if processed[ACTION].shape[:2] != (batch, HORIZON):
        raise AssertionError(f"processed action chunk shape mismatch: {processed[ACTION].shape}")

    loss, _ = model(processed)
    if not torch.isfinite(loss):
        raise AssertionError(f"{policy_family}/{spec.stats_profile_kind} loss is not finite")
    model.zero_grad(set_to_none=True)
    loss.backward()
    if not any(
        parameter.grad is not None and torch.all(torch.isfinite(parameter.grad))
        for parameter in model.parameters()
    ):
        raise AssertionError(f"{policy_family}/{spec.stats_profile_kind} produced no finite grads")
    model.zero_grad(set_to_none=True)

    runner = FullChunkPolicyRunner(
        model,
        preprocessor,
        postprocessor,
        execution_horizon=config.n_action_steps,
    )
    observation = {OBS_STATE: bundle["state"][:1], IMAGE_KEY: images[:1], "task": ["fixture"]}
    with torch.inference_mode():
        first = runner.next_action(observation)
        runner.next_action(observation)
    if first.shape != (1, contract.action_dim):
        raise AssertionError(f"full-chunk action shape mismatch: {tuple(first.shape)}")
    metrics = runner.metrics
    if metrics.preprocessor_calls != 1 or metrics.postprocessor_calls != 1:
        raise AssertionError(f"full chunk was reprocessed per tick: {metrics}")

    manifest = build_manifest_from_processor(config, preprocessor, policy=model)
    validate_action_representation_manifest(
        manifest,
        expected_spec=spec,
        expected_policy_type=policy_family,
    )
    if manifest["resolved_contract_fingerprint"] != contract.fingerprint:
        raise AssertionError("manifest fingerprint does not match the dataset contract")

    return {
        "combination": combination_id(policy_family, spec),
        "action_dim": contract.action_dim,
        "loss": float(loss.detach().cpu()),
        "manifest_sha256": manifest["manifest_sha256"],
        "pre_order": pre_order,
        "post_order": post_order,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--policies", default="act,smolvla,groot")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    families = [name.strip() for name in args.policies.split(",") if name.strip()]
    bundles: dict[str, dict] = {}
    results: list[dict] = []
    failures: list[str] = []
    cached_model: dict = {}

    for family in families:
        for fixture_name, spec in _specs():
            key = f"{fixture_name}:{spec.stats_profile_kind}"
            if key not in bundles:
                bundles[key] = _dataset_bundle(args.fixture_root / fixture_name, spec)
            try:
                result = _run_combination(
                    family,
                    spec,
                    bundles[key],
                    cached_model=cached_model,
                    device=args.device,
                )
            except Exception as exc:  # noqa: BLE001 - 조합별 실패를 모아 보고한다
                failures.append(f"{combination_id(family, spec)}: {type(exc).__name__}: {exc}")
                print(f"FAIL {combination_id(family, spec)}: {exc}")
                continue
            results.append(result)
            print(f"PASS {result['combination']} dim={result['action_dim']}")

    summary = {
        "total": len(families) * 8,
        "passed": len(results),
        "failed": len(failures),
        "failures": failures,
        "results": results,
    }
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"\n{summary['passed']}/{summary['total']} combinations passed "
        f"({len(families)} policies x 8 representations)"
    )
    if failures:
        for failure in failures:
            print(f"  FAILED {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
