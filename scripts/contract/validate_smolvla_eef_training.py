#!/usr/bin/env python3
"""Cached SmolVLA checkpoint + EEF-relative 1-batch forward/backward 검증."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "data"))

from generate_relative_action_stats import _write_self_check_dataset  # noqa: E402
from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig  # noqa: E402
from lerobot.configs.action_representation import ActionRepresentationConfig  # noqa: E402
from lerobot.policies import make_pre_post_processors  # noqa: E402
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: E402,F401
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402
from so101_contract.eef_relative_stats import (  # noqa: E402
    RelativeActionSamplingConfig,
    calculate_relative_action_stats,
    write_relative_action_stats_profile,
)

MODEL_ID = "lerobot/smolvla_base"


def _stats(dim: int) -> dict[str, list[float]]:
    return {
        "mean": [0.0] * dim,
        "std": [1.0] * dim,
        "min": [-1.0] * dim,
        "max": [1.0] * dim,
        "q01": [-1.0] * dim,
        "q10": [-0.5] * dim,
        "q50": [0.0] * dim,
        "q90": [0.5] * dim,
        "q99": [1.0] * dim,
    }


def _write_v2_stats(dataset_root, horizon):
    """schema v2 stats artifact를 만들어 factory가 profile을 찾게 한다."""
    from so101_contract.action_dataset_contract import resolve_action_representation_contract
    from so101_contract.action_representation import ActionRepresentationMode, ActionRepresentationSpec, PoseFormat
    from so101_contract.action_representation_stats import (
        ActionStatsSampling,
        calculate_action_representation_stats,
        empty_stats_artifact,
        load_lerobot_v3_episodes,
        upsert_stats_profile,
        write_action_stats_artifact,
    )

    spec = ActionRepresentationSpec(
        mode=ActionRepresentationMode.EEF_RELATIVE,
        pose_format=PoseFormat.XYZ_ROT6D_ROWS,
    )
    contract = resolve_action_representation_contract(dataset_root, spec)
    episodes = load_lerobot_v3_episodes(
        dataset_root,
        state_key=contract.state_key,
        action_key=contract.action_key,
        state_dim=contract.state_dim,
        action_dim=contract.action_dim,
    )
    result = calculate_action_representation_stats(
        episodes,
        ActionStatsSampling(action_delta_indices=tuple(range(horizon))),
        contract.transform,
        dataset_fingerprint=contract.fingerprint,
    )
    artifact, _ = upsert_stats_profile(empty_stats_artifact(), result)
    write_action_stats_artifact(dataset_root, artifact, output_file=spec.stats_file)
    return result


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("SmolVLA 1-batch validation requires CUDA")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    scratch = ROOT / "scratch"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="smolvla-eef-train-", dir=scratch) as directory:
        dataset_root = Path(directory)
        _write_self_check_dataset(dataset_root)
        sampling = RelativeActionSamplingConfig(action_delta_indices=(0, 1, 2, 3))
        stats_result = calculate_relative_action_stats(
            dataset_root,
            sampling,
            scratch_dir=scratch,
        )
        write_relative_action_stats_profile(dataset_root, stats_result)
        _write_v2_stats(dataset_root, 4)  # 아래 config.chunk_size와 동일

        config = PreTrainedConfig.from_pretrained(MODEL_ID, local_files_only=True)
        if not isinstance(config, SmolVLAConfig):
            raise TypeError(f"unexpected SmolVLA config: {type(config).__name__}")
        config.device = "cuda"
        config.chunk_size = 4
        config.n_action_steps = 2
        config.num_steps = 1
        config.num_vlm_layers = 1
        config.num_expert_layers = 1
        config.load_vlm_weights = False
        config.resize_imgs_with_padding = (64, 64)
        config.input_features = {
            **config.input_features,
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,)),
        }
        config.output_features = {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(10,)),
        }
        config.action_representation = ActionRepresentationConfig(mode="eef_relative", pose_format="xyz_rot6d_rows")

        dataset_stats = {
            OBS_STATE: _stats(10),
            ACTION: _stats(10),
        }
        preprocessor, _ = make_pre_post_processors(
            config,
            pretrained_path=MODEL_ID,
            dataset_stats=dataset_stats,
            dataset_meta=SimpleNamespace(root=dataset_root),
        )
        normalizer = next(
            step for step in preprocessor.steps if step.__class__._registry_name == "normalizer_processor"
        )
        if normalizer.features[ACTION].shape != (10,):
            raise AssertionError(
                "pretrained SmolVLA normalizer kept stale joint action schema: "
                f"{normalizer.features[ACTION]}"
            )

        policy = SmolVLAPolicy.from_pretrained(
            MODEL_ID,
            config=config,
            local_files_only=True,
            strict=False,
        )
        policy.train()
        batch_size = 1
        identity = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32)
        state = torch.zeros(batch_size, 10)
        state[:, :3] = torch.tensor([[0.10, 0.20, 0.30]])
        state[:, 3:9] = identity
        state[:, 9] = 40.0
        action = state[:, None, :].repeat(1, config.chunk_size, 1)
        action[:, :, 0] += torch.tensor([0.01, 0.02, 0.03, 0.04])
        action[:, :, 9] = torch.tensor([20.0, 40.0, 60.0, 80.0])
        raw_batch = {
            OBS_STATE: state,
            ACTION: action,
            "action_is_pad": torch.zeros(batch_size, config.chunk_size, dtype=torch.bool),
            "task": ["pick up the cube and place it in the bowl"],
        }
        for key in config.image_features:
            raw_batch[key] = torch.rand(batch_size, 3, 64, 64)

        processed = preprocessor(raw_batch)
        if processed[ACTION].shape != (batch_size, config.chunk_size, 10):
            raise AssertionError(f"processed SmolVLA action shape mismatch: {processed[ACTION].shape}")
        loss, _ = policy(processed)
        if not torch.isfinite(loss):
            raise AssertionError(f"SmolVLA EEF-relative loss is not finite: {loss}")
        loss.backward()
        if not any(
            parameter.grad is not None and torch.all(torch.isfinite(parameter.grad))
            for parameter in policy.parameters()
        ):
            raise AssertionError("SmolVLA EEF-relative backward produced no finite gradients")

    print("[smolvla-eef-training] PASS")


if __name__ == "__main__":
    main()
