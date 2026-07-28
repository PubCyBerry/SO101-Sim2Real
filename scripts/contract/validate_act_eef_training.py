#!/usr/bin/env python3
"""LeRobot v0.6 ACT + 공통 EEF-relative processor 1-batch 학습/추론 검증."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "data"))

from generate_relative_action_stats import _write_self_check_dataset  # noqa: E402
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402
from lerobot.configs.action_representation import ActionRepresentationConfig  # noqa: E402
from lerobot.policies import make_pre_post_processors  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE  # noqa: E402
from so101_contract.eef_relative_stats import (  # noqa: E402
    RelativeActionSamplingConfig,
    calculate_relative_action_stats,
    write_relative_action_stats_profile,
)
from so101_contract.lerobot_full_chunk import FullChunkPolicyRunner  # noqa: E402


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
    if "action_representation" not in getattr(ACTConfig, "__dataclass_fields__", {}):
        raise RuntimeError("LeRobot v0.6 EEF-relative patch is not installed")
    scratch = ROOT / "scratch"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="act-eef-train-", dir=scratch) as directory:
        dataset_root = Path(directory)
        _write_self_check_dataset(dataset_root)
        sampling = RelativeActionSamplingConfig(action_delta_indices=(0, 1, 2))
        stats_result = calculate_relative_action_stats(
            dataset_root,
            sampling,
            scratch_dir=scratch,
        )
        write_relative_action_stats_profile(dataset_root, stats_result)
        v2_stats = _write_v2_stats(dataset_root, 3)

        config = ACTConfig(
            chunk_size=3,
            n_action_steps=2,
            input_features={
                OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,)),
                OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(1,)),
            },
            output_features={
                ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(10,)),
            },
            normalization_mapping={
                "STATE": NormalizationMode.MEAN_STD,
                "ENV": NormalizationMode.IDENTITY,
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
            action_representation=ActionRepresentationConfig(mode="eef_relative", pose_format="xyz_rot6d_rows"),
        )
        dataset_stats = {
            OBS_STATE: _stats(10),
            OBS_ENV_STATE: _stats(1),
            ACTION: _stats(10),
        }
        preprocessor, postprocessor = make_pre_post_processors(
            config,
            dataset_stats=dataset_stats,
            dataset_meta=SimpleNamespace(root=dataset_root),
        )

        model = ACTPolicy(config)
        batch_size = 2
        identity = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32)
        state = torch.zeros(batch_size, 10)
        state[:, :3] = torch.tensor([[0.10, 0.20, 0.30], [-0.10, 0.25, 0.35]])
        state[:, 3:9] = identity
        state[:, 9] = torch.tensor([30.0, 70.0])
        action = state[:, None, :].repeat(1, config.chunk_size, 1)
        action[:, :, 0] += torch.tensor([0.01, 0.02, 0.03])
        action[:, :, 9] = torch.tensor([25.0, 50.0, 75.0])
        raw_batch = {
            OBS_STATE: state,
            OBS_ENV_STATE: torch.zeros(batch_size, 1),
            ACTION: action,
            "action_is_pad": torch.zeros(batch_size, config.chunk_size, dtype=torch.bool),
        }
        processed = preprocessor(raw_batch)
        if processed[ACTION].shape != (batch_size, config.chunk_size, 10):
            raise AssertionError(f"processed action shape mismatch: {processed[ACTION].shape}")
        loss, _ = model(processed)
        if not torch.isfinite(loss):
            raise AssertionError(f"ACT EEF-relative loss is not finite: {loss}")
        loss.backward()
        if not any(
            parameter.grad is not None and torch.all(torch.isfinite(parameter.grad))
            for parameter in model.parameters()
        ):
            raise AssertionError("ACT EEF-relative backward produced no finite gradients")

        # 실제 ACT predict_action_chunk + full postprocessor가 한 anchor state로 동작하는지 확인.
        runner = FullChunkPolicyRunner(
            model,
            preprocessor,
            postprocessor,
            execution_horizon=config.n_action_steps,
        )
        inference_observation = {
            OBS_STATE: state[:1],
            OBS_ENV_STATE: torch.zeros(1, 1),
        }
        with torch.inference_mode():
            first = runner.next_action(inference_observation)
            second = runner.next_action(
                {
                    OBS_STATE: state[:1] + 100.0,
                    OBS_ENV_STATE: torch.zeros(1, 1),
                }
            )
        if first.shape != (1, 10) or second.shape != (1, 10):
            raise AssertionError(f"ACT full-chunk output shape mismatch: {first.shape}/{second.shape}")
        metrics = runner.metrics
        if metrics.preprocessor_calls != 1 or metrics.postprocessor_calls != 1:
            raise AssertionError(f"ACT full chunk was reprocessed per tick: {metrics}")

        # Relative stats가 absolute dataset action stats를 실제로 대체했는지 수치로 확인한다.
        relative_mean = np.asarray(v2_stats.profile["stats"]["action"]["mean"])
        normalizer = next(
            step for step in preprocessor.steps if step.__class__._registry_name == "normalizer_processor"
        )
        injected_mean = np.asarray(normalizer.stats[ACTION]["mean"])
        np.testing.assert_allclose(injected_mean, relative_mean, atol=0.0)

    print("[act-eef-training] PASS")


if __name__ == "__main__":
    main()
