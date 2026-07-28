#!/usr/bin/env python3
"""Cached GR00T-N1.7 base + 공통 EEF-relative 1-batch 학습 검증."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lerobot.configs import FeatureType, PolicyFeature  # noqa: E402
from lerobot.configs.action_representation import ActionRepresentationConfig  # noqa: E402
from lerobot.policies import make_pre_post_processors  # noqa: E402
from lerobot.policies.groot.configuration_groot import GrootConfig  # noqa: E402
from lerobot.policies.groot.modeling_groot import GrootPolicy  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402
from so101_contract.eef_action_contract import CANONICAL_ACTION_NAMES  # noqa: E402
from so101_contract.eef_relative_action import (  # noqa: E402
    matrix_to_rot6d_rows,
    relative_actions_to_absolute,
)
from so101_contract.eef_relative_stats import (  # noqa: E402
    RelativeActionSamplingConfig,
    calculate_relative_action_stats,
    write_relative_action_stats_profile,
)

MODEL_ID = "nvidia/GR00T-N1.7-3B"
HORIZON = 40


def _numeric_stats(values: np.ndarray) -> dict[str, list]:
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": np.maximum(values.std(axis=0), 1e-4).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def _write_dataset(root: Path) -> tuple[np.ndarray, np.ndarray]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    frame_count = HORIZON + 4
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    feature = {
        "dtype": "float32",
        "shape": [10],
        "names": list(CANONICAL_ACTION_NAMES),
    }
    info = {
        "codebase_version": "v3.0",
        "fps": 30,
        "total_episodes": 1,
        "total_frames": frame_count,
        "features": {
            OBS_STATE: feature,
            ACTION: feature,
        },
        "so101_eef_conversion": {
            "base_frame": "base_link",
            "eef_frame": "tcp_grasp",
            "eef_kinematics_version": "so101_base_tcp_grasp_fk_v2",
            "rotation_representation": "rot6d",
            "rotation_format": "xyz+rot6d_rows",
            "gripper_format": "canonical_policy_feature_[0,100]",
            "keep_joints": False,
            "urdf_sha256": "a" * 64,
            "robot_yaml_sha256": "b" * 64,
        },
    }
    modality = {
        name: {
            "eef_9d": {"start": 0, "end": 9},
            "gripper_position": {"start": 9, "end": 10},
        }
        for name in ("state", "action")
    }
    (root / "meta" / "info.json").write_text(
        json.dumps(info, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "meta" / "modality.json").write_text(
        json.dumps(modality, indent=2) + "\n",
        encoding="utf-8",
    )

    frames = np.arange(frame_count, dtype=np.int64)
    angles = np.linspace(-0.2, 0.2, frame_count, dtype=np.float32)
    rotations = np.zeros((frame_count, 3, 3), dtype=np.float32)
    rotations[:, 0, 0] = np.cos(angles)
    rotations[:, 0, 1] = -np.sin(angles)
    rotations[:, 1, 0] = np.sin(angles)
    rotations[:, 1, 1] = np.cos(angles)
    rotations[:, 2, 2] = 1.0
    states = np.concatenate(
        [
            np.stack(
                [
                    frames.astype(np.float32) * 0.001,
                    np.full(frame_count, 0.20, dtype=np.float32),
                    np.full(frame_count, 0.25, dtype=np.float32),
                ],
                axis=1,
            ),
            matrix_to_rot6d_rows(rotations),
            np.linspace(20.0, 80.0, frame_count, dtype=np.float32)[:, None],
        ],
        axis=1,
    ).astype(np.float32)
    relative = np.zeros((frame_count, 1, 10), dtype=np.float32)
    relative[:, 0, :3] = np.asarray([0.002, -0.001, 0.001], dtype=np.float32)
    relative[:, 0, 3:9] = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
    relative[:, 0, 9] = np.linspace(25.0, 75.0, frame_count, dtype=np.float32)
    actions = relative_actions_to_absolute(states, relative)[:, 0]
    table = pa.table(
        {
            "episode_index": pa.array(np.zeros(frame_count, dtype=np.int64)),
            "frame_index": pa.array(frames),
            OBS_STATE: pa.array(states.tolist(), type=pa.list_(pa.float32(), 10)),
            ACTION: pa.array(actions.tolist(), type=pa.list_(pa.float32(), 10)),
        }
    )
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")
    return states, actions


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
        raise RuntimeError("GR00T-N1.7 validation requires CUDA")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    scratch = ROOT / "scratch"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="groot-n17-eef-train-", dir=scratch) as directory:
        dataset_root = Path(directory)
        states, absolute_actions = _write_dataset(dataset_root)
        sampling = RelativeActionSamplingConfig(
            action_delta_indices=tuple(range(HORIZON))
        )
        stats_result = calculate_relative_action_stats(
            dataset_root,
            sampling,
            scratch_dir=scratch,
        )
        write_relative_action_stats_profile(dataset_root, stats_result)
        _write_v2_stats(dataset_root, HORIZON)

        image_key = "observation.images.top"
        config = GrootConfig(
            base_model_path=MODEL_ID,
            embodiment_tag="new_embodiment",
            chunk_size=HORIZON,
            n_action_steps=16,
            input_features={
                OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,)),
                image_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            },
            output_features={
                ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(10,)),
            },
            device="cuda",
            use_bf16=True,
            model_params_fp32=False,
            tune_llm=False,
            tune_visual=False,
            tune_projector=True,
            tune_diffusion_model=True,
            tune_vlln=True,
            num_inference_timesteps=1,
            use_relative_actions=False,
            action_representation=ActionRepresentationConfig(mode="eef_relative", pose_format="xyz_rot6d_rows"),
        )
        dataset_stats = {
            OBS_STATE: _numeric_stats(states),
            ACTION: _numeric_stats(absolute_actions),
        }
        dataset_meta = SimpleNamespace(
            root=dataset_root,
            features={
                OBS_STATE: {
                    "dtype": "float32",
                    "shape": (10,),
                    "names": list(CANONICAL_ACTION_NAMES),
                },
                ACTION: {
                    "dtype": "float32",
                    "shape": (10,),
                    "names": list(CANONICAL_ACTION_NAMES),
                },
                image_key: {
                    "dtype": "video",
                    "shape": (3, 256, 256),
                    "names": None,
                },
            },
            stats=dataset_stats,
            fps=30,
            repo_id="local/groot_n17_eef_fixture",
            revision=None,
        )
        preprocessor, _ = make_pre_post_processors(
            config,
            dataset_stats=dataset_stats,
            dataset_meta=dataset_meta,
        )
        registry_names = [
            getattr(step.__class__, "_registry_name", None)
            for step in preprocessor.steps
        ]
        relative_index = registry_names.index("so101_action_representation_encode_v2")
        pack_index = registry_names.index("groot_n1_7_pack_inputs_v1")
        if relative_index >= pack_index:
            raise AssertionError(f"GR00T common relative step is after pack: {registry_names}")

        policy = GrootPolicy(config).to(config.device)
        policy.train()
        state = torch.as_tensor(states[:1], dtype=torch.float32)
        action_indices = np.arange(HORIZON)
        action = torch.as_tensor(
            absolute_actions[action_indices][None, :, :],
            dtype=torch.float32,
        )
        raw_batch = {
            OBS_STATE: state,
            ACTION: action,
            "action_is_pad": torch.zeros(1, HORIZON, dtype=torch.bool),
            image_key: torch.rand(1, 3, 256, 256),
            "task": ["pick up the cube and place it in the bowl"],
        }
        processed = preprocessor(raw_batch)
        if processed["action"].shape != (1, HORIZON, config.max_action_dim):
            raise AssertionError(f"GR00T packed action shape mismatch: {processed['action'].shape}")
        loss, _ = policy(processed)
        if not torch.isfinite(loss):
            raise AssertionError(f"GR00T-N1.7 EEF-relative loss is not finite: {loss}")
        loss.backward()
        if not any(
            parameter.grad is not None and torch.all(torch.isfinite(parameter.grad))
            for parameter in policy.parameters()
        ):
            raise AssertionError("GR00T-N1.7 EEF-relative backward produced no finite gradients")

    print("[groot-n17-eef-training] PASS")


if __name__ == "__main__":
    main()
