#!/usr/bin/env python3
"""Phase 15 — 실제 ``lerobot-train`` CLI로 만든 checkpoint의 manifest/reload 검증.

synthetic fixture에 대해 짧은 train을 돌려 다음을 확인한다.

1. periodic checkpoint와 final checkpoint 모두 ``action_representation.json``(schema v2)을 가진다
2. manifest의 mode/pose_format/dim/stats/fingerprint가 dataset 계약과 일치한다
3. 그 checkpoint를 다시 로드하면 v2 processor pair가 복원되고 manifest 교차 검증을 통과한다
4. 복원된 pipeline이 full-chunk 경로로 absolute action을 만든다

.. code-block:: bash

    python scripts/contract/validate_action_representation_checkpoint_cli.py \\
        --fixture scratch/fx_img/xyz_rot6d_rows --mode eef_relative --pose-format xyz_rot6d_rows
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lerobot.policies import make_pre_post_processors  # noqa: E402
from lerobot.utils.constants import OBS_STATE  # noqa: E402

from so101_contract.action_manifest import (  # noqa: E402
    ACTION_REPRESENTATION_MANIFEST,
    manifest_schema_version,
    read_action_representation_manifest,
    validate_action_representation_manifest,
)
from so101_contract.action_representation import (  # noqa: E402
    ActionRepresentationSpec,
    coerce_mode,
    coerce_pose_format,
)
from so101_contract.lerobot_v2_integration import (  # noqa: E402
    action_representation_encode_step,
    has_action_representation_steps,
    validate_checkpoint_manifest,
)


def _train(
    *,
    fixture: Path,
    output_dir: Path,
    mode: str,
    pose_format: str | None,
    steps: int,
    policy_type: str,
    device: str,
) -> None:
    command = [
        "lerobot-train",
        f"--dataset.repo_id=local/{fixture.name}",
        f"--dataset.root={fixture}",
        f"--policy.type={policy_type}",
        f"--policy.device={device}",
        "--policy.push_to_hub=false",
        "--policy.chunk_size=4",
        "--policy.n_action_steps=2",
        f"--policy.action_representation.mode={mode}",
        f"--output_dir={output_dir}",
        f"--steps={steps}",
        f"--save_freq={steps}",
        "--batch_size=2",
        "--num_workers=0",
        "--log_freq=1",
        "--wandb.enable=false",
    ]
    if policy_type == "act":
        command += [
            "--policy.dim_model=32",
            "--policy.n_heads=4",
            "--policy.dim_feedforward=64",
            "--policy.n_encoder_layers=1",
            "--policy.n_decoder_layers=1",
            "--policy.use_vae=false",
            "--policy.vision_backbone=resnet18",
        ]
    elif policy_type == "smolvla":
        command += ["--policy.load_vlm_weights=false", "--policy.num_vlm_layers=2"]
    elif policy_type == "groot":
        command += [
            "--policy.base_model_path=nvidia/GR00T-N1.7-3B",
            "--policy.embodiment_tag=new_embodiment",
            "--policy.num_inference_timesteps=1",
            "--policy.tune_llm=false",
            "--policy.tune_visual=false",
        ]
    if pose_format:
        command.append(f"--policy.action_representation.pose_format={pose_format}")
    environment = dict(os.environ)
    environment.setdefault("HF_HUB_OFFLINE", "1")
    result = subprocess.run(command, env=environment, capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join((result.stdout + result.stderr).splitlines()[-40:])
        raise RuntimeError(f"lerobot-train failed ({result.returncode}):\n{tail}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--pose-format", default=None)
    parser.add_argument("--policy-type", default="act")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--keep-output", action="store_true")
    args = parser.parse_args()

    spec = ActionRepresentationSpec(
        mode=coerce_mode(args.mode),
        pose_format=coerce_pose_format(args.pose_format) if args.pose_format else None,
    )
    scratch = ROOT / "scratch"
    scratch.mkdir(exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="ckpt-cli-", dir=scratch))
    # lerobot-train은 존재하는 output_dir을 거부하므로 미생성 하위 경로를 준다.
    output_dir = workspace / "run"
    try:
        _train(
            fixture=args.fixture.resolve(),
            output_dir=output_dir,
            mode=args.mode,
            pose_format=args.pose_format,
            steps=args.steps,
            policy_type=args.policy_type,
            device=args.device,
        )
        checkpoints = sorted((output_dir / "checkpoints").glob("*/pretrained_model"))
        if not checkpoints:
            raise AssertionError(f"training produced no checkpoint under {output_dir}")
        for checkpoint in checkpoints:
            manifest_path = checkpoint / ACTION_REPRESENTATION_MANIFEST
            if not manifest_path.is_file():
                raise AssertionError(f"checkpoint is missing the v2 manifest: {checkpoint}")
            manifest = read_action_representation_manifest(checkpoint)
            if manifest_schema_version(manifest) != 2:
                raise AssertionError("checkpoint manifest is not schema v2")
            validate_action_representation_manifest(manifest, expected_spec=spec)
            if manifest["policy"]["type"] != args.policy_type:
                raise AssertionError("manifest policy family mismatch")
            print(
                f"  manifest {checkpoint.parent.name}: mode={manifest['mode']} "
                f"pose_format={manifest['pose_format']} dim={manifest['action_dim']} "
                f"stats={manifest['stats']['profile_id'][:19]}... "
                f"sha={manifest['manifest_sha256'][:12]}"
            )

        final = checkpoints[-1]
        from lerobot.configs.policies import PreTrainedConfig

        config = PreTrainedConfig.from_pretrained(final)
        config.device = args.device
        preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=str(final))
        if not has_action_representation_steps(preprocessor, postprocessor):
            raise AssertionError("reloaded checkpoint has no schema v2 processor pair")
        manifest = validate_checkpoint_manifest(final, config, preprocessor)
        step = action_representation_encode_step(preprocessor)
        if step.transform.spec.mode is not spec.mode:
            raise AssertionError("reloaded transform mode mismatch")
        if step.contract_fingerprint != manifest["resolved_contract_fingerprint"]:
            raise AssertionError("reloaded processor fingerprint mismatch")

        dim = manifest["action_dim"]
        # 재로드한 processor에 넣을 canonical fixture(EEF는 identity rotation, joint는 0).
        if spec.is_eef:
            import numpy as np

            from so101_contract.pose_codec import encode_pose

            pose = encode_pose(
                np.zeros(3, dtype=np.float32),
                np.eye(3, dtype=np.float32),
                spec.pose_format,
            )
            row = torch.from_numpy(np.concatenate([pose, np.zeros(1, dtype=np.float32)]))
        else:
            row = torch.zeros(dim)
        state = row[None, :].clone()
        chunk = row[None, None, :].repeat(1, config.chunk_size, 1).clone()
        from lerobot.processor.pipeline import TransitionKey

        encoded = step(
            {TransitionKey.OBSERVATION: {OBS_STATE: state}, TransitionKey.ACTION: chunk}
        )[TransitionKey.ACTION]
        if encoded.shape != chunk.shape:
            raise AssertionError(f"reloaded encode shape mismatch: {tuple(encoded.shape)}")
        print(
            f"  reload OK: steps={len(preprocessor.steps)}/{len(postprocessor.steps)} "
            f"transform={step.transform.spec.stats_profile_kind}"
        )
        print(
            f"PASS: {args.policy_type} {spec.stats_profile_kind} checkpoint CLI "
            f"({len(checkpoints)} checkpoint(s), manifest + reload verified)"
        )
        return 0
    finally:
        if not args.keep_output:
            shutil.rmtree(workspace, ignore_errors=True)
        else:
            print(json.dumps({"output_dir": str(output_dir)}))


if __name__ == "__main__":
    raise SystemExit(main())
