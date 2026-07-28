#!/usr/bin/env python3
"""LeRobot EEF-relative pre/post processor의 chunk·직렬화 계약 검증."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    from lerobot.types import TransitionKey
except ImportError:
    from lerobot.processor.pipeline import TransitionKey

from lerobot.processor.pipeline import DataProcessorPipeline
from lerobot.utils.constants import OBS_STATE

from so101_contract.lerobot_eef_processor import (  # noqa: E402
    SE3AbsoluteActionsProcessorStep,
    SE3RelativeActionsProcessorStep,
    make_eef_relative_processor_steps,
    reconnect_eef_relative_processor_steps,
)
from so101_contract.lerobot_policy_integration import (  # noqa: E402
    attach_eef_relative_processor_steps,
    has_eef_relative_processor_steps,
)


class _BoundaryStep:
    def __init__(self, registry_name: str):
        self.__class__ = type(
            f"Boundary_{registry_name}",
            (),
            {"_registry_name": registry_name},
        )


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    identity = torch.tensor(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        dtype=torch.float32,
    )
    state = torch.zeros((2, 10), dtype=torch.float32)
    state[:, :3] = torch.tensor([[0.1, 0.2, 0.3], [-0.2, 0.4, 0.5]])
    state[:, 3:9] = identity
    state[:, 9] = torch.tensor([25.0, 75.0])

    action = state[:, None, :].repeat(1, 4, 1)
    action[:, :, 0] += torch.tensor([0.01, 0.02, 0.03, 0.04])
    action[:, :, 2] += torch.tensor([0.04, 0.03, 0.02, 0.01])
    action[:, :, 9] = torch.tensor([10.0, 20.0, 30.0, 40.0])
    return state, action


def _roundtrip(
    relative_step: SE3RelativeActionsProcessorStep,
    absolute_step: SE3AbsoluteActionsProcessorStep,
) -> None:
    state, action = _inputs()
    relative_transition = relative_step(
        {
            TransitionKey.OBSERVATION: {OBS_STATE: state},
            TransitionKey.ACTION: action,
        }
    )
    relative = relative_transition[TransitionKey.ACTION]
    if not torch.equal(relative[..., 9], action[..., 9]):
        raise AssertionError("absolute gripper passthrough changed")
    restored = absolute_step({TransitionKey.ACTION: relative})[TransitionKey.ACTION]
    if not torch.allclose(restored, action, atol=2e-6, rtol=2e-6):
        raise AssertionError("LeRobot EEF processor round-trip mismatch")


def _serialization_check() -> None:
    fingerprint = "a" * 64
    relative_step, absolute_step = make_eef_relative_processor_steps(
        state_pose_indices=tuple(range(9)),
        action_pose_indices=tuple(range(9)),
        passthrough_action_indices=(9,),
        contract_fingerprint=fingerprint,
        manifest_context={"contract_fingerprint": fingerprint},
    )
    preprocessor = DataProcessorPipeline(
        steps=[relative_step],
        name="policy_preprocessor",
    )
    postprocessor = DataProcessorPipeline(
        steps=[absolute_step],
        name="policy_postprocessor",
    )
    scratch = REPO_ROOT / "scratch"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="eef-processor-check-", dir=scratch) as directory:
        root = Path(directory)
        preprocessor.save_pretrained(root, config_filename="policy_preprocessor.json")
        postprocessor.save_pretrained(root, config_filename="policy_postprocessor.json")
        loaded_pre = DataProcessorPipeline.from_pretrained(
            root,
            config_filename="policy_preprocessor.json",
            local_files_only=True,
        )
        loaded_post = DataProcessorPipeline.from_pretrained(
            root,
            config_filename="policy_postprocessor.json",
            local_files_only=True,
        )
        reconnect_eef_relative_processor_steps(loaded_pre, loaded_post)
        loaded_relative = loaded_pre.steps[0]
        loaded_absolute = loaded_post.steps[0]
        if not isinstance(loaded_relative, SE3RelativeActionsProcessorStep):
            raise AssertionError("serialized relative processor type mismatch")
        if not isinstance(loaded_absolute, SE3AbsoluteActionsProcessorStep):
            raise AssertionError("serialized absolute processor type mismatch")
        _roundtrip(loaded_relative, loaded_absolute)


def _fail_fast_check() -> None:
    fingerprint = "b" * 64
    relative_step, absolute_step = make_eef_relative_processor_steps(
        state_pose_indices=tuple(range(9)),
        action_pose_indices=tuple(range(9)),
        passthrough_action_indices=(9,),
        contract_fingerprint=fingerprint,
        manifest_context={"contract_fingerprint": fingerprint},
    )
    state, action = _inputs()
    relative_step(
        {
            TransitionKey.OBSERVATION: {OBS_STATE: state},
            TransitionKey.ACTION: action,
        }
    )
    try:
        absolute_step({TransitionKey.ACTION: action[:, 0]})
    except NotImplementedError as exc:
        if "full (B,H,D) chunk" not in str(exc):
            raise
    else:
        raise AssertionError("single-step EEF-relative decode was accepted")

    relative_step.reset()
    try:
        absolute_step({TransitionKey.ACTION: action})
    except RuntimeError as exc:
        if "prediction-time state" not in str(exc):
            raise
    else:
        raise AssertionError("decode without cached prediction-time state was accepted")

    try:
        SE3RelativeActionsProcessorStep(contract_fingerprint="")
    except ValueError as exc:
        if "fingerprint" not in str(exc):
            raise
    else:
        raise AssertionError("strict processor accepted an empty contract fingerprint")


def _policy_integration_order_check() -> None:
    context = SimpleNamespace(
        representation=SimpleNamespace(strict=True),
        contract=SimpleNamespace(
            state_pose_indices=tuple(range(9)),
            action_pose_indices=tuple(range(9)),
            passthrough_action_indices=(9,),
            fingerprint="c" * 64,
        ),
        manifest_context={"contract_fingerprint": "c" * 64},
    )
    for boundary in ("normalizer_processor", "groot_n1_7_pack_inputs_v1"):
        preprocessor = SimpleNamespace(
            steps=[
                _BoundaryStep("to_batch_processor"),
                _BoundaryStep(boundary),
            ]
        )
        postprocessor = SimpleNamespace(
            steps=[
                _BoundaryStep("unnormalizer_processor"),
                _BoundaryStep("device_processor"),
            ]
        )
        attach_eef_relative_processor_steps(preprocessor, postprocessor, context)
        pre_names = [
            getattr(step.__class__, "_registry_name", None)
            for step in preprocessor.steps
        ]
        post_names = [
            getattr(step.__class__, "_registry_name", None)
            for step in postprocessor.steps
        ]
        if pre_names != [
            "to_batch_processor",
            "so101_se3_relative_actions_processor",
            boundary,
        ]:
            raise AssertionError(f"preprocessor insertion order mismatch: {pre_names}")
        if post_names != [
            "unnormalizer_processor",
            "so101_se3_absolute_actions_processor",
            "device_processor",
        ]:
            raise AssertionError(f"postprocessor insertion order mismatch: {post_names}")
        if not has_eef_relative_processor_steps(preprocessor, postprocessor):
            raise AssertionError("attached EEF processor pair was not detected")
        reconnect_eef_relative_processor_steps(preprocessor, postprocessor)


def main() -> None:
    relative_step, absolute_step = make_eef_relative_processor_steps(
        state_pose_indices=tuple(range(9)),
        action_pose_indices=tuple(range(9)),
        passthrough_action_indices=(9,),
        contract_fingerprint="0" * 64,
        manifest_context={"contract_fingerprint": "0" * 64},
    )
    _roundtrip(relative_step, absolute_step)
    _serialization_check()
    _fail_fast_check()
    _policy_integration_order_check()
    print("PASS: LeRobot full-chunk EEF-relative processor")


if __name__ == "__main__":
    main()
