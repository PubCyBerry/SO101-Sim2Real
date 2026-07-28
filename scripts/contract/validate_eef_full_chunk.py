#!/usr/bin/env python3
"""EEF-relative full-chunk runner의 cache/call-count/queue 회귀 검증."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from so101_contract.eef_relative_action import relative_actions_to_absolute  # noqa: E402
from so101_contract.lerobot_full_chunk import FullChunkPolicyRunner  # noqa: E402


class _Preprocessor:
    def __init__(self) -> None:
        self.steps = [object()]
        self.state = None
        self.calls = 0

    def __call__(self, observation):
        self.calls += 1
        self.state = observation["observation.state"].clone()
        return observation

    def reset(self):
        self.state = None


class _Postprocessor:
    def __init__(self, preprocessor: _Preprocessor) -> None:
        self.steps = [object()]
        self.preprocessor = preprocessor
        self.calls = 0

    def __call__(self, action):
        self.calls += 1
        return relative_actions_to_absolute(self.preprocessor.state, action)

    def reset(self):
        pass


class _Policy:
    config = SimpleNamespace(
        n_action_steps=3,
        temporal_ensemble_coeff=None,
        rtc_config=None,
        use_relative_actions=False,
    )

    def __init__(self) -> None:
        self.calls = 0

    def predict_action_chunk(self, observation):
        self.calls += 1
        batch = observation["observation.state"].shape[0]
        identity_rows = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32)
        chunk = torch.zeros(batch, 4, 10)
        chunk[..., 3:9] = identity_rows
        chunk[..., 0] = torch.arange(4, dtype=torch.float32) * 0.01
        chunk[..., 9] = 50.0
        return chunk

    def reset(self):
        pass


def _absolute_state(x: float) -> torch.Tensor:
    state = torch.tensor(
        [[x, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 50.0]],
        dtype=torch.float32,
    )
    return state


def _validate_async_server_branching() -> None:
    """EEF만 1회 full-chunk, absolute fallback은 기존 step별 호출인지 확인."""
    import lerobot.async_inference.policy_server as server_module
    from lerobot.async_inference.helpers import TimedObservation

    server = object.__new__(server_module.PolicyServer)
    server.config = SimpleNamespace(environment_dt=1.0 / 30.0)
    server.lerobot_features = {}
    server.actions_per_chunk = 3
    server.last_processed_obs = None
    # patch가 `_load_policy()`에서 만드는 상태. 이 fixture는 `__init__`을 건너뛰므로
    # 직접 채워 준다(카메라 없는 케이스 = 빈 dict).
    server._raw_policy_image_features = {}
    server._full_chunk_actions = False

    class _AsyncPolicy:
        config = SimpleNamespace(image_features={})

        @staticmethod
        def predict_action_chunk(_observation):
            return torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)

    class _AsyncPostprocessor:
        def __init__(self, expected_rank: int, offset: float) -> None:
            self.expected_rank = expected_rank
            self.offset = offset
            self.calls: list[tuple[int, ...]] = []

        def __call__(self, action):
            if action.ndim != self.expected_rank:
                raise AssertionError(
                    f"postprocessor rank mismatch: {action.ndim} != {self.expected_rank}"
                )
            self.calls.append(tuple(action.shape))
            return action + self.offset

    server.policy = _AsyncPolicy()
    server.preprocessor = lambda observation: observation
    observation = TimedObservation(
        timestamp=10.0,
        timestep=7,
        observation={"observation.state": torch.zeros(1, 10)},
    )
    original_converter = server_module.raw_observation_to_observation
    server_module.raw_observation_to_observation = lambda raw, *_args: raw
    try:
        eef_postprocessor = _AsyncPostprocessor(expected_rank=3, offset=1.0)
        server.postprocessor = eef_postprocessor
        server._full_chunk_actions = True
        eef_actions = server._predict_action_chunk(observation)
        if eef_postprocessor.calls != [(1, 4, 2)] or len(eef_actions) != 3:
            raise AssertionError(
                "EEF async postprocessor must receive one unsliced full chunk: "
                f"calls={eef_postprocessor.calls}, returned={len(eef_actions)}"
            )

        absolute_postprocessor = _AsyncPostprocessor(expected_rank=2, offset=2.0)
        server.postprocessor = absolute_postprocessor
        server._full_chunk_actions = False
        absolute_actions = server._predict_action_chunk(observation)
        if absolute_postprocessor.calls != [(1, 2), (1, 2), (1, 2)]:
            raise AssertionError(
                "absolute async fallback must preserve stock step-wise postprocessing: "
                f"{absolute_postprocessor.calls}"
            )
        if len(absolute_actions) != 3:
            raise AssertionError("absolute async fallback did not honor actions_per_chunk")
    finally:
        server_module.raw_observation_to_observation = original_converter


def main() -> None:
    # Unit fixture에서는 실제 LeRobot step class가 없으므로 pair detector만 이 테스트 범위에서 대체한다.
    import so101_contract.lerobot_full_chunk as module

    original = module.has_eef_relative_processor_steps
    module.has_eef_relative_processor_steps = lambda _pre, _post: True
    try:
        preprocessor = _Preprocessor()
        postprocessor = _Postprocessor(preprocessor)
        policy = _Policy()
        runner = FullChunkPolicyRunner(policy, preprocessor, postprocessor)

        first = runner.next_action({"observation.state": _absolute_state(0.1)})
        # Queue를 소비하는 동안 새 observation을 전달해도 pre/cache가 갱신되면 안 된다.
        second = runner.next_action({"observation.state": _absolute_state(9.0)})
        third = runner.next_action({"observation.state": _absolute_state(9.0)})
        torch.testing.assert_close(first[:, 0], torch.tensor([0.1]))
        torch.testing.assert_close(second[:, 0], torch.tensor([0.11]))
        torch.testing.assert_close(third[:, 0], torch.tensor([0.12]))
        if preprocessor.calls != 1 or postprocessor.calls != 1 or policy.calls != 1:
            raise AssertionError(
                "full chunk must call pre/policy/post exactly once: "
                f"{preprocessor.calls}/{policy.calls}/{postprocessor.calls}"
            )

        fourth = runner.next_action({"observation.state": _absolute_state(0.5)})
        torch.testing.assert_close(fourth[:, 0], torch.tensor([0.5]))
        metrics = runner.metrics
        if (
            metrics.chunks_predicted != 2
            or metrics.preprocessor_calls != 2
            or metrics.postprocessor_calls != 2
            or metrics.actions_dequeued != 4
        ):
            raise AssertionError(f"unexpected runner metrics: {metrics}")
    finally:
        module.has_eef_relative_processor_steps = original
    _validate_async_server_branching()
    print("[eef-full-chunk] PASS")


if __name__ == "__main__":
    main()
