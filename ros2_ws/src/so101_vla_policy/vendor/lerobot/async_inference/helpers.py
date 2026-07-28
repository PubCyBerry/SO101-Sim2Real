"""lerobot.async_inference.helpers 의 **최소 vendored 사본** (pickle 호환 전용).

policy-server(실 LeRobot 0.6.0)와 주고받는 pickle 객체의 클래스 경로
(`lerobot.async_inference.helpers.{RemotePolicyConfig,TimedObservation,TimedAction}`)를
ROS 컨테이너(py3.12)에서 재현하기 위한 dataclass 만 담는다. 실 lerobot 은 transformers/
datasets/diffusers 까지 끌어와(import 체인) 컨테이너에 부적합 → 여기서는 직렬화에 필요한
필드/메서드만 미러한다. 필드 이름·기본값은 실 lerobot 과 **정확히 일치**해야 한다
(pickle 은 __dict__ 기반이라 server 측 실 클래스로 복원됨).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

RawObservation = dict[str, Any]
Action = Any  # 실제로는 torch.Tensor (unpickle 시 torch 필요 — 컨테이너에 torch-cpu 설치)


@dataclass
class TimedData:
    timestamp: float
    timestep: int

    def get_timestamp(self):
        return self.timestamp

    def get_timestep(self):
        return self.timestep


@dataclass
class TimedAction(TimedData):
    action: Action

    def get_action(self):
        return self.action


@dataclass
class TimedObservation(TimedData):
    observation: RawObservation
    must_go: bool = False

    def get_observation(self):
        return self.observation


@dataclass
class RemotePolicyConfig:
    policy_type: str
    pretrained_name_or_path: str
    lerobot_features: dict[str, dict]
    actions_per_chunk: int
    device: str = "cpu"
    rename_map: dict[str, str] = field(default_factory=dict)
