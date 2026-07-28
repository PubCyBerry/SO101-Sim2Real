#!/usr/bin/env python3
"""AffineAdapterServer — policy-server 에 real↔sim joint frame affine 어댑터.

`JOINT_FRAME_MODE`(4-case) 에 따라 정책 I/O 를 변환한다:
  observation.state (수신, client frame → policy frame)
  action            (반환, policy frame → client frame)
정책 자체는 학습 frame 그대로 — affine 은 정책 normalize **바깥**(정규화 통계 불변).
**이미지는 무변환**(시각 gap 은 affine 아니라 DR 영역). 어느 client(sim=policy-feature·
real=real-follower)·어느 정책 frame 이든 single flag 로 처리하므로 양쪽 client 무변경.

  JOINT_FRAME_MODE ∈ {sim-to-sim, real-to-real, sim-to-real, real-to-sim}
    = (학습데이터 도메인 → 추론 플랫폼). 같은 도메인 = passthrough.

frame 정의: sim = policy-feature(sim 학습공간, feature_codec) · real = real-follower 단위.
변환은 follower_calibration 의 composite(`real_follower_to_policy_feature` 등) 재사용.

실행: `policy-entrypoint.sh` 의 `policy-server-affine` 모드. lerobot 의 stock
`policy_server.serve()` 를 AffineAdapterServer 로만 대체(host/port/fps 등 인자·config 동일).
"""

import os
import sys
from concurrent import futures

import draccus
import grpc
import numpy as np
import torch

# so101_contract(src) — policy-server 컨테이너는 ../src:/workspace/src 마운트 필요(compose).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from lerobot.async_inference.configs import PolicyServerConfig  # noqa: E402
from lerobot.async_inference.policy_server import PolicyServer  # noqa: E402
from lerobot.transport import services_pb2_grpc  # type: ignore  # noqa: E402
from lerobot.utils.constants import OBS_STATE  # noqa: E402

from so101_contract.follower_calibration import (  # noqa: E402
    policy_feature_to_real_follower,
    real_follower_to_policy_feature,
)


def _ident(x: np.ndarray) -> np.ndarray:
    return x


# mode → (obs 변환: client→policy, action 변환: policy→client).
#   sim frame = policy-feature(sim 학습공간) · real frame = real-follower 단위.
_MODES = {
    "sim-to-sim": (_ident, _ident),     # client sim · policy sim → passthrough
    "real-to-real": (_ident, _ident),   # client real · policy real → passthrough
    # client real(real-follower) ↔ policy sim(feature)
    "sim-to-real": (real_follower_to_policy_feature, policy_feature_to_real_follower),
    # client sim(feature) ↔ policy real(real-follower)
    "real-to-sim": (policy_feature_to_real_follower, real_follower_to_policy_feature),
}


class AffineAdapterServer(PolicyServer):
    """PolicyServer + JOINT_FRAME_MODE 별 observation.state / action affine 변환."""

    def __init__(self, config: PolicyServerConfig):
        super().__init__(config)
        mode = os.getenv("JOINT_FRAME_MODE", "sim-to-sim").strip()
        if mode not in _MODES:
            raise ValueError(f"JOINT_FRAME_MODE must be one of {list(_MODES)}, got {mode!r}")
        self._mode = mode
        self._obs_fn, self._act_fn = _MODES[mode]
        passthrough = self._obs_fn is _ident
        self.logger.info(
            f"[affine] JOINT_FRAME_MODE={mode} → "
            f"{'passthrough(변환 없음)' if passthrough else 'observation.state·action follower 변환'}"
        )

    def _enqueue_observation(self, obs) -> bool:
        # observation.state(client frame) → policy frame, in-place. 이미지 등 다른 키 무변환.
        if self._obs_fn is not _ident:
            o = obs.get_observation()
            if OBS_STATE in o:
                o[OBS_STATE] = self._obs_fn(np.asarray(o[OBS_STATE], dtype=np.float32)).tolist()
        return super()._enqueue_observation(obs)

    def _predict_action_chunk(self, observation_t):
        chunk = super()._predict_action_chunk(observation_t)  # policy frame action 들
        if self._act_fn is not _ident:
            for ta in chunk:
                a = ta.get_action()  # torch.Tensor (action_dim,)
                conv = self._act_fn(a.detach().cpu().numpy().astype(np.float32))
                ta.action = torch.as_tensor(conv, dtype=a.dtype, device=a.device)
        return chunk


@draccus.wrap()
def serve(cfg: PolicyServerConfig) -> None:
    """stock policy_server.serve() 와 동일하되 AffineAdapterServer 로 기동."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    adapter = AffineAdapterServer(cfg)
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(adapter, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")
    adapter.logger.info(f"AffineAdapterServer started on {cfg.host}:{cfg.port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
