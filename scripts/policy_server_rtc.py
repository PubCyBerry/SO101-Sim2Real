#!/usr/bin/env python
"""
scripts/policy_server_rtc.py
============================
Real-Time Chunking (RTC) 가 통합된 Async Inference Policy Server.

lerobot.async_inference.policy_server.PolicyServer 를 서브클래싱해
gRPC 프로토콜·클라이언트 변경 없이 서버 측 _get_action_chunk 에
RTC 가이던스(prev_chunk_left_over + inference_delay)를 주입한다.

────────────────────────────────────────────────────────
작동 방식
────────────────────────────────────────────────────────
1. SendPolicyInstructions RPC 로 SmolVLA 가 로드되면 RTCConfig 를 주입하고
   init_rtc_processor() 를 호출한다.

2. 매 추론마다 _get_action_chunk 가 호출될 때:
   - elapsed_steps = (현재 시각 - 이전 청크 요청 시각) × fps
     → 추론 중 로봇이 실행했을 스텝 수 추정 = inference_delay
   - prev_chunk_left_over = prev_chunk[:, inference_delay:, :]
     (이미 실행됐을 앞부분을 제외한 나머지)
   - predict_action_chunk(obs, inference_delay=N, prev_chunk_left_over=prev)
     → flow-matching 디노이징 루프에 guidance term 주입

3. 반환된 청크(정규화된 action space)를 저장하고 다음 호출에서 재사용.

────────────────────────────────────────────────────────
클라이언트 측 변경 없음
────────────────────────────────────────────────────────
기존 policy-client(lerobot-entrypoint.sh policy-client 모드)를 그대로 사용한다.
RTC 는 서버 내부에서 투명하게 동작하며, 클라이언트 측 weighted_average 청크
블렌딩과 중첩되어 최대 부드러움을 제공한다.

────────────────────────────────────────────────────────
지원 정책
────────────────────────────────────────────────────────
init_rtc_processor() 를 구현한 flow-matching 정책만 지원한다:
  - SmolVLA (lerobot[smolvla])
  - Pi0 / Pi0.5 (lerobot[pi])

GR00T 는 flash-attn 의존 구조가 달라 별도 확인 필요.
RTC 미지원 정책이 로드되면 경고를 출력하고 표준 추론(RTC 없음)으로 폴백한다.

────────────────────────────────────────────────────────
실행 예시 (Docker 컨테이너 안에서)
────────────────────────────────────────────────────────
  # 기본값으로 시작
  python /workspace/scripts/policy_server_rtc.py

  # RTC 파라미터 커스터마이즈
  python /workspace/scripts/policy_server_rtc.py \\
      --host 0.0.0.0 --port 8080 --fps 30 \\
      --rtc_execution_horizon 10 \\
      --rtc_max_guidance_weight 10.0 \\
      --rtc_prefix_attention_schedule EXP

  # entrypoint 를 통해 기동 (policy-entrypoint.sh policy-server-rtc 모드)
  docker compose --env-file .env -f docker/docker-compose.yaml up -d policy-server
  # docker-compose CMD = "policy-server-rtc" 일 때 자동으로 이 스크립트 호출
"""

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pprint import pformat

import grpc
import torch

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.transport import services_pb2_grpc

logger = logging.getLogger(__name__)


class RTCPolicyServer(PolicyServer):
    """RTC 가이던스를 서버 측 추론에 통합한 PolicyServer."""

    def __init__(self, config: PolicyServerConfig, rtc_config: RTCConfig) -> None:
        super().__init__(config)
        self._rtc_config = rtc_config
        self._rtc_available = False

        # 이전 청크 상태 (정규화된 action space, CPU)
        self._prev_chunk: torch.Tensor | None = None
        # 이전 청크가 클라이언트에 전달된 시각 (로봇이 실행 시작한 시점 추정)
        self._prev_chunk_delivery_time: float | None = None
        # 직전 추론 소요 시간 × fps = inference_delay 추정값 (self-correcting)
        # 첫 번째 청크에서는 0, 이후 실제 측정값으로 업데이트된다.
        self._prev_inference_steps: int = 0

        self._chunk_count: int = 0  # 로그 식별용

    # ── 서버 리셋 시 RTC 상태도 초기화 ──────────────────────────────────────────
    def _reset_server(self) -> None:
        super()._reset_server()
        self._rtc_available = False
        self._prev_chunk = None
        self._prev_chunk_delivery_time = None
        self._prev_inference_steps = 0
        self._chunk_count = 0
        logger.info("[RTC] 상태 초기화 완료.")

    # ── 정책 로드 후 RTCConfig 주입 ───────────────────────────────────────────────
    def _inject_rtc(self) -> None:
        """로드된 policy 에 RTCConfig 를 주입하고 RTCProcessor 를 초기화한다."""
        if self.policy is None:
            return

        if not hasattr(self.policy, "init_rtc_processor"):
            logger.warning(
                f"[RTC] {type(self.policy).__name__} 는 init_rtc_processor 를 지원하지 않습니다. "
                "표준 추론(RTC 없음)으로 동작합니다. (SmolVLA / Pi0 / Pi0.5 만 지원)"
            )
            self._rtc_available = False
            return

        self.policy.config.rtc_config = self._rtc_config
        self.policy.init_rtc_processor()
        self._rtc_available = True
        logger.info(
            f"[RTC] RTCProcessor 주입 완료 | "
            f"execution_horizon={self._rtc_config.execution_horizon} | "
            f"max_guidance_weight={self._rtc_config.max_guidance_weight} | "
            f"schedule={self._rtc_config.prefix_attention_schedule}"
        )

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        response = super().SendPolicyInstructions(request, context)
        self._inject_rtc()
        return response

    # ── RTC 가이던스 적용 추론 ────────────────────────────────────────────────────
    def _get_action_chunk(self, observation: dict) -> torch.Tensor:
        """
        이전 청크의 leftover 를 guidance 로 사용해 새 청크를 생성한다.

        inference_delay 계산 근거:
          - 서버는 클라이언트 타임스텝을 직접 알 수 없으므로 wall-clock 시간을 사용.
          - elapsed_time × fps = 추론 대기 중 로봇이 실행했을 스텝 수(추정).
          - 실제 소비 스텝보다 과소 추정 시: guidance 겹침 영역 확대 → 더 보수적인 전환.
          - 과대 추정 시: guidance 겹침 영역 축소 → 더 반응적인 전환.
          양쪽 모두 파국적 실패는 없으며, 실제 inference latency(~0.1-0.3s × 30fps = 3-9 steps) 와
          execution_horizon(기본 10) 이 비슷해 일반적으로 잘 동작한다.
        """
        t_inference_start = time.perf_counter()
        self._chunk_count += 1

        # ── RTC 입력 계산 ────────────────────────────────────────────────────────
        #
        # prev_chunk_left_over: 직전 청크 중 로봇이 아직 실행하지 않은 부분.
        #   = prev_chunk[leftover_start:] 에서
        #   leftover_start = (현재 시각 - 직전 청크 전달 시각) × fps
        #
        # inference_delay: 현재 추론이 끝날 때까지 로봇이 실행할 추가 스텝 수 (추정).
        #   = 직전 추론 소요 시간 × fps (self-correcting — 추론 시간이 안정되면 정확해짐)
        #   첫 번째 실제 추론 후 자동 갱신되므로 초기값 0 은 무해하다.
        #
        # 두 값이 다른 이유:
        #   leftover_start ≈ chunk_size × chunk_size_threshold (로봇 실행 시간, ~25 steps)
        #   inference_delay ≈ SmolVLA 추론 시간 × fps (~3-10 steps @ 30fps, Blackwell GPU)
        #
        leftover_start = 0
        inference_delay = 0
        prev_chunk_left_over = None
        guidance_applied = False

        if self._rtc_available and self._prev_chunk is not None and self._prev_chunk_delivery_time is not None:
            # 로봇이 직전 청크를 몇 스텝이나 실행했는지 추정
            elapsed_since_delivery = t_inference_start - self._prev_chunk_delivery_time
            leftover_start = min(
                int(elapsed_since_delivery * self.config.fps),
                self._prev_chunk.shape[1] - 1,
            )
            # 현재 추론 중 로봇이 추가로 실행할 스텝 수 (직전 추론 시간으로 추정)
            inference_delay = self._prev_inference_steps
            prev_chunk_left_over = self._prev_chunk[:, leftover_start:, :].to(
                next(self.policy.parameters()).device
            )
            guidance_applied = True

        # ── 청크 생성 ────────────────────────────────────────────────────────────
        if self._rtc_available:
            chunk = self.policy.predict_action_chunk(
                observation,
                inference_delay=inference_delay,
                prev_chunk_left_over=prev_chunk_left_over,
            )
        else:
            chunk = self.policy.predict_action_chunk(observation)

        # 실제 추론 소요 시간 측정 → 다음 inference_delay 추정에 사용
        actual_inference_s = time.perf_counter() - t_inference_start
        measured_inference_steps = int(actual_inference_s * self.config.fps)

        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)  # → (B, chunk_size, action_dim)
        chunk = chunk[:, : self.actions_per_chunk, :]

        # ── INFO 로그: docker logs -f 로 RTC 적용 여부 즉시 확인 ─────────────────
        if guidance_applied:
            logger.info(
                f"[RTC] chunk #{self._chunk_count} | guidance ✅ | "
                f"leftover_start={leftover_start} | "
                f"inference_delay={inference_delay} (prev) → {measured_inference_steps} (now) | "
                f"leftover={prev_chunk_left_over.shape[1]} steps "
                f"(horizon={self._rtc_config.execution_horizon})"
            )
        elif self._rtc_available:
            logger.info(
                f"[RTC] chunk #{self._chunk_count} | guidance ❌ "
                f"(첫 번째 청크 — 다음 청크부터 적용) | "
                f"inference={actual_inference_s*1000:.0f}ms={measured_inference_steps} steps"
            )
        else:
            logger.info(
                f"[RTC] chunk #{self._chunk_count} | guidance ❌ (RTC 미지원 정책)"
            )

        # 다음 호출을 위해 상태 저장
        if self._rtc_available:
            self._prev_chunk = chunk.detach().cpu()
            self._prev_chunk_delivery_time = time.perf_counter()  # 전달 시각 기록
            self._prev_inference_steps = measured_inference_steps  # 추론 시간 갱신

        return chunk


def serve_rtc(cfg: PolicyServerConfig, rtc_config: RTCConfig) -> None:
    """RTCPolicyServer 를 gRPC 서버로 기동한다."""
    logging.info(pformat({"server_config": cfg.__dict__, "rtc_config": rtc_config.__dict__}))

    server_instance = RTCPolicyServer(cfg, rtc_config)

    server = grpc.server(ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(server_instance, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    server_instance.logger.info(
        f"RTCPolicyServer (lerobot {_lerobot_ver()}) 기동 | "
        f"{cfg.host}:{cfg.port} | fps={cfg.fps} | "
        f"rtc execution_horizon={rtc_config.execution_horizon} "
        f"max_guidance_weight={rtc_config.max_guidance_weight}"
    )
    server.start()
    server.wait_for_termination()
    server_instance.logger.info("Server terminated")


def _lerobot_ver() -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version("lerobot")
    except Exception:
        return "unknown"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Async Inference Policy Server with server-side Real-Time Chunking (RTC)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── PolicyServerConfig ─────────────────────────────────────────────────────
    g = parser.add_argument_group("서버 설정")
    g.add_argument("--host", default="0.0.0.0", help="gRPC bind 주소")
    g.add_argument("--port", type=int, default=8080, help="gRPC 포트")
    g.add_argument("--fps", type=int, default=30, help="제어 루프 FPS (로봇과 동일해야 함)")
    g.add_argument("--inference_latency", type=float, default=0.033,
                   help="목표 추론 레이턴시(초). 클라이언트 chunk_size_threshold 와 함께 동작")
    g.add_argument("--obs_queue_timeout", type=float, default=2.0,
                   help="관측 큐 타임아웃(초)")

    # ── RTCConfig ──────────────────────────────────────────────────────────────
    r = parser.add_argument_group("RTC 설정")
    r.add_argument("--rtc_execution_horizon", type=int, default=10,
                   help="이전 청크와 일관성 유지 스텝 수 (권장 8~12)")
    r.add_argument("--rtc_max_guidance_weight", type=float, default=10.0,
                   help="가이던스 강도. 10스텝 flow-matching 에 최적화된 기본값")
    r.add_argument(
        "--rtc_prefix_attention_schedule",
        choices=["EXP", "LINEAR", "ONES", "ZEROS"],
        default="EXP",
        help="겹침 구간 가중치 방식. EXP = 지수 감쇠(권장)",
    )

    args = parser.parse_args()

    schedule_map = {
        "EXP": RTCAttentionSchedule.EXP,
        "LINEAR": RTCAttentionSchedule.LINEAR,
        "ONES": RTCAttentionSchedule.ONES,
        "ZEROS": RTCAttentionSchedule.ZEROS,
    }

    server_config = PolicyServerConfig(
        host=args.host,
        port=args.port,
        fps=args.fps,
        inference_latency=args.inference_latency,
        obs_queue_timeout=args.obs_queue_timeout,
    )

    rtc_config = RTCConfig(
        enabled=True,
        execution_horizon=args.rtc_execution_horizon,
        max_guidance_weight=args.rtc_max_guidance_weight,
        prefix_attention_schedule=schedule_map[args.rtc_prefix_attention_schedule],
    )

    serve_rtc(server_config, rtc_config)


if __name__ == "__main__":
    main()
