#!/usr/bin/env python
"""
scripts/policy_server_attention_bridge.py
=========================================
SmolVLA cross-attention 시각화용 Async Inference Policy Server.

lerobot.async_inference.policy_server.PolicyServer 를 서브클래싱해 gRPC 프로토콜·
클라이언트(vla_policy_node)·action 경로를 **전혀 바꾸지 않고**, 추론마다 SmolVLA 의
expert cross-attention(action 토큰 → 이미지 패치 토큰)을 캡처해 카메라별 히트맵으로
ZMQ PUB(:5556) 한다. Isaac Sim 브리지(run_cube_desk_ros_bridge.py --attention_overlay)가
SUB 해 top/wrist/front 뷰에 오버레이한다.

────────────────────────────────────────────────────────
SmolVLA 전용 — 타 모델 무영향
────────────────────────────────────────────────────────
정책 로드 후 SmolVLA(=self.policy.model.vlm_with_expert 존재) 가 아니면 monkey-patch·
PUB 을 **스킵**하고 표준 PolicyServer 로 동작한다(경고 1줄, 크래시·동작 변경 없음).
groot/act 등에 실수로 물려도 안전.

────────────────────────────────────────────────────────
캡처 방식 (StanleyChueh/lerobot record_attention_plot_cross 이식)
────────────────────────────────────────────────────────
우리 lerobot 0.5.2 SmolVLA 는 attention_mode="cross_attn" 가 기본이고
SmolVLMWithExpertModel.get_attention_interface() 가 항상 eager_attention_forward 를
쓴다(softmax probs 가 materialize됨). 다만 추출 훅이 없어 아래 3곳을 instance-level
monkey-patch 한다(vendored lerobot 무수정 — docker/groot_compat_patch.py 선례).

  1. vlm_with_expert.embed_image      → 카메라별 패치 토큰 수(num_img_embs) 기록
  2. model.embed_prefix               → prefix 내 카메라별 토큰 span + prefix_len 기록
  3. vlm_with_expert.eager_attention_forward
                                      → expert cross-attn probs(마지막 cross 레이어) 캡처

추론 후: probs[1,H,50,prefix_len] → head 평균 → action-step 평균 → cam span slice →
sqrt(N) grid → 0~98 percentile 정규화 → ZMQ PUB.

카메라 매핑(고정): SmolVLA 는 입력 순서로 카메라 구분. env/smolvla.env RENAME_MAP =
top→camera1, wrist→camera2, front→camera3 → slot0=top, slot1=wrist, slot2=front.

────────────────────────────────────────────────────────
실행 (Docker 컨테이너 안)
────────────────────────────────────────────────────────
  python /workspace/scripts/policy_server_attention_bridge.py \\
      --host 0.0.0.0 --port 8080 --fps 30 \\
      --attn_zmq_host 0.0.0.0 --attn_zmq_port 5556

  # entrypoint 경유 (policy-entrypoint.sh policy-server-attn 모드)
  docker compose --env-file .env -f docker/docker-compose.yaml run --rm \\
      -e POLICY_PROFILE=smolvla policy-server policy-server-attn
"""

import argparse
import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pprint import pformat

import grpc
import numpy as np
import torch
import zmq
from torch import nn

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.transport import services_pb2_grpc

logger = logging.getLogger(__name__)

# 카메라 슬롯(입력 순서) → 물리 카메라 이름. RENAME_MAP top→camera1/wrist→camera2/front→camera3.
_SLOT_TO_CAM = ["top", "wrist", "front"]


def _assemble_spans(counts: list[int], add_special_tokens: bool) -> list[tuple]:
    """카메라별 패치 토큰 수 리스트 → prefix 내 (slot, start, count, grid_h, grid_w) span.

    embed_prefix 레이아웃: 이미지마다 [start_token(1)?][패치 count개][end_token(1)?] 반복
    → lang → state. add_special_tokens 면 start/end 각 1토큰씩 포함.
    """
    spans: list[tuple] = []
    offset = 0
    for slot, count in enumerate(counts):
        if add_special_tokens:
            offset += 1  # image_start_token
        s = int(round(math.sqrt(count)))
        if s * s == count:
            gh = gw = s
        else:  # 비정사각 폴백 — gh*gw ≤ count (앞쪽 토큰만 사용)
            gh = max(1, s)
            gw = max(1, count // gh)
        spans.append((slot, offset, count, gh, gw))
        offset += count
        if add_special_tokens:
            offset += 1  # image_end_token
    return spans


def _patched_eager_attention_forward(
    self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
):
    """SmolVLMWithExpertModel.eager_attention_forward 의 instance-level 대체.

    원본(smolvlm_with_expert.py:516-561) 본문을 그대로 복제하고, 기록 활성 시 expert
    cross-attn probs(query=action 토큰, key=prefix)만 단일 슬롯에 덮어써 마지막 cross
    레이어·마지막 denoise step 의 attention 만 남긴다.

    ⚠ 트립와이어: lerobot 업그레이드로 원본 본문이 바뀌면 이 복제도 점검할 것.
    """
    # ── 기록 판정용 입력 seq 길이(transpose/expand 전) ──
    q_len = int(query_states.shape[1])
    k_len = int(key_states.shape[1])

    num_att_heads = self.num_attention_heads
    num_key_value_heads = self.num_key_value_heads
    num_key_value_groups = num_att_heads // num_key_value_heads

    sequence_length = key_states.shape[1]

    key_states = key_states[:, :, :, None, :].expand(
        batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
    )
    key_states = key_states.reshape(
        batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
    )

    value_states = value_states[:, :, :, None, :].expand(
        batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
    )
    value_states = value_states.reshape(
        batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
    )

    query_states = query_states.to(dtype=torch.float32)
    key_states = key_states.to(dtype=torch.float32)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)

    att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
    att_weights *= head_dim**-0.5

    att_weights = att_weights.to(dtype=torch.float32)
    big_neg = torch.finfo(att_weights.dtype).min
    masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
    probs = nn.functional.softmax(masked_att_weights, dim=-1)

    # ── 캡처: expert cross-attn(action→prefix) 호출만. k_len==prefix_len 이 prefix KV 캐시 ──
    if getattr(self, "_attn_record", False):
        try:
            if k_len == int(getattr(self, "_attn_prefix_len", -1)) and q_len <= int(
                getattr(self, "_attn_chunk_size", 50)
            ):
                # probs: [B, H, q_len, k_len] — 마지막 매칭(=마지막 cross 레이어·step)만 보관
                self._attn_cross_probs = probs.detach().to("cpu", dtype=torch.float32)
        except Exception:  # noqa: BLE001  캡처 실패는 추론에 영향 없도록 무시
            pass

    probs = probs.to(dtype=value_states.dtype)
    att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))
    att_output = att_output.permute(0, 2, 1, 3)
    att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)

    return att_output


class AttentionBridgeServer(PolicyServer):
    """SmolVLA cross-attention 을 캡처해 ZMQ PUB 하는 PolicyServer."""

    def __init__(self, config: PolicyServerConfig, attn_zmq_host: str, attn_zmq_port: int) -> None:
        super().__init__(config)
        self._attn_host = attn_zmq_host
        self._attn_port = attn_zmq_port
        self._attn_active = False          # SmolVLA 일 때만 True
        self._attn_patched = False
        self._attn_seq = 0
        self._pub_lock = threading.Lock()  # gRPC worker 스레드 간 PUB 직렬화

        # ZMQ PUB — bind 1회. network_mode: host 라 루프백으로 브리지(호스트) 접속.
        self._zmq_ctx = zmq.Context.instance()
        self._pub = self._zmq_ctx.socket(zmq.PUB)
        self._pub.setsockopt(zmq.SNDHWM, 2)     # 큐 짧게 — 최신 위주
        self._pub.setsockopt(zmq.LINGER, 0)
        self._pub.bind(f"tcp://{attn_zmq_host}:{attn_zmq_port}")
        logger.info(f"[ATTN] ZMQ PUB bind tcp://{attn_zmq_host}:{attn_zmq_port}")

    # ── 정책 로드 후 attention 캡처 patch 주입 ───────────────────────────────────
    def SendPolicyInstructions(self, request, context):  # noqa: N802
        response = super().SendPolicyInstructions(request, context)
        self._inject_attention()
        return response

    def _reset_server(self) -> None:
        super()._reset_server()
        self._attn_active = False
        # 패치는 instance 영속(멱등) — 새 정책이 SmolVLA 아니면 _inject 가 다시 False 로.

    def _inject_attention(self) -> None:
        """SmolVLA 면 embed_image/embed_prefix/eager_attention_forward 를 patch 한다."""
        model = getattr(self.policy, "model", None)
        vw = getattr(model, "vlm_with_expert", None) if model is not None else None
        if vw is None or not hasattr(vw, "eager_attention_forward"):
            logger.warning(
                f"[ATTN] {type(self.policy).__name__} 는 SmolVLA 가 아님 → attention 캡처 "
                "스킵, 표준 추론으로 동작."
            )
            self._attn_active = False
            return

        # 캡처 상태 변수 초기화
        vw._attn_record = False
        vw._attn_img_counts = []
        vw._attn_prefix_len = -1
        vw._attn_cross_probs = None
        vw._attn_chunk_size = int(getattr(self.policy.config, "chunk_size", 50))
        model._attn_img_spans = []

        if not self._attn_patched:
            _orig_embed_image = vw.embed_image
            _orig_embed_prefix = model.embed_prefix

            def _patched_embed_image(img, _o=_orig_embed_image):
                out = _o(img)
                try:
                    vw._attn_img_counts.append(int(out.shape[1]))
                except Exception:  # noqa: BLE001
                    pass
                return out

            def _patched_embed_prefix(*a, _o=_orig_embed_prefix, **k):
                vw._attn_img_counts = []          # 이미지 루프 시작 전 리셋
                res = _o(*a, **k)
                try:
                    vw._attn_prefix_len = int(res[0].shape[1])
                    model._attn_img_spans = _assemble_spans(
                        list(vw._attn_img_counts),
                        bool(getattr(model, "add_image_special_tokens", False)),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"[ATTN] embed_prefix span 조립 실패: {exc}")
                return res

            vw.embed_image = _patched_embed_image
            model.embed_prefix = _patched_embed_prefix
            vw.eager_attention_forward = _patched_eager_attention_forward.__get__(vw, type(vw))
            self._attn_patched = True
            logger.info("[ATTN] monkey-patch 주입 완료 (embed_image·embed_prefix·eager_attention)")

        self._attn_active = True
        try:
            feats = list(getattr(self.policy.config, "image_features", []) or [])
            logger.info(
                f"[ATTN] SmolVLA 캡처 활성 | chunk_size={vw._attn_chunk_size} | "
                f"image_features={feats} | slot→cam={_SLOT_TO_CAM}"
            )
        except Exception:  # noqa: BLE001
            pass

    # ── 추론 + attention 캡처 → ZMQ PUB ─────────────────────────────────────────
    def _get_action_chunk(self, observation: dict) -> torch.Tensor:
        if not self._attn_active:
            return super()._get_action_chunk(observation)

        model = self.policy.model
        vw = model.vlm_with_expert
        vw._attn_cross_probs = None
        vw._attn_record = True
        try:
            chunk = super()._get_action_chunk(observation)
        finally:
            vw._attn_record = False

        try:
            self._publish_attention(model, vw)
        except Exception as exc:  # noqa: BLE001  시각화 실패는 추론에 영향 없도록
            logger.debug(f"[ATTN] publish 실패: {exc}")
        return chunk

    def _publish_attention(self, model, vw) -> None:
        probs = vw._attn_cross_probs            # [1, H, q, prefix_len] cpu float32
        spans = getattr(model, "_attn_img_spans", [])
        if probs is None or not spans:
            return
        # head 평균 → action-step 평균 → [prefix_len]
        attn = probs[0].mean(dim=0).mean(dim=0)  # [prefix_len]
        attn_np = attn.numpy()

        # ⚠ numpy ndarray 를 pickle(send_pyobj)하면 numpy 1.x↔2.x 간 pickle 비호환으로 깨진다
        #   (서버 컨테이너=numpy 2.x `numpy._core` ↔ Isaac 브리지 호스트=numpy 1.26 핀). 따라서
        #   히트맵은 **plain Python list**(grid.tolist())로 직렬화한다(8×8라 사소, numpy 버전 무관).
        heatmaps: dict[str, list] = {}
        shapes: dict[str, tuple] = {}
        for slot, start, count, gh, gw in spans:
            cam = _SLOT_TO_CAM[slot] if slot < len(_SLOT_TO_CAM) else f"cam{slot}"
            n = gh * gw
            vec = attn_np[start : start + n]
            if vec.shape[0] < n:
                continue
            grid = vec.reshape(gh, gw).astype(np.float32)
            lo, hi = np.percentile(grid, [0, 98])
            grid = np.clip((grid - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)
            heatmaps[cam] = grid.tolist()   # ← list (numpy-version-agnostic pickle)
            shapes[cam] = (gh, gw)

        if not heatmaps:
            return
        self._attn_seq += 1
        payload = {"seq": self._attn_seq, "heatmaps": heatmaps}
        with self._pub_lock:
            self._pub.send_pyobj(payload)
        if self._attn_seq <= 3 or self._attn_seq % 100 == 0:
            logger.info(
                f"[ATTN] PUB #{self._attn_seq} | prefix_len={vw._attn_prefix_len} | "
                f"probs={tuple(probs.shape)} | grids={shapes}"
            )


def serve_attention(cfg: PolicyServerConfig, attn_host: str, attn_port: int) -> None:
    logging.info(pformat({"server_config": cfg.__dict__, "attn": {"host": attn_host, "port": attn_port}}))

    server_instance = AttentionBridgeServer(cfg, attn_host, attn_port)
    server = grpc.server(ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(server_instance, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    server_instance.logger.info(
        f"AttentionBridgeServer (lerobot {_lerobot_ver()}) 기동 | {cfg.host}:{cfg.port} | "
        f"fps={cfg.fps} | attention ZMQ tcp://{attn_host}:{attn_port}"
    )
    server.start()
    server.wait_for_termination()
    server_instance.logger.info("Server terminated")


def _lerobot_ver() -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version("lerobot")
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Async Inference Policy Server + SmolVLA cross-attention ZMQ bridge",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = parser.add_argument_group("서버 설정")
    g.add_argument("--host", default="0.0.0.0", help="gRPC bind 주소")
    g.add_argument("--port", type=int, default=8080, help="gRPC 포트")
    g.add_argument("--fps", type=int, default=30, help="제어 루프 FPS (로봇과 동일)")
    g.add_argument("--inference_latency", type=float, default=0.033, help="목표 추론 레이턴시(초)")
    g.add_argument("--obs_queue_timeout", type=float, default=2.0, help="관측 큐 타임아웃(초)")

    a = parser.add_argument_group("Attention ZMQ 설정")
    a.add_argument("--attn_zmq_host", default="0.0.0.0", help="히트맵 PUB bind 주소")
    a.add_argument("--attn_zmq_port", type=int, default=5556, help="히트맵 PUB 포트(gRPC 8080·GR00T 5555와 구분)")

    args = parser.parse_args()

    server_config = PolicyServerConfig(
        host=args.host,
        port=args.port,
        fps=args.fps,
        inference_latency=args.inference_latency,
        obs_queue_timeout=args.obs_queue_timeout,
    )
    serve_attention(server_config, args.attn_zmq_host, args.attn_zmq_port)


if __name__ == "__main__":
    main()
