"""lerobot.transport.utils 의 최소 vendored 사본 — gRPC chunk 송신 + 채널 옵션만.

실 lerobot.transport.utils 와 동일 동작(send_bytes_in_chunks / grpc_channel_options).
services_pb2 는 같은 vendored 패키지의 것을 쓴다.
"""

from __future__ import annotations

import io
import json
import logging
from typing import Any

from lerobot.transport import services_pb2

TransferState = services_pb2.TransferState

CHUNK_SIZE = 2 * 1024 * 1024  # 2 MB
MAX_MESSAGE_SIZE = 4 * 1024 * 1024  # 4 MB


def send_bytes_in_chunks(buffer: bytes, message_class: Any, log_prefix: str = "", silent: bool = True):
    bytes_buffer = io.BytesIO(buffer)
    bytes_buffer.seek(0, io.SEEK_END)
    size_in_bytes = bytes_buffer.tell()
    bytes_buffer.seek(0)

    sent_bytes = 0
    log = logging.debug if silent else logging.info
    while sent_bytes < size_in_bytes:
        transfer_state = TransferState.TRANSFER_MIDDLE
        if sent_bytes + CHUNK_SIZE >= size_in_bytes:
            transfer_state = TransferState.TRANSFER_END
        elif sent_bytes == 0:
            transfer_state = TransferState.TRANSFER_BEGIN

        size_to_read = min(CHUNK_SIZE, size_in_bytes - sent_bytes)
        chunk = bytes_buffer.read(size_to_read)
        yield message_class(transfer_state=transfer_state, data=chunk)
        sent_bytes += size_to_read
        log(f"{log_prefix} Sent {sent_bytes}/{size_in_bytes} bytes ({transfer_state})")
    return bytes_buffer.getvalue()


def grpc_channel_options(
    max_receive_message_length: int = MAX_MESSAGE_SIZE,
    max_send_message_length: int = MAX_MESSAGE_SIZE,
    enable_retries: bool = True,
    initial_backoff: str = "0.1s",
    max_attempts: int = 5,
    backoff_multiplier: float = 2,
    max_backoff: str = "2s",
):
    service_config = {
        "methodConfig": [
            {
                "name": [{}],
                "retryPolicy": {
                    "maxAttempts": max_attempts,
                    "initialBackoff": initial_backoff,
                    "maxBackoff": max_backoff,
                    "backoffMultiplier": backoff_multiplier,
                    "retryableStatusCodes": ["UNAVAILABLE", "DEADLINE_EXCEEDED"],
                },
            }
        ]
    }
    return [
        ("grpc.max_receive_message_length", max_receive_message_length),
        ("grpc.max_send_message_length", max_send_message_length),
        ("grpc.enable_retries", 1 if enable_retries else 0),
        ("grpc.service_config", json.dumps(service_config)),
    ]
