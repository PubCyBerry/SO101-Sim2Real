"""단일 active motion client lease."""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass


class LeaseError(PermissionError):
    """Lease 획득·검증 실패."""


@dataclass(frozen=True)
class LeaseSnapshot:
    client_id: str
    token: str
    expires_at_ns: int


class MotionLease:
    def __init__(self, duration_ms: int = 5000) -> None:
        if duration_ms <= 0:
            raise ValueError("lease duration은 양수여야 한다")
        self.duration_ns = int(duration_ms * 1_000_000)
        self._lock = threading.Lock()
        self._lease: LeaseSnapshot | None = None

    def _active(self, now_ns: int) -> LeaseSnapshot | None:
        if self._lease is not None and self._lease.expires_at_ns <= now_ns:
            self._lease = None
        return self._lease

    def acquire(self, client_id: str, *, now_ns: int | None = None) -> LeaseSnapshot:
        if not client_id.strip():
            raise LeaseError("client_id가 비어 있다")
        current_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
        with self._lock:
            active = self._active(current_ns)
            if active is not None and active.client_id != client_id:
                raise LeaseError(f"motion lease는 이미 {active.client_id!r}가 보유 중이다")
            token = active.token if active is not None else secrets.token_urlsafe(24)
            self._lease = LeaseSnapshot(client_id, token, current_ns + self.duration_ns)
            return self._lease

    def validate_and_renew(
        self,
        client_id: str,
        token: str,
        *,
        now_ns: int | None = None,
    ) -> LeaseSnapshot:
        current_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
        with self._lock:
            active = self._active(current_ns)
            if active is None:
                raise LeaseError("active motion lease가 없다")
            if active.client_id != client_id or not secrets.compare_digest(active.token, token):
                raise LeaseError("motion lease client/token 불일치")
            self._lease = LeaseSnapshot(client_id, token, current_ns + self.duration_ns)
            return self._lease

    def snapshot(self, *, now_ns: int | None = None) -> LeaseSnapshot | None:
        current_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
        with self._lock:
            return self._active(current_ns)

    def release(
        self,
        client_id: str,
        token: str,
        *,
        now_ns: int | None = None,
    ) -> None:
        current_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
        with self._lock:
            active = self._active(current_ns)
            if active is None:
                return
            if active.client_id != client_id or not secrets.compare_digest(active.token, token):
                raise LeaseError("motion lease release client/token 불일치")
            self._lease = None
