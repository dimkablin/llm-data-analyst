"""Singleton manager for per-session sandboxes."""
from __future__ import annotations

import logging
import threading
import time

from backend.tools.sandbox import SessionSandbox

logger = logging.getLogger(__name__)


class SandboxManager:
    """Maps ``session_id`` → :class:`SessionSandbox`, with TTL eviction."""

    _instance: SandboxManager | None = None
    _init_lock = threading.Lock()

    def __init__(self) -> None:
        self._sandboxes: dict[str, SessionSandbox] = {}
        self._last_access: dict[str, float] = {}
        self._lock = threading.Lock()

    # ── singleton ────────────────────────────────────────────────────
    @classmethod
    def get_instance(cls) -> SandboxManager:
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── public API ───────────────────────────────────────────────────
    def get(self, session_id: str) -> SessionSandbox | None:
        with self._lock:
            return self._sandboxes.get(session_id)

    def get_or_create(self, session_id: str) -> SessionSandbox:
        with self._lock:
            sandbox = self._sandboxes.get(session_id)
            if sandbox is None:
                sandbox = SessionSandbox()
                self._sandboxes[session_id] = sandbox
                logger.debug("SandboxManager: created sandbox for session %s", session_id)
            self._last_access[session_id] = time.monotonic()
            return sandbox

    def remove(self, session_id: str) -> None:
        with self._lock:
            sb = self._sandboxes.pop(session_id, None)
            self._last_access.pop(session_id, None)
            if sb is not None:
                sb.clear()
                logger.debug("SandboxManager: removed sandbox for session %s", session_id)

    def cleanup_expired(self, ttl_sec: float = 7200.0) -> int:
        """Remove sandboxes not accessed within *ttl_sec*.  Returns count removed."""
        now = time.monotonic()
        expired: list[str] = []
        with self._lock:
            for sid, ts in self._last_access.items():
                if now - ts > ttl_sec:
                    expired.append(sid)
            for sid in expired:
                sb = self._sandboxes.pop(sid, None)
                self._last_access.pop(sid, None)
                if sb is not None:
                    sb.clear()
        if expired:
            logger.info("SandboxManager: evicted %d expired sandbox(es)", len(expired))
        return len(expired)

    def __len__(self) -> int:
        return len(self._sandboxes)
