from __future__ import annotations

from contextlib import contextmanager
import logging
import socket
import threading
import time
from urllib.parse import urlparse
from typing import Any, Iterator

from backend.config import settings


logger = logging.getLogger(__name__)
_init_lock = threading.Lock()
_initialized = False
_import_warned = False
_connect_warned = False
_connectivity_checked_at = 0.0
_connectivity_ok = False
_CONNECTIVITY_TTL_SEC = 10.0


def _is_phoenix_reachable() -> bool:
    global _connectivity_checked_at, _connectivity_ok

    now = time.monotonic()
    if now - _connectivity_checked_at < _CONNECTIVITY_TTL_SEC:
        return _connectivity_ok

    parsed = urlparse(settings.phoenix_collector_endpoint)
    host = parsed.hostname
    if not host:
        _connectivity_checked_at = now
        _connectivity_ok = False
        return False

    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80

    try:
        with socket.create_connection((host, port), timeout=0.6):
            _connectivity_ok = True
    except OSError:
        _connectivity_ok = False

    _connectivity_checked_at = now
    return _connectivity_ok


def initialize_phoenix() -> None:
    global _initialized, _import_warned, _connect_warned

    if _initialized or not settings.phoenix_enabled:
        return

    if not _is_phoenix_reachable():
        if not _connect_warned:
            logger.warning(
                "Phoenix collector '%s' is unreachable. Tracing is disabled until "
                "connectivity is restored.",
                settings.phoenix_collector_endpoint,
            )
            _connect_warned = True
        return
    _connect_warned = False

    with _init_lock:
        if _initialized:
            return

        try:
            from openinference.instrumentation.langchain import LangChainInstrumentor
            from phoenix.otel import register
        except Exception as exc:
            if not _import_warned:
                logger.warning(
                    "Phoenix instrumentation packages are unavailable (%s). "
                    "Tracing is disabled.",
                    exc,
                )
                _import_warned = True
            return

        try:
            tracer_provider = register(
                project_name=settings.phoenix_project_name,
                endpoint=settings.phoenix_collector_endpoint,
                auto_instrument=False,
            )
            if settings.phoenix_auto_instrument:
                LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
            _initialized = True
            logger.info(
                "Phoenix tracing is enabled (project=%s, endpoint=%s).",
                settings.phoenix_project_name,
                settings.phoenix_collector_endpoint,
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialize Phoenix tracing (%s). Tracing is disabled.",
                exc,
            )


def build_trace_context(
    *,
    session_id: str,
    user_id: int,
    username: str,
    request_kind: str,
    use_history: bool,
    include_reasoning: bool,
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "user_id": str(user_id),
        "username": username,
        "request_kind": request_kind,
        "use_history": bool(use_history),
        "include_reasoning": bool(include_reasoning),
    }


@contextmanager
def query_trace_context(
    *,
    session_id: str,
    user_id: int,
    username: str,
    request_kind: str,
    use_history: bool,
    include_reasoning: bool,
    query: str,
) -> Iterator[None]:
    if not settings.phoenix_enabled:
        yield
        return

    # Ленивая реинициализация: если backend стартовал раньше Phoenix, пробуем включить
    # tracing при каждом запросе.
    initialize_phoenix()
    if not _initialized:
        yield
        return

    try:
        from openinference.instrumentation import using_attributes
    except Exception:
        yield
        return

    query_preview = query.strip().replace("\n", " ")
    if len(query_preview) > 300:
        query_preview = f"{query_preview[:300]}..."

    metadata = {
        "username": username,
        "request_kind": request_kind,
        "use_history": bool(use_history),
        "include_reasoning": bool(include_reasoning),
        "query_preview": query_preview,
    }
    tags = [
        "backend",
        "reason-action",
        f"request:{request_kind}",
        f"user:{username}",
    ]

    with using_attributes(
        session_id=session_id,
        user_id=str(user_id),
        metadata=metadata,
        tags=tags,
    ):
        yield
