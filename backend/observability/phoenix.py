from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
import socket
import threading
import time
from urllib.parse import urlparse
from typing import Any, Iterator

from backend.core.config import settings


logger = logging.getLogger(__name__)
_init_lock = threading.Lock()
_initialized = False
_import_warned = False
_connect_warned = False
_connectivity_checked_at = 0.0
_connectivity_ok = False
_CONNECTIVITY_TTL_SEC = 10.0


@dataclass(frozen=True)
class LLMUsageSnapshot:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    model_name: str | None = None
    provider: str | None = None

    @property
    def is_empty(self) -> bool:
        return all(
            value is None
            for value in (
                self.prompt_tokens,
                self.completion_tokens,
                self.total_tokens,
                self.model_name,
                self.provider,
            )
        )


def _normalize_token_int(raw: Any) -> int | None:
    if raw is None or raw == "" or isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    try:
        return int(float(str(raw).strip()))
    except Exception:
        return None


def _first_token_value(payload: Any, *keys: str) -> int | None:
    if not isinstance(payload, dict):
        return None
    for key in keys:
        value = _normalize_token_int(payload.get(key))
        if value is not None:
            return value
    return None


def _first_text_value(payload: Any, *keys: str) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in keys:
        raw = payload.get(key)
        clean = str(raw or "").strip()
        if clean:
            return clean
    return None


def extract_llm_usage(response: Any, *, fallback_model: str | None = None, fallback_provider: str | None = None) -> LLMUsageSnapshot | None:
    usage_metadata = getattr(response, "usage_metadata", None)
    response_metadata = getattr(response, "response_metadata", None)

    prompt_tokens = _first_token_value(
        usage_metadata,
        "input_tokens",
        "prompt_tokens",
    )
    completion_tokens = _first_token_value(
        usage_metadata,
        "output_tokens",
        "completion_tokens",
    )
    total_tokens = _first_token_value(
        usage_metadata,
        "total_tokens",
    )

    if prompt_tokens is None:
        prompt_tokens = _first_token_value(
            response_metadata,
            "input_tokens",
            "prompt_tokens",
        )
    if completion_tokens is None:
        completion_tokens = _first_token_value(
            response_metadata,
            "output_tokens",
            "completion_tokens",
        )
    if total_tokens is None:
        total_tokens = _first_token_value(
            response_metadata,
            "total_tokens",
        )

    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    model_name = (
        _first_text_value(response_metadata, "model_name", "model")
        or _first_text_value(usage_metadata, "model_name", "model")
        or (str(fallback_model).strip() if fallback_model else None)
    )
    provider = (
        _first_text_value(response_metadata, "provider", "llm_provider")
        or _first_text_value(usage_metadata, "provider", "llm_provider")
        or (str(fallback_provider).strip() if fallback_provider else None)
    )

    snapshot = LLMUsageSnapshot(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        model_name=model_name,
        provider=provider,
    )
    return None if snapshot.is_empty else snapshot


def record_llm_usage_on_active_span(
    response: Any,
    *,
    fallback_model: str | None = None,
    fallback_provider: str | None = None,
) -> LLMUsageSnapshot | None:
    usage = extract_llm_usage(
        response,
        fallback_model=fallback_model,
        fallback_provider=fallback_provider,
    )
    if usage is None:
        return None

    try:
        from opentelemetry import trace
    except Exception:
        return usage

    span = trace.get_current_span()
    if span is None:
        return usage

    try:
        if not span.is_recording():
            return usage
    except Exception:
        return usage

    if usage.prompt_tokens is not None:
        span.set_attribute("llm.token_count.prompt", usage.prompt_tokens)
        span.set_attribute("llm.usage.prompt_tokens", usage.prompt_tokens)
    if usage.completion_tokens is not None:
        span.set_attribute("llm.token_count.completion", usage.completion_tokens)
        span.set_attribute("llm.usage.completion_tokens", usage.completion_tokens)
    if usage.total_tokens is not None:
        span.set_attribute("llm.token_count.total", usage.total_tokens)
        span.set_attribute("llm.usage.total_tokens", usage.total_tokens)
    if usage.model_name:
        span.set_attribute("llm.model_name", usage.model_name)
    if usage.provider:
        span.set_attribute("llm.provider", usage.provider)
    return usage


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
    db_connection_id: str | None = None,
    csv_session_id: str | None = None,
    csv_duckdb_loaded: bool = False,
) -> dict[str, Any]:
    payload = {
        "session_id": session_id,
        "user_id": str(user_id),
        "username": username,
        "request_kind": request_kind,
        "use_history": bool(use_history),
        "include_reasoning": bool(include_reasoning),
        "csv_duckdb_loaded": bool(csv_duckdb_loaded),
    }
    if isinstance(db_connection_id, str) and db_connection_id.strip():
        payload["db_connection_id"] = db_connection_id.strip()
    if isinstance(csv_session_id, str) and csv_session_id.strip():
        payload["csv_session_id"] = csv_session_id.strip()
    return payload


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
    db_connection_id: str | None = None,
    csv_session_id: str | None = None,
    csv_duckdb_loaded: bool = False,
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
    if isinstance(db_connection_id, str) and db_connection_id.strip():
        metadata["db_connection_id"] = db_connection_id.strip()
        metadata["data_source"] = "db_connection"
    elif bool(csv_duckdb_loaded):
        metadata["data_source"] = "csv_duckdb"
    if isinstance(csv_session_id, str) and csv_session_id.strip():
        metadata["csv_session_id"] = csv_session_id.strip()
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


