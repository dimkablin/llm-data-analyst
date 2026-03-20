from __future__ import annotations

import copy
import json
import os
import ssl
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from backend.integration_contract import build_source_descriptor


def _clean_str(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _get_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    value = env.get(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _coerce_positive_int(value: object, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def _coerce_positive_float(value: object, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.1, parsed)


def _ssl_context_for_url(url: str, *, verify_ssl: bool) -> ssl.SSLContext | None:
    if not url.lower().startswith("https://"):
        return None
    if verify_ssl:
        return ssl.create_default_context()
    return ssl._create_unverified_context()


@dataclass(frozen=True)
class RAGConfig:
    enabled: bool
    base_url: str
    query_endpoint: str
    stream_endpoint: str
    timeout_sec: float
    verify_ssl: bool
    query_mode_default: str
    top_k_default: int
    source_type: str = "rag"
    source_ref_id: str = "rag"
    source_label: str = "RAG"
    source_mode: str = "external"

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
    ) -> "RAGConfig":
        source_env = env or os.environ
        base_url = (
            _clean_str(source_env.get("RAG_URL"))
            or _clean_str(source_env.get("RAG_BACKEND_URL"))
            or ""
        )
        enabled_default = bool(base_url)
        enabled = _get_bool(source_env, "RAG_ENABLED", enabled_default)
        query_endpoint = _clean_str(source_env.get("RAG_QUERY_ENDPOINT")) or "/query"
        stream_endpoint = (
            _clean_str(source_env.get("RAG_STREAM_ENDPOINT")) or "/query/stream"
        )
        if not query_endpoint.startswith("/"):
            query_endpoint = f"/{query_endpoint}"
        if not stream_endpoint.startswith("/"):
            stream_endpoint = f"/{stream_endpoint}"
        return cls(
            enabled=enabled,
            base_url=base_url.rstrip("/"),
            query_endpoint=query_endpoint,
            stream_endpoint=stream_endpoint,
            timeout_sec=_coerce_positive_float(
                source_env.get("RAG_TIMEOUT_SEC"),
                default=45.0,
            ),
            verify_ssl=_get_bool(source_env, "RAG_VERIFY_SSL", False),
            query_mode_default=_clean_str(source_env.get("RAG_QUERY_MODE")) or "hybrid",
            top_k_default=_coerce_positive_int(
                source_env.get("RAG_TOP_K"),
                default=5,
            ),
            source_label=_clean_str(source_env.get("RAG_SOURCE_LABEL")) or "RAG",
        )

    @property
    def available(self) -> bool:
        return self.enabled and bool(self.base_url)


@dataclass(frozen=True)
class RAGQueryResult:
    query: str
    answer: str | None
    references: list[str]
    warnings: list[str]
    request_params: dict[str, Any]
    raw_payload: dict[str, Any]


class RAGIntegrationError(RuntimeError):
    pass


RAGTransport = Callable[[str, dict[str, Any], float, bool], dict[str, Any]]
RAGStreamTransport = Callable[[str, dict[str, Any], float, bool], Iterable[str]]


def _default_transport(
    url: str,
    payload: dict[str, Any],
    timeout_sec: float,
    verify_ssl: bool,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(
            request,
            timeout=timeout_sec,
            context=_ssl_context_for_url(url, verify_ssl=verify_ssl),
        ) as response:
            raw_body = response.read()
    except HTTPError as exc:
        body_preview = exc.read().decode("utf-8", errors="replace")[:500]
        raise RAGIntegrationError(
            f"RAG backend returned HTTP {exc.code}: {body_preview}"
        ) from exc
    except URLError as exc:
        raise RAGIntegrationError(
            f"RAG backend is unavailable: {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise RAGIntegrationError("RAG backend request timed out.") from exc

    if not raw_body:
        return {}
    try:
        decoded = json.loads(raw_body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        preview = raw_body.decode("utf-8", errors="replace")[:500]
        raise RAGIntegrationError(
            f"RAG backend returned invalid JSON: {preview!r}"
        ) from exc
    if not isinstance(decoded, dict):
        raise RAGIntegrationError("RAG backend returned a non-object JSON payload.")
    return decoded


def _default_stream_transport(
    url: str,
    payload: dict[str, Any],
    timeout_sec: float,
    verify_ssl: bool,
) -> Iterable[str]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/x-ndjson, application/json, text/plain",
        },
        method="POST",
    )
    try:
        with urlopen(
            request,
            timeout=timeout_sec,
            context=_ssl_context_for_url(url, verify_ssl=verify_ssl),
        ) as response:
            for raw_line in response:
                decoded = raw_line.decode("utf-8", errors="replace").strip()
                if decoded:
                    yield decoded
    except HTTPError as exc:
        body_preview = exc.read().decode("utf-8", errors="replace")[:500]
        raise RAGIntegrationError(
            f"RAG backend returned HTTP {exc.code}: {body_preview}"
        ) from exc
    except URLError as exc:
        raise RAGIntegrationError(
            f"RAG backend is unavailable: {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise RAGIntegrationError("RAG backend request timed out.") from exc


class RAGService:
    def __init__(
        self,
        config: RAGConfig,
        *,
        transport: RAGTransport | None = None,
        stream_transport: RAGStreamTransport | None = None,
    ) -> None:
        self.config = config
        self._transport = transport or _default_transport
        self._stream_transport = stream_transport or _default_stream_transport

    @classmethod
    def from_env(cls) -> "RAGService":
        return cls(RAGConfig.from_env())

    @property
    def is_enabled(self) -> bool:
        return self.config.available

    @property
    def enabled(self) -> bool:
        return self.is_enabled

    def source_ref(self) -> dict[str, str]:
        return {
            "source_type": self.config.source_type,
            "source_ref_id": self.config.source_ref_id,
            "source_label": self.config.source_label,
            "source_mode": self.config.source_mode,
        }

    def source_descriptor(self) -> dict[str, Any]:
        return build_source_descriptor(
            source_type=self.config.source_type,
            source_ref_id=self.config.source_ref_id,
            source_label=self.config.source_label,
            display_name_ru="База знаний",
            source_mode=self.config.source_mode,
            enabled=self.config.enabled,
            available=self.config.available,
            description="External RAG / knowledge-base retrieval integration.",
            description_ru="Поиск и ответ по внешней базе знаний через RAG сервис.",
            capabilities=["knowledge_base_search", "document_answer"],
            requires_session_data=False,
            timeout_hint_sec=self.config.timeout_sec,
        )

    def _endpoint_url(self, endpoint: str) -> str:
        if not self.config.base_url:
            raise RAGIntegrationError(
                "RAG integration is not configured. Set RAG_URL first."
            )
        return urljoin(f"{self.config.base_url}/", endpoint.lstrip("/"))

    def _request(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._transport(
                self._endpoint_url(endpoint),
                payload,
                self.config.timeout_sec,
                self.config.verify_ssl,
            )
        except RAGIntegrationError:
            raise
        except HTTPError as exc:
            body_preview = ""
            try:
                body_preview = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                body_preview = ""
            suffix = f": {body_preview}" if body_preview else ""
            raise RAGIntegrationError(
                f"RAG backend returned HTTP {exc.code}{suffix}"
            ) from exc
        except URLError as exc:
            raise RAGIntegrationError(
                f"RAG backend is unavailable: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise RAGIntegrationError("RAG backend request timed out.") from exc

    def _stream_request(self, endpoint: str, payload: dict[str, Any]) -> Iterable[str]:
        try:
            yield from self._stream_transport(
                self._endpoint_url(endpoint),
                payload,
                self.config.timeout_sec,
                self.config.verify_ssl,
            )
        except RAGIntegrationError:
            raise
        except HTTPError as exc:
            body_preview = ""
            try:
                body_preview = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                body_preview = ""
            suffix = f": {body_preview}" if body_preview else ""
            raise RAGIntegrationError(
                f"RAG backend returned HTTP {exc.code}{suffix}"
            ) from exc
        except URLError as exc:
            raise RAGIntegrationError(
                f"RAG backend is unavailable: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise RAGIntegrationError("RAG backend request timed out.") from exc

    @staticmethod
    def _normalize_references(raw_references: object) -> list[str]:
        result: list[str] = []
        if not isinstance(raw_references, list):
            return result
        seen: set[str] = set()
        for item in raw_references:
            if isinstance(item, str):
                clean = _clean_str(item)
            elif isinstance(item, dict):
                clean = _clean_str(
                    item.get("url") or item.get("link") or item.get("source")
                )
            else:
                clean = None
            if not clean or clean in seen:
                continue
            seen.add(clean)
            result.append(clean)
        return result

    def _normalize_response(
        self,
        *,
        query: str,
        request_params: dict[str, Any],
        payload: dict[str, Any],
    ) -> RAGQueryResult:
        answer = _clean_str(
            payload.get("response") or payload.get("answer") or payload.get("content")
        )
        references = self._normalize_references(
            payload.get("references") or payload.get("sources")
        )
        warnings: list[str] = []
        if not answer:
            warnings.append("RAG backend returned no normalized answer text.")
        return RAGQueryResult(
            query=query,
            answer=answer,
            references=references,
            warnings=warnings,
            request_params=copy.deepcopy(request_params),
            raw_payload=copy.deepcopy(payload),
        )

    def search(
        self,
        *,
        query: str,
        mode: str | None = None,
        top_k: int | None = None,
        include_references: bool = False,
    ) -> RAGQueryResult:
        if not self.is_enabled:
            raise RAGIntegrationError("RAG integration is disabled or not configured.")

        clean_query = _clean_str(query)
        if not clean_query:
            raise RAGIntegrationError("RAG query must not be empty.")

        request_params: dict[str, Any] = {
            "query": clean_query,
            "mode": _clean_str(mode) or self.config.query_mode_default,
            "top_k": _coerce_positive_int(
                top_k,
                default=self.config.top_k_default,
            ),
            "include_references": bool(include_references),
        }
        payload = self._request(self.config.query_endpoint, request_params)
        return self._normalize_response(
            query=clean_query,
            request_params=request_params,
            payload=payload,
        )

    def stream_search(
        self,
        *,
        query: str,
        mode: str | None = None,
        top_k: int | None = None,
        include_references: bool = False,
    ) -> Iterable[str]:
        if not self.is_enabled:
            raise RAGIntegrationError("RAG integration is disabled or not configured.")

        clean_query = _clean_str(query)
        if not clean_query:
            raise RAGIntegrationError("RAG query must not be empty.")

        request_params: dict[str, Any] = {
            "query": clean_query,
            "mode": _clean_str(mode) or self.config.query_mode_default,
            "top_k": _coerce_positive_int(
                top_k,
                default=self.config.top_k_default,
            ),
            "include_references": bool(include_references),
            "stream": True,
        }
        for line in self._stream_request(self.config.stream_endpoint, request_params):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            chunk = payload.get("response") or payload.get("answer")
            if isinstance(chunk, str) and chunk:
                yield chunk

    @staticmethod
    def format_for_user(result: RAGQueryResult | None) -> str:
        if result is None:
            return "RAG не вернул ответ."
        answer = _clean_str(result.answer)
        if answer:
            return answer
        return "RAG не вернул текстовый ответ."
