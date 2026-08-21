from __future__ import annotations

import copy
import json
import os
import ssl
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen

from backend.integrations.contract import build_source_descriptor


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
    # Intentionally bypass certificate verification for internal RAG endpoints.
    # Use RAG_VERIFY_SSL=true in production when a valid cert is available.
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


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
    documents_upload_endpoint: str = "/documents/upload"
    documents_endpoint: str = "/documents"
    documents_delete_endpoint: str = "/documents/delete_document"
    documents_track_status_endpoint: str = "/documents/track_status/{track_id}"
    source_type: str = "rag"
    source_ref_id: str = "rag"
    source_label: str = "RAG"
    source_mode: str = "external"

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
    ) -> RAGConfig:
        source_env = env or os.environ
        base_url = (
            _clean_str(source_env.get("RAG_URL")) or _clean_str(source_env.get("RAG_BACKEND_URL")) or ""
        )
        enabled_default = bool(base_url)
        enabled = _get_bool(source_env, "RAG_ENABLED", enabled_default)
        query_endpoint = _clean_str(source_env.get("RAG_QUERY_ENDPOINT")) or "/query"
        stream_endpoint = _clean_str(source_env.get("RAG_STREAM_ENDPOINT")) or "/query/stream"
        documents_upload_endpoint = (
            _clean_str(source_env.get("RAG_DOCUMENTS_UPLOAD_ENDPOINT")) or "/documents/upload"
        )
        documents_endpoint = _clean_str(source_env.get("RAG_DOCUMENTS_ENDPOINT")) or "/documents"
        documents_delete_endpoint = (
            _clean_str(source_env.get("RAG_DOCUMENTS_DELETE_ENDPOINT")) or "/documents/delete_document"
        )
        documents_track_status_endpoint = (
            _clean_str(source_env.get("RAG_DOCUMENTS_TRACK_STATUS_ENDPOINT"))
            or "/documents/track_status/{track_id}"
        )
        if not query_endpoint.startswith("/"):
            query_endpoint = f"/{query_endpoint}"
        if not stream_endpoint.startswith("/"):
            stream_endpoint = f"/{stream_endpoint}"
        if not documents_upload_endpoint.startswith("/"):
            documents_upload_endpoint = f"/{documents_upload_endpoint}"
        if not documents_endpoint.startswith("/"):
            documents_endpoint = f"/{documents_endpoint}"
        if not documents_delete_endpoint.startswith("/"):
            documents_delete_endpoint = f"/{documents_delete_endpoint}"
        if not documents_track_status_endpoint.startswith("/"):
            documents_track_status_endpoint = f"/{documents_track_status_endpoint}"
        return cls(
            enabled=enabled,
            base_url=base_url.rstrip("/"),
            query_endpoint=query_endpoint,
            stream_endpoint=stream_endpoint,
            documents_upload_endpoint=documents_upload_endpoint,
            documents_endpoint=documents_endpoint,
            documents_delete_endpoint=documents_delete_endpoint,
            documents_track_status_endpoint=documents_track_status_endpoint,
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
RAGUploadTransport = Callable[[str, str, bytes, str, float, bool], dict[str, Any]]
RAGGetTransport = Callable[[str, float, bool], dict[str, Any]]
RAGDeleteTransport = Callable[[str, dict[str, Any], float, bool], dict[str, Any]]


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
        raise RAGIntegrationError(f"RAG backend returned HTTP {exc.code}: {body_preview}") from exc
    except URLError as exc:
        raise RAGIntegrationError(f"RAG backend is unavailable: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RAGIntegrationError("RAG backend request timed out.") from exc

    if not raw_body:
        return {}
    try:
        decoded = json.loads(raw_body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        preview = raw_body.decode("utf-8", errors="replace")[:500]
        raise RAGIntegrationError(f"RAG backend returned invalid JSON: {preview!r}") from exc
    if not isinstance(decoded, dict):
        raise RAGIntegrationError("RAG backend returned a non-object JSON payload.")
    return decoded


def _decode_json_response(raw_body: bytes, *, context: str) -> dict[str, Any]:
    if not raw_body:
        return {}
    try:
        decoded = json.loads(raw_body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        preview = raw_body.decode("utf-8", errors="replace")[:500]
        raise RAGIntegrationError(f"RAG backend returned invalid JSON for {context}: {preview!r}") from exc
    if not isinstance(decoded, dict):
        raise RAGIntegrationError(f"RAG backend returned a non-object JSON payload for {context}.")
    return decoded


def _default_get_transport(
    url: str,
    timeout_sec: float,
    verify_ssl: bool,
) -> dict[str, Any]:
    request = Request(
        url,
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urlopen(
            request,
            timeout=timeout_sec,
            context=_ssl_context_for_url(url, verify_ssl=verify_ssl),
        ) as response:
            return _decode_json_response(response.read(), context="GET request")
    except HTTPError as exc:
        body_preview = exc.read().decode("utf-8", errors="replace")[:500]
        raise RAGIntegrationError(f"RAG backend returned HTTP {exc.code}: {body_preview}") from exc
    except URLError as exc:
        raise RAGIntegrationError(f"RAG backend is unavailable: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RAGIntegrationError("RAG backend request timed out.") from exc


def _default_upload_transport(
    url: str,
    file_name: str,
    content: bytes,
    content_type: str,
    timeout_sec: float,
    verify_ssl: bool,
) -> dict[str, Any]:
    boundary = f"----llmDataAnalyst{uuid.uuid4().hex}"
    safe_file_name = file_name.replace('"', "")
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            (f'Content-Disposition: form-data; name="file"; filename="{safe_file_name}"\r\n').encode(),
            f"Content-Type: {content_type}\r\n\r\n".encode(),
            content,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    request = Request(
        url,
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urlopen(
            request,
            timeout=timeout_sec,
            context=_ssl_context_for_url(url, verify_ssl=verify_ssl),
        ) as response:
            return _decode_json_response(response.read(), context="document upload")
    except HTTPError as exc:
        body_preview = exc.read().decode("utf-8", errors="replace")[:500]
        raise RAGIntegrationError(f"RAG backend returned HTTP {exc.code}: {body_preview}") from exc
    except URLError as exc:
        raise RAGIntegrationError(f"RAG backend is unavailable: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RAGIntegrationError("RAG backend request timed out.") from exc


def _default_delete_transport(
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
        method="DELETE",
    )
    try:
        with urlopen(
            request,
            timeout=timeout_sec,
            context=_ssl_context_for_url(url, verify_ssl=verify_ssl),
        ) as response:
            return _decode_json_response(response.read(), context="document deletion")
    except HTTPError as exc:
        body_preview = exc.read().decode("utf-8", errors="replace")[:500]
        raise RAGIntegrationError(f"RAG backend returned HTTP {exc.code}: {body_preview}") from exc
    except URLError as exc:
        raise RAGIntegrationError(f"RAG backend is unavailable: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RAGIntegrationError("RAG backend request timed out.") from exc


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
        raise RAGIntegrationError(f"RAG backend returned HTTP {exc.code}: {body_preview}") from exc
    except URLError as exc:
        raise RAGIntegrationError(f"RAG backend is unavailable: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RAGIntegrationError("RAG backend request timed out.") from exc


class RAGService:
    def __init__(
        self,
        config: RAGConfig,
        *,
        transport: RAGTransport | None = None,
        stream_transport: RAGStreamTransport | None = None,
        upload_transport: RAGUploadTransport | None = None,
        get_transport: RAGGetTransport | None = None,
        delete_transport: RAGDeleteTransport | None = None,
    ) -> None:
        self.config = config
        self._transport = transport or _default_transport
        self._stream_transport = stream_transport or _default_stream_transport
        self._upload_transport = upload_transport or _default_upload_transport
        self._get_transport = get_transport or _default_get_transport
        self._delete_transport = delete_transport or _default_delete_transport

    @classmethod
    def from_env(cls) -> RAGService:
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
            raise RAGIntegrationError("RAG integration is not configured. Set RAG_URL first.")
        return urljoin(f"{self.config.base_url}/", endpoint.lstrip("/"))

    def _get_request(self, endpoint: str) -> dict[str, Any]:
        try:
            return self._get_transport(
                self._endpoint_url(endpoint),
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
            raise RAGIntegrationError(f"RAG backend returned HTTP {exc.code}{suffix}") from exc
        except URLError as exc:
            raise RAGIntegrationError(f"RAG backend is unavailable: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RAGIntegrationError("RAG backend request timed out.") from exc

    def _delete_request(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._delete_transport(
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
            raise RAGIntegrationError(f"RAG backend returned HTTP {exc.code}{suffix}") from exc
        except URLError as exc:
            raise RAGIntegrationError(f"RAG backend is unavailable: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RAGIntegrationError("RAG backend request timed out.") from exc

    def _request(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._transport(
                self._endpoint_url(endpoint),
                payload,
                self.config.timeout_sec,
                self.config.verify_ssl,
            )
        except RAGIntegrationError:  # pylint: disable=try-except-raise
            raise
        except HTTPError as exc:
            body_preview = ""
            try:
                body_preview = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                body_preview = ""
            suffix = f": {body_preview}" if body_preview else ""
            raise RAGIntegrationError(f"RAG backend returned HTTP {exc.code}{suffix}") from exc
        except URLError as exc:
            raise RAGIntegrationError(f"RAG backend is unavailable: {exc.reason}") from exc
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
        except RAGIntegrationError:  # pylint: disable=try-except-raise
            raise
        except HTTPError as exc:
            body_preview = ""
            try:
                body_preview = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                body_preview = ""
            suffix = f": {body_preview}" if body_preview else ""
            raise RAGIntegrationError(f"RAG backend returned HTTP {exc.code}{suffix}") from exc
        except URLError as exc:
            raise RAGIntegrationError(f"RAG backend is unavailable: {exc.reason}") from exc
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
                clean = _clean_str(item.get("url") or item.get("link") or item.get("source"))
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
        answer = _clean_str(payload.get("response") or payload.get("answer") or payload.get("content"))
        references = self._normalize_references(payload.get("references") or payload.get("sources"))
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

    def retrieve(
        self,
        *,
        query: str,
        mode: str = "naive",
        top_k: int | None = None,
    ) -> RAGQueryResult:
        """Retrieve raw context chunks from LightRAG without LLM generation.

        Uses ``only_need_context=True`` so LightRAG skips the expensive
        LLM-summarisation step and returns the retrieved passages directly.
        """
        if not self.is_enabled:
            raise RAGIntegrationError("RAG integration is disabled or not configured.")

        clean_query = _clean_str(query)
        if not clean_query:
            raise RAGIntegrationError("RAG query must not be empty.")

        request_params: dict[str, Any] = {
            "query": clean_query,
            "mode": _clean_str(mode),
            "top_k": _coerce_positive_int(
                top_k,
                default=self.config.top_k_default,
            ),
            "only_need_context": True,
        }
        payload = self._request(self.config.query_endpoint, request_params)

        # LightRAG returns raw context in `data` field when only_need_context=True
        raw_context = (
            payload.get("data")
            or payload.get("response")
            or payload.get("answer")
            or payload.get("content")
            or ""
        )
        answer = _clean_str(raw_context)
        warnings: list[str] = []
        if not answer:
            warnings.append("RAG backend returned no context chunks.")
        return RAGQueryResult(
            query=clean_query,
            answer=answer,
            references=[],
            warnings=warnings,
            request_params=copy.deepcopy(request_params),
            raw_payload=copy.deepcopy(payload),
        )

    @staticmethod
    def _normalize_status(value: object, *, default: str = "unknown") -> str:
        text = _clean_str(value) or default
        if "." in text:
            text = text.rsplit(".", 1)[-1]
        return text.strip().lower() or default

    @staticmethod
    def _normalize_status_summary(raw: object) -> dict[str, int]:
        if not isinstance(raw, dict):
            return {}
        result: dict[str, int] = {}
        for key, value in raw.items():
            status = RAGService._normalize_status(key)
            try:
                count = int(value)
            except (TypeError, ValueError):
                count = 0
            result[status] = count
        return result

    @staticmethod
    def _optional_int(value: object) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_document(raw: object, *, status_hint: str | None = None) -> dict[str, Any] | None:
        if not isinstance(raw, dict):
            return None
        status = RAGService._normalize_status(raw.get("status") or status_hint)
        return {
            "id": _clean_str(raw.get("id")),
            "file_path": _clean_str(raw.get("file_path") or raw.get("file_name") or raw.get("filename")),
            "status": status,
            "track_id": _clean_str(raw.get("track_id")),
            "chunks_count": RAGService._optional_int(raw.get("chunks_count")),
            "content_length": RAGService._optional_int(raw.get("content_length")),
            "created_at": _clean_str(raw.get("created_at")),
            "updated_at": _clean_str(raw.get("updated_at")),
            "error_msg": _clean_str(raw.get("error_msg") or raw.get("error")),
        }

    @classmethod
    def _normalize_documents_payload(cls, payload: dict[str, Any]) -> list[dict[str, Any]]:
        raw_documents = payload.get("documents")
        if isinstance(raw_documents, list):
            return [doc for item in raw_documents if (doc := cls._normalize_document(item)) is not None]

        raw_statuses = payload.get("statuses") or payload.get("status")
        if isinstance(raw_statuses, dict):
            result: list[dict[str, Any]] = []
            for status, items in raw_statuses.items():
                if not isinstance(items, list):
                    continue
                for item in items:
                    doc = cls._normalize_document(
                        item,
                        status_hint=cls._normalize_status(status),
                    )
                    if doc is not None:
                        result.append(doc)
            return result

        raw_data = payload.get("data")
        if isinstance(raw_data, dict):
            return cls._normalize_documents_payload(raw_data)
        if isinstance(raw_data, list):
            return [doc for item in raw_data if (doc := cls._normalize_document(item)) is not None]
        return []

    def upload_document(
        self,
        *,
        file_name: str,
        content: bytes,
        content_type: str = "application/octet-stream",
    ) -> dict[str, Any]:
        if not self.is_enabled:
            raise RAGIntegrationError("RAG integration is disabled or not configured.")

        clean_file_name = _clean_str(file_name)
        if not clean_file_name:
            raise RAGIntegrationError("RAG document file name must not be empty.")
        if not content:
            raise RAGIntegrationError("RAG document content must not be empty.")

        payload = self._upload_transport(
            self._endpoint_url(self.config.documents_upload_endpoint),
            clean_file_name,
            content,
            _clean_str(content_type) or "application/octet-stream",
            self.config.timeout_sec,
            self.config.verify_ssl,
        )
        track_id = _clean_str(payload.get("track_id"))
        if not track_id:
            raise RAGIntegrationError("RAG backend did not return a track_id.")
        return {
            "status": _clean_str(payload.get("status")) or "success",
            "message": _clean_str(payload.get("message")) or "",
            "track_id": track_id,
        }

    def get_track_status(self, track_id: str) -> dict[str, Any]:
        if not self.is_enabled:
            raise RAGIntegrationError("RAG integration is disabled or not configured.")

        clean_track_id = _clean_str(track_id)
        if not clean_track_id:
            raise RAGIntegrationError("RAG track_id must not be empty.")
        endpoint = self.config.documents_track_status_endpoint.replace(
            "{track_id}",
            quote(clean_track_id, safe=""),
        )
        payload = self._get_request(endpoint)
        return {
            "track_id": _clean_str(payload.get("track_id")) or clean_track_id,
            "documents": self._normalize_documents_payload(payload),
            "status_summary": self._normalize_status_summary(payload.get("status_summary")),
        }

    def list_documents(self) -> dict[str, Any]:
        if not self.is_enabled:
            raise RAGIntegrationError("RAG integration is disabled or not configured.")

        payload = self._get_request(self.config.documents_endpoint)
        documents = self._normalize_documents_payload(payload)
        return {"documents": documents}

    def delete_document(self, document_id: str) -> dict[str, Any]:
        if not self.is_enabled:
            raise RAGIntegrationError("RAG integration is disabled or not configured.")

        clean_document_id = _clean_str(document_id)
        if not clean_document_id:
            raise RAGIntegrationError("RAG document_id must not be empty.")

        payload = self._delete_request(
            self.config.documents_delete_endpoint,
            {
                "doc_ids": [clean_document_id],
                "delete_file": False,
                "delete_llm_cache": False,
            },
        )
        return {
            "status": _clean_str(payload.get("status")) or "deletion_started",
            "message": _clean_str(payload.get("message")) or "",
            "document_id": _clean_str(payload.get("doc_id")) or clean_document_id,
        }

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
