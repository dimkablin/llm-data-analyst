from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from backend.artifacts.artifact_meta import build_source_query_recipe_step
from backend.integrations.contract import build_operation_meta, build_source_descriptor


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


@dataclass(frozen=True)
class SearchIntegrationConfig:
    enabled: bool
    base_url: str
    search_endpoint: str
    fetch_endpoint: str
    timeout_sec: float
    max_results_default: int
    fetch_top_n_default: int
    source_type: str = "search"
    source_ref_id: str = "search"
    source_label: str = "Search"
    source_mode: str = "external"

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
    ) -> "SearchIntegrationConfig":
        source_env = env or os.environ
        base_url = _clean_str(source_env.get("SEARCH_BACKEND_URL")) or ""
        search_endpoint = _clean_str(source_env.get("SEARCH_ENDPOINT")) or "/api/v1/search/"
        if not search_endpoint.startswith("/"):
            search_endpoint = f"/{search_endpoint}"
        fetch_endpoint = _clean_str(source_env.get("SEARCH_FETCH_ENDPOINT")) or "/api/v1/fetch/"
        if not fetch_endpoint.startswith("/"):
            fetch_endpoint = f"/{fetch_endpoint}"
        enabled_default = bool(base_url)
        enabled = _get_bool(source_env, "SEARCH_ENABLED", enabled_default)
        return cls(
            enabled=enabled,
            base_url=base_url.rstrip("/"),
            search_endpoint=search_endpoint,
            fetch_endpoint=fetch_endpoint,
            timeout_sec=_coerce_positive_float(
                source_env.get("SEARCH_TIMEOUT_SEC"),
                default=20.0,
            ),
            max_results_default=_coerce_positive_int(
                source_env.get("SEARCH_MAX_RESULTS_DEFAULT"),
                default=5,
            ),
            fetch_top_n_default=_coerce_positive_int(
                source_env.get("SEARCH_FETCH_TOP_N_DEFAULT"),
                default=3,
            ),
            source_label=_clean_str(source_env.get("SEARCH_SOURCE_LABEL")) or "Search",
        )

    @property
    def available(self) -> bool:
        return self.enabled and bool(self.base_url)


@dataclass(frozen=True)
class FetchedPage:
    url: str
    content: str
    status: str  # "ok" | "error"
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@dataclass(frozen=True)
class SearchResultItem:
    rank: int
    title: str
    url: str
    snippet: str | None = None
    source_name: str | None = None
    published_at: str | None = None

    def as_row(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet or "",
            "source_name": self.source_name or "",
            "published_at": self.published_at or "",
        }


@dataclass(frozen=True)
class SearchQueryResult:
    query: str
    answer: str | None
    results: list[SearchResultItem]
    sources: list[str]
    warnings: list[str]
    request_params: dict[str, Any]

    @property
    def result_count(self) -> int:
        return len(self.results)

    def to_rows(self) -> list[dict[str, Any]]:
        return [item.as_row() for item in self.results]


class SearchIntegrationError(RuntimeError):
    pass


SearchTransport = Callable[[str, dict[str, Any], float], dict[str, Any]]


def _default_transport(url: str, payload: dict[str, Any], timeout_sec: float) -> dict[str, Any]:
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
        with urlopen(request, timeout=timeout_sec) as response:
            raw_body = response.read()
    except HTTPError as exc:
        body_preview = exc.read().decode("utf-8", errors="replace")[:500]
        raise SearchIntegrationError(
            f"Search backend returned HTTP {exc.code}: {body_preview}"
        ) from exc
    except URLError as exc:
        raise SearchIntegrationError(
            f"Search backend is unavailable: {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise SearchIntegrationError("Search backend request timed out.") from exc

    if not raw_body:
        return {}
    try:
        decoded = json.loads(raw_body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        preview = raw_body.decode("utf-8", errors="replace")[:500]
        raise SearchIntegrationError(
            f"Search backend returned invalid JSON: {preview!r}"
        ) from exc
    if not isinstance(decoded, dict):
        raise SearchIntegrationError("Search backend returned a non-object JSON payload.")
    return decoded


class SearchIntegrationService:
    def __init__(
        self,
        config: SearchIntegrationConfig,
        *,
        transport: SearchTransport | None = None,
    ) -> None:
        self.config = config
        self._transport = transport or _default_transport

    @classmethod
    def from_env(cls) -> "SearchIntegrationService":
        return cls(SearchIntegrationConfig.from_env())

    @property
    def is_enabled(self) -> bool:
        return self.config.available

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
            display_name_ru="Поиск",
            source_mode=self.config.source_mode,
            enabled=self.config.enabled,
            available=self.config.available,
            description="External quick search integration.",
            description_ru="Быстрый внешний поиск по теме пользователя.",
            capabilities=["search", "web_results"],
            requires_session_data=False,
            timeout_hint_sec=self.config.timeout_sec,
        )

    @staticmethod
    def _artifact_name(value: str | None) -> str:
        text = str(value or "").strip()
        return text or "search_results"

    def build_artifact_payload(
        self,
        result: SearchQueryResult,
        *,
        artifact_name: str = "search_results",
        tool_name: str = "search_tool",
    ) -> dict[str, Any]:
        return {
            "artifact_name": self._artifact_name(artifact_name),
            "rows": result.to_rows(),
            "source": self.source_ref(),
            "recipe": [
                build_source_query_recipe_step(
                    query=result.query,
                    source_type=self.config.source_type,
                    tool_name=tool_name,
                    title="Search Query",
                    summary=result.answer or f"External search for: {result.query}",
                    params=result.request_params,
                    result_count=result.result_count,
                )
            ],
            "meta": {
                "search": build_operation_meta(
                    status="completed",
                    warnings=result.warnings,
                    request_params=result.request_params,
                    timeout_sec=self.config.timeout_sec,
                    extra={
                        "query": result.query,
                        "answer": result.answer,
                        "result_count": result.result_count,
                        "sources": list(result.sources),
                        "top_titles": [item.title for item in result.results[:3]],
                    },
                )
            },
        }

    def _endpoint_url(self) -> str:
        if not self.config.base_url:
            raise SearchIntegrationError(
                "Search integration is not configured. Set SEARCH_BACKEND_URL first."
            )
        return urljoin(f"{self.config.base_url}/", self.config.search_endpoint.lstrip("/"))

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._transport(
                self._endpoint_url(),
                payload,
                self.config.timeout_sec,
            )
        except SearchIntegrationError:
            raise
        except HTTPError as exc:
            body_preview = ""
            try:
                body_preview = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                body_preview = ""
            suffix = f": {body_preview}" if body_preview else ""
            raise SearchIntegrationError(
                f"Search backend returned HTTP {exc.code}{suffix}"
            ) from exc
        except URLError as exc:
            raise SearchIntegrationError(
                f"Search backend is unavailable: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise SearchIntegrationError("Search backend request timed out.") from exc

    @staticmethod
    def _dedupe_preserve_order(values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            clean = _clean_str(value)
            if not clean or clean in seen:
                continue
            seen.add(clean)
            result.append(clean)
        return result

    @staticmethod
    def _normalize_result_item(raw: object, rank: int) -> SearchResultItem | None:
        if isinstance(raw, str):
            clean = _clean_str(raw)
            if not clean:
                return None
            return SearchResultItem(rank=rank, title=clean, url=clean)
        if not isinstance(raw, dict):
            return None

        title = _clean_str(raw.get("title") or raw.get("name") or raw.get("headline"))
        url = _clean_str(raw.get("url") or raw.get("link"))
        snippet = _clean_str(
            raw.get("snippet") or raw.get("description") or raw.get("content")
        )
        source_name = _clean_str(
            raw.get("source_name") or raw.get("source") or raw.get("engine")
        )
        published_at = _clean_str(
            raw.get("published_at") or raw.get("published") or raw.get("date")
        )

        if not title and url:
            title = url
        if not title:
            return None
        return SearchResultItem(
            rank=rank,
            title=title,
            url=url or "",
            snippet=snippet,
            source_name=source_name,
            published_at=published_at,
        )

    @staticmethod
    def _synthesize_answer_from_results(
        query: str,
        results: list[SearchResultItem],
        *,
        max_items: int = 4,
        max_snippet_chars: int = 420,
    ) -> str | None:
        """
        search_service returns answer=null by design (no LLM there). For the agent we still
        need a compact, query-grounded text block so the inner LLM can answer from the web.
        """
        if not results:
            return None
        lines: list[str] = []
        for item in results[:max_items]:
            title = (item.title or "").strip()
            url = (item.url or "").strip()
            snip = (item.snippet or "").strip().replace("\n", " ")
            if len(snip) > max_snippet_chars:
                snip = snip[: max_snippet_chars - 1] + "…"
            if snip:
                lines.append(f"• {title}: {snip}")
            elif title and url:
                lines.append(f"• {title} — {url}")
        if not lines:
            return None
        return (
            f"Краткая выжимка по запросу «{query}» (топ результатов поиска):\n"
            + "\n".join(lines)
        )

    def _normalize_response(
        self,
        *,
        query: str,
        request_params: dict[str, Any],
        payload: dict[str, Any],
    ) -> SearchQueryResult:
        answer = _clean_str(
            payload.get("answer") or payload.get("summary") or payload.get("content")
        )
        raw_results = payload.get("results") or payload.get("items") or payload.get("data") or []
        results: list[SearchResultItem] = []
        if isinstance(raw_results, list):
            for index, item in enumerate(raw_results, start=1):
                normalized = self._normalize_result_item(item, index)
                if normalized is not None:
                    results.append(normalized)

        sources: list[str] = []
        raw_sources = payload.get("sources")
        if isinstance(raw_sources, list):
            sources.extend(
                clean for clean in (_clean_str(item) for item in raw_sources) if clean
            )
        sources.extend(item.url for item in results if item.url)

        warnings: list[str] = []
        if not results:
            warnings.append("Search backend returned no normalized results.")

        if not answer and results:
            answer = self._synthesize_answer_from_results(query, results)

        return SearchQueryResult(
            query=query,
            answer=answer,
            results=results,
            sources=self._dedupe_preserve_order(sources),
            warnings=warnings,
            request_params=copy.deepcopy(request_params),
        )

    def search(
        self,
        query: str,
        *,
        max_results: int | None = None,
        fetch_top_n: int | None = None,
        language: str | None = None,
        engines: str | list[str] | None = None,
    ) -> SearchQueryResult:
        if not self.is_enabled:
            raise SearchIntegrationError(
                "Search integration is disabled or not configured."
            )

        clean_query = _clean_str(query)
        if not clean_query:
            raise SearchIntegrationError("Search query must not be empty.")

        request_params: dict[str, Any] = {
            "query": clean_query,
            "max_search_results": _coerce_positive_int(
                max_results,
                default=self.config.max_results_default,
            ),
            "fetch_top_n": _coerce_positive_int(
                fetch_top_n,
                default=self.config.fetch_top_n_default,
            ),
        }
        clean_language = _clean_str(language)
        if clean_language:
            request_params["language"] = clean_language

        if isinstance(engines, str):
            clean_engines = _clean_str(engines)
            if clean_engines:
                request_params["engines"] = clean_engines
        elif isinstance(engines, (list, tuple)):
            joined_engines = ",".join(
                clean for clean in (_clean_str(item) for item in engines) if clean
            )
            if joined_engines:
                request_params["engines"] = joined_engines

        payload = self._request(request_params)
        return self._normalize_response(
            query=clean_query,
            request_params=request_params,
            payload=payload,
        )

    def fetch_pages(
        self,
        urls: list[str],
        *,
        max_chars: int | None = None,
    ) -> list[FetchedPage]:
        """
        Fetch the text content of the given URLs via the search_service /fetch/ endpoint.

        Returns a list of FetchedPage (one per URL, preserving order).
        Never raises — per-URL errors are reported in FetchedPage.status / .error.
        """
        if not self.is_enabled:
            raise SearchIntegrationError(
                "Search integration is disabled or not configured."
            )

        clean_urls = [u for u in (_clean_str(u) for u in urls) if u]
        if not clean_urls:
            return []

        fetch_url = urljoin(
            f"{self.config.base_url}/",
            self.config.fetch_endpoint.lstrip("/"),
        )
        body: dict[str, Any] = {"urls": clean_urls}
        if max_chars is not None:
            body["max_chars"] = max_chars

        try:
            raw = self._transport(fetch_url, body, self.config.timeout_sec)
        except SearchIntegrationError:
            raise
        except Exception as exc:
            raise SearchIntegrationError(f"Fetch request failed: {exc}") from exc

        pages_raw = raw.get("pages") or []
        results: list[FetchedPage] = []
        url_set = {u: True for u in clean_urls}

        for item in pages_raw:
            if not isinstance(item, dict):
                continue
            url = _clean_str(item.get("url")) or ""
            if url not in url_set:
                continue
            results.append(
                FetchedPage(
                    url=url,
                    content=str(item.get("content") or ""),
                    status=str(item.get("status") or "ok"),
                    error=_clean_str(item.get("error")),
                )
            )

        # Preserve original order; fill in any missing URLs
        fetched_by_url = {p.url: p for p in results}
        ordered: list[FetchedPage] = []
        for url in clean_urls:
            if url in fetched_by_url:
                ordered.append(fetched_by_url[url])
            else:
                ordered.append(FetchedPage(url=url, content="", status="error", error="not returned"))

        return ordered


