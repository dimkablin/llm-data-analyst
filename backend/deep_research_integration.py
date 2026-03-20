from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from backend.artifact_meta import build_source_query_recipe_step
from backend.integration_contract import build_operation_meta, build_source_descriptor


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
class DeepResearchConfig:
    enabled: bool
    base_url: str
    create_endpoint: str
    execute_endpoint: str
    detail_endpoint: str
    create_timeout_sec: float
    execute_timeout_sec: float
    poll_timeout_sec: float
    poll_interval_sec: float
    max_iterations_default: int
    language_default: str
    source_type: str = "deep_research"
    source_ref_id: str = "deep_research"
    source_label: str = "Deep Research"
    source_mode: str = "external"

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
    ) -> "DeepResearchConfig":
        source_env = env or os.environ
        base_url = _clean_str(source_env.get("DEEP_RESEARCH_BACKEND_URL")) or ""
        enabled_default = bool(base_url)
        enabled = _get_bool(source_env, "DEEP_RESEARCH_ENABLED", enabled_default)
        return cls(
            enabled=enabled,
            base_url=base_url.rstrip("/"),
            create_endpoint=_clean_str(source_env.get("DEEP_RESEARCH_CREATE_ENDPOINT"))
            or "/api/v1/research/",
            execute_endpoint=_clean_str(source_env.get("DEEP_RESEARCH_EXECUTE_ENDPOINT"))
            or "/api/v1/research/{id}/execute",
            detail_endpoint=_clean_str(source_env.get("DEEP_RESEARCH_DETAIL_ENDPOINT"))
            or "/api/v1/research/{id}",
            create_timeout_sec=_coerce_positive_float(
                source_env.get("DEEP_RESEARCH_CREATE_TIMEOUT_SEC"),
                default=30.0,
            ),
            execute_timeout_sec=_coerce_positive_float(
                source_env.get("DEEP_RESEARCH_EXECUTE_TIMEOUT_SEC"),
                default=270.0,
            ),
            poll_timeout_sec=_coerce_positive_float(
                source_env.get("DEEP_RESEARCH_POLL_TIMEOUT_SEC"),
                default=120.0,
            ),
            poll_interval_sec=_coerce_positive_float(
                source_env.get("DEEP_RESEARCH_POLL_INTERVAL_SEC"),
                default=2.0,
            ),
            max_iterations_default=_coerce_positive_int(
                source_env.get("DEEP_RESEARCH_MAX_ITERATIONS_DEFAULT"),
                default=3,
            ),
            language_default=_clean_str(
                source_env.get("DEEP_RESEARCH_LANGUAGE_DEFAULT")
            )
            or "ru",
            source_label=_clean_str(
                source_env.get("DEEP_RESEARCH_SOURCE_LABEL")
            )
            or "Deep Research",
        )

    @property
    def available(self) -> bool:
        return self.enabled and bool(self.base_url)


@dataclass(frozen=True)
class DeepResearchQueryResult:
    query: str
    research_id: str
    status: str | None
    summary: str | None
    report_text: str | None
    rows: list[dict[str, Any]]
    sources: list[str]
    warnings: list[str]
    request_params: dict[str, Any]

    @property
    def result_count(self) -> int:
        return len(self.rows)


class DeepResearchIntegrationError(RuntimeError):
    pass


DeepResearchTransport = Callable[
    [str, str, dict[str, Any] | None, float],
    dict[str, Any],
]


def _default_transport(
    method: str,
    url: str,
    payload: dict[str, Any] | None,
    timeout_sec: float,
) -> dict[str, Any]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout_sec) as response:
            raw_body = response.read()
    except HTTPError as exc:
        body_preview = exc.read().decode("utf-8", errors="replace")[:500]
        raise DeepResearchIntegrationError(
            f"Deep research backend returned HTTP {exc.code}: {body_preview}"
        ) from exc
    except URLError as exc:
        raise DeepResearchIntegrationError(
            f"Deep research backend is unavailable: {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise DeepResearchIntegrationError("Deep research backend request timed out.") from exc

    if not raw_body:
        return {}
    try:
        decoded = json.loads(raw_body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        preview = raw_body.decode("utf-8", errors="replace")[:500]
        raise DeepResearchIntegrationError(
            f"Deep research backend returned invalid JSON: {preview!r}"
        ) from exc
    if not isinstance(decoded, dict):
        raise DeepResearchIntegrationError(
            "Deep research backend returned a non-object JSON payload."
        )
    return decoded


class DeepResearchIntegrationService:
    def __init__(
        self,
        config: DeepResearchConfig,
        *,
        transport: DeepResearchTransport | None = None,
    ) -> None:
        self.config = config
        self._transport = transport or _default_transport

    @classmethod
    def from_env(cls) -> "DeepResearchIntegrationService":
        return cls(DeepResearchConfig.from_env())

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
            display_name_ru="Глубокое исследование",
            source_mode=self.config.source_mode,
            enabled=self.config.enabled,
            available=self.config.available,
            description="External deep research integration.",
            description_ru="Развернутый внешний ресерч с подробным результатом.",
            capabilities=["deep_research", "research_report", "web_results"],
            requires_session_data=False,
            timeout_hint_sec=self.config.execute_timeout_sec,
        )

    @staticmethod
    def _artifact_name(value: str | None) -> str:
        text = str(value or "").strip()
        return text or "deep_research_report"

    def build_artifact_payload(
        self,
        result: DeepResearchQueryResult,
        *,
        artifact_name: str = "deep_research_report",
        tool_name: str = "deep_research_tool",
    ) -> dict[str, Any]:
        return {
            "artifact_name": self._artifact_name(artifact_name),
            "rows": copy.deepcopy(result.rows),
            "source": self.source_ref(),
            "recipe": [
                build_source_query_recipe_step(
                    query=result.query,
                    source_type=self.config.source_type,
                    tool_name=tool_name,
                    title="Deep Research Query",
                    summary=result.summary or result.report_text or f"Deep research for: {result.query}",
                    params=result.request_params,
                    result_count=result.result_count,
                )
            ],
            "meta": {
                "deep_research": build_operation_meta(
                    status=result.status,
                    warnings=result.warnings,
                    request_params=result.request_params,
                    timeout_sec=self.config.execute_timeout_sec,
                    extra={
                        "query": result.query,
                        "research_id": result.research_id,
                        "summary": result.summary,
                        "report_text": result.report_text,
                        "result_count": result.result_count,
                        "sources": list(result.sources),
                        "poll_timeout_sec": self.config.poll_timeout_sec,
                    },
                )
            },
        }

    def _endpoint_url(self, endpoint: str, research_id: str | None = None) -> str:
        if not self.config.base_url:
            raise DeepResearchIntegrationError(
                "Deep research integration is not configured. Set DEEP_RESEARCH_BACKEND_URL first."
            )
        clean_endpoint = endpoint
        if research_id:
            clean_endpoint = clean_endpoint.format(id=research_id)
        return urljoin(f"{self.config.base_url}/", clean_endpoint.lstrip("/"))

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout_sec: float,
        research_id: str | None = None,
    ) -> dict[str, Any]:
        try:
            return self._transport(
                method,
                self._endpoint_url(endpoint, research_id),
                payload,
                timeout_sec,
            )
        except DeepResearchIntegrationError:
            raise
        except HTTPError as exc:
            body_preview = ""
            try:
                body_preview = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                body_preview = ""
            suffix = f": {body_preview}" if body_preview else ""
            raise DeepResearchIntegrationError(
                f"Deep research backend returned HTTP {exc.code}{suffix}"
            ) from exc
        except URLError as exc:
            raise DeepResearchIntegrationError(
                f"Deep research backend is unavailable: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise DeepResearchIntegrationError(
                "Deep research backend request timed out."
            ) from exc

    @staticmethod
    def _extract_research_id(payload: dict[str, Any]) -> str | None:
        for key in ("research_id", "id"):
            clean = _clean_str(payload.get(key))
            if clean:
                return clean
        research = payload.get("research")
        if isinstance(research, dict):
            for key in ("research_id", "id"):
                clean = _clean_str(research.get(key))
                if clean:
                    return clean
        return None

    @staticmethod
    def _extract_status(*payloads: object) -> str | None:
        for payload in payloads:
            if not isinstance(payload, dict):
                continue
            clean = _clean_str(payload.get("status"))
            if clean:
                return clean
            nested = payload.get("research")
            if isinstance(nested, dict):
                clean = _clean_str(nested.get("status"))
                if clean:
                    return clean
        return None

    @staticmethod
    def _extract_report_container(payload: object) -> dict[str, Any] | str | None:
        if not isinstance(payload, dict):
            return None
        for key in ("report", "final_report"):
            value = payload.get(key)
            if isinstance(value, (dict, str)):
                return value
        research = payload.get("research")
        if isinstance(research, dict):
            for key in ("report", "final_report"):
                value = research.get(key)
                if isinstance(value, (dict, str)):
                    return value
        return None

    @staticmethod
    def _normalize_source_rows(raw_sources: object) -> tuple[list[dict[str, Any]], list[str]]:
        rows: list[dict[str, Any]] = []
        urls: list[str] = []
        if not isinstance(raw_sources, list):
            return rows, urls

        for index, item in enumerate(raw_sources, start=1):
            if isinstance(item, str):
                clean = _clean_str(item)
                if not clean:
                    continue
                urls.append(clean)
                rows.append(
                    {
                        "rank": index,
                        "kind": "source",
                        "title": clean,
                        "content": "",
                        "url": clean,
                        "source_name": "",
                    }
                )
                continue
            if not isinstance(item, dict):
                continue
            title = _clean_str(item.get("title") or item.get("name")) or _clean_str(
                item.get("url") or item.get("link")
            )
            url = _clean_str(item.get("url") or item.get("link")) or ""
            snippet = _clean_str(
                item.get("snippet") or item.get("description") or item.get("content")
            ) or ""
            source_name = _clean_str(item.get("source") or item.get("source_name")) or ""
            if url:
                urls.append(url)
            if not title:
                continue
            rows.append(
                {
                    "rank": index,
                    "kind": "source",
                    "title": title,
                    "content": snippet,
                    "url": url,
                    "source_name": source_name,
                }
            )
        return rows, urls

    @staticmethod
    def _normalize_report(
        report: dict[str, Any] | str | None,
        detail_payload: dict[str, Any] | None,
    ) -> tuple[str | None, str | None, list[dict[str, Any]], list[str], list[str]]:
        warnings: list[str] = []
        rows: list[dict[str, Any]] = []
        sources: list[str] = []
        summary: str | None = None
        report_text: str | None = None

        if isinstance(report, str):
            clean = _clean_str(report)
            if clean:
                summary = clean
                report_text = clean

        if isinstance(report, dict):
            summary = _clean_str(
                report.get("summary") or report.get("answer") or report.get("title")
            )
            report_text = _clean_str(
                report.get("content") or report.get("report") or report.get("body")
            )
            raw_sections = report.get("sections")
            if isinstance(raw_sections, list):
                for index, section in enumerate(raw_sections, start=1):
                    if not isinstance(section, dict):
                        continue
                    title = _clean_str(section.get("title") or section.get("name"))
                    content = _clean_str(
                        section.get("content") or section.get("summary") or section.get("body")
                    )
                    if not title and not content:
                        continue
                    rows.append(
                        {
                            "rank": index,
                            "kind": "section",
                            "title": title or f"Section {index}",
                            "content": content or "",
                            "url": "",
                            "source_name": "",
                        }
                    )
            raw_findings = report.get("findings")
            if isinstance(raw_findings, list):
                start_rank = len(rows) + 1
                for offset, finding in enumerate(raw_findings, start=start_rank):
                    content = _clean_str(
                        finding.get("content") if isinstance(finding, dict) else finding
                    )
                    if not content:
                        continue
                    title = _clean_str(
                        finding.get("title") if isinstance(finding, dict) else None
                    ) or f"Finding {offset - start_rank + 1}"
                    rows.append(
                        {
                            "rank": offset,
                            "kind": "finding",
                            "title": title,
                            "content": content,
                            "url": "",
                            "source_name": "",
                        }
                    )
            source_rows, source_urls = DeepResearchIntegrationService._normalize_source_rows(
                report.get("sources")
            )
            sources.extend(source_urls)
            if source_rows and not rows:
                rows.extend(source_rows)

        if isinstance(detail_payload, dict):
            detail_source_rows, detail_urls = DeepResearchIntegrationService._normalize_source_rows(
                detail_payload.get("sources")
            )
            sources.extend(detail_urls)
            if detail_source_rows and not rows:
                rows.extend(detail_source_rows)
            if not summary:
                summary = _clean_str(
                    detail_payload.get("summary") or detail_payload.get("answer")
                )
            if not report_text:
                report_text = _clean_str(detail_payload.get("content"))

        deduped_sources: list[str] = []
        seen_sources: set[str] = set()
        for url in sources:
            clean = _clean_str(url)
            if not clean or clean in seen_sources:
                continue
            seen_sources.add(clean)
            deduped_sources.append(clean)

        if not rows:
            warnings.append("Deep research backend returned no normalized research rows.")
        return summary, report_text, rows, deduped_sources, warnings

    def _wait_for_report(self, research_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.config.poll_timeout_sec
        last_detail: dict[str, Any] = {}
        terminal_statuses = {
            "done",
            "completed",
            "completed_with_warnings",
            "finished",
            "success",
            "failed",
            "cancelled",
            "error",
        }

        while time.monotonic() < deadline:
            last_detail = self._request(
                "GET",
                self.config.detail_endpoint,
                timeout_sec=self.config.create_timeout_sec,
                research_id=research_id,
            )
            report = self._extract_report_container(last_detail)
            status = (self._extract_status(last_detail) or "").strip().lower()
            if report is not None or status in terminal_statuses:
                return last_detail
            time.sleep(self.config.poll_interval_sec)
        return last_detail

    def run_research(
        self,
        query: str,
        *,
        max_iterations: int | None = None,
        language: str | None = None,
    ) -> DeepResearchQueryResult:
        if not self.is_enabled:
            raise DeepResearchIntegrationError(
                "Deep research integration is disabled or not configured."
            )

        clean_query = _clean_str(query)
        if not clean_query:
            raise DeepResearchIntegrationError("Deep research query must not be empty.")

        request_params: dict[str, Any] = {
            "query": clean_query,
            "max_iterations": _coerce_positive_int(
                max_iterations,
                default=self.config.max_iterations_default,
            ),
            "language": _clean_str(language) or self.config.language_default,
        }

        created = self._request(
            "POST",
            self.config.create_endpoint,
            payload=request_params,
            timeout_sec=self.config.create_timeout_sec,
        )
        research_id = self._extract_research_id(created)
        if not research_id:
            raise DeepResearchIntegrationError(
                "Deep research backend did not return research_id."
            )

        executed = self._request(
            "POST",
            self.config.execute_endpoint,
            timeout_sec=self.config.execute_timeout_sec,
            research_id=research_id,
        )

        report = self._extract_report_container(executed)
        detail: dict[str, Any] = executed
        if report is None:
            polled = self._wait_for_report(research_id)
            if polled:
                detail = polled
                report = self._extract_report_container(polled)

        summary, report_text, rows, sources, warnings = self._normalize_report(
            report,
            detail,
        )
        status = self._extract_status(detail, executed, created)
        if report is None:
            warnings.append("Deep research backend did not return a final report payload.")

        return DeepResearchQueryResult(
            query=clean_query,
            research_id=research_id,
            status=status,
            summary=summary,
            report_text=report_text,
            rows=rows,
            sources=sources,
            warnings=warnings,
            request_params=copy.deepcopy(request_params),
        )
