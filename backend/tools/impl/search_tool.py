from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import PrivateAttr

from backend.agent.prompts import search_tool_prompt
from backend.integrations.search import (
    FetchedPage,
    SearchIntegrationError,
    SearchIntegrationService,
)
from backend.tools.impl.base_tool import BaseExecTool


@dataclass
class SearchToolHelper:
    service: SearchIntegrationService
    tool_name: str = "search_tool"

    def search(
        self,
        query: str,
        *,
        max_results: int | None = None,
        fetch_top_n: int | None = None,
        language: str | None = None,
        engines: str | list[str] | None = None,
    ) -> dict[str, Any]:
        result = self.service.search(
            query,
            max_results=max_results,
            fetch_top_n=fetch_top_n,
            language=language,
            engines=engines,
        )
        payload = self.service.build_artifact_payload(
            result,
            tool_name=self.tool_name,
        )
        return {
            "query": result.query,
            "answer": result.answer,
            "results": payload["rows"],
            "sources": list(result.sources),
            "source": payload["source"],
            "recipe": payload["recipe"],
            "meta": payload["meta"],
        }

    def fetch(
        self,
        urls: str | list[str],
        *,
        max_chars: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch the text content of one or more URLs.

        Returns a list of dicts: [{"url": ..., "content": ..., "status": "ok"|"error", "error": ...}]

        Use after search() when you want to read the full text of specific pages
        selected from the search results.
        """
        if isinstance(urls, str):
            urls = [urls]
        pages: list[FetchedPage] = self.service.fetch_pages(urls, max_chars=max_chars)
        return [
            {
                "url": p.url,
                "content": p.content,
                "status": p.status,
                "error": p.error,
            }
            for p in pages
        ]

    def search_result(
        self,
        query: str,
        *,
        artifact_name: str = "search_results",
        max_results: int | None = None,
        fetch_top_n: int | None = None,
        language: str | None = None,
        engines: str | list[str] | None = None,
    ) -> dict[str, Any]:
        result = self.service.search(
            query,
            max_results=max_results,
            fetch_top_n=fetch_top_n,
            language=language,
            engines=engines,
        )
        payload = self.service.build_artifact_payload(
            result,
            artifact_name=artifact_name,
            tool_name=self.tool_name,
        )
        json_data: dict[str, Any] = {
            "query": result.query,
            "answer": result.answer,
            "results": payload["rows"],
            "sources": list(result.sources),
        }
        return {
            "schema_version": "1.0",
            "artifact_type": "json",
            "items": {str(payload["artifact_name"]): json_data},
            "source": payload["source"],
            "recipe": payload["recipe"],
            "meta": payload["meta"],
        }


class SearchTool(BaseExecTool):
    name: str = "search_tool"
    artifact_name: str = "json"
    human_name: str = "результатов поиска"
    description: str = search_tool_prompt
    allowed_libs: set[str] = {"pandas", "numpy"}
    allowed_artifact_types: tuple = (dict,)
    _search_service: SearchIntegrationService = PrivateAttr()

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        search_service: SearchIntegrationService,
        execution_timeout_sec: float = 25.0,
        tool_cache_size: int = 48,
        sandbox: object | None = None,
    ) -> None:
        super().__init__(
            df,
            execution_timeout_sec=execution_timeout_sec,
            include_plotly=False,
            tool_cache_size=tool_cache_size,
            db_runtime_config=None,
            sandbox=sandbox,
        )
        object.__setattr__(self, "_search_service", search_service)

    # Keys that identify a raw search.search() result dict.
    _SEARCH_RAW_KEYS = frozenset({"query", "results", "sources", "source", "recipe", "meta"})

    def _validate_tool_contract(
        self, tool_result: object
    ) -> tuple[dict[str, object] | None, str]:
        """Extend base contract to handle raw search.search() output.

        When LLM calls search.search() instead of search.search_result(), the
        result is a dict with query/answer/results/... keys.  Wrap it as a
        JSON artifact so it passes downstream validation.
        """
        if isinstance(tool_result, dict) and self._SEARCH_RAW_KEYS.issubset(tool_result.keys()):
            json_data: dict[str, Any] = {
                "query": tool_result.get("query"),
                "answer": tool_result.get("answer"),
                "results": tool_result.get("results") or [],
                "sources": tool_result.get("sources") or [],
            }
            return {"search_results": json_data}, ""
        return super()._validate_tool_contract(tool_result)

    def get_execution_scope(self) -> dict[str, Any]:
        return {
            "search": SearchToolHelper(
                service=self._search_service,
                tool_name=self.name,
            )
        }

    @staticmethod
    def _try_parse_dict_input(code: str) -> dict[str, Any] | None:
        """Detect when LLM passes a plain dict {query, language, ...} instead of Python code."""
        stripped = code.strip()
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict) and "query" in parsed:
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, dict) and "query" in parsed:
                return parsed
        except (ValueError, SyntaxError):
            pass
        m = re.search(r"""["']?query["']?\s*[=:]\s*["']([^"']+)["']""", stripped)
        if m:
            lang_m = re.search(r"""["']?language["']?\s*[=:]\s*["']([^"']+)["']""", stripped)
            return {"query": m.group(1), "language": lang_m.group(1) if lang_m else None}
        return None

    _STRUCT_KW_KEYS = frozenset(
        {
            "query",
            "queries",
            "language",
            "max_results",
            "fetch_top_n",
            "artifact_name",
            "engines",
        }
    )

    @staticmethod
    def _normalize_query_dict(d: dict[str, Any]) -> dict[str, Any] | None:
        if "query" in d:
            return d
        if "queries" not in d:
            return None
        qv = d["queries"]
        if isinstance(qv, list):
            query = " ".join(str(q) for q in qv if q)
        else:
            query = str(qv)
        return {**d, "query": query}

    def _run_direct(self, params: dict[str, Any]) -> tuple[str, dict[str, object]]:
        """Execute search directly from parsed dict params (no code execution)."""
        helper = SearchToolHelper(service=self._search_service, tool_name=self.name)
        artifact_name = str(params.get("artifact_name") or "search_results")
        result = helper.search_result(
            str(params["query"]),
            artifact_name=artifact_name,
            max_results=params.get("max_results"),
            language=params.get("language"),
        )
        # Same contract as BaseExecTool._run: artifact under self.artifact_name ("json"),
        # not raw envelope "items" — otherwise ToolCollector never registers the artifact.
        items = result.get("items") if isinstance(result, dict) else None
        meta = result.get("meta") if isinstance(result, dict) else None
        search_meta = meta.get("search") if isinstance(meta, dict) else None
        lines: list[str] = []
        q = str(params.get("query") or "").strip()
        if q:
            lines.append(f"Search completed for: {q}")
        if isinstance(search_meta, dict):
            answer = str(search_meta.get("answer", "")).strip()
            if answer:
                lines.append(answer)
            rc = search_meta.get("result_count")
            if isinstance(rc, int):
                lines.append(f"Normalized search results: {rc}.")
            top_titles = search_meta.get("top_titles")
            if isinstance(top_titles, list) and top_titles:
                preview = "; ".join(
                    str(item).strip() for item in top_titles[:3] if str(item).strip()
                )
                if preview:
                    lines.append(f"Top hits: {preview}")
        enriched = "\n".join(lines).strip() or f"Search completed for: {q}"
        payload: dict[str, object] = {
            "text": enriched,
            "code": f"search.search_result({params['query']!r})",
            self.artifact_name: dict(items) if isinstance(items, dict) else {},
        }
        for k in ("source", "recipe", "meta"):
            if isinstance(result, dict) and k in result:
                payload[k] = result[k]
        return enriched, payload

    def _run(self, code: str = "", **kwargs: Any) -> tuple[str, dict[str, object]]:
        for _k in ("run_manager", "callbacks", "tags", "metadata", "config"):
            kwargs.pop(_k, None)
        if "code" in kwargs:
            cw = kwargs.pop("code")
            if cw is not None and str(cw).strip():
                code = str(cw)
        struct = {
            k: kwargs.pop(k)
            for k in list(kwargs.keys())
            if k in self._STRUCT_KW_KEYS
        }
        kw_params = self._normalize_query_dict(struct) if struct else None

        if not self._search_service.is_enabled:
            text = (
                "Ошибка search_tool: search integration недоступна. "
                "Проверь SEARCH_BACKEND_URL и SEARCH_ENABLED."
            )
            return text, {self.artifact_name: None, "text": text}

        if kw_params:
            try:
                return self._run_direct(kw_params)
            except SearchIntegrationError as exc:
                message = str(exc).strip() or "Search failed."
                return message, {self.artifact_name: None, "text": message}
            except Exception as exc:
                message = f"Search error: {exc}"
                return message, {self.artifact_name: None, "text": message}

        # Fallback: LLM passed a dict {query, language, ...} instead of Python code
        dict_params = self._try_parse_dict_input(str(code or "").strip())
        if dict_params:
            try:
                return self._run_direct(dict_params)
            except SearchIntegrationError as exc:
                message = str(exc).strip() or "Search failed."
                return message, {self.artifact_name: None, "text": message}
            except Exception as exc:
                message = f"Search error: {exc}"
                return message, {self.artifact_name: None, "text": message}

        try:
            text, payload = super()._run(code)
        except SearchIntegrationError as exc:
            message = str(exc).strip() or "Search integration failed."
            return message, {self.artifact_name: None, "text": message}

        meta = payload.get("meta")
        if not isinstance(meta, dict):
            return text, payload
        search_meta = meta.get("search")
        if not isinstance(search_meta, dict):
            return text, payload

        query = str(search_meta.get("query", "")).strip()
        answer = str(search_meta.get("answer", "")).strip()
        result_count = search_meta.get("result_count")
        top_titles = search_meta.get("top_titles")
        lines: list[str] = []
        if query:
            lines.append(f"Search completed for: {query}")
        if answer:
            lines.append(answer)
        if isinstance(result_count, int):
            lines.append(f"Normalized search results: {result_count}.")
        if isinstance(top_titles, list) and top_titles:
            preview = "; ".join(str(item).strip() for item in top_titles[:3] if str(item).strip())
            if preview:
                lines.append(f"Top hits: {preview}")
        enriched_text = "\n".join(lines).strip()
        if enriched_text:
            payload["text"] = enriched_text
            return enriched_text, payload
        return text, payload


