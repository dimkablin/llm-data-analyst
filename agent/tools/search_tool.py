from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import PrivateAttr

from agent.prompts import search_tool_prompt
from agent.tools.base_tool import BaseExecTool
from backend.search_integration import (
    FetchedPage,
    SearchIntegrationError,
    SearchIntegrationService,
)


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
        table = pd.DataFrame(
            payload["rows"],
            columns=[
                "rank",
                "title",
                "url",
                "snippet",
                "source_name",
                "published_at",
            ],
        )
        return {
            "schema_version": "1.0",
            "artifact_type": "table",
            "items": {str(payload["artifact_name"]): table},
            "source": payload["source"],
            "recipe": payload["recipe"],
            "meta": payload["meta"],
        }


class SearchTool(BaseExecTool):
    name: str = "search_tool"
    artifact_name: str = "table"
    human_name: str = "результатов поиска"
    description: str = search_tool_prompt
    allowed_libs: set[str] = {"pandas", "numpy"}
    allowed_artifact_types: tuple = (pd.DataFrame, pd.Series)
    _search_service: SearchIntegrationService = PrivateAttr()

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        search_service: SearchIntegrationService,
        execution_timeout_sec: float = 25.0,
        tool_cache_size: int = 48,
    ) -> None:
        self._search_service = search_service
        super().__init__(
            df,
            execution_timeout_sec=execution_timeout_sec,
            include_plotly=False,
            tool_cache_size=tool_cache_size,
            db_runtime_config=None,
        )

    def get_execution_scope(self) -> dict[str, Any]:
        return {
            "search": SearchToolHelper(
                service=self._search_service,
                tool_name=self.name,
            )
        }

    def _run(self, code: str) -> tuple[str, dict[str, object]]:
        if not self._search_service.is_enabled:
            text = (
                "Ошибка search_tool: search integration недоступна. "
                "Проверь SEARCH_BACKEND_URL и SEARCH_ENABLED."
            )
            return text, {self.artifact_name: None, "text": text}

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
