from __future__ import annotations

from typing import Any, ClassVar

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from backend.integrations.search import (
    SearchIntegrationError,
    SearchIntegrationService,
)
from backend.tools.instructions import tool_description


class SearchToolArgs(BaseModel):
    query: str | None = Field(default=None)
    queries: list[str] | str | None = Field(default=None)
    language: str | None = Field(default=None)
    max_results: int | None = Field(default=None)
    fetch_top_n: int | None = Field(default=None)
    artifact_name: str = Field(default="search_results")
    engines: list[str] | str | None = Field(default=None)

    @model_validator(mode="after")
    def normalize_query(self) -> "SearchToolArgs":
        if self.query is None and self.queries is not None:
            if isinstance(self.queries, list):
                self.query = " ".join(
                    str(item) for item in self.queries if str(item).strip()
                )
            else:
                self.query = str(self.queries)
        return self


class SearchTool(BaseTool):
    name: str = "search_tool"
    artifact_name: str = "json"
    description: str = tool_description("search_tool")
    args_schema: type[BaseModel] = SearchToolArgs
    response_format: str = "content_and_artifact"
    parallel_safe: ClassVar[bool] = True

    _search_service: SearchIntegrationService = PrivateAttr()

    def __init__(self, *, search_service: SearchIntegrationService) -> None:
        super().__init__()
        self._search_service = search_service

    def _run_direct(self, params: dict[str, Any]) -> tuple[str, dict[str, object]]:
        artifact_name = str(params.get("artifact_name") or "search_results")
        search_result = self._search_service.search(
            str(params["query"]),
            max_results=params.get("max_results"),
            fetch_top_n=params.get("fetch_top_n"),
            language=params.get("language"),
            engines=params.get("engines"),
        )
        artifact_payload = self._search_service.build_artifact_payload(
            search_result,
            artifact_name=artifact_name,
            tool_name=self.name,
        )
        items = {
            str(artifact_payload["artifact_name"]): {
                "query": search_result.query,
                "answer": search_result.answer,
                "results": artifact_payload["rows"],
                "sources": list(search_result.sources),
            }
        }
        meta = artifact_payload.get("meta")
        search_meta = meta.get("search") if isinstance(meta, dict) else None
        lines: list[str] = []
        q = str(params.get("query") or "").strip()
        if q:
            lines.append(f"Search completed for: {q}")
        if isinstance(search_meta, dict):
            answer = str(search_meta.get("answer", "")).strip()
            if answer:
                lines.append(answer)
            result_count = search_meta.get("result_count")
            if isinstance(result_count, int):
                lines.append(f"Found results: {result_count}.")
            top_titles = search_meta.get("top_titles")
            if isinstance(top_titles, list) and top_titles:
                preview = "; ".join(
                    str(item).strip() for item in top_titles[:3] if str(item).strip()
                )
                if preview:
                    lines.append(f"Top results: {preview}")
        enriched = "\n".join(lines).strip() or f"Search completed for: {q}"
        payload: dict[str, object] = {
            "text": enriched,
            "code": f"search.search_result({params['query']!r})",
            self.artifact_name: items,
        }
        for key in ("source", "recipe", "meta"):
            if key in artifact_payload:
                payload[key] = artifact_payload[key]
        return enriched, payload

    def _run(
        self,
        query: str | None = None,
        queries: list[str] | str | None = None,
        language: str | None = None,
        max_results: int | None = None,
        fetch_top_n: int | None = None,
        artifact_name: str = "search_results",
        engines: list[str] | str | None = None,
        **kwargs: Any,
    ) -> tuple[str, dict[str, object]]:
        for key in ("run_manager", "callbacks", "tags", "metadata", "config"):
            kwargs.pop(key, None)
        args = SearchToolArgs(
            query=query,
            queries=queries,
            language=language,
            max_results=max_results,
            fetch_top_n=fetch_top_n,
            artifact_name=artifact_name,
            engines=engines,
        )
        if not args.query or not str(args.query).strip():
            text = "Tool error: search_tool query must not be empty."
            return text, {self.artifact_name: None, "text": text}
        if not self._search_service.is_enabled:
            text = "Tool error: search integration is disabled."
            return text, {self.artifact_name: None, "text": text}
        try:
            return self._run_direct(args.model_dump())
        except SearchIntegrationError as exc:
            message = str(exc).strip() or "Search failed."
            return message, {self.artifact_name: None, "text": message}
        except Exception as exc:
            message = f"Tool error: search failed: {exc}"
            return message, {self.artifact_name: None, "text": message}
