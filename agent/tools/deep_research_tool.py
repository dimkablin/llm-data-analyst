from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import PrivateAttr

from agent.prompts import deep_research_tool_prompt
from agent.tools.base_tool import BaseExecTool
from backend.deep_research_integration import (
    DeepResearchIntegrationError,
    DeepResearchIntegrationService,
)


@dataclass
class DeepResearchToolHelper:
    service: DeepResearchIntegrationService
    tool_name: str = "deep_research_tool"

    def research(
        self,
        query: str,
        *,
        max_iterations: int | None = None,
        language: str | None = None,
    ) -> dict[str, Any]:
        result = self.service.run_research(
            query,
            max_iterations=max_iterations,
            language=language,
        )
        payload = self.service.build_artifact_payload(
            result,
            tool_name=self.tool_name,
        )
        deep_meta = payload["meta"]["deep_research"]
        return {
            "query": result.query,
            "research_id": result.research_id,
            "status": result.status,
            "summary": result.summary,
            "report_text": result.report_text,
            "rows": payload["rows"],
            "sources": list(result.sources),
            "source": payload["source"],
            "recipe": payload["recipe"],
            "meta": payload["meta"],
            "warnings": list(deep_meta.get("warnings", [])),
        }

    def research_result(
        self,
        query: str,
        *,
        artifact_name: str = "deep_research_report",
        max_iterations: int | None = None,
        language: str | None = None,
    ) -> dict[str, Any]:
        result = self.service.run_research(
            query,
            max_iterations=max_iterations,
            language=language,
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
                "kind",
                "title",
                "content",
                "url",
                "source_name",
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


class DeepResearchTool(BaseExecTool):
    name: str = "deep_research_tool"
    artifact_name: str = "table"
    human_name: str = "результатов deep research"
    description: str = deep_research_tool_prompt
    allowed_libs: set[str] = {"pandas", "numpy"}
    allowed_artifact_types: tuple = (pd.DataFrame, pd.Series)
    _deep_research_service: DeepResearchIntegrationService = PrivateAttr()

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        deep_research_service: DeepResearchIntegrationService,
        execution_timeout_sec: float = 25.0,
        tool_cache_size: int = 48,
    ) -> None:
        self._deep_research_service = deep_research_service
        super().__init__(
            df,
            execution_timeout_sec=execution_timeout_sec,
            include_plotly=False,
            tool_cache_size=tool_cache_size,
            db_runtime_config=None,
        )

    def get_execution_scope(self) -> dict[str, Any]:
        return {
            "deep_research": DeepResearchToolHelper(
                service=self._deep_research_service,
                tool_name=self.name,
            )
        }

    def _run(self, code: str) -> tuple[str, dict[str, object]]:
        if not self._deep_research_service.is_enabled:
            text = (
                "Ошибка deep_research_tool: deep research integration недоступна. "
                "Проверь DEEP_RESEARCH_BACKEND_URL и DEEP_RESEARCH_ENABLED."
            )
            return text, {self.artifact_name: None, "text": text}

        try:
            text, payload = super()._run(code)
        except DeepResearchIntegrationError as exc:
            message = str(exc).strip() or "Deep research integration failed."
            return message, {self.artifact_name: None, "text": message}

        meta = payload.get("meta")
        if not isinstance(meta, dict):
            return text, payload
        deep_meta = meta.get("deep_research")
        if not isinstance(deep_meta, dict):
            return text, payload

        query = str(deep_meta.get("query", "")).strip()
        research_id = str(deep_meta.get("research_id", "")).strip()
        status = str(deep_meta.get("status", "")).strip()
        summary = str(deep_meta.get("summary", "")).strip()
        result_count = deep_meta.get("result_count")

        lines: list[str] = []
        if query:
            lines.append(f"Deep research completed for: {query}")
        if research_id:
            lines.append(f"Research ID: {research_id}")
        if status:
            lines.append(f"Status: {status}")
        if summary:
            lines.append(summary)
        if isinstance(result_count, int):
            lines.append(f"Normalized research rows: {result_count}.")
        enriched_text = "\n".join(lines).strip()
        if enriched_text:
            payload["text"] = enriched_text
            return enriched_text, payload
        return text, payload
