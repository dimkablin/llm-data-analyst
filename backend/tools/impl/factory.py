"""Tool factories — one class per LangChain tool.

Each factory encapsulates three concerns:
  1. Whether a tool *can* be used given the current context (service live + user allowed + data present).
  2. How to *construct* the tool when it can.
  3. The string key under which the tool is registered.

``ToolFactory`` is a structural ``Protocol``, so factories do not need to inherit from
a common base class — duck typing is enough.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from backend.integrations.anomaly_planfact import AnomalyPlanfactIntegrationService
from backend.integrations.forecast import ForecastIntegrationService
from backend.integrations.rag import RAGService
from backend.integrations.search import SearchIntegrationService
from backend.tools.context import ToolBuildContext
from backend.tools.impl.anomaly_planfact_tool import AnomalyPlanfactTool
from backend.tools.impl.data_catalog_tool import DataCatalogTool
from backend.tools.impl.database_tool import DatabaseTool
from backend.tools.impl.forecast_tool import ForecastTool
from backend.tools.impl.generation_tools import (
    GenerateReportTool,
    GenerateSummaryTool,
    artifact_summaries_from_history,
)
from backend.tools.impl.get_tool_instructions_tool import GetToolInstructionsTool
from backend.tools.impl.memory_tool import MemoryTool, SessionNoteTool
from backend.tools.impl.pandas_tool import PandasTool
from backend.tools.impl.planner_tool import PlannerTool
from backend.tools.impl.plotly_tool import PlotlyTool
from backend.tools.impl.rag_tool import RagTool
from backend.tools.impl.search_tool import SearchTool
from backend.tools.impl.sql_tool import SQLTool
from backend.tools.policy import is_tool_allowed


@runtime_checkable
class ToolFactory(Protocol):
    """Structural protocol every concrete factory must satisfy."""

    key: str

    def is_available(self, ctx: ToolBuildContext) -> bool: ...

    def build(self, ctx: ToolBuildContext) -> BaseTool: ...


# ── Integration-backed factories ──────────────────────────────────────────────


class SearchToolFactory:
    key = "search_tool"

    def __init__(self, service: SearchIntegrationService) -> None:
        self._service = service

    def is_available(self, ctx: ToolBuildContext) -> bool:
        return self._service.is_enabled and is_tool_allowed(self.key, ctx.allowed_tool_keys)

    def build(self, ctx: ToolBuildContext) -> SearchTool:
        return SearchTool(search_service=self._service)


class ForecastToolFactory:
    key = "forecast_tool"

    def __init__(self, service: ForecastIntegrationService) -> None:
        self._service = service

    def is_available(self, ctx: ToolBuildContext) -> bool:
        return (
            self._service.is_enabled
            and ctx.has_data
            and is_tool_allowed(self.key, ctx.allowed_tool_keys)
        )

    def build(self, ctx: ToolBuildContext) -> ForecastTool:
        return ForecastTool(
            ctx.tool_df,
            forecast_service=self._service,
            execution_timeout_sec=max(ctx.settings.tool_exec_timeout_sec, 90.0),
            tool_cache_size=max(4, ctx.settings.tool_cache_size // 3),
            db_runtime_config=ctx.tool_db_runtime,
            csv_session_id=ctx.csv_session_id if ctx.csv_loaded else None,
            sandbox=ctx.sandbox,
        )


class AnomalyPlanfactToolFactory:
    key = "anomaly_planfact_tool"

    def __init__(self, service: AnomalyPlanfactIntegrationService) -> None:
        self._service = service

    def is_available(self, ctx: ToolBuildContext) -> bool:
        return (
            self._service.is_enabled
            and ctx.has_data
            and is_tool_allowed(self.key, ctx.allowed_tool_keys)
        )

    def build(self, ctx: ToolBuildContext) -> AnomalyPlanfactTool:
        return AnomalyPlanfactTool(
            ctx.tool_df,
            anomaly_planfact_service=self._service,
            execution_timeout_sec=max(ctx.settings.tool_exec_timeout_sec, 90.0),
            tool_cache_size=max(4, ctx.settings.tool_cache_size // 3),
            db_runtime_config=ctx.tool_db_runtime,
            csv_session_id=ctx.csv_session_id if ctx.csv_loaded else None,
            sandbox=ctx.sandbox,
        )


class RagToolFactory:
    key = "rag_tool"

    def __init__(self, service: RAGService) -> None:
        self._service = service

    def is_available(self, ctx: ToolBuildContext) -> bool:
        return self._service.is_enabled and is_tool_allowed(self.key, ctx.allowed_tool_keys)

    def build(self, ctx: ToolBuildContext) -> RagTool:
        return RagTool(rag_service=self._service)


# ── Built-in factories ────────────────────────────────────────────────────────


class GenerateSummaryToolFactory(BaseModel):
    key: str = "generate_summary_tool"

    def is_available(self, ctx: ToolBuildContext) -> bool:
        return is_tool_allowed(self.key, ctx.allowed_tool_keys)

    def build(self, ctx: ToolBuildContext) -> GenerateSummaryTool:
        return GenerateSummaryTool(
            history=ctx.history,
            session_notes=ctx.session_notes,
            artifact_summaries=artifact_summaries_from_history(ctx.history),
        )


class GenerateReportToolFactory(BaseModel):
    key: str = "generate_report_tool"

    def is_available(self, ctx: ToolBuildContext) -> bool:
        return is_tool_allowed(self.key, ctx.allowed_tool_keys)

    def build(self, ctx: ToolBuildContext) -> GenerateReportTool:
        return GenerateReportTool(
            session_id=str(ctx.trace_context.get("session_id") or ""),
            storage_dir=ctx.settings.storage_dir,
            session_ttl_days=ctx.settings.session_ttl_days,
        )


class SQLToolFactory:
    key = "sql_tool"

    def is_available(self, ctx: ToolBuildContext) -> bool:
        if not is_tool_allowed(self.key, ctx.allowed_tool_keys):
            return False
        if ctx.tool_db_runtime is not None:
            return True
        return bool(ctx.csv_loaded and (ctx.csv_session_id or "").strip())

    def build(self, ctx: ToolBuildContext) -> SQLTool:
        return SQLTool(
            llm_base_url=ctx.settings.llm_base_url,
            llm_model=ctx.settings.llm_model,
            llm_api_key=ctx.settings.llm_api_key,
            llm_enable_thinking=ctx.settings.llm_enable_thinking,
            llm_chat_template_kwargs_enabled=ctx.settings.llm_chat_template_kwargs_enabled,
            llm_provider=ctx.settings.llm_provider,
            db_runtime_config=ctx.tool_db_runtime,
            csv_loaded=ctx.csv_loaded,
            csv_session_id=ctx.csv_session_id,
            max_rows=200,
            sandbox=ctx.sandbox,
            candidates_cache_key=ctx.candidates_cache_key,
            storage_dir=ctx.settings.storage_dir,
            semantic_context_prompt=ctx.semantic_context_prompt,
            semantic_hints=ctx.semantic_hints,
        )


class DataCatalogToolFactory:
    key = "data_catalog_tool"

    def is_available(self, ctx: ToolBuildContext) -> bool:
        if not is_tool_allowed(self.key, ctx.allowed_tool_keys):
            return False
        return ctx.source_inventory is not None and bool(ctx.source_inventory.tables)

    def build(self, ctx: ToolBuildContext) -> DataCatalogTool:
        if ctx.source_inventory is None:
            msg = "data_catalog_tool cannot be built without source_inventory"
            raise RuntimeError(msg)
        return DataCatalogTool(source_inventory=ctx.source_inventory)


class DatabaseToolFactory:
    key = "database_tool"

    def is_available(self, ctx: ToolBuildContext) -> bool:
        if not is_tool_allowed(self.key, ctx.allowed_tool_keys):
            return False
        return ctx.tool_db_runtime is not None

    def build(self, ctx: ToolBuildContext) -> DatabaseTool:
        return DatabaseTool(
            db_runtime_config=ctx.tool_db_runtime,
            sandbox=ctx.sandbox,
            timeout_sec=15.0,
        )


class PlotlyToolFactory:
    key = "plotly_tool"

    def is_available(self, ctx: ToolBuildContext) -> bool:
        if not is_tool_allowed(self.key, ctx.allowed_tool_keys):
            return False
        if ctx.tool_db_runtime is not None:
            return True
        if ctx.df is not None:
            return True
        return bool(ctx.csv_loaded and (ctx.csv_session_id or "").strip())

    def build(self, ctx: ToolBuildContext) -> PlotlyTool:
        plotly_timeout = max(ctx.settings.tool_exec_timeout_sec * 2, 60.0)
        return PlotlyTool(
            ctx.tool_df,
            execution_timeout_sec=plotly_timeout,
            tool_cache_size=ctx.settings.tool_cache_size,
            db_runtime_config=ctx.tool_db_runtime,
            sandbox=ctx.sandbox,
        )


class PandasToolFactory:
    key = "pandas_tool"

    def is_available(self, ctx: ToolBuildContext) -> bool:
        if not is_tool_allowed(self.key, ctx.allowed_tool_keys):
            return False
        if ctx.tool_db_runtime is not None:
            return True
        if ctx.df is not None:
            return True
        return bool(ctx.csv_loaded and (ctx.csv_session_id or "").strip())

    def build(self, ctx: ToolBuildContext) -> PandasTool:
        return PandasTool(
            ctx.tool_df,
            execution_timeout_sec=ctx.settings.tool_exec_timeout_sec,
            tool_cache_size=ctx.settings.tool_cache_size,
            db_runtime_config=ctx.tool_db_runtime,
            sandbox=ctx.sandbox,
        )


class PlannerToolFactory:
    """Always available; generates analysis plans via internal LLM call."""

    key = "planner_tool"

    def is_available(self, ctx: ToolBuildContext) -> bool:
        return is_tool_allowed("planner_tool", ctx.allowed_tool_keys)

    def build(self, ctx: ToolBuildContext) -> PlannerTool:
        return PlannerTool(
            llm_model=ctx.settings.llm_model,
            llm_base_url=ctx.settings.llm_base_url,
            llm_api_key=ctx.settings.llm_api_key,
            llm_provider=ctx.settings.llm_provider,
            llm_chat_template_kwargs_enabled=ctx.settings.llm_chat_template_kwargs_enabled,
        )


class GetToolInstructionsToolFactory:
    """Always available; returns full skill markdown on demand."""

    key = "get_tool_instructions"

    def __init__(self, skill_registry: Any) -> None:
        self._skill_registry = skill_registry

    def is_available(self, ctx: ToolBuildContext) -> bool:
        return True

    def build(self, ctx: ToolBuildContext) -> GetToolInstructionsTool:
        return GetToolInstructionsTool(
            self._skill_registry,
            allowed_skill_ids=ctx.allowed_skill_ids,
            allowed_tool_keys=ctx.allowed_tool_keys,
        )


class MemoryToolFactory:
    """Always available; saves long-term facts about the user."""

    key = "memory_tool"

    def __init__(self, on_note: Callable[[str], None]) -> None:
        self._on_note = on_note

    def is_available(self, ctx: ToolBuildContext) -> bool:
        return True

    def build(self, ctx: ToolBuildContext) -> MemoryTool:
        return MemoryTool(on_note=self._on_note)


class SessionNoteToolFactory:
    """Always available; saves session-level analysis context."""

    key = "session_note_tool"

    def __init__(self, on_note: Callable[[str], None]) -> None:
        self._on_note = on_note

    def is_available(self, ctx: ToolBuildContext) -> bool:
        return True

    def build(self, ctx: ToolBuildContext) -> SessionNoteTool:
        return SessionNoteTool(on_note=self._on_note)
