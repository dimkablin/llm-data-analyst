"""Tool factories — one class per LangChain tool.

Each factory encapsulates three concerns:
  1. Whether a tool *can* be used given the current context (service live + user allowed + data present).
  2. How to *construct* the tool when it can.
  3. The string key under which the tool is registered.

``ToolFactory`` is a structural ``Protocol``, so factories do not need to inherit from
a common base class — duck typing is enough.
"""
from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from agent.tools.anomaly_planfact_tool import AnomalyPlanfactTool
from agent.tools.base_tool import BaseExecTool
from agent.tools.db_tool import DBTool
from agent.tools.forecast_tool import ForecastTool
from agent.tools.memory_tool import MemoryTool
from agent.tools.pandas_tool import PandasTool
from agent.tools.plotly_tool import PlotlyTool
from agent.tools.search_tool import SearchTool
from agent.tools.value_tool import ValueTool
from backend.anomaly_planfact_integration import AnomalyPlanfactIntegrationService
from backend.forecast_integration import ForecastIntegrationService
from backend.search_integration import SearchIntegrationService
from backend.tool_context import ToolBuildContext
from backend.tool_policy import is_tool_allowed


@runtime_checkable
class ToolFactory(Protocol):
    """Structural protocol every concrete factory must satisfy."""

    key: str

    def is_available(self, ctx: ToolBuildContext) -> bool: ...

    def build(self, ctx: ToolBuildContext) -> BaseExecTool: ...


# ── Integration-backed factories ──────────────────────────────────────────────


class SearchToolFactory:
    key = "search_tool"

    def __init__(self, service: SearchIntegrationService) -> None:
        self._service = service

    def is_available(self, ctx: ToolBuildContext) -> bool:
        return self._service.is_enabled and is_tool_allowed(self.key, ctx.allowed_tool_keys)

    def build(self, ctx: ToolBuildContext) -> SearchTool:
        return SearchTool(
            ctx.tool_df,
            search_service=self._service,
            execution_timeout_sec=ctx.settings.tool_exec_timeout_sec,
            tool_cache_size=max(8, ctx.settings.tool_cache_size // 2),
        )


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
        )


# ── Built-in factories ────────────────────────────────────────────────────────


class PlotlyToolFactory:
    key = "plotly_tool"

    def is_available(self, ctx: ToolBuildContext) -> bool:
        return ctx.has_data and is_tool_allowed(self.key, ctx.allowed_tool_keys)

    def build(self, ctx: ToolBuildContext) -> PlotlyTool:
        # Plotly needs extra time: module reimport in subprocess + chart rendering
        plotly_timeout = max(ctx.settings.tool_exec_timeout_sec * 2, 60.0)
        return PlotlyTool(
            ctx.tool_df,
            execution_timeout_sec=plotly_timeout,
            tool_cache_size=ctx.settings.tool_cache_size,
            db_runtime_config=ctx.tool_db_runtime,
        )


class PandasToolFactory:
    key = "pandas_tool"

    def is_available(self, ctx: ToolBuildContext) -> bool:
        return ctx.df is not None and is_tool_allowed(self.key, ctx.allowed_tool_keys)

    def build(self, ctx: ToolBuildContext) -> PandasTool:
        return PandasTool(
            ctx.tool_df,
            execution_timeout_sec=ctx.settings.tool_exec_timeout_sec,
            tool_cache_size=ctx.settings.tool_cache_size,
            db_runtime_config=ctx.tool_db_runtime,
        )


class ValueToolFactory:
    key = "value_tool"

    def is_available(self, ctx: ToolBuildContext) -> bool:
        return ctx.df is not None and is_tool_allowed(self.key, ctx.allowed_tool_keys)

    def build(self, ctx: ToolBuildContext) -> ValueTool:
        return ValueTool(
            ctx.tool_df,
            execution_timeout_sec=ctx.settings.tool_exec_timeout_sec,
            tool_cache_size=ctx.settings.tool_cache_size,
            db_runtime_config=ctx.tool_db_runtime,
        )


class DBToolFactory:
    key = "db_tool"

    def is_available(self, ctx: ToolBuildContext) -> bool:
        return ctx.tool_db_runtime is not None and is_tool_allowed(self.key, ctx.allowed_tool_keys)

    def build(self, ctx: ToolBuildContext) -> DBTool:
        return DBTool(
            ctx.tool_df,
            execution_timeout_sec=ctx.settings.tool_exec_timeout_sec,
            tool_cache_size=max(8, ctx.settings.tool_cache_size // 2),
            db_runtime_config=ctx.tool_db_runtime,
        )


class MemoryToolFactory:
    """Always available; needs only an `on_note` callback from the runner."""

    key = "memory_tool"

    def __init__(self, on_note: Callable[[str], None]) -> None:
        self._on_note = on_note

    def is_available(self, ctx: ToolBuildContext) -> bool:  # noqa: ARG002
        return True

    def build(self, ctx: ToolBuildContext) -> MemoryTool:  # noqa: ARG002
        return MemoryTool(on_note=self._on_note)
