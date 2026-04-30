from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.agent_graph.runtime import GraphRuntimeContext
from backend.agent_graph.services import AgentRuntimeServices
from backend.core.config import DEPTH_PROFILES
from backend.data_access.db_runtime_service import RuntimeDBConnectionConfig
from backend.tools.capabilities import build_runtime_capability_context
from backend.tools.context import ToolBuildContext
from backend.tools.sandbox_manager import SandboxManager


@dataclass(slots=True)
class AnalysisContext:
    tools: list[Any]
    sandbox: Any | None
    tool_db_runtime: RuntimeDBConnectionConfig | None
    capability_context: dict[str, Any]
    max_steps: int

    @property
    def available_tool_keys(self) -> list[str]:
        return [
            str(getattr(tool, "name", "")).strip()
            for tool in self.tools
            if str(getattr(tool, "name", "")).strip()
        ]


@dataclass(slots=True)
class AnalysisContextBuilder:
    """Build analysis-only runtime dependencies for the graph."""

    hidden_from_agent: frozenset[str] = frozenset({"planner_tool", "review_tool"})

    def build(
        self,
        *,
        services: AgentRuntimeServices,
        runtime_context: GraphRuntimeContext,
        state: dict[str, Any],
    ) -> AnalysisContext:
        trace_context = dict(state.get("trace_context") or {})
        session_source = dict(state.get("session_source") or {})
        df = runtime_context.df

        tool_db_runtime = self._resolve_tool_db_runtime_config(
            services=services,
            session_source=session_source,
            trace_context=trace_context,
        )
        csv_loaded, csv_session_id = self._resolve_csv_runtime_state(
            session_source=session_source,
            trace_context=trace_context,
        )
        csv_duckdb_mode = bool(csv_loaded and str(csv_session_id or "").strip())
        tool_df = None if csv_duckdb_mode else df

        sandbox = runtime_context.sandbox or self._build_sandbox(
            services=services,
            trace_context=trace_context,
            df=df,
            tool_db_runtime=tool_db_runtime,
        )

        tool_context = ToolBuildContext(
            settings=services.settings,
            allowed_tool_keys=services.allowed_tool_keys,
            df=tool_df,
            tool_db_runtime=tool_db_runtime,
            csv_loaded=csv_loaded,
            csv_session_id=csv_session_id,
            sandbox=sandbox,
        )
        if runtime_context.tools:
            tools = list(runtime_context.tools)
            tool_descriptions = "\n".join(f"- `{name}`" for name in self._tool_names(tools))
        else:
            tools = services.tool_registry.build_tools(tool_context)
            tool_descriptions = services.tool_registry.describe_available_tools(tool_context)

        planner_descriptions = self._planner_descriptions(
            services=services,
            tool_descriptions=tool_descriptions,
        )
        for tool in tools:
            if hasattr(tool, "set_tool_descriptions"):
                tool.set_tool_descriptions(planner_descriptions)

        visible_tool_keys = [
            key for key in self._tool_names(tools) if key not in self.hidden_from_agent
        ]
        capability_context = build_runtime_capability_context(
            available_tool_keys=visible_tool_keys,
            has_dataframe=tool_df is not None,
            has_db_source=(tool_db_runtime is not None) or csv_duckdb_mode,
            csv_table_names=list(session_source.get("csv_table_names") or []) or None,
        )
        capability_context["tool_descriptions"] = self._visible_tool_descriptions(
            tool_descriptions,
        )

        depth = services.settings.agent_analysis_depth
        depth_profile = DEPTH_PROFILES.get(depth, DEPTH_PROFILES["light"])
        depth_inner_limit = depth_profile.get("inner_recursion_limit")
        max_steps = max(
            1,
            depth_inner_limit
            if isinstance(depth_inner_limit, int)
            else services.settings.agent_inner_recursion_limit,
        )

        runtime_context.tools = tools
        runtime_context.sandbox = sandbox
        runtime_context.tool_db_runtime = tool_db_runtime

        return AnalysisContext(
            tools=tools,
            sandbox=sandbox,
            tool_db_runtime=tool_db_runtime,
            capability_context=capability_context,
            max_steps=max_steps,
        )

    def _build_sandbox(
        self,
        *,
        services: AgentRuntimeServices,
        trace_context: dict[str, Any],
        df: Any | None,
        tool_db_runtime: RuntimeDBConnectionConfig | None,
    ) -> Any:
        session_id = str(trace_context.get("session_id") or "default")
        sandbox = SandboxManager.get_instance().get_or_create(session_id)
        sandbox.ensure_storage_dir(Path(services.settings.storage_dir) / session_id)
        if df is not None:
            source_label = str(trace_context.get("dataset_name") or "")
            sandbox.bind_dataframe(
                df,
                source_label=source_label,
                db_runtime_config=tool_db_runtime,
            )
        return sandbox

    def _planner_descriptions(
        self,
        *,
        services: AgentRuntimeServices,
        tool_descriptions: str,
    ) -> str:
        lines = [
            line
            for line in tool_descriptions.splitlines()
            if "planner_tool" not in line
        ]
        analytical_block = services.skill_registry.build_analytical_skills_brief_block(
            enabled_skill_ids=services.enabled_analytical_skill_ids,
        )
        if analytical_block:
            lines.extend(["", analytical_block])
        return "\n".join(lines).strip()

    def _visible_tool_descriptions(self, tool_descriptions: str) -> str:
        return "\n".join(
            line
            for line in tool_descriptions.splitlines()
            if not any(f"`{key}`" in line for key in self.hidden_from_agent)
        ).strip()

    @staticmethod
    def _tool_names(tools: list[Any]) -> list[str]:
        return [
            str(getattr(tool, "name", "")).strip()
            for tool in tools
            if str(getattr(tool, "name", "")).strip()
        ]

    @staticmethod
    def _extract_db_connection_id(
        *,
        session_source: dict[str, Any],
        trace_context: dict[str, Any],
    ) -> str | None:
        source_type = str(session_source.get("source_type") or "").strip().lower()
        source_ref_id = session_source.get("source_ref_id")
        if source_type == "db_connection" and isinstance(source_ref_id, str) and source_ref_id.strip():
            return source_ref_id.strip()

        for key in ("db_connection_id", "connection_id"):
            value = trace_context.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _resolve_tool_db_runtime_config(
        self,
        *,
        services: AgentRuntimeServices,
        session_source: dict[str, Any],
        trace_context: dict[str, Any],
    ) -> RuntimeDBConnectionConfig | None:
        connection_id = self._extract_db_connection_id(
            session_source=session_source,
            trace_context=trace_context,
        )
        if not connection_id:
            return None
        if services.db_runtime_service is None:
            raise RuntimeError("DB runtime service is not configured.")

        user_id_raw = trace_context.get("user_id")
        try:
            user_id = int(user_id_raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "trace_context.user_id is required for DB tool runtime."
            ) from exc

        return services.db_runtime_service.get_runtime_config(
            user_id=user_id,
            connection_id=connection_id,
        )

    @staticmethod
    def _resolve_csv_runtime_state(
        *,
        session_source: dict[str, Any],
        trace_context: dict[str, Any],
    ) -> tuple[bool, str | None]:
        direct_loaded = bool(session_source.get("csv_loaded"))
        direct_sid = session_source.get("csv_session_id")
        if direct_loaded and isinstance(direct_sid, str) and direct_sid.strip():
            return True, direct_sid.strip()

        source_type = str(session_source.get("source_type") or "").strip().lower()
        if source_type == "csv" and isinstance(direct_sid, str) and direct_sid.strip():
            return True, direct_sid.strip()

        if bool(trace_context.get("csv_duckdb_loaded")):
            sid = trace_context.get("csv_session_id") or trace_context.get("session_id")
            if isinstance(sid, str) and sid.strip():
                return True, sid.strip()

        return False, None
