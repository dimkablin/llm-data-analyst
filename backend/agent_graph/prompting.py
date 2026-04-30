from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from backend.agent.prompts import execution_agent_prompt, get_detailed_data_info
from backend.agent_graph.services import AgentRuntimeServices


@dataclass(slots=True)
class ExecutionPromptBuilder:
    """Builds the system prompt for analysis/tool-calling graph nodes."""

    services: AgentRuntimeServices

    def build(
        self,
        *,
        capability_context: dict[str, Any],
        sandbox: Any | None,
        selected_skill_ids: list[str],
        df: Any | None,
        session_source: dict[str, Any],
        tool_db_runtime: Any | None,
        user_prompt: str | None,
    ) -> str:
        available_tools = [
            str(item).strip()
            for item in capability_context.get("available_tool_keys", [])
            if str(item).strip()
        ]
        tool_descriptions = str(capability_context.get("tool_descriptions") or "").strip()
        source_mode = str(capability_context.get("source_mode") or "dataset").strip()

        sections: list[str] = [execution_agent_prompt.strip()]
        sections.append(
            self._runtime_section(
                source_mode=source_mode,
                available_tools=available_tools,
                tool_descriptions=tool_descriptions,
            ),
        )

        prompt_block = str(capability_context.get("prompt_block") or "").strip()
        if prompt_block:
            sections.append(prompt_block)

        if sandbox is not None and hasattr(sandbox, "describe_for_prompt"):
            sandbox_block = str(sandbox.describe_for_prompt() or "").strip()
            if sandbox_block:
                sections.append(sandbox_block)

        tool_skills_block = self.services.skill_registry.build_tool_skills_brief_block(
            set(available_tools),
        )
        if tool_skills_block:
            sections.append(tool_skills_block)

        analytical_skills_block = self.services.skill_registry.build_analytical_skills_brief_block(
            enabled_skill_ids=self.services.enabled_analytical_skill_ids,
            user_prompt=user_prompt,
        )
        if analytical_skills_block:
            sections.append(analytical_skills_block)

        selected_skills_block = self._selected_skills_block(selected_skill_ids)
        if selected_skills_block:
            sections.append(selected_skills_block)

        data_context = self._data_context(
            df=df,
            session_source=session_source,
            tool_db_runtime=tool_db_runtime,
        )
        if data_context:
            sections.append(data_context)

        return "\n\n".join(section for section in sections if section).strip()

    def _runtime_section(
        self,
        *,
        source_mode: str,
        available_tools: list[str],
        tool_descriptions: str,
    ) -> str:
        tool_list = ", ".join(f"`{item}`" for item in available_tools) if available_tools else "none"
        lines = [
            f"Today: {date.today().strftime('%Y-%m-%d')}.",
            f"Data mode: `{source_mode}`.",
            f"Available tools in this run: {tool_list}.",
        ]
        if tool_descriptions:
            lines.extend(["Available tool descriptions:", tool_descriptions])
        return "\n".join(lines)

    def _selected_skills_block(self, selected_skill_ids: list[str]) -> str:
        if not selected_skill_ids:
            return ""
        allowed = self.services.enabled_analytical_skill_ids
        filtered = (
            [skill_id for skill_id in selected_skill_ids if skill_id in allowed]
            if allowed is not None
            else selected_skill_ids
        )
        if not filtered:
            return ""
        return self.services.skill_registry.build_prompt_block(filtered)

    def _data_context(
        self,
        *,
        df: Any | None,
        session_source: dict[str, Any],
        tool_db_runtime: Any | None,
    ) -> str:
        parts: list[str] = []
        if df is not None:
            try:
                parts.append(
                    get_detailed_data_info(
                        df,
                        max_columns=self.services.settings.agent_prompt_max_columns,
                    ),
                )
            except Exception:
                shape = getattr(df, "shape", None)
                if isinstance(shape, tuple) and len(shape) == 2:
                    parts.append(f"Dataset: {shape[0]} rows, {shape[1]} columns.")

        db_block = self._db_session_prompt_block(
            session_source=session_source,
            runtime=tool_db_runtime,
        )
        if db_block:
            parts.append(db_block)
        return "\n\n".join(parts).strip()

    @staticmethod
    def _db_session_prompt_block(
        *,
        session_source: dict[str, Any],
        runtime: Any | None,
    ) -> str:
        if runtime is None:
            return ""

        lines = [
            "Data source: database connection.",
            f"Connection name: {getattr(runtime, 'name', '')}",
            f"DB type: {getattr(runtime, 'db_type', '')}",
            f"Connection id: {getattr(runtime, 'connection_id', '')}",
        ]
        database = getattr(runtime, "database", "")
        if database:
            lines.append(f"Database/catalog: {database}")
        label = str(session_source.get("source_label") or "").strip()
        if label:
            lines.append(f"Session source label: {label}")
        lines.append(
            "For primary database table retrieval use sql_tool with both "
            "`question` and `artifact_name` arguments.",
        )
        return "\n".join(line for line in lines if line)
