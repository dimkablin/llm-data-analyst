"""On-demand tool instruction loader — claude-code ToolSearch pattern.

The agent calls ``get_tool_instructions(tool_name)`` before using a complex
tool to receive its full ``.md`` skill file: variables in scope, code patterns,
examples, and rules.  This keeps the base execution prompt compact while
giving the LLM full context exactly when it needs it.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from backend.skills.models import SkillError

logger = logging.getLogger(__name__)


class _Input(BaseModel):
    tool_name: str = Field(
        description=(
            "ID of the tool or analytical skill to fetch instructions for. "
            "For tools: 'plotly_tool', 'sql_tool', 'pandas_tool', etc. "
            "For analytical methods: 'auto_eda', 'cohort_analysis', 'ab_test_analysis', etc."
        )
    )


class GetToolInstructionsTool(BaseTool):
    """Returns the full skill markdown for *tool_name* from the SkillRegistry."""

    name: str = "get_tool_instructions"
    description: str = (
        "Get full usage instructions for a tool or analytical method before using it. "
        "Returns scope variables, code patterns, rules, and step-by-step algorithms. "
        "MUST be called before using any tool (plotly_tool, sql_tool, pandas_tool, etc.) "
        "AND before executing any analytical method listed in 'Аналитические методы' section "
        "(auto_eda, cohort_analysis, ab_test_analysis, statistical_analysis, etc.)."
    )
    args_schema: type[BaseModel] = _Input
    response_format: str = "content"

    _skill_registry: Any = PrivateAttr()

    def __init__(self, skill_registry: Any) -> None:
        super().__init__()
        self._skill_registry = skill_registry

    _EXECUTION_PREAMBLE = (
        "ВЫПОЛНИ ВСЕ ШАГИ НИЖЕ ЧЕРЕЗ TOOL CALLS. Не пиши текст — только вызовы tools.\n\n"
    )

    def _run(self, tool_name: str) -> str:
        try:
            skill = self._skill_registry.get(str(tool_name).strip())
            return self._EXECUTION_PREAMBLE + skill.instructions_markdown
        except SkillError:
            pass
        except Exception:
            logger.exception("Unexpected error looking up skill '%s'", tool_name)
        all_skills = self._skill_registry.list_skills()
        tool_ids = sorted(s.skill_id for s in all_skills if s.kind == "tool")
        analytical_ids = sorted(s.skill_id for s in all_skills if s.kind == "analytical")
        parts = []
        if tool_ids:
            parts.append(f"tools: {', '.join(tool_ids)}")
        if analytical_ids:
            parts.append(f"analytical methods: {', '.join(analytical_ids)}")
        available_str = "; ".join(parts) or "none"
        return (
            f"Unknown skill '{tool_name}'. "
            f"Available: {available_str}."
        )
