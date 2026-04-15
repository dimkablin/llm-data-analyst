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
            "ID инструмента или аналитического скила для загрузки инструкций. "
            "Инструменты: 'plotly_tool', 'sql_tool', 'pandas_tool' и др. "
            "Аналитические методы: 'auto_eda', 'cohort_analysis', 'ab_test_analysis' и др."
        )
    )


class GetToolInstructionsTool(BaseTool):
    """Returns the full skill markdown for *tool_name* from the SkillRegistry."""

    name: str = "get_tool_instructions"
    description: str = (
        "Загрузи полные инструкции по использованию инструмента или аналитического метода. "
        "Возвращает переменные scope, паттерны кода, правила и пошаговые алгоритмы. "
        "Вызывай для сложных инструментов (plotly_tool, sql_tool, forecast_tool и др.) "
        "и аналитических методов (auto_eda, cohort_analysis, ab_test_analysis и др.), "
        "когда нужен полный контракт перед выполнением."
    )
    args_schema: type[BaseModel] = _Input
    response_format: str = "content"

    _skill_registry: Any = PrivateAttr()

    def __init__(self, skill_registry: Any) -> None:
        super().__init__()
        self._skill_registry = skill_registry

    def _run(self, tool_name: str) -> str:
        try:
            skill = self._skill_registry.get(str(tool_name).strip())
            return skill.instructions_markdown
        except SkillError:
            pass
        except Exception:
            logger.exception("Unexpected error looking up skill '%s'", tool_name)
        all_skills = self._skill_registry.list_skills()
        tool_ids = sorted(s.skill_id for s in all_skills if s.kind == "tool")
        analytical_ids = sorted(s.skill_id for s in all_skills if s.kind == "analytical")
        parts = []
        if tool_ids:
            parts.append(f"инструменты: {', '.join(tool_ids)}")
        if analytical_ids:
            parts.append(f"аналитические методы: {', '.join(analytical_ids)}")
        available_str = "; ".join(parts) or "нет"
        return (
            f"Скил '{tool_name}' не найден. "
            f"Доступные: {available_str}."
        )
