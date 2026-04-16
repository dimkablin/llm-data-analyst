"""On-demand tool instruction loader with two-level retrieval (core / details).

The agent calls get_tool_instructions(skill_id) to receive core instructions
(API signatures, behavioral rules). For code examples and scenarios the agent
calls get_tool_instructions(skill_id, details=True).

Retrieval policy is encoded in the tool description so the model follows
deterministic rules rather than guessing.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from backend.skills.models import SkillError

logger = logging.getLogger(__name__)

_EXTENDED_HINT_TEMPLATE = (
    "\n\n[Extended available: "
    "call get_tool_instructions('{skill_id}', details=True) "
    "for code scenarios, error patterns, and edge cases.]"
)

_RETRIEVAL_POLICY = (
    "Загрузи инструкции по инструменту или аналитическому методу.\n"
    "\n"
    "ПРАВИЛА ЗАГРУЗКИ:\n"
    "1. Перед первым использованием незнакомого или нетривиального инструмента"
    " → вызови без details (по умолчанию).\n"
    "2. Перед сложными сценариями (несколько графиков, DB-режим, JOIN-цепочки, многошаговый анализ) "
    "→ вызови с details=True.\n"
    "3. После сбоя tool-вызова → вызови с details=True перед повтором.\n"
    "4. Никогда не вызывай get_tool_instructions для одного и того же skill_id + details дважды за сессию.\n"
    "\n"
    "details=False (по умолчанию): API-сигнатуры, правила поведения, контракт.\n"
    "details=True: примеры кода, паттерны ошибок, граничные случаи (только по запросу)."
)


class _Input(BaseModel):
    skill_id: str = Field(
        description=(
            "ID инструмента или аналитического скила. "
            "Инструменты: 'plotly_tool', 'sql_tool', 'pandas_tool', 'database_tool' и др. "
            "Аналитические методы: 'auto_eda', 'cohort_analysis', 'ab_test_analysis' и др."
        )
    )
    details: bool = Field(
        default=False,
        description=(
            "False (по умолчанию) — вернуть core: API-сигнатуры, правила поведения, контракт. "
            "True — вернуть DETAILS.md: примеры кода, паттерны ошибок, граничные случаи."
        ),
    )


class GetToolInstructionsTool(BaseTool):
    """Returns skill instructions at the requested detail level."""

    name: str = "get_tool_instructions"
    description: str = _RETRIEVAL_POLICY
    args_schema: type[BaseModel] = _Input
    response_format: str = "content"

    _skill_registry: Any = PrivateAttr()

    def __init__(self, skill_registry: Any) -> None:
        super().__init__()
        self._skill_registry = skill_registry

    def _run(self, skill_id: str, details: bool = False) -> str:
        try:
            skill = self._skill_registry.get(str(skill_id).strip())
        except SkillError:
            return self._not_found_response(str(skill_id).strip())
        except Exception:
            logger.exception("Unexpected error looking up skill '%s'", skill_id)
            return self._not_found_response(str(skill_id).strip())

        if details:
            if not skill.has_details:
                return (
                    f"Extended instructions not available for '{skill.skill_id}'. "
                    f"Core instructions were already provided."
                )
            return skill.details_markdown  # type: ignore[return-value]

        # Core — inject hint if details are available
        content = skill.core_markdown
        if skill.has_details:
            content += _EXTENDED_HINT_TEMPLATE.format(skill_id=skill.skill_id)
        return content

    def _not_found_response(self, skill_id: str) -> str:
        all_skills = self._skill_registry.list_skills()
        tool_ids = sorted(s.skill_id for s in all_skills if s.kind == "tool")
        analytical_ids = sorted(s.skill_id for s in all_skills if s.kind == "analytical")
        parts = []
        if tool_ids:
            parts.append(f"инструменты: {', '.join(tool_ids)}")
        if analytical_ids:
            parts.append(f"аналитические методы: {', '.join(analytical_ids)}")
        available_str = "; ".join(parts) or "нет"
        return f"Скил '{skill_id}' не найден. Доступные: {available_str}."
