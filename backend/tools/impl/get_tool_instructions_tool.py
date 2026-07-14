"""On-demand tool instruction loader with two-level retrieval (core / details).

The agent calls get_tool_instructions(skill_id) to receive core instructions
(API signatures, behavioral rules). For code examples and scenarios the agent
calls get_tool_instructions(skill_id, details=True).

Retrieval policy is sourced from tools/get_tool_instructions/TOOL.md so the
model sees the same markdown contract as the rest of the tool catalog.
"""
from __future__ import annotations

import logging
from typing import Any, ClassVar

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from backend.instructions import InstructionDocument
from backend.skills.models import SkillError
from backend.tools.instructions import (
    ToolInstructionRegistry,
    get_default_tool_instruction_registry,
    tool_description,
)

logger = logging.getLogger(__name__)

_EXTENDED_HINT_TEMPLATE = (
    "\n\n[Extended available: "
    "call get_tool_instructions('{skill_id}', details=True) "
    "for code scenarios, error patterns, and edge cases.]"
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
    description: str = tool_description("get_tool_instructions")
    args_schema: type[BaseModel] = _Input
    response_format: str = "content"
    parallel_safe: ClassVar[bool] = True

    _skill_registry: Any = PrivateAttr()
    _tool_instruction_registry: ToolInstructionRegistry = PrivateAttr()
    _allowed_skill_ids: set[str] | None = PrivateAttr(default=None)
    _allowed_tool_keys: set[str] | None = PrivateAttr(default=None)

    def __init__(
        self,
        skill_registry: Any,
        tool_instruction_registry: ToolInstructionRegistry | None = None,
        allowed_skill_ids: set[str] | None = None,
        allowed_tool_keys: set[str] | None = None,
    ) -> None:
        super().__init__()
        self._skill_registry = skill_registry
        self._tool_instruction_registry = (
            tool_instruction_registry or get_default_tool_instruction_registry()
        )

        self._allowed_skill_ids = (
            {str(skill_id).strip() for skill_id in allowed_skill_ids if str(skill_id).strip()}
            if allowed_skill_ids is not None
            else None
        )

        self._allowed_tool_keys = (
            {str(tool_key).strip() for tool_key in allowed_tool_keys if str(tool_key).strip()}
            if allowed_tool_keys is not None
            else None
        )

    def _run(self, skill_id: str, details: bool = False) -> str:
        normalized_skill_id = str(skill_id).strip()

        tool_document = self._tool_instruction_registry.get_optional(normalized_skill_id)
        if tool_document is not None:
            return self._tool_response(tool_document, details=details)

        try:
            skill = self._skill_registry.get(normalized_skill_id)
        except SkillError:
            return self._not_found_response(normalized_skill_id)
        except Exception:
            logger.exception("Unexpected error looking up skill '%s'", skill_id)
            return self._not_found_response(normalized_skill_id)

        if skill.kind == "analytical":
            if (
                self._allowed_skill_ids is not None
                and normalized_skill_id not in self._allowed_skill_ids
            ):
                return f"Скил '{normalized_skill_id}' отключен для этого пользователя."

        elif skill.kind == "tool":
            tool_key = str(skill.tool_key or skill.skill_id).strip()

            if (
                self._allowed_tool_keys is not None
                and tool_key not in self._allowed_tool_keys
                and normalized_skill_id not in self._allowed_tool_keys
            ):
                return f"Инструкции для tool '{normalized_skill_id}' недоступны: сам tool отключен."

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

    def _tool_response(self, document: InstructionDocument, *, details: bool) -> str:
        tool_key = str(document.metadata.tool_key or document.metadata.id).strip()
        if (
            self._allowed_tool_keys is not None
            and tool_key not in self._allowed_tool_keys
            and document.metadata.id not in self._allowed_tool_keys
        ):
            return f"Инструкции для tool '{tool_key}' недоступны: сам tool отключен."

        if details:
            if not document.has_details:
                return (
                    f"Extended instructions not available for '{tool_key}'. "
                    f"Core instructions were already provided."
                )
            return document.details_markdown  # type: ignore[return-value]

        content = document.body
        if document.has_details:
            content += _EXTENDED_HINT_TEMPLATE.format(skill_id=tool_key)
        return content

    def _not_found_response(self, skill_id: str) -> str:
        all_skills = self._skill_registry.list_skills()
        tool_ids = sorted(
            str(document.metadata.tool_key or document.metadata.id)
            for document in self._tool_instruction_registry.list_tools()
        )
        analytical_ids = sorted(s.skill_id for s in all_skills if s.kind == "analytical")
        parts = []
        if tool_ids:
            parts.append(f"инструменты: {', '.join(tool_ids)}")
        if analytical_ids:
            parts.append(f"аналитические методы: {', '.join(analytical_ids)}")
        available_str = "; ".join(parts) or "нет"
        return f"Скил '{skill_id}' не найден. Доступные: {available_str}."
