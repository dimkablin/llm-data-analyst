"""Planner tool — on-demand analysis plan generation.

The main agent calls this tool when it decides a multi-step plan is needed.
Internally calls LLM with a compact planning prompt + available tool list.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from backend.agent.llm_client import ThinkingAwareChatOpenAI

logger = logging.getLogger(__name__)

_PLANNER_SYSTEM_PROMPT = (
    "Ты — планировщик аналитического агента. Составь КРАТКИЙ план действий.\n\n"
    "## Формат плана\n"
    "Для каждого шага укажи: номер, tool, что получить.\n"
    "Пример: 1. `pandas_tool` → показать первые строки таблицы.\n\n"
    "## Правила\n"
    "- Используй ТОЛЬКО инструменты из списка доступных.\n"
    "- Минимум шагов: не добавляй лишние.\n"
    "- Для простых запросов (показать данные, структура) → 1 шаг.\n"
    "- Для графиков → обязательно `plotly_tool`.\n"
    "- Не путай `value_tool` (метрики из df) с `search_tool` (веб-поиск).\n"
)


class _Input(BaseModel):
    question: str = Field(
        description="Вопрос пользователя, для которого нужно составить план анализа."
    )


class PlannerTool(BaseTool):
    """Generates a structured analysis plan by calling LLM with a compact planning prompt."""

    name: str = "planner_tool"
    description: str = (
        "Составь план анализа перед любой задачей с данными (CSV, БД, статистика, графики, метрики, прогноз). "
        "Вызывай ПЕРВЫМ — до sql_tool, pandas_tool, plotly_tool и других инструментов данных. "
        "Исключение: тривиальные выборки ('покажи первые строки') и веб-поиск. "
        "Input: question (вопрос пользователя)."
    )
    args_schema: type[BaseModel] = _Input
    response_format: str = "content"

    _llm_model: str = PrivateAttr()
    _llm_base_url: str = PrivateAttr()
    _llm_api_key: str | None = PrivateAttr()
    _tool_descriptions: str = PrivateAttr()

    def __init__(
        self,
        *,
        llm_model: str,
        llm_base_url: str,
        llm_api_key: str | None = None,
        tool_descriptions: str = "",
    ) -> None:
        super().__init__()
        self._llm_model = llm_model
        self._llm_base_url = llm_base_url
        self._llm_api_key = llm_api_key
        self._tool_descriptions = tool_descriptions

    def _run(self, question: str) -> str:
        system_content = _PLANNER_SYSTEM_PROMPT
        if self._tool_descriptions:
            system_content += f"\n[ДОСТУПНЫЕ ИНСТРУМЕНТЫ]\n{self._tool_descriptions}\n"

        llm = ThinkingAwareChatOpenAI(
            model=self._llm_model,
            base_url=self._llm_base_url,
            api_key=self._llm_api_key,
            temperature=0.3,
            max_tokens=256,
            streaming=False,
        )

        try:
            response = llm.invoke([
                SystemMessage(content=system_content),
                HumanMessage(content=question),
            ])
            plan = str(getattr(response, "content", "")).strip()
            reasoning = response.additional_kwargs.get("reasoning", "")
            if not plan:
                plan = reasoning or "1. Выполнить запрос напрямую."
            return plan
        except Exception as exc:
            logger.warning("PlannerTool LLM call failed: %s", exc)
            return "Не удалось сгенерировать план. Выполни запрос напрямую без планирования."
