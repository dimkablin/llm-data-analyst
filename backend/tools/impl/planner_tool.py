"""Planner tool — on-demand analysis plan generation.

The main agent calls this tool when it decides a multi-step plan is needed.
Internally calls LLM with a compact planning prompt + available tool list.
"""
from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from backend.agent.llm_client import ReasoningChatOpenAI
from backend.core.llm_provider import get_provider_policy

logger = logging.getLogger(__name__)

_PLANNER_SYSTEM_PROMPT = (
    "Ты — специалист по планированию анализа. Получаешь вопрос пользователя и список доступных инструментов, "
    "затем составляешь чёткий план анализа.\n\n"
    "НЕ выполняй ничего. Только анализируй и планируй.\n\n"
    "## Цель\n"
    "Одно предложение — суть задачи.\n\n"
    "## План\n"
    "Нумерованные шаги, каждый — конкретное действие:\n"
    "1. Шаг один — какой инструмент использовать и что получить\n"
    "2. Шаг два — какой инструмент использовать и что получить\n"
    "3. ...\n\n"
    "## Используемые инструменты\n"
    "- `tool_name` — для чего\n\n"
    "## Риски\n"
    "На что обратить внимание.\n\n"
    "План должен быть конкретным. Агент выполнит его шаг за шагом.\n\n"
    "Правила:\n"
    "- Используй ТОЛЬКО инструменты из списка доступных.\n"
    "- Минимум шагов: не добавляй лишних.\n"
    "- Для простых запросов (показать данные, структуру) → 1 шаг.\n"
    "- Для графиков → всегда используй `plotly_tool`.\n"
    "- Не путай `value_tool` (метрики датафрейма) и `search_tool` (веб-поиск).\n"
    "- ВАЖНО: названия аналитических скилов (auto_eda, cohort_analysis, ab_test_analysis и др.) "
    "— это НЕ callable инструменты, а идентификаторы методов. "
    "Если контекст упоминает аналитический скил, ПЕРВЫЙ шаг плана ОБЯЗАН быть: "
    "`get_tool_instructions(\"<skill_id>\")` — это загружает пошаговый алгоритм. "
    "Пример: если применим auto_eda, шаг 1 плана = `get_tool_instructions(\"auto_eda\")`.\n"
)


class _Input(BaseModel):
    question: str = Field(
        description="Вопрос пользователя, для которого нужно составить план анализа."
    )
    context: str = Field(
        default="",
        description="Краткий контекст последних сообщений чата (опционально).",
    )


class PlannerTool(BaseTool):
    """Generates a structured analysis plan by calling LLM with a compact planning prompt."""

    name: str = "planner_tool"
    description: str = (
        "Составь план анализа перед любой задачей с данными "
        "(CSV, БД, статистика, графики, метрики, прогноз). "
        "Вызывай ПЕРВЫМ — до sql_tool, pandas_tool, plotly_tool и других инструментов данных. "
        "Исключение: тривиальные выборки ('покажи первые строки') и веб-поиск. "
        "Input: question (вопрос пользователя)."
    )
    args_schema: type[BaseModel] = _Input
    response_format: str = "content"

    _llm_model: str = PrivateAttr()
    _llm_base_url: str = PrivateAttr()
    _llm_api_key: str | None = PrivateAttr()
    _llm_provider: str | None = PrivateAttr()
    _tool_descriptions: str = PrivateAttr()

    def __init__(
        self,
        *,
        llm_model: str,
        llm_base_url: str,
        llm_api_key: str | None = None,
        llm_provider: str | None = None,
        tool_descriptions: str = "",
    ) -> None:
        super().__init__()
        self._llm_model = llm_model
        self._llm_base_url = llm_base_url
        self._llm_api_key = llm_api_key
        self._llm_provider = llm_provider
        self._tool_descriptions = tool_descriptions

    def set_tool_descriptions(self, descriptions: str) -> None:
        """Inject available tool descriptions after the tool registry is built."""
        self._tool_descriptions = descriptions

    def _run(self, question: str, context: str = "") -> str:
        system_content = _PLANNER_SYSTEM_PROMPT
        if self._tool_descriptions:
            system_content += f"\n[ДОСТУПНЫЕ ИНСТРУМЕНТЫ]\n{self._tool_descriptions}\n"

        extra_body = get_provider_policy(self._llm_provider).build_extra_body(
            enable_thinking=False
        )
        llm = ReasoningChatOpenAI(
            model=self._llm_model,
            base_url=self._llm_base_url,
            api_key=self._llm_api_key,
            temperature=0.3,
            max_tokens=1024,
            streaming=False,
            **({"extra_body": extra_body} if extra_body else {}),
        )

        no_think_prefix = get_provider_policy(self._llm_provider).get_thinking_message_prefix(
            enable_thinking=False
        )
        user_content = question
        if context:
            user_content = f"[Контекст предыдущих сообщений]\n{context}\n\n[Текущий запрос]\n{question}"
        if no_think_prefix:
            user_content = f"{no_think_prefix}{user_content}"

        try:
            response = llm.invoke([
                SystemMessage(content=system_content),
                HumanMessage(content=user_content),
            ])
            plan = str(getattr(response, "content", "")).strip()
            reasoning = response.additional_kwargs.get("reasoning", "")
            if not plan:
                plan = reasoning or "1. Выполнить запрос напрямую."
            return plan
        except Exception as exc:
            logger.warning("PlannerTool LLM call failed: %s", exc)
            return "Не удалось сгенерировать план. Выполни запрос напрямую без планирования."
