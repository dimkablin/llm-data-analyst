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
    "You are a planning specialist. You receive a user question and available tools, "
    "then produce a clear analysis plan.\n\n"
    "You must NOT execute anything. Only analyze and plan.\n\n"
    "## Goal\n"
    "One sentence summary of what needs to be done.\n\n"
    "## Plan\n"
    "Numbered steps, each small and actionable:\n"
    "1. Step one - which tool to use and what to get\n"
    "2. Step two - which tool to use and what to get\n"
    "3. ...\n\n"
    "## Tools to Use\n"
    "- `tool_name` - what for\n\n"
    "## Risks\n"
    "Anything to watch out for.\n\n"
    "Keep the plan concrete. The agent will execute it step by step.\n\n"
    "Rules:\n"
    "- Use ONLY tools from the available tools list.\n"
    "- Minimum steps: don't add unnecessary ones.\n"
    "- For simple requests (show data, structure) → 1 step.\n"
    "- For charts → always use `plotly_tool`.\n"
    "- Don't confuse `value_tool` (df metrics) with `search_tool` (web search).\n"
    "- IMPORTANT: Analytical skill names (auto_eda, cohort_analysis, ab_test_analysis, etc.) "
    "are NOT callable tools — they are method identifiers. "
    "When context mentions an analytical skill, the FIRST step in the plan MUST be: "
    "`get_tool_instructions(\"<skill_id>\")` — this fetches the step-by-step algorithm. "
    "Example: if context says auto_eda applies, plan step 1 = `get_tool_instructions(\"auto_eda\")`.\n"
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

        user_content = question
        if context:
            user_content = f"[Контекст предыдущих сообщений]\n{context}\n\n[Текущий запрос]\n{question}"

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
