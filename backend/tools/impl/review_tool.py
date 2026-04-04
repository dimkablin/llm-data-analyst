"""Review tool — hybrid quality check for agent answers.

Phase 1: fast heuristic checks (0ms).
Phase 2: LLM-based review for complex queries (>2 tool calls).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from backend.agent.llm_client import ThinkingAwareChatOpenAI

logger = logging.getLogger(__name__)

_PLOT_KEYWORDS = ("график", "диаграмм", "plot", "chart", "визуализ", "гистограмм", "scatter")

_EVALUATE_SYSTEM_PROMPT = (
    "Ты — ревьюер качества ответов аналитического агента.\n"
    "Оцени ответ строго в формате JSON:\n"
    '{"pass": true/false, "reason": "краткое обоснование"}\n\n'
    "pass=true если:\n"
    "- Ответ прямо и по существу отвечает на вопрос\n"
    "- Для аналитических вопросов ответ подкреплён артефактами или числами\n"
    "- Если данных недостаточно — честное объяснение допустимо\n\n"
    "pass=false только если:\n"
    "- Ответ пустой или полностью нерелевантен вопросу\n"
    "- Содержит только технические ошибки\n"
    "- LLM придумала данные без инструмента\n\n"
    "Только JSON, без пояснений."
)


class _Input(BaseModel):
    question: str = Field(description="Вопрос пользователя")
    answer: str = Field(description="Предлагаемый ответ агента")
    tool_calls_count: int = Field(default=0, description="Количество вызванных инструментов")
    artifact_count: int = Field(default=0, description="Количество полученных артефактов")


class ReviewTool(BaseTool):
    """Hybrid review: fast heuristics + optional LLM check for complex queries."""

    name: str = "review_tool"
    description: str = (
        "Проверь качество своего ответа перед финализацией. "
        "Вызывай после сложного анализа (3+ tool calls). "
        "Input: question, answer, tool_calls_count, artifact_count."
    )
    args_schema: type[BaseModel] = _Input
    response_format: str = "content"

    _llm_model: str = PrivateAttr()
    _llm_base_url: str = PrivateAttr()
    _llm_api_key: str | None = PrivateAttr()

    def __init__(
        self,
        *,
        llm_model: str,
        llm_base_url: str,
        llm_api_key: str | None = None,
    ) -> None:
        super().__init__()
        self._llm_model = llm_model
        self._llm_base_url = llm_base_url
        self._llm_api_key = llm_api_key

    def _run(
        self,
        question: str,
        answer: str,
        tool_calls_count: int = 0,
        artifact_count: int = 0,
    ) -> str:
        # ── Phase 1: fast heuristic checks ──────────────────────────────
        issues: list[str] = []

        if not answer.strip():
            issues.append("Ответ пустой")

        if len(answer.strip()) < 20 and artifact_count == 0:
            issues.append("Ответ слишком короткий без артефактов")

        q_lower = question.strip().lower()
        needs_plot = any(kw in q_lower for kw in _PLOT_KEYWORDS)
        if needs_plot and artifact_count == 0:
            issues.append("Запрос требует визуализацию, но нет графика")

        if issues:
            return json.dumps({"pass": False, "issues": issues}, ensure_ascii=False)

        # ── Phase 2: LLM review for complex queries ─────────────────────
        if tool_calls_count > 2:
            try:
                llm = ThinkingAwareChatOpenAI(
                    model=self._llm_model,
                    base_url=self._llm_base_url,
                    api_key=self._llm_api_key,
                    temperature=0.1,
                    max_tokens=300,
                    streaming=False,
                )
                eval_prompt = (
                    f"Вопрос пользователя: {question}\n"
                    f"Ответ агента: {answer[:1000]}\n"
                    f"Артефактов: {artifact_count}\n"
                    f"Tool calls: {tool_calls_count}\n"
                )
                response = llm.invoke([
                    SystemMessage(content=_EVALUATE_SYSTEM_PROMPT),
                    HumanMessage(content=eval_prompt),
                ])
                result = str(getattr(response, "content", "")).strip()
                # Try to extract JSON from response
                if result:
                    return result
            except Exception as exc:
                logger.warning("ReviewTool LLM check failed: %s", exc)

        return json.dumps({"pass": True, "issues": []}, ensure_ascii=False)
