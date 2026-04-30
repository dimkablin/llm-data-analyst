from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


RouteDecision = Literal["chat", "summary", "analysis"]


def is_chat_query(normalized_prompt: str, *, has_data: bool = False) -> bool:
    """Return True when a prompt should use the lightweight chat fallback path."""
    return RouteClassifier().is_chat(normalized_prompt, has_data=has_data)


@dataclass(slots=True, frozen=True)
class RouteClassifier:
    """Deterministic first-pass request classifier.

    This keeps cheap routing outside the LLM path. It intentionally errs on
    the side of `analysis` whenever the prompt may require data access.
    """

    summary_markers: tuple[str, ...] = (
        "executive summary",
        "summary",
        "итоги анализа",
        "резюмируй",
        "подведи итог",
        "сделай отчет",
        "сделай отчёт",
        "управленческ",
    )
    chat_markers: tuple[str, ...] = (
        "hello",
        "hi",
        "hey",
        "thanks",
        "thank you",
        "привет",
        "здравствуй",
        "добрый день",
        "доброе утро",
        "добрый вечер",
        "спасибо",
        "как дела",
        "как ты",
        "как поживаешь",
        "кто ты",
        "что ты умеешь",
        "расскажи о себе",
        "ты кто",
        "что можешь",
        "помоги",
        "help",
        "что ты такое",
        "как тебя зовут",
        "твоё имя",
    )
    data_markers: tuple[str, ...] = (
        "analyze",
        "analyse",
        "analysis",
        "dataset",
        "dataframe",
        "sql",
        "query",
        "uploaded",
        "attached",
        "connected",
        "chart",
        "plot",
        "table",
        "загрузил",
        "загрузила",
        "загружено",
        "залил",
        "залила",
        "добавил",
        "добавила",
        "подключил",
        "подключила",
        "анализ",
        "проанализ",
        "датасет",
        "таблиц",
        "данн",
        "покажи",
        "построй",
        "посчитай",
        "сколько",
        "график",
        "срез",
        "метрик",
    )

    def classify(self, prompt: str, *, has_data: bool = False) -> RouteDecision:
        normalized = prompt.strip().lower()
        if not normalized:
            return "chat"
        if any(marker in normalized for marker in self.summary_markers):
            return "summary"
        if self.is_chat(normalized, has_data=has_data):
            return "chat"
        return "analysis"

    def is_chat(self, normalized_prompt: str, *, has_data: bool = False) -> bool:
        """Classify small-talk and assistant-help prompts without invoking an LLM."""
        prompt = normalized_prompt.strip().lower()
        if not prompt:
            return True
        if not has_data and any(
            prompt == marker or prompt.startswith(marker) for marker in self.chat_markers
        ):
            return True
        if any(marker in prompt for marker in self.data_markers):
            return False
        if any(prompt == marker or prompt.startswith(marker) for marker in self.chat_markers):
            return True
        if len(prompt) < 4 and not has_data:
            return True
        return False
