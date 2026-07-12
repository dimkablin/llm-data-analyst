from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Максимальная длина одного шага (символы) — защита от разбухания state.json
MAX_STEP_CONTENT_LEN = 8_000
# Максимальное число шагов — защита от очень длинных сессий
MAX_REASONING_STEPS = 20

ReasoningStepKind = Literal["planning", "tool_synthesis", "final_synthesis", "unknown"]


@dataclass
class ReasoningStep:
    """Один LLM-вызов с thinking блоком в ходе ответа агента.

    Attributes:
        step_index: Порядковый номер LLM-вызова (0-based).
        kind: Семантика шага (авто-детектируется в _build_reasoning_steps).
        content: Сырой текст thinking блока.
        tool_name: Инструмент, вызванный ПОСЛЕ этого шага (если есть).
    """

    step_index: int
    kind: ReasoningStepKind = "unknown"
    content: str = ""
    tool_name: str | None = None

    def to_dict(self) -> dict:
        d: dict = {
            "step_index": self.step_index,
            "kind": self.kind,
            "content": self.content,
        }
        if self.tool_name:
            d["tool_name"] = self.tool_name
        return d

    @staticmethod
    def from_dict(d: dict) -> ReasoningStep:
        return ReasoningStep(
            step_index=int(d.get("step_index", 0)),
            kind=d.get("kind", "unknown"),
            content=str(d.get("content", "")),
            tool_name=d.get("tool_name") or None,
        )

    def truncated(self) -> ReasoningStep:
        """Возвращает копию с content, обрезанным до MAX_STEP_CONTENT_LEN."""
        if len(self.content) <= MAX_STEP_CONTENT_LEN:
            return self
        return ReasoningStep(
            step_index=self.step_index,
            kind=self.kind,
            content=self.content[:MAX_STEP_CONTENT_LEN] + "…",
            tool_name=self.tool_name,
        )
