"""Memory tools — let the agent persist observations.

Two tools:

* **memory** — saves long-term facts *about the user* (preferences,
  role, expertise) into SQLite via the user-memory consolidation pipeline.

* **session_note** — saves context *about the current analysis session*
  (data descriptions, key findings, intermediate conclusions) into the
  session's ``state.json``.

Neither tool is a sandboxed code-execution tool — both are direct
LangChain ``BaseTool`` instances backed by simple in-memory callbacks.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, model_validator

from backend.tools.instructions import tool_description

# ── User Memory Tool ──────────────────────────────────────────────────────────

class _MemoryInput(BaseModel):
    text: str = Field(default="", description="Дистиллированное наблюдение о пользователе (1–2 предложения).")

    @model_validator(mode="before")
    @classmethod
    def _coerce_to_text(cls, values: Any) -> Any:
        if isinstance(values, dict) and "text" not in values:
            for v in values.values():
                if isinstance(v, str):
                    return {"text": v}
        return values


class MemoryTool(BaseTool):
    """Append a user-level note to the user memory buffer."""

    name: str = "memory"
    description: str = tool_description("memory_tool")
    args_schema: type[BaseModel] = _MemoryInput
    parallel_safe: ClassVar[bool] = False

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, on_note: Callable[[str], None], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_memo_cb", on_note)

    def _run(self, text: str, *args: Any, **_kwargs: Any) -> str:
        note = text.strip()
        if not note:
            return "Ничего не сохранено — заметка пустая."
        cb: Callable[[str], None] = object.__getattribute__(self, "_memo_cb")
        cb(note)
        return f"Сохранено в память пользователя: {note[:120]}{'…' if len(note) > 120 else ''}"

    async def _arun(self, text: str, *args: Any, **_kwargs: Any) -> str:
        return self._run(text)


# ── Session Note Tool ─────────────────────────────────────────────────────────

class _SessionNoteInput(BaseModel):
    text: str = Field(
        default="",
        description="Дистиллированный контекст или вывод о сессии (1–3 предложения).",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_to_text(cls, values: Any) -> Any:
        if isinstance(values, dict) and "text" not in values:
            for v in values.values():
                if isinstance(v, str):
                    return {"text": v}
        return values


class SessionNoteTool(BaseTool):
    """Append a session-level note to the session memory buffer."""

    name: str = "session_note"
    description: str = tool_description("session_note_tool")
    args_schema: type[BaseModel] = _SessionNoteInput
    parallel_safe: ClassVar[bool] = False

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, on_note: Callable[[str], None], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_memo_cb", on_note)

    def _run(self, text: str, *args: Any, **_kwargs: Any) -> str:
        note = text.strip()
        if not note:
            return "Ничего не сохранено — заметка пустая."
        cb: Callable[[str], None] = object.__getattribute__(self, "_memo_cb")
        cb(note)
        return f"Сохранено в заметки сессии: {note[:120]}{'…' if len(note) > 120 else ''}"

    async def _arun(self, text: str, *args: Any, **_kwargs: Any) -> str:
        return self._run(text)
