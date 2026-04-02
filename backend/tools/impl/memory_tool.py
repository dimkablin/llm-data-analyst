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

from typing import Any, Callable

from langchain_core.tools import BaseTool


# ── User Memory Tool ──────────────────────────────────────────────────────────

_USER_MEMORY_DESCRIPTION = """\
Save a long-term fact about the user that will be useful in future \
conversations. Call this when you notice something about the user's \
name, role, expertise, preferences, communication style, or recurring \
patterns.

Do NOT use this for session-specific data (datasets, findings, schemas) \
— use session_note for that.

Input: plain text (1-3 sentences max).
Output: confirmation message.

Example: memory.save_note("User is a data analyst at a retail company. \
Prefers monthly aggregations and Plotly bar charts.")
"""


class MemoryTool(BaseTool):
    """Append a user-level note to the user memory buffer."""

    name: str = "memory"
    description: str = _USER_MEMORY_DESCRIPTION

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, on_note: Callable[[str], None], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_memo_cb", on_note)

    def _run(self, text: str, *args: Any, **_kwargs: Any) -> str:  # noqa: ARG002
        note = text.strip()
        if not note:
            return "Nothing saved — the note was empty."
        cb: Callable[[str], None] = object.__getattribute__(self, "_memo_cb")
        cb(note)
        return f"Saved to user memory: {note[:120]}{'…' if len(note) > 120 else ''}"

    async def _arun(self, text: str, *args: Any, **_kwargs: Any) -> str:  # noqa: ARG002
        return self._run(text)


# ── Session Note Tool ─────────────────────────────────────────────────────────

_SESSION_NOTE_DESCRIPTION = """\
Save a note about the current analysis session — data descriptions, \
key findings, intermediate conclusions, data-quality observations, \
column semantics, or any context useful for follow-up questions in \
this session.

Do NOT use this for facts about the user (preferences, role) — use \
memory for that.

Input: plain text (1-3 sentences max).
Output: confirmation message.

Example: session_note.save_note("Dataset contains 12 months of sales \
data with columns: date, product_id, revenue, region. Revenue has \
3% null values in Q2.")
"""


class SessionNoteTool(BaseTool):
    """Append a session-level note to the session memory buffer."""

    name: str = "session_note"
    description: str = _SESSION_NOTE_DESCRIPTION

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, on_note: Callable[[str], None], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_memo_cb", on_note)

    def _run(self, text: str, *args: Any, **_kwargs: Any) -> str:  # noqa: ARG002
        note = text.strip()
        if not note:
            return "Nothing saved — the note was empty."
        cb: Callable[[str], None] = object.__getattribute__(self, "_memo_cb")
        cb(note)
        return f"Saved to session notes: {note[:120]}{'…' if len(note) > 120 else ''}"

    async def _arun(self, text: str, *args: Any, **_kwargs: Any) -> str:  # noqa: ARG002
        return self._run(text)
