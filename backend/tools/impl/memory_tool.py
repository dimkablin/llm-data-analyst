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
from typing import Any

from langchain_core.tools import BaseTool

# ── User Memory Tool ──────────────────────────────────────────────────────────

_USER_MEMORY_DESCRIPTION = """\
Persist a long-term insight about the user that will be useful across \
future conversations. Call this when you infer something meaningful \
about the user — their role, domain, expertise level, analytical \
goals, preferred output format, communication style, or recurring \
patterns in how they work.

Write a distilled, semantic note — NOT a verbatim quote. Capture \
*what it means*, not *what was said*. For example, instead of \
"User said they work in retail", write "User is a retail industry \
analyst focused on sales performance".

Guidelines:
- Infer and abstract: translate observations into durable facts.
- Be specific: "prefers Plotly bar charts with monthly granularity" \
  is better than "likes charts".
- One atomic fact per call; call multiple times for multiple facts.
- Do NOT save session-specific data (datasets, schemas, findings) \
  — use session_note for that.
- Do NOT call this to recall history — it is already in the prompt.

Input: 1–2 sentences of distilled insight.
Output: confirmation message.

Example: memory("User is a retail data analyst. \
Prefers concise monthly aggregations and Plotly bar charts over tables.")
"""


class MemoryTool(BaseTool):
    """Append a user-level note to the user memory buffer."""

    name: str = "memory"
    description: str = _USER_MEMORY_DESCRIPTION

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, on_note: Callable[[str], None], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_memo_cb", on_note)

    def _run(self, text: str, *args: Any, **_kwargs: Any) -> str:
        note = text.strip()
        if not note:
            return "Nothing saved — the note was empty."
        cb: Callable[[str], None] = object.__getattribute__(self, "_memo_cb")
        cb(note)
        return f"Saved to user memory: {note[:120]}{'…' if len(note) > 120 else ''}"

    async def _arun(self, text: str, *args: Any, **_kwargs: Any) -> str:
        return self._run(text)


# ── Session Note Tool ─────────────────────────────────────────────────────────

_SESSION_NOTE_DESCRIPTION = """\
Persist a semantic insight about the current analysis session so it \
can be referenced in follow-up questions without re-running analysis.

Use this for:
- The *meaning* of loaded data (domain, granularity, time range, \
  business purpose) — not just column names.
- Key findings and conclusions: what patterns, anomalies, or answers \
  were discovered and *why they matter*.
- Data-quality issues that affect interpretation (e.g., "Q2 revenue \
  has 3% nulls — likely export gap, not zero sales").
- Decisions made during analysis (filters applied, assumptions used).

Write a distilled, semantic note — NOT a verbatim copy of data or \
output. Capture *understanding*, not raw facts. For example, instead \
of listing columns, explain what the dataset represents: "Monthly \
retail sales by product and region, 2023 — used to analyze \
regional revenue trends".

Do NOT use this for facts about the user (role, preferences) \
— use memory for that.

Input: 1–3 sentences of distilled context or insight.
Output: confirmation message.

Example: session_note("Sales dataset covers Jan–Dec 2023 with \
revenue by product and region. Q2 revenue has 3% nulls likely \
due to a data export gap, not actual zero sales.")
"""


class SessionNoteTool(BaseTool):
    """Append a session-level note to the session memory buffer."""

    name: str = "session_note"
    description: str = _SESSION_NOTE_DESCRIPTION

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, on_note: Callable[[str], None], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_memo_cb", on_note)

    def _run(self, text: str, *args: Any, **_kwargs: Any) -> str:
        note = text.strip()
        if not note:
            return "Nothing saved — the note was empty."
        cb: Callable[[str], None] = object.__getattribute__(self, "_memo_cb")
        cb(note)
        return f"Saved to session notes: {note[:120]}{'…' if len(note) > 120 else ''}"

    async def _arun(self, text: str, *args: Any, **_kwargs: Any) -> str:
        return self._run(text)
