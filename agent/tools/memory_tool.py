"""Memory tool — lets the agent persist observations across sessions.

The agent calls ``memory.save_note(text)`` when it wants to remember
something useful about the user for future conversations. Notes are
buffered in the current ``AgentRunner`` instance and consolidated into
SQLite asynchronously after the response is sent.

This is *not* a sandboxed code-execution tool — it is a direct
LangChain ``BaseTool`` backed by a simple in-memory list.
"""
from __future__ import annotations

from typing import Any, Callable

from langchain_core.tools import BaseTool


_TOOL_DESCRIPTION = """\
Save a short observation or fact about the user that will be useful \
in future conversations. Call this when you notice something about \
the user's preferences, domain knowledge, recurring data patterns, \
or any context worth remembering.

Input: plain text (1-3 sentences max).
Output: confirmation message.

Example: memory.save_note("User works with retail sales data. \
Prefers monthly aggregations and Plotly bar charts.")
"""


class MemoryTool(BaseTool):
    """Append a note to the session memory buffer."""

    name: str = "memory"
    description: str = _TOOL_DESCRIPTION

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, on_note: Callable[[str], None], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Bypass Pydantic descriptor protocol: store callback in plain __dict__.
        object.__setattr__(self, "_memo_cb", on_note)

    # ── BaseTool interface ────────────────────────────────────────────────────

    def _run(self, text: str, *args: Any, **_kwargs: Any) -> str:  # noqa: ARG002
        note = text.strip()
        if not note:
            return "Nothing saved — the note was empty."
        cb: Callable[[str], None] = object.__getattribute__(self, "_memo_cb")
        cb(note)
        return f"Saved to memory: {note[:120]}{'…' if len(note) > 120 else ''}"

    async def _arun(self, text: str, *args: Any, **_kwargs: Any) -> str:  # noqa: ARG002
        return self._run(text)
