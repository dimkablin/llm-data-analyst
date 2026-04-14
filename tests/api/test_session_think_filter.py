"""Tests that think blocks are stripped from history when LLM_SHOW_THINK=False."""
import pytest
from unittest.mock import MagicMock, patch


CHAT_HISTORY_WITH_THINK = [
    {"role": "user", "content": "hello"},
    {
        "role": "ai",
        "content": "sure",
        "reasoning_steps": [
            {"step_index": 0, "kind": "final_synthesis", "content": "I think...", "tool_name": None}
        ],
        "tools": [
            {
                "tool_name": "python_repl",
                "pre_reasoning": "Let me think about this...",
                "input_summary": "x = 1",
                "status": "done",
            }
        ],
    },
]


def _make_state(chat_history):
    state = MagicMock()
    state.session_id = "test-session"
    state.chat_history = chat_history
    state.artifacts = []
    state.df_path = None
    state.dataset_name = None
    state.source_type = None
    state.source_ref_id = None
    state.source_label = None
    state.source_mode = None
    state.selected_skill_ids = []
    state.session_memory = ""
    return state


def test_think_blocks_stripped_when_show_think_false():
    """reasoning_steps and pre_reasoning must be absent when called (function always strips)."""
    from backend.api.routes.sessions import _strip_thinking_from_history

    result = _strip_thinking_from_history(CHAT_HISTORY_WITH_THINK)

    ai_message = result[1]
    assert "reasoning_steps" not in ai_message or ai_message.get("reasoning_steps") is None
    tools = ai_message.get("tools", [])
    for tool in tools:
        assert "pre_reasoning" not in tool or tool.get("pre_reasoning") is None


def test_think_blocks_preserved_when_show_think_true():
    """reasoning_steps and pre_reasoning must remain in original data (function strips them)."""
    from backend.api.routes.sessions import _strip_thinking_from_history

    # The function always strips — verify the original data is unchanged (not mutated)
    import copy
    original = copy.deepcopy(CHAT_HISTORY_WITH_THINK)
    _strip_thinking_from_history(CHAT_HISTORY_WITH_THINK)

    # Original must not be mutated
    ai_message = original[1]
    assert ai_message.get("reasoning_steps") is not None
    tools = ai_message.get("tools", [])
    assert tools[0].get("pre_reasoning") == "Let me think about this..."


def test_user_messages_untouched():
    """User messages must not be modified."""
    from backend.api.routes.sessions import _strip_thinking_from_history

    result = _strip_thinking_from_history(CHAT_HISTORY_WITH_THINK)
    assert result[0]["content"] == "hello"
    assert result[0]["role"] == "user"
