"""Tests that think blocks are stripped from history when LLM_SHOW_THINK=False."""
import copy
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
                "pre_text": "Checking `x` before the tool call.",
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
    state.context_usage = {}
    return state


def test_strip_helper_removes_thinking_fields():
    """_strip_thinking_from_history removes reasoning_steps and pre_reasoning from AI messages."""
    from backend.api.routes.sessions import _strip_thinking_from_history

    result = _strip_thinking_from_history(copy.deepcopy(CHAT_HISTORY_WITH_THINK))

    ai_message = result[1]
    assert "reasoning_steps" not in ai_message
    for tool in ai_message.get("tools", []):
        assert "pre_reasoning" not in tool
        assert tool["pre_text"] == "Checking `x` before the tool call."


def test_strip_helper_does_not_mutate_original():
    """_strip_thinking_from_history must not modify the original list or dicts."""
    from backend.api.routes.sessions import _strip_thinking_from_history

    original = copy.deepcopy(CHAT_HISTORY_WITH_THINK)
    _strip_thinking_from_history(original)

    # Original must be untouched
    assert "reasoning_steps" in original[1]
    assert original[1]["tools"][0].get("pre_reasoning") == "Let me think about this..."


def test_strip_helper_leaves_user_messages_untouched():
    """User messages must not be modified."""
    from backend.api.routes.sessions import _strip_thinking_from_history

    result = _strip_thinking_from_history(copy.deepcopy(CHAT_HISTORY_WITH_THINK))

    assert result[0]["content"] == "hello"
    assert result[0]["role"] == "user"


def test_get_session_strips_think_blocks_when_show_think_false():
    """get_session must filter reasoning_steps and pre_reasoning when llm_show_think=False."""
    from backend.api.routes.sessions import get_session

    state = _make_state(copy.deepcopy(CHAT_HISTORY_WITH_THINK))
    user = MagicMock()
    user.id = "u1"

    with patch("backend.api.routes.sessions._store") as mock_store, \
         patch("backend.api.routes.sessions._auth_db") as mock_auth_db, \
         patch("backend.api.routes.sessions._manifest_store", None), \
         patch("backend.api.routes.sessions.settings") as mock_settings:
        mock_settings.llm_show_think = False
        mock_store.load_session.return_value = state
        mock_auth_db.is_session_owner.return_value = True
        mock_auth_db.get_session_metadata.return_value = {"title": "t"}

        response = get_session("test-session", user)

    ai_msg = response.chat_history[1]
    assert "reasoning_steps" not in ai_msg
    for tool in ai_msg.get("tools", []):
        assert "pre_reasoning" not in tool
        assert tool["pre_text"] == "Checking `x` before the tool call."


def test_get_session_preserves_think_blocks_when_show_think_true():
    """get_session must NOT filter blocks when llm_show_think=True."""
    from backend.api.routes.sessions import get_session

    state = _make_state(copy.deepcopy(CHAT_HISTORY_WITH_THINK))
    user = MagicMock()
    user.id = "u1"

    with patch("backend.api.routes.sessions._store") as mock_store, \
         patch("backend.api.routes.sessions._auth_db") as mock_auth_db, \
         patch("backend.api.routes.sessions._manifest_store", None), \
         patch("backend.api.routes.sessions.settings") as mock_settings:
        mock_settings.llm_show_think = True
        mock_store.load_session.return_value = state
        mock_auth_db.is_session_owner.return_value = True
        mock_auth_db.get_session_metadata.return_value = {"title": "t"}

        response = get_session("test-session", user)

    ai_msg = response.chat_history[1]
    assert "reasoning_steps" in ai_msg
    assert ai_msg["tools"][0].get("pre_reasoning") == "Let me think about this..."
