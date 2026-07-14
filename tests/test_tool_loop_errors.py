from __future__ import annotations

from backend.agent.callbacks import ToolCollector
from backend.agent.tool_loop import (
    _compact_tool_error_message,
    _is_source_unavailable_observation,
    _is_tool_error_observation,
)


def test_tool_error_observation_accepts_normal_russian_error_text() -> None:
    assert _is_tool_error_observation("❌ Ошибка при создании таблиц: failed") is True
    assert _is_tool_error_observation("Ошибка при выполнении SQL: failed") is True


def test_source_unavailable_observation_detects_transport_errors() -> None:
    assert _is_source_unavailable_observation(
        "Tool error: connection to host.docker.internal failed: Network is unreachable"
    ) is True
    assert _is_source_unavailable_observation(
        "Tool error: server closed the connection unexpectedly"
    ) is True


def test_compact_tool_error_message_keeps_error_and_line_without_traceback() -> None:
    text = _compact_tool_error_message(
        "Tool error: Traceback (most recent call last):\n"
        '  File "/app/backend/tools/impl/pandas_tool.py", line 42, in _run\n'
        "    result = run_code()\n"
        "ValueError: column 'sales' not found"
    )

    assert "Traceback" not in text
    assert "/app/backend" not in text
    assert "line 42" in text
    assert "ValueError: column 'sales' not found" in text


def test_compact_tool_error_message_preserves_keyerror() -> None:
    text = _compact_tool_error_message("Tool error: KeyError: 'portfolio_id'")

    assert text == "KeyError: 'portfolio_id'"


def test_tool_collector_error_event_is_compact_for_frontend() -> None:
    collector = ToolCollector()

    collector.on_tool_error(
        "Traceback (most recent call last):\n"
        '  File "/app/backend/tools/impl/sql_tool.py", line 17, in _run\n'
        "RuntimeError: bad SQL",
        tool="sql_tool",
    )

    event = collector.events[-1]
    assert event["error"] == "line 17: RuntimeError: bad SQL"
