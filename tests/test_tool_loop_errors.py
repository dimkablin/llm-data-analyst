from __future__ import annotations

from langchain_core.messages import ToolMessage

from backend.agent.callbacks import ToolCollector
from backend.agent.tool_loop import (
    ToolFailureSummary,
    _compact_tool_error_message,
)


def test_tool_failure_uses_structured_message_status_not_prose() -> None:
    success = ToolMessage(
        content="Документ объясняет ошибку при заполнении формы.",
        name="rag_tool",
        tool_call_id="success-1",
        status="success",
    )
    failure = ToolMessage(
        content="Database request failed.",
        name="database_tool",
        tool_call_id="failure-1",
        status="error",
    )

    assert ToolFailureSummary.from_tool_message(output=success, message=str(success.content)) is None
    summary = ToolFailureSummary.from_tool_message(output=failure, message=str(failure.content))
    assert summary is not None
    assert summary.tool_name == "database_tool"


def test_tool_failure_uses_structured_artifact_status() -> None:
    output = ToolMessage(
        content="SQL execution details.",
        artifact={"status": "error", "error": "bad column"},
        name="sql_tool",
        tool_call_id="failure-2",
        status="success",
    )

    summary = ToolFailureSummary.from_tool_message(output=output, message=str(output.content))

    assert summary is not None
    assert summary.tool_name == "sql_tool"


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
