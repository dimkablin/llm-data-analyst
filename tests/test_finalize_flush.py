"""Tests for Task 4: flush working_memory → StructuredSessionMemory."""
from __future__ import annotations

import pytest

from backend.agent.runner import _extract_findings_from_actions
from backend.agent.working_memory import AnalysisWorkingMemory, ArtifactHandle
from backend.sessions.session_memory import SessionArtifactRef, StructuredSessionMemory


# ---------------------------------------------------------------------------
# _extract_findings_from_actions
# ---------------------------------------------------------------------------

def test_extract_findings_skips_infra_tools():
    actions = [
        "database_tool → schema",
        "planner_tool → plan",
        "sql_tool → revenue_table (100 rows)",
        "pandas_tool → agg_df",
    ]
    result = _extract_findings_from_actions(actions, turn_index=2)
    assert len(result) == 2
    assert any("sql_tool" in r for r in result)
    assert any("pandas_tool" in r for r in result)
    assert not any("database_tool" in r for r in result)
    assert not any("planner_tool" in r for r in result)
    # Each entry has [turn N] prefix
    for r in result:
        assert r.startswith("[turn 2]")


def test_extract_findings_empty_input():
    result = _extract_findings_from_actions([], turn_index=0)
    assert result == []


def test_extract_findings_all_infra():
    actions = [
        "database_tool → schema",
        "planner_tool → plan",
        "get_tool_instructions → ...",
    ]
    result = _extract_findings_from_actions(actions, turn_index=1)
    assert result == []


# ---------------------------------------------------------------------------
# flush logic (inline simulation — same logic as run_query)
# ---------------------------------------------------------------------------

def _do_flush(
    working_memory: AnalysisWorkingMemory,
    structured: StructuredSessionMemory,
) -> None:
    """Replicate the flush logic from AgentRunner.run_query()."""
    for handle in working_memory.artifact_handles:
        ref = SessionArtifactRef(
            id=handle.id,
            name=handle.name,
            type=handle.type,
            turn_index=structured.turn_count,
            schema=handle.schema,
            row_count=handle.row_count,
            summary=handle.summary,
        )
        structured.artifact_index.append(ref)
    new_findings = _extract_findings_from_actions(
        working_memory.completed_actions,
        turn_index=structured.turn_count,
    )
    structured.key_findings = (structured.key_findings + new_findings)[-30:]
    structured.turn_count += 1


def _make_handle(name: str, type_: str = "table") -> ArtifactHandle:
    return ArtifactHandle(
        id=f"id-{name}",
        name=name,
        type=type_,
        tool_name="sql_tool",
        step_index=0,
        schema={"col_a": "int64", "col_b": "str"},
        row_count=10,
        summary=f"summary of {name}",
    )


def test_flush_populates_artifact_index():
    wm = AnalysisWorkingMemory(goal="test")
    wm.artifact_handles.append(_make_handle("revenue_table"))
    wm.artifact_handles.append(_make_handle("cost_table", type_="table"))

    structured = StructuredSessionMemory(turn_count=0)
    _do_flush(wm, structured)

    assert len(structured.artifact_index) == 2
    names = {ref.name for ref in structured.artifact_index}
    assert "revenue_table" in names
    assert "cost_table" in names
    for ref in structured.artifact_index:
        assert ref.turn_index == 0
        assert ref.type == "table"
    assert structured.turn_count == 1


def test_flush_key_findings_accumulate():
    structured = StructuredSessionMemory(turn_count=0)

    for i in range(3):
        wm = AnalysisWorkingMemory(goal=f"turn {i}")
        wm.completed_actions.append(f"sql_tool → result_{i}")
        _do_flush(wm, structured)

    assert len(structured.key_findings) == 3
    assert structured.turn_count == 3
    # Each finding has the correct turn index
    assert "[turn 0]" in structured.key_findings[0]
    assert "[turn 1]" in structured.key_findings[1]
    assert "[turn 2]" in structured.key_findings[2]


def test_flush_key_findings_capped_at_30():
    structured = StructuredSessionMemory(
        turn_count=5,
        key_findings=[f"[turn {i}] old finding" for i in range(28)],
    )

    wm = AnalysisWorkingMemory(goal="cap test")
    for j in range(5):
        wm.completed_actions.append(f"sql_tool → new_result_{j}")

    _do_flush(wm, structured)

    assert len(structured.key_findings) <= 30


def test_flush_empty_working_memory():
    wm = AnalysisWorkingMemory(goal="test")
    structured = StructuredSessionMemory(turn_count=0)

    _do_flush(wm, structured)

    assert structured.artifact_index == []
    assert structured.key_findings == []
    assert structured.turn_count == 1
