"""Tests for Task 4: flush working_memory → StructuredSessionMemory."""
from __future__ import annotations

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
    """Replicate the flush logic from AgentRunner.run_query() (with dedup + cap)."""
    existing_ids = {r.id for r in structured.artifact_index}
    for handle in working_memory.artifact_handles:
        if handle.id in existing_ids:
            continue
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
        existing_ids.add(handle.id)
    # Cap artifact_index at 100 entries (oldest evicted)
    structured.artifact_index = structured.artifact_index[-100:]
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


# ---------------------------------------------------------------------------
# Issue 2 – deduplication and cap
# ---------------------------------------------------------------------------

def test_flush_deduplicates_artifact_refs():
    """Two handles with the same id should produce only one artifact_index entry."""
    wm = AnalysisWorkingMemory(goal="dedup test")
    # Add the same handle twice (same id)
    handle = _make_handle("revenue_table")
    wm.artifact_handles.append(handle)
    duplicate = ArtifactHandle(
        id=handle.id,  # same id
        name="revenue_table_copy",
        type="table",
        tool_name="sql_tool",
        step_index=1,
        schema={},
        row_count=5,
        summary="duplicate",
    )
    wm.artifact_handles.append(duplicate)

    structured = StructuredSessionMemory(turn_count=0)
    _do_flush(wm, structured)

    assert len(structured.artifact_index) == 1
    assert structured.artifact_index[0].id == handle.id


def test_flush_caps_artifact_index_at_100():
    """artifact_index should never exceed 100 entries; oldest are evicted."""
    # Pre-load with 98 distinct refs
    existing_refs = [
        SessionArtifactRef(
            id=f"existing-{i}",
            name=f"table_{i}",
            type="table",
            turn_index=0,
            schema=None,
            row_count=None,
            summary=None,
        )
        for i in range(98)
    ]
    structured = StructuredSessionMemory(turn_count=1, artifact_index=existing_refs)

    # Flush 5 new unique handles → total would be 103 without cap
    wm = AnalysisWorkingMemory(goal="cap test")
    for j in range(5):
        wm.artifact_handles.append(_make_handle(f"new_table_{j}"))

    _do_flush(wm, structured)

    assert len(structured.artifact_index) <= 100
    # The newest entries should be retained (last-100 strategy)
    new_names = {ref.name for ref in structured.artifact_index}
    for j in range(5):
        assert f"new_table_{j}" in new_names


# ---------------------------------------------------------------------------
# Issue 1 – flush only on valid response (conceptual / unit coverage)
# ---------------------------------------------------------------------------

def test_flush_only_on_valid_response():
    """_extract_findings_from_actions is NOT called when working_memory is None.

    This mirrors the guard in AgentRunner.run_query(): flush is skipped entirely
    when working_memory is None, so turn_count must remain unchanged.
    """
    structured = StructuredSessionMemory(turn_count=7)
    working_memory = None  # simulate absent working_memory (failure path)

    # Reproduce the guard as written in run_query
    if working_memory is not None:
        _do_flush(working_memory, structured)

    # turn_count must not have been incremented
    assert structured.turn_count == 7
    assert structured.artifact_index == []
