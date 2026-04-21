"""
E2E user-story tests for the B3 context state pipeline.

These tests use real components (no LLM calls). They verify user-observable
behavior from tool result → handle → session memory → prompt block.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from langchain_core.messages import ToolMessage

from backend.agent.runner import (
    _apply_observation_masking,
    _build_tool_message_text,
    _extract_findings_from_actions,
)
from backend.agent.working_memory import AnalysisWorkingMemory, ArtifactHandle
from backend.sessions.session_memory import SessionArtifactRef, StructuredSessionMemory
from backend.sessions.session_store import SessionStore

# ---------------------------------------------------------------------------
# Story 1: After a tool call produces a table, the agent sees its metadata
#           in the next turn.
# ---------------------------------------------------------------------------

def test_table_handle_created_and_flushed_to_session():
    """
    User story: After the agent runs a SQL/pandas tool and produces a table,
    the next query should know that table exists (name, shape, columns).
    This is verified by: handle created from tool result → flushed to session
    → build_block() includes it.
    """
    # 1. Create a table tool result artifact dict
    artifact = {
        "schema_version": "1.0",
        "artifact_type": "table",
        "items": {
            "revenue_by_region": pd.DataFrame(
                {"region": ["A", "B"], "revenue": [100, 200]}
            )
        },
    }

    # 2. Wrap it in a mock result object with content="" and artifact=...
    result_obj = SimpleNamespace(content="", artifact=artifact)

    # 3. Call _build_tool_message_text(result_obj) → get (text, handle)
    _text, handle = _build_tool_message_text(result_obj)

    # 4. Assert handle properties
    assert handle is not None, "Expected a handle to be created from table artifact"
    assert handle.name == "revenue_by_region"
    assert handle.type == "table"
    assert handle.row_count == 2
    assert handle.schema is not None
    assert "region" in handle.schema
    assert "revenue" in handle.schema

    # 5. Create AnalysisWorkingMemory
    working_memory = AnalysisWorkingMemory(goal="analyze revenue")

    # 6. Fill in caller-side fields
    handle.tool_name = "sql_tool"
    handle.step_index = 0

    # 7. Append handle to working_memory
    working_memory.artifact_handles.append(handle)

    # 8. Create StructuredSessionMemory
    session_memory = StructuredSessionMemory(turn_count=0)

    # 9. Simulate the flush: create SessionArtifactRef from handle, append to session_memory
    ref = SessionArtifactRef(
        id=handle.id,
        name=handle.name,
        type=handle.type,
        turn_index=0,
        schema=handle.schema,
        row_count=handle.row_count,
        summary=handle.summary,
    )
    session_memory.artifact_index.append(ref)

    # 10. Call session_memory.build_block()
    block = session_memory.build_block()

    # 11. Assert build_block() contains artifact name and type — but NOT raw row values
    assert "revenue_by_region" in block
    assert "table" in block
    # Raw row values should NOT appear inline
    assert "\"A\"" not in block and "'A'" not in block
    assert "\"B\"" not in block and "'B'" not in block


# ---------------------------------------------------------------------------
# Story 2: Findings accumulate across multiple turns.
# ---------------------------------------------------------------------------

def test_key_findings_accumulate_across_turns():
    """
    User story: After running 3 queries with analytical tool calls,
    the session memory's key_findings list should contain entries from all 3 turns.
    This helps the agent recall previous analytical steps when answering
    follow-up questions.
    """
    # 1. Create StructuredSessionMemory
    session_memory = StructuredSessionMemory(turn_count=0)

    # 2. Turn 0
    wm0 = AnalysisWorkingMemory(goal="Q1")
    wm0.completed_actions = [
        "sql_tool → revenue_table (100 rows)",
        "plotly_tool → revenue_chart",
    ]
    findings0 = _extract_findings_from_actions(wm0.completed_actions, turn_index=0)
    session_memory.key_findings.extend(findings0)
    session_memory.turn_count += 1

    # 3. Turn 1
    wm1 = AnalysisWorkingMemory(goal="Q2")
    wm1.completed_actions = [
        "pandas_tool → aggregated_df (50 rows)",
        "plotly_tool → trend_chart",
    ]
    findings1 = _extract_findings_from_actions(wm1.completed_actions, turn_index=1)
    session_memory.key_findings.extend(findings1)
    session_memory.turn_count += 1

    # 4. Turn 2
    wm2 = AnalysisWorkingMemory(goal="Q3")
    wm2.completed_actions = [
        "sql_tool → churn_table (200 rows)",
        "pandas_tool → churn_summary",
    ]
    findings2 = _extract_findings_from_actions(wm2.completed_actions, turn_index=2)
    session_memory.key_findings.extend(findings2)
    session_memory.turn_count += 1

    # 5. Assert at least one finding per turn passed the filter
    assert len(session_memory.key_findings) >= 3

    # 6. Assert all turns are represented
    all_findings_str = " ".join(session_memory.key_findings)
    assert "[turn 0]" in all_findings_str
    assert "[turn 1]" in all_findings_str
    assert "[turn 2]" in all_findings_str

    # 7. Assert turn_count == 3
    assert session_memory.turn_count == 3

    # 8. Assert build_block() includes "Key findings" section
    block = session_memory.build_block()
    assert "Key findings" in block
    # At least one finding should be in the block
    assert "[turn 0]" in block or "[turn 1]" in block or "[turn 2]" in block


# ---------------------------------------------------------------------------
# Story 3: Old session (string-only notes) migrates transparently.
# ---------------------------------------------------------------------------

def test_old_session_migrates_to_structured_memory(tmp_path: Path):
    """
    User story: A user with an existing session (created before this feature)
    can continue using the app without any errors or data loss.
    Their old notes are preserved and accessible.
    """
    # 1. Create a SessionStore and session
    store = SessionStore(str(tmp_path), ttl_days=7)
    session = store.create_session()
    session_id = session.session_id

    # 2. Manually patch the state file to OLD format
    state_path = tmp_path / session_id / "state.json"
    raw = json.loads(state_path.read_text())
    raw["session_memory"] = "some old notes"
    raw.pop("artifact_index_json", None)
    raw.pop("key_findings", None)
    raw.pop("session_turn_count", None)
    state_path.write_text(json.dumps(raw))

    # 3. Call store.get_structured_memory(session_id)
    memory = store.get_structured_memory(session_id)

    # 4-9. Assert migration results
    assert isinstance(memory, StructuredSessionMemory)
    assert memory.notes == "some old notes"
    assert memory.artifact_index == []
    assert memory.key_findings == []
    assert memory.turn_count == 0

    # 10. Call memory.build_block() → assert contains old notes
    block = memory.build_block()
    assert "some old notes" in block


# ---------------------------------------------------------------------------
# Story 4: Session memory persists and reloads correctly across queries.
# ---------------------------------------------------------------------------

def test_structured_memory_persists_and_reloads(tmp_path: Path):
    """
    User story: After a query produces artifacts and findings, the next query
    should see them in the context. This requires persist (set_structured_memory)
    then reload (get_structured_memory) to be symmetric.
    """
    # 1. Create store and session
    store = SessionStore(str(tmp_path), ttl_days=7)
    session = store.create_session()
    session_id = session.session_id

    # 2. Build a StructuredSessionMemory
    ref = SessionArtifactRef(
        id="a1",
        name="sales_table",
        type="table",
        turn_index=0,
        schema={"date": "str", "sales": "int64"},
        row_count=50,
        summary="sales_table summary",
    )
    memory = StructuredSessionMemory(
        notes="user prefers bar charts",
        artifact_index=[ref],
        key_findings=["[turn 0] sql_tool → sales_table (50 rows)"],
        turn_count=1,
    )

    # 3. Persist
    store.set_structured_memory(session_id, memory)

    # 4. Reload
    loaded = store.get_structured_memory(session_id)

    # 5-10. Assert all fields survived the round-trip
    assert loaded.notes == "user prefers bar charts"
    assert len(loaded.artifact_index) == 1
    assert loaded.artifact_index[0].name == "sales_table"
    assert loaded.artifact_index[0].row_count == 50
    assert loaded.key_findings == ["[turn 0] sql_tool → sales_table (50 rows)"]
    assert loaded.turn_count == 1


# ---------------------------------------------------------------------------
# Story 5: Observation masking: after 4+ tool calls, early messages are compact.
# ---------------------------------------------------------------------------

def test_observation_masking_compresses_early_tool_messages():
    """
    User story: When the agent runs many tools in sequence, the message history
    stays manageable. Early tool results are replaced with compact metadata
    so the LLM context doesn't get overloaded with raw data.
    """
    # 1. Create AnalysisWorkingMemory
    _wm = AnalysisWorkingMemory(goal="big analysis")

    # 2. Build 5 handles for steps 0-4
    handles = [
        ArtifactHandle(
            id=f"id_{i}",
            name=f"table_{i}",
            type="table",
            tool_name="sql_tool",
            step_index=i,
            schema={"col_a": "int64", "col_b": "object"},
            row_count=100 + i * 10,
            summary=f"table_{i} summary",
        )
        for i in range(5)
    ]

    # 3. Build tc_id_to_handle and tc_id_to_step dicts
    tc_id_to_handle = {f"tc{i}": handles[i] for i in range(5)}
    tc_id_to_step = {f"tc{i}": i for i in range(5)}

    # 4. Build 5 ToolMessage objects with long content (200+ chars each)
    long_content = "x" * 250
    messages = [
        ToolMessage(content=f"tc{i}_full_data: {long_content}", tool_call_id=f"tc{i}")
        for i in range(5)
    ]

    # 5. Create masked_tc_ids set
    masked_tc_ids: set = set()

    # 6. Call _apply_observation_masking with current_step=5
    _apply_observation_masking(
        messages, tc_id_to_handle, tc_id_to_step,
        current_step=5,
        masked_tc_ids=masked_tc_ids,
    )

    # 7-8. Assert masking policy:
    # steps_ago = 5 - step_index
    # step 0 → steps_ago=5 >= 3 → masked
    # step 1 → steps_ago=4 >= 3 → masked
    # step 2 → steps_ago=3 >= 3 → masked (boundary: masked when steps_ago >= KEEP_LAST_N)
    # step 3 → steps_ago=2 < 3 → kept
    # step 4 → steps_ago=1 < 3 → kept
    for i in range(3):
        assert messages[i].content.startswith("[artifact:"), (
            f"tc{i} (steps_ago={5 - i}) should be masked, got: {messages[i].content!r}"
        )
    for i in range(3, 5):
        assert not messages[i].content.startswith("[artifact:"), (
            f"tc{i} (steps_ago={5 - i}) should NOT be masked, got: {messages[i].content!r}"
        )

    # 9. Assert masked content contains artifact metadata
    masked_msg = messages[0]
    assert "table_0" in masked_msg.content
    assert "table" in masked_msg.content


# ---------------------------------------------------------------------------
# Story 6: Infra tool calls don't pollute findings.
# ---------------------------------------------------------------------------

def test_infrastructure_tool_calls_excluded_from_findings():
    """
    User story: When the agent uses infrastructure tools (database_tool,
    planner_tool) to set up context, those internal steps don't appear in the
    session's key findings. Only meaningful analytical results are retained.
    """
    actions = [
        "database_tool → schema",
        "planner_tool → [5-step plan]",
        "sql_tool → monthly_revenue (1200 rows)",
        "pandas_tool → aggregated_df",
        "get_tool_instructions → auto_eda",
    ]

    # 1. Extract findings
    findings = _extract_findings_from_actions(actions, turn_index=1)

    # 2. Assert exactly 2 entries (sql_tool and pandas_tool)
    assert len(findings) == 2, f"Expected 2 findings, got {len(findings)}: {findings}"

    # 3. Assert each entry starts with "[turn 1]"
    for f in findings:
        assert f.startswith("[turn 1]"), f"Finding missing turn prefix: {f!r}"

    # 4. Assert infra tools are excluded
    findings_str = " ".join(findings)
    assert "database_tool" not in findings_str
    assert "planner_tool" not in findings_str
    assert "get_tool_instructions" not in findings_str
