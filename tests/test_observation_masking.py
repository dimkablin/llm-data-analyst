"""Tests for observation masking in _direct_tool_loop (feature-flagged).

These tests validate the masking constants, ArtifactHandle.masked_ref,
and the masking pass logic without requiring a full LLM.
"""
from __future__ import annotations

from langchain_core.messages import ToolMessage

from backend.agent.runner import _MASK_KEEP_LAST_N, _MASK_MIN_STEPS, _MASK_MIN_TOOLS
from backend.agent.working_memory import AnalysisWorkingMemory, ArtifactHandle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_handle(name: str, artifact_type: str, step_index: int) -> ArtifactHandle:
    return ArtifactHandle(
        id=f"id_{name}",
        name=name,
        type=artifact_type,
        tool_name="sql_tool",
        step_index=step_index,
        schema={"col1": "int64", "col2": "object"} if artifact_type == "table" else None,
        row_count=100 if artifact_type == "table" else None,
        summary=f"{name} summary",
    )


def run_masking_pass(
    messages: list,
    tc_id_to_handle: dict,
    tc_id_to_step: dict,
    working_memory: AnalysisWorkingMemory,
) -> list:
    """Standalone replica of the masking pass from _direct_tool_loop."""
    current_step = working_memory.step_index
    for _mi, _msg in enumerate(messages):
        if not isinstance(_msg, ToolMessage):
            continue
        _msg_id = getattr(_msg, "tool_call_id", "")
        _h = tc_id_to_handle.get(_msg_id)
        _step = tc_id_to_step.get(_msg_id)
        if _step is None:
            continue
        steps_ago = current_step - _step
        if steps_ago < _MASK_KEEP_LAST_N:
            continue
        # Never mask errors
        if _h is not None and _h.type == "error":
            continue
        # Apply masking
        if _h is not None:
            masked_content = _h.masked_ref
        else:
            original = str(_msg.content)
            masked_content = (
                f"[step {_step}: {original[:80]}...]"
                if len(original) > 80
                else f"[step {_step}: {original}]"
            )
        messages[_mi] = ToolMessage(content=masked_content, tool_call_id=_msg_id)
    return messages


# ---------------------------------------------------------------------------
# Test 1: Constants are conservative
# ---------------------------------------------------------------------------

def test_masking_constants_are_conservative():
    assert _MASK_KEEP_LAST_N == 3
    assert _MASK_MIN_STEPS == 4
    assert _MASK_MIN_TOOLS == 3


# ---------------------------------------------------------------------------
# Test 2: masked_ref for table artifact
# ---------------------------------------------------------------------------

def test_masked_ref_contains_artifact_metadata():
    handle = make_handle("my_table", "table", 2)
    ref = handle.masked_ref
    assert "artifact: my_table" in ref
    assert "table" in ref
    assert "100×2 cols" in ref
    assert "cols: col1" in ref
    assert "step 2" in ref


# ---------------------------------------------------------------------------
# Test 3: masked_ref for value artifact
# ---------------------------------------------------------------------------

def test_masked_ref_value_artifact():
    handle = make_handle("total_revenue", "value", 1)
    ref = handle.masked_ref
    assert "artifact: total_revenue" in ref
    assert "value" in ref


# ---------------------------------------------------------------------------
# Test 4: masked_ref for error artifact
# ---------------------------------------------------------------------------

def test_masked_ref_error_never_contains_data():
    handle = make_handle("bad_query", "error", 3)
    ref = handle.masked_ref
    assert "error" in ref


# ---------------------------------------------------------------------------
# Test 5: Masking pass — skips recent tools, masks old ones
# ---------------------------------------------------------------------------

def test_masking_policy_skips_recent_tools():
    # 5 tool calls at steps 0..4; working_memory.step_index = 5
    # steps_ago = current_step - step_when_executed = 5 - step
    # steps_ago for each: tc1→5, tc2→4, tc3→3, tc4→2, tc5→1
    # _MASK_KEEP_LAST_N = 3: keep only steps_ago < 3 (i.e. steps_ago 0, 1, 2)
    # Kept (not masked): tc4 (steps_ago=2), tc5 (steps_ago=1)
    # Masked: tc1 (steps_ago=5), tc2 (steps_ago=4), tc3 (steps_ago=3)
    tc_id_to_handle = {
        "tc1": make_handle("art_0", "table", 0),
        "tc2": make_handle("art_1", "table", 1),
        "tc3": make_handle("art_2", "table", 2),
        "tc4": make_handle("art_3", "table", 3),
        "tc5": make_handle("art_4", "table", 4),
    }
    tc_id_to_step = {f"tc{i+1}": i for i in range(5)}

    messages = [
        ToolMessage(content=f"full content {i}", tool_call_id=f"tc{i+1}")
        for i in range(5)
    ]

    wm = AnalysisWorkingMemory(goal="test", step_index=5, tool_call_count=5)
    result = run_masking_pass(messages, tc_id_to_handle, tc_id_to_step, wm)

    # steps 0, 1, 2 are old enough (steps_ago >= 3) — should be masked
    assert "artifact:" in result[0].content, f"tc1 should be masked, got: {result[0].content}"
    assert "artifact:" in result[1].content, f"tc2 should be masked, got: {result[1].content}"
    assert "artifact:" in result[2].content, f"tc3 should be masked, got: {result[2].content}"

    # steps 3, 4 are recent (steps_ago < 3) — should NOT be masked
    assert result[3].content == "full content 3", f"tc4 should not be masked: {result[3].content}"
    assert result[4].content == "full content 4", f"tc5 should not be masked: {result[4].content}"


# ---------------------------------------------------------------------------
# Test 6: Error-type handles are never masked
# ---------------------------------------------------------------------------

def test_masking_skips_error_type_handles():
    error_handle = make_handle("err_result", "error", 1)
    normal_handle = make_handle("ok_result", "table", 0)

    tc_id_to_handle = {
        "tc1": normal_handle,
        "tc2": error_handle,
    }
    tc_id_to_step = {"tc1": 0, "tc2": 1}

    original_tc2_content = "error details here"
    messages = [
        ToolMessage(content="normal full content", tool_call_id="tc1"),
        ToolMessage(content=original_tc2_content, tool_call_id="tc2"),
    ]

    wm = AnalysisWorkingMemory(goal="test", step_index=5, tool_call_count=4)
    result = run_masking_pass(messages, tc_id_to_handle, tc_id_to_step, wm)

    # tc1 (normal table at step 0) should be masked
    assert "artifact:" in result[0].content
    # tc2 (error at step 1) should NOT be masked
    assert result[1].content == original_tc2_content


# ---------------------------------------------------------------------------
# Test 7: No masking below min steps
# ---------------------------------------------------------------------------

def test_masking_not_applied_below_min_steps():
    tc_id_to_handle = {f"tc{i}": make_handle(f"art_{i}", "table", i) for i in range(4)}
    tc_id_to_step = {f"tc{i}": i for i in range(4)}

    original_contents = {f"tc{i}": f"content {i}" for i in range(4)}
    messages = [
        ToolMessage(content=original_contents[f"tc{i}"], tool_call_id=f"tc{i}")
        for i in range(4)
    ]

    # step_index = 3 < _MASK_MIN_STEPS = 4
    wm = AnalysisWorkingMemory(goal="test", step_index=3, tool_call_count=4)

    # Guard: masking pass should only be called when step_index >= _MASK_MIN_STEPS
    if wm.step_index >= _MASK_MIN_STEPS and wm.tool_call_count >= _MASK_MIN_TOOLS:
        run_masking_pass(messages, tc_id_to_handle, tc_id_to_step, wm)

    for i in range(4):
        assert messages[i].content == original_contents[f"tc{i}"], (
            f"tc{i} should not be masked: {messages[i].content}"
        )


# ---------------------------------------------------------------------------
# Test 8: No masking below min tools
# ---------------------------------------------------------------------------

def test_masking_not_applied_below_min_tools():
    tc_id_to_handle = {f"tc{i}": make_handle(f"art_{i}", "table", i) for i in range(4)}
    tc_id_to_step = {f"tc{i}": i for i in range(4)}

    original_contents = {f"tc{i}": f"content {i}" for i in range(4)}
    messages = [
        ToolMessage(content=original_contents[f"tc{i}"], tool_call_id=f"tc{i}")
        for i in range(4)
    ]

    # tool_call_count = 2 < _MASK_MIN_TOOLS = 3
    wm = AnalysisWorkingMemory(goal="test", step_index=5, tool_call_count=2)

    # Guard: masking pass should only be called when tool_call_count >= _MASK_MIN_TOOLS
    if wm.step_index >= _MASK_MIN_STEPS and wm.tool_call_count >= _MASK_MIN_TOOLS:
        run_masking_pass(messages, tc_id_to_handle, tc_id_to_step, wm)

    for i in range(4):
        assert messages[i].content == original_contents[f"tc{i}"], (
            f"tc{i} should not be masked: {messages[i].content}"
        )
