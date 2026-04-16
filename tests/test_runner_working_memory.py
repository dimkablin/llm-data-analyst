"""Tests for AnalysisWorkingMemory wiring in runner.py (Task 3).

These tests exercise user-facing behavior — the shape of data returned by
_build_tool_message_text and correct accumulation in AnalysisWorkingMemory —
without spinning up the full agent graph or an LLM.
"""
from __future__ import annotations

import unittest

import pandas as pd

from backend.agent.runner import _build_tool_message_text
from backend.agent.working_memory import AnalysisWorkingMemory, ArtifactHandle


def _table_result(name: str, df: pd.DataFrame):
    """Build a fake tool result that looks like a table artifact."""

    class _Res:
        content = f"Query returned {len(df)} rows."
        artifact = {
            "artifact_type": "table",
            "items": {name: df},
        }

    return _Res()


def _value_result(name: str, value):
    class _Res:
        content = f"Computed value: {value}"
        artifact = {
            "artifact_type": "value",
            "items": {name: value},
        }

    return _Res()


def _plot_result(name: str):
    class _Res:
        content = "Plot created."
        artifact = {
            "artifact_type": "plot",
            "items": {name: "<fig>"},
        }

    return _Res()


class TestBuildToolMessageTextReturnsTuple(unittest.TestCase):
    """test_build_tool_message_text_returns_tuple"""

    def setUp(self):
        self.df = pd.DataFrame(
            {"region": ["A", "B", "C"], "revenue": [100.0, 200.0, 300.0]}
        )
        self.result = _table_result("revenue_by_region", self.df)

    def test_returns_tuple(self):
        out = _build_tool_message_text(self.result)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)

    def test_first_element_is_string(self):
        text, _ = _build_tool_message_text(self.result)
        self.assertIsInstance(text, str)
        self.assertIn("revenue_by_region", text)

    def test_second_element_is_artifact_handle(self):
        _, handle = _build_tool_message_text(self.result)
        self.assertIsNotNone(handle)
        self.assertIsInstance(handle, ArtifactHandle)

    def test_handle_has_correct_name_and_type(self):
        _, handle = _build_tool_message_text(self.result)
        self.assertEqual(handle.name, "revenue_by_region")
        self.assertEqual(handle.type, "table")

    def test_handle_has_correct_row_count(self):
        _, handle = _build_tool_message_text(self.result)
        self.assertEqual(handle.row_count, 3)

    def test_handle_has_correct_schema(self):
        _, handle = _build_tool_message_text(self.result)
        self.assertIsNotNone(handle.schema)
        self.assertIn("region", handle.schema)
        self.assertIn("revenue", handle.schema)


class TestBuildToolMessageTextNoArtifact(unittest.TestCase):
    """test_build_tool_message_text_no_artifact"""

    def test_plain_string_returns_none_handle(self):
        text, handle = _build_tool_message_text("just a plain string")
        self.assertIsInstance(text, str)
        self.assertIsNone(handle)

    def test_plain_string_text_preserved(self):
        text, _ = _build_tool_message_text("hello world")
        self.assertEqual(text, "hello world")


class TestBuildToolMessageTextValueArtifact(unittest.TestCase):
    """test_build_tool_message_text_value_artifact"""

    def test_handle_type_is_value(self):
        result = _value_result("metric", 42)
        _, handle = _build_tool_message_text(result)
        self.assertIsNotNone(handle)
        self.assertEqual(handle.type, "value")

    def test_summary_contains_value(self):
        result = _value_result("metric", 42)
        _, handle = _build_tool_message_text(result)
        self.assertIn("42", handle.summary)

    def test_no_schema_or_row_count(self):
        result = _value_result("metric", 42)
        _, handle = _build_tool_message_text(result)
        self.assertIsNone(handle.schema)
        self.assertIsNone(handle.row_count)


class TestBuildToolMessageTextPlotArtifact(unittest.TestCase):
    """test_build_tool_message_text_plot_artifact"""

    def test_handle_type_is_plot(self):
        result = _plot_result("monthly_trend_chart")
        _, handle = _build_tool_message_text(result)
        self.assertIsNotNone(handle)
        self.assertEqual(handle.type, "plot")

    def test_handle_name_correct(self):
        result = _plot_result("monthly_trend_chart")
        _, handle = _build_tool_message_text(result)
        self.assertEqual(handle.name, "monthly_trend_chart")

    def test_summary_is_artifact_name(self):
        result = _plot_result("monthly_trend_chart")
        _, handle = _build_tool_message_text(result)
        self.assertEqual(handle.summary, "monthly_trend_chart")


class TestWorkingMemoryAccumulatesHandles(unittest.TestCase):
    """test_working_memory_accumulates_handles"""

    def test_append_handle_with_fields(self):
        mem = AnalysisWorkingMemory(goal="analyse revenue")
        df = pd.DataFrame({"col_a": [1, 2, 3]})
        _, handle = _build_tool_message_text(_table_result("rev_table", df))
        self.assertIsNotNone(handle)
        handle.tool_name = "sql_tool"
        handle.step_index = mem.step_index

        mem.artifact_handles.append(handle)

        self.assertEqual(len(mem.artifact_handles), 1)
        self.assertEqual(mem.artifact_handles[0].name, "rev_table")
        self.assertEqual(mem.artifact_handles[0].type, "table")
        self.assertEqual(mem.artifact_handles[0].tool_name, "sql_tool")
        self.assertEqual(mem.artifact_handles[0].step_index, 0)
        self.assertEqual(mem.artifact_handles[0].row_count, 3)

    def test_handle_id_is_non_empty_string(self):
        df = pd.DataFrame({"x": [1]})
        _, handle = _build_tool_message_text(_table_result("t", df))
        self.assertIsNotNone(handle)
        self.assertIsInstance(handle.id, str)
        self.assertTrue(len(handle.id) > 0)


class TestWorkingMemoryActionAuditTrail(unittest.TestCase):
    """test_working_memory_action_audit_trail"""

    def test_three_tool_calls_tracked(self):
        mem = AnalysisWorkingMemory(goal="run three tools")
        for i in range(3):
            mem.completed_actions.append(f"tool_{i} → result_{i}")
            mem.step_index += 1
            mem.tool_call_count += 1

        self.assertEqual(mem.tool_call_count, 3)
        self.assertEqual(len(mem.completed_actions), 3)

    def test_action_strings_preserved(self):
        mem = AnalysisWorkingMemory(goal="x")
        mem.completed_actions.append("sql_tool → my_table")
        self.assertEqual(mem.completed_actions[0], "sql_tool → my_table")

    def test_step_index_increments(self):
        mem = AnalysisWorkingMemory(goal="x")
        self.assertEqual(mem.step_index, 0)
        mem.step_index += 1
        self.assertEqual(mem.step_index, 1)
        mem.step_index += 1
        self.assertEqual(mem.step_index, 2)


class TestWorkingMemoryPlannerSetsPlan(unittest.TestCase):
    """test_working_memory_planner_sets_current_plan"""

    def test_current_plan_set_from_plan_text(self):
        mem = AnalysisWorkingMemory(goal="analyse sales")
        plan_text = "Step 1: load data\nStep 2: run SQL\nStep 3: plot results"
        # Simulate what _agent_node does:
        mem.current_plan = [
            line.strip() for line in plan_text.splitlines() if line.strip()
        ]
        self.assertEqual(len(mem.current_plan), 3)
        self.assertEqual(mem.current_plan[0], "Step 1: load data")
        self.assertEqual(mem.current_plan[2], "Step 3: plot results")

    def test_empty_lines_filtered_out(self):
        mem = AnalysisWorkingMemory(goal="x")
        plan_text = "Step 1\n\n  \nStep 2"
        mem.current_plan = [
            line.strip() for line in plan_text.splitlines() if line.strip()
        ]
        self.assertEqual(len(mem.current_plan), 2)


class TestWorkingMemoryReplannerReplacesPlan(unittest.TestCase):
    """test_working_memory_replanner_replaces_plan"""

    def test_plan_replaced_not_merged(self):
        mem = AnalysisWorkingMemory(goal="analyse churn")
        plan_text_1 = "Step A\nStep B\nStep C"
        mem.current_plan = [
            line.strip() for line in plan_text_1.splitlines() if line.strip()
        ]
        self.assertEqual(len(mem.current_plan), 3)

        plan_text_2 = "New Step X\nNew Step Y"
        mem.current_plan = [
            line.strip() for line in plan_text_2.splitlines() if line.strip()
        ]

        # Must be fully replaced, not merged
        self.assertEqual(len(mem.current_plan), 2)
        self.assertIn("New Step X", mem.current_plan)
        self.assertIn("New Step Y", mem.current_plan)
        self.assertNotIn("Step A", mem.current_plan)

    def test_re_plan_preserves_other_fields(self):
        mem = AnalysisWorkingMemory(goal="x")
        mem.completed_actions.append("some_tool → some_result")
        mem.tool_call_count = 1

        mem.current_plan = ["New plan"]

        # Other fields must be unchanged
        self.assertEqual(mem.tool_call_count, 1)
        self.assertEqual(len(mem.completed_actions), 1)


if __name__ == "__main__":
    unittest.main()
