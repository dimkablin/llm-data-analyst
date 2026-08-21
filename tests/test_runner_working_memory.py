"""Tests for AnalysisWorkingMemory wiring in runner.py (Task 3).

These tests exercise user-facing behavior — the shape of data returned by
_build_tool_message_text and correct accumulation in AnalysisWorkingMemory —
without spinning up the full agent graph or an LLM.
"""

from __future__ import annotations

import unittest
from typing import ClassVar

import pandas as pd

from backend.agent.tool_loop import _build_tool_message_text
from backend.agent.working_memory import AnalysisWorkingMemory, ArtifactHandle


def _table_result(name: str, df: pd.DataFrame):
    """Build a fake tool result that looks like a table artifact."""

    class _Res:
        content = f"Query returned {len(df)} rows."
        artifact: ClassVar[dict] = {
            "artifact_type": "table",
            "items": {name: df},
        }

    return _Res()


def _value_result(name: str, value):
    class _Res:
        content = f"Computed value: {value}"
        artifact: ClassVar[dict] = {
            "artifact_type": "value",
            "items": {name: value},
        }

    return _Res()


def _plot_result(name: str):
    class _Res:
        content = "Plot created."
        artifact: ClassVar[dict] = {
            "artifact_type": "plot",
            "items": {name: "<fig>"},
        }

    return _Res()


def _artifact_result(kind: str, name: str, payload, meta: dict):
    class _Res:
        content = f"{kind} created."
        artifact: ClassVar[dict] = {
            "artifact_type": kind,
            "items": {name: payload},
            "meta": meta,
        }

    return _Res()


class TestBuildToolMessageTextReturnsTuple(unittest.TestCase):
    """test_build_tool_message_text_returns_tuple"""

    def setUp(self):
        self.df = pd.DataFrame({"region": ["A", "B", "C"], "revenue": [100.0, 200.0, 300.0]})
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

    def test_multi_table_only_first_handle_returned(self):
        """When a tool returns multiple tables, only the first gets a handle (current behaviour)."""
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"c": [5, 6]})
        result = {
            "schema_version": "1.0",
            "artifact_type": "table",
            "items": {"first_table": df1, "second_table": df2},
        }

        # Wrap as a mock result object
        class _R:
            content = ""
            artifact = result

        _, handle = _build_tool_message_text(_R())
        self.assertIsNotNone(handle)
        self.assertEqual(handle.name, "first_table")  # documents that only first table gets a handle

    def test_bundled_artifact_returns_separate_typed_handles(self):
        class _R:
            content = "Forecast ready."
            artifact: ClassVar[dict] = {
                "artifact_type": "json",
                "items": {"forecast": {"rows": []}},
                "table": {"forecast_rows": pd.DataFrame({"y": [12.5]})},
                "plot": {"forecast_plot": "<fig>"},
            }

        text, handles = _build_tool_message_text(_R())

        self.assertIsInstance(handles, list)
        self.assertEqual(
            [(handle.name, handle.type) for handle in handles],
            [
                ("forecast", "json"),
                ("forecast_rows", "table"),
                ("forecast_plot", "plot"),
            ],
        )
        self.assertIn("AVAILABLE_ARTIFACT_HANDLES", text)

    def test_table_preview_contains_only_published_dataframe_rows(self):
        text, _ = _build_tool_message_text(self.result)

        self.assertNotIn("numeric_summary_rows_appended", text)
        self.assertNotIn("__sum__", text)
        self.assertNotIn("__mean__", text)
        self.assertIn("100.0", text)
        self.assertIn("300.0", text)

    def test_table_observation_includes_low_cardinality_values(self):
        data = pd.DataFrame(
            {
                "branch": ["North", "South", "Overall", "North"],
                "section": ["month", "month", "month", "month"],
                "value": [1.0, 2.0, 3.0, 4.0],
            }
        )

        text, _ = _build_tool_message_text(_table_result("member_values", data))

        self.assertIn("low_cardinality_values_in_result", text)
        self.assertIn('\"branch\": [', text)
        self.assertIn('\"Overall\"', text)
        self.assertIn('\"section\": [', text)
        self.assertIn('\"month\"', text)

    def test_table_observation_profiles_all_null_and_constant_columns(self):
        data = pd.DataFrame(
            {
                "branch": ["A"] * 40 + ["B"] * 40,
                "score": [None] * 80,
                "complaints": [0.0] * 80,
                "value": list(range(80)),
            }
        )

        text, _ = _build_tool_message_text(_table_result("joined_values", data))

        self.assertIn("column_value_profile", text)
        self.assertIn('\"all_null_columns\": [', text)
        self.assertIn('\"score\"', text)
        self.assertIn('\"constant_columns\": {', text)
        self.assertIn('\"complaints\": \"0.0\"', text)
        self.assertNotIn('\"value\": \"0\"', text)
        self.assertIn('\"A\"', text)
        self.assertIn('\"B\"', text)
        self.assertIn("cannot satisfy the requested grouping dimension", text)
        self.assertIn("do not relabel or repeat", text)
        self.assertIn("peer numeric columns", text)
        self.assertIn("dimension/value rows", text)

    def test_empty_table_tells_model_to_change_query_strategy(self):
        text, _ = _build_tool_message_text(_table_result("empty_result", self.df.iloc[:0]))

        self.assertIn("EMPTY_RESULT", text)
        self.assertIn("SELECT DISTINCT", text)
        self.assertIn("next planned source", text)
        self.assertIn("replan once", text)
        self.assertIn("Do not repeat an equivalent query", text)

    def test_truncated_table_tells_model_to_aggregate_in_sql(self):
        result = _artifact_result(
            "table",
            "limited_rows",
            self.df,
            {
                "query": {
                    "max_rows": 200,
                    "returned_rows": 200,
                    "truncated": True,
                }
            },
        )

        text, handle = _build_tool_message_text(result)

        self.assertIn("TRUNCATED_RESULT", text)
        self.assertIn("Do not analyze this preview", text)
        self.assertIn("aggregate to the final requested grain in sql", text.lower())
        self.assertIn("artifact_name does not change", text)
        self.assertIn("explicit time bucket", text)
        self.assertIn("SELECT and GROUP BY", text)
        self.assertIn("truncated", str(handle.summary).lower())

    def test_bounded_table_tells_model_not_to_treat_limit_as_complete(self):
        result = _artifact_result(
            "table",
            "top_rows",
            self.df,
            {
                "query": {
                    "requested_limit": 2,
                    "returned_rows": 2,
                    "truncated": False,
                    "has_more_rows": True,
                }
            },
        )

        text, handle = _build_tool_message_text(result)

        self.assertIn("BOUNDED_RESULT", text)
        self.assertIn("exact top-N", text)
        self.assertIn("do not increase LIMIT incrementally", text)
        self.assertIn("bounded", str(handle.summary).lower())


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


class TestBuildToolMessageTextArtifactMeta(unittest.TestCase):
    """Artifact meta should be available to the next LLM turn for any artifact type."""

    def test_json_artifact_meta_is_forwarded(self):
        result = _artifact_result(
            "json",
            "search_results",
            {"items": [{"title": "A"}]},
            {
                "search": {
                    "summary": "Search completed",
                    "warnings": ["low recall"],
                    "metrics": {"result_count": 1},
                }
            },
        )

        text, _ = _build_tool_message_text(result)

        self.assertIn("TOOL_RESULT_CONTEXT_FOR_LLM", text)
        self.assertIn("Search completed", text)
        self.assertIn("low recall", text)
        self.assertIn("JSON_RESULT", text)

    def test_plot_artifact_meta_is_forwarded(self):
        result = _artifact_result(
            "plot",
            "revenue_chart",
            {"data": [], "layout": {}},
            {
                "chart_recipe": {
                    "summary": "Revenue chart built",
                    "params": {"x": "month", "y": "revenue"},
                }
            },
        )

        text, _ = _build_tool_message_text(result)

        self.assertIn("TOOL_RESULT_CONTEXT_FOR_LLM", text)
        self.assertIn("Revenue chart built", text)
        self.assertIn("revenue_chart", text)
        self.assertIn("PLOT_RESULT", text)


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
        mem.current_plan = [line.strip() for line in plan_text.splitlines() if line.strip()]
        self.assertEqual(len(mem.current_plan), 3)
        self.assertEqual(mem.current_plan[0], "Step 1: load data")
        self.assertEqual(mem.current_plan[2], "Step 3: plot results")

    def test_empty_lines_filtered_out(self):
        mem = AnalysisWorkingMemory(goal="x")
        plan_text = "Step 1\n\n  \nStep 2"
        mem.current_plan = [line.strip() for line in plan_text.splitlines() if line.strip()]
        self.assertEqual(len(mem.current_plan), 2)


class TestWorkingMemoryReplannerReplacesPlan(unittest.TestCase):
    """test_working_memory_replanner_replaces_plan"""

    def test_plan_replaced_not_merged(self):
        mem = AnalysisWorkingMemory(goal="analyse churn")
        plan_text_1 = "Step A\nStep B\nStep C"
        mem.current_plan = [line.strip() for line in plan_text_1.splitlines() if line.strip()]
        self.assertEqual(len(mem.current_plan), 3)

        plan_text_2 = "New Step X\nNew Step Y"
        mem.current_plan = [line.strip() for line in plan_text_2.splitlines() if line.strip()]

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
