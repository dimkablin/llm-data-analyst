from __future__ import annotations

import unittest

from backend.agent.working_memory import ArtifactHandle, AnalysisWorkingMemory
from backend.sessions.session_memory import (
    SessionArtifactRef,
    StructuredSessionMemory,
    SessionMemory,
)


def _make_handle(**kwargs) -> ArtifactHandle:
    defaults = dict(
        id="art-1",
        name="my_artifact",
        type="table",
        tool_name="sql_tool",
        step_index=1,
        schema=None,
        row_count=None,
        summary=None,
    )
    defaults.update(kwargs)
    return ArtifactHandle(**defaults)


def _make_ref(**kwargs) -> SessionArtifactRef:
    defaults = dict(
        id="art-1",
        name="my_artifact",
        type="table",
        turn_index=0,
        schema=None,
        row_count=None,
        summary=None,
    )
    defaults.update(kwargs)
    return SessionArtifactRef(**defaults)


class TestArtifactHandleMaskedRef(unittest.TestCase):
    def test_table_with_schema_and_row_count(self) -> None:
        handle = _make_handle(
            name="revenue_by_region",
            type="table",
            step_index=2,
            schema={"region": "str", "revenue": "float", "growth_pct": "float", "rank": "int", "period": "str"},
            row_count=1200,
            summary=None,
        )
        ref = handle.masked_ref
        self.assertIn("artifact: revenue_by_region", ref)
        self.assertIn("table", ref)
        self.assertIn("1200×5 cols", ref)
        self.assertIn("cols: region, revenue, growth_pct, rank, period", ref)
        self.assertIn("step 2", ref)
        self.assertTrue(ref.startswith("["))
        self.assertTrue(ref.endswith("]"))

    def test_table_caps_col_display_at_five(self) -> None:
        schema = {f"col{i}": "str" for i in range(8)}
        handle = _make_handle(
            name="wide_table",
            type="table",
            step_index=1,
            schema=schema,
            row_count=50,
            summary=None,
        )
        ref = handle.masked_ref
        # should show only first 5 col names
        cols_part = [p for p in ref.split(" | ") if p.startswith("cols:")]
        self.assertEqual(len(cols_part), 1)
        listed_cols = cols_part[0].replace("cols: ", "").split(", ")
        self.assertEqual(len(listed_cols), 5)

    def test_plot_type_with_summary(self) -> None:
        handle = _make_handle(
            name="monthly_trend_chart",
            type="plot",
            step_index=3,
            schema=None,
            row_count=None,
            summary="Revenue by month 2024",
        )
        ref = handle.masked_ref
        self.assertIn("artifact: monthly_trend_chart", ref)
        self.assertIn("plot", ref)
        self.assertIn("Revenue by month 2024", ref)
        self.assertIn("step 3", ref)
        self.assertNotIn("cols:", ref)

    def test_value_type_minimal(self) -> None:
        handle = _make_handle(
            name="total_revenue",
            type="value",
            step_index=1,
            schema=None,
            row_count=None,
            summary=None,
        )
        ref = handle.masked_ref
        self.assertEqual(ref, "[artifact: total_revenue | value | step 1]")

    def test_error_type(self) -> None:
        handle = _make_handle(
            name="failed_query",
            type="error",
            step_index=4,
            schema=None,
            row_count=None,
            summary="SQL syntax error",
        )
        ref = handle.masked_ref
        self.assertIn("artifact: failed_query", ref)
        self.assertIn("error", ref)
        self.assertIn("SQL syntax error", ref)
        self.assertIn("step 4", ref)

    def test_no_schema_no_summary_graceful(self) -> None:
        handle = _make_handle(
            name="orphan",
            type="table",
            step_index=5,
            schema=None,
            row_count=None,
            summary=None,
        )
        ref = handle.masked_ref
        # Should not crash and should form a valid bracket string
        self.assertTrue(ref.startswith("["))
        self.assertTrue(ref.endswith("]"))
        self.assertIn("orphan", ref)
        self.assertIn("step 5", ref)
        self.assertNotIn("cols:", ref)
        self.assertNotIn("rows", ref)

    def test_row_count_without_schema_shows_rows(self) -> None:
        handle = _make_handle(
            name="count_result",
            type="table",
            step_index=2,
            schema=None,
            row_count=42,
            summary=None,
        )
        ref = handle.masked_ref
        self.assertIn("42 rows", ref)
        self.assertNotIn("cols:", ref)


class TestStructuredSessionMemoryIsEmpty(unittest.TestCase):
    def test_empty_default(self) -> None:
        mem = StructuredSessionMemory()
        self.assertTrue(mem.is_empty())

    def test_whitespace_notes_still_empty(self) -> None:
        mem = StructuredSessionMemory(notes="   \n  ")
        self.assertTrue(mem.is_empty())

    def test_not_empty_with_notes(self) -> None:
        mem = StructuredSessionMemory(notes="some note")
        self.assertFalse(mem.is_empty())

    def test_not_empty_with_artifact_index(self) -> None:
        mem = StructuredSessionMemory(artifact_index=[_make_ref()])
        self.assertFalse(mem.is_empty())

    def test_not_empty_with_key_findings(self) -> None:
        mem = StructuredSessionMemory(key_findings=["revenue grew 10%"])
        self.assertFalse(mem.is_empty())


class TestStructuredSessionMemoryBuildBlock(unittest.TestCase):
    def test_empty_returns_empty_string(self) -> None:
        mem = StructuredSessionMemory()
        self.assertEqual(mem.build_block(), "")

    def test_notes_only(self) -> None:
        mem = StructuredSessionMemory(notes="Data loaded from sales DB.")
        block = mem.build_block()
        self.assertIn("## Session notes", block)
        self.assertIn("Data loaded from sales DB.", block)
        self.assertNotIn("## Key findings", block)
        self.assertNotIn("## Artifacts", block)

    def test_artifact_index_only_compact_ref_format(self) -> None:
        ref = _make_ref(
            name="orders_table",
            type="table",
            schema={"order_id": "int", "amount": "float"},
            row_count=500,
            summary="Orders Q1",
        )
        mem = StructuredSessionMemory(artifact_index=[ref])
        block = mem.build_block()
        self.assertIn("## Artifacts from this session", block)
        self.assertIn("orders_table", block)
        self.assertIn("table, 500×2", block)
        self.assertIn("Orders Q1", block)
        self.assertNotIn("## Session notes", block)

    def test_key_findings_only(self) -> None:
        mem = StructuredSessionMemory(key_findings=["Revenue up 15%", "Churn down 3%"])
        block = mem.build_block()
        self.assertIn("## Key findings from this session", block)
        self.assertIn("- Revenue up 15%", block)
        self.assertIn("- Churn down 3%", block)
        self.assertNotIn("## Session notes", block)

    def test_key_findings_capped_at_last_10(self) -> None:
        findings = [f"finding_{i:03d}" for i in range(15)]
        mem = StructuredSessionMemory(key_findings=findings)
        block = mem.build_block()
        # Last 10 findings (5–14) must appear; first 5 (0–4) must not
        for i in range(5, 15):
            self.assertIn(f"finding_{i:03d}", block)
        for i in range(5):
            self.assertNotIn(f"finding_{i:03d}", block)

    def test_artifact_index_capped_at_last_20(self) -> None:
        refs = [_make_ref(id=f"art-{i:03d}", name=f"artifact_{i:03d}") for i in range(25)]
        mem = StructuredSessionMemory(artifact_index=refs)
        block = mem.build_block()
        # Last 20 artifacts (5–24) must appear; first 5 (0–4) must not
        for i in range(5, 25):
            self.assertIn(f"artifact_{i:03d}", block)
        for i in range(5):
            self.assertNotIn(f"artifact_{i:03d}", block)

    def test_all_sections_present(self) -> None:
        mem = StructuredSessionMemory(
            notes="Session note here",
            key_findings=["Finding A"],
            artifact_index=[_make_ref(name="my_table", type="table")],
        )
        block = mem.build_block()
        self.assertIn("## Session notes", block)
        self.assertIn("## Key findings from this session", block)
        self.assertIn("## Artifacts from this session", block)


class TestSessionMemoryAlias(unittest.TestCase):
    def test_session_memory_alias_imports_correctly(self) -> None:
        self.assertIs(SessionMemory, StructuredSessionMemory)

    def test_session_memory_alias_is_functional(self) -> None:
        mem = SessionMemory(notes="alias works")
        self.assertFalse(mem.is_empty())
        self.assertIsInstance(mem, StructuredSessionMemory)


class TestAnalysisWorkingMemoryDefaults(unittest.TestCase):
    def test_defaults_are_correct(self) -> None:
        mem = AnalysisWorkingMemory(goal="show top products")
        self.assertEqual(mem.goal, "show top products")
        self.assertEqual(mem.step_index, 0)
        self.assertEqual(mem.artifact_handles, [])
        self.assertEqual(mem.sandbox_var_names, [])
        self.assertEqual(mem.tool_call_count, 0)
        self.assertEqual(mem.current_plan, [])
        self.assertEqual(mem.completed_actions, [])
        self.assertEqual(mem.last_tool_result_summary, "")

    def test_current_plan_starts_as_empty_list(self) -> None:
        mem = AnalysisWorkingMemory(goal="x")
        self.assertIsInstance(mem.current_plan, list)
        self.assertEqual(len(mem.current_plan), 0)

    def test_mutable_defaults_are_independent(self) -> None:
        mem1 = AnalysisWorkingMemory(goal="a")
        mem2 = AnalysisWorkingMemory(goal="b")
        mem1.artifact_handles.append(_make_handle())
        self.assertEqual(len(mem2.artifact_handles), 0)


if __name__ == "__main__":
    unittest.main()
