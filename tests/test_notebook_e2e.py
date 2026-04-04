"""End-to-end tests for the notebook-first architecture.

Covers the full lifecycle: models → store → orchestrator → cell builder →
manifest → kernel manager.  No LLM or HTTP server needed — all tests
run against in-memory and temp-directory state.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backend.notebook.cell_builder import (
    build_cell_from_tool_result,
    build_preamble_cell,
    build_source_binding_cell,
)
from backend.notebook.manifest_store import ManifestStore
from backend.notebook.models import (
    CellMetadata,
    CellOutput,
    NotebookCell,
    NotebookDocument,
    utcnow_iso,
)
from backend.notebook.orchestrator import (
    CellOp,
    NotebookEdit,
    NotebookOrchestrator,
)
from backend.notebook.session_source import (
    SessionManifest,
    SessionSource,
    alias_to_variable_name,
    make_source_alias,
)
from backend.notebook.store import NotebookStore

# ── 1. NotebookDocument model tests ─────────────────────────────────────────


class TestNotebookDocumentModel(unittest.TestCase):
    """NotebookDocument ↔ ipynb dict roundtrip."""

    def test_empty_notebook_roundtrip(self) -> None:
        nb = NotebookDocument(session_id="s1", created_at="2026-01-01T00:00:00Z")
        raw = nb.to_ipynb_dict()
        nb2 = NotebookDocument.from_ipynb_dict(raw)

        self.assertEqual(nb2.session_id, "s1")
        self.assertEqual(nb2.created_at, "2026-01-01T00:00:00Z")
        self.assertEqual(len(nb2.cells), 0)
        self.assertEqual(raw["nbformat"], 4)
        self.assertEqual(raw["nbformat_minor"], 5)

    def test_code_cell_roundtrip(self) -> None:
        cell = NotebookCell(
            id="c1",
            cell_type="code",
            source="x = 1 + 2",
            metadata=CellMetadata(purpose="test", produces=["x"]),
            outputs=[CellOutput(data={"text/plain": "3"})],
            execution_count=1,
        )
        nb = NotebookDocument(session_id="s1", cells=[cell])
        raw = nb.to_ipynb_dict()

        # Verify ipynb source is a list of lines.
        self.assertIsInstance(raw["cells"][0]["source"], list)

        nb2 = NotebookDocument.from_ipynb_dict(raw)
        c = nb2.cells[0]
        self.assertEqual(c.id, "c1")
        self.assertEqual(c.source, "x = 1 + 2")
        self.assertEqual(c.metadata.purpose, "test")
        self.assertEqual(c.metadata.produces, ["x"])
        self.assertEqual(c.execution_count, 1)
        self.assertEqual(c.outputs[0].data["text/plain"], "3")

    def test_markdown_cell_roundtrip(self) -> None:
        cell = NotebookCell(
            id="m1",
            cell_type="markdown",
            source="# Hello\n\nWorld",
            metadata=CellMetadata(tags=["preamble"]),
        )
        nb = NotebookDocument(session_id="s1", cells=[cell])
        nb2 = NotebookDocument.from_ipynb_dict(nb.to_ipynb_dict())

        self.assertEqual(nb2.cells[0].cell_type, "markdown")
        self.assertEqual(nb2.cells[0].source, "# Hello\n\nWorld")
        self.assertEqual(nb2.cells[0].metadata.tags, ["preamble"])

    def test_error_output_roundtrip(self) -> None:
        cell = NotebookCell(id="e1", cell_type="code", source="1/0")
        cell.set_error(ZeroDivisionError("division by zero"), execution_count=5)

        nb = NotebookDocument(session_id="s1", cells=[cell])
        nb2 = NotebookDocument.from_ipynb_dict(nb.to_ipynb_dict())
        out = nb2.cells[0].outputs[0]

        self.assertEqual(out.output_type, "error")
        self.assertEqual(out.ename, "ZeroDivisionError")
        self.assertEqual(out.evalue, "division by zero")

    def test_cell_lookup_helpers(self) -> None:
        nb = NotebookDocument(session_id="s1")
        c1 = NotebookCell(id="a", source="x=1")
        c2 = NotebookCell(id="b", source="y=2")
        nb.append_cell(c1)
        nb.append_cell(c2)

        self.assertIs(nb.cell_by_id("a"), c1)
        self.assertIsNone(nb.cell_by_id("z"))
        self.assertEqual(nb.cell_index("b"), 1)
        self.assertIsNone(nb.cell_index("z"))

    def test_source_binding_property(self) -> None:
        cell = NotebookCell(
            id="sb1",
            metadata=CellMetadata(tags=["source_binding"]),
        )
        self.assertTrue(cell.is_source_binding)

        cell2 = NotebookCell(id="sb2", metadata=CellMetadata(tags=["analysis"]))
        self.assertFalse(cell2.is_source_binding)

    def test_next_execution_count(self) -> None:
        nb = NotebookDocument(session_id="s1")
        self.assertEqual(nb.next_execution_count, 1)

        nb.append_cell(NotebookCell(id="c1", execution_count=3))
        self.assertEqual(nb.next_execution_count, 4)

    def test_ipynb_json_is_valid_json(self) -> None:
        """Verify the produced dict serializes to valid JSON."""
        nb = NotebookDocument(session_id="s1")
        nb.append_cell(NotebookCell(source="print('hello')"))
        raw_json = json.dumps(nb.to_ipynb_dict(), ensure_ascii=False)
        parsed = json.loads(raw_json)
        self.assertEqual(parsed["nbformat"], 4)


# ── 2. NotebookStore persistence tests ──────────────────────────────────────


class TestNotebookStore(unittest.TestCase):
    """NotebookStore read/write against a temp directory."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.store = NotebookStore(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_save_and_load(self) -> None:
        nb = NotebookDocument(session_id="s1", created_at="2026-01-01T00:00:00Z")
        nb.append_cell(NotebookCell(id="c1", source="x=1"))

        self.store.save("s1", nb)
        self.assertTrue(self.store.exists("s1"))

        loaded = self.store.load("s1")
        self.assertEqual(loaded.session_id, "s1")
        self.assertEqual(len(loaded.cells), 1)
        self.assertEqual(loaded.cells[0].source, "x=1")

    def test_load_nonexistent_returns_empty(self) -> None:
        nb = self.store.load("nonexistent")
        self.assertEqual(nb.session_id, "nonexistent")
        self.assertEqual(len(nb.cells), 0)

    def test_append_cell(self) -> None:
        self.store.create_empty("s1")
        cell = NotebookCell(id="c1", source="y=2")
        nb = self.store.append_cell("s1", cell)
        self.assertEqual(len(nb.cells), 1)

        # Verify persistence
        nb2 = self.store.load("s1")
        self.assertEqual(len(nb2.cells), 1)
        self.assertEqual(nb2.cells[0].source, "y=2")

    def test_delete(self) -> None:
        self.store.create_empty("s1")
        self.assertTrue(self.store.exists("s1"))
        self.store.delete("s1")
        self.assertFalse(self.store.exists("s1"))

    def test_ipynb_file_is_valid_json(self) -> None:
        """The saved file must be valid JSON and openable by Jupyter."""
        nb = NotebookDocument(session_id="s1")
        nb.append_cell(NotebookCell(source="import pandas as pd"))
        path = self.store.save("s1", nb)

        raw = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(raw["nbformat"], 4)
        self.assertIn("cells", raw)
        self.assertIn("metadata", raw)


# ── 3. NotebookOrchestrator tests ───────────────────────────────────────────


class TestNotebookOrchestrator(unittest.TestCase):
    """Orchestrator validates and applies edits."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.store = NotebookStore(self._tmpdir.name)
        self.orch = NotebookOrchestrator(self.store)
        self.store.create_empty("s1")

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_insert_cell(self) -> None:
        cell = NotebookCell(id="c1", source="x=1")
        result = self.orch.apply("s1", NotebookEdit(op=CellOp.INSERT, cell=cell))
        self.assertTrue(result.ok)
        self.assertEqual(result.cell_id, "c1")

        nb = self.store.load("s1")
        self.assertEqual(len(nb.cells), 1)

    def test_insert_at_position(self) -> None:
        self.orch.apply("s1", NotebookEdit(op=CellOp.INSERT, cell=NotebookCell(id="a", source="a")))
        self.orch.apply("s1", NotebookEdit(op=CellOp.INSERT, cell=NotebookCell(id="b", source="b")))
        self.orch.apply(
            "s1",
            NotebookEdit(op=CellOp.INSERT, cell=NotebookCell(id="mid", source="mid"), position=1),
        )

        nb = self.store.load("s1")
        ids = [c.id for c in nb.cells]
        self.assertEqual(ids, ["a", "mid", "b"])

    def test_insert_duplicate_id_fails(self) -> None:
        self.orch.apply("s1", NotebookEdit(op=CellOp.INSERT, cell=NotebookCell(id="c1", source="x")))
        result = self.orch.apply("s1", NotebookEdit(op=CellOp.INSERT, cell=NotebookCell(id="c1", source="y")))
        self.assertFalse(result.ok)
        self.assertIn("Duplicate", result.error)

    def test_update_cell(self) -> None:
        self.orch.apply("s1", NotebookEdit(op=CellOp.INSERT, cell=NotebookCell(id="c1", source="old")))
        result = self.orch.apply(
            "s1",
            NotebookEdit(op=CellOp.UPDATE, cell_id="c1", cell=NotebookCell(source="new")),
        )
        self.assertTrue(result.ok)

        nb = self.store.load("s1")
        self.assertEqual(nb.cells[0].source, "new")
        self.assertEqual(nb.cells[0].id, "c1")  # id preserved

    def test_delete_cell(self) -> None:
        self.orch.apply("s1", NotebookEdit(op=CellOp.INSERT, cell=NotebookCell(id="c1", source="x")))
        result = self.orch.apply("s1", NotebookEdit(op=CellOp.DELETE, cell_id="c1"))
        self.assertTrue(result.ok)

        nb = self.store.load("s1")
        self.assertEqual(len(nb.cells), 0)

    def test_delete_source_binding_fails(self) -> None:
        cell = NotebookCell(
            id="sb1",
            source="df = pd.read_parquet('...')",
            metadata=CellMetadata(tags=["source_binding"]),
        )
        self.orch.apply("s1", NotebookEdit(op=CellOp.INSERT, cell=cell))
        result = self.orch.apply("s1", NotebookEdit(op=CellOp.DELETE, cell_id="sb1"))

        self.assertFalse(result.ok)
        self.assertIn("source_binding", result.error)

    def test_move_cell(self) -> None:
        self.orch.apply("s1", NotebookEdit(op=CellOp.INSERT, cell=NotebookCell(id="a", source="a")))
        self.orch.apply("s1", NotebookEdit(op=CellOp.INSERT, cell=NotebookCell(id="b", source="b")))
        self.orch.apply("s1", NotebookEdit(op=CellOp.INSERT, cell=NotebookCell(id="c", source="c")))

        result = self.orch.apply("s1", NotebookEdit(op=CellOp.MOVE, cell_id="c", position=0))
        self.assertTrue(result.ok)

        nb = self.store.load("s1")
        self.assertEqual([c.id for c in nb.cells], ["c", "a", "b"])

    def test_execute_records_outputs(self) -> None:
        self.orch.apply("s1", NotebookEdit(op=CellOp.INSERT, cell=NotebookCell(id="c1", source="x=1")))

        outputs = [CellOutput(data={"text/plain": "1"})]
        result = self.orch.apply(
            "s1",
            NotebookEdit(op=CellOp.EXECUTE, cell_id="c1", outputs=outputs, execution_count=1),
        )
        self.assertTrue(result.ok)

        nb = self.store.load("s1")
        self.assertEqual(nb.cells[0].execution_count, 1)
        self.assertEqual(nb.cells[0].outputs[0].data["text/plain"], "1")

    def test_batch_apply(self) -> None:
        edits = [
            NotebookEdit(op=CellOp.INSERT, cell=NotebookCell(id="a", source="a")),
            NotebookEdit(op=CellOp.INSERT, cell=NotebookCell(id="b", source="b")),
        ]
        results = self.orch.apply_batch("s1", edits)
        self.assertTrue(all(r.ok for r in results))

        nb = self.store.load("s1")
        self.assertEqual(len(nb.cells), 2)

    def test_batch_stops_on_failure(self) -> None:
        edits = [
            NotebookEdit(op=CellOp.INSERT, cell=NotebookCell(id="a", source="a")),
            NotebookEdit(op=CellOp.DELETE, cell_id="nonexistent"),  # fails
            NotebookEdit(op=CellOp.INSERT, cell=NotebookCell(id="c", source="c")),
        ]
        results = self.orch.apply_batch("s1", edits)
        self.assertEqual(len(results), 2)  # stopped at failure
        self.assertTrue(results[0].ok)
        self.assertFalse(results[1].ok)


# ── 4. SessionSource / SessionManifest tests ────────────────────────────────


class TestSessionSource(unittest.TestCase):
    """SessionSource and SessionManifest model tests."""

    def test_source_roundtrip(self) -> None:
        src = SessionSource(
            alias="sales_csv",
            source_type="csv",
            display_name="Sales Q4",
            variable_name="sales_df",
            file_name="sales.csv",
            parquet_path="sources/sales_csv.parquet",
            schema_hint={"amount": "float64", "date": "datetime64[ns]"},
        )
        raw = src.to_dict()
        src2 = SessionSource.from_dict(raw)
        self.assertEqual(src2.alias, "sales_csv")
        self.assertEqual(src2.variable_name, "sales_df")
        self.assertEqual(src2.schema_hint["amount"], "float64")

    def test_manifest_multi_source(self) -> None:
        m = SessionManifest(session_id="s1")
        m.add_source(SessionSource(alias="csv1", source_type="csv", variable_name="df1"))
        m.add_source(SessionSource(alias="db1", source_type="db_connection", variable_name="conn1"))

        self.assertEqual(len(m.sources), 2)
        self.assertTrue(m.has_csv())
        self.assertTrue(m.has_db())
        self.assertIsNotNone(m.source_by_alias("csv1"))
        self.assertIsNone(m.source_by_alias("nonexistent"))

    def test_manifest_primary_source(self) -> None:
        m = SessionManifest(session_id="s1")
        self.assertIsNone(m.primary_source())

        m.add_source(SessionSource(alias="csv1", source_type="csv"))
        self.assertEqual(m.primary_source().alias, "csv1")

    def test_manifest_remove_source(self) -> None:
        m = SessionManifest(session_id="s1")
        m.add_source(SessionSource(alias="csv1", source_type="csv"))
        m.add_source(SessionSource(alias="csv2", source_type="csv"))

        removed = m.remove_source("csv1")
        self.assertIsNotNone(removed)
        self.assertEqual(len(m.sources), 1)

        removed2 = m.remove_source("nonexistent")
        self.assertIsNone(removed2)

    def test_manifest_add_source_replaces_existing_alias(self) -> None:
        m = SessionManifest(session_id="s1")
        m.add_source(SessionSource(alias="csv1", source_type="csv", display_name="v1"))
        m.add_source(SessionSource(alias="csv1", source_type="csv", display_name="v2"))
        self.assertEqual(len(m.sources), 1)
        self.assertEqual(m.sources[0].display_name, "v2")

    def test_manifest_roundtrip(self) -> None:
        m = SessionManifest(
            session_id="s1",
            sources=[
                SessionSource(alias="csv1", source_type="csv", variable_name="df1"),
                SessionSource(alias="db1", source_type="db_connection", variable_name="conn1"),
            ],
            selected_skill_ids=["skill1"],
        )
        m2 = SessionManifest.from_dict(m.to_dict())
        self.assertEqual(m2.session_id, "s1")
        self.assertEqual(len(m2.sources), 2)
        self.assertEqual(m2.selected_skill_ids, ["skill1"])


class TestAliasGeneration(unittest.TestCase):
    """Alias and variable name generation."""

    def test_csv_alias(self) -> None:
        self.assertEqual(make_source_alias("Sales Q4.csv", "csv", []), "sales_q4_csv")

    def test_db_alias(self) -> None:
        self.assertEqual(make_source_alias("Warehouse", "db_connection", []), "warehouse_db")

    def test_deduplication(self) -> None:
        existing = ["sales_q4_csv"]
        self.assertEqual(make_source_alias("Sales Q4.csv", "csv", existing), "sales_q4_csv_2")

    def test_triple_dedup(self) -> None:
        existing = ["sales_csv", "sales_csv_2"]
        self.assertEqual(make_source_alias("sales.csv", "csv", existing), "sales_csv_3")

    def test_empty_name(self) -> None:
        alias = make_source_alias("", "csv", [])
        self.assertEqual(alias, "source_csv")

    def test_variable_name_csv(self) -> None:
        self.assertEqual(alias_to_variable_name("sales_csv"), "sales_df")

    def test_variable_name_db(self) -> None:
        self.assertEqual(alias_to_variable_name("warehouse_db"), "warehouse_conn")

    def test_variable_name_other(self) -> None:
        self.assertEqual(alias_to_variable_name("custom"), "custom_data")


# ── 5. ManifestStore persistence tests ──────────────────────────────────────


class TestManifestStore(unittest.TestCase):
    """ManifestStore read/write."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.store = ManifestStore(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_save_and_load(self) -> None:
        m = SessionManifest(session_id="s1")
        m.add_source(SessionSource(alias="csv1", source_type="csv"))
        self.store.save("s1", m)

        self.assertTrue(self.store.exists("s1"))
        m2 = self.store.load("s1")
        self.assertEqual(m2.session_id, "s1")
        self.assertEqual(len(m2.sources), 1)

    def test_load_nonexistent(self) -> None:
        m = self.store.load("nonexistent")
        self.assertEqual(m.session_id, "nonexistent")
        self.assertEqual(len(m.sources), 0)

    def test_delete(self) -> None:
        m = SessionManifest(session_id="s1")
        self.store.save("s1", m)
        self.store.delete("s1")
        self.assertFalse(self.store.exists("s1"))


# ── 6. CellBuilder tests ───────────────────────────────────────────────────


class TestCellBuilder(unittest.TestCase):
    """Cell builder produces correct cells from tool results."""

    def test_build_from_tool_result(self) -> None:
        cell = build_cell_from_tool_result(
            code='agg = df.groupby("region")["amount"].sum()\ntool_result = {"schema_version": "1.0", "artifact_type": "table", "items": {"agg": agg}}',
            tool_name="pandas_tool",
            result={"artifact_type": "table", "items": {"agg": "df"}},
            plan_step="Aggregate revenue by region",
            execution_count=3,
            new_variables=["agg"],
        )

        self.assertEqual(cell.cell_type, "code")
        self.assertEqual(cell.metadata.tool_name, "pandas_tool")
        self.assertEqual(cell.metadata.purpose, "Aggregate revenue by region")
        self.assertEqual(cell.metadata.produces, ["agg"])
        self.assertIn("df", cell.metadata.depends_on)  # extracted from AST
        self.assertEqual(cell.execution_count, 3)
        self.assertIn("analysis", cell.metadata.tags)

        # Verify breadcrumb comments in source.
        self.assertIn("# PURPOSE: Aggregate revenue by region", cell.source)
        self.assertIn("# PRODUCES: agg", cell.source)
        self.assertIn("# DEPENDS_ON:", cell.source)

    def test_build_plotly_cell(self) -> None:
        cell = build_cell_from_tool_result(
            code='fig = px.bar(monthly, x="month", y="revenue")\ntool_result = chart.result(fig, "rev_chart")',
            tool_name="plotly_tool",
            result={"artifact_type": "plot"},
            plan_step="Revenue bar chart",
            new_variables=[],
        )
        self.assertIn("visualization", cell.metadata.tags)
        self.assertIn("monthly", cell.metadata.depends_on)

    def test_build_source_binding_cell(self) -> None:
        cell = build_source_binding_cell(
            alias="sales_csv",
            variable_name="sales_df",
            source_type="csv",
            display_name="Sales Q4",
            load_code='sales_df = pd.read_parquet("sales.parquet")',
        )
        self.assertTrue(cell.is_source_binding)
        self.assertEqual(cell.metadata.source_alias, "sales_csv")
        self.assertEqual(cell.metadata.produces, ["sales_df"])
        self.assertEqual(cell.metadata.created_by, "system")
        self.assertIn("# PURPOSE: Load source: Sales Q4", cell.source)

    def test_build_preamble_cell(self) -> None:
        cell = build_preamble_cell("s1", "Sources: sales.csv")
        self.assertEqual(cell.cell_type, "markdown")
        self.assertIn("preamble", cell.metadata.tags)
        self.assertIn("Sources: sales.csv", cell.source)

    def test_depends_on_extracts_variable_refs(self) -> None:
        cell = build_cell_from_tool_result(
            code="result = sales_df.merge(orders_df, on='id')",
            tool_name="pandas_tool",
            result=None,
            new_variables=["result"],
        )
        deps = cell.metadata.depends_on
        self.assertIn("sales_df", deps)
        self.assertIn("orders_df", deps)
        # 'result' should NOT be in depends_on (it's stored, not loaded).
        self.assertNotIn("result", deps)

    def test_builtin_names_not_in_depends_on(self) -> None:
        cell = build_cell_from_tool_result(
            code="x = len(df) + int('5')",
            tool_name="pandas_tool",
            result=None,
            new_variables=["x"],
        )
        deps = cell.metadata.depends_on
        self.assertNotIn("len", deps)
        self.assertNotIn("int", deps)
        self.assertIn("df", deps)


# ── 7. Full pipeline e2e test ───────────────────────────────────────────────


class TestFullPipelineE2E(unittest.TestCase):
    """Simulate a complete session lifecycle: create → sources → analysis → persist → restore."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.nb_store = NotebookStore(self._tmpdir.name)
        self.mf_store = ManifestStore(self._tmpdir.name)
        self.orch = NotebookOrchestrator(self.nb_store)
        self.session_id = "e2e-test-session"

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_full_session_lifecycle(self) -> None:
        sid = self.session_id

        # 1. Create empty notebook and manifest.
        nb = self.nb_store.create_empty(sid)
        manifest = SessionManifest(session_id=sid, created_at=utcnow_iso())
        self.mf_store.save(sid, manifest)

        # 2. Add preamble.
        preamble = build_preamble_cell(sid)
        self.orch.apply(sid, NotebookEdit(op=CellOp.INSERT, cell=preamble))

        # 3. Add CSV source.
        csv_source = SessionSource(
            alias="sales_csv",
            source_type="csv",
            display_name="Sales Q4",
            variable_name="sales_df",
            file_name="sales.csv",
            parquet_path="sources/sales_csv.parquet",
            schema_hint={"amount": "float64", "region": "object"},
        )
        manifest.add_source(csv_source)
        self.mf_store.save(sid, manifest)

        binding_cell = build_source_binding_cell(
            alias="sales_csv",
            variable_name="sales_df",
            source_type="csv",
            display_name="Sales Q4",
            load_code='sales_df = pd.read_parquet("sources/sales_csv.parquet")',
        )
        self.orch.apply(sid, NotebookEdit(op=CellOp.INSERT, cell=binding_cell))

        # 4. Simulate tool execution: pandas aggregation.
        analysis_cell = build_cell_from_tool_result(
            code='monthly = sales_df.groupby("month")["amount"].sum()\ntool_result = {"schema_version": "1.0", "artifact_type": "table", "items": {"monthly_revenue": monthly}}',
            tool_name="pandas_tool",
            result={"artifact_type": "table", "items": {"monthly_revenue": "df"}},
            plan_step="Monthly revenue aggregation",
            execution_count=1,
            new_variables=["monthly"],
            artifact_ids=["art-001"],
        )
        self.orch.apply(sid, NotebookEdit(op=CellOp.INSERT, cell=analysis_cell))

        # 5. Simulate tool execution: plotly chart.
        chart_cell = build_cell_from_tool_result(
            code='fig = px.line(monthly, title="Revenue Trend")\ntool_result = chart.result(fig, "revenue_trend")',
            tool_name="plotly_tool",
            result={"artifact_type": "plot", "items": {"revenue_trend": "fig"}},
            plan_step="Revenue trend chart",
            execution_count=2,
            artifact_ids=["art-002"],
        )
        self.orch.apply(sid, NotebookEdit(op=CellOp.INSERT, cell=chart_cell))

        # 6. Add DB source mid-session.
        db_source = SessionSource(
            alias="warehouse_db",
            source_type="db_connection",
            display_name="Warehouse PG",
            variable_name="warehouse_conn",
            connection_id="conn-123",
        )
        manifest.add_source(db_source)
        self.mf_store.save(sid, manifest)

        db_binding = build_source_binding_cell(
            alias="warehouse_db",
            variable_name="warehouse_conn",
            source_type="db_connection",
            display_name="Warehouse PG",
            load_code='warehouse_conn = _restore_db_connection("warehouse_db")',
        )
        self.orch.apply(sid, NotebookEdit(op=CellOp.INSERT, cell=db_binding))

        # ── VERIFY NOTEBOOK STATE ──

        nb = self.nb_store.load(sid)
        self.assertEqual(len(nb.cells), 5)

        # Preamble
        self.assertEqual(nb.cells[0].cell_type, "markdown")
        self.assertIn("preamble", nb.cells[0].metadata.tags)

        # Source binding (CSV)
        self.assertTrue(nb.cells[1].is_source_binding)
        self.assertEqual(nb.cells[1].metadata.source_alias, "sales_csv")
        self.assertEqual(nb.cells[1].metadata.produces, ["sales_df"])

        # Analysis cell
        self.assertEqual(nb.cells[2].metadata.tool_name, "pandas_tool")
        self.assertEqual(nb.cells[2].metadata.produces, ["monthly"])
        self.assertIn("sales_df", nb.cells[2].metadata.depends_on)
        self.assertEqual(nb.cells[2].metadata.artifact_refs, ["art-001"])
        self.assertIn("# PURPOSE: Monthly revenue aggregation", nb.cells[2].source)

        # Visualization cell
        self.assertEqual(nb.cells[3].metadata.tool_name, "plotly_tool")
        self.assertIn("visualization", nb.cells[3].metadata.tags)
        self.assertIn("monthly", nb.cells[3].metadata.depends_on)

        # Source binding (DB)
        self.assertTrue(nb.cells[4].is_source_binding)
        self.assertEqual(nb.cells[4].metadata.source_alias, "warehouse_db")

        # ── VERIFY MANIFEST ──

        m = self.mf_store.load(sid)
        self.assertEqual(len(m.sources), 2)
        self.assertTrue(m.has_csv())
        self.assertTrue(m.has_db())
        self.assertEqual(m.primary_source().alias, "sales_csv")

        # ── VERIFY IPYNB COMPATIBILITY ──

        ipynb = nb.to_ipynb_dict()
        raw_json = json.dumps(ipynb, ensure_ascii=False)
        parsed = json.loads(raw_json)
        self.assertEqual(parsed["nbformat"], 4)
        self.assertEqual(len(parsed["cells"]), 5)

        # Roundtrip
        nb_restored = NotebookDocument.from_ipynb_dict(parsed)
        self.assertEqual(len(nb_restored.cells), 5)
        self.assertEqual(nb_restored.cells[2].metadata.tool_name, "pandas_tool")

    def test_source_removal_does_not_delete_binding_cell(self) -> None:
        """Removing a source from manifest doesn't auto-delete the binding cell."""
        sid = self.session_id
        self.nb_store.create_empty(sid)
        manifest = SessionManifest(session_id=sid)

        # Add source + cell.
        manifest.add_source(SessionSource(alias="csv1", source_type="csv", variable_name="df1"))
        self.mf_store.save(sid, manifest)

        cell = build_source_binding_cell(
            alias="csv1", variable_name="df1",
            source_type="csv", display_name="test.csv",
            load_code='df1 = pd.read_parquet("test.parquet")',
        )
        self.orch.apply(sid, NotebookEdit(op=CellOp.INSERT, cell=cell))

        # Remove from manifest.
        manifest.remove_source("csv1")
        self.mf_store.save(sid, manifest)

        # Binding cell still in notebook (can't be deleted via normal DELETE).
        nb = self.nb_store.load(sid)
        self.assertEqual(len(nb.cells), 1)
        self.assertTrue(nb.cells[0].is_source_binding)


# ── 8. KernelManager restore test ───────────────────────────────────────────


class TestKernelManagerRestore(unittest.TestCase):
    """KernelManager restores kernel by replaying notebook cells."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        base = self._tmpdir.name
        self.nb_store = NotebookStore(base)
        self.mf_store = ManifestStore(base)
        self.session_id = "km-test"

        # Create a session dir with a CSV file (avoid pyarrow dependency).
        session_dir = Path(base) / "sessions" / self.session_id / "sources"
        session_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
        self.csv_path = session_dir / "data.csv"
        df.to_csv(self.csv_path, index=False)

        # Create manifest with CSV source.
        manifest = SessionManifest(session_id=self.session_id)
        manifest.add_source(SessionSource(
            alias="data_csv",
            source_type="csv",
            display_name="data.csv",
            variable_name="data_df",
            parquet_path="sources/data.csv",
        ))
        self.mf_store.save(self.session_id, manifest)

        # Create notebook with source binding + one analysis cell.
        # Use pd.read_csv for the test (parquet not available in test env).
        nb = NotebookDocument(session_id=self.session_id)
        nb.append_cell(build_source_binding_cell(
            alias="data_csv",
            variable_name="data_df",
            source_type="csv",
            display_name="data.csv",
            load_code=f'data_df = pd.read_csv("{self.csv_path.as_posix()}")',
        ))
        nb.append_cell(NotebookCell(
            id="calc",
            source="total = data_df['x'].sum()",
            metadata=CellMetadata(
                purpose="Sum of x",
                produces=["total"],
                depends_on=["data_df"],
                tool_name="pandas_tool",
            ),
        ))
        self.nb_store.save(self.session_id, nb)

    def tearDown(self) -> None:
        # Clean up SandboxManager state.
        from backend.tools.sandbox_manager import SandboxManager
        SandboxManager.get_instance().remove(self.session_id)
        self._tmpdir.cleanup()

    def test_restore_replays_cells(self) -> None:
        from backend.notebook.kernel_manager import KernelManager

        km = KernelManager(
            notebook_store=self.nb_store,
            manifest_store=self.mf_store,
            storage_dir=self._tmpdir.name,
        )
        state = km.get_or_restore(self.session_id)

        self.assertTrue(state.restored)
        self.assertEqual(len(state.restore_errors), 0)
        self.assertEqual(state.cell_execution_count, 2)  # source_binding + analysis cell

        # Verify kernel scope has the variables.
        from backend.tools.sandbox_manager import SandboxManager
        sandbox = SandboxManager.get_instance().get(self.session_id)
        self.assertIsNotNone(sandbox)

        # data_df was bound from source.
        result = sandbox.execute("tool_result = len(data_df)", tool_name="test")
        self.assertEqual(result, 3)

        # total was computed during replay.
        result2 = sandbox.execute("tool_result = total", tool_name="test")
        self.assertEqual(result2, 6)

    def test_restore_is_idempotent(self) -> None:
        """Second call returns cached kernel, not a re-restore."""
        from backend.notebook.kernel_manager import KernelManager

        km = KernelManager(
            notebook_store=self.nb_store,
            manifest_store=self.mf_store,
            storage_dir=self._tmpdir.name,
        )
        state1 = km.get_or_restore(self.session_id)
        state2 = km.get_or_restore(self.session_id)

        self.assertTrue(state1.restored)
        # Second call should return same state (not re-restored).
        self.assertIs(state1, state2)


if __name__ == "__main__":
    unittest.main()
