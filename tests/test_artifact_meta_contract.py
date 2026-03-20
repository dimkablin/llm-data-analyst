from __future__ import annotations

import unittest

from backend.artifact_meta import (
    RECIPE_SCHEMA_VERSION,
    build_artifact_meta,
    build_chart_recipe_step,
    build_db_metadata_recipe_step,
    build_python_recipe_step,
    build_sql_recipe_step,
    normalize_recipe_steps,
)


class ArtifactMetaContractTests(unittest.TestCase):
    def test_normalize_recipe_steps_enforces_minimal_contract(self) -> None:
        steps = normalize_recipe_steps(
            [
                {"sql": "select 1", "tool": "db_tool"},
                {"kind": "python", "code": "tool_result = 1", "tool_name": "pandas_tool"},
                {"kind": "metadata", "summary": "Read schema"},
                {"kind": "plot", "summary": "Rendered chart", "depends_on": ["1", "2"]},
            ]
        )

        self.assertEqual([step["kind"] for step in steps], ["sql", "python", "db_metadata", "chart"])
        self.assertEqual(steps[0]["language"], "sql")
        self.assertEqual(steps[1]["language"], "python")
        self.assertEqual(steps[2]["title"], "DB metadata")
        self.assertEqual(steps[3]["depends_on"], ["1", "2"])

    def test_build_artifact_meta_uses_recipe_as_canonical_source(self) -> None:
        meta = build_artifact_meta(
            base_meta={
                "code": "print('legacy')",
                "source": {"source_label": "base label"},
                "recipe": [build_python_recipe_step(tool_name="plotly_tool", code="fig = go.Figure()")],
            },
            tool_name="plotly_tool",
            tool_code="print('newer but compat only')",
            source_context={
                "source_type": "db_connection",
                "source_ref_id": "conn-1",
                "source_mode": "read_only",
            },
            artifact_hints={
                "source": {"source_label": "Sales DB"},
                "recipe": [build_sql_recipe_step(sql="SELECT 1", tool_name="db_tool")],
            },
        )

        self.assertEqual(meta["code"], "fig = go.Figure()")
        self.assertEqual(meta["source"]["source_type"], "db_connection")
        self.assertEqual(meta["source"]["source_label"], "Sales DB")
        self.assertEqual(meta["provenance"]["recipe_schema_version"], RECIPE_SCHEMA_VERSION)
        self.assertEqual(meta["provenance"]["recipe"], meta["recipe"])
        self.assertEqual(meta["provenance"]["source"], meta["source"])
        self.assertEqual(meta["provenance"]["step_count"], len(meta["recipe"]))
        self.assertEqual(meta["recipe"][0]["kind"], "python")
        self.assertEqual(meta["recipe"][1]["kind"], "sql")

    def test_build_artifact_meta_adds_python_step_from_compat_code(self) -> None:
        meta = build_artifact_meta(
            tool_name="pandas_tool",
            tool_code="tool_result = df.head()",
            source_context={"source_type": "csv", "source_ref_id": "file.csv"},
        )

        self.assertEqual(meta["code"], "tool_result = df.head()")
        self.assertEqual(meta["recipe"][0]["kind"], "python")
        self.assertEqual(meta["recipe"][0]["code"], "tool_result = df.head()")

    def test_recipe_builders_produce_reserved_future_chart_shape(self) -> None:
        sql_step = build_sql_recipe_step(sql="SELECT * FROM orders LIMIT 10", tool_name="db_tool")
        metadata_step = build_db_metadata_recipe_step(
            action="describe_table",
            tool_name="db_tool",
            summary="Read table metadata",
        )
        chart_step = build_chart_recipe_step(
            tool_name="plotly_tool",
            summary="Rendered grouped bar chart",
            depends_on=["1", "2"],
        )

        self.assertEqual(sql_step["kind"], "sql")
        self.assertEqual(metadata_step["kind"], "db_metadata")
        self.assertEqual(chart_step["kind"], "chart")
        self.assertEqual(chart_step["depends_on"], ["1", "2"])

    def test_build_artifact_meta_inserts_python_before_chart_step(self) -> None:
        meta = build_artifact_meta(
            tool_name="plotly_tool",
            tool_code="fig = px.bar(rows, x='month', y='revenue')",
            source_context={"source_type": "db_connection", "source_ref_id": "conn-1"},
            artifact_hints={
                "source": {"source_label": "Sales DB"},
                "recipe": [
                    build_sql_recipe_step(
                        sql="SELECT month, revenue FROM monthly_revenue LIMIT 12",
                        tool_name="db_tool",
                    ),
                    build_chart_recipe_step(
                        tool_name="plotly_tool",
                        summary="Bar chart of monthly revenue",
                    ),
                ],
            },
        )

        self.assertEqual(
            [step["kind"] for step in meta["recipe"]],
            ["sql", "python", "chart"],
        )
        self.assertEqual(meta["code"], "fig = px.bar(rows, x='month', y='revenue')")
        self.assertEqual(meta["provenance"]["recipe"], meta["recipe"])
        self.assertEqual(meta["provenance"]["source"], meta["source"])
        self.assertEqual(meta["source"]["source_label"], "Sales DB")


if __name__ == "__main__":
    unittest.main()
