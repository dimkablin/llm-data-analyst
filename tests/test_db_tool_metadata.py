from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from backend.data_access import RuntimeDBConnectionConfig
from backend.tools.impl.db_helpers import DBDemoHelper


def _runtime() -> RuntimeDBConnectionConfig:
    return RuntimeDBConnectionConfig(
        connection_id="conn-1",
        user_id=7,
        name="Sales DB",
        db_type="postgresql",
        host="localhost",
        port=5432,
        database="analytics",
        username="analyst",
        password="secret",
        options={"schema": "public"},
    )


class DBToolMetadataResultTests(unittest.TestCase):
    def test_validate_sql_appends_limit_and_warnings(self) -> None:
        helper = DBDemoHelper(runtime=_runtime())
        result = helper.validate_sql("SELECT * FROM public.orders", max_rows=25)

        self.assertEqual(result["requested_sql"], "SELECT * FROM public.orders")
        self.assertEqual(result["max_rows"], 25)
        self.assertIn("LIMIT 26", result["normalized_sql"])
        self.assertTrue(result["warnings"])

    def test_validate_sql_rejects_mutating_statement(self) -> None:
        helper = DBDemoHelper(runtime=_runtime())
        with self.assertRaises(ValueError):
            helper.validate_sql("DELETE FROM public.orders")

    def test_list_schemas_result_contains_source_and_recipe(self) -> None:
        helper = DBDemoHelper(runtime=_runtime())
        with patch.object(
            DBDemoHelper,
            "list_schemas",
            return_value=[
                {"name": "public", "display_name": "public"},
                {"name": "mart", "display_name": "mart"},
            ],
        ):
            result = helper.list_schemas_result()

        self.assertEqual(result["schema_version"], "1.0")
        self.assertEqual(result["artifact_type"], "table")
        self.assertEqual(result["source"]["source_type"], "db_connection")
        self.assertEqual(result["source"]["source_ref_id"], "conn-1")
        self.assertEqual(result["recipe"][0]["kind"], "db_metadata")
        self.assertEqual(result["recipe"][0]["tool_name"], "sql_tool")
        rows = result["items"]["db_schemas"]
        self.assertEqual(list(rows.columns), ["name", "display_name"])
        self.assertEqual(len(rows), 2)

    def test_list_effective_tables_with_columns_uses_configured_schema_only(self) -> None:
        helper = DBDemoHelper(runtime=_runtime())
        with (
            patch.object(DBDemoHelper, "list_tables_with_columns") as list_mock,
            patch.object(DBDemoHelper, "list_all_tables_with_columns") as all_mock,
        ):
            list_mock.return_value = [
                {
                    "schema": "public",
                    "table_name": "orders",
                    "table_type": "table",
                    "qualified_name": "public.orders",
                    "columns": ["id"],
                }
            ]
            rows = helper.list_effective_tables_with_columns()

        list_mock.assert_called_once_with()
        all_mock.assert_not_called()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["schema"], "public")

    def test_list_effective_tables_with_columns_scans_all_when_no_schema_config(self) -> None:
        runtime = RuntimeDBConnectionConfig(
            connection_id="conn-2",
            user_id=7,
            name="Wide DB",
            db_type="postgresql",
            host="localhost",
            port=5432,
            database="analytics",
            username="analyst",
            password="secret",
            options={},
        )
        helper = DBDemoHelper(runtime=runtime)
        with (
            patch.object(DBDemoHelper, "list_all_tables_with_columns") as all_mock,
            patch.object(DBDemoHelper, "list_tables_with_columns") as list_mock,
        ):
            all_mock.return_value = [
                {
                    "schema": "demo_invest",
                    "table_name": "instrument_snapshot_demo",
                    "table_type": "table",
                    "qualified_name": "demo_invest.instrument_snapshot_demo",
                    "columns": ["ticker"],
                }
            ]
            rows = helper.list_effective_tables_with_columns()

        all_mock.assert_called_once_with()
        list_mock.assert_not_called()
        self.assertEqual(rows[0]["schema"], "demo_invest")

    def test_list_all_tables_with_columns_deduplicates(self) -> None:
        helper = DBDemoHelper(runtime=_runtime())
        with (
            patch.object(
                DBDemoHelper,
                "list_schemas",
                return_value=[{"name": "demo_invest"}, {"name": "public"}],
            ),
            patch.object(
                DBDemoHelper,
                "list_tables_with_columns",
                side_effect=[
                    [
                        {
                            "schema": "demo_invest",
                            "table_name": "instrument_snapshot_demo",
                            "table_type": "table",
                            "qualified_name": "demo_invest.instrument_snapshot_demo",
                            "columns": ["ticker"],
                        }
                    ],
                    [
                        {
                            "schema": "public",
                            "table_name": "instrument_snapshot_demo",
                            "table_type": "table",
                            "qualified_name": "public.instrument_snapshot_demo",
                            "columns": ["ticker"],
                        }
                    ],
                ],
            ),
        ):
            rows = helper.list_all_tables_with_columns()

        self.assertEqual(len(rows), 2)
        schemas = {row["schema"] for row in rows}
        self.assertEqual(schemas, {"demo_invest", "public"})

    def test_list_tables_result_uses_schema_specific_artifact_name(self) -> None:
        helper = DBDemoHelper(runtime=_runtime())
        with patch.object(
            DBDemoHelper,
            "list_effective_tables_with_columns",
            return_value=[
                {
                    "schema": "public",
                    "table_name": "orders",
                    "table_type": "table",
                    "qualified_name": "public.orders",
                    "columns": ["order_id"],
                }
            ],
        ):
            result = helper.list_tables_result("public")

        self.assertIn("db_tables_public", result["items"])
        self.assertEqual(result["recipe"][0]["title"], "List Tables")
        self.assertIn("schema=public", result["recipe"][0]["summary"])
        rows = result["items"]["db_tables_public"]
        self.assertEqual(
            list(rows.columns),
            ["schema", "table_name", "table_type", "qualified_name"],
        )

    def test_describe_table_result_returns_column_metadata(self) -> None:
        helper = DBDemoHelper(runtime=_runtime())
        with patch.object(
            DBDemoHelper,
            "describe_table",
            return_value=[
                {
                    "schema": "public",
                    "table_name": "orders",
                    "column_name": "order_id",
                    "data_type": "bigint",
                    "is_nullable": False,
                    "ordinal_position": 1,
                    "default_expression": None,
                }
            ],
        ):
            result = helper.describe_table_result("orders", schema="public")

        self.assertIn("describe_public_orders", result["items"])
        self.assertEqual(result["recipe"][0]["title"], "Describe Table")
        self.assertIn("table=orders", result["recipe"][0]["summary"])
        rows = result["items"]["describe_public_orders"]
        self.assertEqual(
            list(rows.columns),
            [
                "schema",
                "table_name",
                "column_name",
                "data_type",
                "is_nullable",
                "ordinal_position",
                "default_expression",
            ],
        )

    def test_execute_analytic_query_packages_sql_stats_and_truncation(self) -> None:
        helper = DBDemoHelper(runtime=_runtime())
        sample = pd.DataFrame(
            [
                {"month": "2025-01-01", "revenue": 10},
                {"month": "2025-02-01", "revenue": 20},
                {"month": "2025-03-01", "revenue": 30},
            ]
        )
        with patch.object(DBDemoHelper, "query_dataframe", return_value=sample):
            result = helper.execute_analytic_query(
                "SELECT * FROM public.orders",
                purpose="Monthly revenue",
                max_rows=2,
                artifact_name="monthly_revenue",
            )

        self.assertIn("monthly_revenue", result["items"])
        self.assertEqual(result["recipe"][0]["kind"], "sql")
        self.assertIn("LIMIT 3", result["recipe"][0]["code"])
        self.assertEqual(result["meta"]["query"]["requested_sql"], "SELECT * FROM public.orders")
        self.assertEqual(result["meta"]["query"]["max_rows"], 2)
        self.assertEqual(result["meta"]["query"]["returned_rows"], 2)
        self.assertTrue(result["meta"]["query"]["truncated"])
        self.assertTrue(result["meta"]["warnings"])
