from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from agent.tools.db_tool import DBDemoHelper
from backend.db_runtime_service import RuntimeDBConnectionConfig


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
        self.assertEqual(result["recipe"][0]["tool_name"], "db_tool")
        rows = result["items"]["db_schemas"]
        self.assertEqual(list(rows.columns), ["name", "display_name"])
        self.assertEqual(len(rows), 2)

    def test_list_tables_result_uses_schema_specific_artifact_name(self) -> None:
        helper = DBDemoHelper(runtime=_runtime())
        with patch.object(
            DBDemoHelper,
            "list_tables",
            return_value=[
                {
                    "schema": "public",
                    "table_name": "orders",
                    "table_type": "table",
                    "qualified_name": "public.orders",
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


if __name__ == "__main__":
    unittest.main()
