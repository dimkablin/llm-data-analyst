from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

import backend.tools.impl  # noqa: F401 - side-effect import avoids sql_table_service circular import
from backend.data_access import RuntimeDBConnectionConfig
from backend.data_access.csv_session_runtime import CSVSessionRuntime
from backend.data_access.sql_table_service import SQLTableService, TableCandidate
from backend.tools.impl.db_helpers import DBAnalyticsHelper
from backend.tools.impl.sql_tool import SQLToolArgs, SQLToolPayload


def _runtime() -> RuntimeDBConnectionConfig:
    return RuntimeDBConnectionConfig(
        connection_id="conn-1",
        user_id=7,
        name="Examples",
        db_type="postgresql",
        host="localhost",
        port=5432,
        database="examples",
        username="analyst",
        password="secret",
        options={"schema": "public"},
    )


class SQLToolArgsContractTests(unittest.TestCase):
    def test_natural_language_requires_an_explicit_mode_and_sql(self) -> None:
        with self.assertRaisesRegex(ValueError, "mode is required"):
            SQLToolArgs(question="Analyze the orders")

        with self.assertRaises(ValueError):
            SQLToolArgs(mode="nl_query", question="Analyze the orders")

    def test_back_compat_raw_select_question_is_inferred_as_execute_sql(self) -> None:
        args = SQLToolArgs(question="SELECT * FROM orders LIMIT 5", artifact_name="orders_sample")

        self.assertEqual(args.mode, "execute_sql")
        self.assertEqual(args.sql, "SELECT * FROM orders LIMIT 5")
        self.assertEqual(args.artifact_name, "orders_sample")

    def test_describe_mode_requires_table_names(self) -> None:
        with self.assertRaises(ValueError):
            SQLToolArgs(mode="describe_table")

    def test_describe_mode_accepts_legacy_table_alias(self) -> None:
        args = SQLToolArgs(mode="describe_table", table="fact")

        self.assertEqual(args.mode, "describe_table")
        self.assertEqual(args.table_names, ["fact"])

    def test_describe_mode_accepts_single_table_name_string(self) -> None:
        args = SQLToolArgs(mode="describe_table", table_names="fact")

        self.assertEqual(args.mode, "describe_table")
        self.assertEqual(args.table_names, ["fact"])

    def test_semantic_query_mode_is_inferred_from_typed_payload(self) -> None:
        args = SQLToolArgs(
            metrics=["service_resolution_index"],
            dimensions=["channel"],
            limit=25,
        )

        self.assertEqual(args.mode, "semantic_query")
        self.assertEqual(args.to_semantic_query().metrics, ["service_resolution_index"])
        self.assertEqual(args.to_semantic_query().dimensions, ["channel"])

    def test_semantic_query_mode_requires_typed_payload(self) -> None:
        with self.assertRaises(ValueError):
            SQLToolArgs(mode="semantic_query")

    def test_semantic_query_schema_separates_time_grain_from_dimensions(self) -> None:
        properties = SQLToolArgs.model_json_schema()["properties"]

        self.assertIn("Do not put time grain labels", properties["dimensions"]["description"])
        self.assertIn("Do not duplicate this grain", properties["time_grain"]["description"])

    def test_semantic_transport_schema_exposes_native_top_level_fields(self) -> None:
        properties = SQLToolArgs.model_json_schema()["properties"]

        self.assertNotIn("semantic_query", properties)
        self.assertEqual(properties["metrics"]["type"], "array")
        self.assertEqual(properties["filters"]["type"], "array")

    def test_payload_contract_requires_table_items(self) -> None:
        with self.assertRaises(ValueError):
            SQLToolPayload.model_validate(
                {
                    "schema_version": "1.0",
                    "artifact_type": "table",
                }
            )


class BuildTableArtifactCatalogTests(unittest.TestCase):
    def test_sql_queries_use_a_30_second_statement_timeout(self) -> None:
        service = SQLTableService(db_runtime_config=_runtime())

        self.assertEqual(service._db_helper().timeout_sec, 30.0)

    def test_referenced_candidates_for_schema_qualified_sql_ignores_same_bare_name(self) -> None:
        svc = SQLTableService()
        public_orders = TableCandidate(
            source_kind="db",
            dialect="postgresql",
            table_name="orders",
            qualified_name="public.orders",
            schema="public",
            columns=["order_id"],
            source_label="DB",
            source_ref_id="conn-1",
        )
        mart_orders = TableCandidate(
            source_kind="db",
            dialect="postgresql",
            table_name="orders",
            qualified_name="mart.orders",
            schema="mart",
            columns=["order_id"],
            source_label="DB",
            source_ref_id="conn-1",
        )
        with patch.object(
            SQLTableService,
            "collect_candidates",
            return_value=[public_orders, mart_orders],
        ):
            referenced = svc._referenced_candidates_for_sql("SELECT * FROM mart.orders")

        self.assertEqual([candidate.qualified_name for candidate in referenced], ["mart.orders"])

    def test_describe_table_splits_schema_qualified_names(self) -> None:
        svc = SQLTableService(
            db_runtime_config=_runtime(),
            csv_loaded=False,
        )
        payload = {
            "schema_version": "1.0",
            "artifact_type": "table",
            "items": {"describe_mart_orders": pd.DataFrame()},
            "recipe": [],
        }

        with patch.object(
            DBAnalyticsHelper,
            "describe_table_result",
            return_value=payload,
        ) as describe_mock:
            svc.build_describe_tables_artifact(["mart.orders"])

        describe_mock.assert_called_once_with("orders", schema="mart")

    def test_db_execute_final_query_adds_lineage_for_selected_table(self) -> None:
        runtime = _runtime()
        svc = SQLTableService(
            db_runtime_config=runtime,
        )
        candidate = TableCandidate(
            source_kind="db",
            dialect="postgresql",
            table_name="orders",
            qualified_name="mart.orders",
            schema="mart",
            columns=["order_id", "amount"],
            source_label="DB",
            source_ref_id="conn-1",
            db_runtime=runtime,
        )
        payload = {
            "schema_version": "1.0",
            "artifact_type": "table",
            "items": {"orders_result": pd.DataFrame({"order_id": [1]})},
            "source": {"source_type": "db_connection"},
            "recipe": [],
            "meta": {"query": {"requested_sql": "SELECT order_id FROM mart.orders"}},
        }
        with (
            patch.object(SQLTableService, "collect_candidates", return_value=[candidate]),
            patch.object(DBAnalyticsHelper, "execute_analytic_query", return_value=payload),
        ):
            out = svc.execute_final_query(
                question="show orders",
                candidate=candidate,
                sql="SELECT order_id FROM mart.orders",
                artifact_name="orders_result",
            )

        self.assertEqual(out["meta"]["lineage"]["source_table_names"], ["mart.orders"])
        self.assertEqual(out["meta"]["lineage"]["source_tables"][0]["schema"], "mart")

    def test_catalog_path_skips_llm_and_uses_list_tables_result(self) -> None:
        svc = SQLTableService(
            db_runtime_config=_runtime(),
            csv_loaded=False,
        )
        mock_payload = {
            "schema_version": "1.0",
            "artifact_type": "table",
            "items": {"db_tables_public": MagicMock()},
            "source": {"source_type": "db_connection"},
        }
        with patch.object(DBAnalyticsHelper, "list_tables_result", return_value=mock_payload) as list_mock:
            out = svc.build_table_artifact("", mode="catalog_tables")
        list_mock.assert_called_once()
        self.assertIs(out, mock_payload)

    def test_catalog_path_merges_meta_flag(self) -> None:
        svc = SQLTableService(
            db_runtime_config=_runtime(),
            csv_loaded=False,
        )
        mock_payload: dict = {
            "schema_version": "1.0",
            "artifact_type": "table",
            "items": {"db_tables_public": MagicMock()},
            "source": {"source_type": "db_connection"},
        }
        with patch.object(DBAnalyticsHelper, "list_tables_result", return_value=mock_payload):
            out = svc.build_table_artifact("", mode="catalog_tables")
        self.assertTrue(out.get("meta", {}).get("catalog_listing"))

    def test_natural_language_requires_an_explicit_sql_mode(self) -> None:
        svc = SQLTableService()
        with self.assertRaisesRegex(ValueError, "mode is required"):
            svc.build_table_artifact("list tables")

    def test_raw_information_schema_sql_executes_without_llm_generation(self) -> None:
        runtime = CSVSessionRuntime(base_dir=self._tmp_path(), default_ttl_sec=3600)
        runtime.register_dataframes(
            session_id="sid",
            tables={"orders": pd.DataFrame({"order_id": [1], "amount": [10]})},
            ttl_seconds=3600,
        )

        svc = SQLTableService(
            csv_loaded=True,
            csv_session_id="sid",
            max_rows=50,
        )
        svc.csv_runtime = runtime

        out = svc.build_table_artifact(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name",
            artifact_name="table_list",
        )

        table = out["items"]["table_list"]
        self.assertIn("orders", set(table["table_name"].astype(str)))

    @staticmethod
    def _tmp_path():
        import tempfile
        from pathlib import Path

        return Path(tempfile.mkdtemp())
