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


class CatalogTableListIntentTests(unittest.TestCase):
    def test_wants_catalog_matches_listing_phrases(self) -> None:
        yes = [
            "SHOW TABLES",
            "list tables",
            "tables",
            "what tables are there",
        ]
        for q in yes:
            with self.subTest(q=q):
                self.assertTrue(SQLTableService._wants_catalog_table_list(q), msg=repr(q))

    def test_wants_catalog_rejects_analytic_phrases(self) -> None:
        no = [
            "show tables with sales",
            "how many rows are in orders",
        ]
        for q in no:
            with self.subTest(q=q):
                self.assertFalse(SQLTableService._wants_catalog_table_list(q), msg=repr(q))


class SQLToolArgsContractTests(unittest.TestCase):
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

    def test_payload_contract_requires_table_items(self) -> None:
        with self.assertRaises(ValueError):
            SQLToolPayload.model_validate(
                {
                    "schema_version": "1.0",
                    "artifact_type": "table",
                }
            )


class BuildTableArtifactCatalogTests(unittest.TestCase):
    def test_explicit_schema_qualified_name_wins_over_bare_table_name(self) -> None:
        svc = SQLTableService(
            llm_base_url="http://localhost:9",
            llm_model="dummy",
            llm_api_key="dummy",
        )
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
            columns=["order_id", "net_revenue"],
            source_label="DB",
            source_ref_id="conn-1",
        )

        picked = svc._find_explicit_table(
            "show mart.orders revenue",
            [public_orders, mart_orders],
        )

        self.assertIs(picked, mart_orders)

    def test_referenced_candidates_for_schema_qualified_sql_ignores_same_bare_name(self) -> None:
        svc = SQLTableService(
            llm_base_url="http://localhost:9",
            llm_model="dummy",
            llm_api_key="dummy",
        )
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
            llm_base_url="http://localhost:9",
            llm_model="dummy",
            llm_api_key="dummy",
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
            llm_base_url="http://localhost:9",
            llm_model="dummy",
            llm_api_key="dummy",
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
            llm_base_url="http://localhost:9",
            llm_model="dummy",
            llm_api_key="dummy",
            db_runtime_config=_runtime(),
            csv_loaded=False,
        )
        mock_payload = {
            "schema_version": "1.0",
            "artifact_type": "table",
            "items": {"db_tables_public": MagicMock()},
            "source": {"source_type": "db_connection"},
        }
        with (
            patch.object(DBAnalyticsHelper, "list_tables_result", return_value=mock_payload) as list_mock,
            patch.object(SQLTableService, "resolve_table", side_effect=AssertionError("resolve_table must not run")),
        ):
            out = svc.build_table_artifact("list tables")
        list_mock.assert_called_once()
        self.assertIs(out, mock_payload)

    def test_catalog_path_merges_meta_flag(self) -> None:
        svc = SQLTableService(
            llm_base_url="http://localhost:9",
            llm_model="dummy",
            llm_api_key="dummy",
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
            out = svc.build_table_artifact("list tables")
        self.assertTrue(out.get("meta", {}).get("catalog_listing"))

    def test_join_question_passes_related_csv_table_to_sql_generation(self) -> None:
        svc = SQLTableService(
            llm_base_url="http://localhost:9",
            llm_model="dummy",
            llm_api_key="dummy",
            csv_loaded=True,
            csv_session_id="sid",
        )
        orders = TableCandidate(
            source_kind="csv_session",
            dialect="duckdb",
            table_name="orders",
            qualified_name="orders",
            schema="main",
            columns=["order_id", "customer_id", "amount"],
            source_label="CSV session sid",
            source_ref_id="sid",
            csv_session_id="sid",
        )
        customers = TableCandidate(
            source_kind="csv_session",
            dialect="duckdb",
            table_name="customers",
            qualified_name="customers",
            schema="main",
            columns=["customer_id", "segment"],
            source_label="CSV session sid",
            source_ref_id="sid",
            csv_session_id="sid",
        )
        with (
            patch.object(SQLTableService, "resolve_table", return_value=orders),
            patch.object(SQLTableService, "collect_candidates", return_value=[orders, customers]),
            patch.object(
                SQLTableService,
                "generate_sql_with_retries",
                return_value={"ok": True, "sql": "SELECT 1"},
            ) as generate_mock,
            patch.object(
                SQLTableService,
                "execute_final_query",
                return_value={"schema_version": "1.0", "artifact_type": "table", "items": {}},
            ),
        ):
            svc.build_table_artifact("join orders and customers by customer_id")

        related = generate_mock.call_args.kwargs["additional_candidates"]
        self.assertEqual([candidate.table_name for candidate in related], ["customers"])

    def test_raw_information_schema_sql_executes_without_llm_generation(self) -> None:
        runtime = CSVSessionRuntime(base_dir=self._tmp_path(), default_ttl_sec=3600)
        runtime.register_dataframes(
            session_id="sid",
            tables={"orders": pd.DataFrame({"order_id": [1], "amount": [10]})},
            ttl_seconds=3600,
        )

        svc = SQLTableService(
            llm_base_url="http://localhost:9",
            llm_model="dummy",
            llm_api_key="dummy",
            csv_loaded=True,
            csv_session_id="sid",
            max_rows=50,
        )
        svc.csv_runtime = runtime

        with (
            patch.object(SQLTableService, "resolve_table", side_effect=AssertionError("raw SQL must not resolve one table")),
            patch.object(SQLTableService, "generate_sql_with_retries", side_effect=AssertionError("raw SQL must not call LLM generation")),
        ):
            out = svc.build_table_artifact(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY table_name",
                artifact_name="table_list",
            )

        table = out["items"]["table_list"]
        self.assertIn("orders", set(table["table_name"].astype(str)))

    def test_semantic_join_mapping_adds_candidate_without_shared_column_names(self) -> None:
        svc = SQLTableService(
            llm_base_url="http://localhost:9",
            llm_model="dummy",
            llm_api_key="dummy",
            csv_loaded=True,
            csv_session_id="sid",
        )
        actuals = TableCandidate(
            source_kind="csv_session",
            dialect="duckdb",
            table_name="actuals",
            qualified_name="actuals",
            schema="main",
            columns=["Статья ДДС", "ЦФО (Документ)", "Сумма"],
            source_label="CSV session sid",
            source_ref_id="sid",
            csv_session_id="sid",
        )
        plan = TableCandidate(
            source_kind="csv_session",
            dialect="duckdb",
            table_name="plan",
            qualified_name="plan",
            schema="main",
            columns=["статья CF", "ЦФО", "CF Mar"],
            source_label="CSV session sid",
            source_ref_id="sid",
            csv_session_id="sid",
        )

        related = svc._additional_candidates_for_question(
            "Сопоставь Статья ДДС со статья CF и ЦФО (Документ) с ЦФО",
            actuals,
            [actuals, plan],
        )

        self.assertEqual([candidate.table_name for candidate in related], ["plan"])

    @staticmethod
    def _tmp_path():
        import tempfile
        from pathlib import Path

        return Path(tempfile.mkdtemp())
