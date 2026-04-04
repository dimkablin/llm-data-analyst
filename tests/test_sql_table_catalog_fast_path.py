from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from backend.data_access import RuntimeDBConnectionConfig
from backend.data_access.sql_table_service import SQLTableService
from backend.tools.impl.db_helpers import DBAnalyticsHelper


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


class BuildTableArtifactCatalogTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
