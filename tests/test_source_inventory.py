from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from backend.data_access.csv_session_runtime import CSVSessionRuntime
from backend.data_access.db_runtime_service import RuntimeDBConnectionConfig
from backend.data_access.source_inventory import (
    SourceInventory,
    SourceInventorySource,
    SourceInventoryTable,
    build_source_inventory,
    format_source_inventory_prompt,
)
from backend.data_access.tabular_upload_service import TabularUploadFile, TabularUploadService
from backend.notebook.manifest_store import ManifestStore
from backend.notebook.orchestrator import NotebookOrchestrator
from backend.notebook.store import NotebookStore
from backend.sessions.session_store import SessionStore
from backend.tools.impl.data_catalog_tool import DataCatalogTool
from backend.tools.impl.db_helpers import DBDemoHelper


def _runtime() -> RuntimeDBConnectionConfig:
    return RuntimeDBConnectionConfig(
        connection_id="conn-1",
        user_id=7,
        name="Warehouse",
        db_type="postgresql",
        host="localhost",
        port=5432,
        database="analytics",
        username="analyst",
        password="secret",
        options={},
    )


def test_source_inventory_merges_csv_tables_and_db_schema_tables(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path / "sessions_state"), ttl_days=7)
    session = store.create_session()
    csv_runtime = CSVSessionRuntime(base_dir=tmp_path / "duckdb", default_ttl_sec=3600)
    manifest_store = ManifestStore(tmp_path)
    upload_service = TabularUploadService(
        store=store,
        csv_runtime=csv_runtime,
        manifest_store=manifest_store,
        notebook_orchestrator=NotebookOrchestrator(NotebookStore(tmp_path)),
        storage_dir=tmp_path,
    )
    upload_service.ingest_files(
        session_id=session.session_id,
        files=[
            TabularUploadFile(
                file_name="orders.csv",
                content=b"order_id,customer_id,amount\n1,10,120\n",
            ),
            TabularUploadFile(
                file_name="customers.csv",
                content=b"customer_id,segment\n10,enterprise\n",
            ),
        ],
        ttl_seconds=3600,
    )

    with patch.object(
        DBDemoHelper,
        "list_effective_tables_with_columns",
        return_value=[
            {
                "schema": "mart",
                "table_name": "orders",
                "table_type": "table",
                "qualified_name": "mart.orders",
                "columns": ["order_id", "net_revenue"],
                "column_types": {
                    "order_id": "integer",
                    "net_revenue": "numeric",
                },
            }
        ],
    ):
        inventory = build_source_inventory(
            session_id=session.session_id,
            session_source={
                "source_type": "csv",
                "csv_loaded": True,
                "csv_session_id": session.session_id,
            },
            manifest_store=manifest_store,
            csv_runtime=csv_runtime,
            db_runtime=_runtime(),
        )

    assert [table.qualified_name for table in inventory.tables] == [
        "customers",
        "orders",
        "mart.orders",
    ]
    assert {table.source_type for table in inventory.tables} == {
        "csv",
        "db_connection",
    }
    prompt = format_source_inventory_prompt(inventory)
    assert "orders.csv" in prompt
    assert "mart.orders" in prompt
    assert "net_revenue" in prompt
    assert "`net_revenue` (numeric)" in prompt


def test_data_catalog_tool_lists_and_describes_inventory() -> None:
    inventory = SourceInventory(
        session_id="session-1",
        sources=[
            SourceInventorySource(
                source_id="orders_csv",
                source_type="csv",
                label="orders.csv",
                alias="orders_csv",
            ),
            SourceInventorySource(
                source_id="db:conn-1",
                source_type="db_connection",
                label="Warehouse",
            ),
        ],
        tables=[
            SourceInventoryTable(
                source_id="orders_csv",
                source_type="csv",
                table_name="orders",
                qualified_name="orders",
                schema_name="main",
                source_label="orders.csv",
                source_alias="orders_csv",
                columns=["order_id", "amount"],
            ),
            SourceInventoryTable(
                source_id="db:conn-1",
                source_type="db_connection",
                table_name="orders",
                qualified_name="mart.orders",
                schema_name="mart",
                source_label="Warehouse",
                columns=["order_id", "net_revenue"],
            ),
        ],
    )
    tool = DataCatalogTool(source_inventory=inventory)

    listed = json.loads(tool.invoke({"action": "list_tables"}))
    assert listed["action"] == "list_tables"
    assert [row["qualified_name"] for row in listed["tables"]] == ["orders", "mart.orders"]

    described = json.loads(tool.invoke({"action": "describe_table", "table": "mart.orders"}))
    assert described["action"] == "describe_table"
    assert described["selected_table"]["qualified_name"] == "mart.orders"
    assert described["selected_table"]["columns"] == ["order_id", "net_revenue"]

    ambiguous = json.loads(tool.invoke({"action": "describe_table", "table": "orders"}))
    assert ambiguous["status"] == "ambiguous"
    assert [row["qualified_name"] for row in ambiguous["tables"]] == ["orders", "mart.orders"]
