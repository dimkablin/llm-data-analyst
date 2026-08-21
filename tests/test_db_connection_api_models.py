from __future__ import annotations

from backend.api.models import DBConnectionTableResponse


def test_db_connection_table_response_serializes_schema_alias() -> None:
    response = DBConnectionTableResponse(
        schema_name="public",
        name="orders",
        table_type="table",
        qualified_name="public.orders",
    )

    assert response.model_dump(mode="json", by_alias=True) == {
        "schema": "public",
        "name": "orders",
        "table_type": "table",
        "qualified_name": "public.orders",
    }
