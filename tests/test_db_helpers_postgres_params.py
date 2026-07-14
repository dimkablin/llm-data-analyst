"""PostgreSQL ILIKE with Cyrillic must use query params (psycopg % placeholders)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from backend.data_access.db_runtime_service import RuntimeDBConnectionConfig
from backend.tools.impl.db_helpers import (
    DBAnalyticsHelper,
    _escape_literal_percent_for_psycopg,
)


def test_escape_literal_percent_doubles_like_wildcards() -> None:
    sql = "WHERE company_name ILIKE '%Норд%' AND sector ILIKE '%энергет%'"
    assert _escape_literal_percent_for_psycopg(sql) == (
        "WHERE company_name ILIKE '%%Норд%%' AND sector ILIKE '%%энергет%%'"
    )


def test_escape_literal_percent_keeps_psycopg_placeholders() -> None:
    sql = "WHERE company_name ILIKE %s AND id = %s"
    assert _escape_literal_percent_for_psycopg(sql) == sql


def test_query_dataframe_passes_params_for_ilike() -> None:
    runtime = RuntimeDBConnectionConfig(
        connection_id="c1",
        user_id=1,
        name="test",
        db_type="postgresql",
        host="localhost",
        port=5432,
        database="tabular",
        username="analyst",
        password="analyst",
        options={"schema": "demo_invest"},
    )
    helper = DBAnalyticsHelper(runtime=runtime, timeout_sec=5.0)
    captured: dict[str, object] = {}

    def fake_postgres(sql: str, params: tuple | None = None) -> pd.DataFrame:
        captured["sql"] = sql
        captured["params"] = params
        return pd.DataFrame([{"ticker": "NREH"}])

    with patch.object(helper, "_postgres_query_dataframe", side_effect=fake_postgres):
        df = helper.query_dataframe(
            "SELECT ticker FROM demo_invest.instrument_snapshot_demo "
            "WHERE company_name ILIKE %s LIMIT 1",
            (f"%{'Норд'}%",),
        )

    assert captured["sql"] == (
        "SELECT ticker FROM demo_invest.instrument_snapshot_demo "
        "WHERE company_name ILIKE %s LIMIT 1"
    )
    assert captured["params"] == ("%Норд%",)
    assert df.iloc[0]["ticker"] == "NREH"
