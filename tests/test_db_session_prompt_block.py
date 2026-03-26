"""Regression: DB-only sessions need explicit connection text in prompts."""

from __future__ import annotations

import pandas as pd

from backend.agent import AgentRunner
from backend.data_access import RuntimeDBConnectionConfig


def test_db_session_prompt_block_includes_connection_and_sql_hint_when_no_csv() -> None:
    rt = RuntimeDBConnectionConfig(
        connection_id="c1",
        user_id=1,
        name="Prod PG",
        db_type="postgresql",
        host="localhost",
        port=5432,
        database="analytics",
        username="u",
        password="p",
    )
    text = AgentRunner._db_session_prompt_block(
        session_source={"source_label": "Моя БД"},
        runtime=rt,
        df=None,
    )
    assert "Prod PG" in text
    assert "postgresql" in text
    assert "analytics" in text
    assert "c1" in text
    assert "Моя БД" in text
    assert "sql_table_tool" in text


def test_db_session_prompt_block_empty_without_runtime() -> None:
    assert (
        AgentRunner._db_session_prompt_block(
            session_source={},
            runtime=None,
            df=None,
        )
        == ""
    )


def test_db_session_prompt_block_omits_sql_paragraph_when_csv_loaded() -> None:
    rt = RuntimeDBConnectionConfig(
        connection_id="c1",
        user_id=1,
        name="DB",
        db_type="postgresql",
        host="h",
        port=None,
        database=None,
        username=None,
        password=None,
    )
    df = pd.DataFrame({"a": [1]})
    text = AgentRunner._db_session_prompt_block(
        session_source=None,
        runtime=rt,
        df=df,
    )
    assert "DB" in text
    assert "sql_table_tool" not in text
