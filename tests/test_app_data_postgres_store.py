from __future__ import annotations

import psycopg
from psycopg.types.json import Jsonb

from backend.auth.app_data_postgres import (
    AppDataPostgresConnection,
    load_app_data_migrations,
)
from backend.auth.migrate_sqlite import _coerce_value
from backend.sessions.postgres_session_store import PostgresSessionStore


def test_initial_app_data_migration_defines_persistent_boundaries() -> None:
    migrations = load_app_data_migrations()

    assert migrations[0].version == 1
    sql = migrations[0].sql.casefold()
    required_tables = (
        "users",
        "auth_tokens",
        "user_settings",
        "user_db_connections",
        "chat_sessions",
        "session_state",
        "session_messages",
        "session_artifacts",
        "session_manifests",
        "session_notebooks",
        "stored_blobs",
    )
    assert all(f"create table {table}" in sql for table in required_tables)


def test_postgres_connection_adapts_existing_qmark_queries(monkeypatch) -> None:
    class FakeConnection:
        def __init__(self) -> None:
            self.calls: list[tuple[object, tuple[object, ...]]] = []

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def execute(self, query: object, params: tuple[object, ...] = ()) -> object:
            self.calls.append((query, params))
            return query

    raw = FakeConnection()
    monkeypatch.setattr(psycopg, "connect", lambda *_args, **_kwargs: raw)

    with AppDataPostgresConnection("postgresql://unused", schema="app") as connection:
        connection.execute("SELECT ?", ("value",))

    assert raw.calls[-1] == ("SELECT %s", ("value",))


def test_sqlite_migration_coerces_postgres_types() -> None:
    assert _coerce_value(1, "boolean") is True
    assert _coerce_value(0, "boolean") is False
    value = _coerce_value('{"key": "value"}', "jsonb")
    assert isinstance(value, Jsonb)
    assert value.obj == {"key": "value"}


def test_postgres_session_initialization_does_not_expire_persistent_sessions(tmp_path) -> None:
    class AppDataStore:
        initialized = False

        def ensure_schema(self) -> None:
            self.initialized = True

        def connect(self):
            raise AssertionError("Persistent session initialization must not delete by age")

    app_data_store = AppDataStore()
    store = PostgresSessionStore(
        str(tmp_path),
        app_data_store=app_data_store,  # type: ignore[arg-type]
    )

    store.initialize()

    assert app_data_store.initialized is True
