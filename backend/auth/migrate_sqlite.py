from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from psycopg import sql
from psycopg.types.json import Jsonb

from backend.auth.app_data_postgres import AppDataPostgresStore
from backend.core.config import settings

TABLES = (
    "users",
    "auth_tokens",
    "user_settings",
    "user_tool_settings",
    "user_skill_settings",
    "mcp_server_configs",
    "mcp_server_secrets",
    "user_mcp_server_settings",
    "user_db_connections",
    "user_db_connection_secrets",
    "user_db_connection_access",
    "user_memories",
    "skill_overrides",
    "chat_sessions",
)


def _coerce_value(value: object, data_type: str) -> object:
    if value is None:
        return None
    if data_type == "boolean":
        return bool(value)
    if data_type in {"json", "jsonb"}:
        parsed = value if isinstance(value, (dict, list)) else json.loads(str(value))
        return Jsonb(parsed)
    if data_type == "timestamp with time zone" and isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, UTC)
    return value


def migrate_sqlite_app_data(sqlite_path: Path, store: AppDataPostgresStore) -> dict[str, int]:
    if not sqlite_path.is_file():
        raise FileNotFoundError(sqlite_path)
    store.ensure_schema()
    source = sqlite3.connect(sqlite_path)
    source.row_factory = sqlite3.Row
    copied: dict[str, int] = {}
    try:
        with store.connect() as target:
            target.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", ("app_data_sqlite_migration",))
            existing = {
                table: int(target.execute(f'SELECT COUNT(*) AS count FROM "{table}"').fetchone()["count"])
                for table in TABLES
            }
            populated = {table: count for table, count in existing.items() if count}
            if populated:
                raise RuntimeError(f"app_data is not empty: {sorted(populated)}")

            for table in TABLES:
                source_exists = source.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                    (table,),
                ).fetchone()
                if source_exists is None:
                    continue
                column_rows = target.execute(
                    """
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                    ORDER BY ordinal_position
                    """,
                    (store.schema, table),
                ).fetchall()
                source_columns = {str(row["name"]) for row in source.execute(f'PRAGMA table_info("{table}")')}
                columns = [
                    str(row["column_name"]) for row in column_rows if row["column_name"] in source_columns
                ]
                types = {str(row["column_name"]): str(row["data_type"]) for row in column_rows}
                selected_columns = ", ".join(f'"{column}"' for column in columns)
                rows = source.execute(f'SELECT {selected_columns} FROM "{table}"').fetchall()
                if not rows:
                    copied[table] = 0
                    continue
                values = [
                    tuple(_coerce_value(row[column], types[column]) for column in columns) for row in rows
                ]
                statement = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                    sql.Identifier(table),
                    sql.SQL(", ").join(map(sql.Identifier, columns)),
                    sql.SQL(", ").join(sql.Placeholder() for _ in columns),
                )
                target.executemany(statement, values)
                copied[table] = len(values)

            target.execute(
                """
                SELECT setval(
                    pg_get_serial_sequence('users', 'id'),
                    COALESCE((SELECT MAX(id) FROM users), 1),
                    EXISTS(SELECT 1 FROM users)
                )
                """
            )
    finally:
        source.close()
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy app data from SQLite to PostgreSQL once.")
    parser.add_argument("--sqlite-path", type=Path, required=True)
    args = parser.parse_args()
    copied = migrate_sqlite_app_data(
        args.sqlite_path,
        AppDataPostgresStore(
            settings.app_data_postgres_dsn,
            schema=settings.app_data_postgres_schema,
        ),
    )
    print(json.dumps(copied, sort_keys=True))


if __name__ == "__main__":
    main()
