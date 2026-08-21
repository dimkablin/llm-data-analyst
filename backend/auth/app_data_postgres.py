from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AppDataMigration:
    version: int
    name: str
    checksum: str
    sql: str


def load_app_data_migrations() -> tuple[AppDataMigration, ...]:
    migrations_dir = Path(__file__).with_name("migrations")
    migrations: list[AppDataMigration] = []
    for path in sorted(migrations_dir.glob("*.sql")):
        version_text, separator, name = path.stem.partition("_")
        if not separator or not version_text.isdigit():
            raise ValueError(f"Invalid app_data migration name: {path.name}")
        sql_text = path.read_text(encoding="utf-8")
        migrations.append(
            AppDataMigration(
                version=int(version_text),
                name=name,
                checksum=hashlib.sha256(sql_text.encode("utf-8")).hexdigest(),
                sql=sql_text,
            )
        )
    versions = [migration.version for migration in migrations]
    if len(versions) != len(set(versions)):
        raise ValueError("Duplicate app_data migration version")
    return tuple(migrations)


class AppDataPostgresStore:
    """Own the app_data PostgreSQL schema lifecycle."""

    def __init__(self, dsn: str, *, schema: str = "public") -> None:
        self.dsn = str(dsn or "").strip()
        self.schema = str(schema or "").strip()
        if not self.dsn:
            raise ValueError("APP_DATABASE_URL must be set")
        if not self.schema:
            raise ValueError("APP_DATABASE_SCHEMA must be set")

    def ensure_schema(self) -> None:
        import psycopg
        from psycopg import sql

        migrations = load_app_data_migrations()
        with psycopg.connect(self.dsn) as connection:
            with connection.transaction():
                connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtext(%s))",
                    (f"app_data_schema:{self.schema}",),
                )
                connection.execute(
                    sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(self.schema))
                )
                connection.execute(sql.SQL("SET LOCAL search_path TO {}").format(sql.Identifier(self.schema)))
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version integer PRIMARY KEY,
                        name text NOT NULL,
                        checksum text NOT NULL,
                        applied_at timestamptz NOT NULL DEFAULT now()
                    )
                    """
                )
                rows = connection.execute("SELECT version, checksum FROM schema_migrations").fetchall()
                applied = {int(row[0]): str(row[1]) for row in rows}
                for migration in migrations:
                    existing_checksum = applied.get(migration.version)
                    if existing_checksum is not None:
                        if existing_checksum != migration.checksum:
                            raise RuntimeError(
                                f"app_data migration checksum mismatch: {migration.version}_{migration.name}"
                            )
                        continue
                    connection.execute(migration.sql)
                    connection.execute(
                        """
                        INSERT INTO schema_migrations(version, name, checksum)
                        VALUES (%s, %s, %s)
                        """,
                        (migration.version, migration.name, migration.checksum),
                    )

    def connect(self) -> AppDataPostgresConnection:
        return AppDataPostgresConnection(self.dsn, schema=self.schema)


class AppDataPostgresConnection:
    """Small DB-API compatibility boundary for the existing app stores."""

    def __init__(self, dsn: str, *, schema: str) -> None:
        import psycopg
        from psycopg.rows import dict_row

        self._connection = psycopg.connect(dsn, row_factory=dict_row)
        self._schema = schema

    def __enter__(self) -> AppDataPostgresConnection:
        from psycopg import sql

        self._connection.__enter__()
        self._connection.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(self._schema)))
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._connection.__exit__(exc_type, exc, traceback)

    def execute(self, query: str, params: tuple[object, ...] = ()) -> Any:
        return self._connection.execute(query.replace("?", "%s"), params)

    def executemany(self, query: object, params: list[tuple[object, ...]]) -> None:
        self._connection.cursor().executemany(query, params)
