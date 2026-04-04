"""Lightweight database exploration tool — no LLM, direct catalog queries.

The agent calls ``database_tool`` for quick structural operations:
list tables, describe columns, preview rows, list schemas.
For complex analytical SQL queries, use ``sql_tool`` instead.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from backend.data_access.db_runtime_service import RuntimeDBConnectionConfig
from backend.tools.impl.db_helpers import DBAnalyticsHelper

if TYPE_CHECKING:
    from backend.tools.sandbox import SessionSandbox


class DatabaseToolArgs(BaseModel):
    action: Literal["list_tables", "describe_table", "preview", "list_schemas"] = Field(
        ...,
        description=(
            "Действие: "
            "list_tables — список таблиц, "
            "describe_table — колонки таблицы, "
            "preview — первые N строк таблицы, "
            "list_schemas — список схем."
        ),
    )
    table: str | None = Field(
        default=None,
        description="Имя таблицы (для describe_table и preview).",
    )
    db_schema: str | None = Field(
        default=None,
        description="Схема БД (опционально, по умолчанию — public/default).",
    )
    limit: int = Field(
        default=10,
        description="Количество строк для preview (макс. 50).",
    )


class DatabaseTool(BaseTool):
    """Direct DB catalog queries without LLM-generated SQL."""

    name: str = "database_tool"
    description: str = (
        "Быстрый просмотр структуры БД: список таблиц, описание колонок, "
        "превью строк, список схем. Не требует SQL — прямые вызовы к каталогу БД. "
        "Используй для разведки данных перед аналитическими запросами."
    )
    args_schema: type[BaseModel] = DatabaseToolArgs
    response_format: str = "content_and_artifact"

    _db_runtime_config: RuntimeDBConnectionConfig = PrivateAttr()
    _sandbox: SessionSandbox | None = PrivateAttr(default=None)
    _timeout_sec: float = PrivateAttr(default=15.0)

    def __init__(
        self,
        *,
        db_runtime_config: RuntimeDBConnectionConfig,
        sandbox: SessionSandbox | None = None,
        timeout_sec: float = 15.0,
    ) -> None:
        super().__init__()
        self._db_runtime_config = db_runtime_config
        self._sandbox = sandbox
        self._timeout_sec = timeout_sec

    def _db(self) -> DBAnalyticsHelper:
        return DBAnalyticsHelper(
            runtime=self._db_runtime_config,
            timeout_sec=self._timeout_sec,
        )

    def _run(
        self,
        action: str,
        table: str | None = None,
        db_schema: str | None = None,
        limit: int = 10,
    ) -> tuple[str, dict[str, Any]]:
        db = self._db()
        limit = max(1, min(limit, 50))

        if action == "list_tables":
            return self._action_list_tables(db, db_schema)
        if action == "describe_table":
            if not table:
                return "Ошибка: укажи имя таблицы (аргумент table).", {}
            return self._action_describe_table(db, table, db_schema)
        if action == "preview":
            if not table:
                return "Ошибка: укажи имя таблицы (аргумент table).", {}
            return self._action_preview(db, table, db_schema, limit)
        if action == "list_schemas":
            return self._action_list_schemas(db)

        return f"Неизвестное действие: {action}", {}

    def _action_list_tables(
        self, db: DBAnalyticsHelper, schema: str | None,
    ) -> tuple[str, dict[str, Any]]:
        rows = db.list_tables_with_columns(schema)

        # Fallback: if the explicitly requested schema is empty,
        # enumerate all real schemas and collect tables from each.
        if not rows and schema is not None:
            rows = self._collect_tables_all_schemas(db)

        if not rows:
            schemas = db.list_schemas()
            schema_hint = (
                f" Доступные схемы: {[s['name'] for s in schemas]}."
                if schemas
                else " Схемы не обнаружены."
            )
            return f"В базе нет таблиц.{schema_hint}", {}

        df = pd.DataFrame(rows)
        artifact_name = "db_tables"
        self._inject(artifact_name, df)

        found_schemas = sorted({r.get("schema", "") for r in rows if r.get("schema")})
        schema_info = f" (схемы: {', '.join(found_schemas)})" if found_schemas else ""
        table_names = sorted({r.get("table_name") or r.get("name", "") for r in rows if r.get("table_name") or r.get("name")})  # noqa: E501
        tables_list = ", ".join(f"`{t}`" for t in table_names) if table_names else "—"
        text = (
            f"✅ Найдено {len(rows)} таблиц{schema_info}: {tables_list}. "
            f"Результат в переменной `{artifact_name}`."
        )
        return text, self._table_artifact(artifact_name, df)

    @staticmethod
    def _collect_tables_all_schemas(db: DBAnalyticsHelper) -> list[dict]:
        """Enumerate every real schema and collect tables across all of them."""
        schemas = db.list_schemas()
        combined: list[dict] = []
        for s in schemas:
            name = s.get("name", "")
            if not name:
                continue
            rows = db.list_tables_with_columns(name)
            combined.extend(rows)
        return combined

    @staticmethod
    def _split_qualified(table: str, schema: str | None) -> tuple[str, str | None]:
        """Split 'schema.table' into (table, schema). Schema arg takes priority."""
        if schema is not None:
            return table, schema
        if "." in table:
            parts = table.split(".", 1)
            return parts[1], parts[0]
        return table, None

    def _action_describe_table(
        self, db: DBAnalyticsHelper, table: str, schema: str | None,
    ) -> tuple[str, dict[str, Any]]:
        table, schema = self._split_qualified(table, schema)
        cols = db.describe_table(table, schema=schema)
        if not cols:
            return f"Таблица '{table}' не найдена или не содержит колонок.", {}

        df = pd.DataFrame(cols)
        artifact_name = f"columns_{table.replace('.', '_')}"
        self._inject(artifact_name, df)

        text = (
            f"✅ Таблица `{table}`: {len(cols)} колонок. "
            f"Результат в переменной `{artifact_name}`."
        )
        return text, self._table_artifact(artifact_name, df)

    def _action_preview(
        self, db: DBAnalyticsHelper, table: str, schema: str | None, limit: int,
    ) -> tuple[str, dict[str, Any]]:
        table, schema = self._split_qualified(table, schema)
        df = db.preview_table(table, schema=schema, limit=limit)
        if df is None or df.empty:
            return f"Таблица '{table}' пуста или не найдена.", {}

        artifact_name = f"preview_{table.replace('.', '_')}"
        self._inject(artifact_name, df)

        text = (
            f"✅ Первые {len(df)} строк таблицы `{table}`. "
            f"Колонки: {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}. "
            f"Результат в переменной `{artifact_name}`."
        )
        return text, self._table_artifact(artifact_name, df)

    def _action_list_schemas(
        self, db: DBAnalyticsHelper,
    ) -> tuple[str, dict[str, Any]]:
        schemas = db.list_schemas()
        if not schemas:
            return "Схемы не найдены.", {}

        df = pd.DataFrame(schemas)
        artifact_name = "db_schemas"
        self._inject(artifact_name, df)

        text = f"✅ Найдено {len(schemas)} схем. Результат в переменной `{artifact_name}`."
        return text, self._table_artifact(artifact_name, df)

    def _inject(self, name: str, df: pd.DataFrame) -> None:
        if self._sandbox is not None:
            self._sandbox.put(name, df)

    @staticmethod
    def _table_artifact(name: str, df: pd.DataFrame) -> dict[str, Any]:
        # Use "table" key so ToolCollector in callbacks.py registers the artifact.
        return {
            "schema_version": "1.0",
            "artifact_type": "table",
            "table": {name: df},
        }
