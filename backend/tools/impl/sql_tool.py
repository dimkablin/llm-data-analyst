from __future__ import annotations

import re
from typing import TYPE_CHECKING, ClassVar, Literal

import pandas as pd
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from backend.data_access.data_catalog import format_dataframe_columns_hint
from backend.data_access.db_runtime_service import RuntimeDBConnectionConfig
from backend.data_access.sql_table_service import SQLTableService
from backend.tools.instructions import tool_description
from backend.tools.schema_registry import (
    ColumnLineage,
    DataFrameSchemaEntry,
    infer_sql_alias_map,
)

if TYPE_CHECKING:
    from backend.tools.sandbox import SessionSandbox


SQLToolMode = Literal["catalog_tables", "describe_table", "execute_sql", "nl_query"]


_RAW_SQL_RE = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
_CATALOG_SQL_RE = re.compile(r"\binformation_schema\.tables\b|\bshow\s+tables\b", re.IGNORECASE)


class SQLToolArgs(BaseModel):
    mode: SQLToolMode | None = Field(
        default=None,
        description=(
            "Режим работы: catalog_tables для каталога, describe_table для схемы, "
            "execute_sql для готового SELECT/WITH, nl_query для аналитического вопроса."
        ),
    )
    question: str | None = Field(
        default=None,
        description="Естественно-языковой аналитический вопрос по доступным таблицам.",
    )
    sql: str | None = Field(
        default=None,
        description="Готовый read-only SQL SELECT/WITH для execute_sql.",
    )
    table_names: list[str] = Field(
        default_factory=list,
        description="Имена таблиц для describe_table.",
    )
    table: str | None = Field(
        default=None,
        description="Backwards-compatible alias for one table name in describe_table mode.",
    )
    table_name: str | None = Field(
        default=None,
        description="Backwards-compatible alias for one table name in describe_table mode.",
    )
    artifact_name: str | None = Field(
        default=None,
        description=(
            "Необязательное желаемое имя Python-переменной для результата "
            "в snake_case, например women_by_district или monthly_sales."
        ),
    )

    @field_validator("table_names", mode="before")
    @classmethod
    def coerce_table_names(cls, value: object) -> object:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (tuple, set)):
            return list(value)
        return value

    @model_validator(mode="after")
    def normalize_and_validate_mode(self) -> SQLToolArgs:
        question_text = str(self.question or "").strip()
        sql_text = str(self.sql or "").strip()
        legacy_table_names = [
            name
            for name in (self.table, self.table_name)
            if str(name or "").strip()
        ]
        if legacy_table_names:
            self.table_names = [*self.table_names, *legacy_table_names]
        mode = self.mode

        if mode is None:
            if self.table_names:
                mode = "describe_table"
            elif sql_text:
                mode = "execute_sql"
            elif _RAW_SQL_RE.match(question_text):
                mode = "execute_sql"
                sql_text = question_text
            elif _CATALOG_SQL_RE.search(question_text):
                mode = "catalog_tables"
            else:
                mode = "nl_query"

        if mode == "execute_sql":
            if not sql_text and question_text and _RAW_SQL_RE.match(question_text):
                sql_text = question_text
            if not sql_text:
                raise ValueError("execute_sql mode requires sql")
        elif mode == "nl_query":
            if not question_text:
                raise ValueError("nl_query mode requires question")
        elif mode == "describe_table":
            cleaned = [str(item).strip() for item in self.table_names if str(item).strip()]
            cleaned = list(dict.fromkeys(cleaned))
            if not cleaned:
                raise ValueError("describe_table mode requires table_names")
            self.table_names = cleaned
        elif mode == "catalog_tables":
            pass

        self.mode = mode
        self.sql = sql_text or None
        self.question = question_text or None
        return self


class SQLToolPayload(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    schema_version: str = Field(default="1.0")
    artifact_type: Literal["table"]
    items: dict[str, object] = Field(min_length=1)
    source: dict[str, object] = Field(default_factory=dict)
    recipe: list[dict[str, object]] = Field(default_factory=list)
    meta: dict[str, object] = Field(default_factory=dict)


class SQLTool(BaseTool):
    name: str = "sql_tool"
    description: str = tool_description("sql_tool")
    args_schema: type[BaseModel] = SQLToolArgs
    response_format: str = "content_and_artifact"
    parallel_safe: ClassVar[bool] = False

    _service: SQLTableService = PrivateAttr()
    _sandbox: SessionSandbox | None = PrivateAttr(default=None)

    def __init__(
        self,
        *,
        llm_base_url: str,
        llm_model: str,
        llm_api_key: str | None,
        llm_enable_thinking: bool = False,
        llm_chat_template_kwargs_enabled: bool = True,
        llm_provider: str = "",
        db_runtime_config: RuntimeDBConnectionConfig | None = None,
        csv_loaded: bool = False,
        csv_session_id: str | None = None,
        max_rows: int = 200,
        sandbox: SessionSandbox | None = None,
        candidates_cache_key: str | None = None,
        storage_dir: str | None = None,
        semantic_context_prompt: str = "",
        semantic_hints: dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        self._service = SQLTableService(
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_enable_thinking=llm_enable_thinking,
            llm_chat_template_kwargs_enabled=llm_chat_template_kwargs_enabled,
            llm_provider=llm_provider,
            db_runtime_config=db_runtime_config,
            csv_loaded=csv_loaded,
            csv_session_id=csv_session_id,
            max_rows=max_rows,
            candidates_cache_key=candidates_cache_key,
            storage_dir=storage_dir,
            semantic_context_prompt=semantic_context_prompt,
            semantic_hints=semantic_hints,
        )
        self._sandbox = sandbox

    def _sanitize_artifact_name(self, value: str | None) -> str | None:
        text = str(value or "").strip().lower()
        if not text:
            return None

        text = re.sub(r"\W+", "_", text, flags=re.UNICODE)
        text = re.sub(r"_+", "_", text).strip("_")

        if not text:
            return None

        if text[0].isdigit():
            text = f"result_{text}"

        return text[:80]

    def _run(
        self,
        question: str | None = None,
        mode: SQLToolMode | None = None,
        sql: str | None = None,
        table_names: list[str] | str | None = None,
        table: str | None = None,
        table_name: str | None = None,
        artifact_name: str | None = None,
    ) -> tuple[str, dict[str, object]]:
        try:
            args = SQLToolArgs(
                mode=mode,
                question=question,
                sql=sql,
                table_names=table_names or [],
                table=table,
                table_name=table_name,
                artifact_name=artifact_name,
            )
            return self._run_query(args)
        except Exception as exc:
            error_text = f"❌ Ошибка sql_tool: {exc}"
            return error_text, {"text": error_text}

    def _run_query(
        self,
        request: SQLToolArgs | str,
        artifact_name: str | None = None,
    ) -> tuple[str, dict[str, object]]:
        import pandas as pd

        if isinstance(request, str):
            request = SQLToolArgs(question=request, artifact_name=artifact_name)
        clean_artifact_name = self._sanitize_artifact_name(request.artifact_name or artifact_name)

        payload = self._service.build_table_artifact(
            request.question or request.sql or "",
            artifact_name=clean_artifact_name,
            mode=request.mode,
            sql=request.sql,
            table_names=request.table_names,
        )
        from backend.data_access.dataframe_utils import deduplicate_dataframe_columns

        for name, data in list(payload.get("items", {}).items()):
            if isinstance(data, pd.DataFrame):
                payload["items"][name] = deduplicate_dataframe_columns(data)

        item_names = ", ".join(payload["items"].keys())

        # Inject result DataFrames into sandbox scope so subsequent tools
        # (plotly_tool, pandas_tool) can reference them by variable name.
        injected: list[str] = []
        if self._sandbox is not None:
            self._register_source_table_schemas(payload)
            for name, data in payload["items"].items():
                if isinstance(data, pd.DataFrame):
                    self._sandbox.put(
                        name,
                        data,
                        schema_entry=self._schema_entry_for_payload_item(name, data, payload),
                    )
                    injected.append(name)

        if injected:
            vars_hint = ", ".join(f"`{v}`" for v in injected)
            schema_hints: list[str] = []
            for name, data in payload["items"].items():
                if isinstance(data, pd.DataFrame):
                    schema_hints.append(format_dataframe_columns_hint(data, name=name))
            schema_block = ""
            if schema_hints:
                schema_block = "\n" + "\n".join(schema_hints)
            text = (
                f"✅ Выполнен sql_tool: {item_names}. "
                f"Результаты доступны как Python-переменные: {vars_hint}. "
                f"Используй эти имена напрямую в pandas_tool и plotly_tool. "
                f"Не выполняй повторное чтение из БД, если нужный датафрейм уже создан."
                f"{schema_block}"
            )
        else:
            text = f"✅ Выполнен sql_tool: {item_names}"

        result: dict[str, object] = {
            "text": text,
            "schema_version": "1.0",
            "artifact_type": "table",
            "items": payload["items"],
            "table": payload["items"],
        }
        if "source" in payload:
            result["source"] = payload["source"]
        if "recipe" in payload:
            result["recipe"] = payload["recipe"]
        if "meta" in payload:
            result["meta"] = payload["meta"]

        SQLToolPayload.model_validate(result)
        self._log_to_notebook(request.question or request.sql or "", payload)
        return text, result

    def _register_source_table_schemas(self, payload: dict) -> None:
        if self._sandbox is None:
            return
        registry = getattr(self._sandbox, "schema_registry", None)
        if registry is None:
            return
        meta = dict(payload.get("meta") or {})
        lineage = dict(meta.get("lineage") or {})
        for raw in lineage.get("source_tables") or []:
            if not isinstance(raw, dict):
                continue
            table_name = str(raw.get("qualified_name") or raw.get("table_name") or "").strip()
            columns = [str(col) for col in raw.get("columns") or [] if str(col).strip()]
            if table_name and columns:
                registry.register_source_table(table_name, columns)

        for data in payload.get("items", {}).values():
            if not isinstance(data, pd.DataFrame):
                continue
            if {"table_name", "column_name"} <= {str(col) for col in data.columns}:
                for table_name, group in data.groupby("table_name", dropna=True):
                    columns = [str(col) for col in group["column_name"].tolist() if str(col).strip()]
                    registry.register_source_table(str(table_name), columns)

    @staticmethod
    def _schema_entry_for_payload_item(
        name: str,
        data: pd.DataFrame,
        payload: dict,
    ) -> DataFrameSchemaEntry:
        meta = dict(payload.get("meta") or {})
        query_meta = dict(meta.get("query") or {})
        lineage_meta = dict(meta.get("lineage") or {})
        sql = str(query_meta.get("requested_sql") or query_meta.get("executed_sql") or "")
        alias_map = infer_sql_alias_map(sql, [str(col) for col in data.columns])
        source_tables = [
            str(item)
            for item in lineage_meta.get("source_table_names") or []
            if str(item).strip()
        ]
        lineage: dict[str, list[ColumnLineage]] = {}
        for source_column, output_column in alias_map.items():
            lineage.setdefault(output_column, []).append(
                ColumnLineage(
                    output_column=output_column,
                    source_column=source_column,
                    source_table=source_tables[0] if len(source_tables) == 1 else None,
                )
            )
        source_kind = "sql_result" if query_meta else "table_schema"
        return DataFrameSchemaEntry(
            variable_name=name,
            source_kind=source_kind,
            source_name=name,
            columns=[str(col) for col in data.columns],
            alias_map=alias_map,
            source_tables=source_tables,
            lineage=lineage,
        )

    def _log_to_notebook(self, question: str, payload: dict) -> None:
        if self._sandbox is None:
            return
        try:
            import pandas as pd

            items = payload.get("items", {})
            parts = []
            for name, data in items.items():
                if isinstance(data, pd.DataFrame):
                    parts.append(f"{name}: {data.shape[0]}x{data.shape[1]}")
                else:
                    parts.append(str(name))
            result_summary = ", ".join(parts) or "—"

            recipe = payload.get("recipe")
            sql = ""
            if isinstance(recipe, dict):
                sql = str(recipe.get("sql") or recipe.get("code") or "")
            elif isinstance(recipe, list):
                first_sql = next(
                    (
                        item
                        for item in recipe
                        if isinstance(item, dict) and (item.get("sql") or item.get("code"))
                    ),
                    {},
                )
                if isinstance(first_sql, dict):
                    sql = str(first_sql.get("sql") or first_sql.get("code") or "")

            self._sandbox.log_code_entry(
                tool_name="sql_tool",
                language="sql",
                question=question,
                code=sql,
                result_summary=result_summary,
            )
        except Exception:
            pass
