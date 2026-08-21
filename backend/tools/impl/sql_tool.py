from __future__ import annotations

import re
from typing import TYPE_CHECKING, ClassVar, Literal

import pandas as pd
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from backend.data_access.data_catalog import format_dataframe_columns_hint
from backend.data_access.db_runtime_service import RuntimeDBConnectionConfig
from backend.data_access.semantic_query import (
    SemanticQuery,
    SemanticQueryFilter,
    SemanticQueryOrder,
)
from backend.data_access.sql_table_service import SQLTableService
from backend.tools.artifact_references import QUERY_META_ATTR, attach_query_metadata
from backend.tools.instructions import tool_description
from backend.tools.observations import exception_metadata
from backend.tools.schema_registry import (
    ColumnLineage,
    DataFrameSchemaEntry,
    infer_sql_alias_map,
)

if TYPE_CHECKING:
    from backend.tools.sandbox import SessionSandbox


SQLToolMode = Literal[
    "catalog_tables",
    "describe_table",
    "execute_sql",
    "semantic_query",
]


_RAW_SQL_RE = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
_CATALOG_SQL_RE = re.compile(r"\binformation_schema\.tables\b|\bshow\s+tables\b", re.IGNORECASE)
_SQL_ERROR_LINE_RE = re.compile(r"\bline\s+(\d+):", re.IGNORECASE)


def _sql_error_context(sql: str | None, error: Exception) -> str:
    match = _SQL_ERROR_LINE_RE.search(str(error))
    lines = str(sql or "").splitlines()
    if match is None or not lines:
        return ""
    line_number = int(match.group(1))
    if not 1 <= line_number <= len(lines):
        return ""
    start = max(1, line_number - 2)
    end = min(len(lines), line_number + 2)
    excerpt = [
        f"{'>' if number == line_number else ' '} {number:>4}: {lines[number - 1][:120]}"
        for number in range(start, end + 1)
    ]
    return f"SQL_CONTEXT around reported line {line_number}:\n" + "\n".join(excerpt)


class SQLToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: SQLToolMode | None = Field(
        default=None,
        description=(
            "Select one mode. When semantic context reports metric_resolution=resolved, use "
            "semantic_query with confirmed_metric_keys; do not inspect coverage or rebuild the "
            "formula in execute_sql. If requested dimensions are not allowed, choose a compatible "
            "complete top-k candidate already in context; resolve only when none is complete and "
            "unambiguous. Use execute_sql for physical SELECT/WITH only when no "
            "confirmed contract covers the calculation, or when resolved metrics span base "
            "tables and semantic execution_mode explicitly requires execute_sql. Use "
            "catalog_tables for table discovery and describe_table for unresolved schema."
        ),
    )
    question: str | None = Field(
        default=None,
        description="Естественно-языковой аналитический вопрос по доступным таблицам.",
    )
    sql: str | None = Field(
        default=None,
        description=(
            "Complete read-only SELECT/WITH for execute_sql. Select aggregate "
            "expressions directly with stable aliases, for example "
            "AVG(value) AS avg_value; apply presentation precision after the "
            "returned table is materialized. For wide-to-long SQL, project the value-table "
            "alias columns into the enclosing SELECT when using CROSS JOIN LATERAL (VALUES ...). "
            "After one alias or syntax failure, switch to explicit UNION ALL branches instead "
            "of retrying an equivalent LATERAL query. "
            "For independent slices, UNION ALL their final aggregates, not source rows; never "
            "form joint member combinations across unrelated wide families. Each member label "
            "must map only to its corresponding value column. When one request compares several "
            "independent dimensions, return their normalized rows from one final UNION ALL query "
            "instead of stitching separate SQL results in pandas. Unpivot peer entity columns inside "
            "their source table; do not join a lookup table unless the fact row has a verified key, "
            "and never use OR-ed nonzero measures as a join condition. For a latest-observed window, "
            "anchor filters to the source MAX(date), not today's date; a latest snapshot filters to "
            "that date instead of averaging all history beside MAX(date). Qualify each joined field "
            "with the alias of the table that declares it. Sandbox artifact names are not "
            "database relations: transform them with pandas_tool or filter the source query. "
            "Before sending, check exact table spelling, balanced CTEs, and every alias reference. "
            "After an error, never resend the same SQL; a missing column must be added to the "
            "immediate input SELECT or removed from the next query."
        ),
        examples=["SELECT segment, AVG(value) AS avg_value FROM source GROUP BY segment"],
    )
    metrics: list[str] = Field(
        default_factory=list,
        description=(
            "Confirmed metric keys returned by semantic resolution. Top-k candidates are context "
            "for disambiguation, not executable keys. Pass confirmed keys as a JSON array, not a "
            "serialized JSON string. All keys must share one compatible base table; otherwise use "
            "execute_sql with the catalog contracts and declared relationships."
        ),
    )
    dimensions: list[str] = Field(
        default_factory=list,
        description=(
            "Non-time semantic dimension references for semantic_query mode. Do not put time "
            "grain labels such as month or year here: time_dimension plus time_grain adds the "
            "grouped time column."
        ),
    )
    time_dimension: str | None = Field(
        default=None,
        description=(
            "Active semantic time dimension reference. Set it together with time_grain whenever "
            "the request specifies a temporal result grain."
        ),
    )
    time_grain: Literal["day", "week", "month", "quarter", "year"] | None = Field(
        default=None,
        description=(
            "Time grouping grain for semantic_query mode. Set it when the request specifies a "
            "temporal grain; omission keeps raw time values and does not infer a grain. Encode "
            "every requested temporal boundary in filters. Do not duplicate this grain in dimensions."
        ),
    )
    filters: list[SemanticQueryFilter] = Field(
        default_factory=list,
        description=(
            "Typed semantic filters as a JSON array of objects. Metric-defined filters are "
            "already compiled; do not repeat them. Every query filter field must be allowed "
            "by each selected metric. Include the complete requested time window in the same "
            "call; time_grain and limit do not bound dates."
        ),
    )
    order_by: list[SemanticQueryOrder] = Field(
        default_factory=list,
        description="Typed semantic ordering as a JSON array of objects.",
    )
    limit: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum semantic result rows.",
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
            "в snake_case, например women_by_district или monthly_sales. "
            "Every result reused later needs a distinct name because a later result with the "
            "same name replaces the earlier sandbox dataframe."
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
        legacy_table_names = [name for name in (self.table, self.table_name) if str(name or "").strip()]
        if legacy_table_names:
            self.table_names = [*self.table_names, *legacy_table_names]
        mode = self.mode

        if mode is None:
            if self.metrics:
                mode = "semantic_query"
            elif self.table_names:
                mode = "describe_table"
            elif sql_text:
                mode = "execute_sql"
            elif _RAW_SQL_RE.match(question_text):
                mode = "execute_sql"
                sql_text = question_text
            elif _CATALOG_SQL_RE.search(question_text):
                mode = "catalog_tables"
            else:
                raise ValueError("sql_tool mode is required")

        if mode == "execute_sql":
            if not sql_text and question_text and _RAW_SQL_RE.match(question_text):
                sql_text = question_text
            if not sql_text:
                raise ValueError("execute_sql mode requires sql")
        elif mode == "semantic_query":
            if not self.metrics:
                raise ValueError("semantic_query mode requires top-level metrics")
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

    def to_semantic_query(self) -> SemanticQuery | None:
        if self.mode != "semantic_query":
            return None
        return SemanticQuery(
            metrics=self.metrics,
            dimensions=self.dimensions,
            time_dimension=self.time_dimension,
            time_grain=self.time_grain,
            filters=self.filters,
            order_by=self.order_by,
            limit=self.limit,
        )


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
        db_runtime_config: RuntimeDBConnectionConfig | None = None,
        csv_loaded: bool = False,
        csv_session_id: str | None = None,
        max_rows: int = 1000,
        query_timeout_sec: float = 30.0,
        sandbox: SessionSandbox | None = None,
        candidates_cache_key: str | None = None,
        storage_dir: str | None = None,
        manifest_store: object | None = None,
        semantic_hints: dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        self._service = SQLTableService(
            db_runtime_config=db_runtime_config,
            csv_loaded=csv_loaded,
            csv_session_id=csv_session_id,
            max_rows=max_rows,
            query_timeout_sec=query_timeout_sec,
            candidates_cache_key=candidates_cache_key,
            storage_dir=storage_dir,
            manifest_store=manifest_store,
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
        metrics: list[str] | None = None,
        dimensions: list[str] | None = None,
        time_dimension: str | None = None,
        time_grain: Literal["day", "week", "month", "quarter", "year"] | None = None,
        filters: list[SemanticQueryFilter] | None = None,
        order_by: list[SemanticQueryOrder] | None = None,
        limit: int = 100,
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
                metrics=metrics or [],
                dimensions=dimensions or [],
                time_dimension=time_dimension,
                time_grain=time_grain,
                filters=filters or [],
                order_by=order_by or [],
                limit=limit,
                table_names=table_names or [],
                table=table,
                table_name=table_name,
                artifact_name=artifact_name,
            )
            return self._run_query(args)
        except Exception as exc:
            raw_error = str(exc)
            sql_excerpt = _sql_error_context(sql, exc)
            error_text = f"❌ Ошибка sql_tool: {raw_error}"
            if sql_excerpt:
                error_text = f"{error_text}\n{sql_excerpt}"
            return error_text, {
                "text": error_text,
                "status": "error",
                "error": raw_error,
                "sql_excerpt": sql_excerpt,
                **exception_metadata(exc),
            }

    def _run_query(
        self,
        request: SQLToolArgs | str,
        artifact_name: str | None = None,
    ) -> tuple[str, dict[str, object]]:
        import pandas as pd

        if isinstance(request, str):
            request = SQLToolArgs(question=request, artifact_name=artifact_name)
        clean_artifact_name = self._sanitize_artifact_name(request.artifact_name or artifact_name)
        if clean_artifact_name is None and request.mode == "execute_sql" and self._sandbox is not None:
            used_names = self._sandbox.get_user_scope()
            requested_sql = str(request.sql or "").strip()
            clean_artifact_name = next(
                (
                    name
                    for name, value in used_names.items()
                    if isinstance(value, pd.DataFrame)
                    and str(dict(value.attrs.get(QUERY_META_ATTR) or {}).get("requested_sql") or "").strip()
                    == requested_sql
                ),
                None,
            )
            if clean_artifact_name is None:
                clean_artifact_name = "sql_result"
                suffix = 2
                while clean_artifact_name in used_names:
                    clean_artifact_name = f"sql_result_{suffix}"
                    suffix += 1

        payload = self._service.build_table_artifact(
            request.question or request.sql or "",
            artifact_name=clean_artifact_name,
            mode=request.mode,
            sql=request.sql,
            table_names=request.table_names,
            semantic_query=request.to_semantic_query(),
        )
        from backend.data_access.dataframe_utils import deduplicate_dataframe_columns

        for name, data in list(payload.get("items", {}).items()):
            if isinstance(data, pd.DataFrame):
                payload["items"][name] = deduplicate_dataframe_columns(data)

        query_metadata = dict(dict(payload.get("meta") or {}).get("query") or {})
        truncated_result = query_metadata.get("truncated") is True
        bounded_result = query_metadata.get("has_more_rows") is True
        if query_metadata:
            for data in payload.get("items", {}).values():
                if isinstance(data, pd.DataFrame):
                    attach_query_metadata(data, query_metadata)

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
            row_previews: list[str] = []
            for name, data in payload["items"].items():
                if isinstance(data, pd.DataFrame):
                    schema_hints.append(format_dataframe_columns_hint(data, name=name))
                    if not truncated_result and not bounded_result and len(data) <= 20:
                        preview = repr(data.to_dict(orient="records"))[:4000]
                        row_previews.append(f"{name}: {preview}")
            schema_block = ""
            if schema_hints:
                schema_block = "\n" + "\n".join(schema_hints)
            if row_previews:
                schema_block += "\nROW_PREVIEW_FOR_LLM_CONTEXT:\n" + "\n".join(row_previews)
            if truncated_result:
                text = (
                    f"⚠️ TRUNCATED_RESULT from sql_tool: {item_names}. "
                    f"Variables {vars_hint} contain only a capped preview and are not "
                    "analysis-ready. Do not use them in pandas_tool or plotly_tool. "
                    "If the query is not yet at the final requested grain, aggregate it "
                    "in SQL. If it is already at the final grain, fetch complete "
                    "non-overlapping partitions with exhaustive predicates (prefer typed "
                    "time ranges), save each under a distinct artifact_name, and "
                    "concatenate only complete partitions once in pandas_tool. Do not "
                    "raise LIMIT or repeat an equivalent query under another artifact "
                    "name."
                    f"{schema_block}"
                )
            elif bounded_result:
                text = (
                    f"⚠️ BOUNDED_RESULT from sql_tool: {item_names}. Variables {vars_hint} "
                    "contain the requested explicit LIMIT, but additional rows exist. "
                    "Use this result only when that exact top-N is the intended final "
                    "output after complete aggregation. Otherwise remove LIMIT and "
                    "aggregate to the final requested grain before analysis or plotting. "
                    "Do not increase LIMIT incrementally."
                    f"{schema_block}"
                )
            else:
                text = (
                    f"✅ Выполнен sql_tool: {item_names}. "
                    f"Результаты доступны как Python-переменные: {vars_hint}. "
                    f"Используй эти имена напрямую в pandas_tool и plotly_tool. "
                    f"Не выполняй повторное чтение из БД, если нужный датафрейм уже создан."
                    f"{schema_block}"
                )
        else:
            text = f"✅ Выполнен sql_tool: {item_names}"

        semantic_metric_meta = dict(dict(payload.get("meta") or {}).get("semantic_metric") or {})
        semantic_metrics = [
            metric for metric in semantic_metric_meta.get("metrics") or [] if isinstance(metric, dict)
        ]
        if semantic_metrics:
            contracts = "; ".join(
                f"{metric.get('name') or metric.get('key')}: {metric.get('formula')}"
                for metric in semantic_metrics
            )
            text += f"\nSEMANTIC METRICS EXECUTED: {contracts}"

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
            str(item) for item in lineage_meta.get("source_table_names") or [] if str(item).strip()
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
