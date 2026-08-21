from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from backend.artifacts.artifact_meta import build_db_metadata_recipe_step, build_sql_recipe_step
from backend.data_access.csv_session_runtime import CSVSessionRuntime
from backend.data_access.db_runtime_service import RuntimeDBConnectionConfig
from backend.data_access.semantic_models import SemanticCatalog
from backend.data_access.semantic_query import (
    SemanticQuery,
    SemanticQueryCompiler,
)
from backend.data_access.session_catalog_cache import get_or_build_candidates
from backend.notebook.manifest_store import ManifestStore
from backend.notebook.session_source import is_duckdb_source_type
from backend.tools.impl.db_helpers import (
    MAX_RESULT_CELLS,
    DBAnalyticsHelper,
    _assert_read_only_sql,
    _normalize_analytic_sql,
    _normalize_dataframe,
)

_SQL_START = re.compile(r"\b(SELECT|WITH)\b", re.IGNORECASE)
_SQL_STOP_MARKERS = [
    "\n```",
    "\n####",
    "\n###",
    "\n##",
    "\n#",
    "\n**",
    "\n-----",
]


@dataclass
class TableCandidate:
    source_kind: str
    dialect: str
    table_name: str
    qualified_name: str
    schema: str | None
    columns: list[str]
    source_label: str
    source_ref_id: str
    db_runtime: RuntimeDBConnectionConfig | None = None
    csv_session_id: str | None = None
    file_name: str | None = None
    display_name: str | None = None
    source_alias: str | None = None
    schema_hint: dict[str, str] = field(default_factory=dict)
    preprocessing_summary: dict[str, Any] = field(default_factory=dict)
    row_count: int | None = None
    column_count: int | None = None


def clean_sql(raw: str) -> str:
    text = str(raw or "").strip()
    text = re.sub(r"^```sql\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^sql\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    text = text.strip("`").strip()

    match = _SQL_START.search(text)
    if not match:
        return text.split("```")[0].strip().strip(";").strip()

    sql = text[match.start() :]
    cut_pos: int | None = None
    for marker in _SQL_STOP_MARKERS:
        pos = sql.find(marker)
        if pos != -1:
            cut_pos = pos if cut_pos is None else min(cut_pos, pos)

    if cut_pos is not None:
        sql = sql[:cut_pos]

    semi = sql.find(";")
    if semi != -1:
        sql = sql[:semi]

    return sql.strip().strip(";").strip()


def is_select_or_with(sql: str) -> bool:
    s = str(sql or "").lstrip().lower()
    return s.startswith("select") or s.startswith("with")


def wrap_limit0(sql: str) -> str:
    return f"SELECT * FROM ({sql}) AS q LIMIT 0"


def _cap_dataframe_rows(rows: pd.DataFrame, *, max_rows: int) -> tuple[pd.DataFrame, bool]:
    if len(rows) <= max_rows:
        return rows, False
    return rows.head(max_rows).copy(), True


def _shrink_for_cell_budget(
    rows: pd.DataFrame,
    *,
    cell_budget: int = MAX_RESULT_CELLS,
) -> tuple[pd.DataFrame, bool]:
    if rows.empty or rows.shape[1] <= 0:
        return rows, False
    max_rows = max(1, cell_budget // max(1, rows.shape[1]))
    if len(rows) <= max_rows:
        return rows, False
    return rows.head(max_rows).copy(), True


class SQLTableService:
    def __init__(
        self,
        *,
        db_runtime_config: RuntimeDBConnectionConfig | None = None,
        csv_loaded: bool = False,
        csv_session_id: str | None = None,
        max_rows: int = 200,
        query_timeout_sec: float = 30.0,
        candidates_cache_key: str | None = None,
        storage_dir: str | Path | None = None,
        manifest_store: ManifestStore | None = None,
        semantic_hints: dict[str, Any] | None = None,
    ) -> None:
        self.db_runtime_config = db_runtime_config
        self.csv_loaded = bool(csv_loaded)
        self.csv_session_id = str(csv_session_id or "").strip() or None
        self.max_rows = max(1, min(int(max_rows), 1000))
        self.query_timeout_sec = max(1.0, float(query_timeout_sec))
        self.csv_runtime = CSVSessionRuntime()
        self._cached_db_helper: DBAnalyticsHelper | None = None
        self._cached_candidates: list[TableCandidate] | None = None
        self._candidates_cache_key = str(candidates_cache_key or "").strip() or None
        self.storage_dir = Path(storage_dir).resolve() if storage_dir is not None else None
        self.manifest_store = manifest_store
        self._cached_csv_source_metadata: dict[str, dict[str, Any]] | None = None
        self.semantic_hints = dict(semantic_hints or {})

    @staticmethod
    def _sanitize_artifact_name(value: str | None) -> str | None:
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

    def _db_helper(self) -> DBAnalyticsHelper:
        if self.db_runtime_config is None:
            raise ValueError("DB runtime is not configured")
        if self._cached_db_helper is None:
            self._cached_db_helper = DBAnalyticsHelper(
                runtime=self.db_runtime_config,
                timeout_sec=self.query_timeout_sec,
            )
        return self._cached_db_helper

    def _collect_db_candidates(self) -> list[TableCandidate]:
        if self.db_runtime_config is None:
            return []

        helper = self._db_helper()
        rows = helper.list_effective_tables_with_columns()
        out: list[TableCandidate] = []
        for row in rows:
            schema = row.get("schema")
            table_name = str(row.get("table_name") or "").strip()
            if not table_name:
                continue
            qualified_name = str(row.get("qualified_name") or table_name).strip()
            columns = [str(c) for c in row.get("columns", []) if str(c).strip()]
            out.append(
                TableCandidate(
                    source_kind="db",
                    dialect=str(self.db_runtime_config.db_type or "sql"),
                    table_name=table_name,
                    qualified_name=qualified_name,
                    schema=str(schema).strip() if schema else None,
                    columns=columns,
                    source_label=str(self.db_runtime_config.name or "DB source"),
                    source_ref_id=str(self.db_runtime_config.connection_id),
                    db_runtime=self.db_runtime_config,
                )
            )
        return out

    def _csv_source_metadata_by_table(self) -> dict[str, dict[str, Any]]:
        if self._cached_csv_source_metadata is not None:
            return self._cached_csv_source_metadata
        metadata: dict[str, dict[str, Any]] = {}
        if self.storage_dir is None or not self.csv_session_id:
            self._cached_csv_source_metadata = metadata
            return metadata
        try:
            manifest = (self.manifest_store or ManifestStore(self.storage_dir)).load(
                self.csv_session_id
            )
        except Exception:
            self._cached_csv_source_metadata = metadata
            return metadata

        for source in manifest.sources:
            if not is_duckdb_source_type(source.source_type):
                continue
            for table_name in source.csv_table_names:
                clean_name = str(table_name or "").strip()
                if not clean_name:
                    continue
                metadata[clean_name] = {
                    "file_name": source.file_name,
                    "display_name": source.display_name,
                    "source_alias": source.alias,
                    "schema_hint": dict(source.schema_hint or {}),
                    "preprocessing_summary": dict(getattr(source, "preprocessing_summary", {}) or {}),
                    "row_count": getattr(source, "row_count", None),
                    "column_count": getattr(source, "column_count", None),
                }
        self._cached_csv_source_metadata = metadata
        return metadata

    def _collect_csv_candidates(self) -> list[TableCandidate]:
        if not self.csv_loaded or not self.csv_session_id:
            return []

        rows = self.csv_runtime.list_tables(self.csv_session_id)
        source_metadata = self._csv_source_metadata_by_table()
        out: list[TableCandidate] = []
        for row in rows:
            table_name = str(row.get("table_name") or "").strip()
            if not table_name:
                continue
            columns_meta = self.csv_runtime.describe_table(self.csv_session_id, table_name)
            columns = [
                str(item.get("column_name") or "").strip()
                for item in columns_meta
                if str(item.get("column_name") or "").strip()
            ]
            source_meta = source_metadata.get(table_name, {})
            out.append(
                TableCandidate(
                    source_kind="csv_session",
                    dialect="duckdb",
                    table_name=table_name,
                    qualified_name=table_name,
                    schema="main",
                    columns=columns,
                    source_label=str(
                        source_meta.get("display_name")
                        or source_meta.get("file_name")
                        or f"CSV session {self.csv_session_id}"
                    ),
                    source_ref_id=self.csv_session_id,
                    csv_session_id=self.csv_session_id,
                    file_name=source_meta.get("file_name"),
                    display_name=source_meta.get("display_name"),
                    source_alias=source_meta.get("source_alias"),
                    schema_hint=dict(source_meta.get("schema_hint") or {}),
                    preprocessing_summary=dict(source_meta.get("preprocessing_summary") or {}),
                    row_count=source_meta.get("row_count"),
                    column_count=source_meta.get("column_count"),
                )
            )
        return out

    def collect_candidates(self) -> list[TableCandidate]:
        if self._cached_candidates is not None:
            return self._cached_candidates

        def _build() -> list[TableCandidate]:
            return self._collect_db_candidates() + self._collect_csv_candidates()

        if self._candidates_cache_key:
            self._cached_candidates = get_or_build_candidates(
                self._candidates_cache_key,
                _build,
            )
        else:
            self._cached_candidates = _build()
        return self._cached_candidates

    def _csv_list_tables_catalog_payload(self) -> dict[str, Any]:
        sid = str(self.csv_session_id or "").strip()
        if not sid:
            raise ValueError("CSV session is not configured.")
        rows = self.csv_runtime.list_tables(sid)
        df = pd.DataFrame(
            rows,
            columns=["schema", "table_name", "table_type", "qualified_name"],
        )
        df = _normalize_dataframe(df)
        return {
            "schema_version": "1.0",
            "artifact_type": "table",
            "items": {"csv_tables": df},
            "source": {
                "source_type": "csv_session",
                "source_ref_id": sid,
                "source_label": f"CSV session {sid}",
                "source_mode": "read_only",
            },
            "recipe": [
                build_db_metadata_recipe_step(
                    action="list_tables",
                    title="List Tables",
                    tool_name="sql_tool",
                    summary="CSV session table catalog",
                )
            ],
            "meta": {"catalog_listing": True},
        }

    def _build_catalog_table_list_artifact(self) -> dict[str, Any]:
        has_db = self.db_runtime_config is not None
        has_csv = bool(self.csv_loaded and self.csv_session_id)

        if has_db and not has_csv:
            payload = self._db_helper().list_tables_result()
            meta = dict(payload.get("meta") or {})
            meta["catalog_listing"] = True
            payload["meta"] = meta
            return payload

        if has_csv and not has_db:
            return self._csv_list_tables_catalog_payload()

        if has_db and has_csv:
            helper = self._db_helper()
            db_payload = helper.list_tables_result(artifact_name="catalog_tables_db")
            db_df = next(iter(db_payload["items"].values())).copy()
            db_df.insert(0, "source", "database")

            sid = str(self.csv_session_id or "").strip()
            rows = self.csv_runtime.list_tables(sid)
            csv_df = pd.DataFrame(
                rows,
                columns=["schema", "table_name", "table_type", "qualified_name"],
            )
            csv_df = _normalize_dataframe(csv_df)
            csv_df.insert(0, "source", "csv_session")

            combined = pd.concat([db_df, csv_df], ignore_index=True)
            recipe = list(db_payload.get("recipe") or [])
            recipe.append(
                build_db_metadata_recipe_step(
                    action="list_tables",
                    title="List Tables (CSV)",
                    tool_name="sql_tool",
                    summary="CSV session table catalog",
                )
            )
            meta_db = dict(db_payload.get("meta") or {})
            meta_db["catalog_listing"] = True
            meta_db["includes_csv_session"] = True
            return {
                "schema_version": "1.0",
                "artifact_type": "table",
                "items": {"catalog_tables": _normalize_dataframe(combined)},
                "source": db_payload["source"],
                "recipe": recipe,
                "meta": meta_db,
            }

        raise ValueError("Нет доступных таблиц ни из DB runtime, ни из CSV session.")

    @staticmethod
    def _candidate_descriptor(candidate: TableCandidate) -> dict[str, Any]:
        preprocessing_summary = {
            key: value
            for key, value in dict(candidate.preprocessing_summary or {}).items()
            if key != "planfact_config"
        }
        return {
            "source_kind": candidate.source_kind,
            "dialect": candidate.dialect,
            "table_name": candidate.table_name,
            "qualified_name": candidate.qualified_name,
            "schema": candidate.schema,
            "columns": list(candidate.columns),
            "source_label": candidate.source_label,
            "source_ref_id": candidate.source_ref_id,
            "file_name": candidate.file_name,
            "display_name": candidate.display_name,
            "source_alias": candidate.source_alias,
            "schema_hint": dict(candidate.schema_hint or {}),
            "preprocessing_summary": preprocessing_summary,
            "row_count": candidate.row_count,
            "column_count": candidate.column_count,
        }

    def _referenced_candidates_for_sql(self, sql: str) -> list[TableCandidate]:
        normalized = re.sub(r"\s+", " ", str(sql or "")).lower()
        candidates = self.collect_candidates()
        qualified_matches = [
            candidate
            for candidate in candidates
            if self._contains_identifier(
                normalized,
                str(candidate.qualified_name or "").lower(),
            )
        ]
        if qualified_matches:
            return qualified_matches

        return [
            candidate
            for candidate in candidates
            if self._contains_identifier(
                normalized,
                str(candidate.table_name or "").lower(),
            )
        ]

    def _csv_describe_tables_artifact(
        self,
        table_names: list[str],
        *,
        artifact_name: str | None = None,
    ) -> dict[str, Any]:
        sid = str(self.csv_session_id or "").strip()
        if not sid:
            raise ValueError("CSV session is not configured.")
        rows: list[dict[str, Any]] = []
        for table_name in table_names:
            rows.extend(self.csv_runtime.describe_table(sid, table_name))
        df = pd.DataFrame(
            rows,
            columns=[
                "schema",
                "table_name",
                "column_name",
                "data_type",
                "is_nullable",
                "ordinal_position",
                "default_expression",
            ],
        )
        name = self._sanitize_artifact_name(artifact_name) or "table_schema"
        return {
            "schema_version": "1.0",
            "artifact_type": "table",
            "items": {name: _normalize_dataframe(df)},
            "source": {
                "source_type": "csv_session",
                "source_ref_id": sid,
                "source_label": f"CSV session {sid}",
                "source_mode": "read_only",
            },
            "recipe": [
                build_db_metadata_recipe_step(
                    action="describe_table",
                    title="Describe Tables",
                    tool_name="sql_tool",
                    summary=f"CSV session schemas: {', '.join(table_names)}",
                )
            ],
            "meta": {
                "schema_description": True,
                "described_tables": list(table_names),
            },
        }

    def build_describe_tables_artifact(
        self,
        table_names: list[str],
        *,
        artifact_name: str | None = None,
    ) -> dict[str, Any]:
        clean_names = [str(item).strip() for item in table_names if str(item).strip()]
        if not clean_names:
            raise ValueError("No table names provided for describe_table.")

        if self.csv_loaded and self.csv_session_id:
            return self._csv_describe_tables_artifact(clean_names, artifact_name=artifact_name)

        if self.db_runtime_config is None:
            raise ValueError("No DB or CSV runtime is configured for describe_table.")

        frames: list[pd.DataFrame] = []
        recipe: list[dict[str, Any]] = []
        for table_name in clean_names:
            table, schema = self._split_schema_qualified_name(table_name)
            payload = self._db_helper().describe_table_result(table, schema=schema)
            frame = next(iter(payload.get("items", {}).values()))
            if isinstance(frame, pd.DataFrame):
                frames.append(frame)
            recipe.extend(list(payload.get("recipe") or []))
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return {
            "schema_version": "1.0",
            "artifact_type": "table",
            "items": {
                self._sanitize_artifact_name(artifact_name) or "table_schema": _normalize_dataframe(df),
            },
            "source": {"source_type": "db_connection"},
            "recipe": recipe,
            "meta": {
                "schema_description": True,
                "described_tables": clean_names,
            },
        }

    @staticmethod
    def _normalized_question(question: str) -> str:
        return re.sub(r"\s+", " ", str(question or "").lower()).strip()

    @staticmethod
    def _split_schema_qualified_name(name: str) -> tuple[str, str | None]:
        text = str(name or "").strip().strip('"').strip("`")
        if "." not in text:
            return text, None
        schema, table = text.split(".", 1)
        clean_schema = schema.strip().strip('"').strip("`")
        clean_table = table.strip().strip('"').strip("`")
        return clean_table, clean_schema or None

    @staticmethod
    def _contains_identifier(text: str, identifier: str) -> bool:
        clean_identifier = str(identifier or "").strip().lower()
        if not clean_identifier:
            return False
        return bool(
            re.search(
                rf"(?<![\w.]){re.escape(clean_identifier)}(?![\w.])",
                str(text or "").lower(),
            )
        )

    def _configured_db_schema(self) -> str:
        if self.db_runtime_config is None:
            return ""
        schema = self.db_runtime_config.options.get("schema")
        if isinstance(schema, str):
            return schema.strip()
        return ""

    def _execute_semantic_query(
        self,
        query: SemanticQuery,
        *,
        artifact_name: str | None,
        purpose: str,
    ) -> dict[str, Any]:
        hints = self.semantic_hints if isinstance(self.semantic_hints, dict) else {}
        catalog_payload = hints.get("catalog")
        if not isinstance(catalog_payload, dict):
            raise ValueError("Semantic catalog is unavailable for the requested metric")
        catalog = SemanticCatalog.model_validate(catalog_payload)
        metrics_by_key = {metric.key: metric for metric in catalog.metrics if metric.is_active}
        selected_metrics = [metrics_by_key[key] for key in query.metrics if key in metrics_by_key]
        dialect = (
            "duckdb"
            if self.csv_loaded
            else str(getattr(self.db_runtime_config, "db_type", "postgres") or "postgres")
        )
        compiler = SemanticQueryCompiler(catalog, dialect=dialect)
        default_time_dimension = compiler.shared_default_time_dimension(selected_metrics)
        effective_query = query
        if query.time_grain and not query.time_dimension and default_time_dimension:
            effective_query = query.model_copy(update={"time_dimension": default_time_dimension})
        semantic_sql = compiler.compile(effective_query)
        payload = self.execute_sql_artifact(
            semantic_sql,
            artifact_name=artifact_name,
            purpose=purpose,
        )
        meta = dict(payload.get("meta") or {})
        meta["semantic_metric"] = {
            "catalog_id": catalog.catalog_id,
            "metric_keys": list(effective_query.metrics),
            "metrics": [
                {
                    "key": metric.key,
                    "name": metric.name,
                    "formula": metric.formula,
                    "format": metric.format,
                    "base_table": metric.base_table,
                    "default_time_dimension": metric.default_time_dimension,
                    "allowed_dimensions": list(metric.allowed_dimensions),
                    "filters": [item.model_dump(mode="json") for item in metric.filters],
                }
                for key in effective_query.metrics
                if (metric := metrics_by_key.get(key)) is not None
            ],
            "query": effective_query.model_dump(mode="json"),
            "compiled_sql": semantic_sql,
        }
        payload["meta"] = meta
        return payload

    def _csv_source_ref(self, candidate: TableCandidate) -> dict[str, Any]:
        return {
            "source_type": "csv_session",
            "source_ref_id": str(candidate.csv_session_id or ""),
            "source_label": candidate.source_label,
            "source_mode": "read_only",
        }

    def _package_csv_query_result(
        self,
        *,
        candidate: TableCandidate,
        question: str,
        sql: str,
        artifact_name: str,
    ) -> dict[str, Any]:
        executed_sql, validation = _normalize_analytic_sql(sql, max_rows=self.max_rows)
        started_at = time.perf_counter()
        rows = self.csv_runtime.query_dataframe(str(candidate.csv_session_id), executed_sql)
        execution_time_ms = int((time.perf_counter() - started_at) * 1000)

        warnings = list(validation["warnings"])
        truncated = False

        rows, capped = _cap_dataframe_rows(rows, max_rows=self.max_rows)
        truncated = truncated or capped
        if capped:
            warnings.append(f"Result exceeded max_rows={self.max_rows} and was truncated.")

        rows, cell_capped = _shrink_for_cell_budget(rows)
        truncated = truncated or cell_capped
        if cell_capped:
            warnings.append(f"Result exceeded cell budget={MAX_RESULT_CELLS} and was truncated.")

        safe_rows = _normalize_dataframe(rows)
        referenced_candidates = self._referenced_candidates_for_sql(sql)
        if not referenced_candidates:
            referenced_candidates = [candidate]
        referenced_tables = [self._candidate_descriptor(item) for item in referenced_candidates]
        query_meta = {
            "purpose": question,
            "requested_sql": validation["requested_sql"],
            "executed_sql": executed_sql,
            "max_rows": self.max_rows,
            "requested_limit": validation["requested_limit"],
            "returned_rows": len(safe_rows),
            "column_count": len(safe_rows.columns),
            "truncated": bool(truncated),
            "execution_time_ms": int(execution_time_ms),
            "warnings": list(warnings),
        }
        meta = {
            "query": query_meta,
            "execution_stats": {
                "row_count": query_meta["returned_rows"],
                "column_count": query_meta["column_count"],
                "truncated": query_meta["truncated"],
                "execution_time_ms": query_meta["execution_time_ms"],
            },
            "table_selection": {
                **self._candidate_descriptor(candidate),
                "additional_tables": [
                    self._candidate_descriptor(item)
                    for item in referenced_candidates
                    if item.qualified_name != candidate.qualified_name
                ],
            },
            "lineage": {
                "source_tables": referenced_tables,
                "source_table_names": [item["qualified_name"] for item in referenced_tables],
            },
            "warnings": list(warnings),
        }
        return {
            "schema_version": "1.0",
            "artifact_type": "table",
            "items": {artifact_name: safe_rows},
            "source": self._csv_source_ref(candidate),
            "recipe": [
                build_sql_recipe_step(
                    sql=executed_sql,
                    title="Executed SQL",
                    tool_name="sql_tool",
                    summary=f"Analytical read query; max_rows={self.max_rows}",
                )
            ],
            "meta": meta,
        }

    def _package_csv_raw_sql_result(
        self,
        *,
        sql: str,
        purpose: str,
        artifact_name: str,
    ) -> dict[str, Any]:
        sid = str(self.csv_session_id or "").strip()
        if not sid:
            raise ValueError("CSV session is not configured.")

        executed_sql, validation = _normalize_analytic_sql(sql, max_rows=self.max_rows)
        started_at = time.perf_counter()
        rows = self.csv_runtime.query_dataframe(sid, executed_sql)
        execution_time_ms = int((time.perf_counter() - started_at) * 1000)

        warnings = list(validation["warnings"])
        truncated = False
        rows, capped = _cap_dataframe_rows(rows, max_rows=self.max_rows)
        truncated = truncated or capped
        if capped:
            warnings.append(f"Result exceeded max_rows={self.max_rows} and was truncated.")
        rows, cell_capped = _shrink_for_cell_budget(rows)
        truncated = truncated or cell_capped
        if cell_capped:
            warnings.append(f"Result exceeded cell budget={MAX_RESULT_CELLS} and was truncated.")

        safe_rows = _normalize_dataframe(rows)
        referenced_candidates = self._referenced_candidates_for_sql(sql)
        referenced_tables = [self._candidate_descriptor(candidate) for candidate in referenced_candidates]
        query_meta = {
            "purpose": purpose,
            "requested_sql": validation["requested_sql"],
            "executed_sql": executed_sql,
            "max_rows": self.max_rows,
            "requested_limit": validation["requested_limit"],
            "returned_rows": len(safe_rows),
            "column_count": len(safe_rows.columns),
            "truncated": bool(truncated),
            "execution_time_ms": int(execution_time_ms),
            "warnings": list(warnings),
        }
        return {
            "schema_version": "1.0",
            "artifact_type": "table",
            "items": {artifact_name: safe_rows},
            "source": {
                "source_type": "csv_session",
                "source_ref_id": sid,
                "source_label": f"CSV session {sid}",
                "source_mode": "read_only",
            },
            "recipe": [
                build_sql_recipe_step(
                    sql=executed_sql,
                    title="Executed SQL",
                    tool_name="sql_tool",
                    summary=f"Direct read query; max_rows={self.max_rows}",
                )
            ],
            "meta": {
                "query": query_meta,
                "execution_stats": {
                    "row_count": query_meta["returned_rows"],
                    "column_count": query_meta["column_count"],
                    "truncated": query_meta["truncated"],
                    "execution_time_ms": query_meta["execution_time_ms"],
                },
                "lineage": {
                    "source_tables": referenced_tables,
                    "source_table_names": [item["qualified_name"] for item in referenced_tables],
                },
                "warnings": list(warnings),
                "direct_sql": True,
            },
        }

    def execute_sql_artifact(
        self,
        sql: str,
        *,
        artifact_name: str | None = None,
        purpose: str | None = None,
    ) -> dict[str, Any]:
        clean = clean_sql(sql)
        if not is_select_or_with(clean):
            raise ValueError("Only SELECT/WITH SQL is allowed.")
        _assert_read_only_sql(clean)
        final_artifact_name = self._sanitize_artifact_name(artifact_name) or "sql_result"

        if self.csv_loaded and self.csv_session_id:
            return self._package_csv_raw_sql_result(
                sql=clean,
                purpose=purpose or clean,
                artifact_name=final_artifact_name,
            )

        if self.db_runtime_config is not None:
            payload = self._db_helper().execute_analytic_query(
                clean,
                purpose=purpose or clean,
                max_rows=self.max_rows,
                artifact_name=final_artifact_name,
            )
            referenced_candidates = self._referenced_candidates_for_sql(clean)
            referenced_tables = [self._candidate_descriptor(candidate) for candidate in referenced_candidates]
            meta = dict(payload.get("meta") or {})
            meta["direct_sql"] = True
            if referenced_tables:
                meta["lineage"] = {
                    "source_tables": referenced_tables,
                    "source_table_names": [item["qualified_name"] for item in referenced_tables],
                }
            payload["meta"] = meta
            return payload

        raise ValueError("No DB or CSV runtime is configured for execute_sql.")

    def execute_final_query(
        self,
        *,
        question: str,
        candidate: TableCandidate,
        sql: str,
        artifact_name: str | None = None,
    ) -> dict[str, Any]:
        clean_artifact_name = self._sanitize_artifact_name(artifact_name)
        fallback_artifact_name = self._sanitize_artifact_name(f"sql_{candidate.table_name}") or "sql_result"
        final_artifact_name = clean_artifact_name or fallback_artifact_name

        if candidate.source_kind == "db":
            payload = self._db_helper().execute_analytic_query(
                sql,
                purpose=question,
                max_rows=self.max_rows,
                artifact_name=final_artifact_name,
            )
            meta = dict(payload.get("meta") or {})
            meta["table_selection"] = self._candidate_descriptor(candidate)
            referenced_candidates = self._referenced_candidates_for_sql(sql)
            if not referenced_candidates:
                referenced_candidates = [candidate]
            referenced_tables = [self._candidate_descriptor(item) for item in referenced_candidates]
            meta["lineage"] = {
                "source_tables": referenced_tables,
                "source_table_names": [item["qualified_name"] for item in referenced_tables],
            }
            payload["meta"] = meta
            return payload

        return self._package_csv_query_result(
            candidate=candidate,
            question=question,
            sql=sql,
            artifact_name=final_artifact_name,
        )

    def build_table_artifact(
        self,
        question: str,
        artifact_name: str | None = None,
        *,
        mode: str | None = None,
        sql: str | None = None,
        table_names: list[str] | None = None,
        semantic_query: SemanticQuery | None = None,
    ) -> dict[str, Any]:
        effective_mode = str(mode or "").strip() or None
        if effective_mode is None:
            if semantic_query is not None:
                effective_mode = "semantic_query"
            elif sql or is_select_or_with(str(question or "")):
                effective_mode = "execute_sql"
            else:
                raise ValueError("SQL tool mode is required")

        if effective_mode == "catalog_tables":
            return self._build_catalog_table_list_artifact()

        if effective_mode == "describe_table":
            return self.build_describe_tables_artifact(
                list(table_names or []),
                artifact_name=artifact_name,
            )

        if effective_mode == "execute_sql":
            return self.execute_sql_artifact(
                sql or question,
                artifact_name=artifact_name,
                purpose=question or sql,
            )

        if effective_mode == "semantic_query":
            if semantic_query is None:
                raise ValueError("semantic_query mode requires semantic_query")
            return self._execute_semantic_query(
                semantic_query,
                artifact_name=artifact_name,
                purpose=question or f"Semantic query: {', '.join(semantic_query.metrics)}",
            )

        raise ValueError(f"Unsupported SQL tool mode: {effective_mode}")
