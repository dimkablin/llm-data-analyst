from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from backend.agent.callbacks import strip_thinking
from backend.agent.dataset_profiles import build_sql_generation_hints
from backend.agent.llm_client import make_reasoning_llm
from backend.artifacts.artifact_meta import build_db_metadata_recipe_step, build_sql_recipe_step
from backend.data_access.csv_session_runtime import CSVSessionRuntime
from backend.data_access.db_runtime_service import RuntimeDBConnectionConfig
from backend.data_access.semantic_models import SemanticCatalog
from backend.data_access.semantic_query import SemanticQueryCompiler, semantic_query_from_hints
from backend.data_access.session_catalog_cache import get_or_build_candidates
from backend.notebook.manifest_store import ManifestStore
from backend.tools.impl.db_helpers import (
    IDENTIFIER_RE,
    MAX_RESULT_CELLS,
    DBAnalyticsHelper,
    _assert_read_only_sql,
    _normalize_analytic_sql,
    _normalize_dataframe,
)

_SQL_START = re.compile(r"\b(SELECT|WITH)\b", re.IGNORECASE)
# Questions that only ask for table names from the catalog (no analytic SQL / LLM).
_CATALOG_TABLE_LIST_RE = re.compile(
    r"(?is)^\s*("
    r"показать\s+все\s+таблицы(\s+(в|из)\s+(базе?(\s+данных)?|бд|базы(\s+данных)?|db))?|"
    r"покажи\s+все\s+таблицы(\s+(в|из)\s+(базе?(\s+данных)?|бд|базы(\s+данных)?|db))?|"
    r"покажи\s+таблицы(\s+(в|из)\s+(базе?(\s+данных)?|бд|базы(\s+данных)?|db))?\s*|"
    r"показать\s+таблицы(\s+(в|из)\s+(базе?(\s+данных)?|бд|базы(\s+данных)?|db))?\s*|"
    r"покажи\s+список\s+таблиц(\s+(в|из)\s+(базе?(\s+данных)?|бд|базы(\s+данных)?|db))?|"
    r"показать\s+список\s+таблиц(\s+(в|из)\s+(базе?(\s+данных)?|бд|базы(\s+данных)?|db))?|"
    r"получи(ть)?\s+список\s+(всех\s+)?таблиц(\s+(в|из)\s+(базе?(\s+данных)?|бд|базы(\s+данных)?|db))?|"
    r"список\s+таблиц(\s+(в|из)\s+(базе?(\s+данных)?|бд|базы(\s+данных)?|db))?|"
    r"перечень\s+таблиц|"
    r"перечисли\s+таблицы|"
    r"какие\s+таблицы\s+(есть|в\s+(базе|бд|базе\s+данных))|"
    r"какие\s+есть\s+таблицы|"
    r"выведи\s+список\s+таблиц|"
    r"назови\s+таблицы|"
    r"(show|list)\s+tables(\s+(in|from)\s+[\w\s]+)?|"
    r"(what|which)\s+tables\s+(are\s+there|exist|do\s+i\s+have|in\s+the\s+database)"
    r"|таблицы\.?\s*$|"
    r"tables\.?\s*$"
    r")\s*\??\s*$"
)
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


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content or "")


def _extract_json_obj(text: str) -> str | None:
    t = str(text or "").strip()
    match = re.search(r"\{[\s\S]*\}", t)
    return match.group(0) if match else None


def _safe_json_loads(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except Exception:
        raw = _extract_json_obj(text)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None


def clean_sql(raw: str) -> str:
    text = str(raw or "").strip()
    text = re.sub(r"^```sql\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^sql\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    text = text.strip("`").strip()

    match = _SQL_START.search(text)
    if not match:
        return text.split("```")[0].strip().strip(";").strip()

    sql = text[match.start():]
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


def wrap_sample(sql: str, n: int = 5) -> str:
    return f"SELECT * FROM ({sql}) AS q LIMIT {int(n)}"


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
    # Thinking default for SQL generation LLM calls.
    # Effective thinking = settings.llm_enable_thinking AND TOOL_ENABLE_THINKING.
    TOOL_ENABLE_THINKING: ClassVar[bool] = False

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
        candidates_cache_key: str | None = None,
        storage_dir: str | Path | None = None,
        semantic_context_prompt: str = "",
        semantic_hints: dict[str, Any] | None = None,
    ) -> None:
        self.db_runtime_config = db_runtime_config
        self.csv_loaded = bool(csv_loaded)
        self.csv_session_id = str(csv_session_id or "").strip() or None
        self.max_rows = max(1, min(int(max_rows), 1000))
        self.csv_runtime = CSVSessionRuntime()
        self._cached_db_helper: DBAnalyticsHelper | None = None
        self._cached_candidates: list[TableCandidate] | None = None
        self._candidates_cache_key = str(candidates_cache_key or "").strip() or None
        self.storage_dir = Path(storage_dir).resolve() if storage_dir is not None else None
        self._cached_csv_source_metadata: dict[str, dict[str, Any]] | None = None
        self.semantic_context_prompt = str(semantic_context_prompt or "").strip()
        self.semantic_hints = dict(semantic_hints or {})

        self.llm = make_reasoning_llm(
            provider=llm_provider,
            model=llm_model,
            base_url=llm_base_url,
            api_key=llm_api_key,
            enable_thinking=llm_enable_thinking and SQLTableService.TOOL_ENABLE_THINKING,
            temperature=0.0,
            max_tokens=2048,
            streaming=False,
            timeout=120.0,
            chat_template_kwargs_enabled=llm_chat_template_kwargs_enabled,
        )

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
            self._cached_db_helper = DBAnalyticsHelper(runtime=self.db_runtime_config, timeout_sec=15.0)
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
            manifest = ManifestStore(self.storage_dir).load(self.csv_session_id)
        except Exception:
            self._cached_csv_source_metadata = metadata
            return metadata

        for source in manifest.sources:
            if source.source_type != "csv":
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

    @staticmethod
    def _wants_catalog_table_list(question: str) -> bool:
        q = re.sub(r"\s+", " ", str(question or "").strip())
        return bool(_CATALOG_TABLE_LIST_RE.match(q))

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
            "preprocessing_summary": dict(candidate.preprocessing_summary or {}),
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

    def _join_related_candidates(
        self,
        question: str,
        primary: TableCandidate,
        candidates: list[TableCandidate],
    ) -> list[TableCandidate]:
        q = self._normalized_question(question)
        join_intent = any(
            token in q
            for token in (
                "join",
                "merge",
                "using",
                "объедин",
                "соедин",
                "связ",
                "сопостав",
                "соотнес",
                "джойн",
                "джоин",
                "по столбц",
                "по колон",
            )
        )
        primary_columns = {
            str(column).strip().lower() for column in primary.columns if str(column).strip()
        }
        scored: list[tuple[int, TableCandidate]] = []
        for candidate in candidates:
            if (
                candidate.source_kind == primary.source_kind
                and candidate.qualified_name == primary.qualified_name
            ):
                continue

            score = 0
            table_names = (candidate.table_name.lower(), candidate.qualified_name.lower())
            if any(name and name in q for name in table_names):
                score += 30

            candidate_columns = {
                str(column).strip().lower()
                for column in candidate.columns
                if str(column).strip()
            }
            shared_columns = primary_columns & candidate_columns
            if shared_columns:
                score += 20 + min(len(shared_columns), 5)
            primary_column_mentions = sum(
                1
                for column in primary_columns
                if len(column) >= 3 and column in q
            )
            for column in candidate_columns:
                if len(column) >= 3 and column in q:
                    score += 10
            if primary_column_mentions and any(
                len(column) >= 3 and column in q for column in candidate_columns
            ):
                score += 12

            if score > 0 and (join_intent or any(name and name in q for name in table_names)):
                scored.append((score, candidate))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [candidate for _, candidate in scored[:4]]

    def _additional_candidates_for_question(
        self,
        question: str,
        primary: TableCandidate,
        candidates: list[TableCandidate],
    ) -> list[TableCandidate]:
        result: list[TableCandidate] = []
        seen: set[tuple[str, str, str]] = set()
        for candidate in self._join_related_candidates(question, primary, candidates):
            key = (candidate.source_kind, candidate.source_ref_id, candidate.qualified_name)
            if key in seen:
                continue
            seen.add(key)
            result.append(candidate)
        for candidate in self._semantic_related_candidates(primary, candidates):
            key = (candidate.source_kind, candidate.source_ref_id, candidate.qualified_name)
            if key in seen:
                continue
            seen.add(key)
            result.append(candidate)
        return result

    def _semantic_table_hint(self, candidates: list[TableCandidate]) -> TableCandidate | None:
        names: list[str] = []
        hints = self.semantic_hints if isinstance(self.semantic_hints, dict) else {}
        for metric in hints.get("metrics") or []:
            if isinstance(metric, dict):
                names.append(str(metric.get("base_table") or ""))
        for table in hints.get("tables") or []:
            if isinstance(table, dict):
                names.append(str(table.get("qualified_name") or ""))
                names.append(str(table.get("table_name") or ""))
        for raw_name in names:
            name = raw_name.strip().lower()
            if not name:
                continue
            for candidate in candidates:
                if name in {
                    str(candidate.qualified_name or "").lower(),
                    str(candidate.table_name or "").lower(),
                }:
                    return candidate
        return None

    def _semantic_related_candidates(
        self,
        primary: TableCandidate,
        candidates: list[TableCandidate],
    ) -> list[TableCandidate]:
        hints = self.semantic_hints if isinstance(self.semantic_hints, dict) else {}
        related_names: set[str] = set()
        primary_names = {
            str(primary.qualified_name or "").lower(),
            str(primary.table_name or "").lower(),
        }
        for rel in hints.get("relationships") or []:
            if not isinstance(rel, dict):
                continue
            from_table = str(rel.get("from_table") or "").lower()
            to_table = str(rel.get("to_table") or "").lower()
            if from_table in primary_names:
                related_names.add(to_table)
            if to_table in primary_names:
                related_names.add(from_table)
        result: list[TableCandidate] = []
        for candidate in candidates:
            candidate_names = {
                str(candidate.qualified_name or "").lower(),
                str(candidate.table_name or "").lower(),
            }
            if related_names & candidate_names:
                result.append(candidate)
        return result

    def _semantic_prompt_block(self) -> str:
        if not self.semantic_context_prompt:
            return ""
        return "\n\nSEMANTIC_HINTS:\n" + self.semantic_context_prompt[:4000]

    def _try_compile_semantic_sql(self, question: str) -> str | None:
        hints = self.semantic_hints if isinstance(self.semantic_hints, dict) else {}
        catalog_payload = hints.get("catalog")
        if not isinstance(catalog_payload, dict):
            return None
        try:
            catalog = SemanticCatalog.model_validate(catalog_payload)
            semantic_query = semantic_query_from_hints(hints, question=question, catalog=catalog)
            if semantic_query is None:
                return None
            dialect = "duckdb" if self.csv_loaded else str(getattr(self.db_runtime_config, "db_type", "postgres") or "postgres")
            return SemanticQueryCompiler(catalog, dialect=dialect).compile(semantic_query)
        except Exception:
            return None

    def _find_explicit_table(self, question: str, candidates: list[TableCandidate]) -> TableCandidate | None:
        normalized_question = self._normalized_question(question)
        for candidate in candidates:
            qualified = str(candidate.qualified_name or "").lower()
            if qualified and self._contains_identifier(normalized_question, qualified):
                return candidate
        for candidate in candidates:
            table_name = str(candidate.table_name or "").lower()
            if table_name and self._contains_identifier(normalized_question, table_name):
                return candidate
        return None

    @staticmethod
    def _score_table_candidate(question: str, candidate: TableCandidate) -> int:
        """Rank tables so analytic questions prefer relevant schemas over import/meta noise."""
        q = re.sub(r"\s+", " ", str(question or "").lower()).strip()
        score = 0
        schema = str(candidate.schema or "").lower()
        if schema.endswith("_meta") or schema in {"information_schema", "pg_catalog"}:
            score -= 80

        for col in candidate.columns:
            col_l = str(col).lower()
            if len(col_l) >= 4 and col_l in q:
                score += 10

        if candidate.table_name.lower() in q:
            score += 25
        if candidate.qualified_name.lower() in q:
            score += 30

        return score

    def _rank_candidates(self, question: str, candidates: list[TableCandidate]) -> list[TableCandidate]:
        return sorted(
            candidates,
            key=lambda c: self._score_table_candidate(question, c),
            reverse=True,
        )

    def _choose_table_via_llm(self, question: str, candidates: list[TableCandidate]) -> TableCandidate:
        ranked = self._rank_candidates(question, candidates)
        preview_rows = []
        for idx, candidate in enumerate(ranked[:40], start=1):
            preview_rows.append(
                {
                    "idx": idx,
                    "source_kind": candidate.source_kind,
                    "dialect": candidate.dialect,
                    "table_name": candidate.table_name,
                    "qualified_name": candidate.qualified_name,
                    "columns": candidate.columns[:15],
                    "source_label": candidate.source_label,
                }
            )

        prompt = f"""
Выбери ОДНУ таблицу, которая лучше всего подходит для вопроса пользователя.
Верни только JSON вида {{"idx": number, "reason": "short"}}.

QUESTION:
{question}

CANDIDATES:
{json.dumps(preview_rows, ensure_ascii=False)}
""".strip()

        resp = self.llm.invoke(
            [
                SystemMessage(content="Верни только валидный JSON."),
                HumanMessage(content=prompt),
            ]
        )
        obj = _safe_json_loads(_message_text(resp.content))
        if obj and isinstance(obj.get("idx"), int):
            idx = int(obj["idx"])
            if 1 <= idx <= len(preview_rows):
                return ranked[idx - 1]

        return ranked[0]

    def resolve_table(self, question: str) -> TableCandidate:
        candidates = self.collect_candidates()
        if not candidates:
            raise ValueError("Нет доступных таблиц ни из DB runtime, ни из CSV session.")

        # Single candidate — no disambiguation needed.
        if len(candidates) == 1:
            return candidates[0]

        explicit = self._find_explicit_table(question, candidates)
        if explicit is not None:
            return explicit

        semantic = self._semantic_table_hint(candidates)
        if semantic is not None:
            return semantic

        return self._choose_table_via_llm(question, candidates)

    def _run_query_no_throw(
        self,
        candidate: TableCandidate,
        sql: str,
    ) -> tuple[list[dict[str, Any]] | None, str | None]:
        try:
            if candidate.source_kind == "db":
                rows = self._db_helper().query_dataframe(sql)
            else:
                rows = self.csv_runtime.query_dataframe(str(candidate.csv_session_id), sql)
            return rows.to_dict(orient="records"), None
        except Exception as exc:
            return None, str(exc)

    def _safe_sample_sql(self, candidate: TableCandidate) -> str:
        """Build a safe LIMIT-5 sample query with properly quoted identifiers.

        Never interpolate ``qualified_name`` directly — it originates from the
        database catalog and could contain unexpected characters.
        """
        if candidate.source_kind == "db":
            helper = self._db_helper()
            schema = candidate.schema
            table = candidate.table_name
            quoted_table = helper._quote_identifier(table)  # noqa: SLF001
            if schema:
                quoted_schema = helper._quote_identifier(schema)  # noqa: SLF001
                qualified = f"{quoted_schema}.{quoted_table}"
            else:
                qualified = quoted_table
        else:
            # DuckDB / CSV session — table names come from our own catalog.
            table = candidate.table_name
            if not IDENTIFIER_RE.match(table):
                raise ValueError(f"Unsafe CSV table identifier: {table!r}")
            qualified = f'"{table}"'
        return f"SELECT * FROM {qualified} LIMIT 5"

    def _table_sample(self, candidate: TableCandidate) -> dict[str, Any]:
        try:
            sample_sql = self._safe_sample_sql(candidate)
            _assert_read_only_sql(sample_sql)
            if candidate.source_kind == "db":
                rows = self._db_helper().query_dataframe(sample_sql)
            else:
                rows = self.csv_runtime.query_dataframe(str(candidate.csv_session_id), sample_sql)
            return {"first_rows": rows.to_dict(orient="records")}
        except Exception:
            return {"first_rows": []}

    @staticmethod
    def _quoted_columns_str(columns: list[str], dialect: str) -> str:
        """Return column names in the safest syntax for the SQL dialect."""
        needs_sql_quotes = dialect.lower() in ("postgresql", "postgres", "duckdb")
        parts: list[str] = []
        for col in columns:
            name = str(col)
            if needs_sql_quotes:
                escaped = name.replace('"', '""')
                parts.append(f'"{escaped}"')
            elif IDENTIFIER_RE.fullmatch(name):
                parts.append(name)
            else:
                escaped = name.replace('"', '""')
                parts.append(f'"{escaped}"')
        return ", ".join(parts)

    def _call_llm_sql_only(
        self,
        *,
        question: str,
        candidate: TableCandidate,
        sample: dict[str, Any] | None = None,
        previous_sql: str | None = None,
        feedback: str | None = None,
        additional_candidates: list[TableCandidate] | None = None,
    ) -> str:
        table_name = candidate.qualified_name
        columns_str = self._quoted_columns_str(candidate.columns, candidate.dialect)
        schema_hints_block = build_sql_generation_hints(
            candidate.columns,
            db_schema=self._configured_db_schema(),
        )
        semantic_block = self._semantic_prompt_block()

        extra_tables_block = ""
        if additional_candidates:
            parts = []
            for ac in additional_candidates:
                ac_cols = self._quoted_columns_str(ac.columns, ac.dialect)
                source_hint = (
                    f" (source: {ac.display_name or ac.file_name or ac.source_label})"
                    if (ac.display_name or ac.file_name or ac.source_label)
                    else ""
                )
                parts.append(f"- {ac.qualified_name}{source_hint}: {ac_cols}")
            extra_tables_intro = "\nAdditional table context for JOIN when multiple sources are needed:\n"
            extra_tables_block = extra_tables_intro + "\n".join(parts)
        primary_source_hint = (
            f" (source: {candidate.display_name or candidate.file_name or candidate.source_label})"
            if (candidate.display_name or candidate.file_name or candidate.source_label)
            else ""
        )

        if previous_sql and feedback:
            user_prompt = f"""
Оригинальный вопрос:
{question}

Предыдущий SQL:
{previous_sql}

Проблема:
{feedback}

Сгенерируй исправленный SQL для {candidate.dialect}.

Правила:
- верни только SQL
- только SELECT/WITH
 - стартовая таблица: {table_name}{primary_source_hint}
 - колонки основной таблицы: {columns_str}{extra_tables_block}{schema_hints_block}
 - If the question needs comparison or JOIN across tables, use all relevant context tables.
 - старайся вернуть компактный результат
""".strip()
        else:
            user_prompt = f"""
Сгенерируй один SQL-запрос для {candidate.dialect}.

Правила:
- верни только SQL
- только SELECT/WITH
 - стартовая таблица: {table_name}{primary_source_hint}
 - колонки основной таблицы: {columns_str}{extra_tables_block}{schema_hints_block}
 - If the question needs comparison or JOIN across tables, use all relevant context tables.
 - старайся вернуть компактный результат

Вопрос:
{question}
""".strip()

        if semantic_block:
            user_prompt += semantic_block

        if sample and sample.get("first_rows"):
            user_prompt += "\n\nSAMPLE_ROWS:\n" + json.dumps(sample["first_rows"], ensure_ascii=False)

        resp = self.llm.invoke(
            [
                SystemMessage(content="Ты SQL-генератор. Верни только SQL SELECT/WITH."),
                HumanMessage(content=user_prompt),
            ]
        )
        return clean_sql(strip_thinking(_message_text(resp.content)))

    @staticmethod
    def _is_trivial_sql(sql: str) -> bool:
        """Detect trivial queries that don't need LLM judge validation."""
        s = re.sub(r"\s+", " ", sql.strip().upper())
        # Only one SELECT (no subqueries)
        if s.count("SELECT") != 1:
            return False
        # Simple SELECT ... FROM ... with optional WHERE/ORDER BY/LIMIT/GROUP BY/HAVING (no JOINs)
        _trivial_pattern = r"^SELECT\s+.+?\s+FROM\s+\S+(\s+(WHERE|LIMIT|ORDER\s+BY|OFFSET|GROUP\s+BY|HAVING)\s+.*)?\s*;?\s*$"  # noqa: E501
        if re.match(_trivial_pattern, s):
            if "JOIN" not in s:
                return True
        # Simple aggregate: SELECT COUNT/SUM/AVG/MIN/MAX(...) [optional GROUP BY]
        if re.match(r"^SELECT\s+(COUNT|SUM|AVG|MIN|MAX)\s*\(", s):
            return True
        # Multi-aggregate: SELECT COUNT(...), SUM(...) etc. from single table
        if re.match(r"^SELECT\s+((COUNT|SUM|AVG|MIN|MAX)\s*\([^)]*\)\s*,?\s*)+\s*FROM\s+\S+", s):
            return True
        return False

    def _judge_sql(
        self,
        *,
        question: str,
        candidate: TableCandidate,
        sql: str,
        sample_result: list[dict[str, Any]],
    ) -> tuple[bool, str, str]:
        columns_str = self._quoted_columns_str(candidate.columns, candidate.dialect)
        prompt = f"""
Ты строгий ревьюер SQL ({candidate.dialect}).
Проверь, решает ли SQL вопрос пользователя.
Верни только JSON формата {{"ok": true/false, "reason": "...", "fix_hint": "..."}}.

TABLE: {candidate.qualified_name}
COLUMNS: {columns_str}

QUESTION:
{question}

SQL:
{sql}

SAMPLE_RESULT:
{json.dumps(sample_result, ensure_ascii=False)}
""".strip()

        resp = self.llm.invoke(
            [
                SystemMessage(content="Верни только JSON."),
                HumanMessage(content=prompt),
            ]
        )
        obj = _safe_json_loads(_message_text(resp.content))
        if not obj:
            return False, "Judge parsing failed", "Верни JSON строго формата ok/reason/fix_hint"

        return (
            bool(obj.get("ok", False)),
            str(obj.get("reason", "")),
            str(obj.get("fix_hint", "")),
        )

    def generate_sql_with_retries(
        self,
        *,
        question: str,
        candidate: TableCandidate,
        max_attempts: int = 3,
        sample_rows: int = 5,
        additional_candidates: list[TableCandidate] | None = None,
    ) -> dict[str, Any]:
        previous_sql: str | None = None
        feedback: str | None = None

        # Fetch sample rows once (reused across retries and LLM prompts).
        cached_sample = self._table_sample(candidate)

        for attempt in range(1, max_attempts + 1):
            sql = self._call_llm_sql_only(
                question=question,
                candidate=candidate,
                sample=cached_sample,
                previous_sql=previous_sql,
                feedback=feedback,
                additional_candidates=additional_candidates,
            )
            sql = clean_sql(sql)
            previous_sql = sql

            if not is_select_or_with(sql):
                feedback = "Разрешены только SELECT/WITH."
                continue

            try:
                _assert_read_only_sql(sql)
            except Exception as exc:
                feedback = f"SQL validation error: {exc}"
                continue

            # Run sample query directly — it validates both syntax and data.
            sample_res, err = self._run_query_no_throw(candidate, wrap_sample(sql, sample_rows))
            if err:
                feedback = f"DB error: {err}"
                continue

            # Skip judge for trivial queries that compiled and returned data.
            if sample_res and self._is_trivial_sql(sql):
                return {
                    "ok": True,
                    "sql": sql,
                    "attempts": attempt,
                    "judge_reason": "trivial_skip",
                }

            ok, reason, fix_hint = self._judge_sql(
                question=question,
                candidate=candidate,
                sql=sql,
                sample_result=sample_res or [],
            )
            if ok:
                return {
                    "ok": True,
                    "sql": sql,
                    "attempts": attempt,
                    "judge_reason": reason,
                }

            feedback = f"Judge reject: {reason}" + (f" | hint: {fix_hint}" if fix_hint else "")

        return {
            "ok": False,
            "sql": previous_sql,
            "attempts": max_attempts,
            "error": feedback or "SQL generation failed",
        }

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
            referenced_tables = [
                self._candidate_descriptor(candidate)
                for candidate in referenced_candidates
            ]
            meta = dict(payload.get("meta") or {})
            meta["direct_sql"] = True
            if referenced_tables:
                meta["lineage"] = {
                    "source_tables": referenced_tables,
                    "source_table_names": [
                        item["qualified_name"] for item in referenced_tables
                    ],
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
            referenced_tables = [
                self._candidate_descriptor(item)
                for item in referenced_candidates
            ]
            meta["lineage"] = {
                "source_tables": referenced_tables,
                "source_table_names": [
                    item["qualified_name"] for item in referenced_tables
                ],
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
    ) -> dict[str, Any]:
        effective_mode = str(mode or "").strip() or None
        if effective_mode is None:
            if sql or is_select_or_with(str(question or "")):
                effective_mode = "execute_sql"
            elif self._wants_catalog_table_list(question):
                effective_mode = "catalog_tables"
            else:
                effective_mode = "nl_query"

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

        if effective_mode != "nl_query":
            raise ValueError(f"Unsupported SQL tool mode: {effective_mode}")

        semantic_sql = self._try_compile_semantic_sql(question)
        if semantic_sql:
            return self.execute_sql_artifact(
                semantic_sql,
                artifact_name=artifact_name,
                purpose=question,
            )

        candidate = self.resolve_table(question)
        additional = self._additional_candidates_for_question(
            question,
            candidate,
            self.collect_candidates(),
        )
        gen = self.generate_sql_with_retries(
            question=question,
            candidate=candidate,
            additional_candidates=additional or None,
        )
        if not gen["ok"]:
            raise ValueError(str(gen.get("error") or "SQL generation failed"))
        return self.execute_final_query(
            question=question,
            candidate=candidate,
            sql=str(gen["sql"]),
            artifact_name=artifact_name,
        )
