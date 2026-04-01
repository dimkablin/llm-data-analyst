from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from backend.tools.impl.db_helpers import (
    DBAnalyticsHelper,
    MAX_RESULT_CELLS,
    _assert_read_only_sql,
    _normalize_analytic_sql,
    _normalize_dataframe,
)
from backend.artifacts.artifact_meta import build_db_metadata_recipe_step, build_sql_recipe_step
from backend.data_access.csv_session_runtime import CSVSessionRuntime
from backend.data_access.db_runtime_service import RuntimeDBConnectionConfig


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
    def __init__(
        self,
        *,
        llm_base_url: str,
        llm_model: str,
        llm_api_key: str | None,
        llm_enable_thinking: bool = False,
        llm_chat_template_kwargs_enabled: bool = True,
        db_runtime_config: RuntimeDBConnectionConfig | None = None,
        csv_loaded: bool = False,
        csv_session_id: str | None = None,
        max_rows: int = 200,
    ) -> None:
        self.db_runtime_config = db_runtime_config
        self.csv_loaded = bool(csv_loaded)
        self.csv_session_id = str(csv_session_id or "").strip() or None
        self.max_rows = max(1, min(int(max_rows), 1000))
        self.csv_runtime = CSVSessionRuntime()

        llm_kwargs: dict[str, Any] = {
            "model": llm_model,
            "base_url": llm_base_url,
            "api_key": llm_api_key,
            "streaming": False,
            "temperature": 0.0,
            "timeout": 120.0,
        }
        if llm_chat_template_kwargs_enabled:
            llm_kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": llm_enable_thinking}
            }
        self.llm = ChatOpenAI(**llm_kwargs)

    def _db_helper(self) -> DBAnalyticsHelper:
        if self.db_runtime_config is None:
            raise ValueError("DB runtime is not configured")
        return DBAnalyticsHelper(runtime=self.db_runtime_config, timeout_sec=15.0)

    def _collect_db_candidates(self) -> list[TableCandidate]:
        if self.db_runtime_config is None:
            return []

        helper = self._db_helper()
        rows = helper.list_tables()
        out: list[TableCandidate] = []
        for row in rows:
            schema = row.get("schema")
            table_name = str(row.get("table_name") or "").strip()
            if not table_name:
                continue
            qualified_name = str(row.get("qualified_name") or table_name).strip()
            columns_meta = helper.describe_table(table_name, schema=schema)
            columns = [
                str(item.get("column_name") or "").strip()
                for item in columns_meta
                if str(item.get("column_name") or "").strip()
            ]
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

    def _collect_csv_candidates(self) -> list[TableCandidate]:
        if not self.csv_loaded or not self.csv_session_id:
            return []

        rows = self.csv_runtime.list_tables(self.csv_session_id)
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
            out.append(
                TableCandidate(
                    source_kind="csv_session",
                    dialect="duckdb",
                    table_name=table_name,
                    qualified_name=table_name,
                    schema="main",
                    columns=columns,
                    source_label=f"CSV session {self.csv_session_id}",
                    source_ref_id=self.csv_session_id,
                    csv_session_id=self.csv_session_id,
                )
            )
        return out

    def collect_candidates(self) -> list[TableCandidate]:
        return self._collect_db_candidates() + self._collect_csv_candidates()

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
                    tool_name="sql_table_tool",
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
                    tool_name="sql_table_tool",
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
    def _normalized_question(question: str) -> str:
        return re.sub(r"\s+", " ", str(question or "").lower()).strip()

    def _find_explicit_table(self, question: str, candidates: list[TableCandidate]) -> TableCandidate | None:
        normalized_question = self._normalized_question(question)
        for candidate in candidates:
            for name in (candidate.table_name.lower(), candidate.qualified_name.lower()):
                if name and name in normalized_question:
                    return candidate
        return None

    def _choose_table_via_llm(self, question: str, candidates: list[TableCandidate]) -> TableCandidate:
        preview_rows = []
        for idx, candidate in enumerate(candidates[:40], start=1):
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
                SystemMessage(content="/no_think Верни только валидный JSON."),
                HumanMessage(content=prompt),
            ]
        )
        obj = _safe_json_loads(_message_text(resp.content))
        if obj and isinstance(obj.get("idx"), int):
            idx = int(obj["idx"])
            if 1 <= idx <= len(preview_rows):
                return candidates[idx - 1]

        return candidates[0]

    def resolve_table(self, question: str) -> TableCandidate:
        candidates = self.collect_candidates()
        if not candidates:
            raise ValueError("Нет доступных таблиц ни из DB runtime, ни из CSV session.")

        explicit = self._find_explicit_table(question, candidates)
        if explicit is not None:
            return explicit
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

    def _table_sample(self, candidate: TableCandidate) -> dict[str, Any]:
        try:
            sample_sql = f"SELECT * FROM {candidate.qualified_name} LIMIT 5"
            if candidate.source_kind == "db":
                rows = self._db_helper().query_dataframe(sample_sql)
            else:
                rows = self.csv_runtime.query_dataframe(str(candidate.csv_session_id), sample_sql)
            return {"first_rows": rows.to_dict(orient="records")}
        except Exception:
            return {"first_rows": []}

    def _call_llm_sql_only(
        self,
        *,
        question: str,
        candidate: TableCandidate,
        previous_sql: str | None = None,
        feedback: str | None = None,
    ) -> str:
        table_name = candidate.qualified_name
        columns_str = ", ".join(candidate.columns)

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
- используй таблицу {table_name}
- используй только эти колонки: {columns_str}
- старайся вернуть компактный результат
""".strip()
        else:
            user_prompt = f"""
Сгенерируй один SQL-запрос для {candidate.dialect}.

Правила:
- верни только SQL
- только SELECT/WITH
- используй таблицу {table_name}
- используй только эти колонки: {columns_str}
- старайся вернуть компактный результат

Вопрос:
{question}
""".strip()

        sample = self._table_sample(candidate)
        if sample.get("first_rows"):
            user_prompt += "\n\nSAMPLE_ROWS:\n" + json.dumps(sample["first_rows"], ensure_ascii=False)

        resp = self.llm.invoke(
            [
                SystemMessage(content="/no_think Ты SQL-генератор. Верни только SQL SELECT/WITH."),
                HumanMessage(content=user_prompt),
            ]
        )
        return clean_sql(_message_text(resp.content))

    @staticmethod
    def _is_trivial_sql(sql: str) -> bool:
        """Detect trivial queries that don't need LLM judge validation."""
        s = re.sub(r"\s+", " ", sql.strip().upper())
        # SELECT * / SELECT col, col ... FROM ... (no subqueries, no joins)
        if re.match(r"^SELECT\s+.+?\s+FROM\s+\S+(\s+(WHERE|LIMIT|ORDER\s+BY|OFFSET)\s+.*)?\s*;?\s*$", s):
            # No subquery, no JOIN
            if "JOIN" not in s and s.count("SELECT") == 1:
                return True
        # Simple aggregate: SELECT COUNT/SUM/AVG/MIN/MAX(...)
        if re.match(r"^SELECT\s+(COUNT|SUM|AVG|MIN|MAX)\s*\(", s) and s.count("SELECT") == 1:
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
        prompt = f"""
Ты строгий ревьюер SQL ({candidate.dialect}).
Проверь, решает ли SQL вопрос пользователя.
Верни только JSON формата {{"ok": true/false, "reason": "...", "fix_hint": "..."}}.

QUESTION:
{question}

SQL:
{sql}

SAMPLE_RESULT:
{json.dumps(sample_result, ensure_ascii=False)}
""".strip()

        resp = self.llm.invoke(
            [
                SystemMessage(content="/no_think Верни только JSON."),
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
    ) -> dict[str, Any]:
        previous_sql: str | None = None
        feedback: str | None = None

        for attempt in range(1, max_attempts + 1):
            sql = self._call_llm_sql_only(
                question=question,
                candidate=candidate,
                previous_sql=previous_sql,
                feedback=feedback,
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

            _, err = self._run_query_no_throw(candidate, wrap_limit0(sql))
            if err:
                feedback = f"DB compile error: {err}"
                continue

            sample_res, err = self._run_query_no_throw(candidate, wrap_sample(sql, sample_rows))
            if err:
                feedback = f"DB runtime error: {err}"
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
        query_meta = {
            "purpose": question,
            "requested_sql": validation["requested_sql"],
            "executed_sql": executed_sql,
            "max_rows": self.max_rows,
            "requested_limit": validation["requested_limit"],
            "returned_rows": int(len(safe_rows)),
            "column_count": int(len(safe_rows.columns)),
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
                "source_kind": candidate.source_kind,
                "dialect": candidate.dialect,
                "table_name": candidate.table_name,
                "qualified_name": candidate.qualified_name,
                "columns": list(candidate.columns),
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
                    tool_name="sql_table_tool",
                    summary=f"Analytical read query; max_rows={self.max_rows}",
                )
            ],
            "meta": meta,
        }

    def execute_final_query(
        self,
        *,
        question: str,
        candidate: TableCandidate,
        sql: str,
    ) -> dict[str, Any]:
        artifact_name = f"sql_{candidate.table_name}"

        if candidate.source_kind == "db":
            payload = self._db_helper().execute_analytic_query(
                sql,
                purpose=question,
                max_rows=self.max_rows,
                artifact_name=artifact_name,
            )
            meta = dict(payload.get("meta") or {})
            meta["table_selection"] = {
                "source_kind": candidate.source_kind,
                "dialect": candidate.dialect,
                "table_name": candidate.table_name,
                "qualified_name": candidate.qualified_name,
                "columns": list(candidate.columns),
            }
            payload["meta"] = meta
            return payload

        return self._package_csv_query_result(
            candidate=candidate,
            question=question,
            sql=sql,
            artifact_name=artifact_name,
        )

    def build_table_artifact(self, question: str) -> dict[str, Any]:
        if self._wants_catalog_table_list(question):
            return self._build_catalog_table_list_artifact()
        candidate = self.resolve_table(question)
        gen = self.generate_sql_with_retries(question=question, candidate=candidate)
        if not gen["ok"]:
            raise ValueError(str(gen.get("error") or "SQL generation failed"))
        return self.execute_final_query(
            question=question,
            candidate=candidate,
            sql=str(gen["sql"]),
        )


