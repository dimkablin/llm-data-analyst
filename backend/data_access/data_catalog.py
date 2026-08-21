"""Session data catalog: table/column metadata for prompts and SQL routing."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

_CATALOG_VERSION = "1.0"
_MAX_TABLES_IN_PROMPT = 24
_MAX_COLUMNS_PER_TABLE = 48


@dataclass
class CatalogColumn:
    name: str
    dtype: str = ""
    nullable: bool | None = None
    null_ratio: float | None = None
    distinct_count: int | None = None
    examples: list[str] = field(default_factory=list)
    min_value: str | None = None
    max_value: str | None = None
    top_values: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "dtype": self.dtype}
        if self.nullable is not None:
            out["nullable"] = self.nullable
        if self.null_ratio is not None:
            out["null_ratio"] = self.null_ratio
        if self.distinct_count is not None:
            out["distinct_count"] = self.distinct_count
        if self.examples:
            out["examples"] = self.examples
        if self.min_value is not None:
            out["min_value"] = self.min_value
        if self.max_value is not None:
            out["max_value"] = self.max_value
        if self.top_values:
            out["top_values"] = self.top_values
        return out

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CatalogColumn:
        return cls(
            name=str(raw.get("name") or "").strip(),
            dtype=str(raw.get("dtype") or "").strip(),
            nullable=raw.get("nullable") if "nullable" in raw else None,
            null_ratio=float(raw["null_ratio"]) if raw.get("null_ratio") is not None else None,
            distinct_count=(
                int(raw["distinct_count"])
                if raw.get("distinct_count") is not None
                else None
            ),
            examples=[str(item) for item in raw.get("examples") or []],
            min_value=str(raw["min_value"]) if raw.get("min_value") is not None else None,
            max_value=str(raw["max_value"]) if raw.get("max_value") is not None else None,
            top_values=[str(item) for item in raw.get("top_values") or []],
        )


@dataclass
class CatalogTable:
    qualified_name: str
    table_name: str
    source_kind: str
    columns: list[CatalogColumn] = field(default_factory=list)
    schema: str | None = None
    table_type: str | None = None
    row_estimate: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "qualified_name": self.qualified_name,
            "table_name": self.table_name,
            "source_kind": self.source_kind,
            "schema": self.schema,
            "table_type": self.table_type,
            "row_estimate": self.row_estimate,
            "columns": [c.to_dict() for c in self.columns],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CatalogTable:
        cols = [
            CatalogColumn.from_dict(item)
            for item in (raw.get("columns") or [])
            if isinstance(item, dict)
        ]
        return cls(
            qualified_name=str(raw.get("qualified_name") or "").strip(),
            table_name=str(raw.get("table_name") or "").strip(),
            source_kind=str(raw.get("source_kind") or "").strip(),
            columns=[c for c in cols if c.name],
            schema=str(raw.get("schema") or "").strip() or None,
            table_type=str(raw.get("table_type") or "").strip() or None,
            row_estimate=(
                int(raw["row_estimate"])
                if raw.get("row_estimate") is not None
                else None
            ),
        )


@dataclass
class DataCatalogSnapshot:
    version: str = _CATALOG_VERSION
    built_at: str = ""
    source_fingerprint: str = ""
    profile_sample_strategy: str = ""
    profile_sample_limit: int | None = None
    tables: list[CatalogTable] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "built_at": self.built_at,
            "source_fingerprint": self.source_fingerprint,
            "profile_sample_strategy": self.profile_sample_strategy,
            "profile_sample_limit": self.profile_sample_limit,
            "tables": [t.to_dict() for t in self.tables],
            "errors": list(self.errors),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DataCatalogSnapshot:
        tables = [
            CatalogTable.from_dict(item)
            for item in (raw.get("tables") or [])
            if isinstance(item, dict)
        ]
        return cls(
            version=str(raw.get("version") or _CATALOG_VERSION),
            built_at=str(raw.get("built_at") or ""),
            source_fingerprint=str(raw.get("source_fingerprint") or ""),
            profile_sample_strategy=str(raw.get("profile_sample_strategy") or ""),
            profile_sample_limit=(
                int(raw["profile_sample_limit"]) if raw.get("profile_sample_limit") is not None else None
            ),
            tables=[t for t in tables if t.qualified_name],
            errors=[str(item) for item in raw.get("errors") or [] if str(item).strip()],
        )


@dataclass(frozen=True)
class CatalogProfileOptions:
    max_tables: int = 100
    max_columns_per_table: int = 200
    sample_rows: int = 1000
    top_values: int = 20


def _dtype_label(series: pd.Series) -> str:
    if is_datetime64_any_dtype(series):
        return "datetime"
    if is_numeric_dtype(series):
        return str(series.dtype)
    return "string"


_SENSITIVE_COLUMN_RE = re.compile(
    r"(password|passwd|secret|token|api[_-]?key|credential|dsn|email|phone)",
    re.IGNORECASE,
)


def _safe_profile_value(column: str, value: Any) -> str:
    if _SENSITIVE_COLUMN_RE.search(column):
        return "<redacted>"
    text = str(value)
    return text if len(text) <= 80 else text[:77] + "..."


def _hashable_profile_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except TypeError:
            return str(value)
    return value


def _profile_column(
    name: str,
    series: pd.Series,
    *,
    dtype: str = "",
    nullable: bool | None = None,
    top_values: int,
) -> CatalogColumn:
    non_null = series.dropna()
    profile_values = non_null.map(_hashable_profile_value)
    examples = [
        _safe_profile_value(name, value)
        for value in profile_values.drop_duplicates().head(5).tolist()
    ]
    popular = [
        _safe_profile_value(name, value)
        for value in profile_values.value_counts(dropna=True).head(top_values).index.tolist()
    ]
    min_value: str | None = None
    max_value: str | None = None
    if len(non_null) and (is_numeric_dtype(series) or is_datetime64_any_dtype(series)):
        min_value = _safe_profile_value(name, non_null.min())
        max_value = _safe_profile_value(name, non_null.max())
    return CatalogColumn(
        name=name,
        dtype=dtype or _dtype_label(series),
        nullable=nullable if nullable is not None else bool(series.isna().any()),
        null_ratio=round(float(series.isna().mean()), 6) if len(series) else 0.0,
        distinct_count=int(profile_values.nunique(dropna=True)),
        examples=examples,
        min_value=min_value,
        max_value=max_value,
        top_values=popular,
    )


def build_table_from_dataframe(
    df: pd.DataFrame,
    *,
    qualified_name: str = "df",
    source_kind: str = "dataframe",
    options: CatalogProfileOptions | None = None,
) -> CatalogTable:
    profile = options or CatalogProfileOptions()
    sample = df.head(profile.sample_rows)
    columns: list[CatalogColumn] = []
    for col in list(df.columns)[: profile.max_columns_per_table]:
        name = str(col)
        columns.append(
            _profile_column(
                name,
                sample[col],
                top_values=profile.top_values,
            )
        )
    return CatalogTable(
        qualified_name=qualified_name,
        table_name=qualified_name,
        source_kind=source_kind,
        columns=columns,
        row_estimate=len(df),
    )


def build_snapshot_from_dataframe(
    df: pd.DataFrame,
    *,
    qualified_name: str = "df",
    fingerprint: str = "",
    options: CatalogProfileOptions | None = None,
) -> DataCatalogSnapshot:
    profile = options or CatalogProfileOptions()
    return DataCatalogSnapshot(
        built_at=datetime.now(UTC).isoformat(),
        source_fingerprint=fingerprint,
        profile_sample_strategy="first_rows",
        profile_sample_limit=profile.sample_rows,
        tables=[build_table_from_dataframe(df, qualified_name=qualified_name, options=profile)],
    )


def build_snapshot_from_csv_runtime(
    csv_runtime: Any,
    csv_session_id: str,
    *,
    fingerprint: str = "",
    options: CatalogProfileOptions | None = None,
    deadline: float | None = None,
) -> DataCatalogSnapshot:
    sid = str(csv_session_id or "").strip()
    if not sid:
        return DataCatalogSnapshot(built_at=datetime.now(UTC).isoformat())

    profile = options or CatalogProfileOptions()
    tables: list[CatalogTable] = []
    errors: list[str] = []
    for row in csv_runtime.list_tables(sid)[: profile.max_tables]:
        if deadline is not None and time.monotonic() >= deadline:
            errors.append("CSV profiling stopped after SEMANTIC_PROFILE_TIMEOUT_SEC")
            break
        table_name = str(row.get("table_name") or "").strip()
        if not table_name:
            continue
        meta = csv_runtime.describe_table(sid, table_name)[: profile.max_columns_per_table]
        try:
            quoted = '"' + table_name.replace('"', '""') + '"'
            sample = csv_runtime.query_dataframe(
                sid,
                f"SELECT * FROM {quoted} LIMIT {max(0, profile.sample_rows)}",
            )
        except Exception as exc:
            sample = pd.DataFrame()
            errors.append(f"CSV profile failed for {table_name}: {exc}")
        columns: list[CatalogColumn] = []
        for item in meta:
            name = str(item.get("column_name") or "").strip()
            if not name:
                continue
            if name in sample.columns:
                columns.append(
                    _profile_column(
                        name,
                        sample[name],
                        dtype=str(item.get("data_type") or "").strip(),
                        nullable=(
                            bool(item.get("is_nullable"))
                            if item.get("is_nullable") is not None
                            else None
                        ),
                        top_values=profile.top_values,
                    )
                )
            else:
                columns.append(
                    CatalogColumn(
                        name=name,
                        dtype=str(item.get("data_type") or "").strip(),
                        nullable=(
                            bool(item.get("is_nullable"))
                            if item.get("is_nullable") is not None
                            else None
                        ),
                    )
                )
        tables.append(
            CatalogTable(
                qualified_name=str(row.get("qualified_name") or table_name),
                table_name=table_name,
                source_kind="csv_session",
                schema=str(row.get("schema") or "main") or None,
                table_type=str(row.get("table_type") or "") or None,
                columns=columns,
            )
        )

    return DataCatalogSnapshot(
        built_at=datetime.now(UTC).isoformat(),
        source_fingerprint=fingerprint,
        profile_sample_strategy="first_rows",
        profile_sample_limit=profile.sample_rows,
        tables=tables,
        errors=errors,
    )


def build_snapshot_from_db_helper(
    helper: Any,
    *,
    fingerprint: str = "",
    options: CatalogProfileOptions | None = None,
    deadline: float | None = None,
) -> DataCatalogSnapshot:
    profile = options or CatalogProfileOptions()
    tables: list[CatalogTable] = []
    errors: list[str] = []
    for row in helper.list_tables_with_columns()[: profile.max_tables]:
        if deadline is not None and time.monotonic() >= deadline:
            errors.append("DB profiling stopped after SEMANTIC_PROFILE_TIMEOUT_SEC")
            break
        table_name = str(row.get("table_name") or "").strip()
        if not table_name:
            continue
        col_names = [
            str(c)
            for c in row.get("columns", [])[: profile.max_columns_per_table]
            if str(c).strip()
        ]
        raw_column_types = row.get("column_types")
        column_types = raw_column_types if isinstance(raw_column_types, dict) else {}
        declared_types = {
            name: str(column_types.get(name) or "").strip() for name in col_names
        }
        sample = pd.DataFrame()
        preview = getattr(helper, "preview_table", None)
        if callable(preview) and profile.sample_rows > 0:
            try:
                sample = preview(
                    table_name,
                    schema=str(row.get("schema") or "").strip() or None,
                    limit=profile.sample_rows,
                )
            except Exception as exc:
                errors.append(f"DB profile failed for {row.get('qualified_name') or table_name}: {exc}")
        columns: list[CatalogColumn] = []
        for name in col_names:
            dtype = declared_types[name]
            if name in sample.columns:
                columns.append(
                    _profile_column(
                        name, sample[name], dtype=dtype, top_values=profile.top_values
                    )
                )
            else:
                columns.append(CatalogColumn(name=name, dtype=dtype))
        tables.append(
            CatalogTable(
                qualified_name=str(row.get("qualified_name") or table_name),
                table_name=table_name,
                source_kind="db",
                schema=str(row.get("schema") or "").strip() or None,
                table_type=str(row.get("table_type") or "") or None,
                columns=columns,
            )
        )
    return DataCatalogSnapshot(
        built_at=datetime.now(UTC).isoformat(),
        source_fingerprint=fingerprint,
        profile_sample_strategy="first_rows",
        profile_sample_limit=profile.sample_rows,
        tables=tables,
        errors=errors,
    )


def merge_snapshots(*parts: DataCatalogSnapshot) -> DataCatalogSnapshot:
    seen: set[str] = set()
    merged: list[CatalogTable] = []
    fingerprint_parts: list[str] = []
    built_at = datetime.now(UTC).isoformat()
    errors: list[str] = []
    sample_limits: list[int] = []
    for part in parts:
        errors.extend(part.errors)
        if part.profile_sample_limit is not None:
            sample_limits.append(part.profile_sample_limit)
        if part.source_fingerprint:
            fingerprint_parts.append(part.source_fingerprint)
        for table in part.tables:
            key = f"{table.source_kind}:{table.qualified_name}"
            if key in seen:
                continue
            seen.add(key)
            merged.append(table)
    return DataCatalogSnapshot(
        built_at=built_at,
        source_fingerprint="|".join(fingerprint_parts),
        profile_sample_strategy="first_rows" if sample_limits else "",
        profile_sample_limit=max(sample_limits) if sample_limits else None,
        tables=merged,
        errors=list(dict.fromkeys(errors)),
    )


def format_catalog_prompt_block(
    snapshot: DataCatalogSnapshot | None,
    *,
    max_tables: int = _MAX_TABLES_IN_PROMPT,
    max_columns_per_table: int = _MAX_COLUMNS_PER_TABLE,
) -> str:
    if snapshot is None or not snapshot.tables:
        return ""

    lines = [
        "═══ Каталог данных (схема сессии) ═══",
        "Используй ТОЧНЫЕ имена таблиц и колонок из каталога. "
        "Для SQL — `sql_tool`; для расчётов/графиков — переменные sandbox после sql_tool.",
    ]
    if snapshot.profile_sample_strategy == "first_rows" and snapshot.profile_sample_limit:
        lines.append(
            "Важно: статистика профиля (null_ratio, distinct_count, min/max и top_values) "
            f"рассчитана только по первым {snapshot.profile_sample_limit} строкам каждой таблицы, "
            "а не по всему источнику."
        )

    for table in snapshot.tables[:max_tables]:
        col_parts = [
            _format_catalog_column_prompt_label(col)
            for col in table.columns[:max_columns_per_table]
        ]
        extra = len(table.columns) - max_columns_per_table
        if extra > 0:
            col_parts.append(f"... +{extra} колонок")
        row_hint = f", ~{table.row_estimate} строк" if table.row_estimate else ""
        lines.append(
            f"- **{table.qualified_name}** [{table.source_kind}]{row_hint}: "
            + ", ".join(col_parts) if col_parts else "(нет колонок)"
        )

    if len(snapshot.tables) > max_tables:
        lines.append(f"- ... ещё {len(snapshot.tables) - max_tables} таблиц")

    return "\n".join(lines)


def _format_catalog_column_prompt_label(col: CatalogColumn) -> str:
    """Schema-only column label for LLM prompts."""
    label = f"`{col.name}`"
    if col.dtype:
        label += f" ({col.dtype})"
    return label


def snapshot_to_json(snapshot: DataCatalogSnapshot) -> str:
    return json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=0)


def snapshot_from_json(text: str) -> DataCatalogSnapshot | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        return DataCatalogSnapshot.from_dict(json.loads(raw))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def catalog_cache_key(
    *,
    session_id: str,
    source_type: str | None,
    source_ref_id: str | None,
    csv_session_id: str | None,
) -> str:
    parts = [
        str(session_id or "").strip(),
        str(source_type or "").strip().lower(),
        str(source_ref_id or "").strip(),
        str(csv_session_id or "").strip(),
    ]
    payload = "|".join(parts)
    return payload


_FUZZY_STRIP_RE = re.compile(r"[\s_\-]+")


def fuzzy_match_column(name: str, candidates: list[str]) -> str | None:
    """Return best matching column name or None."""
    return fuzzy_match_identifier(name, candidates)


_SEMANTIC_COLUMN_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"visits", "visit", "traffic", "trafik", "посетители", "посетитель"}),
    frozenset({"conversion", "конверсия", "конверсию", "покупку", "purchase"}),
    frozenset({"discount", "скидка", "скидки"}),
    frozenset({"revenue", "выручка", "sales", "value", "стоимость"}),
    frozenset({"volume", "объём", "объем", "qty", "quantity", "продаж"}),
    frozenset({"pnl", "unrealized", "profit", "loss", "результат", "доход", "убыток"}),
)


def _abs_to_pct_column_match(name: str, candidates: list[str]) -> str | None:
    """Map hallucinated *_abs PnL columns to existing *_pct columns."""
    lower = str(name or "").lower()
    if "abs" not in lower and not lower.endswith("_abs"):
        return None
    pct_cols = [c for c in candidates if "pct" in c.lower() or "percent" in c.lower()]
    if not pct_cols:
        return None
    if "pnl" in lower or "unrealized" in lower:
        for cand in pct_cols:
            if "pnl" in cand.lower() or "unrealized" in cand.lower():
                return cand
        return pct_cols[0]
    return None


def _column_name_tokens(name: str) -> set[str]:
    import re as _re

    raw = str(name or "").lower()
    for prefix in ("total_", "avg_", "sum_", "mean_", "max_", "min_"):
        if raw.startswith(prefix):
            raw = raw[len(prefix) :]
    return {t for t in _re.findall(r"[a-zа-яё0-9]+", raw) if t}


def _semantic_column_match(name: str, candidates: list[str]) -> str | None:
    tokens = _column_name_tokens(name)
    if not tokens:
        return None
    for group in _SEMANTIC_COLUMN_GROUPS:
        if not (tokens & group):
            continue
        for cand in candidates:
            if _column_name_tokens(cand) & group:
                return cand
    return None


def fuzzy_match_identifier(name: str, candidates: list[str]) -> str | None:
    """Return best matching identifier (column or sandbox variable) or None."""
    import difflib

    target = str(name or "").strip()
    if not target or not candidates:
        return None
    if target in candidates:
        return target
    lower_map = {c.lower(): c for c in candidates}
    if target.lower() in lower_map:
        return lower_map[target.lower()]
    norm_target = _FUZZY_STRIP_RE.sub("", target.lower())
    for cand in candidates:
        if _FUZZY_STRIP_RE.sub("", cand.lower()) == norm_target:
            return cand
    semantic = _semantic_column_match(target, candidates)
    if semantic is not None:
        return semantic
    abs_pct = _abs_to_pct_column_match(target, candidates)
    if abs_pct is not None:
        return abs_pct
    close = difflib.get_close_matches(target, candidates, n=1, cutoff=0.72)
    if close:
        return close[0]
    target_parts = [p for p in target.lower().split("_") if p]
    if len(target_parts) >= 2:
        best: str | None = None
        best_score = 0.0
        for cand in candidates:
            cand_parts = [p for p in cand.lower().split("_") if p]
            if not cand_parts:
                continue
            overlap = len(set(target_parts) & set(cand_parts))
            score = overlap / max(len(target_parts), len(cand_parts))
            if score > best_score:
                best_score = score
                best = cand
        if best is not None and best_score >= 0.6:
            return best
    return None


def format_dataframe_columns_hint(df: pd.DataFrame, *, name: str) -> str:
    cols = [str(c) for c in df.columns]
    dtypes = ", ".join(f"`{c}` ({_dtype_label(df[c])})" for c in cols[:40])
    if len(cols) > 40:
        dtypes += f", ... +{len(cols) - 40}"
    return f"Колонки `{name}` ({len(df)}×{len(cols)}): {dtypes}"
