"""Universal segment aggregation: consistent shares across charts and text summaries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from pandas.api.types import is_numeric_dtype

from backend.data_access.dataframe_utils import column_series, deduplicate_dataframe_columns

MetricKind = Literal["weight", "absolute"]

_WEIGHT_COLUMN_RE = re.compile(
    r"(weight_pct|_weight|share_pct|_share|доля|процент\s*портфел)",
    re.IGNORECASE,
)
_WEIGHT_LOOSE_RE = re.compile(
    r"(^weight$|weight_pct|share_pct|доля)",
    re.IGNORECASE,
)
_WEIGHT_EXCLUDE_RE = re.compile(
    r"(pnl|return|yield|margin|discount|conv|rate|volatility|drawdown|roe|pe_|income)",
    re.IGNORECASE,
)
_VALUE_COLUMN_RE = re.compile(
    r"(value_mln|revenue|sales|amount|turnover|qty|quantity|sum|total|"
    r"выруч|продаж|объем|объём|стоимост|сумм)",
    re.IGNORECASE,
)
_OVERLOAD_THRESHOLD_PCT = 35.0


@dataclass(frozen=True)
class SegmentMetric:
    column: str
    kind: MetricKind

    @property
    def unit(self) -> str:
        return "%" if self.kind == "weight" else "value"


@dataclass(frozen=True)
class SegmentShareRow:
    label: str
    value: float
    share_pct: float
    unit: str


def _pick_column(columns: list[str], patterns: tuple[str, ...]) -> str | None:
    lowered = {col: col.lower() for col in columns}
    for pattern in patterns:
        for col, col_lower in lowered.items():
            if pattern in col_lower:
                return col
    return None


def is_weight_share_column(name: str) -> bool:
    col = str(name or "").strip()
    if not col:
        return False
    if _WEIGHT_EXCLUDE_RE.search(col) and not _WEIGHT_COLUMN_RE.search(col):
        if "weight" not in col.lower() and "share" not in col.lower() and "доля" not in col.lower():
            return False
    return bool(_WEIGHT_COLUMN_RE.search(col) or _WEIGHT_LOOSE_RE.search(col))


def resolve_segment_metric(df: pd.DataFrame) -> SegmentMetric | None:
    """Pick one metric for segment shares: weight/share column first, else absolute sum column."""
    df = deduplicate_dataframe_columns(df)
    columns = [str(c) for c in df.columns]

    for col in columns:
        if is_weight_share_column(col):
            series = column_series(df, col)
            if is_numeric_dtype(series) and float(series.fillna(0).abs().sum()) > 0:
                return SegmentMetric(column=col, kind="weight")

    for col in columns:
        if _VALUE_COLUMN_RE.search(col) and is_numeric_dtype(column_series(df, col)):
            if "pct" in col.lower() and "weight" not in col.lower() and "share" not in col.lower():
                continue
            series = pd.to_numeric(column_series(df, col), errors="coerce").fillna(0)
            if float(series.abs().sum()) > 0:
                return SegmentMetric(column=col, kind="absolute")

    numeric = [
        str(c)
        for c in columns
        if is_numeric_dtype(column_series(df, c))
        and not _WEIGHT_EXCLUDE_RE.search(str(c))
    ]
    if numeric:
        best = max(
            numeric,
            key=lambda c: float(
                pd.to_numeric(column_series(df, c), errors="coerce").fillna(0).abs().sum()
            ),
        )
        return SegmentMetric(column=best, kind="absolute")
    return None


def value_unit_hint(column: str) -> str:
    lowered = str(column).lower()
    if "mln" in lowered or "млн" in lowered:
        return "млн руб."
    if "pct" in lowered or "percent" in lowered or "доля" in lowered:
        return "%"
    if "rub" in lowered or "руб" in lowered:
        return "руб."
    return ""


def compute_segment_shares(
    df: pd.DataFrame,
    segment_col: str,
    *,
    metric: SegmentMetric | None = None,
    limit: int = 12,
) -> list[SegmentShareRow]:
    """SUM metric within segment; share_pct = segment_sum / total_sum * 100 (never AVG of pct)."""
    df = deduplicate_dataframe_columns(df)
    if segment_col not in df.columns or df.empty:
        return []

    resolved = metric or resolve_segment_metric(df)
    if not resolved:
        return []

    work = pd.DataFrame(
        {
            "segment": column_series(df, segment_col).astype(str),
            "value": pd.to_numeric(column_series(df, resolved.column), errors="coerce"),
        }
    )
    work = work.dropna(subset=["value"])
    if work.empty:
        return []

    agg = work.groupby("segment", dropna=False)["value"].sum()
    agg = agg.sort_values(ascending=False).head(limit)
    total = float(agg.sum())
    if total <= 0:
        return []

    unit = "%" if resolved.kind == "weight" else value_unit_hint(resolved.column)
    rows: list[SegmentShareRow] = []
    for label, amount in agg.items():
        share = float(amount) / total * 100.0
        rows.append(
            SegmentShareRow(
                label=str(label),
                value=float(amount),
                share_pct=round(share, 2),
                unit=unit,
            )
        )
    return rows


def detect_overload_segments(
    rows: list[SegmentShareRow],
    *,
    threshold_pct: float = _OVERLOAD_THRESHOLD_PCT,
) -> list[SegmentShareRow]:
    return [row for row in rows if row.share_pct >= threshold_pct]


def compose_analytical_brief(
    *,
    essence: str,
    key_sections: list[str],
    insights: list[str],
    chart_lines: list[str],
    next_steps: list[str],
) -> str:
    """Standard five-part analytical answer (domain-agnostic)."""
    parts = [
        "### 1. Суть",
        essence.strip(),
        "",
        "### 2. Ключевые цифры",
        "\n".join(f"- {line.lstrip('- ')}" for line in key_sections if line.strip()),
    ]
    if insights:
        parts.extend(
            [
                "",
                "### 3. Инсайты",
                "\n".join(f"- {line.lstrip('- ')}" for line in insights if line.strip()),
            ]
        )
    if chart_lines:
        parts.extend(
            [
                "",
                "### 4. Графики и артефакты",
                "\n".join(f"- {line.lstrip('- ')}" for line in chart_lines if line.strip()),
            ]
        )
    if next_steps:
        parts.extend(
            [
                "",
                "### 5. Что проверить дальше",
                "\n".join(f"- {line.lstrip('- ')}" for line in next_steps if line.strip()),
            ]
        )
    return "\n".join(parts).strip()


def format_share_line(row: SegmentShareRow, *, show_absolute: bool = False) -> str:
    if row.unit == "%":
        text = f"**{row.label}**: {row.share_pct}%"
        if show_absolute:
            text += f" (сумма весов {row.value:.2f})"
        return text
    unit_suffix = f" {row.unit}" if row.unit else ""
    return (
        f"**{row.label}**: {row.value:.2f}{unit_suffix} "
        f"({row.share_pct}% от суммы)"
    )
