"""Universal chart-type selection from table shape and question intent."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from backend.data_access.dataframe_utils import column_nunique, column_series

ChartKind = Literal["pie", "bar", "bar_h", "scatter", "line"]
PlotIntent = Literal[
    "structure",
    "concentration",
    "dynamics",
    "top_n",
    "comparison",
    "generic",
]

_TIME_COLUMN_RE = re.compile(
    r"(^|_)(date|time|dt|timestamp)(_|$)|month|year|week|period",
    re.IGNORECASE,
)
_METRIC_FALSE_TIME_RE = re.compile(
    r"(sentiment|rate|ratio|amount|count|total|score|value|quantity|qty)",
    re.IGNORECASE,
)
_ID_LIKE_RE = re.compile(r"(^id$|_id$|uuid|guid|hash|code$)", re.IGNORECASE)
_PCT_COLUMN_RE = re.compile(r"(pct|percent|share)", re.IGNORECASE)

_PIE_MAX_CATEGORIES = 8
_PIE_MIN_CATEGORIES = 2
_HORIZONTAL_BAR_MIN_CATEGORIES = 7


@dataclass(frozen=True)
class TabularPlotProfile:
    time_columns: tuple[str, ...]
    dimension_columns: tuple[str, ...]
    metric_columns: tuple[str, ...]


def is_plausible_time_column(name: str, series: pd.Series) -> bool:
    """Return True when a column name and values look like a time axis."""
    col = str(name or "").strip()
    if not col or _METRIC_FALSE_TIME_RE.search(col):
        return False
    if is_datetime64_any_dtype(series):
        return True
    if not _TIME_COLUMN_RE.search(col):
        return False
    parsed = pd.to_datetime(series, errors="coerce")
    return bool(parsed.notna().sum() >= 2 and int(parsed.dt.year.min()) >= 1990)


def segment_category_count(df: pd.DataFrame, segment_col: str) -> int:
    if segment_col not in df.columns:
        return 0
    return column_nunique(df, segment_col)


def infer_tabular_plot_profile(df: pd.DataFrame) -> TabularPlotProfile:
    """Classify columns by dtypes and cardinality only."""
    time_cols: list[str] = []
    dimension_cols: list[str] = []
    metric_cols: list[str] = []

    for column in df.columns:
        name = str(column)
        series = column_series(df, name)

        if is_plausible_time_column(name, series):
            time_cols.append(name)
            continue

        if is_numeric_dtype(series):
            if _PCT_COLUMN_RE.search(name):
                continue
            if _ID_LIKE_RE.search(name) and column_nunique(df, name) > len(df) * 0.9:
                continue
            metric_cols.append(name)
            continue

        unique = column_nunique(df, name)
        if 1 < unique <= 64:
            dimension_cols.append(name)

    dimension_cols.sort(
        key=lambda col: (
            segment_category_count(df, col),
            -int(pd.to_numeric(column_series(df, col), errors="coerce").notna().sum()),
        )
    )
    metric_cols.sort(
        key=lambda col: (
            0 if pd.to_numeric(column_series(df, col), errors="coerce").notna().any() else 1,
            -float(
                pd.to_numeric(column_series(df, col), errors="coerce")
                .fillna(0)
                .abs()
                .sum()
            ),
        )
    )
    return TabularPlotProfile(
        time_columns=tuple(time_cols),
        dimension_columns=tuple(dimension_cols),
        metric_columns=tuple(metric_cols),
    )


def pick_chart_kind(
    *,
    intent: PlotIntent,
    df: pd.DataFrame,
    segment_col: str,
    segment_index: int = 0,
) -> ChartKind:
    """Choose chart type from data shape and analytical intent."""
    category_count = segment_category_count(df, segment_col)

    if intent == "dynamics":
        return "line"

    if intent == "comparison" and segment_index == 0:
        profile = infer_tabular_plot_profile(df)
        if len(profile.metric_columns) >= 2:
            return "scatter"

    if category_count >= _HORIZONTAL_BAR_MIN_CATEGORIES:
        return "bar_h"

    if intent == "top_n":
        return "bar_h"

    if intent == "concentration":
        if segment_index == 0 and _PIE_MIN_CATEGORIES <= category_count <= _PIE_MAX_CATEGORIES:
            return "pie"
        return "bar_h"

    if intent == "structure":
        rotation: tuple[ChartKind, ...] = ("pie", "bar", "bar_h", "bar")
        kind = rotation[segment_index % len(rotation)]
        if kind == "pie" and not (_PIE_MIN_CATEGORIES <= category_count <= _PIE_MAX_CATEGORIES):
            kind = "bar"
        if kind == "bar_h" and category_count < _HORIZONTAL_BAR_MIN_CATEGORIES:
            kind = "bar"
        return kind

    if intent == "comparison":
        return "bar"

    if _PIE_MIN_CATEGORIES <= category_count <= 6:
        return "pie"
    return "bar"


def pick_metric_pair(df: pd.DataFrame) -> tuple[str, str] | None:
    profile = infer_tabular_plot_profile(df)
    metrics = list(profile.metric_columns)
    if len(metrics) >= 2:
        metric_names = set(metrics)
        metrics = [str(column) for column in df.columns if str(column) in metric_names]
    if len(metrics) < 2:
        numeric_cols = [
            str(column)
            for column in df.columns
            if is_numeric_dtype(df[column]) and not _PCT_COLUMN_RE.search(str(column))
        ]
        metrics = numeric_cols[:2]
    if len(metrics) < 2:
        return None
    return metrics[0], metrics[1]


def score_dataframe_for_plot(
    frame_name: str,
    df: pd.DataFrame,
    *,
    intent: PlotIntent = "generic",
) -> int:
    """Rank candidate tables for autogen charts without domain-specific table names."""
    profile = infer_tabular_plot_profile(df)
    score = len(df) * 10 + len(df.columns) * 5
    if profile.dimension_columns:
        score += 120
    if profile.metric_columns:
        score += 80
    if intent == "dynamics" and profile.time_columns:
        score += 500
    if intent in {"structure", "concentration", "top_n"} and profile.dimension_columns:
        score += 300
    if intent == "comparison" and len(profile.metric_columns) >= 2:
        score += 200
    if intent in {"comparison", "top_n"} and profile.dimension_columns and not profile.time_columns:
        score += 350

    lowered = frame_name.lower()
    if intent == "dynamics" and any(token in lowered for token in ("time", "month", "period")):
        score += 100
    if profile.time_columns and any(token in lowered for token in ("history", "timeline")):
        score += 450
    if any(token in lowered for token in ("db_tables", "columns_", "describe_", "preview_")):
        score -= 800
    if "lookup" in lowered:
        score -= 1200
    return score
