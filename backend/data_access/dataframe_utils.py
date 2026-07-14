"""Shared DataFrame hygiene helpers for SQL/sandbox/analytics."""

from __future__ import annotations

import pandas as pd


def deduplicate_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make column names unique (empty names → column_N, duplicates → name_2)."""
    raw_columns = [str(col).replace("\ufeff", "").strip() for col in list(df.columns)]
    normalized: list[str] = []
    seen: dict[str, int] = {}
    for idx, col in enumerate(raw_columns):
        base = col or f"column_{idx + 1}"
        count = seen.get(base, 0) + 1
        seen[base] = count
        normalized.append(base if count == 1 else f"{base}_{count}")
    out = df.copy()
    out.columns = normalized
    return out


def column_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a single Series for *column* (first match if names are duplicated)."""
    if column not in df.columns:
        raise KeyError(column)
    data = df.loc[:, column]
    if isinstance(data, pd.DataFrame):
        return data.iloc[:, 0]
    if isinstance(data, pd.Series):
        return data
    return pd.Series(data, name=column)


def column_nunique(df: pd.DataFrame, column: str) -> int:
    """Scalar nunique for *column* — safe when duplicate column labels exist."""
    return int(column_series(df, column).nunique(dropna=False))


def numeric_summary_rows(df: pd.DataFrame, *, precision: int = 4) -> list[dict[str, object]]:
    """Return ready-to-display sum/mean rows for numeric columns."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []

    numeric_cols = [
        column
        for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column])
        and not pd.api.types.is_bool_dtype(df[column])
    ]
    if not numeric_cols:
        return []

    label_column = next((column for column in df.columns if column not in numeric_cols), df.columns[0])
    rows: list[dict[str, object]] = []
    for metric, values in (
        ("__sum__", df[numeric_cols].sum(numeric_only=True)),
        ("__mean__", df[numeric_cols].mean(numeric_only=True)),
    ):
        row: dict[str, object] = {str(column): "" for column in df.columns}
        row[str(label_column)] = metric
        for column in numeric_cols:
            value = values[column]
            if pd.isna(value):
                row[str(column)] = None
            elif float(value).is_integer():
                row[str(column)] = int(value)
            else:
                row[str(column)] = round(float(value), precision)
        rows.append(row)

    return rows
