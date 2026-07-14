from __future__ import annotations

import csv
from io import BytesIO
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from backend.data_access.csv_session_runtime import CSVSessionRuntime

TabularFileFormat = Literal["csv", "xlsx"]

EMPTY_VALUE_PATTERN = r"^\s*$"
DEFAULT_HEADER_SCAN_ROWS = 50
DEFAULT_SPARSE_ROW_MIN_RATIO = 0.5
CSV_DELIMITER_CANDIDATES = (",", ";", "\t", "|")


class TabularPreprocessingOptions(BaseModel):
    enabled: bool = True
    detect_csv_separator: bool = True
    detect_header_row: bool = True
    normalize_empty_values: bool = True
    drop_empty_rows: bool = True
    drop_empty_columns: bool = True
    drop_sparse_rows: bool = True
    unique_column_names: bool = True
    header_scan_rows: int = Field(default=DEFAULT_HEADER_SCAN_ROWS, ge=1, le=500)
    sparse_row_min_ratio: float = Field(
        default=DEFAULT_SPARSE_ROW_MIN_RATIO,
        ge=0.0,
        le=1.0,
    )


class TabularPreprocessingSummary(BaseModel):
    enabled: bool
    raw_rows: int
    raw_columns: int
    cleaned_rows: int
    cleaned_columns: int
    detected_header_row: int | None = None
    removed_rows: int
    removed_columns: int


class PreprocessedTabularData(BaseModel):
    dataframe: object
    summary: TabularPreprocessingSummary

    model_config = {"arbitrary_types_allowed": True}


def detect_csv_separator(file_content: bytes) -> str:
    if not file_content:
        return ","

    sample = file_content[:4096].decode("utf-8", errors="ignore")
    if not sample.strip():
        return ","

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="".join(CSV_DELIMITER_CANDIDATES))
        if dialect.delimiter in CSV_DELIMITER_CANDIDATES:
            return dialect.delimiter
    except csv.Error:
        pass

    non_empty_lines = [line for line in sample.splitlines() if line.strip()]
    if not non_empty_lines:
        return ","

    def score(delimiter: str) -> tuple[int, int, int]:
        field_counts = [len(line.split(delimiter)) for line in non_empty_lines[:20]]
        useful_counts = [count for count in field_counts if count > 1]
        if not useful_counts:
            return (0, 0, 0)
        most_common_width = max(set(useful_counts), key=useful_counts.count)
        consistency = useful_counts.count(most_common_width)
        return (consistency, most_common_width, sum(useful_counts))

    return max(CSV_DELIMITER_CANDIDATES, key=score)


def _normalize_empty_values(df: pd.DataFrame, options: TabularPreprocessingOptions) -> pd.DataFrame:
    if not options.normalize_empty_values:
        return df.copy()
    return df.replace(EMPTY_VALUE_PATTERN, pd.NA, regex=True)


def _normalize_column_name(value: object, fallback: str) -> str:
    if pd.isna(value):
        return fallback

    name = " ".join(str(value).strip().split())
    if not name or name.lower().startswith("unnamed:"):
        return fallback

    return name


def _make_unique_column_names(values: pd.Series) -> list[str]:
    columns: list[str] = []
    seen: dict[str, int] = {}

    for index, value in enumerate(values, start=1):
        base_name = _normalize_column_name(value, f"column_{index}")
        count = seen.get(base_name, 0) + 1
        seen[base_name] = count
        columns.append(base_name if count == 1 else f"{base_name}__{count}")

    return columns


def _infer_cleaned_column_types(df: pd.DataFrame) -> pd.DataFrame:
    converted = df.copy()
    for column in converted.columns:
        series = converted[column]
        if not pd.api.types.is_object_dtype(series):
            continue
        non_empty = series.dropna()
        if non_empty.empty:
            continue
        numeric = pd.to_numeric(series, errors="coerce")
        if int(numeric.notna().sum()) == int(series.notna().sum()):
            converted[column] = numeric
    return converted


def detect_header_row(raw_df: pd.DataFrame, *, scan_rows: int = DEFAULT_HEADER_SCAN_ROWS) -> int:
    if raw_df.empty:
        return 0

    normalized_df = raw_df.replace(EMPTY_VALUE_PATTERN, pd.NA, regex=True)
    limit = min(scan_rows, len(normalized_df))
    non_empty_counts = normalized_df.iloc[:limit].notna().sum(axis=1)

    if non_empty_counts.max() == 0:
        return 0

    return int(non_empty_counts.idxmax())


def clean_dataframe_for_eda(
    raw_df: pd.DataFrame,
    *,
    detect_header: bool,
    options: TabularPreprocessingOptions,
) -> PreprocessedTabularData:
    raw_rows, raw_columns = raw_df.shape
    normalized_df = _normalize_empty_values(raw_df, options)
    detected_header_index: int | None = None

    if detect_header:
        detected_header_index = detect_header_row(
            normalized_df,
            scan_rows=options.header_scan_rows,
        )
        header_values = normalized_df.iloc[detected_header_index]
        df = normalized_df.iloc[detected_header_index + 1 :].copy()

        if options.drop_empty_columns:
            df = df.dropna(axis=1, how="all")
        header_values = header_values.loc[df.columns]
        if options.unique_column_names:
            df.columns = _make_unique_column_names(header_values)
        else:
            df.columns = [str(value) for value in header_values]

        header_non_empty_count = int(header_values.notna().sum())
        min_non_empty_cells = max(
            1,
            min(
                header_non_empty_count,
                max(3, int(header_non_empty_count * options.sparse_row_min_ratio)),
            ),
        )
    else:
        df = normalized_df.copy()
        if options.unique_column_names:
            df.columns = _make_unique_column_names(pd.Series(df.columns))
        min_non_empty_cells = 1

    if options.drop_empty_rows:
        df = df.dropna(axis=0, how="all")
    if options.drop_empty_columns:
        df = df.dropna(axis=1, how="all")

    if options.drop_sparse_rows and not df.empty:
        non_empty_counts = df.notna().sum(axis=1)
        df = df.loc[non_empty_counts >= min_non_empty_cells]

    df = _infer_cleaned_column_types(df)
    df = CSVSessionRuntime._normalize_columns(df.reset_index(drop=True))  # noqa: SLF001
    summary = TabularPreprocessingSummary(
        enabled=options.enabled,
        raw_rows=raw_rows,
        raw_columns=raw_columns,
        cleaned_rows=len(df),
        cleaned_columns=len(df.columns),
        detected_header_row=detected_header_index + 1 if detected_header_index is not None else None,
        removed_rows=raw_rows - len(df),
        removed_columns=raw_columns - len(df.columns),
    )
    return PreprocessedTabularData(dataframe=df, summary=summary)


def _legacy_summary(df: pd.DataFrame) -> TabularPreprocessingSummary:
    return TabularPreprocessingSummary(
        enabled=False,
        raw_rows=len(df),
        raw_columns=len(df.columns),
        cleaned_rows=len(df),
        cleaned_columns=len(df.columns),
        removed_rows=0,
        removed_columns=0,
    )


def _read_legacy_dataframe(content: bytes, *, file_format: TabularFileFormat) -> pd.DataFrame:
    if file_format == "csv":
        return CSVSessionRuntime._read_csv_resilient(  # noqa: SLF001
            content,
            pandas_read_csv_kwargs={},
        )
    if file_format == "xlsx":
        df = pd.read_excel(BytesIO(content), engine="openpyxl")
        return CSVSessionRuntime._normalize_columns(df)  # noqa: SLF001
    raise ValueError(f"Unsupported tabular format: {file_format}")


def read_tabular_dataframe(
    content: bytes,
    *,
    file_format: TabularFileFormat,
    options: TabularPreprocessingOptions | None = None,
) -> PreprocessedTabularData:
    clean_options = options or TabularPreprocessingOptions()
    if not clean_options.enabled:
        df = _read_legacy_dataframe(content, file_format=file_format)
        return PreprocessedTabularData(dataframe=df, summary=_legacy_summary(df))

    if file_format == "csv":
        separator = detect_csv_separator(content) if clean_options.detect_csv_separator else ","
        read_kwargs: dict[str, object] = {"sep": separator}
        if clean_options.detect_header_row:
            read_kwargs.update({"header": None, "dtype": object})
        df = CSVSessionRuntime._read_csv_resilient(  # noqa: SLF001
            content,
            pandas_read_csv_kwargs=read_kwargs,
        )
        return clean_dataframe_for_eda(
            df,
            detect_header=clean_options.detect_header_row,
            options=clean_options,
        )

    if file_format == "xlsx":
        if clean_options.detect_header_row:
            raw_df = pd.read_excel(
                BytesIO(content),
                sheet_name=0,
                header=None,
                engine="openpyxl",
                dtype=object,
            )
            return clean_dataframe_for_eda(raw_df, detect_header=True, options=clean_options)

        df = pd.read_excel(BytesIO(content), engine="openpyxl")
        normalized = CSVSessionRuntime._normalize_columns(df)  # noqa: SLF001
        return clean_dataframe_for_eda(normalized, detect_header=False, options=clean_options)

    raise ValueError(f"Unsupported tabular format: {file_format}")
