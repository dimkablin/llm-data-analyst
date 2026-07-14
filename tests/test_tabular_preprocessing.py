from __future__ import annotations

from io import BytesIO

import pandas as pd

from backend.data_access.tabular_preprocessing import (
    TabularPreprocessingOptions,
    read_tabular_dataframe,
)


def _workbook_bytes(rows: list[list[object]]) -> bytes:
    buffer = BytesIO()
    pd.DataFrame(rows).to_excel(buffer, index=False, header=False)
    return buffer.getvalue()


def test_xlsx_preprocessing_detects_header_and_removes_sparse_metadata_rows() -> None:
    data = _workbook_bytes(
        [
            ["Sales report", None, None],
            [None, None, None],
            ["customer_id", "segment", "amount"],
            [10, "enterprise", 120],
            [None, "technical note", None],
            [20, "retail", 80],
        ]
    )

    result = read_tabular_dataframe(
        data,
        file_format="xlsx",
        options=TabularPreprocessingOptions(),
    )

    assert result.summary.detected_header_row == 3
    assert result.summary.raw_rows == 6
    assert result.summary.cleaned_rows == 2
    assert list(result.dataframe.columns) == ["customer_id", "segment", "amount"]
    assert result.dataframe.to_dict(orient="records") == [
        {"customer_id": 10, "segment": "enterprise", "amount": 120},
        {"customer_id": 20, "segment": "retail", "amount": 80},
    ]


def test_sparse_row_filter_can_be_disabled_for_uploaded_xlsx() -> None:
    data = _workbook_bytes(
        [
            ["Report", None, None],
            ["customer_id", "segment", "amount"],
            [10, "enterprise", 120],
            [None, "technical note", None],
        ]
    )

    result = read_tabular_dataframe(
        data,
        file_format="xlsx",
        options=TabularPreprocessingOptions(drop_sparse_rows=False),
    )

    assert list(result.dataframe.columns) == ["customer_id", "segment", "amount"]
    assert result.dataframe.loc[0].to_dict() == {
        "customer_id": 10,
        "segment": "enterprise",
        "amount": 120,
    }
    assert pd.isna(result.dataframe.loc[1, "customer_id"])
    assert result.dataframe.loc[1, "segment"] == "technical note"
    assert pd.isna(result.dataframe.loc[1, "amount"])


def test_csv_preprocessing_drops_empty_rows_and_columns() -> None:
    result = read_tabular_dataframe(
        b"name,amount,empty\nA,10,\n,,\nB,20,\n",
        file_format="csv",
        options=TabularPreprocessingOptions(),
    )

    assert list(result.dataframe.columns) == ["name", "amount"]
    assert result.dataframe.to_dict(orient="records") == [
        {"name": "A", "amount": 10},
        {"name": "B", "amount": 20},
    ]


def test_csv_preprocessing_detects_semicolon_separator() -> None:
    result = read_tabular_dataframe(
        "name;amount\nA;10\nB;20\n".encode("utf-8"),
        file_format="csv",
        options=TabularPreprocessingOptions(),
    )

    assert list(result.dataframe.columns) == ["name", "amount"]
    assert result.dataframe.to_dict(orient="records") == [
        {"name": "A", "amount": 10},
        {"name": "B", "amount": 20},
    ]


def test_csv_preprocessing_detects_header_after_metadata_rows() -> None:
    result = read_tabular_dataframe(
        "Sales report,,\ncustomer_id,segment,amount\n10,enterprise,120\n20,retail,80\n".encode("utf-8"),
        file_format="csv",
        options=TabularPreprocessingOptions(),
    )

    assert result.summary.detected_header_row == 2
    assert list(result.dataframe.columns) == ["customer_id", "segment", "amount"]
    assert result.dataframe.to_dict(orient="records") == [
        {"customer_id": 10, "segment": "enterprise", "amount": 120},
        {"customer_id": 20, "segment": "retail", "amount": 80},
    ]


def test_csv_preprocessing_detects_tab_separator() -> None:
    result = read_tabular_dataframe(
        "name\tamount\nA\t10\nB\t20\n".encode("utf-8"),
        file_format="csv",
        options=TabularPreprocessingOptions(),
    )

    assert list(result.dataframe.columns) == ["name", "amount"]
    assert result.dataframe.to_dict(orient="records") == [
        {"name": "A", "amount": 10},
        {"name": "B", "amount": 20},
    ]
