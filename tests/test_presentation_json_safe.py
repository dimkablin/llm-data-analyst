from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from backend.artifacts.bridge import execution_to_api_payload
from backend.artifacts.execution import ExecutionArtifact, ExecArtifactType
from backend.artifacts.presentation import _serialize_table_data
from backend.core.json_utils import NumpyEncoder


def test_serialize_table_data_converts_numpy_values() -> None:
    df = pd.DataFrame(
        {
            "return_30d": [np.float64(1.5), np.float64(-2.0)],
            "ticker": ["NREH", "NREH27"],
        }
    )
    artifact = ExecutionArtifact(
        artifact_type=ExecArtifactType.DATAFRAME,
        name="sector_snapshot",
        data=df,
        producer_tool="sql_tool",
    )
    payload = _serialize_table_data(artifact)
    encoded = json.dumps(payload, cls=NumpyEncoder)
    assert "NREH" in encoded
    assert "numpy" not in encoded.lower()


def test_execution_to_api_payload_is_json_serializable() -> None:
    df = pd.DataFrame({"close": np.array([10.0, 11.5], dtype=np.float64)})
    artifact = ExecutionArtifact(
        artifact_type=ExecArtifactType.DATAFRAME,
        name="nreh_price_history",
        data=df,
        producer_tool="sql_tool",
    )
    api_payload = execution_to_api_payload(artifact)
    json.dumps(api_payload, cls=NumpyEncoder)


def test_numpy_dtype_is_json_serializable() -> None:
    encoded = json.dumps({"dtype": np.dtype("O")}, cls=NumpyEncoder)

    assert encoded == '{"dtype": "object"}'


def test_plot_payload_serializes_datetime64_ns_as_iso_strings() -> None:
    fig = go.Figure(
        go.Scatter(
            x=np.array(["2024-01-31", "2024-02-29"], dtype="datetime64[ns]"),
            y=[1, 2],
        )
    )
    artifact = ExecutionArtifact(
        artifact_type=ExecArtifactType.PLOT,
        name="monthly_trend",
        data=fig,
        producer_tool="plotly_tool",
    )

    api_payload = execution_to_api_payload(artifact)

    assert api_payload["data"]["data"]["data"][0]["x"] == [
        "2024-01-31T00:00:00",
        "2024-02-29T00:00:00",
    ]


def test_table_payload_includes_numeric_summary_rows() -> None:
    df = pd.DataFrame(
        {
            "merchant_category": ["p2p", "travel"],
            "operation_count": [204, 125],
            "bank_income": [25771, -55780],
        }
    )
    artifact = ExecutionArtifact(
        artifact_type=ExecArtifactType.DATAFRAME,
        name="merchant_metrics",
        data=df,
        producer_tool="sql_tool",
    )

    payload = _serialize_table_data(artifact)

    assert payload["summary_rows"] == [
        {"merchant_category": "__sum__", "operation_count": 329, "bank_income": -30009},
        {"merchant_category": "__mean__", "operation_count": 164.5, "bank_income": -15004.5},
    ]
    assert df["merchant_category"].tolist() == ["p2p", "travel"]
