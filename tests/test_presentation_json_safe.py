from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from backend.artifacts.bridge import execution_from_api_payload, execution_to_api_payload
from backend.artifacts.execution import ExecArtifactType, ExecutionArtifact
from backend.artifacts.presentation import _serialize_table_data
from backend.auth.blob_store import StoredBlob
from backend.core.json_utils import NumpyEncoder
from backend.sessions.session_store import SessionStore


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


def test_complete_table_artifact_round_trips_for_execution() -> None:
    frame = pd.DataFrame(
        {
            "month": pd.to_datetime(["2025-01-01", "2025-02-01"]),
            "value": pd.Series([10, 20], dtype="Int64"),
        }
    )
    artifact = ExecutionArtifact(
        artifact_type=ExecArtifactType.DATAFRAME,
        name="monthly_values",
        data=frame,
        producer_tool="sql_tool",
    )

    payload = execution_to_api_payload(artifact)
    restored = execution_from_api_payload(payload, session_id="session-1")

    assert payload["execution"]["data_complete"] is True
    assert restored.id == artifact.id
    assert restored.session_id == "session-1"
    assert restored.data.equals(frame)


def test_legacy_inline_execution_contract_remains_readable() -> None:
    artifact = ExecutionArtifact(
        name="legacy",
        data=pd.DataFrame({"value": [1]}),
    )
    payload = execution_to_api_payload(artifact)
    payload["execution"]["schema_version"] = "1.0"
    payload["execution"].pop("storage")

    restored = execution_from_api_payload(payload, session_id="session-1")

    assert restored.data.equals(artifact.data)


def test_large_execution_artifact_uses_blob_and_keeps_bounded_json_preview(tmp_path) -> None:
    class BlobStore:
        def __init__(self) -> None:
            self.content: dict[str, bytes] = {}

        def put_many(self, *, items, **_kwargs) -> list[str]:
            ids = [f"blob-{index}" for index, _item in enumerate(items)]
            self.content.update((blob_id, item.content) for blob_id, item in zip(ids, items, strict=True))
            return ids

        def get_for_session(self, *, blob_id: str, **_kwargs) -> StoredBlob | None:
            content = self.content.get(blob_id)
            if content is None:
                return None
            return StoredBlob(
                blob_id=blob_id,
                logical_name="artifact.parquet",
                media_type="application/vnd.apache.parquet",
                content=content,
            )

    blob_store = BlobStore()
    store = SessionStore(
        str(tmp_path),
        ttl_days=7,
        artifact_blob_store=blob_store,
    )
    session = store.create_session("session-1")
    frame = pd.DataFrame(
        {
            "row": range(5_000),
            "description": ["large analytical result " * 8] * 5_000,
        }
    )
    artifact = ExecutionArtifact(id="large", name="large", data=frame)
    full_payload = execution_to_api_payload(artifact)

    store.add_chat_message(session.session_id, "ai", "result", artifacts=[full_payload])
    store.add_artifacts(session.session_id, [artifact], user_id=1)

    state = store.load_session(session.session_id)
    assert state is not None
    persisted = state.artifacts[0]
    assert persisted["execution"]["storage"]["kind"] == "blob"
    assert len(persisted["data"]["data"]["data"]) == 100
    assert len(state.chat_history[-1]["artifacts"][0]["data"]["data"]["data"]) == 100
    hydrated = store.get_serialized_artifact(session.session_id, artifact.id)
    assert hydrated is not None
    restored = execution_from_api_payload(hydrated, session_id=session.session_id)
    assert restored.data.equals(frame)


def test_incomplete_table_artifact_cannot_be_restored_for_execution() -> None:
    frame = pd.DataFrame({"value": [1, 2]})
    frame.attrs["llm_data_analyst.query"] = {"truncated": True}
    artifact = ExecutionArtifact(
        artifact_type=ExecArtifactType.DATAFRAME,
        name="preview_only",
        data=frame,
        producer_tool="sql_tool",
    )

    payload = execution_to_api_payload(artifact)

    assert payload["execution"]["data_complete"] is False
    with pytest.raises(ValueError, match="incomplete or preview"):
        execution_from_api_payload(payload, session_id="session-1")


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
