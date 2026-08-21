"""Serialization bridge: ExecutionArtifact → API-ready dict.

Transforms execution artifacts through the presentation layer and produces
the JSON-serializable payload expected by the frontend.
"""

from __future__ import annotations

from typing import Any

from backend.artifacts.execution import (
    ExecArtifactSchema,
    ExecArtifactType,
    ExecutionArtifact,
    artifact_type_label,
    is_tabular_artifact_type,
)
from backend.artifacts.presentation import to_presentation
from backend.core.json_utils import make_json_safe

_EXEC_TO_API_TYPE: dict[ExecArtifactType, str] = {
    ExecArtifactType.DATAFRAME: "table",
    ExecArtifactType.SQL_RESULT: "table",
    ExecArtifactType.SEARCH_RESULT: "table",
    ExecArtifactType.PLOT: "plot",
    ExecArtifactType.SCALAR: "value",
    ExecArtifactType.FORECAST: "table",
    ExecArtifactType.JSON: "json",
}

EXECUTION_CONTRACT_VERSION = "2.0"
_LEGACY_EXECUTION_CONTRACT_VERSION = "1.0"


def execution_data_is_complete(exec_artifact: ExecutionArtifact) -> bool:
    if not is_tabular_artifact_type(exec_artifact.artifact_type):
        return False
    meta = exec_artifact.meta if isinstance(exec_artifact.meta, dict) else {}
    query = meta.get("query") if isinstance(meta.get("query"), dict) else {}
    upstream = (
        meta.get("upstream_completeness") if isinstance(meta.get("upstream_completeness"), dict) else {}
    )
    dataframe_meta: dict[str, Any] = {}
    attrs = getattr(exec_artifact.data, "attrs", None)
    if isinstance(attrs, dict):
        raw = attrs.get("llm_data_analyst.query")
        if isinstance(raw, dict):
            dataframe_meta = raw
    return not any(
        source.get("truncated") is True or source.get("has_more_rows") is True
        for source in (query, upstream, dataframe_meta)
    )


def _execution_contract(exec_artifact: ExecutionArtifact) -> dict[str, Any] | None:
    if not is_tabular_artifact_type(exec_artifact.artifact_type):
        return None
    schema = exec_artifact.schema or exec_artifact.build_schema()
    if schema is None:
        return None
    return {
        "schema_version": EXECUTION_CONTRACT_VERSION,
        "artifact_type": artifact_type_label(exec_artifact.artifact_type),
        "data_format": "dataframe-split",
        "data_complete": execution_data_is_complete(exec_artifact),
        "storage": {"kind": "inline"},
        "schema": {
            "columns": list(schema.columns),
            "dtypes": dict(schema.dtypes),
            "row_count": int(schema.row_count),
        },
    }


def execution_to_api_payload(exec_artifact: ExecutionArtifact) -> dict[str, Any]:
    """Convert ExecutionArtifact → PresentationArtifact → API-ready dict.

    Output shape matches the frontend ``ArtifactPayload`` contract.
    """
    pa = to_presentation(exec_artifact)
    payload = {
        "id": exec_artifact.id,
        "type": _EXEC_TO_API_TYPE.get(exec_artifact.artifact_type, "table"),
        "text": pa.title,
        "role": "ai",
        "meta": pa.meta,
        "timestamp": pa.created_at,
        "data": pa.data,
        "execution_artifact_id": exec_artifact.id,
        "content_hash": exec_artifact.content_hash,
        "version": exec_artifact.version,
        "parent_ids": exec_artifact.parent_ids,
    }
    execution = _execution_contract(exec_artifact)
    if execution is not None:
        payload["execution"] = execution
    return make_json_safe(payload)


def execution_from_api_payload(payload: dict[str, Any], *, session_id: str) -> ExecutionArtifact:
    """Restore a complete tabular artifact persisted in ``session_artifacts``."""
    import pandas as pd

    execution = payload.get("execution")
    if not isinstance(execution, dict) or execution.get("schema_version") not in {
        _LEGACY_EXECUTION_CONTRACT_VERSION,
        EXECUTION_CONTRACT_VERSION,
    }:
        raise ValueError("Artifact has no supported execution contract.")
    if execution.get("data_complete") is not True:
        raise ValueError("Artifact contains only incomplete or preview data.")
    if execution.get("data_format") != "dataframe-split":
        raise ValueError("Artifact data format is not supported.")
    storage = execution.get("storage")
    if isinstance(storage, dict) and storage.get("kind") == "blob":
        raise ValueError("Artifact blob data was not materialized.")

    outer_data = payload.get("data")
    split = outer_data.get("data") if isinstance(outer_data, dict) else None
    if not isinstance(split, dict):
        raise ValueError("Artifact table data is missing.")
    columns = split.get("columns")
    rows = split.get("data")
    index = split.get("index")
    if not isinstance(columns, list) or not isinstance(rows, list):
        raise ValueError("Artifact table data is malformed.")

    dataframe = pd.DataFrame(rows, columns=columns, index=index if isinstance(index, list) else None)
    schema_payload = execution.get("schema")
    if not isinstance(schema_payload, dict):
        raise ValueError("Artifact schema is missing.")
    expected_columns = [str(column) for column in schema_payload.get("columns") or []]
    if list(map(str, dataframe.columns)) != expected_columns:
        raise ValueError("Artifact columns do not match its schema.")
    expected_rows = int(schema_payload.get("row_count") or 0)
    if len(dataframe) != expected_rows:
        raise ValueError("Artifact row count does not match its schema.")

    for column, dtype in dict(schema_payload.get("dtypes") or {}).items():
        if column not in dataframe.columns:
            continue
        try:
            if str(dtype).startswith("datetime"):
                dataframe[column] = pd.to_datetime(dataframe[column])
            elif str(dtype).startswith("timedelta"):
                dataframe[column] = pd.to_timedelta(dataframe[column])
            elif str(dtype) != "object":
                dataframe[column] = dataframe[column].astype(str(dtype))
        except (TypeError, ValueError):
            # The persisted values are still usable; dtype restoration is best effort.
            continue

    meta = dict(payload.get("meta") or {})
    query = meta.get("query")
    if isinstance(query, dict):
        dataframe.attrs["llm_data_analyst.query"] = dict(query)
    artifact_type = ExecArtifactType(str(execution.get("artifact_type") or "dataframe"))
    return ExecutionArtifact(
        id=str(payload.get("execution_artifact_id") or payload.get("id") or ""),
        session_id=session_id,
        artifact_type=artifact_type,
        producer_tool=str(meta.get("producer_tool") or ""),
        data=dataframe,
        name=str(payload.get("text") or ""),
        parent_ids=[str(item) for item in payload.get("parent_ids") or []],
        schema=ExecArtifactSchema(
            columns=expected_columns,
            dtypes={str(key): str(value) for key, value in dict(schema_payload.get("dtypes") or {}).items()},
            row_count=expected_rows,
        ),
        content_hash=str(payload.get("content_hash") or ""),
        meta=meta,
        created_at=str(payload.get("timestamp") or ""),
        version=int(payload.get("version") or 1),
        reusable=True,
    )
