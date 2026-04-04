"""Serialization bridge: ExecutionArtifact → API-ready dict.

Transforms execution artifacts through the presentation layer and produces
the JSON-serializable payload expected by the frontend.
"""
from __future__ import annotations

from typing import Any

from backend.artifacts.execution import (
    ExecArtifactType,
    ExecutionArtifact,
)
from backend.artifacts.presentation import to_presentation

_EXEC_TO_API_TYPE: dict[ExecArtifactType, str] = {
    ExecArtifactType.DATAFRAME: "table",
    ExecArtifactType.SQL_RESULT: "table",
    ExecArtifactType.SEARCH_RESULT: "table",
    ExecArtifactType.PLOT: "plot",
    ExecArtifactType.SCALAR: "value",
    ExecArtifactType.FORECAST: "table",
}


def execution_to_api_payload(exec_artifact: ExecutionArtifact) -> dict[str, Any]:
    """Convert ExecutionArtifact → PresentationArtifact → API-ready dict.

    Output shape matches the frontend ``ArtifactPayload`` contract.
    """
    pa = to_presentation(exec_artifact)
    return {
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
