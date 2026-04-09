"""Presentation artifacts — user-facing objects for UI rendering and reports.

A presentation artifact is always derived from one or more execution artifacts.
It carries only the data needed for display, plus a reference back to its source
execution artifact(s) for traceability.

The transform layer converts ExecutionArtifact → PresentationArtifact,
applying formatting, theme, truncation, and serialization rules that are
purely presentation concerns.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from backend.artifacts.execution import (
    ExecArtifactType,
    ExecutionArtifact,
)


class PresentationType(StrEnum):
    TABLE = "table"
    CHART = "chart"
    VALUE = "value"
    TEXT = "text"
    JSON = "json"


@dataclass
class PresentationArtifact:
    """A UI-ready artifact derived from one or more execution artifacts.

    This is what gets serialized to the API and rendered in the frontend.
    It never contains raw DataFrames or Figures — only serialized representations.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    presentation_type: PresentationType = PresentationType.TEXT
    title: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    source_execution_ids: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )


# ── Transform: ExecutionArtifact → PresentationArtifact ────────────────


_TYPE_MAP: dict[ExecArtifactType, PresentationType] = {
    ExecArtifactType.DATAFRAME: PresentationType.TABLE,
    ExecArtifactType.SQL_RESULT: PresentationType.TABLE,
    ExecArtifactType.SEARCH_RESULT: PresentationType.TABLE,
    ExecArtifactType.PLOT: PresentationType.CHART,
    ExecArtifactType.SCALAR: PresentationType.VALUE,
    ExecArtifactType.FORECAST: PresentationType.TABLE,
    ExecArtifactType.JSON: PresentationType.JSON,
}


def to_presentation(exec_artifact: ExecutionArtifact) -> PresentationArtifact:
    """Transform a single execution artifact into a presentation artifact.

    Serialization of raw data (DataFrame → split dict, Figure → plotly JSON)
    happens here — the presentation layer owns all formatting decisions.
    """
    ptype = _TYPE_MAP.get(exec_artifact.artifact_type, PresentationType.TEXT)

    if ptype == PresentationType.TABLE:
        data = _serialize_table_data(exec_artifact)
    elif ptype == PresentationType.CHART:
        data = _serialize_chart_data(exec_artifact)
    elif ptype == PresentationType.VALUE:
        data = _serialize_value_data(exec_artifact)
    elif ptype == PresentationType.JSON:
        data = _serialize_json_data(exec_artifact)
    else:
        data = {"format": "text", "data": str(exec_artifact.data or "")}

    return PresentationArtifact(
        presentation_type=ptype,
        title=exec_artifact.name or exec_artifact.producer_tool,
        data=data,
        source_execution_ids=[exec_artifact.id],
        meta={
            "producer_tool": exec_artifact.producer_tool,
            "lineage": exec_artifact.parent_ids,
            **(exec_artifact.meta or {}),
        },
    )


# ── Serializers (presentation concerns only) ───────────────────────────


def _serialize_table_data(artifact: ExecutionArtifact) -> dict[str, Any]:
    import pandas as pd
    df = artifact.data
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if not isinstance(df, pd.DataFrame):
        return {"format": "text", "data": str(df)}
    try:
        return {"format": "split", "data": df.to_dict(orient="split")}
    except Exception:
        return {"format": "text", "data": str(df)}


def _serialize_chart_data(artifact: ExecutionArtifact) -> dict[str, Any]:
    fig = artifact.data
    if fig is None:
        return {"format": "plotly-json", "data": {}}
    if isinstance(fig, dict):
        return {"format": "plotly-json", "data": fig}
    if hasattr(fig, "to_plotly_json"):
        return {"format": "plotly-json", "data": fig.to_plotly_json()}
    if hasattr(fig, "to_json"):
        import json
        return {"format": "plotly-json", "data": json.loads(fig.to_json())}
    return {"format": "text", "data": str(fig)}


def _serialize_value_data(artifact: ExecutionArtifact) -> dict[str, Any]:
    data = artifact.data
    if isinstance(data, dict):
        return {"format": "value", "data": data}
    return {"format": "value", "data": {"value": data}}


def _serialize_json_data(artifact: ExecutionArtifact) -> dict[str, Any]:
    import json as _json
    data = artifact.data
    if isinstance(data, dict):
        try:
            _json.dumps(data, default=str)  # validate serialisability
            return {"format": "json", "data": data}
        except Exception:
            pass
    return {"format": "json", "data": {"value": str(data)}}
