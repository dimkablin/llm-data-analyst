from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import plotly.io as pio

from backend.internal_models import ArtifactRecord


TABLE_FORMAT = "split"
PLOT_FORMAT = "plotly-json"


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def serialize_table(df: pd.DataFrame) -> dict[str, Any]:
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return {
        "format": TABLE_FORMAT,
        "data": _to_jsonable(df.to_dict(orient=TABLE_FORMAT)),
    }


def serialize_plot(fig) -> dict[str, Any]:
    return {
        "format": PLOT_FORMAT,
        "data": json.loads(pio.to_json(fig)),
    }


def serialize_artifact(artifact: ArtifactRecord) -> dict[str, Any]:
    data = artifact.data
    if artifact.artifact_type == "table":
        data = serialize_table(data)
    elif artifact.artifact_type == "plot":
        data = serialize_plot(data)
    elif artifact.artifact_type == "value":
        data = {"format": "value", "data": _to_jsonable(data)}
    else:
        data = {"format": "text", "data": _to_jsonable(data)}

    return {
        "id": artifact.artifact_id,
        "type": artifact.artifact_type,
        "text": artifact.text,
        "role": artifact.role,
        "meta": _to_jsonable(artifact.meta),
        "timestamp": artifact.timestamp,
        "data": data,
    }
