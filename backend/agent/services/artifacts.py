from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from backend.artifacts.execution import (
    ExecArtifactType,
    ExecutionArtifact,
    is_tabular_artifact_type,
)
from backend.tools.sandbox_manager import SandboxManager

logger = logging.getLogger(__name__)


class ArtifactRehydrationRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    artifacts: list[Any] = Field(default_factory=list)
    session_id: str = ""
    tool_names: list[str] = Field(default_factory=list)


def table_artifact_to_dataframe(table_data: Any) -> pd.DataFrame | None:
    if isinstance(table_data, pd.DataFrame):
        return table_data.copy()
    if isinstance(table_data, pd.Series):
        return table_data.to_frame()
    return None


def artifact_dataframe_row_count(artifact: object) -> int:
    frame = table_artifact_to_dataframe(getattr(artifact, "data", None))
    return len(frame) if frame is not None else 0


def dataframe_execution_artifacts(
    frames: list[tuple[str, pd.DataFrame]],
    *,
    producer_tool: str = "sql_tool",
) -> list[ExecutionArtifact]:
    artifacts: list[ExecutionArtifact] = []
    for name, frame in frames:
        if frame is None or frame.empty:
            continue
        artifact = ExecutionArtifact(
            artifact_type=ExecArtifactType.DATAFRAME,
            producer_tool=producer_tool,
            data=frame,
            name=str(name),
        )
        artifact.build_schema()
        artifacts.append(artifact)
    return artifacts


def iter_sandbox_tabular_frames(session_id: str) -> list[tuple[str, pd.DataFrame]]:
    """Load analyst tables from the session sandbox when tool artifacts lack DataFrames."""
    session_id = str(session_id or "").strip()
    if not session_id:
        return []
    sandbox = SandboxManager.get_instance().get(session_id)
    if sandbox is None:
        return []
    frames: list[tuple[str, pd.DataFrame]] = []
    for name, obj in sandbox.get_user_scope().items():
        if str(name).startswith("_"):
            continue
        if isinstance(obj, pd.Series):
            obj = obj.to_frame()
        if isinstance(obj, pd.DataFrame) and not obj.empty:
            frames.append((str(name), obj.copy()))
    return frames


def rehydrate_artifacts_from_sandbox(request: ArtifactRehydrationRequest) -> list[Any]:
    """Restore table artifacts from sandbox when callbacks dropped DataFrames."""
    session_id = str(request.session_id or "").strip()
    current = list(request.artifacts or [])
    tool_names = [str(name).strip() for name in request.tool_names if str(name).strip()]
    if not current and not tool_names:
        return current
    if not session_id:
        return current

    has_nonempty_tabular = any(
        is_tabular_artifact_type(getattr(artifact, "artifact_type", ""))
        and artifact_dataframe_row_count(artifact) > 0
        for artifact in current
    )
    frames = iter_sandbox_tabular_frames(session_id)
    if not frames:
        return current

    existing_names = {
        str(getattr(artifact, "name", "") or "").strip().lower()
        for artifact in current
        if is_tabular_artifact_type(getattr(artifact, "artifact_type", ""))
    }
    missing_frames = [
        (name, frame)
        for name, frame in frames
        if frame is not None
        and not frame.empty
        and str(name).strip().lower() not in existing_names
    ]
    if has_nonempty_tabular and not missing_frames:
        return current

    producer = "pandas_tool" if "pandas_tool" in tool_names and "sql_tool" not in tool_names else "sql_tool"
    rebuilt = dataframe_execution_artifacts(
        missing_frames if has_nonempty_tabular else frames,
        producer_tool=producer,
    )
    if not rebuilt:
        return current

    logger.info(
        "Rehydrated %d table artifact(s) from sandbox (session=%s)",
        len(rebuilt),
        session_id[:8],
    )
    return [*current, *rebuilt]
