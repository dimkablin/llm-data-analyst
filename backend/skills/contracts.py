from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

SkillArtifactType = Literal["table", "plot", "value", "json", "note"]

_ARTIFACT_TYPE_ALIASES: dict[str, SkillArtifactType] = {
    "table": "table",
    "dataframe": "table",
    "df": "table",
    "sql_result": "table",
    "plot": "plot",
    "chart": "plot",
    "graph": "plot",
    "value": "value",
    "metric": "value",
    "scalar": "value",
    "json": "json",
    "note": "note",
    "markdown": "note",
}


class SkillArtifactRequirement(BaseModel):
    """Nonblocking metadata describing an artifact a domain extension may produce."""

    model_config = ConfigDict(extra="forbid")

    artifact_type: SkillArtifactType
    name: str | None = None

    @field_validator("artifact_type", mode="before")
    @classmethod
    def _normalize_artifact_type(cls, value: object) -> SkillArtifactType:
        key = str(value or "").strip().lower().replace("-", "_")
        if key not in _ARTIFACT_TYPE_ALIASES:
            raise ValueError(f"Unsupported skill artifact type: {value}")
        return _ARTIFACT_TYPE_ALIASES[key]

    @field_validator("name", mode="before")
    @classmethod
    def _normalize_name(cls, value: object) -> str | None:
        text = str(value or "").strip().strip("`")
        return text or None
