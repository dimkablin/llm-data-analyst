from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.skills.contracts import SkillArtifactRequirement, SkillArtifactType


class DomainArtifactReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_type: SkillArtifactType
    name: str = Field(min_length=1)

    @field_validator("name", mode="before")
    @classmethod
    def _normalize_name(cls, value: object) -> str:
        return str(value or "").strip()


class DomainToolPermission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required_tool_keys: tuple[str, ...]
    required_skill_ids: tuple[str, ...]

    @field_validator("required_tool_keys", "required_skill_ids", mode="before")
    @classmethod
    def _normalize_tuple(cls, value: object) -> tuple[str, ...]:
        return _dedupe_clean_strings(value)


class DomainMCPToolContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_key: str = Field(min_length=1)
    mcp_tool_name: str = Field(min_length=1)
    input_schema_ref: str = Field(min_length=1)
    output_schema_ref: str = Field(min_length=1)
    required_tool_keys: tuple[str, ...]
    produced_artifacts: tuple[SkillArtifactRequirement, ...]

    @field_validator(
        "tool_key",
        "mcp_tool_name",
        "input_schema_ref",
        "output_schema_ref",
        mode="before",
    )
    @classmethod
    def _normalize_name(cls, value: object) -> str:
        return str(value or "").strip()

    @field_validator("required_tool_keys", mode="before")
    @classmethod
    def _normalize_required_tool_keys(cls, value: object) -> tuple[str, ...]:
        return _dedupe_clean_strings(value)


class DomainExtensionManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    extension_id: str = Field(min_length=1)
    skill_id: str = Field(min_length=1)
    capability: str = Field(min_length=1)
    mcp_server_name: str = Field(min_length=1)
    permission: DomainToolPermission
    tools: tuple[DomainMCPToolContract, ...]

    @field_validator("extension_id", "skill_id", "capability", "mcp_server_name", mode="before")
    @classmethod
    def _normalize_name(cls, value: object) -> str:
        return str(value or "").strip()


class DomainAnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_markdown: str = Field(min_length=1)
    artifacts: tuple[DomainArtifactReference, ...] = Field(default_factory=tuple)
    warnings: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("answer_markdown", mode="before")
    @classmethod
    def _normalize_answer(cls, value: object) -> str:
        return str(value or "").strip()

    @field_validator("warnings", mode="before")
    @classmethod
    def _normalize_warnings(cls, value: object) -> tuple[str, ...]:
        return _dedupe_clean_strings(value)


def _dedupe_clean_strings(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw_values = [value]
    else:
        try:
            raw_values = list(value)  # type: ignore[arg-type]
        except TypeError:
            raw_values = [value]

    result: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return tuple(result)
