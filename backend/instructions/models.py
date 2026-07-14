from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class InstructionKind(StrEnum):
    ANALYTICAL = "analytical"
    TOOL = "tool"


def normalize_instruction_id(value: str) -> str:
    return str(value or "").strip().lower()


def normalize_triggers(raw: object) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        items = [item.strip().lower() for item in raw.split(",")]
    elif isinstance(raw, list | tuple):
        items = [str(item).strip().lower() for item in raw]
    else:
        raise TypeError("triggers must be a comma-separated string or a list of strings")
    return tuple(dict.fromkeys(item for item in items if item))


class InstructionMetadata(BaseModel):
    """Typed YAML frontmatter shared by SKILL.md and TOOL.md files."""

    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    description: str
    enabled_by_default: bool = True
    triggers: tuple[str, ...] = Field(default_factory=tuple)
    kind: InstructionKind = InstructionKind.ANALYTICAL
    tool_key: str | None = None

    @field_validator("id", mode="before")
    @classmethod
    def normalize_id(cls, value: object) -> str:
        return normalize_instruction_id(str(value or ""))

    @field_validator("name", "description", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> str:
        return str(value or "").strip()

    @field_validator("triggers", mode="before")
    @classmethod
    def normalize_trigger_values(cls, value: object) -> tuple[str, ...]:
        return normalize_triggers(value)

    @field_validator("kind", mode="before")
    @classmethod
    def normalize_kind(cls, value: object) -> InstructionKind:
        raw = str(value or InstructionKind.ANALYTICAL.value).strip().lower()
        if raw == "skill":
            raw = InstructionKind.ANALYTICAL.value
        return InstructionKind(raw)

    @field_validator("tool_key", mode="before")
    @classmethod
    def normalize_tool_key(cls, value: object) -> str | None:
        if value is None:
            return None
        clean = str(value or "").strip()
        return clean or None

    @model_validator(mode="after")
    def validate_required_fields(self) -> InstructionMetadata:
        if not self.id:
            raise ValueError("id is required")
        if not self.name:
            raise ValueError("name is required")
        if not self.description:
            raise ValueError("description is required")
        if self.kind == InstructionKind.TOOL and not self.tool_key:
            raise ValueError("tool_key is required when kind='tool'")
        return self

    @property
    def extras(self) -> dict[str, Any]:
        return dict(self.model_extra or {})


class InstructionDocument(BaseModel):
    """Parsed markdown instruction document with typed metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    metadata: InstructionMetadata
    body: str
    source_path: Path
    details_markdown: str | None = None

    @property
    def instruction_id(self) -> str:
        return self.metadata.id

    @property
    def source_name(self) -> str:
        return self.source_path.name

    @property
    def has_details(self) -> bool:
        return self.details_markdown is not None
