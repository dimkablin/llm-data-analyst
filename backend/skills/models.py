from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class SkillError(Exception):
    """Base error for skill loading and selection."""


class SkillValidationError(SkillError):
    """Raised when a markdown skill file is malformed or unsafe to load."""


class SkillSelectionError(SkillError):
    """Raised when explicit runtime skill selection is invalid."""


@dataclass(frozen=True)
class SkillExample:
    language: str
    code: str


@dataclass(frozen=True)
class Skill:
    skill_id: str
    name: str
    description: str
    core_markdown: str
    details_markdown: str | None
    source_path: str
    triggers: tuple[str, ...] = ()
    python_examples: tuple[SkillExample, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    kind: str = "analytical"
    tool_key: str | None = None
    enabled_by_default: bool = True

    @property
    def instructions_markdown(self) -> str:
        """Backward-compat alias → core_markdown."""
        return self.core_markdown

    @property
    def has_details(self) -> bool:
        return self.details_markdown is not None

    @property
    def source_name(self) -> str:
        return Path(self.source_path).name


@dataclass(frozen=True)
class SkillSummary:
    skill_id: str
    name: str
    description: str
    triggers: tuple[str, ...]
    source_path: str
    enabled_by_default: bool = True


@dataclass(frozen=True)
class SkillSelectionContext:
    query: str | None = None
    dataset_columns: tuple[str, ...] = ()
    source_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SkillFilter(Protocol):
    def filter(
        self,
        skills: tuple[Skill, ...],
        context: SkillSelectionContext,
    ) -> tuple[Skill, ...]: ...


class SkillMatcher(Protocol):
    def match(
        self,
        skills: tuple[Skill, ...],
        context: SkillSelectionContext,
    ) -> tuple[SkillSummary, ...]: ...


class SkillRanker(Protocol):
    def rank(
        self,
        skills: tuple[SkillSummary, ...],
        context: SkillSelectionContext,
    ) -> tuple[SkillSummary, ...]: ...


class SkillSelector(Protocol):
    def select(
        self,
        skills: tuple[Skill, ...],
        context: SkillSelectionContext,
    ) -> tuple[SkillSummary, ...]: ...
