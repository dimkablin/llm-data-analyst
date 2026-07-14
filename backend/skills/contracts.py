from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SkillArtifactType = Literal["table", "plot", "value", "json", "note"]

_SECTION_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_BULLET_RE = re.compile(r"^\s*[-*]\s+(.*\S)\s*$")

_REQUIRED_TOOLS_SECTION = "required tools"
_REQUIRED_ARTIFACTS_SECTION = "required artifacts"
_EVIDENCE_RULES_SECTION = "evidence rules"

_SECTION_ALIASES: dict[str, str] = {
    _REQUIRED_TOOLS_SECTION: _REQUIRED_TOOLS_SECTION,
    "required tool": _REQUIRED_TOOLS_SECTION,
    "tools": _REQUIRED_TOOLS_SECTION,
    _REQUIRED_ARTIFACTS_SECTION: _REQUIRED_ARTIFACTS_SECTION,
    "required artifact": _REQUIRED_ARTIFACTS_SECTION,
    "artifacts": _REQUIRED_ARTIFACTS_SECTION,
    _EVIDENCE_RULES_SECTION: _EVIDENCE_RULES_SECTION,
    "evidence rule": _EVIDENCE_RULES_SECTION,
    "evidence": _EVIDENCE_RULES_SECTION,
}

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
    model_config = ConfigDict(extra="forbid")

    artifact_type: SkillArtifactType
    name: str | None = None

    @field_validator("artifact_type", mode="before")
    @classmethod
    def _normalize_artifact_type(cls, value: object) -> SkillArtifactType:
        key = str(value or "").strip().lower().replace("-", "_")
        if key not in _ARTIFACT_TYPE_ALIASES:
            msg = f"Unsupported artifact type in skill execution contract: {value}"
            raise ValueError(msg)
        return _ARTIFACT_TYPE_ALIASES[key]

    @field_validator("name", mode="before")
    @classmethod
    def _normalize_name(cls, value: object) -> str | None:
        text = str(value or "").strip().strip("`")
        return text or None


class SkillExecutionContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required_tools: tuple[str, ...] = Field(default_factory=tuple)
    required_artifacts: tuple[SkillArtifactRequirement, ...] = Field(default_factory=tuple)
    evidence_rules: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("required_tools", mode="before")
    @classmethod
    def _normalize_tools(cls, value: object) -> tuple[str, ...]:
        return _dedupe_clean_strings(value)

    @field_validator("evidence_rules", mode="before")
    @classmethod
    def _normalize_rules(cls, value: object) -> tuple[str, ...]:
        return _dedupe_clean_strings(value)

    @property
    def is_empty(self) -> bool:
        return not self.required_tools and not self.required_artifacts and not self.evidence_rules


class SkillExecutionRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill_id: str
    execution_contract: SkillExecutionContract

    @field_validator("skill_id", mode="before")
    @classmethod
    def _normalize_skill_id(cls, value: object) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("Skill execution requirement must include a skill_id.")
        return text


class SkillPermissionValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    reason: str = ""
    missing_tool_keys: tuple[str, ...] = Field(default_factory=tuple)


def parse_skill_execution_contract(markdown: str) -> SkillExecutionContract:
    """Parse optional execution-contract sections from a skill markdown body.

    The parser is intentionally small and declarative: skills remain markdown files,
    while runtime code receives typed requirements it can validate generically.
    """
    sections = _extract_sections(markdown)
    required_tools = tuple(
        _parse_tool_item(item)
        for item in _bullet_items(sections, _REQUIRED_TOOLS_SECTION)
    )
    required_artifacts = tuple(
        _parse_artifact_item(item)
        for item in _bullet_items(sections, _REQUIRED_ARTIFACTS_SECTION)
    )
    evidence_rules = tuple(_bullet_items(sections, _EVIDENCE_RULES_SECTION))
    return SkillExecutionContract(
        required_tools=required_tools,
        required_artifacts=required_artifacts,
        evidence_rules=evidence_rules,
    )


def normalize_skill_execution_requirements(
    requirements: object,
) -> tuple[SkillExecutionRequirement, ...]:
    if requirements is None:
        return ()
    try:
        raw_items = list(requirements)  # type: ignore[arg-type]
    except TypeError:
        raw_items = [requirements]

    normalized: list[SkillExecutionRequirement] = []
    for item in raw_items:
        if isinstance(item, SkillExecutionRequirement):
            requirement = item
        elif isinstance(item, dict):
            payload = dict(item)
            if "execution_contract" not in payload and "contract" in payload:
                payload["execution_contract"] = payload["contract"]
            requirement = SkillExecutionRequirement.model_validate(payload)
        else:
            contract = getattr(item, "execution_contract", None)
            if contract is None:
                contract = getattr(item, "contract", None)
            requirement = SkillExecutionRequirement(
                skill_id=getattr(item, "skill_id", ""),
                execution_contract=contract,
            )
        if not requirement.execution_contract.is_empty:
            normalized.append(requirement)
    return tuple(normalized)


def validate_skill_tool_permissions(
    requirements: Iterable[SkillExecutionRequirement],
    allowed_tool_keys: Iterable[str] | None,
) -> SkillPermissionValidationResult:
    if allowed_tool_keys is None:
        return SkillPermissionValidationResult(passed=True)

    allowed = {str(tool_key).strip() for tool_key in allowed_tool_keys if str(tool_key).strip()}
    missing: list[str] = []
    seen: set[str] = set()
    missing_by_skill: list[str] = []

    for requirement in requirements:
        blocked = [
            tool_key
            for tool_key in requirement.execution_contract.required_tools
            if tool_key not in allowed
        ]
        if not blocked:
            continue
        missing_by_skill.append(f"{requirement.skill_id}: {', '.join(blocked)}")
        for tool_key in blocked:
            if tool_key in seen:
                continue
            seen.add(tool_key)
            missing.append(tool_key)

    if not missing:
        return SkillPermissionValidationResult(passed=True)
    return SkillPermissionValidationResult(
        passed=False,
        reason="blocked required tools: " + "; ".join(missing_by_skill),
        missing_tool_keys=tuple(missing),
    )


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
        text = str(raw or "").strip().strip("`")
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return tuple(result)


def _normalize_heading(raw_heading: str) -> str:
    heading = re.sub(r"\s+", " ", str(raw_heading or "").strip().lower())
    heading = heading.strip(":")
    return _SECTION_ALIASES.get(heading, heading)


def _extract_sections(markdown: str) -> dict[str, str]:
    text = str(markdown or "")
    matches = list(_SECTION_HEADING_RE.finditer(text))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        name = _normalize_heading(match.group(2))
        if name not in {
            _REQUIRED_TOOLS_SECTION,
            _REQUIRED_ARTIFACTS_SECTION,
            _EVIDENCE_RULES_SECTION,
        }:
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[name] = text[start:end].strip()
    return sections


def _bullet_items(sections: dict[str, str], section_name: str) -> list[str]:
    section = sections.get(section_name, "")
    items: list[str] = []
    for line in section.splitlines():
        match = _BULLET_RE.match(line)
        if match is None:
            continue
        item = match.group(1).strip()
        if item:
            items.append(item)
    return items


def _parse_tool_item(item: str) -> str:
    text = item.strip()
    tick_match = re.search(r"`([^`]+)`", text)
    if tick_match:
        return tick_match.group(1).strip()
    return text.split(":", 1)[0].strip()


def _parse_artifact_item(item: str) -> SkillArtifactRequirement:
    text = item.strip()
    if ":" in text:
        raw_type, raw_name = text.split(":", 1)
    else:
        parts = text.split(maxsplit=1)
        raw_type = parts[0] if parts else ""
        raw_name = parts[1] if len(parts) > 1 else ""
    return SkillArtifactRequirement(artifact_type=raw_type, name=raw_name)
