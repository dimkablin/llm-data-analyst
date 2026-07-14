from __future__ import annotations

import re
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from backend.artifacts.execution import artifact_type_label, is_tabular_artifact_type
from backend.skills.contracts import SkillArtifactRequirement, SkillExecutionContract

RequiredOutputKind = Literal[
    "catalog",
    "table_schema",
    "table",
    "aggregation",
    "joined_table",
    "plot",
    "metric",
    "brief",
]


class RequiredOutput(BaseModel):
    kind: RequiredOutputKind
    reason: str
    depends_on: list[str] = Field(default_factory=list)


_CATALOG_REQUEST_TOKENS = ("найди таблиц", "список таблиц", "show tables", "duckdb")
_SCHEMA_REQUEST_TOKENS = ("схем", "колон", "структур", "определи таблиц")
_AGGREGATION_REQUEST_TOKENS = ("агрег", "сумм", "групп", "отклон", "итог", "total")
_JOIN_REQUEST_TOKENS = (
    "join",
    "джойн",
    "джоин",
    "сопостав",
    "соедин",
    "объедин",
    "связ",
)
_BRIEF_REQUEST_TOKENS = ("вывод", "инсайт", "справк", "объясн", "анализ")


class AnalysisTaskContractDetector:
    """Infer required outputs from a user prompt without owning the DTO schema."""

    def detect(self, prompt: str) -> AnalysisTaskContract:
        text = self._normalize_prompt(prompt)
        outputs: list[RequiredOutput] = []

        self._add_when(
            outputs,
            "catalog",
            "user requested table discovery",
            any(token in text for token in _CATALOG_REQUEST_TOKENS),
        )
        self._add_when(
            outputs,
            "table_schema",
            "user requested table/schema identification",
            any(token in text for token in _SCHEMA_REQUEST_TOKENS),
        )
        self._add_when(
            outputs,
            "aggregation",
            "user requested aggregation or variance metrics",
            any(token in text for token in _AGGREGATION_REQUEST_TOKENS),
        )
        self._add_when(
            outputs,
            "joined_table",
            "user requested multi-table matching/join",
            any(token in text for token in _JOIN_REQUEST_TOKENS),
        )
        self._add_when(
            outputs,
            "plot",
            "user explicitly requested a chart",
            _explicit_visual_requested(text),
        )
        self._add_when(
            outputs,
            "brief",
            "user requested analytical explanation",
            any(token in text for token in _BRIEF_REQUEST_TOKENS),
        )
        return AnalysisTaskContract(required_outputs=outputs)

    @staticmethod
    def _normalize_prompt(prompt: str) -> str:
        return re.sub(r"\s+", " ", str(prompt or "").lower()).strip()

    @staticmethod
    def _add_when(
        outputs: list[RequiredOutput],
        kind: RequiredOutputKind,
        reason: str,
        condition: bool,
    ) -> None:
        if condition and not any(item.kind == kind for item in outputs):
            outputs.append(RequiredOutput(kind=kind, reason=reason))


class AnalysisTaskContract(BaseModel):
    required_outputs: list[RequiredOutput] = Field(default_factory=list)

    @classmethod
    def from_prompt(cls, prompt: str) -> AnalysisTaskContract:
        detected = AnalysisTaskContractDetector().detect(prompt)
        return cls(required_outputs=detected.required_outputs)


class ContractValidationResult(BaseModel):
    passed: bool
    reason: str = ""
    missing_requirements: list[str] = Field(default_factory=list)


class SkillExecutionContractValidationResult(BaseModel):
    passed: bool
    reason: str = ""
    missing_tools: list[str] = Field(default_factory=list)
    missing_artifacts: list[str] = Field(default_factory=list)


class ArtifactLineage(BaseModel):
    model_config = ConfigDict(extra="allow")

    source_table_names: list[str] = Field(default_factory=list)
    source_tables: list[dict[str, Any]] = Field(default_factory=list)


class ArtifactContractMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    catalog_listing: bool = False
    schema_description: bool = False
    aggregation: bool = False
    direct_sql: bool = False
    lineage: ArtifactLineage = Field(default_factory=ArtifactLineage)
    table_selection: dict[str, Any] | None = None


def validate_task_contract(
    contract: AnalysisTaskContract,
    response: Any,
) -> ContractValidationResult:
    missing: list[str] = []
    artifacts = list(getattr(response, "artifacts", []) or [])
    final_text = str(getattr(response, "final_text", "") or "")

    for requirement in contract.required_outputs:
        if requirement.kind == "catalog" and not _has_catalog_artifact(artifacts):
            missing.append(requirement.kind)
        elif requirement.kind == "table_schema" and not _has_schema_artifact(artifacts):
            missing.append(requirement.kind)
        elif requirement.kind == "table" and not _has_tabular_artifact(artifacts):
            missing.append(requirement.kind)
        elif requirement.kind == "aggregation" and not _has_aggregation_artifact(artifacts):
            missing.append(requirement.kind)
        elif requirement.kind == "joined_table" and not _has_joined_table_artifact(artifacts):
            missing.append(requirement.kind)
        elif requirement.kind == "plot" and not _has_plot_artifact(artifacts):
            missing.append(requirement.kind)
        elif requirement.kind == "metric" and not _has_metric_artifact(artifacts):
            missing.append(requirement.kind)
        elif requirement.kind == "brief" and not _has_brief(final_text):
            missing.append(requirement.kind)

    if missing:
        return ContractValidationResult(
            passed=False,
            reason=f"missing requirements: {', '.join(missing)}",
            missing_requirements=missing,
        )
    return ContractValidationResult(passed=True)


def validate_skill_execution_contract(
    contract: SkillExecutionContract,
    response: Any,
) -> SkillExecutionContractValidationResult:
    """Validate a skill's declared execution contract against runtime output.

    This is intentionally domain-neutral: the runtime checks that model-directed
    analysis used required generic tools and produced required artifact shapes,
    but it does not calculate domain summaries itself.
    """
    if contract.is_empty:
        return SkillExecutionContractValidationResult(passed=True)

    used_tools = {
        str(tool_name).strip()
        for tool_name in (getattr(response, "tool_names", None) or [])
        if str(tool_name).strip()
    }
    missing_tools = [
        tool_name
        for tool_name in contract.required_tools
        if tool_name not in used_tools
    ]

    artifacts = list(getattr(response, "artifacts", []) or [])
    missing_artifacts = [
        _format_skill_artifact_requirement(requirement)
        for requirement in contract.required_artifacts
        if not _has_skill_artifact_requirement(artifacts, requirement)
    ]

    if missing_tools or missing_artifacts:
        parts: list[str] = []
        if missing_tools:
            parts.append(f"missing tools: {', '.join(missing_tools)}")
        if missing_artifacts:
            parts.append(f"missing artifacts: {', '.join(missing_artifacts)}")
        return SkillExecutionContractValidationResult(
            passed=False,
            reason="; ".join(parts),
            missing_tools=missing_tools,
            missing_artifacts=missing_artifacts,
        )

    return SkillExecutionContractValidationResult(passed=True)


def _explicit_visual_requested(text: str) -> bool:
    if any(token in text for token in ("без граф", "не строй граф", "без визуал")):
        return False
    return any(
        token in text
        for token in (
            "граф",
            "диаграм",
            "визуал",
            "plot",
            "chart",
            "bar",
            "line",
        )
    )


def _artifact_meta(artifact: Any) -> dict[str, Any]:
    meta = getattr(artifact, "meta", None)
    if isinstance(meta, dict):
        return meta
    data = getattr(artifact, "data", None)
    if isinstance(data, dict):
        maybe_meta = data.get("meta")
        if isinstance(maybe_meta, dict):
            return maybe_meta
    return {}


def _artifact_contract_meta(artifact: Any) -> ArtifactContractMetadata:
    try:
        return ArtifactContractMetadata.model_validate(_artifact_meta(artifact))
    except Exception:
        return ArtifactContractMetadata()


def _artifact_name(artifact: Any) -> str:
    return str(getattr(artifact, "name", "") or getattr(artifact, "text", "") or "").lower()


def _normalized_skill_artifact_type(artifact_type: Any) -> str:
    label = artifact_type_label(artifact_type)
    if label in {"dataframe", "sql_result"}:
        return "table"
    if label == "scalar":
        return "value"
    return label


def _format_skill_artifact_requirement(requirement: SkillArtifactRequirement) -> str:
    suffix = f":{requirement.name}" if requirement.name else ""
    return f"{requirement.artifact_type}{suffix}"


def _has_skill_artifact_requirement(
    artifacts: list[Any],
    requirement: SkillArtifactRequirement,
) -> bool:
    expected_type = requirement.artifact_type
    expected_name = str(requirement.name or "").strip().lower()
    for artifact in artifacts:
        current_type = _normalized_skill_artifact_type(getattr(artifact, "artifact_type", ""))
        if current_type != expected_type:
            continue
        if not expected_name:
            return True
        current_name = _artifact_name(artifact)
        if current_name == expected_name:
            return True
    return False


def _has_tabular_artifact(artifacts: list[Any]) -> bool:
    return any(is_tabular_artifact_type(getattr(artifact, "artifact_type", "")) for artifact in artifacts)


def _has_plot_artifact(artifacts: list[Any]) -> bool:
    return any(
        artifact_type_label(getattr(artifact, "artifact_type", "")) == "plot"
        for artifact in artifacts
    )


def _has_metric_artifact(artifacts: list[Any]) -> bool:
    return any(
        artifact_type_label(getattr(artifact, "artifact_type", "")) in {"scalar", "value"}
        for artifact in artifacts
    )


def _has_catalog_artifact(artifacts: list[Any]) -> bool:
    for artifact in artifacts:
        meta = _artifact_contract_meta(artifact)
        if meta.catalog_listing:
            return True
        name = _artifact_name(artifact)
        if "catalog" in name or "tables" in name or "csv_tables" in name:
            return True
    return False


def _has_schema_artifact(artifacts: list[Any]) -> bool:
    for artifact in artifacts:
        meta = _artifact_contract_meta(artifact)
        if meta.schema_description:
            return True
        data = getattr(artifact, "data", None)
        if isinstance(data, pd.DataFrame) and {"table_name", "column_name"} <= set(map(str, data.columns)):
            return True
    return False


def _has_aggregation_artifact(artifacts: list[Any]) -> bool:
    metric_markers = (
        "sum",
        "total",
        "count",
        "avg",
        "fact",
        "plan",
        "delta",
        "variance",
        "отклон",
        "сумм",
        "итог",
    )
    for artifact in artifacts:
        if not is_tabular_artifact_type(getattr(artifact, "artifact_type", "")):
            continue
        meta = _artifact_contract_meta(artifact)
        if meta.aggregation:
            return True
        data = getattr(artifact, "data", None)
        if isinstance(data, pd.DataFrame):
            columns = " ".join(str(column).lower() for column in data.columns)
            if any(marker in columns for marker in metric_markers):
                return True
    return False


def _has_joined_table_artifact(artifacts: list[Any]) -> bool:
    for artifact in artifacts:
        if not is_tabular_artifact_type(getattr(artifact, "artifact_type", "")):
            continue
        meta = _artifact_contract_meta(artifact)
        if len({str(item) for item in meta.lineage.source_table_names if str(item).strip()}) >= 2:
            return True
        selection = meta.table_selection
        if isinstance(selection, dict) and selection.get("additional_tables"):
            return True
    return False


def _has_brief(text: str) -> bool:
    normalized = str(text or "").strip()
    if len(normalized) < 20:
        return False
    return any(token in normalized.lower() for token in ("суть", "вывод", "инсайт", "ключев", "что сделано"))
