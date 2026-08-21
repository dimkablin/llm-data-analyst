from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.tools.capabilities import (
    _CAPABILITY_TABLE_PROMPT_OPTIONS,
    RuntimeTableDescriptor,
    coerce_runtime_table_descriptors,
    format_runtime_table_descriptors,
)


class FrozenDict(dict):
    """JSON-serializable immutable mapping used inside a run snapshot."""

    def _immutable(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("Active registry snapshot mappings are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return FrozenDict({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


class CapabilityDefinition(BaseModel):
    """Provider-agnostic meaning and completion contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str
    semantic_purpose: str
    task_outcome: str
    required_artifact_types: tuple[str, ...] = ()
    provenance_required: bool = False
    completion_semantics: str = ""
    planning_description: str = ""


class CapabilityResolutionStatus(StrEnum):
    RESOLVED = "resolved"
    UNAVAILABLE = "unavailable"
    AMBIGUOUS = "ambiguous"


class ProviderBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    capability_key: str
    provider_identity: str
    bound_tool_key: str
    runtime_route: str
    priority: int = 0
    preferred: bool = False
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] | None = None
    artifact_types: tuple[str, ...] = ()
    provenance: str
    failure_semantics: str
    description: str

    @model_validator(mode="after")
    def _freeze_schemas(self) -> ProviderBinding:
        object.__setattr__(self, "input_schema", _deep_freeze(self.input_schema))
        object.__setattr__(self, "output_schema", _deep_freeze(self.output_schema))
        return self


class CapabilityResolution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    capability_key: str
    status: CapabilityResolutionStatus
    binding: ProviderBinding | None = None
    candidates: tuple[ProviderBinding, ...] = ()
    diagnostic: str | None = None


class ActiveCapability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str
    semantic_purpose: str
    trigger_conditions: tuple[str, ...] = ()
    required_inputs: tuple[str, ...] = ()
    produced_outputs: tuple[str, ...] = ()
    artifact_types: tuple[str, ...] = ()
    provenance: str
    failure_semantics: str
    runtime_route: str
    bound_tool_key: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] | None = None
    specialized: bool = False

    @model_validator(mode="after")
    def _freeze_schemas(self) -> ActiveCapability:
        object.__setattr__(self, "input_schema", _deep_freeze(self.input_schema))
        object.__setattr__(self, "output_schema", _deep_freeze(self.output_schema))
        return self


class ActiveCapabilityCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    capabilities: tuple[ActiveCapability, ...] = ()

    @property
    def tool_keys(self) -> list[str]:
        return [capability.bound_tool_key for capability in self.capabilities]

    @property
    def capability_keys(self) -> list[str]:
        return [capability.key for capability in self.capabilities]

    def capability_for_tool(self, tool_key: str) -> ActiveCapability | None:
        clean = str(tool_key or "").strip()
        return next(
            (capability for capability in self.capabilities if capability.bound_tool_key == clean),
            None,
        )

class ActiveRegistrySnapshot(BaseModel):
    """Immutable run-scoped projection of executable registry state."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    tools: tuple[Any, ...] = ()
    catalog: ActiveCapabilityCatalog = Field(default_factory=ActiveCapabilityCatalog)
    capability_definitions: tuple[CapabilityDefinition, ...] = ()
    resolutions: tuple[CapabilityResolution, ...] = ()
    configuration_revision: str = ""
    fingerprint: str

    def resolution_for(self, capability_key: str) -> CapabilityResolution:
        clean = normalize_capability_key(capability_key)
        resolution = next(
            (item for item in self.resolutions if item.capability_key == clean),
            None,
        )
        return resolution or CapabilityResolution(
            capability_key=clean,
            status=CapabilityResolutionStatus.UNAVAILABLE,
            diagnostic=f"No active provider is bound for capability '{clean}'.",
        )

    @classmethod
    def create(
        cls,
        *,
        tools: Iterable[Any],
        catalog: ActiveCapabilityCatalog,
        capability_definitions: Iterable[CapabilityDefinition] = (),
        resolutions: Iterable[CapabilityResolution],
        configuration_revision: str = "",
    ) -> ActiveRegistrySnapshot:
        ordered_resolutions = tuple(sorted(resolutions, key=lambda item: item.capability_key))
        ordered_definitions = tuple(
            sorted(capability_definitions, key=lambda item: item.key)
        )
        canonical = {
            "configuration_revision": configuration_revision,
            "capability_definitions": [
                definition.model_dump(mode="json") for definition in ordered_definitions
            ],
            "capabilities": [
                capability.model_dump(mode="json")
                for capability in sorted(catalog.capabilities, key=lambda item: item.key)
            ],
            "resolutions": [
                resolution.model_dump(mode="json") for resolution in ordered_resolutions
            ],
        }
        payload = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return cls(
            tools=tuple(tools),
            catalog=catalog,
            capability_definitions=ordered_definitions,
            resolutions=ordered_resolutions,
            configuration_revision=configuration_revision,
            fingerprint=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        )


class ActiveToolSurface(ActiveRegistrySnapshot):
    """Backward-compatible name for the canonical active registry snapshot."""


DEFAULT_CAPABILITY_DEFINITIONS: tuple[CapabilityDefinition, ...] = (
    CapabilityDefinition(
        key="forecast",
        semantic_purpose="Produce a provider-backed time-series forecast.",
        task_outcome="Forecast values for future periods with provider provenance.",
        required_artifact_types=("json",),
        provenance_required=True,
        completion_semantics=(
            "Complete only after a resolved forecasting provider returns a valid artifact."
        ),
        planning_description="Use for requests that require future time-series values.",
    ),
)


def normalize_capability_key(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").casefold()).strip("_")
    return normalized or "unknown"


def schema_property_names(schema: dict[str, Any] | None) -> list[str]:
    root = dict(schema or {})
    definitions = root.get("$defs") if isinstance(root.get("$defs"), dict) else {}
    found: list[str] = []
    visited: set[int] = set()

    def walk(node: Any) -> None:
        if not isinstance(node, dict) or id(node) in visited:
            return
        visited.add(id(node))
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            walk(definitions.get(ref.rsplit("/", 1)[-1]))
        properties = node.get("properties")
        if isinstance(properties, dict):
            for name in properties:
                clean = str(name).strip()
                if clean and clean not in found:
                    found.append(clean)
        for key in ("anyOf", "oneOf", "allOf"):
            variants = node.get(key)
            if isinstance(variants, list):
                for variant in variants:
                    walk(variant)

    walk(root)
    return found


def required_schema_inputs(schema: dict[str, Any] | None) -> list[str]:
    required = (schema or {}).get("required")
    if not isinstance(required, list):
        return []
    return [str(item).strip() for item in required if str(item).strip()]


def build_active_capability_context(
    *,
    catalog: ActiveCapabilityCatalog,
    definitions: Iterable[CapabilityDefinition] = (),
    resolutions: Iterable[CapabilityResolution] = (),
    has_dataframe: bool,
    has_db_source: bool,
    has_knowledge_base: bool = False,
    csv_table_names: list[str] | None = None,
    csv_table_descriptors: list[RuntimeTableDescriptor | dict[str, Any]] | None = None,
    source_table_count: int = 0,
    source_count: int = 0,
) -> dict[str, Any]:
    source_mode = (
        "db"
        if has_db_source
        else "dataset"
        if has_dataframe
        else "knowledge_base"
        if has_knowledge_base
        else "none"
    )
    tool_keys = catalog.tool_keys
    resolution_list = tuple(resolutions)
    definition_list = tuple(definitions)
    tool_text = ", ".join(f"`{item}`" for item in tool_keys) if tool_keys else "none"
    lines = [
        "[ROLE: ACTIVE_CAPABILITY_CATALOG]",
        f"- Active data mode: `{source_mode}`",
        f"- Bound tools: {tool_text}",
        "- This catalog is the complete action surface for the current run.",
        "- Plans may contain only actions resolved by these capability-to-tool bindings.",
    ]
    for capability in catalog.capabilities:
        inputs = ", ".join(capability.required_inputs) or "none"
        outputs = ", ".join(capability.produced_outputs) or "provider-defined"
        artifacts = ", ".join(capability.artifact_types) or "none"
        suffix = "; provider provenance required" if capability.specialized else ""
        lines.append(
            f"  - `{capability.key}` -> `{capability.bound_tool_key}` "
            f"(inputs: {inputs}; outputs: {outputs}; artifacts: {artifacts}; "
            f"route: {capability.runtime_route}{suffix})"
        )
    for resolution in resolution_list:
        if resolution.status is CapabilityResolutionStatus.RESOLVED:
            continue
        lines.append(
            f"  - `{resolution.capability_key}` -> `{resolution.status.value}`"
            + (f" ({resolution.diagnostic})" if resolution.diagnostic else "")
        )
    lines.extend(
        (
            f"  - `{definition.key}` outcome: {definition.task_outcome}; "
            f"required artifacts: {', '.join(definition.required_artifact_types) or 'none'}; "
            f"provenance required: {str(definition.provenance_required).lower()}"
        )
        for definition in definition_list
    )
    lines.extend(
        [
            "- A specialized capability is complete only after its bound tool returns "
            "a successful result with provider provenance.",
            "- Generic dataframe calculations and trend exploration are not substitutes "
            "for a specialized capability.",
            "- Retry a semantic tool error only with changed inputs or strategy. "
            "When bounded recovery is exhausted, return a partial/unavailable outcome.",
        ]
    )
    if csv_table_names:
        lines.append("- DuckDB tables: " + ", ".join(f"`{table}`" for table in csv_table_names))
    lines.extend(
        format_runtime_table_descriptors(
            coerce_runtime_table_descriptors(csv_table_descriptors),
            _CAPABILITY_TABLE_PROMPT_OPTIONS,
        )
    )
    if "data_catalog_tool" in tool_keys and (source_table_count > 1 or source_count > 1):
        lines.append(
            "- CATALOG-FIRST: multiple sources or tables are active; use "
            "`data_catalog_tool` before analysis when table choice is not explicit."
        )

    return {
        "source_mode": source_mode,
        "available_tool_keys": tool_keys,
        "available_capability_keys": catalog.capability_keys,
        "unavailable_capability_keys": [
            item.capability_key
            for item in resolution_list
            if item.status is not CapabilityResolutionStatus.RESOLVED
        ],
        "capability_resolutions": [
            item.model_dump(mode="json") for item in resolution_list
        ],
        "capability_definitions": [
            item.model_dump(mode="json") for item in definition_list
        ],
        "capabilities": [capability.model_dump(mode="json") for capability in catalog.capabilities],
        "prompt_block": "\n".join(lines),
    }


def inferred_artifact_types(
    *,
    declared: Iterable[str] = (),
    capabilities: Iterable[str] = (),
    output_schema: dict[str, Any] | None = None,
) -> list[str]:
    result = [str(item).strip() for item in declared if str(item).strip()]
    capability_set = {str(item).casefold() for item in capabilities}
    output_set = {item.casefold() for item in schema_property_names(output_schema)}
    for artifact_type, present in (
        ("table", "table_artifact" in capability_set or "rows" in output_set),
        ("plot", "chart_artifact" in capability_set or "plot" in output_set),
        ("json", bool(output_schema)),
    ):
        if present and artifact_type not in result:
            result.append(artifact_type)
    return result
