"""Central registry of tool factories.

``ToolRegistry`` owns an ordered map of ``ToolFactory`` instances and exposes two
operations used by ``AgentRunner``:

- ``is_available(key, ctx)`` — lightweight per-key check used during intent routing
  (no tool is actually constructed, just the guard conditions are evaluated).
- ``build_tools(ctx)`` — returns the complete ordered list of tools ready to hand
  to the LLM for the current turn.

Create the registry once in ``AgentRunner.__init__`` via ``ToolRegistry.from_services``
and reuse it across graph invocations.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from backend.tools.active_catalog import (
    DEFAULT_CAPABILITY_DEFINITIONS,
    ActiveCapability,
    ActiveCapabilityCatalog,
    ActiveToolSurface,
    CapabilityDefinition,
    CapabilityResolution,
    CapabilityResolutionStatus,
    ProviderBinding,
    inferred_artifact_types,
    normalize_capability_key,
    required_schema_inputs,
    schema_property_names,
)
from backend.tools.catalog import ALL_TOOL_SPECS, ToolCatalogSpec
from backend.tools.impl.factory import (
    AnomalyPlanfactToolFactory,
    DatabaseToolFactory,
    DataCatalogToolFactory,
    ForecastToolFactory,
    GenerateReportToolFactory,
    GenerateSummaryToolFactory,
    GetToolInstructionsToolFactory,
    MemoryToolFactory,
    PandasToolFactory,
    PlotlyToolFactory,
    RagToolFactory,
    SemanticCatalogEditToolFactory,
    SemanticCatalogGenerateToolFactory,
    SemanticCatalogReadToolFactory,
    SessionNoteToolFactory,
    SQLToolFactory,
    ToolFactory,
    UpdatePlanToolFactory,
)
from backend.tools.impl.mcp_tool import MCPToolFactory
from backend.tools.instructions import get_default_tool_instruction_registry

if TYPE_CHECKING:
    from backend.integrations import (
        AnomalyPlanfactIntegrationService,
        ForecastIntegrationService,
    )
    from backend.integrations.rag import RAGService
    from backend.mcp.models import MCPServerConfig, MCPToolDescriptor
    from backend.mcp.service import MCPToolProvider
    from backend.skills.registry import SkillRegistry
    from backend.tools.context import ToolBuildContext

_TOOL_SPECS_BY_KEY: dict[str, ToolCatalogSpec] = {spec.tool_key: spec for spec in ALL_TOOL_SPECS}
_TOOL_SPECS_BY_CAPABILITY: dict[str, ToolCatalogSpec] = {spec.capability_key: spec for spec in ALL_TOOL_SPECS}


class ToolRegistry:
    """Ordered collection of :class:`ToolFactory` instances."""

    def __init__(
        self,
        factories: list[ToolFactory],
        *,
        capability_definitions: list[CapabilityDefinition]
        | tuple[CapabilityDefinition, ...] = DEFAULT_CAPABILITY_DEFINITIONS,
    ) -> None:
        # Preserve insertion order so LLM tool descriptions come out predictably.
        self._factories: dict[str, ToolFactory] = {f.key: f for f in factories}
        self._capability_definitions = {definition.key: definition for definition in capability_definitions}

    # ── Queries ───────────────────────────────────────────────────────────────

    def is_available(self, key: str, ctx: ToolBuildContext) -> bool:
        """Return True when *key* is bound on the active surface."""
        return key in {tool.name for tool in self.build_active_snapshot(ctx).tools}

    def build_tools(self, ctx: ToolBuildContext) -> list:
        """Materialise every available tool in insertion order."""
        return list(self.build_active_snapshot(ctx).tools)

    def build_active_surface(self, ctx: ToolBuildContext) -> ActiveToolSurface:
        return self.build_active_snapshot(ctx)

    def build_active_snapshot(self, ctx: ToolBuildContext) -> ActiveToolSurface:
        """Build the immutable model/executor snapshot for one run."""
        candidates: list[tuple[Any, ActiveCapability, ProviderBinding]] = []
        for factory in self._factories.values():
            if not factory.is_available(ctx):
                continue
            tool = factory.build(ctx)
            capability = self._capability_for_factory(factory, tool)
            candidates.append((tool, capability, self._provider_binding(factory, capability)))

        grouped: dict[str, list[tuple[Any, ActiveCapability, ProviderBinding]]] = {}
        for candidate in candidates:
            grouped.setdefault(candidate[1].key, []).append(candidate)

        resolutions: list[CapabilityResolution] = []
        selected: list[tuple[Any, ActiveCapability, ProviderBinding]] = []
        capability_keys = sorted(set(grouped) | set(self._capability_definitions))
        for capability_key in capability_keys:
            group = grouped.get(capability_key, [])
            resolution, chosen = self._resolve_capability(capability_key, group)
            resolutions.append(resolution)
            if chosen is not None:
                selected.append(chosen)
        selected.sort(key=lambda item: item[2].bound_tool_key)
        catalog = ActiveCapabilityCatalog(capabilities=tuple(item[1] for item in selected))
        return ActiveToolSurface.create(
            tools=(item[0] for item in selected),
            catalog=catalog,
            capability_definitions=self._capability_definitions.values(),
            resolutions=resolutions,
            configuration_revision=self._configuration_revision(ctx),
        )

    def describe_available_tools(self, ctx: ToolBuildContext) -> str:
        """Return a compact multi-line block describing each available tool to the agent."""
        return self.describe_catalog(self.build_active_surface(ctx).catalog)

    @staticmethod
    def describe_catalog(catalog: ActiveCapabilityCatalog) -> str:
        return "\n".join(
            f"- `{capability.bound_tool_key}`: {capability.description} [capability: {capability.key}]"
            for capability in catalog.capabilities
        )

    @staticmethod
    def _resolve_capability(
        capability_key: str,
        candidates: list[tuple[Any, ActiveCapability, ProviderBinding]],
    ) -> tuple[
        CapabilityResolution,
        tuple[Any, ActiveCapability, ProviderBinding] | None,
    ]:
        if not candidates:
            return (
                CapabilityResolution(
                    capability_key=capability_key,
                    status=CapabilityResolutionStatus.UNAVAILABLE,
                    diagnostic=f"No active provider is bound for capability '{capability_key}'.",
                ),
                None,
            )
        ordered = sorted(
            candidates,
            key=lambda item: (
                item[2].provider_identity,
                item[2].bound_tool_key,
            ),
        )
        preferred = [item for item in ordered if item[2].preferred]
        eligible = preferred or ordered
        highest_priority = max(item[2].priority for item in eligible)
        winners = [item for item in eligible if item[2].priority == highest_priority]
        if len(winners) != 1:
            bindings = tuple(item[2] for item in ordered)
            return (
                CapabilityResolution(
                    capability_key=capability_key,
                    status=CapabilityResolutionStatus.AMBIGUOUS,
                    candidates=bindings,
                    diagnostic=(
                        f"Capability '{capability_key}' has ambiguous active providers: "
                        + ", ".join(binding.provider_identity for binding in bindings)
                    ),
                ),
                None,
            )
        chosen = winners[0]
        return (
            CapabilityResolution(
                capability_key=capability_key,
                status=CapabilityResolutionStatus.RESOLVED,
                binding=chosen[2],
                candidates=tuple(item[2] for item in ordered),
            ),
            chosen,
        )

    def _configuration_revision(self, ctx: ToolBuildContext) -> str:
        allowed = getattr(ctx, "allowed_tool_keys", None)
        mcp_configs = []
        for factory in self._factories.values():
            if not isinstance(factory, MCPToolFactory):
                continue
            config = factory.config
            mcp_configs.append(
                {
                    "server_id": config.server_id,
                    "updated_at": config.updated_at,
                    "timeout_sec": config.timeout_sec,
                    "enabled": config.enabled,
                    "url": config.url,
                    "command": config.command,
                    "args": config.args,
                }
            )
        canonical = {
            "allowed_tool_keys": (sorted(str(item) for item in allowed) if allowed is not None else None),
            "mcp_configs": sorted(mcp_configs, key=lambda item: item["server_id"]),
        }
        payload = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _provider_binding(factory: ToolFactory, capability: ActiveCapability) -> ProviderBinding:
        return ProviderBinding(
            capability_key=capability.key,
            provider_identity=str(
                getattr(factory, "provider_identity", "")
                or (
                    factory.descriptor.provider_identity
                    if isinstance(factory, MCPToolFactory)
                    else factory.key
                )
                or factory.key
            ),
            bound_tool_key=capability.bound_tool_key,
            runtime_route=capability.runtime_route,
            priority=int(
                getattr(factory, "binding_priority", None)
                if getattr(factory, "binding_priority", None) is not None
                else (factory.descriptor.binding_priority if isinstance(factory, MCPToolFactory) else 0)
            ),
            preferred=bool(
                getattr(factory, "binding_preferred", False)
                or (factory.descriptor.binding_preferred if isinstance(factory, MCPToolFactory) else False)
            ),
            input_schema=capability.input_schema,
            output_schema=capability.output_schema,
            artifact_types=capability.artifact_types,
            provenance=capability.provenance,
            failure_semantics=capability.failure_semantics,
            description=capability.description,
        )

    @staticmethod
    def _tool_input_schema(tool: Any) -> dict[str, Any]:
        schema = getattr(tool, "args_schema", None)
        if isinstance(schema, dict):
            return dict(schema)
        if isinstance(schema, type) and hasattr(schema, "model_json_schema"):
            return dict(schema.model_json_schema())
        return {}

    def _capability_for_factory(self, factory: ToolFactory, tool: Any) -> ActiveCapability:
        instruction_registry = get_default_tool_instruction_registry()
        spec = _TOOL_SPECS_BY_KEY.get(factory.key)
        runtime_route = spec.kind if spec is not None else "builtin"
        provenance = runtime_route
        output_schema: dict[str, Any] | None = None

        if isinstance(factory, MCPToolFactory):
            capability_key = (
                normalize_capability_key(factory.descriptor.capability_key)
                if factory.descriptor.capability_key
                else f"mcp_tool:{factory.descriptor.tool_key}"
            )
            semantic_spec = _TOOL_SPECS_BY_CAPABILITY.get(capability_key)
            runtime_route = "mcp"
            provenance = f"mcp:{factory.config.server_id}"
            output_schema = factory.descriptor.output_schema
        else:
            semantic_spec = spec
            capability_key = (
                normalize_capability_key(getattr(factory, "capability_key", ""))
                if getattr(factory, "capability_key", None)
                else spec.capability_key
                if spec is not None
                else normalize_capability_key(factory.key)
            )

        document = instruction_registry.get_optional(factory.key)
        description = str(
            getattr(tool, "description", "")
            or (document.metadata.description if document is not None else "")
            or (spec.description if spec is not None else "")
        ).strip()
        input_schema = self._tool_input_schema(tool)
        capabilities = semantic_spec.capabilities if semantic_spec is not None else ()
        triggers = list(semantic_spec.trigger_conditions if semantic_spec is not None else ())
        if document is not None:
            for trigger in document.metadata.triggers:
                if trigger not in triggers:
                    triggers.append(trigger)
        definition = self._capability_definitions.get(capability_key)
        return ActiveCapability(
            key=capability_key,
            semantic_purpose=(
                definition.semantic_purpose
                if definition is not None
                else description or capability_key.replace("_", " ")
            ),
            trigger_conditions=tuple(triggers),
            required_inputs=required_schema_inputs(input_schema),
            produced_outputs=schema_property_names(output_schema),
            artifact_types=tuple(
                definition.required_artifact_types
                if definition is not None
                else inferred_artifact_types(
                    declared=(semantic_spec.artifact_types if semantic_spec is not None else ()),
                    capabilities=capabilities,
                    output_schema=output_schema,
                )
            ),
            provenance=provenance,
            failure_semantics=(
                semantic_spec.failure_semantics
                if semantic_spec is not None
                else "Return a structured actionable error; retry only with changed inputs."
            ),
            runtime_route=runtime_route,
            bound_tool_key=str(tool.name),
            description=description or capability_key,
            input_schema=input_schema,
            output_schema=output_schema,
            specialized=bool(
                definition.provenance_required
                if definition is not None
                else semantic_spec.specialized
                if semantic_spec is not None
                else False
            ),
        )

    # ── Factory method ────────────────────────────────────────────────────────

    @classmethod
    def from_services(
        cls,
        *,
        forecast_service: ForecastIntegrationService | None = None,
        anomaly_planfact_service: AnomalyPlanfactIntegrationService | None = None,
        rag_service: RAGService | None = None,
        memory_note_callback: Callable[[str], None] | None = None,
        session_note_callback: Callable[[str], None] | None = None,
        skill_registry: SkillRegistry | None = None,
        mcp_tool_provider: MCPToolProvider | None = None,
        mcp_server_configs: dict[str, MCPServerConfig] | None = None,
        mcp_tool_descriptors: list[MCPToolDescriptor] | None = None,
        semantic_catalog_service: object | None = None,
        semantic_generation_service: object | None = None,
    ) -> ToolRegistry:
        """Assemble a registry from optional integration services plus all built-in tools."""
        del semantic_catalog_service, semantic_generation_service
        factories: list[ToolFactory] = []

        # Integration-backed tools (only registered when a service is provided).
        if forecast_service is not None:
            factories.append(ForecastToolFactory(forecast_service))
        if anomaly_planfact_service is not None:
            factories.append(AnomalyPlanfactToolFactory(anomaly_planfact_service))
        if rag_service is not None:
            factories.append(RagToolFactory(rag_service))

        if mcp_tool_provider is not None and mcp_server_configs:
            for descriptor in mcp_tool_descriptors or []:
                config = mcp_server_configs.get(descriptor.server_id)
                if config is None:
                    continue
                factories.append(
                    MCPToolFactory(
                        config=config,
                        descriptor=descriptor,
                        provider=mcp_tool_provider,
                    )
                )

        # Built-in tools are always registered; their own is_available guards handle
        # data-context requirements (df / db_runtime_config / allowed_tool_keys).
        factories.extend(
            [
                UpdatePlanToolFactory(),
                GenerateSummaryToolFactory(),
                GenerateReportToolFactory(),
                DataCatalogToolFactory(),
                SQLToolFactory(),
                DatabaseToolFactory(),
                SemanticCatalogReadToolFactory(),
                SemanticCatalogEditToolFactory(),
                SemanticCatalogGenerateToolFactory(),
                PlotlyToolFactory(),
                PandasToolFactory(),
            ]
        )

        # get_tool_instructions: always available when a skill registry is provided.
        if skill_registry is not None:
            factories.append(GetToolInstructionsToolFactory(skill_registry))

        # Memory tools are always available (no data or service requirements).
        factories.append(MemoryToolFactory(memory_note_callback or (lambda _: None)))
        factories.append(SessionNoteToolFactory(session_note_callback or (lambda _: None)))

        return cls(factories)
