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

from collections.abc import Callable
from typing import TYPE_CHECKING

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
    PlannerToolFactory,
    PlotlyToolFactory,
    RagToolFactory,
    SearchToolFactory,
    SessionNoteToolFactory,
    SQLToolFactory,
    ToolFactory,
)
from backend.tools.impl.mcp_tool import MCPToolFactory
from backend.tools.instructions import get_default_tool_instruction_registry

if TYPE_CHECKING:
    from backend.integrations import (
        AnomalyPlanfactIntegrationService,
        ForecastIntegrationService,
        SearchIntegrationService,
    )
    from backend.integrations.rag import RAGService
    from backend.mcp.models import MCPServerConfig, MCPToolDescriptor
    from backend.mcp.service import MCPToolProvider
    from backend.skills.registry import SkillRegistry
    from backend.tools.context import ToolBuildContext

_TOOL_SPECS_BY_KEY: dict[str, ToolCatalogSpec] = {
    spec.tool_key: spec for spec in ALL_TOOL_SPECS
}


class ToolRegistry:
    """Ordered collection of :class:`ToolFactory` instances."""

    def __init__(self, factories: list[ToolFactory]) -> None:
        # Preserve insertion order so LLM tool descriptions come out predictably.
        self._factories: dict[str, ToolFactory] = {f.key: f for f in factories}

    # ── Queries ───────────────────────────────────────────────────────────────

    def is_available(self, key: str, ctx: ToolBuildContext) -> bool:
        """Return True when the factory for *key* exists and passes its guard."""
        factory = self._factories.get(key)
        return factory is not None and factory.is_available(ctx)

    def build_tools(self, ctx: ToolBuildContext) -> list:
        """Materialise every available tool in insertion order."""
        return [f.build(ctx) for f in self._factories.values() if f.is_available(ctx)]

    def describe_available_tools(self, ctx: ToolBuildContext) -> str:
        """Return a compact multi-line block describing each available tool for the planner."""
        lines: list[str] = []
        instruction_registry = get_default_tool_instruction_registry()
        for factory in self._factories.values():
            if not factory.is_available(ctx):
                continue
            spec = _TOOL_SPECS_BY_KEY.get(factory.key)
            document = instruction_registry.get_optional(factory.key)
            desc = (
                document.metadata.description
                if document is not None
                else (
                    spec.description
                    if spec is not None
                    else str(getattr(factory, "description", "") or "")
                )
            )
            if spec:
                caps = ", ".join(spec.capabilities)
                lines.append(f"- `{factory.key}`: {desc} [{caps}]")
            elif desc:
                lines.append(f"- `{factory.key}`: {desc}")
            else:
                lines.append(f"- `{factory.key}`")
        return "\n".join(lines)

    # ── Factory method ────────────────────────────────────────────────────────

    @classmethod
    def from_services(
        cls,
        *,
        search_service: SearchIntegrationService | None = None,
        forecast_service: ForecastIntegrationService | None = None,
        anomaly_planfact_service: AnomalyPlanfactIntegrationService | None = None,
        rag_service: RAGService | None = None,
        memory_note_callback: Callable[[str], None] | None = None,
        session_note_callback: Callable[[str], None] | None = None,
        skill_registry: SkillRegistry | None = None,
        mcp_tool_provider: MCPToolProvider | None = None,
        mcp_server_configs: dict[str, MCPServerConfig] | None = None,
        mcp_tool_descriptors: list[MCPToolDescriptor] | None = None,
    ) -> ToolRegistry:
        """Assemble a registry from optional integration services plus all built-in tools."""
        factories: list[ToolFactory] = []

        # Integration-backed tools (only registered when a service is provided).
        if search_service is not None:
            factories.append(SearchToolFactory(search_service))
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
        factories.extend([
            PlannerToolFactory(),
            GenerateSummaryToolFactory(),
            GenerateReportToolFactory(),
            DataCatalogToolFactory(),
            SQLToolFactory(),
            DatabaseToolFactory(),
            PlotlyToolFactory(),
            PandasToolFactory(),
        ])

        # get_tool_instructions: always available when a skill registry is provided.
        if skill_registry is not None:
            factories.append(GetToolInstructionsToolFactory(skill_registry))

        # Memory tools are always available (no data or service requirements).
        factories.append(MemoryToolFactory(memory_note_callback or (lambda _: None)))
        factories.append(SessionNoteToolFactory(session_note_callback or (lambda _: None)))

        return cls(factories)
