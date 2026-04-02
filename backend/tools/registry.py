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

from typing import TYPE_CHECKING

from typing import Callable

from backend.tools.catalog import ALL_TOOL_SPECS, ToolCatalogSpec
from backend.tools.impl.factory import (
    AnomalyPlanfactToolFactory,
    ForecastToolFactory,
    GetToolInstructionsToolFactory,
    MemoryToolFactory,
    PandasToolFactory,
    PlotlyToolFactory,
    SearchToolFactory,
    SQLTableToolFactory,
    ToolFactory,
    ValueToolFactory,
)

if TYPE_CHECKING:
    from backend.integrations import (
        AnomalyPlanfactIntegrationService,
        ForecastIntegrationService,
        SearchIntegrationService,
    )
    from backend.skills.registry import SkillRegistry
    from backend.tools.context import ToolBuildContext

# Pre-build a lookup from tool_key → short Russian description for planner prompts.
_TOOL_DESCRIPTIONS_RU: dict[str, str] = {
    spec.tool_key: spec.description_ru for spec in ALL_TOOL_SPECS
}

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
        for factory in self._factories.values():
            if not factory.is_available(ctx):
                continue
            desc = _TOOL_DESCRIPTIONS_RU.get(factory.key, "")
            spec = _TOOL_SPECS_BY_KEY.get(factory.key)
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
        memory_note_callback: Callable[[str], None] | None = None,
        skill_registry: SkillRegistry | None = None,
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

        # Built-in tools are always registered; their own is_available guards handle
        # data-context requirements (df / db_runtime_config / allowed_tool_keys).
        factories.extend([
            SQLTableToolFactory(),
            PlotlyToolFactory(),
            PandasToolFactory(),
            ValueToolFactory(),
        ])

        # get_tool_instructions: always available when a skill registry is provided.
        if skill_registry is not None:
            factories.append(GetToolInstructionsToolFactory(skill_registry))

        # Memory tool is always available (no data or service requirements).
        factories.append(MemoryToolFactory(memory_note_callback or (lambda _: None)))

        return cls(factories)


