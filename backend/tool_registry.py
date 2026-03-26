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

from agent.tools.factory import (
    AnomalyPlanfactToolFactory,
    DBToolFactory,
    ForecastToolFactory,
    MemoryToolFactory,
    PandasToolFactory,
    PlotlyToolFactory,
    SearchToolFactory,
    ToolFactory,
    ValueToolFactory,
)

if TYPE_CHECKING:
    from backend.anomaly_planfact_integration import AnomalyPlanfactIntegrationService
    from backend.forecast_integration import ForecastIntegrationService
    from backend.search_integration import SearchIntegrationService
    from backend.tool_context import ToolBuildContext


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

    # ── Factory method ────────────────────────────────────────────────────────

    @classmethod
    def from_services(
        cls,
        *,
        search_service: SearchIntegrationService | None = None,
        forecast_service: ForecastIntegrationService | None = None,
        anomaly_planfact_service: AnomalyPlanfactIntegrationService | None = None,
        memory_note_callback: Callable[[str], None] | None = None,
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
            PlotlyToolFactory(),
            PandasToolFactory(),
            ValueToolFactory(),
            DBToolFactory(),
        ])

        # Memory tool is always available (no data or service requirements).
        factories.append(MemoryToolFactory(memory_note_callback or (lambda _: None)))

        return cls(factories)
