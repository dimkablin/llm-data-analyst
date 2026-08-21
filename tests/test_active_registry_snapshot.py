from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from backend.tools.active_catalog import (
    CapabilityDefinition,
    CapabilityResolutionStatus,
)
from backend.tools.context import ToolBuildContext
from backend.tools.impl.factory import (
    SemanticCatalogEditToolFactory,
    SemanticCatalogGenerateToolFactory,
    SemanticCatalogReadToolFactory,
)
from backend.tools.registry import ToolRegistry


class _SyntheticInput(BaseModel):
    value: int


class _SyntheticFactory:
    def __init__(
        self,
        *,
        key: str,
        capability_key: str,
        provider: str,
        priority: int = 0,
        preferred: bool = False,
    ) -> None:
        self.key = key
        self.capability_key = capability_key
        self.provider_identity = provider
        self.binding_priority = priority
        self.binding_preferred = preferred

    def is_available(self, _context: ToolBuildContext) -> bool:
        return True

    def build(self, _context: ToolBuildContext) -> StructuredTool:
        return StructuredTool.from_function(
            name=self.key,
            description=f"Synthetic provider {self.provider_identity}",
            args_schema=_SyntheticInput,
            func=lambda value: str(value),
        )


_PROJECTION = CapabilityDefinition(
    key="projection",
    semantic_purpose="Produce a specialized projection.",
    task_outcome="A provider-backed projection artifact.",
    required_artifact_types=("json",),
    provenance_required=True,
)


def _context() -> ToolBuildContext:
    from backend.core.config import Settings

    return ToolBuildContext(settings=Settings(), allowed_tool_keys=None)


def test_capability_registry_does_not_infer_requirements_from_prompt_words() -> None:
    registry = ToolRegistry([], capability_definitions=[_PROJECTION])

    snapshot = registry.build_active_snapshot(_context())

    assert not hasattr(registry, "requirements_for_prompt")
    assert snapshot.resolution_for("projection").status is CapabilityResolutionStatus.UNAVAILABLE


def test_provider_resolution_is_order_independent_and_ambiguous_on_tie() -> None:
    factories = [
        _SyntheticFactory(key="provider_a", capability_key="projection", provider="a"),
        _SyntheticFactory(key="provider_b", capability_key="projection", provider="b"),
    ]

    first = ToolRegistry(factories, capability_definitions=[_PROJECTION]).build_active_snapshot(_context())
    second = ToolRegistry(
        list(reversed(factories)),
        capability_definitions=[_PROJECTION],
    ).build_active_snapshot(_context())

    assert first.fingerprint == second.fingerprint
    assert first.resolution_for("projection").status is CapabilityResolutionStatus.AMBIGUOUS
    assert first.catalog.tool_keys == []
    assert first.resolution_for("projection").diagnostic


def test_explicit_preference_resolves_provider_collision() -> None:
    registry = ToolRegistry(
        [
            _SyntheticFactory(
                key="provider_a",
                capability_key="projection",
                provider="a",
                preferred=True,
            ),
            _SyntheticFactory(key="provider_b", capability_key="projection", provider="b"),
        ],
        capability_definitions=[_PROJECTION],
    )

    snapshot = registry.build_active_snapshot(_context())

    resolution = snapshot.resolution_for("projection")
    assert resolution.status is CapabilityResolutionStatus.RESOLVED
    assert resolution.binding is not None
    assert resolution.binding.bound_tool_key == "provider_a"


def test_snapshot_fingerprint_changes_with_effective_schema() -> None:
    factory = _SyntheticFactory(
        key="provider_a",
        capability_key="projection",
        provider="a",
    )
    first = ToolRegistry([factory], capability_definitions=[_PROJECTION]).build_active_snapshot(_context())
    changed_factory = SimpleNamespace(**factory.__dict__)
    changed_factory.is_available = factory.is_available
    changed_factory.build = lambda _context: StructuredTool.from_function(
        name="provider_a",
        description="Synthetic provider a",
        args_schema=None,
        func=lambda value, unit="day": str((value, unit)),
    )
    second = ToolRegistry(
        [changed_factory],
        capability_definitions=[_PROJECTION],
    ).build_active_snapshot(_context())

    assert first.fingerprint != second.fingerprint


def test_snapshot_nested_schema_is_immutable() -> None:
    snapshot = ToolRegistry(
        [
            _SyntheticFactory(
                key="provider_a",
                capability_key="projection",
                provider="a",
            )
        ],
        capability_definitions=[_PROJECTION],
    ).build_active_snapshot(_context())

    with pytest.raises(TypeError):
        snapshot.catalog.capabilities[0].input_schema["properties"]["value"]["type"] = "string"


def test_semantic_feature_flag_removes_semantic_tools_from_snapshot() -> None:
    from backend.core.config import Settings

    context = ToolBuildContext(
        settings=Settings(semantic_layer_enabled=False),
        semantic_catalog_service=object(),
        semantic_generation_service=object(),
    )
    snapshot = ToolRegistry(
        [
            SemanticCatalogReadToolFactory(),
            SemanticCatalogEditToolFactory(),
            SemanticCatalogGenerateToolFactory(),
        ]
    ).build_active_snapshot(context)

    assert not {
        "semantic_catalog_read_tool",
        "semantic_catalog_edit_tool",
        "semantic_catalog_generate_tool",
    } & set(snapshot.catalog.tool_keys)
