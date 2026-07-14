from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from backend.agent.runner import AgentRunner
from backend.domain_extensions import (
    DOMAIN_EXTENSION_MANIFESTS,
    DomainExtensionRegistry,
    get_domain_extension_registry,
)
from backend.domain_extensions.models import DomainExtensionManifest
from backend.skills import SkillRegistry


def test_default_domain_extension_registry_lists_known_manifests() -> None:
    registry = get_domain_extension_registry()

    assert isinstance(registry, BaseModel)
    assert registry.skill_ids == (
        "investment_market_analysis",
        "portfolio_risk_analysis",
        "retail_sales_analysis",
    )
    assert registry.get_by_skill_id("portfolio_risk_analysis").capability == "portfolio_risk"
    assert registry.get_by_extension_id("retail_sales_analysis").mcp_server_name == "sales-domain-mcp"


def test_domain_extension_registry_links_to_markdown_skills() -> None:
    registry = get_domain_extension_registry()
    skills = SkillRegistry.from_path("skills")
    loaded_skill_ids = {skill.skill_id for skill in skills.list_skills()}

    assert set(registry.skill_ids).issubset(loaded_skill_ids)
    assert registry.required_tool_keys_for_skill("investment_market_analysis") == (
        "database_tool",
        "sql_tool",
        "pandas_tool",
        "plotly_tool",
    )


def test_domain_extension_registry_rejects_duplicate_skill_ids() -> None:
    duplicate_skill = DOMAIN_EXTENSION_MANIFESTS[0].model_copy(
        update={"extension_id": "duplicate-investment-extension"},
    )
    duplicated = (*DOMAIN_EXTENSION_MANIFESTS, duplicate_skill)

    with pytest.raises(ValueError, match="Duplicate domain extension skill id"):
        DomainExtensionRegistry(duplicated)


def test_domain_extension_registry_rejects_duplicate_extension_ids() -> None:
    duplicate_extension = DOMAIN_EXTENSION_MANIFESTS[0].model_copy(
        update={"skill_id": "duplicate-investment-skill"},
    )
    duplicated = (*DOMAIN_EXTENSION_MANIFESTS, duplicate_extension)

    with pytest.raises(ValueError, match="Duplicate domain extension id"):
        DomainExtensionRegistry(duplicated)


def test_domain_extension_manifest_accepts_new_capability_without_core_model_change() -> None:
    base = DOMAIN_EXTENSION_MANIFESTS[0]
    manifest = DomainExtensionManifest(
        extension_id=base.extension_id,
        skill_id=base.skill_id,
        capability=" insurance ",
        mcp_server_name=base.mcp_server_name,
        permission=base.permission,
        tools=base.tools,
    )

    registry = DomainExtensionRegistry((manifest,))

    assert registry.list_by_capability("insurance") == (manifest,)


def test_domain_extension_registry_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        DomainExtensionRegistry(
            manifests=DOMAIN_EXTENSION_MANIFESTS[:1],
            unexpected=True,
        )


def test_domain_extension_registry_model_copy_keeps_lookup_indexes_consistent() -> None:
    registry = DomainExtensionRegistry(DOMAIN_EXTENSION_MANIFESTS[:1])
    replacement = DOMAIN_EXTENSION_MANIFESTS[1]

    copied = registry.model_copy(update={"manifests": (replacement,)})

    assert copied.get_by_skill_id(replacement.skill_id) is replacement
    with pytest.raises(KeyError):
        copied.get_by_skill_id(DOMAIN_EXTENSION_MANIFESTS[0].skill_id)


def test_domain_extension_manifest_tools_are_permission_scoped() -> None:
    registry = get_domain_extension_registry()

    for manifest in registry.list_manifests():
        permission_tools = set(manifest.permission.required_tool_keys)
        assert manifest.permission.required_skill_ids == (manifest.skill_id,)
        for tool in manifest.tools:
            assert set(tool.required_tool_keys).issubset(permission_tools)
            assert tool.produced_artifacts


def test_domain_extension_registry_accepts_typed_manifest_sequence() -> None:
    manifests: tuple[DomainExtensionManifest, ...] = DOMAIN_EXTENSION_MANIFESTS[:1]

    registry = DomainExtensionRegistry(manifests)

    assert registry.list_manifests() == manifests
    assert registry.model_dump()["manifests"][0]["skill_id"] == manifests[0].skill_id


def test_agent_runner_exposes_domain_extension_registry_dependency() -> None:
    registry = DomainExtensionRegistry(DOMAIN_EXTENSION_MANIFESTS[:1])
    runner = AgentRunner(domain_extension_registry=registry)

    assert runner.domain_extension_registry is registry
