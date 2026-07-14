from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from backend.domain_extensions.investment import INVESTMENT_MARKET_EXTENSION
from backend.domain_extensions.models import DomainExtensionManifest, DomainMCPToolContract
from backend.domain_extensions.portfolio import PORTFOLIO_RISK_EXTENSION
from backend.domain_extensions.sales import RETAIL_SALES_EXTENSION

DEFAULT_DOMAIN_EXTENSION_MANIFESTS: tuple[DomainExtensionManifest, ...] = (
    INVESTMENT_MARKET_EXTENSION,
    PORTFOLIO_RISK_EXTENSION,
    RETAIL_SALES_EXTENSION,
)


class DomainExtensionRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manifests: tuple[DomainExtensionManifest, ...] = Field(default_factory=tuple)

    def __init__(
        self,
        manifests: Iterable[DomainExtensionManifest] | None = None,
        **data: object,
    ) -> None:
        if manifests is not None and "manifests" not in data:
            data["manifests"] = tuple(manifests)
        super().__init__(**data)
        _index_by_extension_id(self.manifests)
        _index_by_skill_id(self.manifests)

    @property
    def skill_ids(self) -> tuple[str, ...]:
        return tuple(manifest.skill_id for manifest in self.manifests)

    def list_manifests(self) -> tuple[DomainExtensionManifest, ...]:
        return self.manifests

    def get_by_extension_id(self, extension_id: str) -> DomainExtensionManifest:
        key = str(extension_id or "").strip()
        try:
            return self._extension_index()[key]
        except KeyError as exc:
            raise KeyError(f"Unknown domain extension id: {key}") from exc

    def get_by_skill_id(self, skill_id: str) -> DomainExtensionManifest:
        key = str(skill_id or "").strip()
        try:
            return self._skill_index()[key]
        except KeyError as exc:
            raise KeyError(f"Unknown domain extension skill id: {key}") from exc

    def list_by_capability(self, capability: str) -> tuple[DomainExtensionManifest, ...]:
        key = str(capability or "").strip()
        return tuple(manifest for manifest in self.manifests if manifest.capability == key)

    def tool_contracts_for_skill(self, skill_id: str) -> tuple[DomainMCPToolContract, ...]:
        return self.get_by_skill_id(skill_id).tools

    def required_tool_keys_for_skill(self, skill_id: str) -> tuple[str, ...]:
        return self.get_by_skill_id(skill_id).permission.required_tool_keys

    def _extension_index(self) -> dict[str, DomainExtensionManifest]:
        return _index_by_extension_id(self.manifests)

    def _skill_index(self) -> dict[str, DomainExtensionManifest]:
        return _index_by_skill_id(self.manifests)


def get_domain_extension_registry() -> DomainExtensionRegistry:
    return DomainExtensionRegistry(DEFAULT_DOMAIN_EXTENSION_MANIFESTS)


def _index_by_extension_id(
    manifests: tuple[DomainExtensionManifest, ...],
) -> dict[str, DomainExtensionManifest]:
    indexed: dict[str, DomainExtensionManifest] = {}
    for manifest in manifests:
        if manifest.extension_id in indexed:
            raise ValueError(f"Duplicate domain extension id: {manifest.extension_id}")
        indexed[manifest.extension_id] = manifest
    return indexed


def _index_by_skill_id(
    manifests: tuple[DomainExtensionManifest, ...],
) -> dict[str, DomainExtensionManifest]:
    indexed: dict[str, DomainExtensionManifest] = {}
    for manifest in manifests:
        if manifest.skill_id in indexed:
            raise ValueError(f"Duplicate domain extension skill id: {manifest.skill_id}")
        indexed[manifest.skill_id] = manifest
    return indexed
