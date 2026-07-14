from __future__ import annotations

from backend.domain_extensions.investment import (
    INVESTMENT_MARKET_EXTENSION,
    InvestmentMarketAnalysisRequest,
    InvestmentMarketAnalysisResponse,
)
from backend.domain_extensions.models import (
    DomainAnalysisResponse,
    DomainArtifactReference,
    DomainExtensionManifest,
    DomainMCPToolContract,
    DomainToolPermission,
)
from backend.domain_extensions.portfolio import (
    PORTFOLIO_RISK_EXTENSION,
    PortfolioPositionInput,
    PortfolioRiskAnalysisRequest,
    PortfolioRiskAnalysisResponse,
)
from backend.domain_extensions.registry import (
    DEFAULT_DOMAIN_EXTENSION_MANIFESTS,
    DomainExtensionRegistry,
    get_domain_extension_registry,
)
from backend.domain_extensions.sales import (
    RETAIL_SALES_EXTENSION,
    RetailSalesAnalysisRequest,
    RetailSalesAnalysisResponse,
)

DOMAIN_EXTENSION_MANIFESTS: tuple[DomainExtensionManifest, ...] = DEFAULT_DOMAIN_EXTENSION_MANIFESTS

__all__ = [
    "DEFAULT_DOMAIN_EXTENSION_MANIFESTS",
    "DOMAIN_EXTENSION_MANIFESTS",
    "INVESTMENT_MARKET_EXTENSION",
    "PORTFOLIO_RISK_EXTENSION",
    "RETAIL_SALES_EXTENSION",
    "DomainAnalysisResponse",
    "DomainArtifactReference",
    "DomainExtensionManifest",
    "DomainExtensionRegistry",
    "DomainMCPToolContract",
    "DomainToolPermission",
    "InvestmentMarketAnalysisRequest",
    "InvestmentMarketAnalysisResponse",
    "PortfolioPositionInput",
    "PortfolioRiskAnalysisRequest",
    "PortfolioRiskAnalysisResponse",
    "RetailSalesAnalysisRequest",
    "RetailSalesAnalysisResponse",
    "get_domain_extension_registry",
]
