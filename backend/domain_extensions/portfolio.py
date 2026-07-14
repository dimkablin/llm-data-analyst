from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.domain_extensions.models import (
    DomainAnalysisResponse,
    DomainExtensionManifest,
    DomainMCPToolContract,
    DomainToolPermission,
)
from backend.skills.contracts import SkillArtifactRequirement

PORTFOLIO_RISK_SKILL_ID = "portfolio_risk_analysis"
PORTFOLIO_RISK_REQUIRED_TOOLS = ("sql_tool", "pandas_tool", "plotly_tool")


class PortfolioPositionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(min_length=1)
    weight_pct: Decimal | None = Field(default=None, ge=0, le=100)
    market_value: Decimal | None = Field(default=None, ge=0)
    risk_profile: str | None = None

    @field_validator("symbol", "risk_profile", mode="before")
    @classmethod
    def _normalize_text(cls, value: object) -> str | None:
        text = str(value or "").strip()
        return text or None


class PortfolioRiskAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    source_table: str | None = None
    positions: tuple[PortfolioPositionInput, ...] = Field(default_factory=tuple)
    risk_dimensions: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("question", "source_table", mode="before")
    @classmethod
    def _normalize_text(cls, value: object) -> str | None:
        text = str(value or "").strip()
        return text or None

    @field_validator("risk_dimensions", mode="before")
    @classmethod
    def _normalize_dimensions(cls, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        raw_values = [value] if isinstance(value, str) else list(value)  # type: ignore[arg-type]
        dimensions: list[str] = []
        seen: set[str] = set()
        for raw in raw_values:
            text = str(raw or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            dimensions.append(text)
        return tuple(dimensions)


class PortfolioRiskAnalysisResponse(DomainAnalysisResponse):
    pass


PORTFOLIO_RISK_EXTENSION = DomainExtensionManifest(
    extension_id=PORTFOLIO_RISK_SKILL_ID,
    skill_id=PORTFOLIO_RISK_SKILL_ID,
    capability="portfolio_risk",
    mcp_server_name="portfolio-risk-domain-mcp",
    permission=DomainToolPermission(
        required_tool_keys=PORTFOLIO_RISK_REQUIRED_TOOLS,
        required_skill_ids=(PORTFOLIO_RISK_SKILL_ID,),
    ),
    tools=(
        DomainMCPToolContract(
            tool_key=PORTFOLIO_RISK_SKILL_ID,
            mcp_tool_name="portfolio.risk_analysis",
            input_schema_ref="PortfolioRiskAnalysisRequest",
            output_schema_ref="PortfolioRiskAnalysisResponse",
            required_tool_keys=PORTFOLIO_RISK_REQUIRED_TOOLS,
            produced_artifacts=(
                SkillArtifactRequirement(artifact_type="table", name="portfolio_positions_table"),
                SkillArtifactRequirement(artifact_type="table", name="risk_concentration_table"),
                SkillArtifactRequirement(artifact_type="plot", name="risk_concentration_chart"),
            ),
        ),
    ),
)
