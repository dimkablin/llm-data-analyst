from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.domain_extensions.models import (
    DomainAnalysisResponse,
    DomainExtensionManifest,
    DomainMCPToolContract,
    DomainToolPermission,
)
from backend.skills.contracts import SkillArtifactRequirement

INVESTMENT_MARKET_SKILL_ID = "investment_market_analysis"
INVESTMENT_MARKET_REQUIRED_TOOLS = (
    "database_tool",
    "sql_tool",
    "pandas_tool",
    "plotly_tool",
)


class InvestmentMarketAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    instruments: tuple[str, ...] = Field(default_factory=tuple)
    lookback_days: int = Field(default=90, ge=1, le=3650)
    include_news: bool = True

    @field_validator("question", mode="before")
    @classmethod
    def _normalize_question(cls, value: object) -> str:
        return str(value or "").strip()

    @field_validator("instruments", mode="before")
    @classmethod
    def _normalize_instruments(cls, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        raw_values = [value] if isinstance(value, str) else list(value)  # type: ignore[arg-type]
        instruments: list[str] = []
        seen: set[str] = set()
        for raw in raw_values:
            text = str(raw or "").strip().upper()
            if not text or text in seen:
                continue
            seen.add(text)
            instruments.append(text)
        return tuple(instruments)


class InvestmentMarketAnalysisResponse(DomainAnalysisResponse):
    pass


INVESTMENT_MARKET_EXTENSION = DomainExtensionManifest(
    extension_id=INVESTMENT_MARKET_SKILL_ID,
    skill_id=INVESTMENT_MARKET_SKILL_ID,
    capability="investment",
    mcp_server_name="investment-domain-mcp",
    permission=DomainToolPermission(
        required_tool_keys=INVESTMENT_MARKET_REQUIRED_TOOLS,
        required_skill_ids=(INVESTMENT_MARKET_SKILL_ID,),
    ),
    tools=(
        DomainMCPToolContract(
            tool_key=INVESTMENT_MARKET_SKILL_ID,
            mcp_tool_name="investment.market_analysis",
            input_schema_ref="InvestmentMarketAnalysisRequest",
            output_schema_ref="InvestmentMarketAnalysisResponse",
            required_tool_keys=INVESTMENT_MARKET_REQUIRED_TOOLS,
            produced_artifacts=(
                SkillArtifactRequirement(artifact_type="table", name="market_snapshot_table"),
                SkillArtifactRequirement(artifact_type="table", name="market_price_history_table"),
                SkillArtifactRequirement(artifact_type="table", name="market_news_impact_table"),
                SkillArtifactRequirement(artifact_type="plot", name="market_price_history_chart"),
            ),
        ),
    ),
)
