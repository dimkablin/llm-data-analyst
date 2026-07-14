from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.domain_extensions.models import (
    DomainAnalysisResponse,
    DomainExtensionManifest,
    DomainMCPToolContract,
    DomainToolPermission,
)
from backend.skills.contracts import SkillArtifactRequirement

RETAIL_SALES_SKILL_ID = "retail_sales_analysis"
RETAIL_SALES_REQUIRED_TOOLS = ("sql_tool", "plotly_tool")


class RetailSalesAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    dimensions: tuple[str, ...] = Field(default_factory=tuple)
    metrics: tuple[str, ...] = Field(default_factory=tuple)
    period: str | None = None

    @field_validator("question", "period", mode="before")
    @classmethod
    def _normalize_text(cls, value: object) -> str | None:
        text = str(value or "").strip()
        return text or None

    @field_validator("dimensions", "metrics", mode="before")
    @classmethod
    def _normalize_tuple(cls, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        raw_values = [value] if isinstance(value, str) else list(value)  # type: ignore[arg-type]
        values: list[str] = []
        seen: set[str] = set()
        for raw in raw_values:
            text = str(raw or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            values.append(text)
        return tuple(values)


class RetailSalesAnalysisResponse(DomainAnalysisResponse):
    pass


RETAIL_SALES_EXTENSION = DomainExtensionManifest(
    extension_id=RETAIL_SALES_SKILL_ID,
    skill_id=RETAIL_SALES_SKILL_ID,
    capability="retail_sales",
    mcp_server_name="sales-domain-mcp",
    permission=DomainToolPermission(
        required_tool_keys=RETAIL_SALES_REQUIRED_TOOLS,
        required_skill_ids=(RETAIL_SALES_SKILL_ID,),
    ),
    tools=(
        DomainMCPToolContract(
            tool_key=RETAIL_SALES_SKILL_ID,
            mcp_tool_name="sales.retail_analysis",
            input_schema_ref="RetailSalesAnalysisRequest",
            output_schema_ref="RetailSalesAnalysisResponse",
            required_tool_keys=RETAIL_SALES_REQUIRED_TOOLS,
            produced_artifacts=(
                SkillArtifactRequirement(artifact_type="table", name="retail_sales_slice_table"),
                SkillArtifactRequirement(artifact_type="plot", name="retail_sales_trend_chart"),
            ),
        ),
    ),
)
