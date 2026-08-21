from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from backend.data_access.semantic_models import SemanticMetricCreate, utc_now_iso

ScenarioProposalKind = Literal["metric", "metric_update", "modeling_gap", "rollup_rule"]
ScenarioProposalStatus = Literal["pending", "approved", "rejected", "applied", "failed"]
ScenarioReviewStatus = Literal["draft", "partially_applied", "applied"]


class SemanticScenarioRequest(BaseModel):
    title: str = Field(default="Сценарии аналитики", max_length=160)
    questions: list[str] = Field(..., min_length=1, max_length=50)

    @model_validator(mode="after")
    def normalize(self) -> SemanticScenarioRequest:
        self.title = self.title.strip() or "Сценарии аналитики"
        self.questions = [item.strip() for item in self.questions if item.strip()]
        if not self.questions:
            raise ValueError("At least one question is required")
        return self


class SemanticScenarioCoverage(BaseModel):
    question_index: int = Field(..., ge=0)
    status: Literal["covered", "partial", "gap"] = "gap"
    rationale: str = ""
    existing_metric_keys: list[str] = Field(default_factory=list)
    proposal_ids: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)


class SemanticScenarioProposal(BaseModel):
    proposal_id: str
    kind: ScenarioProposalKind
    title: str
    rationale: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    question_indexes: list[int] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metric: SemanticMetricCreate | None = None
    target_metric_key: str | None = None
    status: ScenarioProposalStatus = "pending"
    error: str | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> SemanticScenarioProposal:
        if self.kind in {"metric", "metric_update"} and self.metric is None:
            raise ValueError("Metric proposal requires metric payload")
        if self.kind not in {"metric", "metric_update"} and self.metric is not None:
            raise ValueError("Only metric proposals may contain a metric payload")
        if self.kind == "metric_update" and not self.target_metric_key:
            raise ValueError("Metric update proposal requires target_metric_key")
        if self.kind != "metric_update" and self.target_metric_key is not None:
            raise ValueError("Only metric update proposals may contain target_metric_key")
        return self


class SemanticScenarioReview(BaseModel):
    review_id: str
    source_key: str
    source_fingerprint: str
    catalog_overlay_version: int = 0
    review_version: int = 1
    title: str
    questions: list[str]
    coverage: list[SemanticScenarioCoverage] = Field(default_factory=list)
    proposals: list[SemanticScenarioProposal] = Field(default_factory=list)
    status: ScenarioReviewStatus = "draft"
    created_by_user_id: int
    applied_by_user_id: int | None = None
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)


class SemanticScenarioApplyRequest(BaseModel):
    proposal_ids: list[str] = Field(..., min_length=1)
    expected_review_version: int = Field(..., ge=1)
    expected_source_fingerprint: str = Field(..., min_length=1)


class SemanticScenarioApplyResponse(BaseModel):
    review: SemanticScenarioReview
    applied_metric_keys: list[str] = Field(default_factory=list)
    rejected_items: list[str] = Field(default_factory=list)
