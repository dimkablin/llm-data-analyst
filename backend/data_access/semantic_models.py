from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SemanticCatalogStatus = Literal["pending", "indexing", "ready", "failed", "stale", "degraded", "unbound"]
SemanticEntityType = Literal[
    "table",
    "column",
    "entity",
    "dimension",
    "fact",
    "metric",
    "relationship",
    "term",
    "saved_query",
]
SemanticColumnRole = Literal[
    "metric_candidate",
    "dimension",
    "time",
    "identifier",
    "foreign_key_candidate",
    "text",
    "flag",
    "unknown",
]
SemanticTableRole = Literal["fact", "dimension", "bridge", "snapshot", "unknown"]
MetricAggregation = Literal["sum", "avg", "count", "count_distinct", "min", "max"]
SemanticEntityKind = Literal["primary", "unique", "foreign", "natural"]
SemanticDimensionKind = Literal["categorical", "time", "boolean", "number"]
SemanticFactKind = Literal["number", "money", "duration", "count"]
SemanticMetricKind = Literal[
    "simple",
    "derived",
    "ratio",
    "cumulative",
    "conversion",
    "filtered",
    "non_additive",
    "time_comparison",
]
SemanticTimeGrain = Literal["day", "week", "month", "quarter", "year"]
RelationshipCardinality = Literal[
    "one_to_one",
    "one_to_many",
    "many_to_one",
    "many_to_many",
    "unknown",
]


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def stable_id(*parts: object) -> str:
    payload = "|".join(str(part or "") for part in parts)
    return uuid.uuid5(uuid.NAMESPACE_URL, payload).hex


def clean_list(items: list[str] | None) -> list[str]:
    return list(dict.fromkeys(str(item).strip() for item in items or [] if str(item).strip()))


def normalize_semantic_name(value: str) -> str:
    return re.sub(r"[^\w]+", "_", value.strip().lower(), flags=re.UNICODE).strip("_")


class SemanticTable(BaseModel):
    table_id: str
    qualified_name: str
    table_name: str
    source_kind: str
    schema_name: str | None = None
    description: str = ""
    semantic_role: SemanticTableRole = "unknown"
    grain: str = ""
    row_count: int | None = None
    columns_count: int = 0
    aliases: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    quality_notes: list[str] = Field(default_factory=list)
    ai_context: str = ""
    is_hidden: bool = False


class SemanticColumn(BaseModel):
    column_id: str
    table: str
    name: str
    dtype: str = ""
    nullable: bool | None = None
    null_ratio: float | None = None
    distinct_count: int | None = None
    semantic_role: SemanticColumnRole = "unknown"
    description: str = ""
    aliases: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    min_value: str | None = None
    max_value: str | None = None
    top_values: list[str] = Field(default_factory=list)
    quality_notes: list[str] = Field(default_factory=list)
    ai_context: str = ""
    is_hidden: bool = False


class SemanticEntity(BaseModel):
    entity_id: str
    name: str = Field(..., min_length=1, max_length=120)
    table: str = Field(..., min_length=1, max_length=240)
    expr: str = Field(..., min_length=1, max_length=240)
    type: SemanticEntityKind
    description: str = ""
    synonyms: list[str] = Field(default_factory=list)
    is_active: bool = True

    @model_validator(mode="after")
    def normalize(self) -> SemanticEntity:
        self.name = normalize_semantic_name(self.name)
        self.synonyms = clean_list(self.synonyms)
        if not self.name:
            raise ValueError("entity name must contain a letter or digit")
        return self


class SemanticDimension(BaseModel):
    dimension_id: str
    name: str = Field(..., min_length=1, max_length=120)
    table: str = Field(..., min_length=1, max_length=240)
    expr: str = Field(..., min_length=1, max_length=240)
    type: SemanticDimensionKind
    grains: list[SemanticTimeGrain] = Field(default_factory=list)
    description: str = ""
    synonyms: list[str] = Field(default_factory=list)
    is_active: bool = True

    @model_validator(mode="after")
    def normalize(self) -> SemanticDimension:
        self.name = normalize_semantic_name(self.name)
        self.synonyms = clean_list(self.synonyms)
        if not self.name:
            raise ValueError("dimension name must contain a letter or digit")
        return self


class SemanticFact(BaseModel):
    fact_id: str
    name: str = Field(..., min_length=1, max_length=120)
    table: str = Field(..., min_length=1, max_length=240)
    expr: str = Field(..., min_length=1, max_length=240)
    type: SemanticFactKind = "number"
    description: str = ""
    synonyms: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize(self) -> SemanticFact:
        self.name = normalize_semantic_name(self.name)
        self.synonyms = clean_list(self.synonyms)
        if not self.name:
            raise ValueError("fact name must contain a letter or digit")
        return self


class SemanticRelationshipBase(BaseModel):
    from_table: str = Field(..., min_length=1, max_length=240)
    from_column: str = Field(..., min_length=1, max_length=240)
    to_table: str = Field(..., min_length=1, max_length=240)
    to_column: str = Field(..., min_length=1, max_length=240)
    description: str = ""
    cardinality: RelationshipCardinality = "unknown"
    is_active: bool = True


class SemanticRelationshipCreate(SemanticRelationshipBase):
    pass


class SemanticRelationshipUpdate(BaseModel):
    from_table: str | None = Field(default=None, min_length=1, max_length=240)
    from_column: str | None = Field(default=None, min_length=1, max_length=240)
    to_table: str | None = Field(default=None, min_length=1, max_length=240)
    to_column: str | None = Field(default=None, min_length=1, max_length=240)
    description: str | None = None
    cardinality: RelationshipCardinality | None = None
    is_active: bool | None = None


class SemanticRelationship(SemanticRelationshipBase):
    relationship_id: str


class SemanticTermBase(BaseModel):
    name: str
    description: str = ""
    synonyms: list[str] = Field(default_factory=list)
    entity_refs: list[str] = Field(default_factory=list)
    is_active: bool = True

    @model_validator(mode="after")
    def normalize_term(self) -> SemanticTermBase:
        self.name = self.name.strip()
        if not self.name:
            raise ValueError("term name must not be empty")
        self.synonyms = clean_list(self.synonyms)
        self.entity_refs = clean_list(self.entity_refs)
        return self


class SemanticTermCreate(SemanticTermBase):
    pass


class SemanticTermUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=160)
    description: str | None = None
    synonyms: list[str] | None = None
    entity_refs: list[str] | None = None
    is_active: bool | None = None


class SemanticTerm(SemanticTermBase):
    term_id: str
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)


class SemanticTablePatch(BaseModel):
    description: str | None = None
    semantic_role: SemanticTableRole | None = None
    grain: str | None = None
    aliases: list[str] | None = None
    tags: list[str] | None = None
    quality_notes: list[str] | None = None
    ai_context: str | None = None
    is_hidden: bool | None = None


class SemanticColumnPatch(BaseModel):
    description: str | None = None
    semantic_role: SemanticColumnRole | None = None
    aliases: list[str] | None = None
    examples: list[str] | None = None
    quality_notes: list[str] | None = None
    ai_context: str | None = None
    is_hidden: bool | None = None


class SemanticMetricBase(BaseModel):
    key: str = Field(..., min_length=1, max_length=80)
    name: str = Field(..., min_length=1, max_length=160)
    type: SemanticMetricKind = "simple"
    base_table: str = Field(..., min_length=1, max_length=240)
    expr: str | None = None
    agg: MetricAggregation | None = None
    numerator: str | None = None
    denominator: str | None = None
    formula: str = ""
    default_time_dimension: str | None = None
    allowed_dimensions: list[str] = Field(default_factory=list)
    filters: list[str] = Field(default_factory=list)
    format: str = "number"
    description: str = ""
    synonyms: list[str] = Field(default_factory=list)
    is_active: bool = True

    @model_validator(mode="after")
    def normalize(self) -> SemanticMetricBase:
        self.key = re.sub(r"[^A-Za-z0-9_]+", "_", self.key.strip().lower()).strip("_")
        self.name = self.name.strip()
        self.base_table = self.base_table.strip()
        self.expr = str(self.expr or "").strip() or None
        self.numerator = str(self.numerator or "").strip() or None
        self.denominator = str(self.denominator or "").strip() or None
        self.default_time_dimension = str(self.default_time_dimension or "").strip() or None
        self.formula = str(self.formula or "").strip()
        self.allowed_dimensions = clean_list(self.allowed_dimensions)
        self.filters = clean_list(self.filters)
        self.synonyms = clean_list(self.synonyms)
        if not self.key:
            raise ValueError("metric key must contain a letter or digit")
        if self.type == "simple" and (not self.expr or not self.agg):
            raise ValueError("simple metric requires expr and agg")
        if self.type == "simple":
            qualified = f"{self.base_table}.{self.expr}"
            self.formula = (
                f"COUNT(DISTINCT {qualified})"
                if self.agg == "count_distinct"
                else f"{str(self.agg).upper()}({qualified})"
            )
        if self.type == "ratio" and (not self.numerator or not self.denominator):
            raise ValueError("ratio metric requires numerator and denominator")
        if self.type == "derived" and not self.formula:
            raise ValueError("derived metric requires formula")
        return self


class SemanticMetricCreate(SemanticMetricBase):
    pass


class SemanticMetricUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=160)
    type: SemanticMetricKind | None = None
    base_table: str | None = Field(default=None, min_length=1, max_length=240)
    expr: str | None = None
    agg: MetricAggregation | None = None
    numerator: str | None = None
    denominator: str | None = None
    formula: str | None = None
    default_time_dimension: str | None = Field(default=None, max_length=240)
    allowed_dimensions: list[str] | None = None
    filters: list[str] | None = None
    format: str | None = None
    description: str | None = None
    synonyms: list[str] | None = None
    is_active: bool | None = None


class SemanticMetric(SemanticMetricBase):
    metric_id: str
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)


class SemanticSavedQuery(BaseModel):
    query_id: str
    name: str = Field(..., min_length=1, max_length=160)
    metrics: list[str] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list)
    filters: list[str] = Field(default_factory=list)
    description: str = ""


class SemanticValidationIssue(BaseModel):
    code: str
    message: str
    object_type: str = ""
    object_id: str = ""


class SemanticValidationResult(BaseModel):
    errors: list[SemanticValidationIssue] = Field(default_factory=list)
    warnings: list[SemanticValidationIssue] = Field(default_factory=list)
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0)


class SemanticCatalog(BaseModel):
    model_config = ConfigDict(extra="ignore")

    catalog_id: str
    user_id: int
    session_id: str
    source_key: str = ""
    source_type: str = ""
    source_ref_id: str = ""
    source_label: str = ""
    source_fingerprint: str = ""
    version: str = "2.0"
    overlay_version: int = 0
    published_version: int = 0
    status: SemanticCatalogStatus = "pending"
    error: str | None = None
    built_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    tables: list[SemanticTable] = Field(default_factory=list)
    columns: list[SemanticColumn] = Field(default_factory=list)
    entities: list[SemanticEntity] = Field(default_factory=list)
    dimensions: list[SemanticDimension] = Field(default_factory=list)
    facts: list[SemanticFact] = Field(default_factory=list)
    relationships: list[SemanticRelationship] = Field(default_factory=list)
    metrics: list[SemanticMetric] = Field(default_factory=list)
    saved_queries: list[SemanticSavedQuery] = Field(default_factory=list)
    terms: list[SemanticTerm] = Field(default_factory=list)
    validation: SemanticValidationResult = Field(default_factory=SemanticValidationResult)


class SemanticCatalogOverlay(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source_key: str
    source_fingerprint: str = ""
    version: int = 0
    updated_at: str = Field(default_factory=utc_now_iso)
    table_patches: dict[str, SemanticTablePatch] = Field(default_factory=dict)
    column_patches: dict[str, SemanticColumnPatch] = Field(default_factory=dict)
    relationships: list[SemanticRelationship] = Field(default_factory=list)
    metrics: list[SemanticMetric] = Field(default_factory=list)
    terms: list[SemanticTerm] = Field(default_factory=list)


class SemanticSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=8, ge=1, le=50)


class SemanticSearchResultItem(BaseModel):
    entity_type: SemanticEntityType
    entity_id: str
    score: float = 0.0
    payload: dict[str, Any] = Field(default_factory=dict)


class SemanticContextResult(BaseModel):
    status: SemanticCatalogStatus | Literal["empty"] = "empty"
    prompt: str = ""
    items: list[SemanticSearchResultItem] = Field(default_factory=list)
    hints: dict[str, Any] = Field(default_factory=dict)
