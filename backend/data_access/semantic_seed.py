from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from backend.data_access.semantic_models import (
    SemanticCatalog,
    SemanticMetric,
    SemanticTable,
    SemanticTerm,
    clean_list,
    stable_id,
    utc_now_iso,
)

logger = logging.getLogger(__name__)

_SEED_PATH = Path(__file__).with_name("semantic_seed.global.json")
_SEED_GLOB = "semantic_seed.*.json"

MetricTemplateKind = Literal["count_rows", "sum", "avg", "count_distinct", "sum_per_count", "ratio_sum"]


class SemanticSeedTerm(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = ""
    synonyms: list[str] = Field(default_factory=list)
    entity_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize(self) -> SemanticSeedTerm:
        self.name = self.name.strip()
        self.synonyms = clean_list(self.synonyms)
        self.entity_refs = clean_list(self.entity_refs)
        return self


class SemanticMetricTemplate(BaseModel):
    key: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    kind: MetricTemplateKind
    pack_name: str = ""
    column_group: str | None = None
    numerator_group: str | None = None
    denominator_group: str | None = None
    distinct_group: str | None = None
    synonyms: list[str] = Field(default_factory=list)
    description: str = ""
    format: str = "number"

    @model_validator(mode="after")
    def normalize(self) -> SemanticMetricTemplate:
        self.key = re.sub(r"[^A-Za-z0-9_]+", "_", self.key.strip().lower()).strip("_")
        self.name = self.name.strip()
        self.synonyms = clean_list(self.synonyms)
        if not self.key:
            raise ValueError("metric template key must contain a letter or digit")
        return self


class SemanticSeedConfig(BaseModel):
    terms: list[SemanticSeedTerm] = Field(default_factory=list)
    metric_columns: dict[str, list[str]] = Field(default_factory=dict)
    metric_templates: list[SemanticMetricTemplate] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize(self) -> SemanticSeedConfig:
        self.metric_columns = {
            str(key).strip(): clean_list(values)
            for key, values in self.metric_columns.items()
            if str(key).strip()
        }
        return self


class SemanticSeedPack(SemanticSeedConfig):
    name: str
    enabled_by_default: bool = True
    domains: list[str] = Field(default_factory=list)


@lru_cache(maxsize=1)
def load_semantic_seed(path: Path = _SEED_PATH) -> SemanticSeedConfig:
    try:
        return SemanticSeedConfig.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        logger.warning("Semantic seed config is invalid: %s", exc)
        return SemanticSeedConfig()


@lru_cache(maxsize=1)
def load_semantic_seed_packs(seed_dir: Path = Path(__file__).parent) -> tuple[SemanticSeedPack, ...]:
    packs: list[SemanticSeedPack] = []
    paths = sorted(
        seed_dir.glob(_SEED_GLOB),
        key=lambda path: (0 if path.name == "semantic_seed.global.json" else 1, path.name),
    )
    for path in paths:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw.setdefault("name", path.stem.replace("semantic_seed.", ""))
            packs.append(SemanticSeedPack.model_validate(raw))
        except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
            logger.warning("Semantic seed pack %s is invalid: %s", path.name, exc)
    return tuple(packs)


def _enabled_seed() -> SemanticSeedConfig:
    terms = []
    metric_columns: dict[str, list[str]] = {}
    metric_templates = []
    for pack in load_semantic_seed_packs():
        if not pack.enabled_by_default:
            continue
        terms.extend(pack.terms)
        for key, values in pack.metric_columns.items():
            metric_columns.setdefault(key, [])
            metric_columns[key].extend(values)
        metric_templates.extend(
            template.model_copy(update={"pack_name": pack.name})
            for template in pack.metric_templates
        )
    return SemanticSeedConfig(
        terms=terms,
        metric_columns=metric_columns,
        metric_templates=metric_templates,
    )


def starter_terms(source_key: str) -> list[SemanticTerm]:
    now = utc_now_iso()
    return [
        SemanticTerm(
            term_id=f"term:{stable_id('term', source_key, item.name)}",
            name=item.name,
            description=item.description,
            synonyms=item.synonyms,
            entity_refs=item.entity_refs,
            created_at=now,
            updated_at=now,
        )
        for item in _enabled_seed().terms
    ]


def starter_metrics(catalog: SemanticCatalog) -> list[SemanticMetric]:
    table = _starter_fact_table(catalog)
    if table is None:
        return []

    columns = [column for column in catalog.columns if column.table == table.qualified_name]
    column_names = [column.name for column in columns]
    time_column = next((column.name for column in columns if column.semantic_role == "time"), None)
    dimensions = [column.name for column in columns if column.semantic_role == "dimension"][:5]
    seed = _enabled_seed()
    resolved_columns = {
        key: _find_column(column_names, candidates)
        for key, candidates in seed.metric_columns.items()
    }

    candidates: list[tuple[int, SemanticMetric]] = []
    for template in seed.metric_templates:
        spec = _metric_spec(template, resolved_columns)
        if spec is None:
            continue
        try:
            candidates.append((
                _metric_match_score(template, resolved_columns, seed.metric_columns),
                SemanticMetric(
                    metric_id=f"metric:{template.key}",
                    base_table=table.qualified_name,
                    default_time_dimension=time_column,
                    allowed_dimensions=dimensions,
                    **spec,
                ),
            ))
        except ValueError:
            continue
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [metric for _score, metric in candidates[:10]]


def _metric_spec(
    template: SemanticMetricTemplate,
    columns: dict[str, str | None],
) -> dict[str, object] | None:
    spec: dict[str, object] = {
        "key": template.key,
        "name": template.name,
        "synonyms": template.synonyms,
        "description": template.description,
        "format": template.format,
    }

    if template.kind in {"sum", "avg", "count_distinct"}:
        column = columns.get(template.column_group or "")
        if not column:
            return None
        return {
            **spec,
            "type": "simple",
            "expr": column,
            "agg": template.kind,
        }

    if template.kind == "count_rows":
        distinct = columns.get(template.distinct_group or "")
        if distinct:
            return {**spec, "type": "simple", "expr": distinct, "agg": "count_distinct"}
        return None

    if template.kind == "sum_per_count":
        numerator = columns.get(template.numerator_group or "")
        if not numerator:
            return None
        distinct = columns.get(template.distinct_group or "")
        denominator = f"COUNT(DISTINCT {distinct})" if distinct else "COUNT(*)"
        return {**spec, "type": "derived", "formula": f"SUM({numerator}) / NULLIF({denominator}, 0)"}

    if template.kind == "ratio_sum":
        numerator = columns.get(template.numerator_group or "")
        denominator = columns.get(template.denominator_group or "")
        if not numerator or not denominator:
            return None
        return {**spec, "type": "derived", "formula": f"SUM({numerator}) / NULLIF(SUM({denominator}), 0)"}

    return None


def _metric_match_score(
    template: SemanticMetricTemplate,
    resolved_columns: dict[str, str | None],
    metric_columns: dict[str, list[str]],
) -> int:
    score = 10
    if template.pack_name and template.pack_name != "global_common":
        score += 20
    for group in _template_groups(template):
        column = resolved_columns.get(group)
        if not column:
            continue
        score += 30
        candidates = metric_columns.get(group, [])
        normalized = _normalize_column_name(column)
        if any(normalized == _normalize_column_name(candidate) for candidate in candidates):
            score += 15
        if _normalize_column_name(group) in _normalize_column_name(template.key):
            score += 10
    return score


def _template_groups(template: SemanticMetricTemplate) -> list[str]:
    return clean_list(
        [
            template.column_group or "",
            template.numerator_group or "",
            template.denominator_group or "",
            template.distinct_group or "",
        ]
    )


def _starter_fact_table(catalog: SemanticCatalog) -> SemanticTable | None:
    if not catalog.tables:
        return None
    return next((table for table in catalog.tables if table.semantic_role == "fact"), catalog.tables[0])


def _find_column(columns: list[str], candidates: list[str]) -> str | None:
    by_normalized = {_normalize_column_name(column): column for column in columns}
    for candidate in candidates:
        normalized = _normalize_column_name(candidate)
        if normalized in by_normalized:
            return by_normalized[normalized]
    for column in columns:
        normalized = _normalize_column_name(column)
        if any(_normalize_column_name(candidate) in normalized for candidate in candidates):
            return column
    return None


def _normalize_column_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())
