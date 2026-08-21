from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from backend.agent.llm_client import make_reasoning_llm
from backend.data_access.semantic_catalog_service import SemanticCatalogService
from backend.data_access.semantic_models import (
    SemanticCatalog,
    SemanticMetric,
    SemanticMetricCreate,
    stable_id,
    utc_now_iso,
)
from backend.data_access.semantic_scenario_models import (
    SemanticScenarioApplyRequest,
    SemanticScenarioApplyResponse,
    SemanticScenarioCoverage,
    SemanticScenarioProposal,
    SemanticScenarioRequest,
    SemanticScenarioReview,
)

_AGGREGATED_EXPR_RE = re.compile(
    r"^\s*(?:"
    r"COUNT\s*\(\s*DISTINCT\s+(?P<distinct_column>[\w.\"]+)\s*\)"
    r"|(?P<agg>SUM|AVG|COUNT_DISTINCT|COUNT|MIN|MAX)\s*\(\s*(?P<column>[\w.\"]+)\s*\)"
    r")\s*$",
    flags=re.IGNORECASE | re.UNICODE,
)

_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
_LATIN_RE = re.compile(r"[A-Za-z]")


class GeneratedScenarioProposal(BaseModel):
    kind: str = "metric"
    title: str
    rationale: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    question_indexes: list[int] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metric: SemanticMetricCreate | None = None


class GeneratedScenarioCoverage(BaseModel):
    question_index: int
    status: str = "gap"
    rationale: str = ""
    existing_metric_keys: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)


class GeneratedScenarioReview(BaseModel):
    coverage: list[GeneratedScenarioCoverage] = Field(default_factory=list)
    proposals: list[GeneratedScenarioProposal] = Field(default_factory=list)


@dataclass
class SemanticScenarioService:
    catalog_service: SemanticCatalogService
    settings: Any
    llm_generate: Callable[[dict[str, Any]], GeneratedScenarioReview] | None = None

    def analyze(
        self,
        *,
        session_id: str,
        user_id: int,
        request: SemanticScenarioRequest,
    ) -> SemanticScenarioReview:
        catalog = self.catalog_service.load_for_session(session_id=session_id, user_id=user_id)
        if catalog is None:
            raise ValueError("Semantic catalog not found")
        if catalog.status in {"stale", "failed", "unbound", "empty", "not_built"}:
            raise ValueError(f"Semantic catalog is not ready: {catalog.status}")

        generated = self._generate(self._prompt_payload(catalog, request))
        proposals: list[SemanticScenarioProposal] = []
        proposed_metric_keys: set[str] = set()
        working_catalog = catalog.model_copy(deep=True)
        existing_metrics = {metric.key: metric for metric in catalog.metrics}
        existing_metric_keys = set(existing_metrics)
        metric_key_aliases: dict[str, str] = {}
        proposed_update_targets: set[str] = set()
        for index, item in enumerate(generated.proposals[:60]):
            kind = item.kind if item.kind in {"metric", "modeling_gap", "rollup_rule"} else "modeling_gap"
            metric = item.metric if kind == "metric" else None
            target_metric_key: str | None = None
            warnings = list(item.warnings)
            if metric is not None:
                try:
                    original_key = metric.key
                    metric, normalization_note = self._normalize_simple_metric_expression(metric)
                    if normalization_note:
                        warnings.append(normalization_note)
                    metric = self._rewrite_metric_references(metric, metric_key_aliases)
                    if metric.key in proposed_metric_keys:
                        raise ValueError("duplicate metric key in this review")
                    same_key = existing_metrics.get(metric.key)
                    if same_key is not None:
                        if self._metric_signature(catalog, same_key) != self._metric_signature(
                            catalog,
                            metric,
                        ):
                            raise ValueError("metric key already exists with a different calculation")
                        equivalent = same_key
                    else:
                        equivalent = self._equivalent_existing_metric(catalog, metric)
                    if equivalent is not None:
                        if equivalent.key in proposed_update_targets:
                            raise ValueError("duplicate update for the same existing metric in this review")
                        target_metric_key = equivalent.key
                        metric_key_aliases[original_key] = equivalent.key
                        metric = self._metric_update_payload(equivalent, metric)
                        kind = "metric_update"
                        candidate = SemanticMetric(
                            **metric.model_dump(),
                            metric_id=equivalent.metric_id,
                            created_at=equivalent.created_at,
                        )
                    else:
                        candidate = SemanticMetric(**metric.model_dump(), metric_id=f"metric:{metric.key}")
                    self.catalog_service.validate_metric_candidate(working_catalog, candidate)
                    metric = SemanticMetricCreate.model_validate(
                        candidate.model_dump(exclude={"metric_id", "created_at", "updated_at"})
                    )
                    evidence_error = self._metric_filter_evidence_error(catalog, metric)
                    if evidence_error:
                        raise ValueError(evidence_error)
                    if kind == "metric":
                        proposed_metric_keys.add(metric.key)
                        working_catalog.metrics.append(candidate)
                    else:
                        working_catalog.metrics = [
                            candidate if current.metric_id == candidate.metric_id else current
                            for current in working_catalog.metrics
                        ]
                        proposed_update_targets.add(candidate.key)
                except ValueError as exc:
                    kind = "modeling_gap"
                    metric = None
                    target_metric_key = None
                    warnings.append(f"Предложение метрики отклонено валидатором: {exc}")
            proposal_id = f"proposal:{stable_id(catalog.source_key, request.title, index, item.title)}"
            proposals.append(
                SemanticScenarioProposal(
                    proposal_id=proposal_id,
                    kind=kind,
                    title=item.title,
                    rationale=item.rationale,
                    confidence=item.confidence,
                    question_indexes=[i for i in item.question_indexes if 0 <= i < len(request.questions)],
                    evidence=item.evidence,
                    warnings=warnings,
                    metric=metric,
                    target_metric_key=target_metric_key,
                )
            )

        coverage_by_index = {item.question_index: item for item in generated.coverage}
        coverage = []
        for question_index in range(len(request.questions)):
            item = coverage_by_index.get(question_index)
            related_proposals = [p for p in proposals if question_index in p.question_indexes]
            related = [p.proposal_id for p in related_proposals]
            known_existing = [
                key for key in (item.existing_metric_keys if item else []) if key in existing_metric_keys
            ]
            known_existing.extend(
                proposal.target_metric_key
                for proposal in related_proposals
                if proposal.kind == "metric_update" and proposal.target_metric_key
            )
            known_existing = list(dict.fromkeys(known_existing))
            missing_requirements = list(dict.fromkeys(item.missing_requirements if item else []))
            status = item.status if item and item.status in {"covered", "partial", "gap"} else "gap"
            if any(proposal.kind in {"modeling_gap", "rollup_rule"} for proposal in related_proposals):
                status = "partial" if known_existing or related else "gap"
            if any(proposal.kind == "metric" for proposal in related_proposals) and status == "covered":
                status = "partial"
            relevant_tables = self._coverage_tables(catalog, known_existing, related_proposals)
            if len(relevant_tables) > 1 and not self._tables_connected(catalog, relevant_tables):
                missing_requirements.append(self._requirement_text("relationships", request.questions))
                status = "partial" if known_existing or related else "gap"
            missing_requirements = list(dict.fromkeys(missing_requirements))
            if missing_requirements and status == "covered":
                status = "partial"
            if status == "covered" and not known_existing and not related:
                status = "gap"
            coverage.append(
                SemanticScenarioCoverage(
                    question_index=question_index,
                    status=status,
                    rationale=item.rationale if item else "Требуется подтверждение семантических объектов.",
                    existing_metric_keys=known_existing,
                    proposal_ids=related,
                    missing_requirements=missing_requirements,
                )
            )

        now = utc_now_iso()
        review = SemanticScenarioReview(
            review_id=f"review:{stable_id(catalog.source_key, request.title, now)}",
            source_key=catalog.source_key,
            source_fingerprint=catalog.source_fingerprint,
            catalog_overlay_version=catalog.overlay_version,
            title=request.title,
            questions=request.questions,
            coverage=coverage,
            proposals=proposals,
            created_by_user_id=user_id,
            created_at=now,
            updated_at=now,
        )
        self.catalog_service.catalog_store.save_scenario_review(review)
        return review

    def get(self, *, session_id: str, user_id: int, review_id: str) -> SemanticScenarioReview:
        catalog = self.catalog_service.load_for_session(session_id=session_id, user_id=user_id)
        if catalog is None:
            raise ValueError("Semantic catalog not found")
        review = self.catalog_service.catalog_store.load_scenario_review(catalog.source_key, review_id)
        if review is None:
            raise ValueError("Scenario review not found")
        return review

    def apply(
        self,
        *,
        session_id: str,
        user_id: int,
        review_id: str,
        request: SemanticScenarioApplyRequest,
    ) -> SemanticScenarioApplyResponse:
        catalog = self.catalog_service.load_for_session(session_id=session_id, user_id=user_id)
        if catalog is None:
            raise ValueError("Semantic catalog not found")
        review = self.get(session_id=session_id, user_id=user_id, review_id=review_id)
        if review.review_version != request.expected_review_version:
            raise ValueError("Scenario review changed; reload it before applying")
        fingerprint_changed = (
            catalog.source_fingerprint != request.expected_source_fingerprint
            or catalog.source_fingerprint != review.source_fingerprint
        )
        if fingerprint_changed:
            raise ValueError("Source schema changed; analyze the questions again")
        if catalog.overlay_version != review.catalog_overlay_version:
            raise ValueError("Semantic catalog changed; analyze the questions again")

        selected_ids = set(request.proposal_ids)
        metrics: list[SemanticMetric] = []
        proposal_by_metric_id: dict[str, SemanticScenarioProposal] = {}
        rejected: list[str] = []
        existing_keys = {metric.key for metric in catalog.metrics}
        for proposal in review.proposals:
            if proposal.proposal_id not in selected_ids:
                continue
            if proposal.kind not in {"metric", "metric_update"} or proposal.metric is None:
                proposal.status = "rejected"
                proposal.error = "Only metric proposals can be applied"
                continue
            if proposal.kind == "metric" and proposal.metric.key in existing_keys:
                proposal.status = "failed"
                proposal.error = "A metric with this key already exists"
                rejected.append(f"Metric {proposal.metric.key}: already exists")
                continue
            existing = next(
                (item for item in catalog.metrics if item.key == proposal.target_metric_key),
                None,
            )
            if proposal.kind == "metric_update" and existing is None:
                proposal.status = "failed"
                proposal.error = "The metric selected for update no longer exists"
                rejected.append(f"Metric {proposal.metric.key}: update target no longer exists")
                continue
            metric = SemanticMetric(
                **proposal.metric.model_dump(),
                metric_id=existing.metric_id if existing else f"metric:{proposal.metric.key}",
                created_at=existing.created_at if existing else utc_now_iso(),
            )
            metrics.append(metric)
            proposal_by_metric_id[metric.metric_id] = proposal

        if not metrics:
            raise ValueError("Select at least one valid metric proposal")
        _, validation_rejected = self.catalog_service.apply_generated_overlay(
            session_id=session_id,
            user_id=user_id,
            metrics=metrics,
            replace_metrics=True,
        )
        rejected.extend(validation_rejected)
        rejected_keys = {
            item.split(":", 1)[0].removeprefix("Metric ").strip()
            for item in rejected
            if item.startswith("Metric ")
        }
        applied_keys = []
        for metric in metrics:
            proposal = proposal_by_metric_id[metric.metric_id]
            if metric.key in rejected_keys:
                proposal.status = "failed"
                proposal.error = next(
                    (item for item in rejected if item.startswith(f"Metric {metric.key}:")),
                    "Rejected",
                )
            else:
                proposal.status = "applied"
                applied_keys.append(metric.key)
        review.review_version += 1
        review.catalog_overlay_version += 1
        review.applied_by_user_id = user_id
        review.updated_at = utc_now_iso()
        applied_count = sum(item.status == "applied" for item in review.proposals)
        pending_count = sum(
            item.status == "pending" and item.kind in {"metric", "metric_update"} for item in review.proposals
        )
        review.status = "applied" if applied_count and not pending_count else "partially_applied"
        self.catalog_service.catalog_store.save_scenario_review(review)
        return SemanticScenarioApplyResponse(
            review=review,
            applied_metric_keys=applied_keys,
            rejected_items=rejected,
        )

    def _generate(self, payload: dict[str, Any]) -> GeneratedScenarioReview:
        if self.llm_generate is not None:
            return self.llm_generate(payload)
        llm = make_reasoning_llm(
            provider=getattr(self.settings, "llm_provider", None),
            model=str(self.settings.llm_model),
            base_url=str(self.settings.llm_base_url),
            api_key=str(getattr(self.settings, "llm_api_key", "")),
            enable_thinking=False,
            temperature=0.1,
            max_tokens=max(8192, int(getattr(self.settings, "llm_max_tokens_reasoning", 4096))),
            streaming=False,
            timeout=float(getattr(self.settings, "backend_query_timeout_sec", 180)),
            top_p=float(getattr(self.settings, "llm_top_p", 1.0)),
            top_k=int(getattr(self.settings, "llm_top_k", 0)),
            num_ctx=int(getattr(self.settings, "llm_num_ctx", 0)),
            presence_penalty=0.0,
            chat_template_kwargs_enabled=bool(
                getattr(self.settings, "llm_chat_template_kwargs_enabled", False)
            ),
        )
        structured_llm = llm.with_structured_output(GeneratedScenarioReview)
        messages = [
            ("system", self._system_prompt(payload)),
            ("human", json.dumps(payload, ensure_ascii=False)),
        ]
        generated = structured_llm.invoke(messages)
        localization_issues = self._localization_issues(generated, payload)
        if localization_issues:
            messages.extend(
                [
                    ("assistant", generated.model_dump_json()),
                    (
                        "human",
                        "Regenerate the complete structured response and fix these localization "
                        "violations: " + "; ".join(localization_issues),
                    ),
                ]
            )
            generated = structured_llm.invoke(messages)
        return generated

    @staticmethod
    def _system_prompt(payload: dict[str, Any]) -> str:
        language = dict(payload.get("output_language") or {})
        language_name = str(language.get("name") or "the dominant language of the questions")
        return (
            "You design a conservative, source-agnostic semantic layer from business questions. "
            "Use only supplied tables, columns, example/top values and existing metrics. "
            "Never invent a column. For a simple metric, expr must be the raw column name only; "
            "never put SUM(), AVG(), COUNT(), MIN() or MAX() in expr because agg is a separate field. "
            "Propose metrics but never claim they are applied. When observed discriminator values "
            "represent mutually exclusive scenarios, create separate simple metrics with structured filters. "
            "A total member inside a dimension can double count with its children: report a "
            "rollup_rule proposal when evidence suggests this, and do not hardcode any organization "
            "name. For ratios and percentages, do not SUM; prefer a modeling_gap unless a valid "
            "formula is supported. Never substitute semantically different measures unless supplied "
            "definitions establish their equivalence. "
            "A multi-table question is not covered without validated relationships and compatible "
            "grain. Treat any requested operation unsupported by the supplied semantic catalog as "
            "a missing requirement. Set coverage.status to covered only when every requested "
            "measure, dimension, filter, time grain, join and operation is supported. Otherwise "
            "use partial or gap and enumerate concrete missing_requirements. "
            "A rollup_rule or modeling_gap always means the linked question is not fully covered. "
            "Every metric must cite evidence, map to question indexes, and use qualified existing tables. "
            f"The required output language for every user-facing field is {language_name}. "
            "This includes proposal.title, proposal.rationale, coverage.rationale, "
            "coverage.missing_requirements, warnings, "
            "metric.name, metric.description, and metric.synonyms. Keep database identifiers, SQL "
            "expressions, filter values, and evidence references verbatim; do not translate them. "
            "Keep metric.key as stable lowercase ASCII snake_case regardless of the output language. "
            f"For every proposed metric, provide 2-5 concise business synonyms in {language_name}, "
            "prefer terminology used in its linked questions, and do not copy a whole question as a "
            "synonym. Return concise structured output only."
        )

    @staticmethod
    def _prompt_payload(catalog: SemanticCatalog, request: SemanticScenarioRequest) -> dict[str, Any]:
        columns_by_table: dict[str, list[dict[str, Any]]] = {}
        for column in catalog.columns:
            columns_by_table.setdefault(column.table, []).append(
                {
                    "name": column.name,
                    "dtype": column.dtype,
                    "role": column.semantic_role,
                    "description": column.description,
                    "examples": column.examples[:8],
                    "top_values": column.top_values[:12],
                }
            )
        return {
            "title": request.title,
            "output_language": SemanticScenarioService._output_language_profile(request.questions),
            "questions": [
                {"index": index, "text": question} for index, question in enumerate(request.questions)
            ],
            "source": {
                "type": catalog.source_type,
                "label": catalog.source_label,
                "fingerprint": catalog.source_fingerprint,
            },
            "tables": [
                {
                    "name": table.qualified_name,
                    "description": table.description,
                    "role": table.semantic_role,
                    "grain": table.grain,
                    "row_count": table.row_count,
                    "columns": columns_by_table.get(table.qualified_name, []),
                }
                for table in catalog.tables
                if not table.is_hidden
            ],
            "existing_metrics": [
                {
                    "key": metric.key,
                    "name": metric.name,
                    "description": metric.description,
                    "synonyms": metric.synonyms,
                    "table": metric.base_table,
                    "formula": metric.formula,
                    "filters": [item.model_dump() for item in metric.filters],
                }
                for metric in catalog.metrics
                if metric.is_active
            ],
            "relationships": [
                {
                    "from_table": relationship.from_table,
                    "from_column": relationship.from_column,
                    "to_table": relationship.to_table,
                    "to_column": relationship.to_column,
                    "cardinality": relationship.cardinality,
                }
                for relationship in catalog.relationships
                if relationship.is_active
            ],
        }

    @staticmethod
    def _output_language_profile(questions: list[str]) -> dict[str, str]:
        text = " ".join(str(question or "") for question in questions)
        cyrillic_count = len(_CYRILLIC_RE.findall(text))
        latin_count = len(_LATIN_RE.findall(text))
        if cyrillic_count > latin_count:
            return {"code": "ru", "name": "Russian"}
        if latin_count > cyrillic_count:
            return {"code": "en", "name": "English"}
        return {"code": "auto", "name": "the dominant language of the questions"}

    @staticmethod
    def _localization_issues(
        generated: GeneratedScenarioReview,
        payload: dict[str, Any],
    ) -> list[str]:
        language_code = str(dict(payload.get("output_language") or {}).get("code") or "auto")
        if language_code not in {"ru", "en"}:
            return []

        def uses_wrong_script(value: str) -> bool:
            text = str(value or "").strip()
            if not text:
                return False
            has_cyrillic = bool(_CYRILLIC_RE.search(text))
            has_latin = bool(_LATIN_RE.search(text))
            if language_code == "ru":
                return has_latin and not has_cyrillic
            return has_cyrillic and not has_latin

        issues = [
            f"coverage[{item.question_index}].rationale"
            for item in generated.coverage
            if uses_wrong_script(item.rationale)
        ]
        issues.extend(
            f"coverage[{item.question_index}].missing_requirements"
            for item in generated.coverage
            if any(uses_wrong_script(requirement) for requirement in item.missing_requirements)
        )
        for index, proposal in enumerate(generated.proposals):
            for field_name, value in (
                ("title", proposal.title),
                ("rationale", proposal.rationale),
            ):
                if uses_wrong_script(value):
                    issues.append(f"proposals[{index}].{field_name}")
            metric = proposal.metric
            if metric is None:
                continue
            for field_name, value in (
                ("name", metric.name),
                ("description", metric.description),
            ):
                if uses_wrong_script(value):
                    issues.append(f"proposals[{index}].metric.{field_name}")
            localized_synonyms = [synonym for synonym in metric.synonyms if not uses_wrong_script(synonym)]
            if len(metric.synonyms) < 2 or not localized_synonyms:
                issues.append(f"proposals[{index}].metric.synonyms")
        return list(dict.fromkeys(issues))

    @staticmethod
    def _rewrite_metric_references(
        metric: SemanticMetricCreate,
        aliases: dict[str, str],
    ) -> SemanticMetricCreate:
        if not aliases:
            return metric
        payload = metric.model_dump()
        for field in ("numerator", "denominator"):
            value = payload.get(field)
            if value in aliases:
                payload[field] = aliases[value]
        formula = str(payload.get("formula") or "")
        for old_key, new_key in aliases.items():
            formula = re.sub(rf"\b{re.escape(old_key)}\b", new_key, formula)
        payload["formula"] = formula
        return SemanticMetricCreate.model_validate(payload)

    @staticmethod
    def _metric_signature(
        catalog: SemanticCatalog,
        metric: SemanticMetricCreate | SemanticMetric,
    ) -> tuple:
        table = next(
            (
                item.qualified_name
                for item in catalog.tables
                if metric.base_table in {item.qualified_name, item.table_name}
            ),
            metric.base_table,
        )
        filters = tuple(
            sorted(
                (
                    item.field.rsplit(".", 1)[-1].casefold(),
                    item.op,
                    json.dumps(item.value, ensure_ascii=False, sort_keys=True).casefold(),
                )
                for item in metric.filters
            )
        )
        return (
            metric.type,
            table.casefold(),
            str(metric.expr or "").casefold(),
            str(metric.agg or "").casefold(),
            str(metric.numerator or "").casefold(),
            str(metric.denominator or "").casefold(),
            (re.sub(r"\s+", "", str(metric.formula or "")).casefold() if metric.type == "derived" else ""),
            filters,
        )

    @classmethod
    def _equivalent_existing_metric(
        cls,
        catalog: SemanticCatalog,
        metric: SemanticMetricCreate,
    ) -> SemanticMetric | None:
        signature = cls._metric_signature(catalog, metric)
        return next(
            (
                existing
                for existing in catalog.metrics
                if existing.is_active and cls._metric_signature(catalog, existing) == signature
            ),
            None,
        )

    @staticmethod
    def _metric_update_payload(
        existing: SemanticMetric,
        generated: SemanticMetricCreate,
    ) -> SemanticMetricCreate:
        payload = existing.model_dump(exclude={"metric_id", "created_at", "updated_at"})
        payload["name"] = generated.name
        payload["description"] = generated.description or existing.description
        payload["synonyms"] = list(dict.fromkeys([*existing.synonyms, *generated.synonyms]))
        return SemanticMetricCreate.model_validate(payload)

    @staticmethod
    def _coverage_tables(
        catalog: SemanticCatalog,
        existing_metric_keys: list[str],
        proposals: list[SemanticScenarioProposal],
    ) -> set[str]:
        keys = set(existing_metric_keys)
        tables = {metric.base_table for metric in catalog.metrics if metric.key in keys and metric.is_active}
        tables.update(
            proposal.metric.base_table
            for proposal in proposals
            if proposal.metric is not None and proposal.kind in {"metric", "metric_update"}
        )
        return tables

    @staticmethod
    def _tables_connected(catalog: SemanticCatalog, tables: set[str]) -> bool:
        if len(tables) < 2:
            return True
        graph: dict[str, set[str]] = {}
        for relationship in catalog.relationships:
            if not relationship.is_active:
                continue
            graph.setdefault(relationship.from_table, set()).add(relationship.to_table)
            graph.setdefault(relationship.to_table, set()).add(relationship.from_table)
        first = next(iter(tables))
        seen = {first}
        pending = [first]
        while pending:
            current = pending.pop()
            for neighbour in graph.get(current, set()) - seen:
                seen.add(neighbour)
                pending.append(neighbour)
        return tables <= seen

    @classmethod
    def _requirement_text(cls, requirement: str, questions: list[str]) -> str:
        language = cls._output_language_profile(questions).get("code")
        messages = {
            "relationships": {
                "ru": ("Нужны подтверждённые связи и совместимая гранулярность между таблицами."),
                "en": "Validated relationships and compatible grain between tables are required.",
            },
        }
        return messages[requirement].get(language, messages[requirement]["en"])

    @staticmethod
    def _metric_filter_evidence_error(
        catalog: SemanticCatalog,
        metric: SemanticMetricCreate,
    ) -> str:
        columns = {(column.table, column.name): column for column in catalog.columns}
        for item in metric.filters:
            column = columns.get((metric.base_table, item.field.rsplit(".", 1)[-1]))
            if column is None:
                continue
            observed = {str(value).casefold() for value in [*column.examples, *column.top_values]}
            if not observed:
                continue
            values = item.value if isinstance(item.value, list) else [item.value]
            if item.op in {"=", "!=", "in", "not_in"}:
                unknown = [str(value) for value in values if str(value).casefold() not in observed]
                if unknown:
                    return f"filter value is not present in the column profile: {', '.join(unknown)}"
            if item.op == "starts_with" and not any(
                value.startswith(str(item.value).casefold()) for value in observed
            ):
                return "filter prefix is not present in the column profile"
        return ""

    @staticmethod
    def _normalize_simple_metric_expression(
        metric: SemanticMetricCreate,
    ) -> tuple[SemanticMetricCreate, str]:
        if metric.type != "simple" or not metric.expr:
            return metric, ""
        match = _AGGREGATED_EXPR_RE.fullmatch(metric.expr)
        if match is None:
            return metric, ""
        wrapped_agg = "count_distinct" if match.group("distinct_column") else str(match.group("agg")).lower()
        raw_identifier = str(match.group("distinct_column") or match.group("column")).replace('"', "")
        identifier_parts = raw_identifier.split(".")
        raw_column = identifier_parts[-1]
        if len(identifier_parts) > 1:
            qualifier = ".".join(identifier_parts[:-1])
            allowed_qualifiers = {metric.base_table, metric.base_table.rsplit(".", 1)[-1]}
            if qualifier not in allowed_qualifiers:
                raise ValueError(f"expr references {qualifier}, but base_table is {metric.base_table}")
        if metric.agg and metric.agg != wrapped_agg:
            raise ValueError(f"conflicting aggregations: expr uses {wrapped_agg}, agg is {metric.agg}")
        payload = metric.model_dump()
        payload.update({"expr": raw_column, "agg": wrapped_agg, "formula": ""})
        normalized = SemanticMetricCreate.model_validate(payload)
        return (
            normalized,
            f"Нормализовано выражение метрики: {metric.expr} → {raw_column} ({wrapped_agg}).",
        )
