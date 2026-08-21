from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.data_access.semantic_models import (
    SemanticCatalog,
    SemanticColumn,
    SemanticMetric,
    SemanticMetricCreate,
    SemanticTable,
)
from backend.data_access.semantic_scenario_models import (
    SemanticScenarioApplyRequest,
    SemanticScenarioRequest,
)
from backend.data_access.semantic_scenario_service import (
    GeneratedScenarioCoverage,
    GeneratedScenarioProposal,
    GeneratedScenarioReview,
    SemanticScenarioService,
)


class MemoryReviewStore:
    def __init__(self) -> None:
        self.reviews = {}

    def save_scenario_review(self, review) -> None:
        self.reviews[(review.source_key, review.review_id)] = review.model_copy(deep=True)

    def load_scenario_review(self, source_key, review_id):
        review = self.reviews.get((source_key, review_id))
        return review.model_copy(deep=True) if review else None


class FakeCatalogService:
    def __init__(self, catalog: SemanticCatalog) -> None:
        self.catalog = catalog
        self.catalog_store = MemoryReviewStore()
        self.applied = []

    def load_for_session(self, **_kwargs):
        return self.catalog

    def validate_metric_candidate(self, catalog, metric) -> None:
        columns = {(item.table, item.name) for item in catalog.columns}
        if metric.type == "simple" and (metric.base_table, metric.expr) not in columns:
            raise ValueError("unknown column")
        known_metric_keys = {item.key for item in catalog.metrics if item.metric_id != metric.metric_id}
        for ref in (metric.numerator, metric.denominator):
            if ref and ref not in known_metric_keys:
                raise ValueError("unknown metric reference")

    def apply_generated_overlay(self, *, metrics, **_kwargs):
        self.applied.extend(metrics)
        for metric in metrics:
            self.catalog.metrics = [
                item for item in self.catalog.metrics if item.metric_id != metric.metric_id
            ]
            self.catalog.metrics.append(metric)
        self.catalog.overlay_version += 1
        return self.catalog, []


def _catalog() -> SemanticCatalog:
    return SemanticCatalog(
        catalog_id="catalog:demo",
        source_key="db_connection:demo",
        source_type="db_connection",
        source_fingerprint="schema-v1",
        status="ready",
        tables=[
            SemanticTable(
                table_id="table:sales",
                qualified_name="public.sales",
                table_name="sales",
                source_kind="postgresql",
            )
        ],
        columns=[
            SemanticColumn(
                column_id="column:sales.amount",
                table="public.sales",
                name="amount",
                dtype="numeric",
            ),
            SemanticColumn(
                column_id="column:sales.kind",
                table="public.sales",
                name="kind",
                dtype="text",
                examples=["fact", "plan"],
            ),
        ],
    )


def _generated_review(_payload) -> GeneratedScenarioReview:
    return GeneratedScenarioReview(
        coverage=[
            GeneratedScenarioCoverage(
                question_index=0,
                status="partial",
                rationale="Нужна метрика факта.",
            )
        ],
        proposals=[
            GeneratedScenarioProposal(
                title="Фактическая выручка",
                rationale="Вопрос просит динамику факта.",
                confidence=0.9,
                question_indexes=[0],
                evidence=["public.sales.amount; kind examples: fact, plan"],
                metric=SemanticMetricCreate(
                    key="actual_revenue",
                    name="Фактическая выручка",
                    description="Сумма фактической выручки.",
                    synonyms=["фактическая выручка", "выручка по факту"],
                    base_table="public.sales",
                    expr="amount",
                    agg="sum",
                    filters=[{"field": "kind", "op": "=", "value": "fact"}],
                ),
            )
        ],
    )


def test_scenario_metrics_remain_drafts_until_confirmed() -> None:
    catalog_service = FakeCatalogService(_catalog())
    service = SemanticScenarioService(
        catalog_service=catalog_service,
        settings=SimpleNamespace(),
        llm_generate=_generated_review,
    )

    review = service.analyze(
        session_id="session-1",
        user_id=7,
        request=SemanticScenarioRequest(questions=["Покажи динамику фактической выручки"]),
    )

    assert catalog_service.applied == []
    assert review.proposals[0].metric.filters[0].value == "fact"
    assert review.proposals[0].metric.description == "Сумма фактической выручки."
    assert review.proposals[0].metric.synonyms == ["фактическая выручка", "выручка по факту"]
    result = service.apply(
        session_id="session-1",
        user_id=7,
        review_id=review.review_id,
        request=SemanticScenarioApplyRequest(
            proposal_ids=[review.proposals[0].proposal_id],
            expected_review_version=review.review_version,
            expected_source_fingerprint=review.source_fingerprint,
        ),
    )

    assert result.applied_metric_keys == ["actual_revenue"]
    assert isinstance(catalog_service.applied[0], SemanticMetric)
    assert result.review.proposals[0].status == "applied"


def test_scenario_apply_rejects_changed_catalog() -> None:
    catalog_service = FakeCatalogService(_catalog())
    service = SemanticScenarioService(
        catalog_service=catalog_service,
        settings=SimpleNamespace(),
        llm_generate=_generated_review,
    )
    review = service.analyze(
        session_id="session-1",
        user_id=7,
        request=SemanticScenarioRequest(questions=["Покажи динамику фактической выручки"]),
    )
    catalog_service.catalog.overlay_version += 1

    with pytest.raises(ValueError, match="catalog changed"):
        service.apply(
            session_id="session-1",
            user_id=7,
            review_id=review.review_id,
            request=SemanticScenarioApplyRequest(
                proposal_ids=[review.proposals[0].proposal_id],
                expected_review_version=review.review_version,
                expected_source_fingerprint=review.source_fingerprint,
            ),
        )


def test_scenario_rejects_unobserved_filter_value() -> None:
    def generated_with_invented_value(_payload):
        draft = _generated_review(_payload)
        draft.proposals[0].metric.filters[0].value = "budget"
        return draft

    catalog_service = FakeCatalogService(_catalog())
    service = SemanticScenarioService(
        catalog_service=catalog_service,
        settings=SimpleNamespace(),
        llm_generate=generated_with_invented_value,
    )

    review = service.analyze(
        session_id="session-1",
        user_id=7,
        request=SemanticScenarioRequest(questions=["Покажи динамику фактической выручки"]),
    )

    assert review.proposals[0].kind == "modeling_gap"
    assert review.proposals[0].metric is None
    assert "not present" in review.proposals[0].warnings[0]


@pytest.mark.parametrize(
    ("expr", "agg", "expected_expr", "expected_agg"),
    [
        ("SUM(amount)", "sum", "amount", "sum"),
        ('AVG("amount")', "avg", "amount", "avg"),
        ("COUNT(DISTINCT public.sales.amount)", "count_distinct", "amount", "count_distinct"),
        ("COUNT_DISTINCT(amount)", "count_distinct", "amount", "count_distinct"),
    ],
)
def test_normalizes_wrapped_simple_metric_expression(
    expr: str,
    agg: str,
    expected_expr: str,
    expected_agg: str,
) -> None:
    metric = SemanticMetricCreate(
        key="generated_metric",
        name="Generated metric",
        base_table="public.sales",
        expr=expr,
        agg=agg,
    )

    normalized, note = SemanticScenarioService._normalize_simple_metric_expression(metric)

    assert normalized.expr == expected_expr
    assert normalized.agg == expected_agg
    assert "Нормализовано" in note


def test_rejects_conflicting_wrapped_aggregation() -> None:
    metric = SemanticMetricCreate(
        key="generated_metric",
        name="Generated metric",
        base_table="public.sales",
        expr="SUM(amount)",
        agg="avg",
    )

    with pytest.raises(ValueError, match="conflicting aggregations"):
        SemanticScenarioService._normalize_simple_metric_expression(metric)


def test_rejects_wrapped_expression_from_another_table() -> None:
    metric = SemanticMetricCreate(
        key="generated_metric",
        name="Generated metric",
        base_table="public.sales",
        expr="SUM(public.budget.amount)",
        agg="sum",
    )

    with pytest.raises(ValueError, match="base_table"):
        SemanticScenarioService._normalize_simple_metric_expression(metric)


def test_wrapped_generated_metric_remains_selectable() -> None:
    def generated_with_wrapped_expression(_payload):
        draft = _generated_review(_payload)
        payload = draft.proposals[0].metric.model_dump()
        payload.update({"expr": "SUM(amount)", "formula": ""})
        draft.proposals[0].metric = SemanticMetricCreate.model_validate(payload)
        return draft

    catalog_service = FakeCatalogService(_catalog())
    service = SemanticScenarioService(
        catalog_service=catalog_service,
        settings=SimpleNamespace(),
        llm_generate=generated_with_wrapped_expression,
    )

    review = service.analyze(
        session_id="session-1",
        user_id=7,
        request=SemanticScenarioRequest(questions=["Покажи динамику фактической выручки"]),
    )

    assert review.proposals[0].kind == "metric"
    assert review.proposals[0].metric.expr == "amount"
    assert any("Нормализовано" in item for item in review.proposals[0].warnings)


def test_russian_questions_request_localized_metric_metadata() -> None:
    request = SemanticScenarioRequest(
        questions=[
            "Проанализируй динамику выручки по каналам за доступный период.",
            "Сравни валовую и чистую выручку.",
        ]
    )

    payload = SemanticScenarioService._prompt_payload(_catalog(), request)
    prompt = SemanticScenarioService._system_prompt(payload)

    assert payload["output_language"] == {"code": "ru", "name": "Russian"}
    assert "required output language for every user-facing field is Russian" in prompt
    assert "metric.name, metric.description, and metric.synonyms" in prompt
    assert "2-5 concise business synonyms in Russian" in prompt
    assert "metric.key as stable lowercase ASCII snake_case" in prompt


def test_english_questions_keep_english_output_language() -> None:
    request = SemanticScenarioRequest(
        questions=["Show the monthly actual revenue trend by branch."],
    )

    payload = SemanticScenarioService._prompt_payload(_catalog(), request)

    assert payload["output_language"] == {"code": "en", "name": "English"}


def test_localization_check_rejects_english_metric_metadata_for_russian_questions() -> None:
    generated = GeneratedScenarioReview(
        coverage=[
            GeneratedScenarioCoverage(
                question_index=0,
                status="partial",
                rationale="A gross revenue metric is required.",
            )
        ],
        proposals=[
            GeneratedScenarioProposal(
                title="Gross Revenue",
                rationale="Revenue before deductions.",
                question_indexes=[0],
                metric=SemanticMetricCreate(
                    key="gross_revenue",
                    name="Gross Revenue",
                    description="Revenue before deductions.",
                    synonyms=["sales revenue", "gross sales"],
                    base_table="public.sales",
                    expr="amount",
                    agg="sum",
                ),
            )
        ],
    )
    payload = {"output_language": {"code": "ru", "name": "Russian"}}

    issues = SemanticScenarioService._localization_issues(generated, payload)

    assert "coverage[0].rationale" in issues
    assert "proposals[0].title" in issues
    assert "proposals[0].metric.name" in issues
    assert "proposals[0].metric.synonyms" in issues


def test_localization_check_accepts_russian_metric_metadata() -> None:
    generated = _generated_review({})
    payload = {"output_language": {"code": "ru", "name": "Russian"}}

    assert SemanticScenarioService._localization_issues(generated, payload) == []


def test_equivalent_metric_is_proposed_as_metadata_update_and_does_not_duplicate() -> None:
    catalog = _catalog()
    catalog.metrics = [
        SemanticMetric(
            metric_id="metric:long_generated_key",
            key="long_generated_key",
            name="Old name",
            description="Old description",
            synonyms=["old synonym"],
            base_table="public.sales",
            expr="amount",
            agg="sum",
            filters=[{"field": "kind", "op": "=", "value": "fact"}],
        )
    ]
    catalog_service = FakeCatalogService(catalog)
    service = SemanticScenarioService(
        catalog_service=catalog_service,
        settings=SimpleNamespace(),
        llm_generate=_generated_review,
    )

    review = service.analyze(
        session_id="session-1",
        user_id=7,
        request=SemanticScenarioRequest(questions=["Покажи фактическую выручку"]),
    )

    proposal = review.proposals[0]
    assert proposal.kind == "metric_update"
    assert proposal.target_metric_key == "long_generated_key"
    assert proposal.metric.key == "long_generated_key"
    assert proposal.metric.name == "Фактическая выручка"
    assert proposal.metric.expr == "amount"
    assert review.coverage[0].existing_metric_keys == ["long_generated_key"]

    result = service.apply(
        session_id="session-1",
        user_id=7,
        review_id=review.review_id,
        request=SemanticScenarioApplyRequest(
            proposal_ids=[proposal.proposal_id],
            expected_review_version=review.review_version,
            expected_source_fingerprint=review.source_fingerprint,
        ),
    )

    assert result.applied_metric_keys == ["long_generated_key"]
    assert len(catalog_service.catalog.metrics) == 1
    assert catalog_service.catalog.metrics[0].name == "Фактическая выручка"
    assert "old synonym" in catalog_service.catalog.metrics[0].synonyms


def test_ratio_may_reference_metrics_proposed_earlier_in_same_review() -> None:
    def generated(_payload):
        return GeneratedScenarioReview(
            coverage=[GeneratedScenarioCoverage(question_index=0, status="partial")],
            proposals=[
                GeneratedScenarioProposal(
                    title="Числитель",
                    question_indexes=[0],
                    metric=SemanticMetricCreate(
                        key="actual_amount",
                        name="Фактическая сумма",
                        base_table="public.sales",
                        expr="amount",
                        agg="sum",
                        filters=[{"field": "kind", "op": "=", "value": "fact"}],
                    ),
                ),
                GeneratedScenarioProposal(
                    title="Знаменатель",
                    question_indexes=[0],
                    metric=SemanticMetricCreate(
                        key="plan_amount",
                        name="Плановая сумма",
                        base_table="public.sales",
                        expr="amount",
                        agg="sum",
                        filters=[{"field": "kind", "op": "=", "value": "plan"}],
                    ),
                ),
                GeneratedScenarioProposal(
                    title="Выполнение плана",
                    question_indexes=[0],
                    metric=SemanticMetricCreate(
                        key="plan_attainment",
                        name="Выполнение плана",
                        type="ratio",
                        base_table="public.sales",
                        numerator="actual_amount",
                        denominator="plan_amount",
                        format="percent",
                    ),
                ),
            ],
        )

    service = SemanticScenarioService(
        catalog_service=FakeCatalogService(_catalog()),
        settings=SimpleNamespace(),
        llm_generate=generated,
    )
    review = service.analyze(
        session_id="session-1",
        user_id=7,
        request=SemanticScenarioRequest(questions=["Сравни факт с планом"]),
    )

    assert [proposal.kind for proposal in review.proposals] == ["metric", "metric", "metric"]
    assert review.proposals[2].metric.numerator == "actual_amount"
    assert review.proposals[2].metric.denominator == "plan_amount"


def test_rollup_rule_prevents_false_covered_status() -> None:
    def generated(_payload):
        return GeneratedScenarioReview(
            coverage=[GeneratedScenarioCoverage(question_index=0, status="covered")],
            proposals=[
                GeneratedScenarioProposal(
                    kind="rollup_rule",
                    title="Правило итоговой строки",
                    question_indexes=[0],
                )
            ],
        )

    service = SemanticScenarioService(
        catalog_service=FakeCatalogService(_catalog()),
        settings=SimpleNamespace(),
        llm_generate=generated,
    )
    review = service.analyze(
        session_id="session-1",
        user_id=7,
        request=SemanticScenarioRequest(questions=["Покажи итоги без двойного счёта"]),
    )

    assert review.coverage[0].status == "partial"


def test_generated_missing_operation_requirement_is_preserved() -> None:
    catalog = _catalog()
    catalog.metrics = [
        SemanticMetric(
            metric_id="metric:revenue",
            key="revenue",
            name="Выручка",
            base_table="public.sales",
            expr="amount",
            agg="sum",
        )
    ]

    def generated(_payload):
        return GeneratedScenarioReview(
            coverage=[
                GeneratedScenarioCoverage(
                    question_index=0,
                    status="partial",
                    existing_metric_keys=["revenue"],
                    missing_requirements=["Нужен настроенный инструмент прогнозирования."],
                )
            ]
        )

    service = SemanticScenarioService(
        catalog_service=FakeCatalogService(catalog),
        settings=SimpleNamespace(),
        llm_generate=generated,
    )
    review = service.analyze(
        session_id="session-1",
        user_id=7,
        request=SemanticScenarioRequest(questions=["Построй прогноз выручки на три месяца"]),
    )

    assert review.coverage[0].status == "partial"
    assert any("прогноз" in item.casefold() for item in review.coverage[0].missing_requirements)


def test_multi_table_question_requires_connected_relationships() -> None:
    catalog = _catalog()
    catalog.tables.append(
        SemanticTable(
            table_id="table:feedback",
            qualified_name="public.feedback",
            table_name="feedback",
            source_kind="postgresql",
        )
    )
    catalog.columns.append(
        SemanticColumn(
            column_id="column:feedback.count",
            table="public.feedback",
            name="count",
            dtype="numeric",
        )
    )
    catalog.metrics = [
        SemanticMetric(
            metric_id="metric:revenue",
            key="revenue",
            name="Выручка",
            base_table="public.sales",
            expr="amount",
            agg="sum",
        ),
        SemanticMetric(
            metric_id="metric:complaints",
            key="complaints",
            name="Обращения",
            base_table="public.feedback",
            expr="count",
            agg="sum",
        ),
    ]

    def generated(_payload):
        return GeneratedScenarioReview(
            coverage=[
                GeneratedScenarioCoverage(
                    question_index=0,
                    status="covered",
                    existing_metric_keys=["revenue", "complaints"],
                )
            ]
        )

    service = SemanticScenarioService(
        catalog_service=FakeCatalogService(catalog),
        settings=SimpleNamespace(),
        llm_generate=generated,
    )
    review = service.analyze(
        session_id="session-1",
        user_id=7,
        request=SemanticScenarioRequest(questions=["Сопоставь выручку и обращения по филиалам"]),
    )

    assert review.coverage[0].status == "partial"
    assert any("связ" in item.casefold() for item in review.coverage[0].missing_requirements)
