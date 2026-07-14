from __future__ import annotations

import re

from backend.data_access.semantic_models import (
    SemanticCatalog,
    SemanticValidationIssue,
    SemanticValidationResult,
)

_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


def validate_semantic_catalog(catalog: SemanticCatalog) -> SemanticValidationResult:
    errors: list[SemanticValidationIssue] = []
    warnings: list[SemanticValidationIssue] = []
    metric_keys = {metric.key for metric in catalog.metrics}
    fact_names = {fact.name for fact in catalog.facts}
    dimension_names = {dimension.name for dimension in catalog.dimensions}
    table_names = {table.qualified_name for table in catalog.tables}
    column_keys = {(column.table, column.name) for column in catalog.columns}

    for metric in catalog.metrics:
        if table_names and metric.base_table not in table_names:
            errors.append(
                _issue("unknown_table", f"Unknown metric table: {metric.base_table}", "metric", metric.metric_id)
            )
        if (
            metric.type == "simple"
            and metric.expr
            and fact_names
            and metric.expr not in fact_names
            and (metric.base_table, metric.expr) not in column_keys
        ):
            errors.append(
                _issue("unknown_metric_expr", f"Unknown metric expression: {metric.expr}", "metric", metric.metric_id)
            )
        for ref in [metric.numerator, metric.denominator]:
            if ref and ref not in metric_keys:
                errors.append(
                    _issue("unknown_metric_reference", f"Unknown metric reference: {ref}", "metric", metric.metric_id)
                )
        for dim in metric.allowed_dimensions:
            if dimension_names and dim not in dimension_names:
                warnings.append(
                    _issue("unknown_dimension", f"Unknown allowed dimension: {dim}", "metric", metric.metric_id)
                )

    errors.extend(_metric_cycle_errors(catalog))
    errors.extend(_relationship_errors(catalog, column_keys))
    score = max(0.0, 1.0 - (len(errors) * 0.2) - (len(warnings) * 0.05))
    return SemanticValidationResult(errors=errors, warnings=warnings, quality_score=round(score, 2))


def _metric_cycle_errors(catalog: SemanticCatalog) -> list[SemanticValidationIssue]:
    refs = {metric.key: _metric_refs(metric) for metric in catalog.metrics}
    visiting: set[str] = set()
    visited: set[str] = set()
    errors: list[SemanticValidationIssue] = []

    def visit(key: str, path: list[str]) -> None:
        if key in visited:
            return
        if key in visiting:
            cycle = " -> ".join([*path, key])
            errors.append(_issue("metric_cycle", f"Metric cycle detected: {cycle}", "metric", f"metric:{key}"))
            return
        visiting.add(key)
        for ref in refs.get(key, set()):
            if ref in refs:
                visit(ref, [*path, key])
        visiting.remove(key)
        visited.add(key)

    for key in refs:
        visit(key, [])
    return errors


def _relationship_errors(
    catalog: SemanticCatalog,
    column_keys: set[tuple[str, str]],
) -> list[SemanticValidationIssue]:
    errors: list[SemanticValidationIssue] = []
    for rel in catalog.relationships:
        if rel.cardinality == "many_to_many":
            errors.append(
                _issue(
                    "unsupported_relationship_cardinality",
                    "many_to_many relationships are not supported yet",
                    "relationship",
                    rel.relationship_id,
                )
            )
        if column_keys and (rel.from_table, rel.from_column) not in column_keys:
            errors.append(
                _issue(
                    "unknown_relationship_column",
                    f"Unknown relationship column: {rel.from_table}.{rel.from_column}",
                    "relationship",
                    rel.relationship_id,
                )
            )
        if column_keys and (rel.to_table, rel.to_column) not in column_keys:
            errors.append(
                _issue(
                    "unknown_relationship_column",
                    f"Unknown relationship column: {rel.to_table}.{rel.to_column}",
                    "relationship",
                    rel.relationship_id,
                )
            )
        if rel.from_table == rel.to_table and rel.from_column == rel.to_column:
            errors.append(
                _issue("self_relationship", "Relationship points to the same column", "relationship", rel.relationship_id)
            )
        safety_error = relationship_safety_error(catalog, rel)
        if safety_error:
            errors.append(_issue("unsafe_relationship", safety_error, "relationship", rel.relationship_id))
    return errors


def relationship_safety_error(catalog: SemanticCatalog, rel) -> str:
    if not getattr(rel, "is_active", True):
        return ""
    if rel.cardinality not in {"many_to_one", "one_to_one"}:
        return f"Unsupported safe cardinality: {rel.cardinality}"
    entities = {
        (entity.table, entity.expr): entity.type
        for entity in catalog.entities
        if entity.is_active
    }
    from_kind = entities.get((rel.from_table, rel.from_column))
    to_kind = entities.get((rel.to_table, rel.to_column))
    if rel.cardinality == "many_to_one":
        if from_kind != "foreign":
            return f"{rel.from_table}.{rel.from_column} is not modeled as a foreign entity"
        if to_kind not in {"primary", "unique", "natural"}:
            return f"{rel.to_table}.{rel.to_column} is not modeled as a unique target entity"
    if rel.cardinality == "one_to_one":
        if from_kind not in {"primary", "unique", "natural"} or to_kind not in {"primary", "unique", "natural"}:
            return "one_to_one relationship requires unique entities on both sides"
    return ""


def _metric_refs(metric) -> set[str]:
    refs = {str(ref) for ref in [metric.numerator, metric.denominator] if str(ref or "").strip()}
    if metric.type == "derived":
        refs.update(_IDENT_RE.findall(metric.formula or ""))
    return refs


def _issue(code: str, message: str, object_type: str, object_id: str) -> SemanticValidationIssue:
    return SemanticValidationIssue(code=code, message=message, object_type=object_type, object_id=object_id)
