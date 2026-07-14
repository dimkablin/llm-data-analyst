from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from backend.data_access.semantic_models import (
    SemanticCatalog,
    SemanticContextResult,
    SemanticMetric,
    SemanticSearchResultItem,
    SemanticTable,
    SemanticTerm,
    utc_now_iso,
)
from backend.sessions.session_store import SessionStore

logger = logging.getLogger(__name__)


@dataclass
class SemanticContextBuilder:
    store: SessionStore
    vector_store: object | None = None
    catalog_service: object | None = None
    top_k: int = 8

    def build(self, *, session_id: str, user_id: int, query: str) -> SemanticContextResult:
        catalog = self._load_catalog(session_id=session_id, user_id=user_id)
        if catalog is None:
            return SemanticContextResult(status="empty")

        if self.catalog_service is None:
            self._mark_stale_if_needed(session_id, catalog)
        items = self._vector_search(catalog=catalog, query=query)
        status = catalog.status
        if not items:
            items = self._lexical_search(catalog=catalog, query=query, top_k=self.top_k)
            if (
                status == "ready"
                and self.vector_store is not None
                and getattr(self.vector_store, "enabled", False)
            ):
                status = "degraded"
        prompt = format_semantic_context_prompt(catalog, items)
        hints = build_semantic_hints(catalog, items)
        return SemanticContextResult(status=status, prompt=prompt, items=items, hints=hints)

    def _mark_stale_if_needed(self, session_id: str, catalog: SemanticCatalog) -> None:
        snapshot = self.store.load_data_catalog(session_id)
        if snapshot is None:
            return
        if snapshot.source_fingerprint == catalog.source_fingerprint:
            return
        catalog.status = "stale"
        catalog.error = "Data catalog source fingerprint changed."
        catalog.updated_at = utc_now_iso()
        self._save_runtime_status(catalog)

    def _vector_search(self, *, catalog: SemanticCatalog, query: str) -> list[SemanticSearchResultItem]:
        if self.vector_store is None or not getattr(self.vector_store, "enabled", False):
            return []
        try:
            return list(
                self.vector_store.search(catalog=catalog, query=query, top_k=max(1, int(self.top_k)))
            )
        except Exception as exc:
            catalog.status = "degraded"
            catalog.error = f"Qdrant search failed: {exc}"
            catalog.updated_at = utc_now_iso()
            self._save_runtime_status(catalog)
            logger.warning("Semantic catalog search failed: %s", exc)
            return []

    def _load_catalog(self, *, session_id: str, user_id: int) -> SemanticCatalog | None:
        if self.catalog_service is not None:
            loader = getattr(self.catalog_service, "load_for_session", None)
            if callable(loader):
                return loader(session_id=session_id, user_id=user_id)
        catalog = self.store.load_semantic_catalog(session_id)
        if catalog is None or catalog.user_id != user_id:
            return None
        return catalog

    def _save_runtime_status(self, catalog: SemanticCatalog) -> None:
        if self.catalog_service is not None:
            saver = getattr(self.catalog_service, "save_runtime_status", None)
            if callable(saver):
                saver(catalog)
                return
        self.store.save_semantic_catalog(catalog.session_id, catalog)

    @staticmethod
    def _lexical_search(
        *,
        catalog: SemanticCatalog,
        query: str,
        top_k: int,
    ) -> list[SemanticSearchResultItem]:
        tokens = _tokens(query)
        if not tokens:
            return []
        scored: list[tuple[float, SemanticSearchResultItem]] = []
        for entity_type, entity_id, text in _searchable_entities(catalog):
            words = _tokens(text)
            if not words:
                continue
            overlap = tokens & words
            if not overlap:
                continue
            score = len(overlap) / max(len(tokens), 1)
            scored.append(
                (
                    score,
                    SemanticSearchResultItem(
                        entity_type=entity_type,
                        entity_id=entity_id,
                        score=score,
                        payload={"source": "lexical"},
                    ),
                )
            )
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item for _score, item in scored[: max(1, int(top_k))]]


def format_semantic_context_prompt(
    catalog: SemanticCatalog,
    items: list[SemanticSearchResultItem],
) -> str:
    selected = _selected_entities(catalog, items)
    lines = [
        "SEMANTIC DATA CONTEXT",
        f"status: {catalog.status}",
        f"catalog_id: {catalog.catalog_id}",
        f"source_fingerprint: {catalog.source_fingerprint}",
    ]
    if catalog.error:
        lines.append(f"note: {catalog.error}")

    visible_tables = [table for table in catalog.tables if not table.is_hidden]
    tables = selected["tables"] or visible_tables[: min(5, len(visible_tables))]
    if tables:
        lines.append("tables:")
        for table in tables[:8]:
            table_columns = [
                column
                for column in catalog.columns
                if column.table == table.qualified_name and not column.is_hidden
            ][:16]
            columns_text = ", ".join(
                f"{column.name}({column.semantic_role})" for column in table_columns
            )
            lines.append(
                f"- {table.qualified_name}: role={table.semantic_role}; columns={columns_text}"
            )

    metrics = selected["metrics"] or [metric for metric in catalog.metrics if metric.is_active][:5]
    if metrics:
        lines.append("metrics:")
        for metric in metrics[:8]:
            parts = [_metric_prompt_text(metric)]
            if metric.synonyms:
                parts.append("synonyms=" + ", ".join(metric.synonyms))
            lines.append("; ".join(parts))

    relationships = selected["relationships"] or catalog.relationships[:5]
    if relationships:
        lines.append("relationships:")
        for rel in relationships[:8]:
            lines.append(f"- {rel.from_table}.{rel.from_column} -> {rel.to_table}.{rel.to_column}")
    terms = selected["terms"] or [term for term in catalog.terms if term.is_active][:5]
    if terms:
        lines.append("terms:")
        for term in terms[:8]:
            parts = [f"- {term.name}: {term.description}".strip()]
            if term.synonyms:
                parts.append("synonyms=" + ", ".join(term.synonyms))
            if term.entity_refs:
                parts.append("refs=" + ", ".join(term.entity_refs))
            lines.append("; ".join(parts))
    return "\n".join(lines).strip()


def build_semantic_hints(
    catalog: SemanticCatalog,
    items: list[SemanticSearchResultItem],
) -> dict[str, object]:
    selected = _selected_entities(catalog, items)
    return {
        "status": catalog.status,
        "catalog_id": catalog.catalog_id,
        "source_fingerprint": catalog.source_fingerprint,
        "tables": [table.model_dump() for table in selected["tables"]],
        "columns": [column.model_dump() for column in selected["columns"]],
        "metrics": [metric.model_dump() for metric in selected["metrics"]],
        "relationships": [rel.model_dump() for rel in selected["relationships"]],
        "terms": [term.model_dump() for term in selected["terms"]],
        "items": [item.model_dump() for item in items],
        "catalog": catalog.model_dump(exclude={"columns": {"__all__": {"examples"}}}),
    }


def _selected_entities(
    catalog: SemanticCatalog,
    items: list[SemanticSearchResultItem],
) -> dict[str, list]:
    tables_by_id = {table.table_id: table for table in catalog.tables}
    tables_by_name = {table.qualified_name: table for table in catalog.tables}
    columns_by_id = {column.column_id: column for column in catalog.columns}
    metrics_by_id = {metric.metric_id: metric for metric in catalog.metrics}
    relationships_by_id = {rel.relationship_id: rel for rel in catalog.relationships}
    terms_by_id = {term.term_id: term for term in catalog.terms}

    tables: list[SemanticTable] = []
    columns = []
    metrics: list[SemanticMetric] = []
    relationships = []
    terms: list[SemanticTerm] = []

    def add_table(name_or_id: str) -> None:
        table = tables_by_id.get(name_or_id) or tables_by_name.get(name_or_id)
        if table is not None and table not in tables:
            tables.append(table)

    for item in items:
        if item.entity_type == "table":
            add_table(item.entity_id)
        elif item.entity_type == "column":
            column = columns_by_id.get(item.entity_id)
            if column is not None:
                columns.append(column)
                add_table(column.table)
        elif item.entity_type == "metric":
            metric = metrics_by_id.get(item.entity_id)
            if metric is not None:
                metrics.append(metric)
                add_table(metric.base_table)
        elif item.entity_type == "relationship":
            rel = relationships_by_id.get(item.entity_id)
            if rel is not None:
                relationships.append(rel)
                add_table(rel.from_table)
                add_table(rel.to_table)
        elif item.entity_type == "term":
            term = terms_by_id.get(item.entity_id)
            if term is not None and term not in terms:
                terms.append(term)

    for metric in metrics:
        table_name = metric.base_table
        metric_columns = [
            metric.expr,
            metric.default_time_dimension,
            *list(metric.allowed_dimensions or []),
        ]
        for column in catalog.columns:
            if column.table == table_name and column.name in metric_columns and not column.is_hidden:
                if column not in columns:
                    columns.append(column)

    return {
        "tables": tables,
        "columns": columns,
        "metrics": metrics,
        "relationships": relationships,
        "terms": terms,
    }


def _searchable_entities(catalog: SemanticCatalog) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for table in catalog.tables:
        if table.is_hidden:
            continue
        rows.append(
            (
                "table",
                table.table_id,
                " ".join([table.qualified_name, table.table_name, table.description]),
            )
        )
    for column in catalog.columns:
        if column.is_hidden:
            continue
        rows.append(
            (
                "column",
                column.column_id,
                " ".join([column.table, column.name, column.description, column.semantic_role]),
            )
        )
    for metric in catalog.metrics:
        rows.append(
            (
                "metric",
                metric.metric_id,
                " ".join(
                    [
                        metric.key,
                        metric.name,
                        metric.description,
                        metric.type,
                        metric.formula,
                        metric.base_table,
                        metric.expr or "",
                        metric.agg or "",
                        metric.numerator or "",
                        metric.denominator or "",
                        " ".join(metric.allowed_dimensions),
                        " ".join(metric.synonyms),
                    ]
                ),
            )
        )
    for rel in catalog.relationships:
        rows.append(
            (
                "relationship",
                rel.relationship_id,
                f"{rel.from_table} {rel.from_column} {rel.to_table} {rel.to_column}",
            )
        )
    for term in catalog.terms:
        rows.append(("term", term.term_id, " ".join([term.name, term.description, " ".join(term.synonyms)])))
    return rows


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[\w]+", str(text or ""), flags=re.UNICODE)
        if len(token) >= 2
    }


def _metric_prompt_text(metric: SemanticMetric) -> str:
    parts = [
        f"- Metric: {metric.key} / {metric.name}",
        f"type={metric.type}",
        f"table={metric.base_table}",
    ]
    if metric.agg:
        parts.append(f"agg={metric.agg}")
    if metric.expr:
        parts.append(f"expr={metric.expr}")
    if metric.default_time_dimension:
        parts.append(f"time={metric.default_time_dimension}")
    if metric.allowed_dimensions:
        parts.append("dimensions=" + ", ".join(metric.allowed_dimensions))
    if metric.formula:
        parts.append(f"formula={metric.formula}")
    return "; ".join(parts)
