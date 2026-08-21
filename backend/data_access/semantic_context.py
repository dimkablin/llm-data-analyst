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

    def build(
        self,
        *,
        session_id: str,
        user_id: int,
        query: str,
    ) -> SemanticContextResult:
        catalog = self._load_catalog(session_id=session_id, user_id=user_id)
        if catalog is None:
            return SemanticContextResult(status="empty")

        if self.catalog_service is None:
            self._mark_stale_if_needed(session_id, catalog)
        items, vector_error = self._vector_search(catalog=catalog, query=query)
        status = "degraded" if vector_error else catalog.status
        items = _merge_search_items(
            _exact_lexical_search(catalog=catalog, query=query),
            items,
            top_k=self.top_k,
        )
        if not items:
            items = self._lexical_search(catalog=catalog, query=query, top_k=self.top_k)
        candidate_keys, confirmed_keys, metric_resolution_status = _metric_resolution(catalog, query, items)
        term_resolution_status = _term_resolution(catalog, query, items)
        items = _limit_items(
            _ensure_metric_items(catalog, items, confirmed_keys),
            top_k=self.top_k,
            confirmed_metric_keys=confirmed_keys,
            catalog=catalog,
        )
        prompt = format_semantic_context_prompt(
            catalog,
            items,
            confirmed_metric_keys=confirmed_keys,
            metric_resolution_status=metric_resolution_status,
            term_resolution_status=term_resolution_status,
        )
        hints = build_semantic_hints(catalog, items)
        hints["query"] = query
        hints["candidate_metric_keys"] = candidate_keys
        hints["confirmed_metric_keys"] = confirmed_keys
        hints["definition_status"] = metric_resolution_status
        hints["metric_resolution_status"] = metric_resolution_status
        hints["term_resolution_status"] = term_resolution_status
        return SemanticContextResult(status=status, prompt=prompt, items=items, hints=hints)

    def build_from_catalog(
        self,
        *,
        catalog: SemanticCatalog,
        query: str,
        items: list[SemanticSearchResultItem] | None = None,
    ) -> SemanticContextResult:
        selected = (
            items
            if items is not None
            else self._lexical_search(
                catalog=catalog,
                query=query,
                top_k=self.top_k,
            )
        )
        selected = _merge_search_items(
            _exact_lexical_search(catalog=catalog, query=query),
            selected,
            top_k=self.top_k,
        )
        candidate_keys, confirmed_keys, metric_resolution_status = _metric_resolution(
            catalog, query, selected
        )
        term_resolution_status = _term_resolution(catalog, query, selected)
        selected = _limit_items(
            _ensure_metric_items(catalog, selected, confirmed_keys),
            top_k=self.top_k,
            confirmed_metric_keys=confirmed_keys,
            catalog=catalog,
        )
        hints = build_semantic_hints(catalog, selected)
        hints["query"] = query
        hints["candidate_metric_keys"] = candidate_keys
        hints["confirmed_metric_keys"] = confirmed_keys
        hints["definition_status"] = metric_resolution_status
        hints["metric_resolution_status"] = metric_resolution_status
        hints["term_resolution_status"] = term_resolution_status
        return SemanticContextResult(
            status=catalog.status,
            prompt=format_semantic_context_prompt(
                catalog,
                selected,
                confirmed_metric_keys=confirmed_keys,
                metric_resolution_status=metric_resolution_status,
                term_resolution_status=term_resolution_status,
            ),
            items=selected,
            hints=hints,
        )

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

    def _vector_search(
        self, *, catalog: SemanticCatalog, query: str
    ) -> tuple[list[SemanticSearchResultItem], str | None]:
        if self.vector_store is None or not getattr(self.vector_store, "enabled", False):
            return [], None
        try:
            return (
                list(
                    self.vector_store.search(
                        catalog=catalog,
                        query=query,
                        top_k=max(1, int(self.top_k)),
                    )
                ),
                None,
            )
        except Exception as exc:
            logger.warning("Semantic catalog search failed: %s", exc)
            return [], str(exc)

    def _load_catalog(self, *, session_id: str, user_id: int) -> SemanticCatalog | None:
        if self.catalog_service is not None:
            loader = getattr(self.catalog_service, "load_for_session", None)
            if callable(loader):
                return loader(session_id=session_id, user_id=user_id)
        return None

    def _save_runtime_status(self, catalog: SemanticCatalog) -> None:
        if self.catalog_service is not None:
            saver = getattr(self.catalog_service, "save_runtime_status", None)
            if callable(saver):
                saver(catalog)
                return

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


def _prompt_hint(value: str, *, limit: int = 180) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _exact_lexical_search(*, catalog: SemanticCatalog, query: str) -> list[SemanticSearchResultItem]:
    text = str(query or "")
    entities = [
        (
            "metric",
            metric.metric_id,
            [metric.key, metric.name, *metric.synonyms],
        )
        for metric in catalog.metrics
        if metric.is_active
    ]
    entities.extend(
        ("term", term.term_id, [term.name, *term.synonyms]) for term in catalog.terms if term.is_active
    )
    matched = [
        SemanticSearchResultItem(
            entity_type=entity_type,
            entity_id=entity_id,
            score=1.0,
            payload={"source": "exact"},
        )
        for entity_type, entity_id, labels in entities
        if any(
            label.strip()
            and re.search(
                rf"(?<!\w){re.escape(label.strip())}(?!\w)",
                text,
                flags=re.IGNORECASE,
            )
            for label in labels
        )
    ]
    referenced_types = {
        **{table.table_id: "table" for table in catalog.tables if not table.is_hidden},
        **{metric.metric_id: "metric" for metric in catalog.metrics if metric.is_active},
        **{
            relationship.relationship_id: "relationship"
            for relationship in catalog.relationships
            if relationship.is_active
        },
    }
    terms_by_id = {term.term_id: term for term in catalog.terms if term.is_active}
    for item in list(matched):
        term = terms_by_id.get(item.entity_id)
        if term is None:
            continue
        matched.extend(
            SemanticSearchResultItem(
                entity_type=referenced_types[ref],
                entity_id=ref,
                score=1.0,
                payload={"source": "term_ref", "term_id": term.term_id},
            )
            for ref in term.entity_refs
            if ref in referenced_types
        )
    return matched


def _merge_search_items(
    preferred: list[SemanticSearchResultItem],
    ranked: list[SemanticSearchResultItem],
    *,
    top_k: int,
) -> list[SemanticSearchResultItem]:
    merged: list[SemanticSearchResultItem] = []
    seen: set[tuple[str, str]] = set()
    for item in [*preferred, *ranked]:
        identity = (item.entity_type, item.entity_id)
        if identity in seen:
            continue
        seen.add(identity)
        merged.append(item)
    return merged[: max(1, int(top_k))]


def format_semantic_context_prompt(
    catalog: SemanticCatalog,
    items: list[SemanticSearchResultItem] | None = None,
    *,
    confirmed_metric_keys: list[str] | None = None,
    metric_resolution_status: str = "",
    term_resolution_status: str = "",
) -> str:
    """Describe the relevant semantic objects for the current request."""
    lines = [
        "SEMANTIC DATA CONTEXT",
        f"status: {catalog.status}",
        f"catalog_id: {catalog.catalog_id}",
        f"source_fingerprint: {catalog.source_fingerprint}",
    ]
    if metric_resolution_status:
        resolution = f"metric_resolution: status={_prompt_hint(metric_resolution_status)}"
        if confirmed_metric_keys:
            resolution += "; confirmed_metric_keys=" + ",".join(confirmed_metric_keys)
            resolution += "; calculation_action=execute_only_if_requested_grain_is_allowed"
        lines.append(resolution)
    if term_resolution_status:
        lines.append(f"term_resolution: status={_prompt_hint(term_resolution_status)}")
    if catalog.error:
        lines.append(f"note: {catalog.error}")
    if catalog.profile_sample_strategy == "first_rows" and catalog.profile_sample_limit:
        lines.append(
            "profile_note: column profile statistics were calculated only from the first "
            f"{catalog.profile_sample_limit} rows of each table, not the complete source; "
            "profile ranges do not prove source coverage, so query requested periods"
        )

    visible_tables = [table for table in catalog.tables if not table.is_hidden]
    if visible_tables:
        lines.append("tables:")
        for table in visible_tables[:8]:
            table_details = [f"role={table.semantic_role}"]
            if table.description:
                table_details.append(f"description={_prompt_hint(table.description)}")
            if table.grain:
                table_details.append(f"grain={_prompt_hint(table.grain, limit=120)}")
            if table.aliases:
                aliases = _prompt_hint(", ".join(table.aliases[:4]), limit=120)
                table_details.append(f"aliases={aliases}")
            lines.append(f"- {table.qualified_name}: {'; '.join(table_details)}")
    if items:
        lines.append("top_k_candidates:")
        lines.append(
            "candidate_policy: entity types are authoritative; a term is not a metric unless "
            "explicitly linked to one. Select metrics whose allowed_dimensions support the "
            "requested grain; resolve a compatible candidate instead of using an incompatible "
            "exact label; inspect catalog relationships before raw schema or SQL"
        )
        metrics_by_id = {metric.metric_id: metric for metric in catalog.metrics if metric.is_active}
        terms_by_id = {term.term_id: term for term in catalog.terms if term.is_active}
        relationships_by_id = {
            relationship.relationship_id: relationship
            for relationship in catalog.relationships
            if relationship.is_active
        }
        for item in items:
            details = [
                f"type={item.entity_type}",
                f"id={item.entity_id}",
                f"score={item.score:.4f}",
            ]
            entity = metrics_by_id.get(item.entity_id) or terms_by_id.get(item.entity_id)
            if entity is not None:
                key = getattr(entity, "key", "")
                if key:
                    details.append(f"key={_prompt_hint(key)}")
                    details.append(f"base_table={_prompt_hint(entity.base_table)}")
                    if entity.allowed_dimensions:
                        details.append(
                            "allowed_dimensions="
                            + _prompt_hint(",".join(entity.allowed_dimensions), limit=160)
                        )
                details.append(f"name={_prompt_hint(entity.name)}")
                if entity.description:
                    details.append(f"description={_prompt_hint(entity.description)}")
            relationship = relationships_by_id.get(item.entity_id)
            if relationship is not None:
                details.append(
                    "contract="
                    f"{relationship.from_table}.{relationship.from_column} -> "
                    f"{relationship.to_table}.{relationship.to_column} "
                    f"({relationship.cardinality})"
                )
                if relationship.description:
                    details.append(f"description={_prompt_hint(relationship.description)}")
            lines.append(f"- {'; '.join(details)}")
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


def _label_matches(
    text: str,
    entities: list[tuple[str, list[str]]],
) -> list[tuple[str, int, int]]:
    matches: list[tuple[str, int, int]] = []
    labels: dict[str, tuple[str, set[str]]] = {}
    for key, values in entities:
        for value in values:
            label = value.strip()
            if not label:
                continue
            stored_label, keys = labels.setdefault(label.casefold(), (label, set()))
            keys.add(key)
            labels[label.casefold()] = (stored_label, keys)
    ordered_labels = sorted(
        labels.values(),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    for label, keys in ordered_labels:
        key = next(iter(keys)) if len(keys) == 1 else ""
        for match in re.finditer(
            rf"(?<!\w){re.escape(label)}(?!\w)",
            text,
            flags=re.IGNORECASE,
        ):
            start, end = match.span()
            if any(start < used_end and end > used_start for _, used_start, used_end in matches):
                continue
            matches.append((key, start, end))
    return matches


def _metric_resolution(
    catalog: SemanticCatalog,
    query: str,
    items: list[SemanticSearchResultItem],
) -> tuple[list[str], list[str], str]:
    text = str(query or "")
    active_metrics = [metric for metric in catalog.metrics if metric.is_active]
    metrics_by_ref = {ref: metric.key for metric in active_metrics for ref in (metric.metric_id, metric.key)}
    metric_labels = [(metric.key, [metric.key, metric.name, *metric.synonyms]) for metric in active_metrics]
    for term in catalog.terms:
        if not term.is_active:
            continue
        metric_refs = [ref for ref in term.entity_refs if str(ref).startswith("metric:")]
        targets = {metrics_by_ref[ref] for ref in metric_refs if ref in metrics_by_ref}
        has_missing_target = any(ref not in metrics_by_ref for ref in metric_refs)
        if len(targets) == 1 and not has_missing_target:
            metric_labels.append((next(iter(targets)), [term.name, *term.synonyms]))
        elif len(targets) > 1 or (targets and has_missing_target):
            metric_labels.append(("", [term.name, *term.synonyms]))
    matches = _label_matches(text, metric_labels)
    confirmed = list(dict.fromkeys(key for key, _, _ in matches if key))
    ambiguous = any(not key for key, _, _ in matches)
    missing_metric_labels = [
        [term.name, *term.synonyms]
        for term in catalog.terms
        if term.is_active
        and any(str(ref).startswith("metric:") and ref not in metrics_by_ref for ref in term.entity_refs)
        and not any(ref in metrics_by_ref for ref in term.entity_refs)
    ]
    missing = bool(_label_matches(text, [("", labels) for labels in missing_metric_labels]))
    metric_keys_by_ref = {
        ref: metric.key for metric in active_metrics for ref in (metric.metric_id, metric.key)
    }
    retrieved = [
        metric_keys_by_ref[item.entity_id]
        for item in items
        if item.entity_type == "metric" and item.entity_id in metric_keys_by_ref
    ]
    candidates = list(dict.fromkeys([*confirmed, *retrieved]))
    if missing:
        status = "missing"
    elif ambiguous:
        status = "ambiguous"
    elif confirmed:
        status = "resolved"
    elif candidates:
        status = "candidates"
    else:
        status = "not_found"
    if status != "resolved":
        confirmed = []
    return candidates, confirmed, status


def _term_resolution(
    catalog: SemanticCatalog,
    query: str,
    items: list[SemanticSearchResultItem],
) -> str:
    active_terms = [term for term in catalog.terms if term.is_active]
    matches = _label_matches(
        str(query or ""),
        [(term.term_id, [term.name, *term.synonyms]) for term in active_terms],
    )
    if any(not term_id for term_id, _start, _end in matches):
        return "ambiguous"
    if any(term_id for term_id, _start, _end in matches):
        return "resolved"
    active_ids = {term.term_id for term in active_terms}
    if any(item.entity_type == "term" and item.entity_id in active_ids for item in items):
        return "candidates"
    return "not_found"


def _ensure_metric_items(
    catalog: SemanticCatalog,
    items: list[SemanticSearchResultItem],
    metric_keys: list[str],
) -> list[SemanticSearchResultItem]:
    metric_ids = {metric.key: metric.metric_id for metric in catalog.metrics if metric.is_active}
    explicit = [
        SemanticSearchResultItem(
            entity_type="metric",
            entity_id=metric_ids[key],
            score=1.0,
        )
        for key in metric_keys
        if key in metric_ids
    ]
    existing_ids = {item.entity_id for item in items}
    return [*items, *(item for item in explicit if item.entity_id not in existing_ids)]


def _limit_items(
    items: list[SemanticSearchResultItem],
    *,
    top_k: int,
    confirmed_metric_keys: list[str],
    catalog: SemanticCatalog,
) -> list[SemanticSearchResultItem]:
    limit = max(1, int(top_k))
    confirmed_ids = {
        metric.metric_id
        for metric in catalog.metrics
        if metric.is_active and metric.key in confirmed_metric_keys
    }
    confirmed = [item for item in items if item.entity_id in confirmed_ids]
    remaining = [item for item in items if item.entity_id not in confirmed_ids]
    return [*confirmed, *remaining][:limit]


def _selected_entities(
    catalog: SemanticCatalog,
    items: list[SemanticSearchResultItem],
) -> dict[str, list]:
    tables_by_id = {table.table_id: table for table in catalog.tables}
    tables_by_name = {table.qualified_name: table for table in catalog.tables}
    columns_by_id = {column.column_id: column for column in catalog.columns}
    metrics_by_id = {metric.metric_id: metric for metric in catalog.metrics if metric.is_active}
    relationships_by_id = {rel.relationship_id: rel for rel in catalog.relationships if rel.is_active}
    terms_by_id = {term.term_id: term for term in catalog.terms if term.is_active}

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
    rows.extend(
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
                    " ".join(f"{item.field} {item.op} {item.value}" for item in metric.filters),
                    " ".join(metric.synonyms),
                ]
            ),
        )
        for metric in catalog.metrics
        if metric.is_active
    )
    rows.extend(
        (
            "relationship",
            rel.relationship_id,
            " ".join(
                [
                    rel.from_table,
                    rel.from_column,
                    rel.to_table,
                    rel.to_column,
                    rel.cardinality,
                    rel.description,
                ]
            ),
        )
        for rel in catalog.relationships
        if rel.is_active
    )
    rows.extend(
        (
            "term",
            term.term_id,
            " ".join([term.name, term.description, " ".join(term.synonyms)]),
        )
        for term in catalog.terms
        if term.is_active
    )
    return rows


def _tokens(text: str) -> set[str]:
    return {
        token.lower() for token in re.findall(r"[\w]+", str(text or ""), flags=re.UNICODE) if len(token) >= 2
    }
