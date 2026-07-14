from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from backend.data_access.data_catalog import CatalogTable, DataCatalogSnapshot
from backend.data_access.semantic_catalog_store import SemanticCatalogFileStore, SemanticCatalogStore
from backend.data_access.semantic_models import (
    SemanticCatalog,
    SemanticCatalogOverlay,
    SemanticColumn,
    SemanticColumnPatch,
    SemanticColumnRole,
    SemanticDimension,
    SemanticEntity,
    SemanticFact,
    SemanticMetric,
    SemanticMetricCreate,
    SemanticMetricUpdate,
    SemanticRelationship,
    SemanticRelationshipCreate,
    SemanticRelationshipUpdate,
    SemanticTable,
    SemanticTablePatch,
    SemanticTableRole,
    SemanticTerm,
    SemanticTermCreate,
    SemanticTermUpdate,
    clean_list,
    stable_id,
    utc_now_iso,
)
from backend.data_access.semantic_seed import starter_metrics, starter_terms
from backend.data_access.semantic_validator import relationship_safety_error, validate_semantic_catalog
from backend.sessions.session_store import SessionState, SessionStore

logger = logging.getLogger(__name__)

GLOBAL_SOURCE_KEY = "global"

_FORBIDDEN_SQL_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|attach|detach|copy|vacuum|call)\b",
    re.IGNORECASE,
)
_SQL_COMMENT_RE = re.compile(r"(--|/\*)")
_TIME_RE = re.compile(r"(date|time|month|year|week|day|period|timestamp|dt\b)", re.I)
_ID_RE = re.compile(r"(^id$|_id$|uuid|guid|code|number|key)", re.I)
_METRIC_RE = re.compile(
    r"(amount|revenue|sales|sum|total|price|qty|quantity|volume|profit|margin|cost)",
    re.I,
)
_FACT_TABLE_RE = re.compile(r"(sale|order|payment|event|transaction|fact)", re.I)
_DIM_TABLE_RE = re.compile(r"(customer|product|region|category|user|client|dim)", re.I)


@dataclass(frozen=True)
class SemanticSourceIdentity:
    source_key: str
    source_type: str
    source_ref_id: str
    source_label: str
    source_fingerprint: str


@dataclass
class SemanticCatalogService:
    store: SessionStore
    vector_store: Any | None = None
    settings: Any | None = None
    semantic_store: SemanticCatalogStore | None = None

    @property
    def catalog_store(self) -> SemanticCatalogStore:
        return self.semantic_store or SemanticCatalogFileStore(self.store.root_dir)

    def refresh(self, *, session_id: str, user_id: int) -> SemanticCatalog:
        state = self.store.load_session(session_id)
        snapshot = self.store.load_data_catalog(session_id)
        if snapshot is None:
            return self._unbound_catalog(session_id=session_id, user_id=user_id, state=state)

        identity = self._source_identity(session_id=session_id, state=state, snapshot=snapshot)
        overlay = self._load_overlay(identity.source_key)
        generated = self._build_catalog(
            snapshot=snapshot,
            identity=identity,
            session_id=session_id,
            user_id=user_id,
        )
        overlay = self._seed_overlay_if_empty(overlay, generated)
        published = self._publish(generated=generated, overlay=overlay)
        if published.status != "failed":
            published.status = "indexing"
        self._save_generated(generated)
        self._save_published(published)
        try:
            if self.vector_store is not None and getattr(self.vector_store, "enabled", False):
                self.vector_store.upsert_catalog(published)
            self._apply_validation_status(published)
        except Exception as exc:
            published.status = "degraded"
            published.error = "; ".join(
                item for item in [published.error, f"Qdrant indexing failed: {exc}"] if item
            )
            logger.warning("Semantic catalog indexing failed: %s", exc)
        published.updated_at = utc_now_iso()
        self._save_published(published)
        return self._for_session(published, session_id=session_id, user_id=user_id)

    def load_for_session(self, *, session_id: str, user_id: int) -> SemanticCatalog | None:
        state = self.store.load_session(session_id)
        snapshot = self.store.load_data_catalog(session_id)
        if snapshot is None:
            legacy = self.store.load_semantic_catalog(session_id)
            if legacy is not None:
                return legacy
            return self._unbound_catalog(session_id=session_id, user_id=user_id, state=state)

        identity = self._source_identity(session_id=session_id, state=state, snapshot=snapshot)
        catalog = self._load_published(identity.source_key)
        if catalog is None:
            legacy = self.store.load_semantic_catalog(session_id)
            if legacy is not None:
                return legacy
            return None

        if catalog.source_fingerprint != identity.source_fingerprint:
            catalog.status = "stale"
            catalog.error = "Data catalog source fingerprint changed."
            catalog.updated_at = utc_now_iso()
            self._save_published(catalog)
        catalog = self._with_current_global_terms(catalog)
        return self._for_session(catalog, session_id=session_id, user_id=user_id)

    def create_metric(
        self,
        *,
        session_id: str,
        user_id: int,
        payload: SemanticMetricCreate,
    ) -> SemanticMetric:
        catalog, overlay = self._catalog_and_overlay(session_id=session_id, user_id=user_id)
        metric = SemanticMetric(**payload.model_dump(), metric_id=f"metric:{payload.key}")
        self._validate_metric(catalog, metric)
        overlay.metrics = [item for item in overlay.metrics if item.metric_id != metric.metric_id]
        overlay.metrics.append(metric)
        self._save_overlay(overlay)
        self._republish_from_overlay(catalog, overlay)
        return metric

    def update_metric(
        self,
        *,
        session_id: str,
        user_id: int,
        metric_id: str,
        payload: SemanticMetricUpdate,
    ) -> SemanticMetric:
        catalog, overlay = self._catalog_and_overlay(session_id=session_id, user_id=user_id)
        current = next((item for item in overlay.metrics if item.metric_id == metric_id), None)
        if current is None:
            current = next((item for item in catalog.metrics if item.metric_id == metric_id), None)
        if current is None:
            raise ValueError("Metric not found")

        data = current.model_dump()
        for key, value in payload.model_dump(exclude_unset=True).items():
            data[key] = value
        data["updated_at"] = utc_now_iso()
        updated = SemanticMetric.model_validate(data)
        self._validate_metric(catalog, updated)
        overlay.metrics = [updated if item.metric_id == metric_id else item for item in overlay.metrics]
        if all(item.metric_id != metric_id for item in overlay.metrics):
            overlay.metrics.append(updated)
        self._save_overlay(overlay)
        self._republish_from_overlay(catalog, overlay)
        return updated

    def delete_metric(self, *, session_id: str, user_id: int, metric_id: str) -> None:
        catalog, overlay = self._catalog_and_overlay(session_id=session_id, user_id=user_id)
        overlay.metrics = [item for item in overlay.metrics if item.metric_id != metric_id]
        self._save_overlay(overlay)
        self._republish_from_overlay(catalog, overlay)

    def create_relationship(
        self,
        *,
        session_id: str,
        user_id: int,
        payload: SemanticRelationshipCreate,
    ) -> SemanticRelationship:
        catalog, overlay = self._catalog_and_overlay(session_id=session_id, user_id=user_id)
        relationship = SemanticRelationship(
            **payload.model_dump(),
            relationship_id=f"relationship:{stable_id(catalog.source_key, payload.from_table, payload.from_column, payload.to_table, payload.to_column)}",
        )
        self._validate_relationship(catalog, relationship)
        overlay.relationships = [
            item for item in overlay.relationships if item.relationship_id != relationship.relationship_id
        ]
        overlay.relationships.append(relationship)
        self._save_overlay(overlay)
        self._republish_from_overlay(catalog, overlay)
        return relationship

    def update_relationship(
        self,
        *,
        session_id: str,
        user_id: int,
        relationship_id: str,
        payload: SemanticRelationshipUpdate,
    ) -> SemanticRelationship:
        catalog, overlay = self._catalog_and_overlay(session_id=session_id, user_id=user_id)
        current = next((item for item in overlay.relationships if item.relationship_id == relationship_id), None)
        if current is None:
            current = next((item for item in catalog.relationships if item.relationship_id == relationship_id), None)
        if current is None:
            raise ValueError("Relationship not found")

        data = current.model_dump()
        data.update(payload.model_dump(exclude_unset=True))
        updated = SemanticRelationship.model_validate(data)
        self._validate_relationship(catalog, updated)
        overlay.relationships = [
            updated if item.relationship_id == relationship_id else item
            for item in overlay.relationships
        ]
        if all(item.relationship_id != relationship_id for item in overlay.relationships):
            overlay.relationships.append(updated)
        self._save_overlay(overlay)
        self._republish_from_overlay(catalog, overlay)
        return updated

    def delete_relationship(self, *, session_id: str, user_id: int, relationship_id: str) -> None:
        catalog, overlay = self._catalog_and_overlay(session_id=session_id, user_id=user_id)
        overlay.relationships = [
            item for item in overlay.relationships if item.relationship_id != relationship_id
        ]
        self._save_overlay(overlay)
        self._republish_from_overlay(catalog, overlay)

    def create_term(
        self,
        *,
        session_id: str,
        user_id: int,
        payload: SemanticTermCreate,
    ) -> SemanticTerm:
        catalog = self.load_for_session(session_id=session_id, user_id=user_id)
        if catalog is None:
            catalog = self.refresh(session_id=session_id, user_id=user_id)
        overlay = self._global_overlay()
        term = SemanticTerm(
            **payload.model_dump(),
            term_id=f"term:{stable_id('term', GLOBAL_SOURCE_KEY, payload.name)}",
        )
        overlay.terms = [item for item in overlay.terms if item.term_id != term.term_id]
        overlay.terms.append(term)
        self._save_overlay(overlay)
        self._republish_terms(catalog)
        return term

    def update_term(
        self,
        *,
        session_id: str,
        user_id: int,
        term_id: str,
        payload: SemanticTermUpdate,
    ) -> SemanticTerm:
        catalog = self.load_for_session(session_id=session_id, user_id=user_id)
        if catalog is None:
            catalog = self.refresh(session_id=session_id, user_id=user_id)
        overlay = self._global_overlay()
        current = next((item for item in overlay.terms if item.term_id == term_id), None)
        if current is None:
            raise ValueError("Term not found")

        data = current.model_dump()
        for key, value in payload.model_dump(exclude_unset=True).items():
            if key in {"synonyms", "entity_refs"} and value is not None:
                data[key] = clean_list(value)
            else:
                data[key] = value
        data["updated_at"] = utc_now_iso()
        updated = SemanticTerm.model_validate(data)
        overlay.terms = [updated if item.term_id == term_id else item for item in overlay.terms]
        if all(item.term_id != term_id for item in overlay.terms):
            overlay.terms.append(updated)
        self._save_overlay(overlay)
        self._republish_terms(catalog)
        return updated

    def delete_term(self, *, session_id: str, user_id: int, term_id: str) -> None:
        catalog = self.load_for_session(session_id=session_id, user_id=user_id)
        if catalog is None:
            catalog = self.refresh(session_id=session_id, user_id=user_id)
        overlay = self._global_overlay()
        overlay.terms = [item for item in overlay.terms if item.term_id != term_id]
        self._save_overlay(overlay)
        self._republish_terms(catalog)

    def patch_table(
        self,
        *,
        session_id: str,
        user_id: int,
        table_id: str,
        payload: SemanticTablePatch,
    ) -> SemanticTable:
        catalog, overlay = self._catalog_and_overlay(session_id=session_id, user_id=user_id)
        table = self._table_for_id(catalog, table_id)
        if table is None:
            raise ValueError("Semantic table not found")
        current = overlay.table_patches.get(table.table_id, SemanticTablePatch())
        overlay.table_patches[table.table_id] = self._merge_patch(current, payload, SemanticTablePatch)
        self._save_overlay(overlay)
        published = self._republish_from_overlay(catalog, overlay)
        return self._table_for_id(published, table.table_id) or table

    def patch_column(
        self,
        *,
        session_id: str,
        user_id: int,
        column_id: str,
        payload: SemanticColumnPatch,
    ) -> SemanticColumn:
        catalog, overlay = self._catalog_and_overlay(session_id=session_id, user_id=user_id)
        column = self._column_for_id(catalog, column_id)
        if column is None:
            raise ValueError("Semantic column not found")
        current = overlay.column_patches.get(column.column_id, SemanticColumnPatch())
        overlay.column_patches[column.column_id] = self._merge_patch(current, payload, SemanticColumnPatch)
        self._save_overlay(overlay)
        published = self._republish_from_overlay(catalog, overlay)
        return self._column_for_id(published, column.column_id) or column

    def search(
        self,
        *,
        session_id: str,
        user_id: int,
        query: str,
        top_k: int = 8,
    ):
        from backend.data_access.semantic_context import SemanticContextBuilder

        return SemanticContextBuilder(
            store=self.store,
            vector_store=self.vector_store,
            catalog_service=self,
            top_k=top_k,
        ).build(session_id=session_id, user_id=user_id, query=query)

    def save_runtime_status(self, catalog: SemanticCatalog) -> None:
        self._save_published(self._for_source(catalog))

    def apply_generated_overlay(
        self,
        *,
        session_id: str,
        user_id: int,
        table_patches: dict[str, SemanticTablePatch] | None = None,
        column_patches: dict[str, SemanticColumnPatch] | None = None,
        metrics: list[SemanticMetric] | None = None,
        relationships: list[SemanticRelationship] | None = None,
        terms: list[SemanticTerm] | None = None,
    ) -> tuple[SemanticCatalog, list[str]]:
        catalog, overlay = self._catalog_and_overlay(session_id=session_id, user_id=user_id)
        rejected: list[str] = []
        known_tables = {table.table_id for table in catalog.tables}
        known_columns = {column.column_id for column in catalog.columns}
        for table_id, patch in (table_patches or {}).items():
            if table_id in known_tables:
                current = overlay.table_patches.get(table_id, SemanticTablePatch())
                overlay.table_patches[table_id] = self._merge_patch(current, patch, SemanticTablePatch)
            else:
                rejected.append(f"Unknown table patch target: {table_id}")
        for column_id, patch in (column_patches or {}).items():
            if column_id in known_columns:
                current = overlay.column_patches.get(column_id, SemanticColumnPatch())
                overlay.column_patches[column_id] = self._merge_patch(current, patch, SemanticColumnPatch)
            else:
                rejected.append(f"Unknown column patch target: {column_id}")
        for metric in metrics or []:
            try:
                self._validate_metric(catalog, metric)
            except ValueError as exc:
                rejected.append(f"Metric {metric.key}: {exc}")
                continue
            overlay.metrics = [item for item in overlay.metrics if item.metric_id != metric.metric_id]
            overlay.metrics.append(metric)
        for relationship in relationships or []:
            try:
                self._validate_relationship(catalog, relationship)
            except ValueError as exc:
                rejected.append(f"Relationship {relationship.relationship_id}: {exc}")
                continue
            overlay.relationships = [
                item for item in overlay.relationships if item.relationship_id != relationship.relationship_id
            ]
            overlay.relationships.append(relationship)
        for term in terms or []:
            overlay.terms = [item for item in overlay.terms if item.term_id != term.term_id]
            overlay.terms.append(term)
        self._save_overlay(overlay)
        return self._republish_from_overlay(catalog, overlay), rejected

    def _catalog_and_overlay(
        self,
        *,
        session_id: str,
        user_id: int,
    ) -> tuple[SemanticCatalog, SemanticCatalogOverlay]:
        catalog = self.load_for_session(session_id=session_id, user_id=user_id)
        if catalog is None:
            catalog = self.refresh(session_id=session_id, user_id=user_id)
        overlay = self._load_overlay(catalog.source_key)
        return catalog, overlay

    def _unbound_catalog(
        self,
        *,
        session_id: str,
        user_id: int,
        state: SessionState | None,
    ) -> SemanticCatalog:
        source_type = str(getattr(state, "source_type", "") or "")
        source_ref_id = str(getattr(state, "source_ref_id", "") or "")
        source_label = str(getattr(state, "source_label", "") or "")
        catalog = SemanticCatalog(
            catalog_id=stable_id("semantic-catalog", GLOBAL_SOURCE_KEY),
            source_key=GLOBAL_SOURCE_KEY,
            user_id=user_id,
            session_id=session_id,
            source_type=source_type,
            source_ref_id=source_ref_id,
            source_label=source_label,
            status="unbound",
            terms=self._global_terms(),
        )
        catalog.validation = validate_semantic_catalog(catalog)
        return catalog

    def _build_catalog(
        self,
        *,
        snapshot: DataCatalogSnapshot,
        identity: SemanticSourceIdentity,
        session_id: str,
        user_id: int,
    ) -> SemanticCatalog:
        catalog_id = stable_id("semantic-catalog", identity.source_key, identity.source_fingerprint)
        tables = [self._semantic_table(table) for table in snapshot.tables]
        columns = [
            self._semantic_column(table, column)
            for table in snapshot.tables
            for column in table.columns
        ]
        entities = self._semantic_entities(columns)
        dimensions = self._semantic_dimensions(columns)
        facts = self._semantic_facts(columns)
        now = utc_now_iso()
        profile_error = "; ".join(snapshot.errors[:3]) or None
        return SemanticCatalog(
            catalog_id=catalog_id,
            source_key=identity.source_key,
            user_id=user_id,
            session_id=session_id,
            source_type=identity.source_type,
            source_ref_id=identity.source_ref_id,
            source_label=identity.source_label,
            source_fingerprint=identity.source_fingerprint,
            status=("failed" if profile_error and not tables else "degraded" if profile_error else "pending"),
            error=profile_error,
            version="2.0",
            built_at=now,
            updated_at=now,
            tables=tables,
            columns=columns,
            entities=entities,
            dimensions=dimensions,
            facts=facts,
        )

    def _publish(
        self,
        *,
        generated: SemanticCatalog,
        overlay: SemanticCatalogOverlay,
    ) -> SemanticCatalog:
        table_patches = dict(overlay.table_patches or {})
        column_patches = dict(overlay.column_patches or {})
        catalog = generated.model_copy(deep=True)
        catalog.tables = [
            self._apply_table_patch(table, table_patches.get(table.table_id))
            for table in catalog.tables
        ]
        catalog.columns = [
            self._apply_column_patch(column, column_patches.get(column.column_id))
            for column in catalog.columns
        ]
        table_names = {table.qualified_name for table in catalog.tables}
        column_keys = {(column.table, column.name) for column in catalog.columns}
        catalog.metrics = [
            metric
            for metric in overlay.metrics
            if self._metric_targets_exist(metric, table_names, column_keys)
        ]
        catalog.relationships = [
            rel
            for rel in overlay.relationships
            if (rel.from_table, rel.from_column) in column_keys
            and (rel.to_table, rel.to_column) in column_keys
        ]
        catalog.terms = _dedupe_terms(self._global_terms() + list(overlay.terms))
        catalog.overlay_version = int(overlay.version or 0)
        catalog.published_version = int(overlay.version or 0)
        catalog.validation = validate_semantic_catalog(catalog)
        self._apply_validation_status(catalog)
        catalog.updated_at = utc_now_iso()
        return catalog

    def _republish_from_overlay(
        self,
        catalog: SemanticCatalog,
        overlay: SemanticCatalogOverlay,
    ) -> SemanticCatalog:
        generated = self._load_generated(catalog.source_key) or self._for_source(catalog)
        published = self._publish(generated=generated, overlay=overlay)
        self._apply_validation_status(published)
        self._save_published(published)
        self._try_reindex(published)
        return self._for_session(published, session_id=catalog.session_id, user_id=catalog.user_id)

    def _republish_terms(self, catalog: SemanticCatalog) -> SemanticCatalog:
        if catalog.source_key == GLOBAL_SOURCE_KEY:
            overlay = self._global_overlay()
        else:
            overlay = self._load_overlay(catalog.source_key)
        return self._republish_from_overlay(catalog, overlay)

    def _global_overlay(self) -> SemanticCatalogOverlay:
        overlay = self._load_overlay(GLOBAL_SOURCE_KEY)
        if overlay.version or overlay.terms:
            return overlay
        overlay.terms = starter_terms(GLOBAL_SOURCE_KEY)
        self._save_overlay(overlay)
        return overlay

    def _global_terms(self) -> list[SemanticTerm]:
        return list(self._global_overlay().terms)

    def _with_current_global_terms(self, catalog: SemanticCatalog) -> SemanticCatalog:
        if catalog.source_key == GLOBAL_SOURCE_KEY:
            return catalog
        overlay = self._load_overlay(catalog.source_key)
        catalog = catalog.model_copy(deep=True)
        catalog.terms = _dedupe_terms(self._global_terms() + list(overlay.terms))
        return catalog

    def _seed_overlay_if_empty(
        self,
        overlay: SemanticCatalogOverlay,
        catalog: SemanticCatalog,
    ) -> SemanticCatalogOverlay:
        if overlay.version or overlay.terms or overlay.metrics:
            return overlay
        if catalog.source_key == GLOBAL_SOURCE_KEY:
            overlay.terms = starter_terms(GLOBAL_SOURCE_KEY)
        else:
            overlay.metrics = starter_metrics(catalog)
        self._save_overlay(overlay)
        return overlay

    def _semantic_table(self, table: CatalogTable) -> SemanticTable:
        return SemanticTable(
            table_id=f"table:{table.qualified_name}",
            qualified_name=table.qualified_name,
            table_name=table.table_name,
            source_kind=table.source_kind,
            schema_name=table.schema,
            semantic_role=self._table_role(table),
            row_count=table.row_estimate,
            columns_count=len(table.columns),
            aliases=[table.table_name] if table.table_name != table.qualified_name else [],
        )

    def _semantic_column(self, table: CatalogTable, column: Any) -> SemanticColumn:
        return SemanticColumn(
            column_id=f"column:{table.qualified_name}.{column.name}",
            table=table.qualified_name,
            name=column.name,
            dtype=column.dtype,
            nullable=column.nullable,
            null_ratio=column.null_ratio,
            distinct_count=column.distinct_count,
            semantic_role=self._column_role(column.name, column.dtype),
            examples=list(column.examples),
            min_value=column.min_value,
            max_value=column.max_value,
            top_values=list(column.top_values),
        )

    def _table_role(self, table: CatalogTable) -> SemanticTableRole:
        name = f"{table.qualified_name} {table.table_name}"
        if _FACT_TABLE_RE.search(name):
            return "fact"
        if _DIM_TABLE_RE.search(name):
            return "dimension"
        return "unknown"

    def _column_role(self, name: str, dtype: str) -> SemanticColumnRole:
        text = f"{name} {dtype}"
        dtype_l = str(dtype or "").lower()
        if _TIME_RE.search(text) or "date" in dtype_l or "time" in dtype_l:
            return "time"
        if _ID_RE.search(name):
            return "identifier"
        if dtype_l in {"bool", "boolean"}:
            return "flag"
        if any(kind in dtype_l for kind in ("int", "float", "double", "decimal", "numeric")):
            return "metric_candidate" if _METRIC_RE.search(text) else "dimension"
        if any(kind in dtype_l for kind in ("str", "text", "char", "varchar")):
            return "dimension"
        return "unknown"

    def _semantic_entities(self, columns: list[SemanticColumn]) -> list[SemanticEntity]:
        entities: list[SemanticEntity] = []
        for column in columns:
            if column.semantic_role not in {"identifier", "foreign_key_candidate"}:
                continue
            name = re.sub(r"(_id|id)$", "", column.name.lower()).strip("_") or column.name.lower()
            kind = "primary" if _column_names_table_entity(column.table, name) else "foreign"
            entities.append(
                SemanticEntity(
                    entity_id=f"entity:{column.table}.{name}",
                    name=name,
                    table=column.table,
                    expr=column.name,
                    type=kind,
                )
            )
        return entities

    def _semantic_dimensions(self, columns: list[SemanticColumn]) -> list[SemanticDimension]:
        dimensions: list[SemanticDimension] = []
        for column in columns:
            if column.semantic_role not in {"dimension", "time", "flag"}:
                continue
            kind = "time" if column.semantic_role == "time" else "boolean" if column.semantic_role == "flag" else "categorical"
            dimensions.append(
                SemanticDimension(
                    dimension_id=f"dimension:{column.table}.{column.name}",
                    name=column.name,
                    table=column.table,
                    expr=column.name,
                    type=kind,
                    grains=["day", "week", "month", "quarter", "year"] if kind == "time" else [],
                )
            )
        return dimensions

    def _semantic_facts(self, columns: list[SemanticColumn]) -> list[SemanticFact]:
        return [
            SemanticFact(
                fact_id=f"fact:{column.table}.{column.name}",
                name=column.name,
                table=column.table,
                expr=column.name,
                type="number",
            )
            for column in columns
            if column.semantic_role == "metric_candidate"
        ]

    def _validate_metric(self, catalog: SemanticCatalog, metric: SemanticMetric) -> None:
        table = self._table_for_metric(catalog, metric.base_table)
        if table is None:
            raise ValueError(f"Unknown metric table: {metric.base_table}")
        metric.base_table = table.qualified_name
        table_columns = {
            column.name
            for column in catalog.columns
            if column.table == table.qualified_name
        }
        if metric.type == "simple" and metric.expr and metric.expr not in table_columns:
            raise ValueError(f"Unknown metric column: {metric.expr}")
        if metric.default_time_dimension and metric.default_time_dimension not in table_columns:
            raise ValueError(f"Unknown metric time column: {metric.default_time_dimension}")
        dimension_names = {dim.name for dim in catalog.dimensions}
        missing_dimensions = [item for item in metric.allowed_dimensions if item not in dimension_names]
        if missing_dimensions:
            raise ValueError(f"Unknown metric dimensions: {', '.join(missing_dimensions)}")
        metric_keys = {item.key for item in catalog.metrics if item.metric_id != metric.metric_id}
        for ref in [metric.numerator, metric.denominator]:
            if ref and ref not in metric_keys:
                raise ValueError(f"Unknown metric reference: {ref}")

    def _validate_relationship(self, catalog: SemanticCatalog, relationship: SemanticRelationship) -> None:
        if relationship.cardinality == "many_to_many":
            raise ValueError("many_to_many relationships are not supported yet")
        column_keys = {(column.table, column.name) for column in catalog.columns}
        missing = [
            f"{table}.{column}"
            for table, column in [
                (relationship.from_table, relationship.from_column),
                (relationship.to_table, relationship.to_column),
            ]
            if (table, column) not in column_keys
        ]
        if missing:
            raise ValueError(f"Unknown relationship columns: {', '.join(missing)}")
        safety_error = relationship_safety_error(catalog, relationship)
        if safety_error:
            raise ValueError(f"Unsafe relationship: {safety_error}")

    def _table_for_metric(self, catalog: SemanticCatalog, name: str) -> SemanticTable | None:
        needle = str(name or "").strip()
        return next(
            (
                table
                for table in catalog.tables
                if table.qualified_name == needle or table.table_name == needle
            ),
            None,
        )

    def _table_for_id(self, catalog: SemanticCatalog, table_id: str) -> SemanticTable | None:
        needle = str(table_id or "").strip()
        return next(
            (
                table
                for table in catalog.tables
                if table.table_id == needle
                or table.qualified_name == needle
                or table.table_name == needle
            ),
            None,
        )

    def _column_for_id(self, catalog: SemanticCatalog, column_id: str) -> SemanticColumn | None:
        needle = str(column_id or "").strip()
        return next(
            (
                column
                for column in catalog.columns
                if column.column_id == needle or f"{column.table}.{column.name}" == needle
            ),
            None,
        )

    def _validate_metric_sql(self, sql: str, allowed_columns: set[str]) -> None:
        if ";" in sql or _SQL_COMMENT_RE.search(sql) or _FORBIDDEN_SQL_RE.search(sql):
            raise ValueError("Only read-only aggregate expressions are allowed")
        if not re.search(r"\b(sum|avg|count|min|max|nullif)\s*\(", sql, re.I):
            raise ValueError("Only read-only aggregate expressions are allowed")
        identifiers = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", sql))
        allowed_words = {
            "sum",
            "avg",
            "count",
            "distinct",
            "min",
            "max",
            "nullif",
            "case",
            "when",
            "then",
            "else",
            "end",
        }
        unknown = [
            item
            for item in identifiers
            if item.lower() not in allowed_words and item not in allowed_columns
        ]
        if unknown:
            raise ValueError(f"Unknown metric SQL identifiers: {', '.join(sorted(unknown))}")

    def _metric_targets_exist(
        self,
        metric: SemanticMetric,
        table_names: set[str],
        column_keys: set[tuple[str, str]],
    ) -> bool:
        if metric.base_table not in table_names:
            return False
        if metric.type == "simple" and metric.expr and (metric.base_table, metric.expr) not in column_keys:
            return False
        return True

    @staticmethod
    def _apply_validation_status(catalog: SemanticCatalog) -> None:
        if catalog.status == "unbound":
            catalog.error = None
            return
        if catalog.status == "failed":
            return
        existing_error = catalog.error
        if catalog.validation.errors:
            catalog.status = "degraded"
            catalog.error = "; ".join(
                item
                for item in [
                    existing_error,
                    "; ".join(issue.message for issue in catalog.validation.errors[:3]),
                ]
                if item
            )
        elif existing_error:
            catalog.status = "degraded"
        else:
            catalog.status = "ready"
            catalog.error = None

    def _try_reindex(self, catalog: SemanticCatalog) -> None:
        try:
            if self.vector_store is not None and getattr(self.vector_store, "enabled", False):
                self.vector_store.upsert_catalog(catalog)
        except Exception as exc:
            catalog.status = "degraded"
            catalog.error = f"Qdrant indexing failed: {exc}"
            catalog.updated_at = utc_now_iso()
            self._save_published(catalog)

    def _source_identity(
        self,
        *,
        session_id: str,
        state: SessionState | None,
        snapshot: DataCatalogSnapshot,
    ) -> SemanticSourceIdentity:
        source_type = str(getattr(state, "source_type", "") or "").strip().lower()
        source_ref_id = str(getattr(state, "source_ref_id", "") or "").strip()
        source_label = str(getattr(state, "source_label", "") or "").strip()
        schema_sig = _snapshot_signature(snapshot)
        if source_type in {"db_connection", "openproject"} and source_ref_id:
            source_key = f"{source_type}:{source_ref_id}"
            source_fingerprint = snapshot.source_fingerprint
        elif source_type == "csv":
            source_fingerprint = _csv_source_fingerprint(
                snapshot.source_fingerprint,
                source_ref_id,
                schema_sig,
            )
            source_key = "csv:" + stable_id("csv-source", source_fingerprint)
        else:
            source_key = "source:" + stable_id("source", source_type, source_ref_id, schema_sig, session_id)
            source_fingerprint = snapshot.source_fingerprint or schema_sig
        return SemanticSourceIdentity(
            source_key=source_key,
            source_type=source_type,
            source_ref_id=source_ref_id,
            source_label=source_label,
            source_fingerprint=source_fingerprint,
        )

    def _save_generated(self, catalog: SemanticCatalog) -> None:
        self.catalog_store.save_generated(catalog)

    def _load_generated(self, source_key: str) -> SemanticCatalog | None:
        return self.catalog_store.load_generated(source_key)

    def _save_published(self, catalog: SemanticCatalog) -> None:
        self.catalog_store.save_published(catalog)

    def _load_published(self, source_key: str) -> SemanticCatalog | None:
        return self.catalog_store.load_published(source_key)

    def _load_overlay(self, source_key: str) -> SemanticCatalogOverlay:
        return self.catalog_store.load_overlay(source_key)

    def _save_overlay(self, overlay: SemanticCatalogOverlay) -> None:
        self.catalog_store.save_overlay(overlay)

    @staticmethod
    def _for_source(catalog: SemanticCatalog) -> SemanticCatalog:
        return catalog.model_copy(update={"session_id": "", "user_id": 0}, deep=True)

    @staticmethod
    def _for_session(catalog: SemanticCatalog, *, session_id: str, user_id: int) -> SemanticCatalog:
        return catalog.model_copy(update={"session_id": session_id, "user_id": user_id}, deep=True)

    @staticmethod
    def _merge_patch(current: Any, payload: Any, model_cls: Any) -> Any:
        data = current.model_dump(exclude_unset=True)
        data.update(payload.model_dump(exclude_unset=True))
        return model_cls.model_validate(data)

    @staticmethod
    def _apply_table_patch(table: SemanticTable, patch: SemanticTablePatch | None) -> SemanticTable:
        if patch is None:
            return table
        data = table.model_dump()
        for key, value in patch.model_dump(exclude_unset=True).items():
            if value is None:
                continue
            if key in {"aliases", "tags", "quality_notes"}:
                data[key] = clean_list(value)
            else:
                data[key] = value
        return SemanticTable.model_validate(data)

    @staticmethod
    def _apply_column_patch(column: SemanticColumn, patch: SemanticColumnPatch | None) -> SemanticColumn:
        if patch is None:
            return column
        data = column.model_dump()
        for key, value in patch.model_dump(exclude_unset=True).items():
            if value is None:
                continue
            if key in {"aliases", "examples", "quality_notes"}:
                data[key] = clean_list(value)
            else:
                data[key] = value
        return SemanticColumn.model_validate(data)


def _snapshot_signature(snapshot: DataCatalogSnapshot) -> str:
    rows = [
        {
            "table": table.qualified_name,
            "kind": table.source_kind,
            "columns": [{"name": col.name, "dtype": col.dtype} for col in table.columns],
        }
        for table in sorted(snapshot.tables, key=lambda item: item.qualified_name)
    ]
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()


def _csv_source_fingerprint(
    snapshot_fingerprint: str,
    source_ref_id: str,
    schema_sig: str,
) -> str:
    for value in (snapshot_fingerprint, source_ref_id):
        clean = str(value or "").strip()
        if clean.startswith("csv:sha256:") or clean.startswith("sha256:"):
            return clean
    return schema_sig


def _dedupe_terms(terms: list[SemanticTerm]) -> list[SemanticTerm]:
    seen: set[str] = set()
    result: list[SemanticTerm] = []
    for term in terms:
        if term.term_id in seen:
            continue
        seen.add(term.term_id)
        result.append(term)
    return result


def _column_names_table_entity(table_name: str, entity_name: str) -> bool:
    table_leaf = str(table_name or "").split(".")[-1].lower()
    entity = str(entity_name or "").lower()
    singular = table_leaf[:-1] if table_leaf.endswith("s") else table_leaf
    return entity in {table_leaf, singular, "id"} or table_leaf.startswith(f"{entity}_")


def catalog_to_json(catalog: SemanticCatalog) -> str:
    return catalog.model_dump_json(indent=0)


def catalog_from_json(text: str) -> SemanticCatalog | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    return SemanticCatalog.model_validate(json.loads(raw))
