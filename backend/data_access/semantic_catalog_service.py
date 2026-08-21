from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from backend.data_access.data_catalog import (
    CatalogTable,
    DataCatalogSnapshot,
    build_snapshot_from_db_helper,
)
from backend.data_access.semantic_catalog_store import SemanticCatalogStore
from backend.data_access.semantic_models import (
    SemanticCatalog,
    SemanticCatalogOperation,
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
from backend.data_access.semantic_seed import starter_terms
from backend.data_access.semantic_validator import (
    metric_dependency_errors,
    metric_references,
    relationship_safety_error,
    validate_semantic_catalog,
)
from backend.sessions.session_store import SessionState, SessionStore

logger = logging.getLogger(__name__)

GLOBAL_SOURCE_KEY = "global"
_OPERATION_STALE_AFTER = timedelta(minutes=30)

_FORBIDDEN_SQL_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|attach|detach|copy|vacuum|call)\b",
    re.IGNORECASE,
)
_SQL_COMMENT_RE = re.compile(r"(--|/\*)")
_SQL_IDENT_RE = re.compile(r"\b[^\W\d]\w*\b", re.UNICODE)
_QUALIFIED_SQL_IDENT_RE = re.compile(
    r"\b[^\W\d]\w*(?:\.[^\W\d]\w*)+\b",
    re.UNICODE,
)
_TIME_RE = re.compile(r"(date|time|month|year|week|day|period|timestamp|dt\b)", re.I)
_ID_RE = re.compile(r"(^id$|_id$|uuid|guid|code|number|key)", re.I)
_METRIC_RE = re.compile(
    r"(amount|revenue|sales|sum|total|price|qty|quantity|volume|profit|margin|cost|"
    r"count|turnover|score|rating|value|measure|usage|capacity)",
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

    @property
    def catalog_store(self) -> SemanticCatalogStore:
        if self.store.metadata_store is None:
            raise RuntimeError("PostgreSQL semantic metadata store is required")
        return self.store.metadata_store

    def refresh(
        self,
        *,
        session_id: str,
        user_id: int,
        operation_id: int | None = None,
    ) -> SemanticCatalog:
        state = self.store.load_session(session_id)
        snapshot = self.store.load_data_catalog(session_id)
        if snapshot is None:
            raise ValueError("Data profile is not available for semantic catalog refresh")

        identity = self._source_identity(
            session_id=session_id,
            user_id=user_id,
            state=state,
            snapshot=snapshot,
        )
        try:
            self._advance_operation(operation_id, stage="profiling")
            overlay = self._load_overlay(identity.source_key)
            generated = self._build_catalog(
                snapshot=snapshot,
                identity=identity,
                session_id=session_id,
                user_id=user_id,
            )
            published = self._publish(generated=generated, overlay=overlay)
            self._advance_operation(operation_id, stage="indexing")
            self._try_reindex(published)
            self._advance_after_reindex(operation_id, published)
            self._commit_build_result(
                operation_id=operation_id,
                generated=generated,
                published=published,
            )
            self._complete_operation(operation_id)
            return self._for_session(published, session_id=session_id, user_id=user_id)
        except Exception as exc:
            self.mark_build_failed(
                source_key=identity.source_key,
                error=str(exc),
                operation_id=operation_id,
            )
            raise

    def build_for_connection(
        self,
        *,
        user_id: int,
        runtime: Any,
        source_label: str = "",
        operation_id: int | None = None,
    ) -> SemanticCatalog:
        from backend.tools.impl.db_helpers import DBAnalyticsHelper

        connection_id = str(getattr(runtime, "connection_id", "") or "").strip()
        if not connection_id:
            raise ValueError("connection_id is required")
        identity = SemanticSourceIdentity(
            source_key=f"db_connection:{connection_id}",
            source_type="db_connection",
            source_ref_id=connection_id,
            source_label=source_label,
            source_fingerprint=f"db:{connection_id}",
        )
        try:
            self._advance_operation(operation_id, stage="profiling")
            helper = DBAnalyticsHelper(runtime=runtime, timeout_sec=15.0)
            snapshot = build_snapshot_from_db_helper(
                helper,
                fingerprint=identity.source_fingerprint,
            )
            generated = self._build_catalog(
                snapshot=snapshot,
                identity=identity,
                session_id="",
                user_id=user_id,
            )
            overlay = self._load_overlay(identity.source_key)
            published = self._publish(generated=generated, overlay=overlay)
            self._advance_operation(operation_id, stage="indexing")
            self._try_reindex(published)
            self._advance_after_reindex(operation_id, published)
            self._commit_build_result(
                operation_id=operation_id,
                generated=generated,
                published=published,
            )
            self._complete_operation(operation_id)
            return published
        except Exception as exc:
            self.mark_build_failed(
                source_key=identity.source_key,
                error=str(exc),
                operation_id=operation_id,
            )
            raise

    def ensure_for_connection(
        self,
        *,
        user_id: int,
        runtime: Any,
        source_label: str = "",
    ) -> SemanticCatalog:
        connection_id = str(getattr(runtime, "connection_id", "") or "").strip()
        pending, operation = self.claim_connection_build(
            connection_id=connection_id,
            user_id=user_id,
            source_label=source_label or str(getattr(runtime, "name", "") or ""),
        )
        if operation is None:
            return pending
        return self.build_for_connection(
            user_id=user_id,
            runtime=runtime,
            source_label=pending.source_label,
            operation_id=operation.operation_id,
        )

    def claim_connection_build(
        self,
        *,
        connection_id: str,
        user_id: int,
        source_label: str = "",
        force: bool = False,
    ) -> tuple[SemanticCatalog, SemanticCatalogOperation | None]:
        clean_id = str(connection_id or "").strip()
        if not clean_id:
            raise ValueError("connection_id is required")
        source_key = f"db_connection:{clean_id}"
        existing = self.load_for_connection(connection_id=clean_id, user_id=user_id)
        if existing is not None and existing.status not in {"not_built", "failed"} and not force:
            return existing, None
        catalog = existing or SemanticCatalog(
            catalog_id=stable_id("semantic-catalog", source_key),
            connection_id=clean_id,
            source_key=source_key,
            source_type="db_connection",
            source_ref_id=clean_id,
            source_label=source_label,
            source_fingerprint=f"db:{clean_id}",
            status="not_built",
            user_id=user_id,
        )
        if existing is None:
            self.catalog_store.save_published_if_absent(self._for_source(catalog))
            catalog = self.load_for_connection(connection_id=clean_id, user_id=user_id) or catalog
        operation = self.catalog_store.claim_operation(
            source_key=source_key,
            catalog_id=catalog.catalog_id,
            connection_id=clean_id,
            operation_type="refresh" if existing is not None else "build",
            actor_user_id=user_id,
            force=force,
        )
        return catalog, operation

    def claim_session_build(
        self,
        *,
        session_id: str,
        user_id: int,
        force: bool = False,
        operation_type: str | None = None,
    ) -> tuple[SemanticCatalog, SemanticCatalogOperation | None]:
        state = self.store.load_session(session_id)
        if state is None:
            raise ValueError("Session not found")
        source_ref_id = str(state.source_ref_id or "").strip()
        snapshot = self.store.load_data_catalog(session_id) or DataCatalogSnapshot(
            source_fingerprint=source_ref_id
        )
        identity = self._source_identity(
            session_id=session_id,
            user_id=user_id,
            state=state,
            snapshot=snapshot,
        )
        existing = self._load_published(identity.source_key)
        if (
            operation_type is None
            and existing is not None
            and existing.status not in {"not_built", "failed"}
            and not force
        ):
            return self._for_session(existing, session_id=session_id, user_id=user_id), None
        catalog = existing or SemanticCatalog(
            catalog_id=stable_id("semantic-catalog", identity.source_key),
            source_key=identity.source_key,
            source_type=identity.source_type,
            source_ref_id=identity.source_ref_id,
            source_label=identity.source_label,
            source_fingerprint=identity.source_fingerprint,
            status="not_built",
            user_id=user_id,
            session_id=session_id,
        )
        if existing is None:
            self.catalog_store.save_published_if_absent(self._for_source(catalog))
            catalog = self._load_published(identity.source_key) or catalog
        operation = self.catalog_store.claim_operation(
            source_key=identity.source_key,
            catalog_id=catalog.catalog_id,
            connection_id=catalog.connection_id,
            operation_type=operation_type or ("refresh" if existing is not None else "build"),
            actor_user_id=user_id,
            force=force,
        )
        return self._for_session(catalog, session_id=session_id, user_id=user_id), operation

    def mark_build_failed(
        self,
        *,
        source_key: str,
        error: str,
        operation_id: int | None = None,
    ) -> None:
        if operation_id is not None:
            self.fail_operation(operation_id=operation_id, error=error)
        catalog = self._load_published(source_key)
        if catalog is None:
            return
        if catalog.status not in {"not_built", "pending", "indexing", "failed"}:
            return
        catalog.status = "failed"
        catalog.error = str(error)
        catalog.updated_at = utc_now_iso()
        self._save_published(catalog)

    def update_operation(
        self,
        *,
        operation_id: int,
        stage: str | None = None,
        status: str | None = None,
        error: str | None = None,
    ) -> SemanticCatalogOperation:
        operation = self.catalog_store.update_operation(
            operation_id,
            stage=stage,
            status=status,
            error=error,
        )
        if operation is None:
            raise RuntimeError("Semantic operation was cancelled or superseded")
        return operation

    def fail_operation(self, *, operation_id: int, error: str) -> None:
        self.catalog_store.update_operation(operation_id, status="failed", error=str(error))

    def latest_operation(self, *, source_key: str) -> SemanticCatalogOperation | None:
        operation = self.catalog_store.load_latest_operation(source_key)
        if operation is None or operation.status != "running":
            return operation
        if datetime.now(UTC) - _parse_utc(operation.updated_at) <= _OPERATION_STALE_AFTER:
            return operation
        return self.catalog_store.update_operation(
            operation.operation_id,
            status="interrupted",
            error="Semantic operation was interrupted. Retry it.",
        )

    def latest_operation_for_connection(
        self,
        *,
        connection_id: str,
    ) -> SemanticCatalogOperation | None:
        return self.latest_operation(source_key=f"db_connection:{connection_id}")

    def latest_operation_for_session(
        self,
        *,
        session_id: str,
        user_id: int,
    ) -> SemanticCatalogOperation | None:
        state = self.store.load_session(session_id)
        snapshot = self.store.load_data_catalog(session_id) or DataCatalogSnapshot(
            source_fingerprint=str(getattr(state, "source_ref_id", "") or "")
        )
        identity = self._source_identity(
            session_id=session_id,
            user_id=user_id,
            state=state,
            snapshot=snapshot,
        )
        return self.latest_operation(source_key=identity.source_key)

    def _advance_operation(self, operation_id: int | None, *, stage: str) -> None:
        if operation_id is None:
            return
        self.update_operation(operation_id=operation_id, stage=stage, error=None)

    def _complete_operation(self, operation_id: int | None) -> None:
        if operation_id is None:
            return
        self.update_operation(operation_id=operation_id, status="completed", error=None)

    def _advance_after_reindex(
        self,
        operation_id: int | None,
        catalog: SemanticCatalog,
    ) -> None:
        try:
            self._advance_operation(operation_id, stage="publishing")
        except RuntimeError:
            if self.vector_store is not None and getattr(self.vector_store, "enabled", False):
                self.vector_store.delete_catalog(
                    catalog,
                    published_version=catalog.published_version,
                )
            raise

    def _commit_build_result(
        self,
        *,
        operation_id: int | None,
        generated: SemanticCatalog,
        published: SemanticCatalog,
    ) -> None:
        published.updated_at = utc_now_iso()
        if operation_id is None:
            self._save_generated(generated)
            self._save_published(published)
            return
        if self.catalog_store.save_build_result_if_current(
            operation_id=operation_id,
            generated=self._for_source(generated),
            published=self._for_source(published),
        ):
            return
        if self.vector_store is not None and getattr(self.vector_store, "enabled", False):
            try:
                self.vector_store.delete_catalog(published)
            except Exception:
                logger.warning(
                    "Failed to clean cancelled semantic index for %s",
                    published.source_key,
                    exc_info=True,
                )
        raise RuntimeError("Semantic operation was cancelled or superseded")

    def load_for_connection(self, *, connection_id: str, user_id: int = 0) -> SemanticCatalog | None:
        catalog = self._load_published(f"db_connection:{connection_id}")
        if catalog is None:
            return None
        return catalog.model_copy(update={"user_id": user_id}, deep=True)

    def clear_for_connection(self, *, connection_id: str, user_id: int) -> None:
        _ = user_id
        source_key = f"db_connection:{connection_id}"
        catalog = self._load_published(source_key)
        self._delete_catalog(source_key=source_key, catalog=catalog)

    def mark_stale_for_connection(self, *, connection_id: str, reason: str) -> None:
        catalog = self._load_published(f"db_connection:{connection_id}")
        if catalog is None:
            return
        catalog.status = "stale"
        catalog.error = reason
        catalog.updated_at = utc_now_iso()
        self._save_published(catalog)

    def create_metric_for_connection(
        self,
        *,
        connection_id: str,
        user_id: int,
        payload: SemanticMetricCreate,
    ) -> SemanticMetric:
        catalog, overlay = self._connection_catalog_and_overlay(connection_id=connection_id, user_id=user_id)
        metric = SemanticMetric(**payload.model_dump(), metric_id=f"metric:{payload.key}")
        self._validate_metric(catalog, metric)
        overlay.metrics = [item for item in overlay.metrics if item.metric_id != metric.metric_id]
        overlay.metrics.append(metric)
        self._save_overlay(overlay)
        self._republish_from_overlay(catalog, overlay)
        return metric

    def update_metric_for_connection(
        self,
        *,
        connection_id: str,
        user_id: int,
        metric_id: str,
        payload: SemanticMetricUpdate,
    ) -> SemanticMetric:
        catalog, overlay = self._connection_catalog_and_overlay(connection_id=connection_id, user_id=user_id)
        current = next((item for item in overlay.metrics if item.metric_id == metric_id), None)
        if current is None:
            current = next((item for item in catalog.metrics if item.metric_id == metric_id), None)
        if current is None:
            raise ValueError("Metric not found")
        data = current.model_dump()
        data.update(payload.model_dump(exclude_unset=True))
        data["updated_at"] = utc_now_iso()
        updated = SemanticMetric.model_validate(data)
        self._validate_metric(catalog, updated)
        overlay.metrics = [updated if item.metric_id == metric_id else item for item in overlay.metrics]
        if all(item.metric_id != metric_id for item in overlay.metrics):
            overlay.metrics.append(updated)
        self._save_overlay(overlay)
        self._republish_from_overlay(catalog, overlay)
        return updated

    def delete_metric_for_connection(self, *, connection_id: str, user_id: int, metric_id: str) -> None:
        catalog, overlay = self._connection_catalog_and_overlay(connection_id=connection_id, user_id=user_id)
        self._validate_metric_delete(catalog, metric_id)
        overlay.metrics = [item for item in overlay.metrics if item.metric_id != metric_id]
        self._save_overlay(overlay)
        self._republish_from_overlay(catalog, overlay)

    def create_relationship_for_connection(
        self,
        *,
        connection_id: str,
        user_id: int,
        payload: SemanticRelationshipCreate,
    ) -> SemanticRelationship:
        catalog, overlay = self._connection_catalog_and_overlay(connection_id=connection_id, user_id=user_id)
        relationship_key = stable_id(
            catalog.source_key,
            payload.from_table,
            payload.from_column,
            payload.to_table,
            payload.to_column,
        )
        relationship = SemanticRelationship(
            **payload.model_dump(),
            relationship_id=f"relationship:{relationship_key}",
        )
        self._validate_relationship(catalog, relationship)
        overlay.relationships = [
            item for item in overlay.relationships if item.relationship_id != relationship.relationship_id
        ]
        overlay.relationships.append(relationship)
        self._save_overlay(overlay)
        self._republish_from_overlay(catalog, overlay)
        return relationship

    def update_relationship_for_connection(
        self,
        *,
        connection_id: str,
        user_id: int,
        relationship_id: str,
        payload: SemanticRelationshipUpdate,
    ) -> SemanticRelationship:
        catalog, overlay = self._connection_catalog_and_overlay(connection_id=connection_id, user_id=user_id)
        current = next(
            (item for item in overlay.relationships if item.relationship_id == relationship_id),
            None,
        )
        if current is None:
            current = next(
                (item for item in catalog.relationships if item.relationship_id == relationship_id),
                None,
            )
        if current is None:
            raise ValueError("Relationship not found")

        data = current.model_dump()
        data.update(payload.model_dump(exclude_unset=True))
        updated = SemanticRelationship.model_validate(data)
        self._validate_relationship(catalog, updated)
        overlay.relationships = [
            updated if item.relationship_id == relationship_id else item for item in overlay.relationships
        ]
        if all(item.relationship_id != relationship_id for item in overlay.relationships):
            overlay.relationships.append(updated)
        self._save_overlay(overlay)
        self._republish_from_overlay(catalog, overlay)
        return updated

    def delete_relationship_for_connection(
        self,
        *,
        connection_id: str,
        user_id: int,
        relationship_id: str,
    ) -> None:
        catalog, overlay = self._connection_catalog_and_overlay(connection_id=connection_id, user_id=user_id)
        overlay.relationships = [
            item for item in overlay.relationships if item.relationship_id != relationship_id
        ]
        self._save_overlay(overlay)
        self._republish_from_overlay(catalog, overlay)

    def search_for_connection(
        self,
        *,
        connection_id: str,
        user_id: int,
        query: str,
        top_k: int = 8,
    ):
        from backend.data_access.semantic_context import SemanticContextBuilder

        catalog = self.load_for_connection(connection_id=connection_id, user_id=user_id)
        if catalog is None:
            return SemanticContextBuilder(
                store=self.store,
                vector_store=self.vector_store,
                catalog_service=None,
                top_k=top_k,
            ).build(session_id="", user_id=user_id, query=query)
        items = []
        search_error: str | None = None
        if self.vector_store is not None and getattr(self.vector_store, "enabled", False):
            try:
                items = list(self.vector_store.search(catalog=catalog, query=query, top_k=top_k))
            except Exception as exc:
                search_error = str(exc)
        if not items:
            items = SemanticContextBuilder._lexical_search(  # noqa: SLF001
                catalog=catalog,
                query=query,
                top_k=top_k,
            )
        context_catalog = (
            catalog.model_copy(
                update={"status": "degraded", "error": f"Qdrant search failed: {search_error}"},
                deep=True,
            )
            if search_error
            else catalog
        )
        return SemanticContextBuilder(
            store=self.store,
            vector_store=None,
            catalog_service=None,
            top_k=top_k,
        ).build_from_catalog(catalog=context_catalog, query=query, items=items)

    def status_for_connection(self, *, connection_id: str) -> SemanticCatalog:
        catalog = self._load_published(f"db_connection:{connection_id}")
        if catalog is not None:
            return catalog
        return SemanticCatalog(
            catalog_id=stable_id("semantic-catalog", f"db_connection:{connection_id}"),
            connection_id=connection_id,
            source_key=f"db_connection:{connection_id}",
            source_type="db_connection",
            source_ref_id=connection_id,
            status="not_built",
        )

    def _connection_catalog_and_overlay(
        self,
        *,
        connection_id: str,
        user_id: int,
    ) -> tuple[SemanticCatalog, SemanticCatalogOverlay]:
        catalog = self.load_for_connection(connection_id=connection_id, user_id=user_id)
        if catalog is None:
            raise ValueError("Semantic catalog not built")
        return catalog, self._load_overlay(catalog.source_key)

    def load_for_session(self, *, session_id: str, user_id: int) -> SemanticCatalog | None:
        state = self.store.load_session(session_id)
        source_type = str(getattr(state, "source_type", "") or "").strip().lower()
        source_ref_id = str(getattr(state, "source_ref_id", "") or "").strip()
        if source_type == "db_connection" and source_ref_id:
            catalog = self.load_for_connection(connection_id=source_ref_id, user_id=user_id)
            if catalog is not None:
                catalog = self._with_current_global_terms(catalog)
                return self._for_session(catalog, session_id=session_id, user_id=user_id)

        if source_type == "csv" and source_ref_id:
            pending_identity = self._source_identity(
                session_id=session_id,
                user_id=user_id,
                state=state,
                snapshot=DataCatalogSnapshot(source_fingerprint=source_ref_id),
            )
            pending = self._load_published(pending_identity.source_key)
            if pending is not None:
                pending = self._with_current_global_terms(pending)
                return self._for_session(pending, session_id=session_id, user_id=user_id)

        snapshot = self.store.load_data_catalog(session_id)
        if snapshot is None:
            return self._unbound_catalog(session_id=session_id, user_id=user_id, state=state)

        identity = self._source_identity(
            session_id=session_id,
            user_id=user_id,
            state=state,
            snapshot=snapshot,
        )
        catalog = self._load_published(identity.source_key)
        if catalog is None:
            return None

        if catalog.source_fingerprint != identity.source_fingerprint:
            catalog.status = "stale"
            catalog.error = "Data catalog source fingerprint changed."
            catalog.updated_at = utc_now_iso()
            self._save_published(catalog)
        catalog = self._with_current_global_terms(catalog)
        return self._for_session(catalog, session_id=session_id, user_id=user_id)

    def clear_for_session(self, *, session_id: str, user_id: int) -> None:
        state = self.store.load_session(session_id)
        source_ref_id = str(getattr(state, "source_ref_id", "") or "").strip()
        source_type = str(getattr(state, "source_type", "") or "").strip()
        snapshot = self.store.load_data_catalog(session_id)
        if not source_type and snapshot is None:
            return
        snapshot = snapshot or DataCatalogSnapshot(source_fingerprint=source_ref_id)
        identity = self._source_identity(
            session_id=session_id,
            user_id=user_id,
            state=state,
            snapshot=snapshot,
        )
        catalog = self._load_published(identity.source_key)
        self._delete_catalog(source_key=identity.source_key, catalog=catalog)

    def clear_source(self, source_key: str) -> None:
        clean_source_key = str(source_key or "").strip()
        if not clean_source_key or clean_source_key == GLOBAL_SOURCE_KEY:
            return
        catalog = self._load_published(clean_source_key)
        self._delete_catalog(source_key=clean_source_key, catalog=catalog)

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
        self._validate_metric_delete(catalog, metric_id)
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
        relationship_key = stable_id(
            catalog.source_key,
            payload.from_table,
            payload.from_column,
            payload.to_table,
            payload.to_column,
        )
        relationship = SemanticRelationship(
            **payload.model_dump(),
            relationship_id=f"relationship:{relationship_key}",
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
        current = next(
            (item for item in overlay.relationships if item.relationship_id == relationship_id),
            None,
        )
        if current is None:
            current = next(
                (item for item in catalog.relationships if item.relationship_id == relationship_id),
                None,
            )
        if current is None:
            raise ValueError("Relationship not found")

        data = current.model_dump()
        data.update(payload.model_dump(exclude_unset=True))
        updated = SemanticRelationship.model_validate(data)
        self._validate_relationship(catalog, updated)
        overlay.relationships = [
            updated if item.relationship_id == relationship_id else item for item in overlay.relationships
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
        catalog, overlay = self._catalog_and_overlay(session_id=session_id, user_id=user_id)
        term = SemanticTerm(
            **payload.model_dump(),
            term_id=f"term:{stable_id('term', catalog.source_key, payload.name)}",
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
        catalog, overlay = self._catalog_and_overlay(session_id=session_id, user_id=user_id)
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
        catalog, overlay = self._catalog_and_overlay(session_id=session_id, user_id=user_id)
        if all(item.term_id != term_id for item in overlay.terms):
            raise ValueError("Term not found or is read-only")
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

    def validate_metric_candidate(self, catalog: SemanticCatalog, metric: SemanticMetric) -> None:
        self._validate_metric(catalog, metric)

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
        replace_metrics: bool = False,
        operation_id: int | None = None,
    ) -> tuple[SemanticCatalog, list[str]]:
        if operation_id is not None:
            self._advance_operation(operation_id, stage="publishing")
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
        working_catalog = catalog.model_copy(deep=True)
        pending_metrics = list(metrics or [])
        # ponytail: generated batches are small; retrying avoids a dependency-graph abstraction.
        while pending_metrics:
            failures: list[tuple[SemanticMetric, ValueError]] = []
            for metric in pending_metrics:
                try:
                    self._validate_metric(working_catalog, metric)
                except ValueError as exc:
                    failures.append((metric, exc))
                    continue
                metric_exists = any(item.metric_id == metric.metric_id for item in overlay.metrics)
                if metric_exists and not replace_metrics:
                    continue
                overlay.metrics = [item for item in overlay.metrics if item.metric_id != metric.metric_id]
                overlay.metrics.append(metric)
                working_catalog.metrics = [
                    item for item in working_catalog.metrics if item.metric_id != metric.metric_id
                ]
                working_catalog.metrics.append(metric)
            if len(failures) == len(pending_metrics):
                rejected.extend(f"Metric {metric.key}: {exc}" for metric, exc in failures)
                break
            pending_metrics = [metric for metric, _exc in failures]
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
        if operation_id is None:
            self._save_overlay(overlay)
            return self._republish_from_overlay(catalog, overlay), rejected

        self._advance_operation(operation_id, stage="indexing")
        overlay.version = int(overlay.version or 0) + 1
        overlay.updated_at = utc_now_iso()
        generated = self._load_generated(catalog.source_key) or self._for_source(catalog)
        published = self._publish(generated=generated, overlay=overlay)
        self._apply_validation_status(published)
        self._try_reindex(published)
        self._advance_after_reindex(operation_id, published)
        if not self.catalog_store.save_generation_result_if_current(
            operation_id=operation_id,
            overlay=overlay,
            published=published,
        ):
            if self.vector_store is not None and getattr(self.vector_store, "enabled", False):
                self.vector_store.delete_catalog(
                    published,
                    published_version=published.published_version,
                )
            raise RuntimeError("Semantic generation was cancelled before publication")
        self._complete_operation(operation_id)
        return self._for_session(published, session_id=session_id, user_id=user_id), rejected

    def _catalog_and_overlay(
        self,
        *,
        session_id: str,
        user_id: int,
    ) -> tuple[SemanticCatalog, SemanticCatalogOverlay]:
        catalog = self.load_for_session(session_id=session_id, user_id=user_id)
        if catalog is None:
            catalog = self.refresh(session_id=session_id, user_id=user_id)
        if catalog.status == "unbound" or catalog.source_key == GLOBAL_SOURCE_KEY:
            raise ValueError("Bind a data source before editing its semantic catalog")
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
        catalog_id = stable_id("semantic-catalog", identity.source_key)
        tables = [self._semantic_table(table) for table in snapshot.tables]
        columns = [
            self._semantic_column(table, column) for table in snapshot.tables for column in table.columns
        ]
        entities = self._semantic_entities(columns)
        dimensions = self._semantic_dimensions(columns)
        facts = self._semantic_facts(columns)
        now = utc_now_iso()
        profile_error = "; ".join(snapshot.errors[:3]) or None
        return SemanticCatalog(
            catalog_id=catalog_id,
            connection_id=identity.source_ref_id if identity.source_type == "db_connection" else "",
            source_key=identity.source_key,
            user_id=user_id,
            session_id=session_id,
            source_type=identity.source_type,
            source_ref_id=identity.source_ref_id,
            source_label=identity.source_label,
            source_fingerprint=identity.source_fingerprint,
            profile_sample_strategy=snapshot.profile_sample_strategy,
            profile_sample_limit=snapshot.profile_sample_limit,
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
            self._apply_table_patch(table, table_patches.get(table.table_id)) for table in catalog.tables
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
        catalog.terms = _dedupe_terms(list(overlay.terms) + self._applicable_global_terms(catalog))
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
        self._try_reindex(published)
        self._save_published(published)
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
        # User-authored terms are source-scoped. Legacy global custom terms remain
        # stored for manual migration but are never exposed across users or sources.
        return starter_terms(GLOBAL_SOURCE_KEY)

    def _applicable_global_terms(self, catalog: SemanticCatalog) -> list[SemanticTerm]:
        """Expose starter glossary terms only when their targets exist in this source."""

        starter_ids = {term.term_id for term in starter_terms(GLOBAL_SOURCE_KEY)}
        known_refs = {metric.metric_id for metric in catalog.metrics}
        known_refs.update(entity.entity_id for entity in catalog.entities)
        known_refs.update(dimension.dimension_id for dimension in catalog.dimensions)
        known_refs.update(fact.fact_id for fact in catalog.facts)
        known_refs.update(table.table_id for table in catalog.tables)
        known_refs.update(table.qualified_name for table in catalog.tables)
        known_refs.update(column.column_id for column in catalog.columns)
        known_refs.update(f"{column.table}.{column.name}" for column in catalog.columns)
        return [
            term
            for term in self._global_terms()
            if term.term_id not in starter_ids
            or not term.entity_refs
            or bool(set(term.entity_refs) & known_refs)
        ]

    def _with_current_global_terms(self, catalog: SemanticCatalog) -> SemanticCatalog:
        if catalog.source_key == GLOBAL_SOURCE_KEY:
            return catalog
        overlay = self._load_overlay(catalog.source_key)
        catalog = catalog.model_copy(deep=True)
        catalog.terms = _dedupe_terms(list(overlay.terms) + self._applicable_global_terms(catalog))
        return catalog

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
            semantic_role=self._column_role(
                column.name,
                column.dtype,
                distinct_count=column.distinct_count,
            ),
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

    def _column_role(
        self,
        name: str,
        dtype: str,
        *,
        distinct_count: int | None = None,
    ) -> SemanticColumnRole:
        text = f"{name} {dtype}"
        dtype_l = str(dtype or "").lower()
        if _TIME_RE.search(text) or "date" in dtype_l or "time" in dtype_l:
            return "time"
        if _ID_RE.search(name):
            return "identifier"
        if dtype_l in {"bool", "boolean"}:
            return "flag"
        if any(kind in dtype_l for kind in ("int", "float", "double", "decimal", "numeric")):
            if _METRIC_RE.search(text):
                return "metric_candidate"
            if distinct_count is not None and distinct_count <= 30:
                return "dimension"
            return "metric_candidate"
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
            kind = (
                "time"
                if column.semantic_role == "time"
                else "boolean"
                if column.semantic_role == "flag"
                else "categorical"
            )
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
        table_columns = {column.name for column in catalog.columns if column.table == table.qualified_name}
        if metric.type == "simple" and metric.expr and metric.expr not in table_columns:
            raise ValueError(f"Unknown metric column: {metric.expr}")
        if metric.default_time_dimension:
            time_ref = metric.default_time_dimension
            exact = [
                dimension
                for dimension in catalog.dimensions
                if dimension.is_active
                and time_ref in {f"{dimension.table}.{dimension.name}", dimension.dimension_id}
            ]
            candidates = exact or [
                dimension
                for dimension in catalog.dimensions
                if dimension.is_active and dimension.name == time_ref
            ]
            local = [dimension for dimension in candidates if dimension.table == table.qualified_name]
            if len(local) == 1:
                time_dimension = local[0]
            elif len(candidates) == 1:
                time_dimension = candidates[0]
            elif candidates:
                raise ValueError(f"Ambiguous metric time dimension: {time_ref}")
            else:
                raise ValueError(f"Unknown active metric time dimension: {time_ref}")
            if time_dimension.type != "time":
                raise ValueError(f"Metric time dimension is not type=time: {time_ref}")
        for item in metric.filters:
            field = item.field
            if field.startswith(f"{table.qualified_name}."):
                field = field[len(table.qualified_name) + 1 :]
            if field not in table_columns:
                raise ValueError(f"Unknown metric filter column: {item.field}")
        dimension_names = {
            ref
            for dim in catalog.dimensions
            for ref in (dim.name, f"{dim.table}.{dim.name}", dim.dimension_id)
        }
        missing_dimensions = [item for item in metric.allowed_dimensions if item not in dimension_names]
        if missing_dimensions:
            raise ValueError(f"Unknown metric dimensions: {', '.join(missing_dimensions)}")
        metric_keys = {item.key for item in catalog.metrics if item.metric_id != metric.metric_id}
        for ref in [metric.numerator, metric.denominator]:
            if ref and ref not in metric_keys:
                raise ValueError(f"Unknown metric reference: {ref}")
        if metric.type == "derived":
            allowed_columns = set(table_columns)
            allowed_columns.update(f"{table.qualified_name}.{column}" for column in table_columns)
            allowed_columns.update(metric_keys)
            self._validate_metric_sql(
                metric.formula,
                allowed_columns,
                aggregate_refs=metric_keys,
            )
        candidate_catalog = catalog.model_copy(deep=True)
        candidate_catalog.metrics = [
            item for item in candidate_catalog.metrics if item.metric_id != metric.metric_id
        ]
        candidate_catalog.metrics.append(metric)
        existing_errors = {(issue.object_id, issue.message) for issue in metric_dependency_errors(catalog)}
        new_errors = [
            issue
            for issue in metric_dependency_errors(candidate_catalog)
            if (issue.object_id, issue.message) not in existing_errors
        ]
        if new_errors:
            raise ValueError(new_errors[0].message)

    @staticmethod
    def _validate_metric_delete(catalog: SemanticCatalog, metric_id: str) -> None:
        target = next((metric for metric in catalog.metrics if metric.metric_id == metric_id), None)
        if target is None:
            return
        dependents = sorted(
            metric.key
            for metric in catalog.metrics
            if metric.is_active and metric.metric_id != metric_id and target.key in metric_references(metric)
        )
        if dependents:
            raise ValueError(
                f"Метрику {target.key} нельзя удалить: от неё зависят активные метрики: "
                f"{', '.join(dependents)}. Сначала измените или удалите зависимые метрики."
            )

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
                if table.table_id == needle or table.qualified_name == needle or table.table_name == needle
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

    def _validate_metric_sql(
        self,
        sql: str,
        allowed_columns: set[str],
        *,
        aggregate_refs: set[str],
    ) -> None:
        if ";" in sql or _SQL_COMMENT_RE.search(sql) or _FORBIDDEN_SQL_RE.search(sql):
            raise ValueError("Only read-only aggregate expressions are allowed")
        qualified_refs = set(_QUALIFIED_SQL_IDENT_RE.findall(sql))
        unknown_qualified = qualified_refs - allowed_columns
        if unknown_qualified:
            raise ValueError(f"Unknown metric SQL identifiers: {', '.join(sorted(unknown_qualified))}")
        unqualified_sql = sql
        for ref in qualified_refs:
            unqualified_sql = re.sub(rf"\b{re.escape(ref)}\b", "", unqualified_sql)
        identifiers = set(_SQL_IDENT_RE.findall(unqualified_sql))
        if not (
            re.search(r"\b(sum|avg|count|min|max|nullif)\s*\(", sql, re.I) or identifiers & aggregate_refs
        ):
            raise ValueError("Only read-only aggregate expressions are allowed")
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
            item for item in identifiers if item.lower() not in allowed_words and item not in allowed_columns
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
            logger.warning("Semantic catalog indexing failed: %s", exc)

    def _delete_catalog(self, *, source_key: str, catalog: SemanticCatalog | None) -> None:
        self.catalog_store.delete_source(source_key)
        if (
            catalog is not None
            and self.vector_store is not None
            and getattr(self.vector_store, "enabled", False)
        ):
            try:
                self.vector_store.delete_catalog(catalog)
            except Exception:
                logger.warning(
                    "Semantic catalog %s was deleted, but vector cleanup failed",
                    source_key,
                    exc_info=True,
                )

    def _source_identity(
        self,
        *,
        session_id: str,
        user_id: int,
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
            source_key = "csv:" + stable_id("csv-source", user_id, source_fingerprint)
        elif source_type == "planfact":
            source_key = "planfact:" + stable_id("planfact-source", user_id, session_id, source_ref_id)
            source_fingerprint = snapshot.source_fingerprint or source_ref_id
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
        catalog = self.catalog_store.load_published(source_key)
        if catalog is None:
            return None
        catalog.overlay_version = int(self.catalog_store.load_overlay(source_key).version or 0)
        return catalog

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


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _csv_source_fingerprint(
    snapshot_fingerprint: str,
    source_ref_id: str,
    schema_sig: str,
) -> str:
    for value in (snapshot_fingerprint, source_ref_id):
        clean = str(value or "").strip()
        if clean.startswith("csv:sha256:") or clean.startswith("sha256:"):
            return clean.removeprefix("csv:")
    return schema_sig


def _dedupe_terms(terms: list[SemanticTerm]) -> list[SemanticTerm]:
    groups: list[dict[str, Any]] = []
    for term in terms:
        name = _term_label(term.name)
        labels = {label for label in [_term_label(term.name), *map(_term_label, term.synonyms)] if label}
        matching = [
            index
            for index, group in enumerate(groups)
            if name in group["labels"] or bool(group["names"] & labels)
        ]
        if not matching:
            groups.append({"terms": [term], "names": {name}, "labels": labels})
            continue
        first = matching[0]
        groups[first]["terms"].append(term)
        groups[first]["names"].add(name)
        groups[first]["labels"].update(labels)
        for index in reversed(matching[1:]):
            groups[first]["terms"].extend(groups[index]["terms"])
            groups[first]["names"].update(groups[index]["names"])
            groups[first]["labels"].update(groups[index]["labels"])
            del groups[index]
    starter_ids = {term.term_id for term in starter_terms(GLOBAL_SOURCE_KEY)}
    return [_merge_term_group(group["terms"], starter_ids) for group in groups]


def _term_label(value: str) -> str:
    return re.sub(r"[\W_]+", " ", str(value or "").casefold(), flags=re.UNICODE).strip()


def _merge_term_group(terms: list[SemanticTerm], starter_ids: set[str]) -> SemanticTerm:
    canonical = max(
        terms,
        key=lambda term: (
            int(term.term_id not in starter_ids),
            int("_" not in term.name and "*" not in term.name),
            int(bool(re.search(r"\s", term.name))),
            len(term.description or ""),
            len(term.entity_refs),
        ),
    )
    synonyms = clean_list(
        [
            re.sub(r"\*+", " ", value).strip()
            for term in terms
            for value in [term.name, *term.synonyms]
            if _term_label(value) != _term_label(canonical.name)
        ]
    )
    return SemanticTerm.model_validate(
        {
            **canonical.model_dump(),
            "description": canonical.description,
            "synonyms": synonyms,
            "entity_refs": clean_list([ref for term in terms for ref in term.entity_refs]),
        }
    )


def _column_names_table_entity(table_name: str, entity_name: str) -> bool:
    table_leaf = str(table_name or "").split(".")[-1].lower()
    entity = str(entity_name or "").lower()
    table_entities = {table_leaf, "id"}
    if table_leaf.endswith("s"):
        table_entities.add(table_leaf[:-1])
    if table_leaf.endswith("es"):
        table_entities.add(table_leaf[:-2])
    return entity in table_entities or table_leaf.startswith(f"{entity}_")
