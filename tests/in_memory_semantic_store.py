from __future__ import annotations

import threading

from backend.data_access.data_catalog import DataCatalogSnapshot
from backend.data_access.semantic_models import (
    SemanticCatalog,
    SemanticCatalogOperation,
    SemanticCatalogOverlay,
    utc_now_iso,
)
from backend.data_access.semantic_scenario_models import SemanticScenarioReview
from backend.sessions.session_store import SessionStore


class InMemorySemanticCatalogStore:
    """Test double for the mandatory PostgreSQL metadata store."""

    def __init__(self) -> None:
        self.profiles: dict[str, DataCatalogSnapshot] = {}
        self.generated: dict[str, SemanticCatalog] = {}
        self.published: dict[str, SemanticCatalog] = {}
        self.overlays: dict[str, SemanticCatalogOverlay] = {}
        self.reviews: dict[tuple[str, str], SemanticScenarioReview] = {}
        self.operations: dict[int, SemanticCatalogOperation] = {}
        self._next_operation_id = 1
        self._lock = threading.RLock()

    def save_data_profile(self, session_id: str, snapshot: DataCatalogSnapshot) -> None:
        self.profiles[session_id] = DataCatalogSnapshot.from_dict(snapshot.to_dict())

    def load_data_profile(self, session_id: str) -> DataCatalogSnapshot | None:
        snapshot = self.profiles.get(session_id)
        return DataCatalogSnapshot.from_dict(snapshot.to_dict()) if snapshot else None

    def delete_data_profile(self, session_id: str) -> None:
        self.profiles.pop(session_id, None)

    def save_generated(self, catalog: SemanticCatalog) -> None:
        self.generated[catalog.source_key] = catalog.model_copy(deep=True)

    def load_generated(self, source_key: str) -> SemanticCatalog | None:
        catalog = self.generated.get(source_key)
        return catalog.model_copy(deep=True) if catalog else None

    def save_published(self, catalog: SemanticCatalog) -> None:
        self.published[catalog.source_key] = catalog.model_copy(deep=True)

    def save_published_if_absent(self, catalog: SemanticCatalog) -> bool:
        with self._lock:
            if catalog.source_key in self.published:
                return False
            self.save_published(catalog)
            return True

    def load_published(self, source_key: str) -> SemanticCatalog | None:
        catalog = self.published.get(source_key)
        return catalog.model_copy(deep=True) if catalog else None

    def load_overlay(self, source_key: str) -> SemanticCatalogOverlay:
        overlay = self.overlays.get(source_key)
        return overlay.model_copy(deep=True) if overlay else SemanticCatalogOverlay(source_key=source_key)

    def save_overlay(self, overlay: SemanticCatalogOverlay) -> None:
        overlay.version = int(overlay.version or 0) + 1
        overlay.updated_at = utc_now_iso()
        self.overlays[overlay.source_key] = overlay.model_copy(deep=True)

    def save_scenario_review(self, review: SemanticScenarioReview) -> None:
        self.reviews[(review.source_key, review.review_id)] = review.model_copy(deep=True)

    def load_scenario_review(
        self,
        source_key: str,
        review_id: str,
    ) -> SemanticScenarioReview | None:
        review = self.reviews.get((source_key, review_id))
        return review.model_copy(deep=True) if review else None

    def claim_operation(
        self,
        *,
        source_key: str,
        catalog_id: str,
        connection_id: str,
        operation_type: str,
        actor_user_id: int,
        force: bool = False,
    ) -> SemanticCatalogOperation | None:
        with self._lock:
            active = next(
                (
                    item
                    for item in reversed(list(self.operations.values()))
                    if item.source_key == source_key and item.status == "running"
                ),
                None,
            )
            if active is not None and not force:
                return None
            if active is not None:
                self.update_operation(
                    active.operation_id,
                    status="cancelled",
                    error="Superseded by a newer semantic operation.",
                )
            operation = SemanticCatalogOperation(
                operation_id=self._next_operation_id,
                source_key=source_key,
                catalog_id=catalog_id,
                connection_id=connection_id,
                operation_type=operation_type,
                actor_user_id=actor_user_id,
            )
            self._next_operation_id += 1
            self.operations[operation.operation_id] = operation
            return operation.model_copy(deep=True)

    def load_latest_operation(self, source_key: str) -> SemanticCatalogOperation | None:
        with self._lock:
            operation = next(
                (item for item in reversed(list(self.operations.values())) if item.source_key == source_key),
                None,
            )
            return operation.model_copy(deep=True) if operation else None

    def update_operation(
        self,
        operation_id: int,
        *,
        stage: str | None = None,
        status: str | None = None,
        error: str | None = None,
    ) -> SemanticCatalogOperation | None:
        with self._lock:
            current = self.operations.get(operation_id)
            if current is None or current.status != "running":
                return None
            now = utc_now_iso()
            updated = current.model_copy(
                update={
                    "stage": stage or current.stage,
                    "status": status or current.status,
                    "error": error,
                    "updated_at": now,
                    "finished_at": (
                        now
                        if status in {"completed", "failed", "cancelled", "interrupted"}
                        else current.finished_at
                    ),
                },
                deep=True,
            )
            self.operations[operation_id] = updated
            return updated.model_copy(deep=True)

    def save_build_result_if_current(
        self,
        *,
        operation_id: int,
        generated: SemanticCatalog,
        published: SemanticCatalog,
    ) -> bool:
        with self._lock:
            operation = self.operations.get(operation_id)
            if (
                operation is None
                or operation.status != "running"
                or operation.source_key != published.source_key
            ):
                return False
            self.save_generated(generated)
            self.save_published(published)
            return True

    def save_generation_result_if_current(
        self,
        *,
        operation_id: int,
        overlay: SemanticCatalogOverlay,
        published: SemanticCatalog,
    ) -> bool:
        with self._lock:
            operation = self.operations.get(operation_id)
            if (
                operation is None
                or operation.status != "running"
                or operation.source_key != published.source_key
            ):
                return False
            self.overlays[overlay.source_key] = overlay.model_copy(deep=True)
            self.save_published(published)
            return True

    def cancel_operations(self, source_key: str) -> None:
        with self._lock:
            for operation in list(self.operations.values()):
                if operation.source_key == source_key and operation.status == "running":
                    self.update_operation(
                        operation.operation_id,
                        status="cancelled",
                        error="Semantic catalog was cleared.",
                    )

    def delete_source(self, source_key: str) -> None:
        with self._lock:
            self.cancel_operations(source_key)
            self.generated.pop(source_key, None)
            self.published.pop(source_key, None)
            self.overlays.pop(source_key, None)
            for key in [key for key in self.reviews if key[0] == source_key]:
                self.reviews.pop(key)


class SemanticSessionStore(SessionStore):
    def __init__(self, root_dir: str, ttl_days: int) -> None:
        super().__init__(
            root_dir,
            ttl_days=ttl_days,
            data_catalog_store=InMemorySemanticCatalogStore(),
        )
