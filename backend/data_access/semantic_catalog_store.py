from __future__ import annotations

import json
import logging
import re
from contextlib import nullcontext
from typing import Any, Protocol

from backend.data_access.data_catalog import DataCatalogSnapshot
from backend.data_access.semantic_models import (
    SemanticCatalog,
    SemanticCatalogOperation,
    SemanticCatalogOverlay,
    SemanticDimension,
    SemanticEntity,
    SemanticFact,
    stable_id,
    utc_now_iso,
)
from backend.data_access.semantic_scenario_models import SemanticScenarioReview

_DOCUMENT_TABLE_NAME = "semantic_catalog_documents"
logger = logging.getLogger(__name__)


class SemanticCatalogStore(Protocol):
    def save_data_profile(self, session_id: str, snapshot: DataCatalogSnapshot) -> None: ...
    def load_data_profile(self, session_id: str) -> DataCatalogSnapshot | None: ...
    def delete_data_profile(self, session_id: str) -> None: ...
    def save_generated(self, catalog: SemanticCatalog) -> None: ...
    def load_generated(self, source_key: str) -> SemanticCatalog | None: ...
    def save_published(self, catalog: SemanticCatalog) -> None: ...
    def save_published_if_absent(self, catalog: SemanticCatalog) -> bool: ...
    def load_published(self, source_key: str) -> SemanticCatalog | None: ...
    def load_overlay(self, source_key: str) -> SemanticCatalogOverlay: ...
    def save_overlay(self, overlay: SemanticCatalogOverlay) -> None: ...
    def save_scenario_review(self, review: SemanticScenarioReview) -> None: ...
    def load_scenario_review(self, source_key: str, review_id: str) -> SemanticScenarioReview | None: ...
    def claim_operation(
        self,
        *,
        source_key: str,
        catalog_id: str,
        connection_id: str,
        operation_type: str,
        actor_user_id: int,
        force: bool = False,
    ) -> SemanticCatalogOperation | None: ...
    def load_latest_operation(self, source_key: str) -> SemanticCatalogOperation | None: ...
    def update_operation(
        self,
        operation_id: int,
        *,
        stage: str | None = None,
        status: str | None = None,
        error: str | None = None,
    ) -> SemanticCatalogOperation | None: ...
    def save_build_result_if_current(
        self,
        *,
        operation_id: int,
        generated: SemanticCatalog,
        published: SemanticCatalog,
    ) -> bool: ...
    def save_generation_result_if_current(
        self,
        *,
        operation_id: int,
        overlay: SemanticCatalogOverlay,
        published: SemanticCatalog,
    ) -> bool: ...
    def cancel_operations(self, source_key: str) -> None: ...
    def delete_source(self, source_key: str) -> None: ...


class SemanticCatalogPostgresStore:
    def __init__(self, dsn: str, *, schema: str = "public") -> None:
        clean = str(dsn or "").strip()
        if not clean:
            raise ValueError("SEMANTIC_METADATA_DATABASE_URL must be set")
        clean_schema = str(schema or "public").strip() or "public"
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", clean_schema):
            raise ValueError("SEMANTIC_METADATA_SCHEMA must be a valid PostgreSQL identifier")
        self.dsn = clean
        self.schema = clean_schema
        self._ready = False

    @staticmethod
    def _profile_source_key(session_id: str) -> str:
        return f"session:{session_id}"

    def save_data_profile(self, session_id: str, snapshot: DataCatalogSnapshot) -> None:
        self._save(
            "data_profile",
            self._profile_source_key(session_id),
            snapshot.to_dict(),
            snapshot,
        )

    def load_data_profile(self, session_id: str) -> DataCatalogSnapshot | None:
        payload = self._load("data_profile", self._profile_source_key(session_id))
        return DataCatalogSnapshot.from_dict(payload) if payload is not None else None

    def delete_data_profile(self, session_id: str) -> None:
        import psycopg

        with psycopg.connect(self.dsn) as conn:
            self._configure_connection(conn)
            self._ensure_schema(conn)
            conn.execute(
                f"DELETE FROM {_DOCUMENT_TABLE_NAME} WHERE source_key = %s AND doc_type = %s",
                (self._profile_source_key(session_id), "data_profile"),
            )

    def save_generated(self, catalog: SemanticCatalog) -> None:
        self._save("generated_catalog", catalog.source_key, catalog.model_dump(), catalog)

    def load_generated(self, source_key: str) -> SemanticCatalog | None:
        return _model_from_payload(self._load("generated_catalog", source_key), SemanticCatalog)

    def save_published(self, catalog: SemanticCatalog) -> None:
        self._save_catalog(catalog)

    def save_published_if_absent(self, catalog: SemanticCatalog) -> bool:
        import psycopg
        from psycopg.types.json import Jsonb

        with psycopg.connect(self.dsn) as conn:
            self._configure_connection(conn)
            self._ensure_schema(conn)
            row = conn.execute(
                """
                INSERT INTO semantic_catalogs(
                    id, source_key, connection_id, source_type, source_ref_id,
                    source_label, source_fingerprint, status, error,
                    built_at, updated_at, built_by_user_id, updated_by_user_id,
                    validation_json, overlay_version, published_version
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (source_key) DO NOTHING
                RETURNING source_key
                """,
                (
                    catalog.catalog_id,
                    catalog.source_key,
                    catalog.connection_id,
                    catalog.source_type,
                    catalog.source_ref_id,
                    catalog.source_label,
                    catalog.source_fingerprint,
                    catalog.status,
                    catalog.error,
                    catalog.built_at,
                    catalog.updated_at,
                    catalog.user_id or None,
                    catalog.user_id or None,
                    Jsonb(catalog.validation.model_dump()),
                    catalog.overlay_version,
                    catalog.published_version,
                ),
            ).fetchone()
        return row is not None

    def load_published(self, source_key: str) -> SemanticCatalog | None:
        return self._load_catalog(source_key)

    def load_overlay(self, source_key: str) -> SemanticCatalogOverlay:
        overlay = _model_from_payload(self._load("overlay", source_key), SemanticCatalogOverlay)
        return overlay or SemanticCatalogOverlay(source_key=source_key)

    def save_overlay(self, overlay: SemanticCatalogOverlay) -> None:
        overlay.version = int(overlay.version or 0) + 1
        overlay.updated_at = utc_now_iso()
        self._save("overlay", overlay.source_key, overlay.model_dump(), overlay)

    def save_scenario_review(self, review: SemanticScenarioReview) -> None:
        self._save(
            f"scenario_review:{review.review_id}",
            review.source_key,
            review.model_dump(),
            review,
        )

    def load_scenario_review(self, source_key: str, review_id: str) -> SemanticScenarioReview | None:
        payload = self._load(f"scenario_review:{review_id}", source_key)
        return _model_from_payload(payload, SemanticScenarioReview)

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
        import psycopg

        with psycopg.connect(self.dsn) as conn:
            self._configure_connection(conn)
            self._ensure_schema(conn)
            with conn.transaction():
                conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (source_key,))
                active = conn.execute(
                    """
                    SELECT id FROM semantic_refresh_jobs
                    WHERE source_key = %s AND status = 'running'
                    ORDER BY id DESC LIMIT 1
                    """,
                    (source_key,),
                ).fetchone()
                if active is not None and not force:
                    return None
                if active is not None:
                    conn.execute(
                        """
                        UPDATE semantic_refresh_jobs
                        SET status = 'cancelled', error = %s,
                            updated_at = now(), finished_at = now()
                        WHERE source_key = %s AND status = 'running'
                        """,
                        ("Superseded by a newer semantic operation.", source_key),
                    )
                row = conn.execute(
                    """
                    INSERT INTO semantic_refresh_jobs(
                        catalog_id, connection_id, source_key, operation_type,
                        stage, status, actor_user_id, started_at, updated_at
                    )
                    VALUES (%s, %s, %s, %s, 'queued', 'running', %s, now(), now())
                    RETURNING id, source_key, catalog_id, connection_id, operation_type,
                              stage, status, actor_user_id, error,
                              started_at, updated_at, finished_at
                    """,
                    (catalog_id, connection_id, source_key, operation_type, actor_user_id or None),
                ).fetchone()
        return _operation_from_row(row)

    def load_latest_operation(self, source_key: str) -> SemanticCatalogOperation | None:
        import psycopg

        with psycopg.connect(self.dsn) as conn:
            self._configure_connection(conn)
            self._ensure_schema(conn)
            row = conn.execute(
                """
                SELECT id, source_key, catalog_id, connection_id, operation_type,
                       stage, status, actor_user_id, error,
                       started_at, updated_at, finished_at
                FROM semantic_refresh_jobs
                WHERE source_key = %s
                ORDER BY id DESC LIMIT 1
                """,
                (source_key,),
            ).fetchone()
        return _operation_from_row(row) if row is not None else None

    def update_operation(
        self,
        operation_id: int,
        *,
        stage: str | None = None,
        status: str | None = None,
        error: str | None = None,
    ) -> SemanticCatalogOperation | None:
        import psycopg

        terminal = status in {"completed", "failed", "cancelled", "interrupted"}
        with psycopg.connect(self.dsn) as conn:
            self._configure_connection(conn)
            self._ensure_schema(conn)
            row = conn.execute(
                """
                UPDATE semantic_refresh_jobs
                SET stage = COALESCE(%s, stage),
                    status = COALESCE(%s, status),
                    error = %s,
                    updated_at = now(),
                    finished_at = CASE WHEN %s THEN now() ELSE finished_at END
                WHERE id = %s AND status = 'running'
                RETURNING id, source_key, catalog_id, connection_id, operation_type,
                          stage, status, actor_user_id, error,
                          started_at, updated_at, finished_at
                """,
                (stage, status, error, terminal, operation_id),
            ).fetchone()
        return _operation_from_row(row) if row is not None else None

    def save_build_result_if_current(
        self,
        *,
        operation_id: int,
        generated: SemanticCatalog,
        published: SemanticCatalog,
    ) -> bool:
        import psycopg

        with psycopg.connect(self.dsn) as conn:
            self._configure_connection(conn)
            self._ensure_schema(conn)
            with conn.transaction():
                current = conn.execute(
                    """
                    SELECT source_key FROM semantic_refresh_jobs
                    WHERE id = %s AND status = 'running'
                    FOR UPDATE
                    """,
                    (operation_id,),
                ).fetchone()
                if current is None or str(current[0]) != published.source_key:
                    return False
                self._save(
                    "generated_catalog",
                    generated.source_key,
                    generated.model_dump(),
                    generated,
                    conn=conn,
                )
                self._save_catalog(published, conn=conn)
        return True

    def save_generation_result_if_current(
        self,
        *,
        operation_id: int,
        overlay: SemanticCatalogOverlay,
        published: SemanticCatalog,
    ) -> bool:
        import psycopg

        with psycopg.connect(self.dsn) as conn:
            self._configure_connection(conn)
            self._ensure_schema(conn)
            with conn.transaction():
                current = conn.execute(
                    """
                    SELECT source_key FROM semantic_refresh_jobs
                    WHERE id = %s AND status = 'running'
                    FOR UPDATE
                    """,
                    (operation_id,),
                ).fetchone()
                if current is None or str(current[0]) != published.source_key:
                    return False
                self._save(
                    "overlay",
                    overlay.source_key,
                    overlay.model_dump(),
                    overlay,
                    conn=conn,
                )
                self._save_catalog(published, conn=conn)
        return True

    def cancel_operations(self, source_key: str) -> None:
        import psycopg

        with psycopg.connect(self.dsn) as conn:
            self._configure_connection(conn)
            self._ensure_schema(conn)
            conn.execute(
                """
                UPDATE semantic_refresh_jobs
                SET status = 'cancelled', error = %s,
                    updated_at = now(), finished_at = now()
                WHERE source_key = %s AND status = 'running'
                """,
                ("Semantic catalog was cleared.", source_key),
            )

    def delete_source(self, source_key: str) -> None:
        import psycopg

        with psycopg.connect(self.dsn) as conn:
            self._configure_connection(conn)
            self._ensure_schema(conn)
            with conn.transaction():
                conn.execute(
                    """
                    UPDATE semantic_refresh_jobs
                    SET status = 'cancelled', error = %s,
                        updated_at = now(), finished_at = now()
                    WHERE source_key = %s AND status = 'running'
                    """,
                    ("Semantic catalog was cleared.", source_key),
                )
                conn.execute(
                    f"DELETE FROM {_DOCUMENT_TABLE_NAME} WHERE source_key = %s",
                    (source_key,),
                )
                conn.execute(
                    "DELETE FROM semantic_catalogs WHERE source_key = %s",
                    (source_key,),
                )

    def _save(
        self,
        doc_type: str,
        source_key: str,
        payload: dict[str, Any],
        model: Any,
        *,
        conn: Any | None = None,
    ) -> None:
        import psycopg
        from psycopg.types.json import Jsonb

        owns_connection = conn is None
        connection = psycopg.connect(self.dsn) if owns_connection else nullcontext(conn)
        with connection as conn:
            if owns_connection:
                self._configure_connection(conn)
                self._ensure_schema(conn)
            conn.execute(
                f"""
                INSERT INTO {_DOCUMENT_TABLE_NAME}(
                    source_key, doc_type, catalog_id, source_fingerprint,
                    version, payload, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (source_key, doc_type) DO UPDATE SET
                    catalog_id = EXCLUDED.catalog_id,
                    source_fingerprint = EXCLUDED.source_fingerprint,
                    version = EXCLUDED.version,
                    payload = EXCLUDED.payload,
                    updated_at = now()
                """,
                (
                    source_key,
                    doc_type,
                    getattr(model, "catalog_id", None),
                    getattr(model, "source_fingerprint", ""),
                    _document_version(model),
                    Jsonb(payload),
                ),
            )

    def _load(self, doc_type: str, source_key: str) -> dict[str, Any] | None:
        import psycopg

        with psycopg.connect(self.dsn) as conn:
            self._configure_connection(conn)
            self._ensure_schema(conn)
            row = conn.execute(
                f"SELECT payload FROM {_DOCUMENT_TABLE_NAME} WHERE source_key = %s AND doc_type = %s",
                (source_key, doc_type),
            ).fetchone()
        if row is None:
            return None
        payload = row[0]
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, str):
            return json.loads(payload)
        return None

    def _configure_connection(self, conn: Any) -> None:
        from psycopg import sql

        if not self._ready:
            conn.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s))",
                (f"semantic_catalog_schema:{self.schema}",),
            )
            conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(self.schema)))
        conn.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(self.schema)))

    def _ensure_schema(self, conn: Any) -> None:
        if self._ready:
            return
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_DOCUMENT_TABLE_NAME} (
                source_key text NOT NULL,
                doc_type text NOT NULL,
                catalog_id text,
                source_fingerprint text NOT NULL DEFAULT '',
                version integer NOT NULL DEFAULT 0,
                payload jsonb NOT NULL,
                updated_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (source_key, doc_type)
            )
            """
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{_DOCUMENT_TABLE_NAME}_catalog_id "
            f"ON {_DOCUMENT_TABLE_NAME}(catalog_id)"
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{_DOCUMENT_TABLE_NAME}_fingerprint "
            f"ON {_DOCUMENT_TABLE_NAME}(source_fingerprint)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_catalogs (
                id text PRIMARY KEY,
                source_key text NOT NULL UNIQUE,
                connection_id text NOT NULL DEFAULT '',
                source_type text NOT NULL DEFAULT '',
                source_ref_id text NOT NULL DEFAULT '',
                source_label text NOT NULL DEFAULT '',
                source_fingerprint text NOT NULL DEFAULT '',
                status text NOT NULL,
                error text,
                built_at timestamptz NOT NULL,
                updated_at timestamptz NOT NULL,
                built_by_user_id integer,
                updated_by_user_id integer,
                validation_json jsonb NOT NULL DEFAULT '{}'::jsonb,
                overlay_version integer NOT NULL DEFAULT 0,
                published_version integer NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            "ALTER TABLE semantic_catalogs ADD COLUMN IF NOT EXISTS profile_json "
            "jsonb NOT NULL DEFAULT '{}'::jsonb"
        )
        _ensure_column(conn, "semantic_catalogs", "overlay_version", "integer")
        _ensure_column(conn, "semantic_catalogs", "published_version", "integer")
        conn.execute(
            f"""
            UPDATE semantic_catalogs AS catalog
            SET overlay_version = COALESCE(catalog.overlay_version, document.version, 0),
                published_version = COALESCE(catalog.published_version, document.version, 0)
            FROM {_DOCUMENT_TABLE_NAME} AS document
            WHERE document.source_key = catalog.source_key
              AND document.doc_type = 'overlay'
              AND (catalog.overlay_version IS NULL OR catalog.published_version IS NULL)
            """
        )
        conn.execute(
            "UPDATE semantic_catalogs "
            "SET overlay_version = COALESCE(overlay_version, 0), "
            "published_version = COALESCE(published_version, 0)"
        )
        conn.execute(
            "ALTER TABLE semantic_catalogs ALTER COLUMN overlay_version SET DEFAULT 0, "
            "ALTER COLUMN overlay_version SET NOT NULL, "
            "ALTER COLUMN published_version SET DEFAULT 0, "
            "ALTER COLUMN published_version SET NOT NULL"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_semantic_catalogs_connection ON semantic_catalogs(connection_id)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_tables (
                id text PRIMARY KEY,
                object_id text NOT NULL DEFAULT '',
                catalog_id text NOT NULL REFERENCES semantic_catalogs(id) ON DELETE CASCADE,
                qualified_name text NOT NULL,
                schema_name text,
                table_name text NOT NULL,
                source_kind text NOT NULL,
                description text NOT NULL DEFAULT '',
                semantic_role text NOT NULL DEFAULT 'unknown',
                grain text NOT NULL DEFAULT '',
                row_count bigint,
                columns_count integer NOT NULL DEFAULT 0,
                aliases_json jsonb NOT NULL DEFAULT '[]'::jsonb,
                tags_json jsonb NOT NULL DEFAULT '[]'::jsonb,
                quality_notes_json jsonb NOT NULL DEFAULT '[]'::jsonb,
                ai_context text NOT NULL DEFAULT '',
                is_hidden boolean NOT NULL DEFAULT false
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_semantic_tables_catalog ON semantic_tables(catalog_id)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_columns (
                id text PRIMARY KEY,
                object_id text NOT NULL DEFAULT '',
                catalog_id text NOT NULL REFERENCES semantic_catalogs(id) ON DELETE CASCADE,
                table_name text NOT NULL,
                name text NOT NULL,
                dtype text NOT NULL DEFAULT '',
                nullable boolean,
                semantic_role text NOT NULL DEFAULT 'unknown',
                description text NOT NULL DEFAULT '',
                aliases_json jsonb NOT NULL DEFAULT '[]'::jsonb,
                examples_json jsonb NOT NULL DEFAULT '[]'::jsonb,
                quality_notes_json jsonb NOT NULL DEFAULT '[]'::jsonb,
                ai_context text NOT NULL DEFAULT '',
                is_hidden boolean NOT NULL DEFAULT false
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_semantic_columns_catalog ON semantic_columns(catalog_id)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_column_profiles (
                column_id text PRIMARY KEY REFERENCES semantic_columns(id) ON DELETE CASCADE,
                catalog_id text NOT NULL REFERENCES semantic_catalogs(id) ON DELETE CASCADE,
                null_ratio double precision,
                distinct_count bigint,
                min_value text,
                max_value text,
                top_values_json jsonb NOT NULL DEFAULT '[]'::jsonb
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_relationships (
                id text PRIMARY KEY,
                object_id text NOT NULL DEFAULT '',
                catalog_id text NOT NULL REFERENCES semantic_catalogs(id) ON DELETE CASCADE,
                from_table text NOT NULL,
                from_column text NOT NULL,
                to_table text NOT NULL,
                to_column text NOT NULL,
                description text NOT NULL DEFAULT '',
                cardinality text NOT NULL DEFAULT 'unknown',
                is_active boolean NOT NULL DEFAULT true
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_metrics (
                id text PRIMARY KEY,
                object_id text NOT NULL DEFAULT '',
                catalog_id text NOT NULL REFERENCES semantic_catalogs(id) ON DELETE CASCADE,
                key text NOT NULL,
                name text NOT NULL,
                type text NOT NULL,
                base_table text NOT NULL,
                expr text,
                agg text,
                numerator text,
                denominator text,
                formula text NOT NULL DEFAULT '',
                default_time_dimension text,
                allowed_dimensions_json jsonb NOT NULL DEFAULT '[]'::jsonb,
                filters_json jsonb NOT NULL DEFAULT '[]'::jsonb,
                format text NOT NULL DEFAULT 'number',
                description text NOT NULL DEFAULT '',
                synonyms_json jsonb NOT NULL DEFAULT '[]'::jsonb,
                is_active boolean NOT NULL DEFAULT true,
                created_at timestamptz NOT NULL,
                updated_at timestamptz NOT NULL,
                UNIQUE(catalog_id, key)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_terms (
                id text PRIMARY KEY,
                object_id text NOT NULL DEFAULT '',
                catalog_id text NOT NULL REFERENCES semantic_catalogs(id) ON DELETE CASCADE,
                name text NOT NULL,
                description text NOT NULL DEFAULT '',
                synonyms_json jsonb NOT NULL DEFAULT '[]'::jsonb,
                entity_refs_json jsonb NOT NULL DEFAULT '[]'::jsonb,
                is_active boolean NOT NULL DEFAULT true,
                created_at timestamptz NOT NULL,
                updated_at timestamptz NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_refresh_jobs (
                id bigserial PRIMARY KEY,
                catalog_id text,
                connection_id text NOT NULL DEFAULT '',
                source_key text NOT NULL DEFAULT '',
                operation_type text NOT NULL DEFAULT 'build',
                stage text NOT NULL DEFAULT 'queued',
                status text NOT NULL,
                actor_user_id integer,
                error text,
                started_at timestamptz NOT NULL DEFAULT now(),
                updated_at timestamptz NOT NULL DEFAULT now(),
                finished_at timestamptz
            )
            """
        )
        _ensure_column(conn, "semantic_refresh_jobs", "source_key", "text NOT NULL DEFAULT ''")
        _ensure_column(
            conn,
            "semantic_refresh_jobs",
            "operation_type",
            "text NOT NULL DEFAULT 'build'",
        )
        _ensure_column(conn, "semantic_refresh_jobs", "stage", "text NOT NULL DEFAULT 'queued'")
        _ensure_column(conn, "semantic_refresh_jobs", "actor_user_id", "integer")
        _ensure_column(
            conn,
            "semantic_refresh_jobs",
            "updated_at",
            "timestamptz NOT NULL DEFAULT now()",
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_semantic_refresh_jobs_source "
            "ON semantic_refresh_jobs(source_key, id DESC)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_audit_log (
                id bigserial PRIMARY KEY,
                catalog_id text,
                connection_id text NOT NULL DEFAULT '',
                action text NOT NULL,
                actor_user_id integer,
                created_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        for table_name in (
            "semantic_tables",
            "semantic_columns",
            "semantic_relationships",
            "semantic_metrics",
            "semantic_terms",
        ):
            _ensure_column(conn, table_name, "object_id", "text NOT NULL DEFAULT ''")
            conn.execute(f"UPDATE {table_name} SET object_id = id WHERE object_id = ''")
        self._ready = True

    def _save_catalog(self, catalog: SemanticCatalog, *, conn: Any | None = None) -> None:
        import psycopg
        from psycopg.types.json import Jsonb

        owns_connection = conn is None
        connection = psycopg.connect(self.dsn) if owns_connection else nullcontext(conn)
        with connection as conn:
            if owns_connection:
                self._configure_connection(conn)
                self._ensure_schema(conn)
            transaction = conn.transaction() if owns_connection else nullcontext()
            with transaction:
                conn.execute(
                    """
                    INSERT INTO semantic_catalogs(
                        id, source_key, connection_id, source_type, source_ref_id,
                        source_label, source_fingerprint, status, error,
                        built_at, updated_at, built_by_user_id, updated_by_user_id,
                        validation_json, profile_json, overlay_version, published_version
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (source_key) DO UPDATE SET
                        id = EXCLUDED.id,
                        connection_id = EXCLUDED.connection_id,
                        source_type = EXCLUDED.source_type,
                        source_ref_id = EXCLUDED.source_ref_id,
                        source_label = EXCLUDED.source_label,
                        source_fingerprint = EXCLUDED.source_fingerprint,
                        status = EXCLUDED.status,
                        error = EXCLUDED.error,
                        built_at = EXCLUDED.built_at,
                        updated_at = EXCLUDED.updated_at,
                        built_by_user_id = EXCLUDED.built_by_user_id,
                        updated_by_user_id = EXCLUDED.updated_by_user_id,
                        validation_json = EXCLUDED.validation_json,
                        profile_json = EXCLUDED.profile_json,
                        overlay_version = EXCLUDED.overlay_version,
                        published_version = EXCLUDED.published_version
                    """,
                    (
                        catalog.catalog_id,
                        catalog.source_key,
                        catalog.connection_id,
                        catalog.source_type,
                        catalog.source_ref_id,
                        catalog.source_label,
                        catalog.source_fingerprint,
                        catalog.status,
                        catalog.error,
                        catalog.built_at,
                        catalog.updated_at,
                        catalog.user_id or None,
                        catalog.user_id or None,
                        Jsonb(catalog.validation.model_dump()),
                        Jsonb(
                            {
                                "sample_strategy": catalog.profile_sample_strategy,
                                "sample_limit": catalog.profile_sample_limit,
                            }
                        ),
                        catalog.overlay_version,
                        catalog.published_version,
                    ),
                )
                conn.execute("DELETE FROM semantic_tables WHERE catalog_id = %s", (catalog.catalog_id,))
                conn.execute("DELETE FROM semantic_columns WHERE catalog_id = %s", (catalog.catalog_id,))
                conn.execute(
                    "DELETE FROM semantic_column_profiles WHERE catalog_id = %s", (catalog.catalog_id,)
                )
                conn.execute(
                    "DELETE FROM semantic_relationships WHERE catalog_id = %s", (catalog.catalog_id,)
                )
                conn.execute("DELETE FROM semantic_metrics WHERE catalog_id = %s", (catalog.catalog_id,))
                conn.execute("DELETE FROM semantic_terms WHERE catalog_id = %s", (catalog.catalog_id,))
                for table in catalog.tables:
                    conn.execute(
                        """
                        INSERT INTO semantic_tables(
                            id, object_id, catalog_id, qualified_name, schema_name, table_name,
                            source_kind, description, semantic_role, grain, row_count,
                            columns_count, aliases_json, tags_json, quality_notes_json,
                            ai_context, is_hidden
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            _storage_id(catalog.catalog_id, table.table_id),
                            table.table_id,
                            catalog.catalog_id,
                            table.qualified_name,
                            table.schema_name,
                            table.table_name,
                            table.source_kind,
                            table.description,
                            table.semantic_role,
                            table.grain,
                            table.row_count,
                            table.columns_count,
                            Jsonb(table.aliases),
                            Jsonb(table.tags),
                            Jsonb(table.quality_notes),
                            table.ai_context,
                            table.is_hidden,
                        ),
                    )
                for column in catalog.columns:
                    conn.execute(
                        """
                        INSERT INTO semantic_columns(
                            id, object_id, catalog_id, table_name, name, dtype, nullable,
                            semantic_role, description, aliases_json, examples_json,
                            quality_notes_json, ai_context, is_hidden
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            _storage_id(catalog.catalog_id, column.column_id),
                            column.column_id,
                            catalog.catalog_id,
                            column.table,
                            column.name,
                            column.dtype,
                            column.nullable,
                            column.semantic_role,
                            column.description,
                            Jsonb(column.aliases),
                            Jsonb(column.examples),
                            Jsonb(column.quality_notes),
                            column.ai_context,
                            column.is_hidden,
                        ),
                    )
                    conn.execute(
                        """
                        INSERT INTO semantic_column_profiles(
                            column_id, catalog_id, null_ratio, distinct_count,
                            min_value, max_value, top_values_json
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            _storage_id(catalog.catalog_id, column.column_id),
                            catalog.catalog_id,
                            column.null_ratio,
                            column.distinct_count,
                            column.min_value,
                            column.max_value,
                            Jsonb(column.top_values),
                        ),
                    )
                for rel in catalog.relationships:
                    conn.execute(
                        """
                        INSERT INTO semantic_relationships(
                            id, object_id, catalog_id, from_table, from_column, to_table,
                            to_column, description, cardinality, is_active
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            _storage_id(catalog.catalog_id, rel.relationship_id),
                            rel.relationship_id,
                            catalog.catalog_id,
                            rel.from_table,
                            rel.from_column,
                            rel.to_table,
                            rel.to_column,
                            rel.description,
                            rel.cardinality,
                            rel.is_active,
                        ),
                    )
                for metric in catalog.metrics:
                    conn.execute(
                        """
                        INSERT INTO semantic_metrics(
                            id, object_id, catalog_id, key, name, type, base_table, expr, agg,
                            numerator, denominator, formula, default_time_dimension,
                            allowed_dimensions_json, filters_json, format,
                            description, synonyms_json, is_active, created_at, updated_at
                        )
                        VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                        """,
                        (
                            _storage_id(catalog.catalog_id, metric.metric_id),
                            metric.metric_id,
                            catalog.catalog_id,
                            metric.key,
                            metric.name,
                            metric.type,
                            metric.base_table,
                            metric.expr,
                            metric.agg,
                            metric.numerator,
                            metric.denominator,
                            metric.formula,
                            metric.default_time_dimension,
                            Jsonb(metric.allowed_dimensions),
                            Jsonb([item.model_dump() for item in metric.filters]),
                            metric.format,
                            metric.description,
                            Jsonb(metric.synonyms),
                            metric.is_active,
                            metric.created_at,
                            metric.updated_at,
                        ),
                    )
                for term in catalog.terms:
                    conn.execute(
                        """
                        INSERT INTO semantic_terms(
                            id, object_id, catalog_id, name, description, synonyms_json,
                            entity_refs_json, is_active, created_at, updated_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            _storage_id(catalog.catalog_id, term.term_id),
                            term.term_id,
                            catalog.catalog_id,
                            term.name,
                            term.description,
                            Jsonb(term.synonyms),
                            Jsonb(term.entity_refs),
                            term.is_active,
                            term.created_at,
                            term.updated_at,
                        ),
                    )
                conn.execute(
                    """
                    INSERT INTO semantic_audit_log(
                        catalog_id, connection_id, action, actor_user_id, created_at
                    )
                    VALUES (%s, %s, %s, %s, now())
                    """,
                    (catalog.catalog_id, catalog.connection_id, "save_catalog", catalog.user_id or None),
                )

    def _load_catalog(self, source_key: str) -> SemanticCatalog | None:
        import psycopg

        with psycopg.connect(self.dsn) as conn:
            self._configure_connection(conn)
            self._ensure_schema(conn)
            row = conn.execute(
                """
                SELECT
                    id, source_key, connection_id, source_type, source_ref_id,
                    source_label, source_fingerprint, status, error,
                    built_at, updated_at, built_by_user_id, updated_by_user_id,
                    validation_json, profile_json, overlay_version, published_version
                FROM semantic_catalogs
                WHERE source_key = %s
                """,
                (source_key,),
            ).fetchone()
            if row is None:
                return None
            catalog_id = row[0]
            tables = conn.execute(
                """
                SELECT
                    id, object_id, catalog_id, qualified_name, schema_name, table_name,
                    source_kind, description, semantic_role, grain, row_count,
                    columns_count, aliases_json, tags_json, quality_notes_json,
                    ai_context, is_hidden
                FROM semantic_tables
                WHERE catalog_id = %s
                ORDER BY qualified_name
                """,
                (catalog_id,),
            ).fetchall()
            columns = conn.execute(
                """
                SELECT
                    c.id, c.object_id, c.catalog_id, c.table_name, c.name, c.dtype,
                    c.nullable, c.semantic_role, c.description, c.aliases_json,
                    c.examples_json, c.quality_notes_json, c.ai_context, c.is_hidden,
                    p.null_ratio, p.distinct_count, p.min_value, p.max_value,
                    p.top_values_json
                FROM semantic_columns c
                LEFT JOIN semantic_column_profiles p ON p.column_id = c.id AND p.catalog_id = c.catalog_id
                WHERE c.catalog_id = %s
                ORDER BY c.table_name, c.name
                """,
                (catalog_id,),
            ).fetchall()
            relationships = conn.execute(
                """
                SELECT
                    id, object_id, catalog_id, from_table, from_column, to_table,
                    to_column, description, cardinality, is_active
                FROM semantic_relationships
                WHERE catalog_id = %s
                ORDER BY id
                """,
                (catalog_id,),
            ).fetchall()
            metrics = conn.execute(
                """
                SELECT
                    id, object_id, catalog_id, key, name, type, base_table, expr, agg,
                    numerator, denominator, formula, default_time_dimension,
                    allowed_dimensions_json, filters_json, format, description,
                    synonyms_json, is_active, created_at, updated_at
                FROM semantic_metrics
                WHERE catalog_id = %s
                ORDER BY key
                """,
                (catalog_id,),
            ).fetchall()
            terms = conn.execute(
                """
                SELECT
                    id, object_id, catalog_id, name, description, synonyms_json,
                    entity_refs_json, is_active, created_at, updated_at
                FROM semantic_terms
                WHERE catalog_id = %s
                ORDER BY name
                """,
                (catalog_id,),
            ).fetchall()
        catalog = SemanticCatalog.model_validate(
            {
                "catalog_id": row[0],
                "source_key": row[1],
                "connection_id": row[2] or "",
                "source_type": row[3] or "",
                "source_ref_id": row[4] or "",
                "source_label": row[5] or "",
                "source_fingerprint": row[6] or "",
                "status": row[7] or "not_built",
                "error": row[8],
                "built_at": str(row[9]),
                "updated_at": str(row[10]),
                "user_id": int(row[11] or 0),
                "profile_sample_strategy": str((row[14] or {}).get("sample_strategy") or ""),
                "profile_sample_limit": (row[14] or {}).get("sample_limit"),
                "overlay_version": int(row[15] or 0),
                "published_version": int(row[16] or 0),
                "tables": [
                    {
                        "table_id": item[1],
                        "qualified_name": item[3],
                        "schema_name": item[4],
                        "table_name": item[5],
                        "source_kind": item[6],
                        "description": item[7] or "",
                        "semantic_role": item[8] or "unknown",
                        "grain": item[9] or "",
                        "row_count": item[10],
                        "columns_count": item[11] or 0,
                        "aliases": _json_value(item[12], []),
                        "tags": _json_value(item[13], []),
                        "quality_notes": _json_value(item[14], []),
                        "ai_context": item[15] or "",
                        "is_hidden": bool(item[16]),
                    }
                    for item in tables
                ],
                "columns": [
                    {
                        "column_id": item[1],
                        "table": item[3],
                        "name": item[4],
                        "dtype": item[5] or "",
                        "nullable": item[6],
                        "semantic_role": item[7] or "unknown",
                        "description": item[8] or "",
                        "aliases": _json_value(item[9], []),
                        "examples": _json_value(item[10], []),
                        "quality_notes": _json_value(item[11], []),
                        "ai_context": item[12] or "",
                        "is_hidden": bool(item[13]),
                        "null_ratio": item[14],
                        "distinct_count": item[15],
                        "min_value": item[16],
                        "max_value": item[17],
                        "top_values": _json_value(item[18], []),
                    }
                    for item in columns
                ],
                "relationships": [
                    {
                        "relationship_id": item[1],
                        "from_table": item[3],
                        "from_column": item[4],
                        "to_table": item[5],
                        "to_column": item[6],
                        "description": item[7] or "",
                        "cardinality": item[8] or "unknown",
                        "is_active": bool(item[9]),
                    }
                    for item in relationships
                ],
                "metrics": [
                    {
                        "metric_id": item[1],
                        "key": item[3],
                        "name": item[4],
                        "type": item[5],
                        "base_table": item[6],
                        "expr": item[7],
                        "agg": item[8],
                        "numerator": item[9],
                        "denominator": item[10],
                        "formula": item[11] or "",
                        "default_time_dimension": item[12],
                        "allowed_dimensions": _json_value(item[13], []),
                        "filters": _json_value(item[14], []),
                        "format": item[15] or "number",
                        "description": item[16] or "",
                        "synonyms": _json_value(item[17], []),
                        "is_active": bool(item[18]),
                        "created_at": str(item[19]),
                        "updated_at": str(item[20]),
                    }
                    for item in metrics
                ],
                "terms": [
                    {
                        "term_id": item[1],
                        "name": item[3],
                        "description": item[4] or "",
                        "synonyms": _json_value(item[5], []),
                        "entity_refs": _json_value(item[6], []),
                        "is_active": bool(item[7]),
                        "created_at": str(item[8]),
                        "updated_at": str(item[9]),
                    }
                    for item in terms
                ],
                "validation": _json_value(row[13], {}),
            }
        )
        if not catalog.entities:
            catalog.entities = _semantic_entities(catalog)
        if not catalog.dimensions:
            catalog.dimensions = _semantic_dimensions(catalog)
        if not catalog.facts:
            catalog.facts = _semantic_facts(catalog)
        return catalog


def _operation_from_row(row: Any) -> SemanticCatalogOperation:
    return SemanticCatalogOperation(
        operation_id=int(row[0]),
        source_key=str(row[1] or ""),
        catalog_id=str(row[2] or ""),
        connection_id=str(row[3] or ""),
        operation_type=str(row[4] or "build"),
        stage=str(row[5] or "queued"),
        status=str(row[6] or "running"),
        actor_user_id=int(row[7] or 0),
        error=str(row[8]) if row[8] is not None else None,
        started_at=str(row[9]),
        updated_at=str(row[10]),
        finished_at=str(row[11]) if row[11] is not None else None,
    )


def semantic_catalog_store_from_settings(settings: Any) -> SemanticCatalogStore:
    dsn = str(getattr(settings, "semantic_catalog_postgres_dsn", "") or "").strip()
    if not dsn:
        raise ValueError("SEMANTIC_METADATA_DATABASE_URL must be set for PostgreSQL metadata")
    schema = str(getattr(settings, "semantic_catalog_postgres_schema", "public") or "public").strip()
    return SemanticCatalogPostgresStore(dsn, schema=schema)


def _model_from_payload(payload: dict[str, Any] | None, model_cls: Any) -> Any | None:
    if payload is None:
        return None
    try:
        return model_cls.model_validate(payload)
    except (ValueError, TypeError):
        return None


def _json_value(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


def _semantic_dimensions(catalog: SemanticCatalog) -> list[SemanticDimension]:
    dimensions: list[SemanticDimension] = []
    for column in catalog.columns:
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


def _semantic_entities(catalog: SemanticCatalog) -> list[SemanticEntity]:
    entities: list[SemanticEntity] = []
    for column in catalog.columns:
        if column.semantic_role not in {"identifier", "foreign_key_candidate"}:
            continue
        name = re.sub(r"(_id|id)$", "", column.name.lower()).strip("_") or column.name.lower()
        table_leaf = column.table.split(".")[-1].lower()
        table_entities = {table_leaf, "id"}
        if table_leaf.endswith("s"):
            table_entities.add(table_leaf[:-1])
        if table_leaf.endswith("es"):
            table_entities.add(table_leaf[:-2])
        is_primary = name in table_entities or table_leaf.startswith(f"{name}_")
        kind = "primary" if is_primary else "foreign"
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


def _semantic_facts(catalog: SemanticCatalog) -> list[SemanticFact]:
    return [
        SemanticFact(
            fact_id=f"fact:{column.table}.{column.name}",
            name=column.name,
            table=column.table,
            expr=column.name,
            type="number",
        )
        for column in catalog.columns
        if column.semantic_role == "metric_candidate"
    ]


def _storage_id(catalog_id: str, object_id: str) -> str:
    return stable_id("semantic-storage-row", catalog_id, object_id)


def _ensure_column(conn: Any, table_name: str, column_name: str, definition: str) -> None:
    row = conn.execute(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = %s AND column_name = %s
        """,
        (table_name, column_name),
    ).fetchone()
    if row is None:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")


def _document_version(model: Any) -> int:
    for attr in ("review_version", "published_version", "overlay_version", "version"):
        raw = getattr(model, attr, None)
        if raw is None:
            continue
        try:
            return int(raw or 0)
        except (TypeError, ValueError):
            continue
    return 0
