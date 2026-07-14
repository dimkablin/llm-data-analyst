from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from backend.agent.llm_client import make_reasoning_llm
from backend.data_access.catalog_refresh import refresh_session_catalog
from backend.data_access.semantic_catalog_service import SemanticCatalogService
from backend.data_access.semantic_models import (
    MetricAggregation,
    RelationshipCardinality,
    SemanticCatalog,
    SemanticColumnPatch,
    SemanticColumnRole,
    SemanticMetric,
    SemanticMetricKind,
    SemanticRelationship,
    SemanticTablePatch,
    SemanticTableRole,
    SemanticTerm,
    clean_list,
    stable_id,
)
from backend.data_access.db_runtime_service import DBRuntimeService, RuntimeDBConnectionConfig
from backend.sessions.session_store import SessionStore
from backend.tools.impl.db_helpers import DBAnalyticsHelper

_SENSITIVE_RE = re.compile(r"(password|passwd|secret|token|api[_-]?key|credential|dsn)", re.I)
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class GeneratedTablePatch(BaseModel):
    table: str
    description: str = ""
    semantic_role: SemanticTableRole | None = None
    grain: str = ""
    aliases: list[str] = Field(default_factory=list)


class GeneratedColumnPatch(BaseModel):
    table: str
    column: str
    description: str = ""
    semantic_role: SemanticColumnRole | None = None
    aliases: list[str] = Field(default_factory=list)


class GeneratedMetricDraft(BaseModel):
    key: str
    name: str
    type: SemanticMetricKind = "simple"
    base_table: str
    expr: str | None = None
    agg: MetricAggregation | None = None
    numerator: str | None = None
    denominator: str | None = None
    formula: str = ""
    default_time_dimension: str | None = None
    allowed_dimensions: list[str] = Field(default_factory=list)
    format: str = "number"
    description: str = ""
    synonyms: list[str] = Field(default_factory=list)


class GeneratedRelationshipDraft(BaseModel):
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    cardinality: RelationshipCardinality = "many_to_one"
    description: str = ""


class GeneratedTermDraft(BaseModel):
    name: str
    description: str = ""
    synonyms: list[str] = Field(default_factory=list)
    entity_refs: list[str] = Field(default_factory=list)


class SemanticGenerationDraft(BaseModel):
    tables: list[GeneratedTablePatch] = Field(default_factory=list)
    columns: list[GeneratedColumnPatch] = Field(default_factory=list)
    metrics: list[GeneratedMetricDraft] = Field(default_factory=list)
    relationships: list[GeneratedRelationshipDraft] = Field(default_factory=list)
    terms: list[GeneratedTermDraft] = Field(default_factory=list)


class SemanticCatalogGenerationRequest(BaseModel):
    sample_rows: int = Field(default=3, ge=0, le=10)
    max_tables: int = Field(default=30, ge=1, le=200)


class SemanticCatalogGenerationSummary(BaseModel):
    tables_scanned: int = 0
    sample_tables: int = 0
    table_patches: int = 0
    column_patches: int = 0
    metrics_added: int = 0
    terms_added: int = 0
    relationships_added: int = 0
    rejected_items: list[str] = Field(default_factory=list)


class SemanticCatalogGenerationResponse(BaseModel):
    catalog: SemanticCatalog
    summary: SemanticCatalogGenerationSummary


@dataclass
class SemanticCatalogGenerationService:
    store: SessionStore
    catalog_service: SemanticCatalogService
    db_runtime_service: DBRuntimeService
    settings: Any
    llm_generate: Callable[[dict[str, Any]], SemanticGenerationDraft] | None = None

    def generate(
        self,
        *,
        session_id: str,
        user_id: int,
        request: SemanticCatalogGenerationRequest | None = None,
    ) -> SemanticCatalogGenerationResponse:
        options = request or SemanticCatalogGenerationRequest()
        state = self.store.load_session(session_id)
        if state is None:
            raise ValueError("Session not found")
        source_type = str(state.source_type or "").lower()
        helper: DBAnalyticsHelper | None = None
        runtime: RuntimeDBConnectionConfig | None = None
        if source_type == "db_connection" and state.source_ref_id:
            runtime = self.db_runtime_service.get_runtime_config(
                user_id=user_id,
                connection_id=state.source_ref_id,
            )
            refresh_session_catalog(self.store, session_id, db_runtime=runtime)
            helper = DBAnalyticsHelper(runtime=runtime, timeout_sec=15.0)
        elif source_type != "csv":
            raise ValueError(
                "AI semantic generation is available only for database or uploaded CSV/XLSX sources"
            )

        catalog = self.catalog_service.refresh(session_id=session_id, user_id=user_id)
        prompt_payload = self._prompt_payload(
            catalog=catalog,
            helper=helper,
            runtime=runtime,
            sample_rows=options.sample_rows,
            max_tables=options.max_tables,
        )
        draft = self._generate_draft(prompt_payload)
        return self._apply_draft(
            session_id=session_id,
            user_id=user_id,
            catalog=catalog,
            draft=draft,
            relationships_from_db=prompt_payload["relationships_from_db"],
            tables_scanned=len(prompt_payload["tables"]),
            sample_tables=len(prompt_payload["samples"]),
        )

    def _generate_draft(self, prompt_payload: dict[str, Any]) -> SemanticGenerationDraft:
        if self.llm_generate is not None:
            return self.llm_generate(prompt_payload)
        llm = make_reasoning_llm(
            provider=getattr(self.settings, "llm_provider", None),
            model=str(getattr(self.settings, "llm_model")),
            base_url=str(getattr(self.settings, "llm_base_url")),
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
        structured = llm.with_structured_output(SemanticGenerationDraft)
        return structured.invoke(
            [
                (
                    "system",
                    "You generate a conservative semantic layer for analytics. "
                    "Use only provided tables and columns. Prefer simple metrics, "
                    "many_to_one relationships, concise business terms, and safe aliases. "
                    "Do not invent columns. Do not include raw sample values in descriptions.",
                ),
                ("human", json.dumps(prompt_payload, ensure_ascii=False)),
            ]
        )

    def _prompt_payload(
        self,
        *,
        catalog: SemanticCatalog,
        helper: DBAnalyticsHelper | None,
        runtime: RuntimeDBConnectionConfig | None,
        sample_rows: int,
        max_tables: int,
    ) -> dict[str, Any]:
        tables = [table for table in catalog.tables if not table.is_hidden][:max_tables]
        columns_by_table = {
            table.qualified_name: [
                {
                    "name": column.name,
                    "dtype": column.dtype,
                    "role": column.semantic_role,
                }
                for column in catalog.columns
                if column.table == table.qualified_name and not column.is_hidden
            ]
            for table in tables
        }
        samples: dict[str, list[dict[str, Any]]] = {}
        if sample_rows > 0:
            for table in tables:
                rows = (
                    self._sample_table(helper, table.schema_name, table.table_name, sample_rows)
                    if helper is not None
                    else _profile_samples(catalog, table.qualified_name, sample_rows)
                )
                if rows:
                    samples[table.qualified_name] = rows
        return {
            "db_type": runtime.db_type if runtime is not None else "csv_duckdb",
            "source_label": catalog.source_label,
            "tables": [
                {
                    "qualified_name": table.qualified_name,
                    "table_name": table.table_name,
                    "schema_name": table.schema_name,
                    "role": table.semantic_role,
                    "columns": columns_by_table.get(table.qualified_name, []),
                }
                for table in tables
            ],
            "relationships_from_db": (
                self._relationships_from_db(helper, catalog) if helper is not None else []
            ),
            "samples": samples,
        }

    def _sample_table(
        self,
        helper: DBAnalyticsHelper,
        schema: str | None,
        table: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        try:
            df = helper.preview_table(table, schema=schema, limit=limit)
        except Exception:
            return []
        rows = df.head(limit).to_dict(orient="records")
        return [
            {
                str(key): _safe_sample_value(str(key), value)
                for key, value in row.items()
            }
            for row in rows
        ]

    def _relationships_from_db(
        self,
        helper: DBAnalyticsHelper,
        catalog: SemanticCatalog,
    ) -> list[dict[str, str]]:
        table_names = {table.qualified_name for table in catalog.tables}
        rows: list[dict[str, str]] = []
        try:
            relationships = helper.list_effective_relationships()
        except Exception:
            return rows
        for rel in relationships:
            from_table = _qualified(rel.get("from_schema"), rel.get("from_table"))
            to_table = _qualified(rel.get("to_schema"), rel.get("to_table"))
            if from_table not in table_names or to_table not in table_names:
                continue
            rows.append(
                {
                    "from_table": from_table,
                    "from_column": str(rel.get("from_column") or ""),
                    "to_table": to_table,
                    "to_column": str(rel.get("to_column") or ""),
                    "cardinality": "many_to_one",
                }
            )
        return rows

    def _apply_draft(
        self,
        *,
        session_id: str,
        user_id: int,
        catalog: SemanticCatalog,
        draft: SemanticGenerationDraft,
        relationships_from_db: list[dict[str, str]],
        tables_scanned: int,
        sample_tables: int,
    ) -> SemanticCatalogGenerationResponse:
        table_map = _table_lookup(catalog)
        column_ids = {
            (column.table, column.name): column.column_id
            for column in catalog.columns
        }
        table_patches = _table_patches(draft, table_map)
        column_patches = _column_patches(draft, table_map, column_ids)
        metrics = _metrics(draft, catalog, table_map)
        relationships = _relationships(draft, catalog, table_map, relationships_from_db)
        terms = _terms(draft, catalog)
        before_metrics = {item.metric_id for item in catalog.metrics}
        before_terms = {item.term_id for item in catalog.terms}
        before_relationships = {item.relationship_id for item in catalog.relationships}
        published, rejected = self.catalog_service.apply_generated_overlay(
            session_id=session_id,
            user_id=user_id,
            table_patches=table_patches,
            column_patches=column_patches,
            metrics=metrics,
            relationships=relationships,
            terms=terms,
        )
        return SemanticCatalogGenerationResponse(
            catalog=published,
            summary=SemanticCatalogGenerationSummary(
                tables_scanned=tables_scanned,
                sample_tables=sample_tables,
                table_patches=len(table_patches),
                column_patches=len(column_patches),
                metrics_added=len({item.metric_id for item in published.metrics} - before_metrics),
                terms_added=len({item.term_id for item in published.terms} - before_terms),
                relationships_added=len(
                    {item.relationship_id for item in published.relationships} - before_relationships
                ),
                rejected_items=rejected,
            ),
        )


def _table_lookup(catalog: SemanticCatalog) -> dict[str, str]:
    result: dict[str, str] = {}
    for table in catalog.tables:
        result[table.qualified_name] = table.qualified_name
        result[table.table_name] = table.qualified_name
    return result


def _table_patches(
    draft: SemanticGenerationDraft,
    table_map: dict[str, str],
) -> dict[str, SemanticTablePatch]:
    patches: dict[str, SemanticTablePatch] = {}
    for item in draft.tables:
        table = table_map.get(item.table)
        if not table:
            continue
        patches[f"table:{table}"] = SemanticTablePatch(
            description=item.description or None,
            semantic_role=item.semantic_role,
            grain=item.grain or None,
            aliases=clean_list(item.aliases) or None,
        )
    return patches


def _column_patches(
    draft: SemanticGenerationDraft,
    table_map: dict[str, str],
    column_ids: dict[tuple[str, str], str],
) -> dict[str, SemanticColumnPatch]:
    patches: dict[str, SemanticColumnPatch] = {}
    for item in draft.columns:
        table = table_map.get(item.table)
        if not table:
            continue
        column_id = column_ids.get((table, item.column))
        if not column_id:
            continue
        patches[column_id] = SemanticColumnPatch(
            description=item.description or None,
            semantic_role=item.semantic_role,
            aliases=clean_list(item.aliases) or None,
        )
    return patches


def _metrics(
    draft: SemanticGenerationDraft,
    catalog: SemanticCatalog,
    table_map: dict[str, str],
) -> list[SemanticMetric]:
    metrics: list[SemanticMetric] = []
    for item in draft.metrics:
        table = table_map.get(item.base_table)
        if not table:
            continue
        try:
            data = item.model_dump()
            data["base_table"] = table
            metric = SemanticMetric(
                **data,
                metric_id=f"metric:{item.key}",
            )
        except ValueError:
            continue
        metrics.append(metric)
    return metrics


def _relationships(
    draft: SemanticGenerationDraft,
    catalog: SemanticCatalog,
    table_map: dict[str, str],
    relationships_from_db: list[dict[str, str]],
) -> list[SemanticRelationship]:
    rows = [
        {
            "from_table": item.from_table,
            "from_column": item.from_column,
            "to_table": item.to_table,
            "to_column": item.to_column,
            "cardinality": item.cardinality,
            "description": item.description,
        }
        for item in draft.relationships
    ]
    rows.extend(relationships_from_db)
    relationships: list[SemanticRelationship] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        from_table = table_map.get(str(row.get("from_table") or ""))
        to_table = table_map.get(str(row.get("to_table") or ""))
        from_column = str(row.get("from_column") or "").strip()
        to_column = str(row.get("to_column") or "").strip()
        if not from_table or not to_table or not from_column or not to_column:
            continue
        key = (from_table, from_column, to_table, to_column)
        if key in seen:
            continue
        seen.add(key)
        try:
            relationship = SemanticRelationship(
                relationship_id=f"relationship:{stable_id(catalog.source_key, *key)}",
                from_table=from_table,
                from_column=from_column,
                to_table=to_table,
                to_column=to_column,
                cardinality=str(row.get("cardinality") or "many_to_one"),  # type: ignore[arg-type]
                description=str(row.get("description") or ""),
            )
        except ValueError:
            continue
        relationships.append(relationship)
    return relationships


def _terms(draft: SemanticGenerationDraft, catalog: SemanticCatalog) -> list[SemanticTerm]:
    terms: list[SemanticTerm] = []
    for item in draft.terms:
        name = str(item.name or "").strip()
        if not name:
            continue
        terms.append(
            SemanticTerm(
                term_id=f"term:{stable_id('term', catalog.source_key, name)}",
                name=name,
                description=item.description,
                synonyms=item.synonyms,
                entity_refs=item.entity_refs,
            )
        )
    return terms


def _safe_sample_value(column: str, value: Any) -> Any:
    if value is None:
        return None
    if _SENSITIVE_RE.search(column):
        return "<redacted>"
    if isinstance(value, str):
        if _EMAIL_RE.match(value):
            return "<email>"
        if len(value) > 80:
            return value[:77] + "..."
    return value


def _profile_samples(
    catalog: SemanticCatalog,
    table: str,
    limit: int,
) -> list[dict[str, Any]]:
    columns = [column for column in catalog.columns if column.table == table and column.examples]
    return [
        {
            column.name: _safe_sample_value(column.name, column.examples[index])
            for column in columns
            if index < len(column.examples)
        }
        for index in range(limit)
        if any(index < len(column.examples) for column in columns)
    ]


def _qualified(schema: Any, table: Any) -> str:
    clean_schema = str(schema or "").strip()
    clean_table = str(table or "").strip()
    return f"{clean_schema}.{clean_table}" if clean_schema else clean_table
