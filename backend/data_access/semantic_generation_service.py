from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from backend.agent.llm_client import make_reasoning_llm
from backend.data_access.catalog_refresh import refresh_session_catalog
from backend.data_access.db_runtime_service import DBRuntimeService, RuntimeDBConnectionConfig
from backend.data_access.semantic_catalog_service import SemanticCatalogService
from backend.data_access.semantic_models import (
    MetricAggregation,
    RelationshipCardinality,
    SemanticCatalog,
    SemanticColumnPatch,
    SemanticColumnRole,
    SemanticMetric,
    SemanticMetricFilter,
    SemanticMetricKind,
    SemanticRelationship,
    SemanticTablePatch,
    SemanticTableRole,
    SemanticTerm,
    clean_list,
    stable_id,
)
from backend.sessions.session_store import SessionStore
from backend.tools.impl.db_helpers import DBAnalyticsHelper

_SENSITIVE_RE = re.compile(r"(password|passwd|secret|token|api[_-]?key|credential|dsn)", re.I)
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_TECHNICAL_DIMENSION_RE = re.compile(
    r"^(metric|is_?deleted|deleted|delete_?flag|row_?version|etl_?.*|ingest(?:ed|ion)?_?.*)$",
    re.I,
)
logger = logging.getLogger(__name__)


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
    filters: list[SemanticMetricFilter] = Field(default_factory=list)
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
    ensure_metrics: bool = False


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
        operation_id: int | None = None,
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

        if operation_id is not None:
            self.catalog_service.update_operation(operation_id=operation_id, stage="generating")
        catalog = self.catalog_service.load_for_session(session_id=session_id, user_id=user_id)
        if catalog is None:
            if operation_id is not None:
                raise ValueError("Build the semantic catalog before AI generation")
            catalog = self.catalog_service.refresh(session_id=session_id, user_id=user_id)
        tables = [table for table in catalog.tables if not table.is_hidden][: options.max_tables]
        batch_size = max(
            1,
            min(
                int(getattr(self.settings, "semantic_generation_batch_tables", 2) or 2),
                options.max_tables,
            ),
        )
        relationships_from_db = self._relationships_from_db(helper, catalog) if helper is not None else []
        drafts, prompt_payloads = self._generate_drafts_in_batches(
            catalog=catalog,
            helper=helper,
            runtime=runtime,
            sample_rows=options.sample_rows,
            max_tables=options.max_tables,
            tables=tables,
            batch_size=batch_size,
            relationships_from_db=relationships_from_db,
        )
        draft = _merge_generation_drafts(drafts)
        draft = _repair_generated_column_roles(draft=draft, catalog=catalog)
        if options.ensure_metrics:
            draft = self._complete_missing_metrics(
                catalog=catalog,
                draft=draft,
                helper=helper,
                runtime=runtime,
                tables=tables,
                sample_rows=options.sample_rows,
                max_tables=options.max_tables,
                relationships_from_db=relationships_from_db,
                batch_size=batch_size,
            )
        return self._apply_draft(
            session_id=session_id,
            user_id=user_id,
            catalog=catalog,
            draft=draft,
            relationships_from_db=relationships_from_db,
            tables_scanned=len(tables),
            sample_tables=sum(len(payload["samples"]) for payload in prompt_payloads),
            operation_id=operation_id,
        )

    def _generate_drafts_in_batches(
        self,
        *,
        catalog: SemanticCatalog,
        helper: DBAnalyticsHelper | None,
        runtime: RuntimeDBConnectionConfig | None,
        sample_rows: int,
        max_tables: int,
        tables: list[Any],
        batch_size: int,
        relationships_from_db: list[dict[str, str]],
    ) -> tuple[list[SemanticGenerationDraft], list[dict[str, Any]]]:
        drafts: list[SemanticGenerationDraft] = []
        successful_payloads: list[dict[str, Any]] = []

        def generate_group(selected_tables: list[Any]) -> None:
            prompt_payload = self._prompt_payload(
                catalog=catalog,
                helper=helper,
                runtime=runtime,
                sample_rows=sample_rows,
                max_tables=max_tables,
                selected_tables=selected_tables,
                relationships_from_db=relationships_from_db,
            )
            try:
                draft = self._generate_draft(prompt_payload)
            except Exception as exc:
                if len(selected_tables) <= 1 or not _is_output_length_error(exc):
                    raise
                midpoint = len(selected_tables) // 2
                logger.warning(
                    "Semantic generation output exceeded the model limit for %s tables; "
                    "retrying as %s and %s table batches",
                    len(selected_tables),
                    midpoint,
                    len(selected_tables) - midpoint,
                )
                generate_group(selected_tables[:midpoint])
                generate_group(selected_tables[midpoint:])
                return
            drafts.append(draft)
            successful_payloads.append(prompt_payload)

        for start in range(0, len(tables), batch_size):
            generate_group(tables[start : start + batch_size])
        return drafts, successful_payloads

    def _complete_missing_metrics(
        self,
        *,
        catalog: SemanticCatalog,
        draft: SemanticGenerationDraft,
        helper: DBAnalyticsHelper | None,
        runtime: RuntimeDBConnectionConfig | None,
        tables: list[Any],
        sample_rows: int,
        max_tables: int,
        relationships_from_db: list[dict[str, str]],
        batch_size: int,
    ) -> SemanticGenerationDraft:
        table_map = _table_lookup(catalog)
        covered_tables = {
            table_map.get(metric.base_table, metric.base_table)
            for metric in [*catalog.metrics, *draft.metrics]
        }
        candidate_tables = {
            column.table
            for column in catalog.columns
            if column.semantic_role == "metric_candidate" and not column.is_hidden
        }
        missing = [
            table
            for table in tables
            if table.qualified_name in candidate_tables and table.qualified_name not in covered_tables
        ]
        if not missing:
            return draft

        metric_drafts: list[SemanticGenerationDraft] = [draft]
        for start in range(0, len(missing), batch_size):
            selected = missing[start : start + batch_size]
            payload = self._prompt_payload(
                catalog=catalog,
                helper=helper,
                runtime=runtime,
                sample_rows=sample_rows,
                max_tables=max_tables,
                selected_tables=selected,
                relationships_from_db=relationships_from_db,
            )
            payload["task"] = (
                "metrics_only: create 1-3 conservative simple business metrics for every "
                "provided table that has clear numeric measures; return all other arrays empty"
            )
            payload["response_limits"].update(
                {
                    "table_patches": 0,
                    "column_patches": 0,
                    "metrics": max(1, min(6, len(selected) * 3)),
                    "relationships": 0,
                    "terms": 0,
                }
            )
            focused = self._generate_draft(payload)
            metric_drafts.append(SemanticGenerationDraft(metrics=focused.metrics))
        return _merge_generation_drafts(metric_drafts)

    def _generate_draft(self, prompt_payload: dict[str, Any]) -> SemanticGenerationDraft:
        if self.llm_generate is not None:
            return self.llm_generate(prompt_payload)
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
        structured = llm.with_structured_output(SemanticGenerationDraft)
        return structured.invoke(
            [
                (
                    "system",
                    "You generate a conservative semantic layer for analytics. "
                    "Use only provided tables and columns. Prefer simple metrics, "
                    "many_to_one relationships, concise business terms, and safe aliases. "
                    "Do not invent columns. Do not include raw sample values in descriptions. "
                    "Keep the JSON compact: patch only columns whose business meaning or role "
                    "you can improve. "
                    "Create a simple metric only when row grain and column semantics make its "
                    "aggregation unambiguous. Set agg explicitly; omit ambiguous metrics rather "
                    "than guessing. Numeric identifiers and category codes remain dimensions or "
                    "identifiers. Strictly obey response_limits and use short descriptions. "
                    "Prioritize identifiers, times, and measures within the column cap. When task "
                    "is metrics_only, return only metrics for tables with an unambiguous numeric "
                    "business measure. If a metric applies to only a subset of rows, represent "
                    "that condition as a structured metric filter and use only observed example "
                    "values. "
                    "Return no prose outside the structured response.",
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
        selected_tables: list[Any] | None = None,
        relationships_from_db: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        tables = (
            selected_tables
            if selected_tables is not None
            else [table for table in catalog.tables if not table.is_hidden][:max_tables]
        )
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
        table_names = {table.qualified_name for table in tables}
        db_relationships = (
            relationships_from_db
            if relationships_from_db is not None
            else self._relationships_from_db(helper, catalog)
            if helper is not None
            else []
        )
        visible_column_count = sum(len(columns_by_table.get(table.qualified_name, [])) for table in tables)
        return {
            "db_type": runtime.db_type if runtime is not None else "csv_duckdb",
            "source_label": catalog.source_label,
            "profile": {
                "sample_strategy": catalog.profile_sample_strategy,
                "sample_limit": catalog.profile_sample_limit,
                "notice": (
                    "Column profile statistics were calculated only from the first "
                    f"{catalog.profile_sample_limit} rows of each table, not the complete source."
                    if catalog.profile_sample_strategy == "first_rows" and catalog.profile_sample_limit
                    else ""
                ),
            },
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
            "relationships_from_db": [
                rel
                for rel in db_relationships
                if rel.get("from_table") in table_names and rel.get("to_table") in table_names
            ],
            "samples": samples,
            "response_limits": {
                "table_patches": len(tables),
                "column_patches": min(visible_column_count, 16),
                "metrics": 6,
                "relationships": 4,
                "terms": 4,
                "description_chars": 200,
                "aliases_per_item": 2,
            },
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
        return [{str(key): _safe_sample_value(str(key), value) for key, value in row.items()} for row in rows]

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
        operation_id: int | None = None,
    ) -> SemanticCatalogGenerationResponse:
        table_map = _table_lookup(catalog)
        column_ids = {(column.table, column.name): column.column_id for column in catalog.columns}
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
            operation_id=operation_id,
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


def _merge_generation_drafts(drafts: list[SemanticGenerationDraft]) -> SemanticGenerationDraft:
    """Merge successful batch drafts deterministically before applying one overlay."""

    tables: dict[str, GeneratedTablePatch] = {}
    columns: dict[tuple[str, str], GeneratedColumnPatch] = {}
    metrics: dict[tuple[str, str], GeneratedMetricDraft] = {}
    relationships: dict[tuple[str, str, str, str], GeneratedRelationshipDraft] = {}
    terms: dict[str, GeneratedTermDraft] = {}
    for draft in drafts:
        for item in draft.tables:
            tables[item.table] = item
        for item in draft.columns:
            columns[(item.table, item.column)] = item
        for item in draft.metrics:
            metrics[(item.base_table, item.key)] = item
        for item in draft.relationships:
            relationships[(item.from_table, item.from_column, item.to_table, item.to_column)] = item
        for item in draft.terms:
            terms[item.name.strip().casefold()] = item
    metric_groups: dict[str, list[GeneratedMetricDraft]] = {}
    for item in metrics.values():
        metric_groups.setdefault(_generated_metric_key(item), []).append(item)
    merged_metrics: list[GeneratedMetricDraft] = []
    for normalized_key, items in metric_groups.items():
        if len(items) == 1:
            merged_metrics.append(items[0].model_copy(update={"key": normalized_key}))
            continue
        merged_metrics.extend(
            item.model_copy(update={"key": _qualified_metric_key(item.base_table, normalized_key)})
            for item in items
        )
    return SemanticGenerationDraft(
        tables=list(tables.values()),
        columns=list(columns.values()),
        metrics=merged_metrics,
        relationships=list(relationships.values()),
        terms=list(terms.values()),
    )


def _is_output_length_error(exc: BaseException) -> bool:
    """Recognize OpenAI-compatible structured-output truncation without coupling to one SDK version."""

    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if current.__class__.__name__ == "LengthFinishReasonError":
            return True
        message = str(current).casefold()
        if "length limit was reached" in message or ("finish_reason" in message and "length" in message):
            return True
        current = current.__cause__ or current.__context__
    return False


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


def _repair_generated_column_roles(
    *,
    draft: SemanticGenerationDraft,
    catalog: SemanticCatalog,
) -> SemanticGenerationDraft:
    """Let an explicit simple metric own its value column."""

    table_map = _table_lookup(catalog)
    catalog_columns = {(column.table, column.name) for column in catalog.columns}
    metric_columns = {
        (table, str(item.expr))
        for item in draft.metrics
        if item.type == "simple"
        and item.expr
        and item.agg
        and (table := table_map.get(item.base_table))
        and (table, str(item.expr)) in catalog_columns
    }
    repaired: list[GeneratedColumnPatch] = []
    for item in draft.columns:
        table = table_map.get(item.table)
        if table and item.semantic_role == "dimension" and (table, item.column) in metric_columns:
            repaired.append(item.model_copy(update={"semantic_role": "metric_candidate"}))
        else:
            repaired.append(item)
    return draft.model_copy(update={"columns": repaired})


def _effective_column_roles(
    *,
    draft: SemanticGenerationDraft,
    catalog: SemanticCatalog,
) -> dict[tuple[str, str], SemanticColumnRole]:
    table_map = _table_lookup(catalog)
    roles = {(column.table, column.name): column.semantic_role for column in catalog.columns}
    for item in draft.columns:
        table = table_map.get(item.table)
        if table and item.semantic_role is not None and (table, item.column) in roles:
            roles[(table, item.column)] = item.semantic_role
    return roles


def _safe_allowed_dimensions(
    *,
    draft: SemanticGenerationDraft,
    catalog: SemanticCatalog,
    table: str,
    requested: list[str],
) -> list[str]:
    roles = _effective_column_roles(draft=draft, catalog=catalog)
    columns = {
        column.name: column for column in catalog.columns if column.table == table and not column.is_hidden
    }
    result: list[str] = []
    for raw_ref in requested:
        ref = str(raw_ref or "").strip()
        name = ref
        if ref.startswith("dimension:"):
            name = ref.removeprefix("dimension:").removeprefix(f"{table}.")
        elif ref.startswith(f"{table}."):
            name = ref[len(table) + 1 :]
        column = columns.get(name)
        if column is None or roles.get((table, name)) not in {"dimension", "time", "flag"}:
            continue
        if _TECHNICAL_DIMENSION_RE.fullmatch(name):
            continue
        if name not in result:
            result.append(name)
    return result


def _metrics(
    draft: SemanticGenerationDraft,
    catalog: SemanticCatalog,
    table_map: dict[str, str],
) -> list[SemanticMetric]:
    metrics: list[SemanticMetric] = []
    existing_metric_tables = {metric.key: metric.base_table for metric in catalog.metrics}
    for item in draft.metrics:
        table = table_map.get(item.base_table)
        if not table:
            continue
        try:
            data = item.model_dump()
            data["base_table"] = table
            normalized_key = _generated_metric_key(item)
            existing_table = existing_metric_tables.get(normalized_key)
            data["key"] = (
                _qualified_metric_key(table, normalized_key)
                if existing_table is not None and existing_table != table
                else normalized_key
            )
            data["allowed_dimensions"] = _safe_allowed_dimensions(
                draft=draft,
                catalog=catalog,
                table=table,
                requested=item.allowed_dimensions,
            )
            metric = SemanticMetric(
                **data,
                metric_id=f"metric:{data['key']}",
            )
        except ValueError:
            continue
        metrics.append(metric)
    return metrics


def _metric_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")


def _generated_metric_key(item: GeneratedMetricDraft) -> str:
    for value in (item.key, item.expr or "", item.name):
        candidate = _metric_key(value)
        if not candidate:
            continue
        if len(candidate) <= 80:
            return candidate
        suffix = stable_id("metric-key", item.base_table, candidate)[:8]
        return f"{candidate[:71].rstrip('_')}_{suffix}"
    return f"measure_{stable_id('metric-key', item.base_table, item.key, item.name)[:8]}"


def _qualified_metric_key(table: str, key: str) -> str:
    prefix = _metric_key(table) or "table"
    candidate = f"{prefix}_{_metric_key(key)}".strip("_")
    if len(candidate) <= 80:
        return candidate
    suffix = stable_id("metric-key", table, key)[:8]
    return f"{candidate[:71].rstrip('_')}_{suffix}"


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
