from __future__ import annotations

import json
from typing import Any, ClassVar, Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from backend.data_access.semantic_models import (
    SemanticColumnPatch,
    SemanticMetricCreate,
    SemanticMetricUpdate,
    SemanticRelationshipCreate,
    SemanticRelationshipUpdate,
    SemanticSearchRequest,
    SemanticTablePatch,
    SemanticTermCreate,
    SemanticTermUpdate,
)
from backend.data_access.semantic_validator import validate_semantic_catalog
from backend.tools.instructions import tool_description

ReadAction = Literal[
    "status",
    "get_catalog",
    "search",
    "resolve",
    "list_tables",
    "list_columns",
    "list_metrics",
    "list_relationships",
    "list_terms",
    "validate",
]
EditAction = Literal[
    "patch_table",
    "patch_column",
    "create_metric",
    "update_metric",
    "delete_metric",
    "create_relationship",
    "update_relationship",
    "delete_relationship",
    "create_term",
    "update_term",
    "delete_term",
]


class SemanticCatalogReadToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: ReadAction
    query: str | None = Field(default=None, description="Search text for action=search or resolve.")
    top_k: int = Field(default=8, ge=1, le=50)


class SemanticCatalogEditToolArgs(BaseModel):
    action: EditAction
    object_id: str | None = Field(
        default=None,
        description="Target table_id, column_id, metric_id, relationship_id, or term_id.",
    )
    table_patch: SemanticTablePatch | None = None
    column_patch: SemanticColumnPatch | None = None
    metric: SemanticMetricCreate | None = None
    metric_update: SemanticMetricUpdate | None = None
    relationship: SemanticRelationshipCreate | None = None
    relationship_update: SemanticRelationshipUpdate | None = None
    term: SemanticTermCreate | None = None
    term_update: SemanticTermUpdate | None = None

    @field_validator(
        "table_patch",
        "column_patch",
        "metric",
        "metric_update",
        "relationship",
        "relationship_update",
        "term",
        "term_update",
        mode="before",
    )
    @classmethod
    def parse_serialized_object(cls, value: Any) -> Any:
        """Accept structured tool arguments accidentally serialized as JSON strings."""
        if not isinstance(value, str):
            return value
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        return parsed if isinstance(parsed, dict) else value


class SemanticCatalogGenerateToolArgs(BaseModel):
    apply: bool = Field(
        default=False,
        description="Must be true to mutate the semantic catalog.",
    )
    sample_rows: int | None = Field(default=None, ge=0, le=10)
    max_tables: int | None = Field(default=None, ge=1, le=200)


class _SemanticCatalogToolMixin:
    _catalog_service: Any = PrivateAttr()
    _session_id: str = PrivateAttr()
    _user_id: int = PrivateAttr()
    _connection_id: str = PrivateAttr(default="")

    def _init_context(
        self,
        *,
        catalog_service: Any,
        session_id: str,
        user_id: int,
        connection_id: str = "",
    ) -> None:
        self._catalog_service = catalog_service
        self._session_id = session_id
        self._user_id = user_id
        self._connection_id = connection_id

    @staticmethod
    def _plain(payload: Any) -> Any:
        if isinstance(payload, BaseModel):
            return payload.model_dump()
        if isinstance(payload, list):
            return [_SemanticCatalogToolMixin._plain(item) for item in payload]
        if isinstance(payload, tuple):
            return [_SemanticCatalogToolMixin._plain(item) for item in payload]
        if isinstance(payload, dict):
            return {str(key): _SemanticCatalogToolMixin._plain(value) for key, value in payload.items()}
        return payload

    @staticmethod
    def _json(payload: Any) -> str:
        return json.dumps(
            _SemanticCatalogToolMixin._plain(payload),
            ensure_ascii=False,
            default=str,
            indent=2,
        )

    def _load_catalog(self):
        catalog = self._catalog_service.load_for_session(
            session_id=self._session_id,
            user_id=self._user_id,
        )
        if catalog is None:
            raise ValueError("Semantic catalog not found")
        return catalog


class SemanticCatalogReadTool(_SemanticCatalogToolMixin, BaseTool):
    name: str = "semantic_catalog_read_tool"
    description: str = tool_description("semantic_catalog_read_tool")
    args_schema: type[BaseModel] = SemanticCatalogReadToolArgs
    response_format: str = "content"
    parallel_safe: ClassVar[bool] = True

    def __init__(
        self,
        *,
        catalog_service: Any,
        session_id: str,
        user_id: int,
        connection_id: str = "",
    ) -> None:
        super().__init__()
        self._init_context(
            catalog_service=catalog_service,
            session_id=session_id,
            user_id=user_id,
            connection_id=connection_id,
        )

    def _run(
        self,
        action: ReadAction,
        query: str | None = None,
        top_k: int = 8,
    ) -> str:
        return self._json(
            self._run_action(
                action,
                query=query,
                top_k=top_k,
            )
        )

    def _run_action(
        self,
        action: ReadAction,
        *,
        query: str | None,
        top_k: int,
    ) -> Any:
        target_connection_id = self._connection_id.strip()
        if action == "status":
            if target_connection_id:
                return self._catalog_service.status_for_connection(connection_id=target_connection_id)
            catalog = self._catalog_service.load_for_session(
                session_id=self._session_id,
                user_id=self._user_id,
            )
            return {"status": "empty"} if catalog is None else catalog
        if action == "get_catalog":
            if target_connection_id:
                catalog = self._catalog_service.load_for_connection(
                    connection_id=target_connection_id,
                    user_id=self._user_id,
                )
                if catalog is None:
                    raise ValueError("Semantic catalog not found")
                return catalog
            return self._load_catalog()
        if action in {"search", "resolve"}:
            request = SemanticSearchRequest(query=str(query or ""), top_k=top_k)
            if target_connection_id:
                result = self._catalog_service.search_for_connection(
                    connection_id=target_connection_id,
                    user_id=self._user_id,
                    query=request.query,
                    top_k=request.top_k,
                )
            else:
                result = self._catalog_service.search(
                    session_id=self._session_id,
                    user_id=self._user_id,
                    query=request.query,
                    top_k=request.top_k,
                )
            if action == "search":
                catalog = result.hints.get("catalog")
                catalog = catalog if isinstance(catalog, dict) else {}

                def rows(name: str) -> list[Any]:
                    values = catalog.get(name) or result.hints.get(name, [])
                    return values if isinstance(values, list) else []

                entity_specs = {
                    "table": (
                        rows("tables"),
                        "table_id",
                        ("qualified_name", "description", "semantic_role", "grain", "aliases"),
                    ),
                    "column": (
                        rows("columns"),
                        "column_id",
                        ("table", "name", "description", "semantic_role", "aliases"),
                    ),
                    "entity": (
                        rows("entities"),
                        "entity_id",
                        ("name", "table", "expr", "type", "description", "synonyms"),
                    ),
                    "dimension": (
                        rows("dimensions"),
                        "dimension_id",
                        ("name", "table", "expr", "type", "grains", "description", "synonyms"),
                    ),
                    "fact": (
                        rows("facts"),
                        "fact_id",
                        ("name", "table", "expr", "type", "description", "synonyms"),
                    ),
                    "metric": (
                        rows("metrics"),
                        "metric_id",
                        ("key", "name", "description", "type", "base_table", "synonyms"),
                    ),
                    "relationship": (
                        rows("relationships"),
                        "relationship_id",
                        (
                            "from_table",
                            "from_column",
                            "to_table",
                            "to_column",
                            "cardinality",
                            "description",
                        ),
                    ),
                    "term": (
                        rows("terms"),
                        "term_id",
                        ("name", "description", "synonyms", "entity_refs"),
                    ),
                    "saved_query": (
                        rows("saved_queries"),
                        "query_id",
                        ("name", "metrics", "dimensions", "filters", "description"),
                    ),
                }
                entities = {
                    (entity_type, str(entity[id_field])): (entity, fields)
                    for entity_type, (rows, id_field, fields) in entity_specs.items()
                    for entity in rows
                    if isinstance(entity, dict) and entity.get(id_field)
                }
                candidates = []
                for item in result.items:
                    resolved = entities.get((item.entity_type, item.entity_id))
                    if resolved is None:
                        continue
                    entity, fields = resolved
                    candidate = {
                        "entity_type": item.entity_type,
                        "entity_id": item.entity_id,
                        "score": item.score,
                    }
                    candidate.update(
                        {
                            field: value
                            for field in fields
                            if (value := entity.get(field)) not in (None, "", [])
                        }
                    )
                    candidates.append(candidate)
                return {
                    "status": result.status,
                    "matched": bool(candidates),
                    "query": request.query,
                    "candidates": candidates,
                }
            keys = set(result.hints.get("confirmed_metric_keys") or [])
            candidate_keys = set(result.hints.get("candidate_metric_keys") or [])
            metrics = [item for item in result.hints.get("metrics", []) if isinstance(item, dict)]

            def metric_contract(metric: dict[str, Any]) -> dict[str, Any]:
                return {
                    key: metric.get(key)
                    for key in (
                        "key",
                        "name",
                        "type",
                        "base_table",
                        "expr",
                        "agg",
                        "numerator",
                        "denominator",
                        "formula",
                        "format",
                        "default_time_dimension",
                        "allowed_dimensions",
                        "filters",
                    )
                }

            def metric_candidate(metric: dict[str, Any]) -> dict[str, Any]:
                return {key: metric.get(key) for key in ("key", "name", "description")}

            def relationship_contract(relationship: dict[str, Any]) -> dict[str, Any]:
                return {
                    key: relationship.get(key)
                    for key in (
                        "from_table",
                        "from_column",
                        "to_table",
                        "to_column",
                        "cardinality",
                        "description",
                    )
                }

            resolved_metrics = [metric for metric in metrics if metric.get("key") in keys]
            base_tables = {str(metric.get("base_table") or "") for metric in resolved_metrics}
            relationship_rows = [
                relationship
                for relationship in result.hints.get("relationships", [])
                if isinstance(relationship, dict)
            ]
            if len(base_tables) > 1:
                catalog = result.hints.get("catalog")
                catalog = catalog if isinstance(catalog, dict) else {}
                relationship_rows.extend(
                    relationship
                    for relationship in catalog.get("relationships", [])
                    if isinstance(relationship, dict)
                    and (
                        relationship.get("from_table") in base_tables
                        or relationship.get("to_table") in base_tables
                    )
                )
            relationships = list(
                {
                    tuple(relationship_contract(item).values()): relationship_contract(item)
                    for item in relationship_rows
                }.values()
            )
            terms = [
                {
                    "name": term.get("name"),
                    "description": term.get("description"),
                    "synonyms": term.get("synonyms", []),
                    "entity_refs": term.get("entity_refs", []),
                }
                for term in result.hints.get("terms", [])
            ]
            metric_resolution_status = result.hints.get(
                "metric_resolution_status",
                result.hints.get("definition_status", "not_found"),
            )
            term_resolution_status = result.hints.get("term_resolution_status", "not_found")
            payload: dict[str, Any] = {
                "matched": bool(keys or terms or relationships),
                "definition_status": metric_resolution_status,
                "metric_resolution_status": metric_resolution_status,
                "term_resolution_status": term_resolution_status,
            }
            if keys:
                payload["execution_mode"] = "semantic_query" if len(base_tables) == 1 else "execute_sql"
            if len(base_tables) > 1:
                payload["execution_note"] = (
                    "Metrics use different base tables. Select only contracts whose "
                    "allowed_dimensions support the requested join grain. Do not call "
                    "semantic_query. Aggregate each selected contract in its own CTE to the "
                    "join grain, then join those CTEs through the declared relationships."
                )
            payload.update(
                {
                    "confirmed_metric_keys": sorted(keys),
                    "candidate_metric_keys": sorted(candidate_keys),
                    "metrics": [metric_contract(metric) for metric in resolved_metrics],
                    "candidates": [
                        metric_candidate(metric)
                        for metric in metrics
                        if metric.get("key") in candidate_keys and metric.get("key") not in keys
                    ],
                    "terms": terms,
                    "relationships": relationships,
                }
            )
            if metric_resolution_status == "missing":
                payload["formula_required"] = True
            return payload

        catalog = self._load_catalog()
        if action == "list_tables":
            return catalog.tables
        if action == "list_columns":
            return catalog.columns
        if action == "list_metrics":
            return catalog.metrics
        if action == "list_relationships":
            return catalog.relationships
        if action == "list_terms":
            return catalog.terms
        if action == "validate":
            return validate_semantic_catalog(catalog)
        raise ValueError(f"Unknown semantic catalog read action: {action}")


class SemanticCatalogEditTool(_SemanticCatalogToolMixin, BaseTool):
    name: str = "semantic_catalog_edit_tool"
    description: str = tool_description("semantic_catalog_edit_tool")
    args_schema: type[BaseModel] = SemanticCatalogEditToolArgs
    response_format: str = "content"
    parallel_safe: ClassVar[bool] = False

    def __init__(self, *, catalog_service: Any, session_id: str, user_id: int) -> None:
        super().__init__()
        self._init_context(
            catalog_service=catalog_service,
            session_id=session_id,
            user_id=user_id,
        )

    def _run(
        self,
        action: EditAction,
        object_id: str | None = None,
        table_patch: SemanticTablePatch | None = None,
        column_patch: SemanticColumnPatch | None = None,
        metric: SemanticMetricCreate | None = None,
        metric_update: SemanticMetricUpdate | None = None,
        relationship: SemanticRelationshipCreate | None = None,
        relationship_update: SemanticRelationshipUpdate | None = None,
        term: SemanticTermCreate | None = None,
        term_update: SemanticTermUpdate | None = None,
    ) -> str:
        result = self._run_action(
            action,
            object_id=str(object_id or ""),
            table_patch=table_patch,
            column_patch=column_patch,
            metric=metric,
            metric_update=metric_update,
            relationship=relationship,
            relationship_update=relationship_update,
            term=term,
            term_update=term_update,
        )
        return self._json(result)

    def _run_action(
        self,
        action: EditAction,
        *,
        object_id: str,
        table_patch: SemanticTablePatch | None,
        column_patch: SemanticColumnPatch | None,
        metric: SemanticMetricCreate | None,
        metric_update: SemanticMetricUpdate | None,
        relationship: SemanticRelationshipCreate | None,
        relationship_update: SemanticRelationshipUpdate | None,
        term: SemanticTermCreate | None,
        term_update: SemanticTermUpdate | None,
    ) -> Any:
        if action == "patch_table":
            if table_patch is None:
                raise ValueError("table_patch is required for patch_table")
            return self._catalog_service.patch_table(
                session_id=self._session_id,
                user_id=self._user_id,
                table_id=object_id,
                payload=table_patch,
            )
        if action == "patch_column":
            if column_patch is None:
                raise ValueError("column_patch is required for patch_column")
            return self._catalog_service.patch_column(
                session_id=self._session_id,
                user_id=self._user_id,
                column_id=object_id,
                payload=column_patch,
            )
        if action == "create_metric":
            if metric is None:
                raise ValueError("metric is required for create_metric")
            return self._catalog_service.create_metric(
                session_id=self._session_id,
                user_id=self._user_id,
                payload=metric,
            )
        if action == "update_metric":
            if metric_update is None:
                raise ValueError("metric_update is required for update_metric")
            return self._catalog_service.update_metric(
                session_id=self._session_id,
                user_id=self._user_id,
                metric_id=object_id,
                payload=metric_update,
            )
        if action == "delete_metric":
            self._catalog_service.delete_metric(
                session_id=self._session_id,
                user_id=self._user_id,
                metric_id=object_id,
            )
            return {"status": "deleted", "object_id": object_id}
        if action == "create_relationship":
            if relationship is None:
                raise ValueError("relationship is required for create_relationship")
            return self._catalog_service.create_relationship(
                session_id=self._session_id,
                user_id=self._user_id,
                payload=relationship,
            )
        if action == "update_relationship":
            if relationship_update is None:
                raise ValueError("relationship_update is required for update_relationship")
            return self._catalog_service.update_relationship(
                session_id=self._session_id,
                user_id=self._user_id,
                relationship_id=object_id,
                payload=relationship_update,
            )
        if action == "delete_relationship":
            self._catalog_service.delete_relationship(
                session_id=self._session_id,
                user_id=self._user_id,
                relationship_id=object_id,
            )
            return {"status": "deleted", "object_id": object_id}
        if action == "create_term":
            if term is None:
                raise ValueError("term is required for create_term")
            return self._catalog_service.create_term(
                session_id=self._session_id,
                user_id=self._user_id,
                payload=term,
            )
        if action == "update_term":
            if term_update is None:
                raise ValueError("term_update is required for update_term")
            return self._catalog_service.update_term(
                session_id=self._session_id,
                user_id=self._user_id,
                term_id=object_id,
                payload=term_update,
            )
        if action == "delete_term":
            self._catalog_service.delete_term(
                session_id=self._session_id,
                user_id=self._user_id,
                term_id=object_id,
            )
            return {"status": "deleted", "object_id": object_id}
        raise ValueError(f"Unknown semantic catalog edit action: {action}")


class SemanticCatalogGenerateTool(BaseTool):
    name: str = "semantic_catalog_generate_tool"
    description: str = tool_description("semantic_catalog_generate_tool")
    args_schema: type[BaseModel] = SemanticCatalogGenerateToolArgs
    response_format: str = "content"
    parallel_safe: ClassVar[bool] = False

    _generation_service: Any = PrivateAttr()
    _session_id: str = PrivateAttr()
    _user_id: int = PrivateAttr()

    def __init__(self, *, generation_service: Any, session_id: str, user_id: int) -> None:
        super().__init__()
        self._generation_service = generation_service
        self._session_id = session_id
        self._user_id = user_id

    def _run(
        self,
        apply: bool = False,
        sample_rows: int | None = None,
        max_tables: int | None = None,
    ) -> str:
        if not apply:
            return json.dumps(
                {
                    "status": "confirmation_required",
                    "message": (
                        "This tool mutates the semantic catalog. Ask the user to confirm, "
                        "then call again with apply=true."
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )

        from backend.data_access.semantic_generation_service import (
            SemanticCatalogGenerationRequest,
        )

        payload = {
            key: value
            for key, value in {
                "sample_rows": sample_rows,
                "max_tables": max_tables,
                "ensure_metrics": True,
            }.items()
            if value is not None
        }
        result = self._generation_service.generate(
            session_id=self._session_id,
            user_id=self._user_id,
            request=SemanticCatalogGenerationRequest.model_validate(payload),
        )
        catalog = result.catalog
        return json.dumps(
            {
                "status": "applied",
                "applied": True,
                "catalog_id": catalog.catalog_id,
                "source_key": catalog.source_key,
                "catalog_status": catalog.status,
                "published_version": catalog.published_version,
                "summary": result.summary.model_dump(),
                "validation": catalog.validation.model_dump(),
            },
            ensure_ascii=False,
            default=str,
            indent=2,
        )
