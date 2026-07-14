from __future__ import annotations

import hashlib
import math
import re
import uuid
from dataclasses import dataclass
from typing import Any

from backend.data_access.semantic_models import (
    SemanticCatalog,
    SemanticColumn,
    SemanticDimension,
    SemanticEntity,
    SemanticFact,
    SemanticMetric,
    SemanticRelationship,
    SemanticSavedQuery,
    SemanticSearchResultItem,
    SemanticTable,
    SemanticTerm,
    stable_id,
)


@dataclass(frozen=True)
class SemanticChunk:
    point_id: str
    text: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class LocalHashEmbeddings:
    dimension: int = 1536

    def embed_documents(self, texts: list[str], **_: Any) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        tokens = re.findall(r"[^\W_]+", str(text).lower(), flags=re.UNICODE)
        features = [*tokens, *(f"{left}_{right}" for left, right in zip(tokens, tokens[1:]))]
        vector = [0.0] * max(1, int(self.dimension))
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % len(vector)
            vector[bucket] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


@dataclass
class SemanticVectorStore:
    url: str
    collection: str
    vector_enabled: bool = False
    api_key: str = ""
    timeout_sec: float = 10.0
    embedding_model: str = ""
    embedding_provider: str = "local"
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_dim: int = 1536
    embedding_timeout_sec: float = 30.0
    embedding_batch_size: int = 64

    _client: Any | None = None
    _embeddings_client: Any | None = None

    @classmethod
    def from_settings(cls, settings: Any) -> SemanticVectorStore:
        return cls(
            url=str(getattr(settings, "semantic_qdrant_url", "") or "").strip(),
            api_key=str(getattr(settings, "semantic_qdrant_api_key", "") or "").strip(),
            collection=str(
                getattr(settings, "semantic_qdrant_collection", "semantic_catalog_chunks") or ""
            ).strip()
            or "semantic_catalog_chunks",
            vector_enabled=bool(getattr(settings, "semantic_vector_enabled", False)),
            timeout_sec=float(getattr(settings, "semantic_qdrant_timeout_sec", 10) or 10),
            embedding_model=str(getattr(settings, "semantic_embedding_model", "") or "").strip(),
            embedding_provider=str(
                getattr(settings, "semantic_embedding_provider", "local") or "local"
            ).strip().lower(),
            embedding_base_url=str(getattr(settings, "semantic_embedding_base_url", "") or "").strip(),
            embedding_api_key=str(getattr(settings, "semantic_embedding_api_key", "") or "").strip(),
            embedding_dim=int(getattr(settings, "semantic_embedding_dim", 1536) or 1536),
            embedding_timeout_sec=float(getattr(settings, "semantic_embedding_timeout_sec", 30) or 30),
            embedding_batch_size=max(1, int(getattr(settings, "semantic_embedding_batch_size", 64) or 64)),
        )

    @property
    def enabled(self) -> bool:
        provider_ready = self.embedding_provider in {"local", "hash"} or bool(self.embedding_model)
        return bool(self.vector_enabled and self.url and self.collection and provider_ready)

    def _embeddings(self) -> Any:
        if self._embeddings_client is not None:
            return self._embeddings_client
        if self.embedding_provider in {"local", "hash"}:
            self._embeddings_client = LocalHashEmbeddings(max(1, int(self.embedding_dim)))
            return self._embeddings_client
        if self.embedding_provider != "openai":
            raise ValueError(f"Unsupported semantic embedding provider: {self.embedding_provider}")
        try:
            from langchain_openai import OpenAIEmbeddings
        except Exception as exc:  # pragma: no cover - installed in runtime requirements
            raise RuntimeError("langchain-openai is required for semantic embeddings") from exc

        kwargs: dict[str, Any] = {
            "model": self.embedding_model,
            "timeout": self.embedding_timeout_sec,
        }
        if self.embedding_base_url:
            kwargs["base_url"] = self.embedding_base_url
        if self.embedding_api_key:
            kwargs["api_key"] = self.embedding_api_key
        if self.embedding_dim > 0:
            kwargs["dimensions"] = self.embedding_dim
        self._embeddings_client = OpenAIEmbeddings(**kwargs)
        return self._embeddings_client

    def _qdrant(self) -> tuple[Any, Any]:
        try:
            from qdrant_client import QdrantClient, models
        except Exception as exc:  # pragma: no cover - depends on optional runtime package
            raise RuntimeError("qdrant-client is required for semantic vector search") from exc
        return QdrantClient, models

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        QdrantClient, _models = self._qdrant()
        kwargs: dict[str, Any] = {"url": self.url, "timeout": self.timeout_sec}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        self._client = QdrantClient(**kwargs)
        return self._client

    def _ensure_collection(self) -> None:
        client = self._get_client()
        _QdrantClient, models = self._qdrant()
        if client.collection_exists(self.collection):
            return
        client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=int(self.embedding_dim or 1536),
                distance=models.Distance.COSINE,
            ),
        )

    def upsert_catalog(self, catalog: SemanticCatalog) -> None:
        if not self.enabled:
            return
        chunks = catalog_chunks(catalog)
        self._ensure_collection()
        self.delete_catalog(catalog)
        if not chunks:
            return
        _QdrantClient, models = self._qdrant()
        vectors: list[list[float]] = []
        texts = [chunk.text for chunk in chunks]
        embeddings = self._embeddings()
        for start in range(0, len(texts), self.embedding_batch_size):
            vectors.extend(
                embeddings.embed_documents(
                    texts[start : start + self.embedding_batch_size],
                    chunk_size=self.embedding_batch_size,
                )
            )
        points = [
            models.PointStruct(
                id=chunk.point_id,
                vector=vector,
                payload=chunk.payload,
            )
            for chunk, vector in zip(chunks, vectors, strict=False)
        ]
        if points:
            self._get_client().upsert(collection_name=self.collection, points=points)

    def delete_catalog(self, catalog: SemanticCatalog) -> None:
        if not self.enabled:
            return
        _QdrantClient, models = self._qdrant()
        self._get_client().delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=self._filter(
                    source_key=catalog.source_key,
                    catalog_id=catalog.catalog_id,
                    source_fingerprint=catalog.source_fingerprint,
                )
            ),
        )

    def search(
        self,
        *,
        catalog: SemanticCatalog,
        query: str,
        top_k: int,
        entity_type: str | None = None,
    ) -> list[SemanticSearchResultItem]:
        if not self.enabled:
            return []
        vector = self._embeddings().embed_query(query)
        hits = self._get_client().search(
            collection_name=self.collection,
            query_vector=vector,
            query_filter=self._filter(
                source_key=catalog.source_key,
                catalog_id=catalog.catalog_id,
                source_fingerprint=catalog.source_fingerprint,
                entity_type=entity_type,
            ),
            limit=max(1, int(top_k)),
            with_payload=True,
        )
        return [
            SemanticSearchResultItem(
                entity_type=str(hit.payload.get("entity_type")),
                entity_id=str(hit.payload.get("entity_id")),
                score=float(getattr(hit, "score", 0.0) or 0.0),
                payload=dict(hit.payload or {}),
            )
            for hit in hits
            if getattr(hit, "payload", None)
        ]

    def _filter(
        self,
        *,
        source_key: str,
        catalog_id: str,
        source_fingerprint: str,
        entity_type: str | None = None,
    ) -> Any:
        _QdrantClient, models = self._qdrant()
        conditions = [
            models.FieldCondition(key="source_key", match=models.MatchValue(value=source_key)),
            models.FieldCondition(key="catalog_id", match=models.MatchValue(value=catalog_id)),
            models.FieldCondition(
                key="source_fingerprint",
                match=models.MatchValue(value=source_fingerprint),
            ),
        ]
        if entity_type:
            conditions.append(
                models.FieldCondition(key="entity_type", match=models.MatchValue(value=entity_type))
            )
        return models.Filter(must=conditions)


def catalog_chunks(catalog: SemanticCatalog) -> list[SemanticChunk]:
    chunks: list[SemanticChunk] = []
    seen: set[tuple[str, str]] = set()

    def add(entity_type: str, entity_id: str, text: str) -> None:
        key = (entity_type, entity_id)
        if key in seen:
            return
        seen.add(key)
        chunks.append(_chunk(catalog, entity_type, entity_id, text))

    for table in catalog.tables:
        if table.is_hidden:
            continue
        add("table", table.table_id, _table_text(table))
    for column in catalog.columns:
        if column.is_hidden:
            continue
        add("column", column.column_id, _column_text(column))
    for entity in catalog.entities:
        add("entity", entity.entity_id, _entity_text(entity))
    for dimension in catalog.dimensions:
        add("dimension", dimension.dimension_id, _dimension_text(dimension))
    for fact in catalog.facts:
        add("fact", fact.fact_id, _fact_text(fact))
    for metric in catalog.metrics:
        add("metric", metric.metric_id, _metric_text(metric))
    for relationship in catalog.relationships:
        add("relationship", relationship.relationship_id, _relationship_text(relationship))
    for term in catalog.terms:
        add("term", term.term_id, _term_text(term))
    for query in catalog.saved_queries:
        add("saved_query", query.query_id, _saved_query_text(query))
    return [chunk for chunk in chunks if chunk.text.strip()]


def _chunk(
    catalog: SemanticCatalog,
    entity_type: str,
    entity_id: str,
    text: str,
) -> SemanticChunk:
    payload = {
        "user_id": catalog.user_id,
        "session_id": catalog.session_id,
        "source_key": catalog.source_key,
        "catalog_id": catalog.catalog_id,
        "source_fingerprint": catalog.source_fingerprint,
        "published_version": catalog.published_version,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "text": text[:2000],
    }
    return SemanticChunk(
        point_id=str(uuid.UUID(stable_id("semantic", catalog.catalog_id, entity_type, entity_id))),
        text=text,
        payload=payload,
    )


def _table_text(table: SemanticTable) -> str:
    return " ".join(
        [
            "table",
            table.qualified_name,
            table.table_name,
            table.description,
            table.ai_context,
            table.semantic_role,
            table.grain,
            " ".join(table.aliases),
            " ".join(table.tags),
        ]
    )


def _column_text(column: SemanticColumn) -> str:
    return " ".join(
        [
            "column",
            column.table,
            column.name,
            column.dtype,
            column.semantic_role,
            column.description,
            column.ai_context,
            " ".join(column.aliases),
        ]
    )


def _entity_text(entity: SemanticEntity) -> str:
    return " ".join(
        [
            "entity",
            entity.name,
            entity.type,
            entity.table,
            entity.expr,
            entity.description,
            " ".join(entity.synonyms),
        ]
    )


def _dimension_text(dimension: SemanticDimension) -> str:
    return " ".join(
        [
            "dimension",
            dimension.name,
            dimension.type,
            dimension.table,
            dimension.expr,
            " ".join(dimension.grains),
            dimension.description,
            " ".join(dimension.synonyms),
        ]
    )


def _fact_text(fact: SemanticFact) -> str:
    return " ".join(["fact", fact.name, fact.type, fact.table, fact.expr, fact.description, " ".join(fact.synonyms)])


def _metric_text(metric: SemanticMetric) -> str:
    return " ".join(
        [
            "metric",
            metric.key,
            metric.name,
            metric.type,
            metric.description,
            metric.base_table,
            metric.expr or "",
            metric.agg or "",
            metric.formula,
            metric.numerator or "",
            metric.denominator or "",
            metric.default_time_dimension or "",
            " ".join(metric.allowed_dimensions),
            " ".join(metric.synonyms),
        ]
    )


def _relationship_text(relationship: SemanticRelationship) -> str:
    return " ".join(
        [
            "relationship",
            relationship.from_table,
            relationship.from_column,
            relationship.to_table,
            relationship.to_column,
            relationship.description,
        ]
    )


def _term_text(term: SemanticTerm) -> str:
    return " ".join(["term", term.name, term.description, " ".join(term.synonyms)])


def _saved_query_text(query: SemanticSavedQuery) -> str:
    return " ".join(
        [
            "saved_query",
            query.name,
            query.description,
            " ".join(query.metrics),
            " ".join(query.dimensions),
            " ".join(query.filters),
        ]
    )
