from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, model_validator

from backend.data_access.semantic_models import (
    SemanticTerm,
    clean_list,
    stable_id,
    utc_now_iso,
)

logger = logging.getLogger(__name__)

_SEED_GLOB = "semantic_seed.*.json"


class SemanticSeedTerm(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = ""
    synonyms: list[str] = Field(default_factory=list)
    entity_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize(self) -> SemanticSeedTerm:
        self.name = self.name.strip()
        self.synonyms = clean_list(self.synonyms)
        self.entity_refs = clean_list(self.entity_refs)
        return self


class SemanticSeedPack(BaseModel):
    name: str
    enabled_by_default: bool = True
    domains: list[str] = Field(default_factory=list)
    terms: list[SemanticSeedTerm] = Field(default_factory=list)


@lru_cache(maxsize=1)
def load_semantic_seed_packs(seed_dir: Path = Path(__file__).parent) -> tuple[SemanticSeedPack, ...]:
    packs: list[SemanticSeedPack] = []
    paths = sorted(
        seed_dir.glob(_SEED_GLOB),
        key=lambda path: (0 if path.name == "semantic_seed.global.json" else 1, path.name),
    )
    for path in paths:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw.setdefault("name", path.stem.replace("semantic_seed.", ""))
            packs.append(SemanticSeedPack.model_validate(raw))
        except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
            logger.warning("Semantic seed pack %s is invalid: %s", path.name, exc)
    return tuple(packs)


def _enabled_terms() -> list[SemanticSeedTerm]:
    return [term for pack in load_semantic_seed_packs() if pack.enabled_by_default for term in pack.terms]


def starter_terms(source_key: str) -> list[SemanticTerm]:
    now = utc_now_iso()
    return [
        SemanticTerm(
            term_id=f"term:{stable_id('term', source_key, item.name)}",
            name=item.name,
            description=item.description,
            synonyms=item.synonyms,
            entity_refs=item.entity_refs,
            created_at=now,
            updated_at=now,
        )
        for item in _enabled_terms()
    ]
