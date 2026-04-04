"""Multi-source session model.

A session can bind N CSV files and M DB connections simultaneously.
Each source gets a stable ``alias`` (e.g. ``sales_csv``, ``warehouse_pg``)
and a ``variable_name`` that the kernel injects into scope (e.g. ``sales_df``).

``SessionManifest`` is the lightweight identity object that replaces the
monolithic ``SessionState`` for source management.  Chat history and
artifacts stay in the existing ``state.json`` — the manifest only tracks
sources and session identity.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from backend.notebook.models import utcnow_iso

# ── SessionSource ────────────────────────────────────────────────────────────


@dataclass
class SessionSource:
    """A named data source bound to a session."""

    alias: str
    """Stable identifier within the session: ``sales_csv``, ``warehouse_pg``."""

    source_type: Literal["csv", "db_connection"]

    display_name: str = ""
    """Human-readable label shown in UI."""

    variable_name: str = ""
    """Variable name injected into kernel scope: ``sales_df``."""

    # CSV-specific
    file_name: str | None = None
    parquet_path: str | None = None

    # DB-specific
    connection_id: str | None = None
    connection_name: str | None = None

    # DuckDB CSV runtime state
    csv_session_id: str | None = None
    csv_table_names: list[str] = field(default_factory=list)
    csv_expires_at: int | None = None

    # Common
    bound_at: str = field(default_factory=utcnow_iso)
    schema_hint: dict[str, str] = field(default_factory=dict)
    """column_name → dtype string, populated after first load."""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "alias": self.alias,
            "source_type": self.source_type,
            "display_name": self.display_name,
            "variable_name": self.variable_name,
            "bound_at": self.bound_at,
        }
        if self.file_name:
            d["file_name"] = self.file_name
        if self.parquet_path:
            d["parquet_path"] = self.parquet_path
        if self.connection_id:
            d["connection_id"] = self.connection_id
        if self.connection_name:
            d["connection_name"] = self.connection_name
        if self.csv_session_id:
            d["csv_session_id"] = self.csv_session_id
        if self.csv_table_names:
            d["csv_table_names"] = self.csv_table_names
        if self.csv_expires_at is not None:
            d["csv_expires_at"] = self.csv_expires_at
        if self.schema_hint:
            d["schema_hint"] = self.schema_hint
        return d

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SessionSource:
        return cls(
            alias=raw["alias"],
            source_type=raw["source_type"],
            display_name=raw.get("display_name", ""),
            variable_name=raw.get("variable_name", ""),
            file_name=raw.get("file_name"),
            parquet_path=raw.get("parquet_path"),
            connection_id=raw.get("connection_id"),
            connection_name=raw.get("connection_name"),
            csv_session_id=raw.get("csv_session_id"),
            csv_table_names=raw.get("csv_table_names", []),
            csv_expires_at=raw.get("csv_expires_at"),
            bound_at=raw.get("bound_at", ""),
            schema_hint=raw.get("schema_hint", {}),
        )


# ── SessionManifest ──────────────────────────────────────────────────────────


@dataclass
class SessionManifest:
    """Lightweight session identity + source list.

    The manifest is persisted alongside the notebook.  Chat history and
    artifacts remain in the existing ``state.json`` for backward
    compatibility.
    """

    session_id: str = ""
    created_at: str = ""
    last_access: str = ""
    sources: list[SessionSource] = field(default_factory=list)
    selected_skill_ids: list[str] = field(default_factory=list)

    # ── Source access ────────────────────────────────────────────────────

    def source_by_alias(self, alias: str) -> SessionSource | None:
        return next((s for s in self.sources if s.alias == alias), None)

    def primary_source(self) -> SessionSource | None:
        """First source — used for backward-compat with single-source APIs."""
        return self.sources[0] if self.sources else None

    def add_source(self, source: SessionSource) -> None:
        existing = self.source_by_alias(source.alias)
        if existing is not None:
            self.sources.remove(existing)
        self.sources.append(source)

    def remove_source(self, alias: str) -> SessionSource | None:
        source = self.source_by_alias(alias)
        if source is not None:
            self.sources.remove(source)
        return source

    def has_csv(self) -> bool:
        return any(s.source_type == "csv" for s in self.sources)

    def has_db(self) -> bool:
        return any(s.source_type == "db_connection" for s in self.sources)

    # ── Serialisation ────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_access": self.last_access,
            "sources": [s.to_dict() for s in self.sources],
            "selected_skill_ids": self.selected_skill_ids,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SessionManifest:
        sources_raw = raw.get("sources", [])
        return cls(
            session_id=raw.get("session_id", ""),
            created_at=raw.get("created_at", ""),
            last_access=raw.get("last_access", ""),
            sources=[SessionSource.from_dict(s) for s in sources_raw],
            selected_skill_ids=raw.get("selected_skill_ids", []),
        )


# ── Alias / variable name generation ────────────────────────────────────────

_SAFE_CHARS = re.compile(r"[^a-z0-9_]")


def make_source_alias(name: str, source_type: str, existing: list[str]) -> str:
    """Generate a unique alias from a display name.

    Examples::

        make_source_alias("Sales Q4.csv", "csv", [])          → "sales_q4_csv"
        make_source_alias("Sales Q4.csv", "csv", ["sales_q4_csv"]) → "sales_q4_csv_2"
        make_source_alias("Warehouse", "db_connection", [])    → "warehouse_db"
    """
    stem = name.rsplit(".", 1)[0] if "." in name else name
    slug = _SAFE_CHARS.sub("_", stem.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug:
        slug = "source"

    suffix = "_csv" if source_type == "csv" else "_db"
    if not slug.endswith(suffix):
        slug += suffix

    # Deduplicate
    base = slug
    counter = 2
    while slug in existing:
        slug = f"{base}_{counter}"
        counter += 1

    return slug


def alias_to_variable_name(alias: str) -> str:
    """Convert source alias to a Python variable name.

    ``sales_csv`` → ``sales_df``,  ``warehouse_db`` → ``warehouse_conn``
    """
    if alias.endswith("_csv"):
        return alias[:-4] + "_df"
    if alias.endswith("_db"):
        return alias[:-3] + "_conn"
    return alias + "_data"
