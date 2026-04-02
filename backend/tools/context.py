from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from backend.core.config import Settings
from backend.data_access.db_runtime_service import RuntimeDBConnectionConfig

if TYPE_CHECKING:
    from backend.tools.sandbox import SessionSandbox


@dataclass
class ToolBuildContext:
    """Immutable snapshot of everything a ToolFactory needs to check availability and build."""

    settings: Settings
    allowed_tool_keys: set[str] | None = None
    df: pd.DataFrame | None = None
    tool_db_runtime: RuntimeDBConnectionConfig | None = None
    csv_loaded: bool = False
    csv_session_id: str | None = None
    sandbox: SessionSandbox | None = None

    @property
    def has_data(self) -> bool:
        """True when at least a DataFrame or a live DB connection is present."""
        return (
            self.df is not None
            or self.tool_db_runtime is not None
            or (self.csv_loaded and bool((self.csv_session_id or "").strip()))
        )

    @property
    def tool_df(self) -> pd.DataFrame:
        """Normalised DataFrame (never None) safe to pass to tool constructors."""
        return self.df if self.df is not None else pd.DataFrame()


