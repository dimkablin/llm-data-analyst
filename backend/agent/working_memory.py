from __future__ import annotations
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ArtifactHandle:
    """
    Lightweight read-only projection of an artifact's metadata.
    Canonical source: ArtifactStore (lookup by id to get actual data).
    Created once at artifact creation time. Immutable after creation.
    """
    id: str                           # matches ArtifactStore.artifact.id
    name: str                         # artifact_name from tool_result
    type: str                         # "table" | "plot" | "value" | "error"
    tool_name: str                    # "sql_tool" | "pandas_tool" | "plotly_tool" | ...
    step_index: int                   # step counter when created
    schema: dict[str, str] | None     # col → dtype, tables only
    row_count: int | None             # tables only
    summary: str | None               # deterministic one-liner

    @property
    def masked_ref(self) -> str:
        """
        Informative compact string used to replace ToolMessage content after masking.
        Examples:
          [artifact: revenue_by_region | table | 1200×5 cols | cols: region, revenue, growth_pct, rank, period | step 2]
          [artifact: monthly_trend_chart | plot | Revenue by month 2024 | step 3]
          [artifact: total_revenue | value | step 1]
        """
        parts = [f"artifact: {self.name}", self.type]
        if self.row_count is not None and self.schema:
            parts.append(f"{self.row_count}×{len(self.schema)} cols")
            top_cols = ", ".join(list(self.schema.keys())[:5])
            parts.append(f"cols: {top_cols}")
        elif self.row_count is not None:
            parts.append(f"{self.row_count} rows")
        if self.summary:
            parts.append(self.summary)
        parts.append(f"step {self.step_index}")
        return "[" + " | ".join(parts) + "]"


@dataclass
class AnalysisWorkingMemory:
    """
    Per-query ephemeral state. Initialized at dispatch, flushed to SessionStore at finalize.
    All fields have explicit defaults — safe to construct with goal= only.
    """
    goal: str                               # current user request (set at dispatch)
    step_index: int = 0                     # incremented after each tool call
    artifact_handles: list[ArtifactHandle] = field(default_factory=list)
    sandbox_var_names: list[str] = field(default_factory=list)
    tool_call_count: int = 0

    # current_plan: set ONLY by planner_tool. Default [] means planner was not called.
    # Fully replaced (not merged) when planner_tool runs again.
    current_plan: list[str] = field(default_factory=list)

    # completed_actions: full audit trail. One entry per tool call, always, deterministic.
    # Format: "{tool_name} → {artifact_name_or_summary}"
    completed_actions: list[str] = field(default_factory=list)

    # last_tool_result_summary: compact summary of the most recent tool result.
    # Written deterministically from artifact metadata — not by the LLM.
    last_tool_result_summary: str = ""
