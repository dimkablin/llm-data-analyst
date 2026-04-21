# Agent Context & State — B3 Design Spec
_Date: 2026-04-16_

## Problem

The agent loses context between iterations at two levels:

1. **Within a single query** — tool results are passed as raw text inline in the message thread.
   The model has no structured view of what artifacts exist, what variables are in sandbox scope,
   or what steps have been completed.

2. **Between user turns** — `SessionMemory` is an unstructured string. Artifact data is serialized
   inline into chat history as table previews (up to 20 rows of data per artifact). Old tool results
   are never masked or compacted.

The root cause is not prompting — it is the absence of a proper **state contract** between iterations
and the absence of **artifact references** as a first-class concept.

---

## Design Principles

1. `ArtifactStore` is the canonical source of artifact data. `ArtifactHandle` is a read-only
   memory projection created once at artifact creation time.
2. State assembly changes (history masking, context block generation) are consequences of better
   structured state — not prompt engineering.
3. The design is an evolutionary intermediate step toward a full `AgentWorkingState` redesign (variant C).
   Schema fields and write paths are chosen to allow natural extension without breaking changes.
4. No changes to: graph routing logic, streaming callbacks, tool implementations, CSV/DB runtime
   services, or the tool loop itself.

---

## Layer 1: `AnalysisWorkingMemory` (per-query, ephemeral)

Lives in `AgentGraphState`. Created at dispatch, discarded after finalize flushes to session store.

```python
# backend/agent/working_memory.py

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ArtifactHandle:
    """
    Lightweight read-only projection of an artifact's metadata.

    Canonical source: ArtifactStore (lookup by id to get actual data).
    This object is created once at artifact creation time and is immutable.
    """
    id: str                           # matches ArtifactStore.artifact.id
    name: str                         # artifact_name from tool_result
    type: str                         # "table" | "plot" | "value" | "error"
    tool_name: str                    # "sql_tool" | "pandas_tool" | "plotly_tool" | ...
    step_index: int                   # AgentGraphState.step_index when created
    schema: dict[str, str] | None     # col → dtype, tables only
    row_count: int | None             # tables only
    summary: str | None               # deterministic one-liner, e.g. "Revenue by region, 1200×5"

    @property
    def masked_ref(self) -> str:
        """
        Informative compact string used to replace ToolMessage content after masking.
        Gives the model enough metadata to decide whether to re-query.
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
    # Immutable within a query after first set. If planner_tool is called again (re-plan),
    # fully replaces the list — no merge.
    current_plan: list[str] = field(default_factory=list)

    # completed_actions: full audit trail. One entry per tool call, always, deterministic.
    # Format: "{tool_name} → {artifact_name_or_summary}"
    # Example: "sql_tool → revenue_by_region (1200 rows)"
    # Ephemeral — lives only in working memory, not persisted directly.
    completed_actions: list[str] = field(default_factory=list)

    # last_tool_result_summary: compact summary of the most recent tool result.
    # Written deterministically from artifact metadata — not by the LLM.
    # Example: "table: monthly_revenue, 36×4, jan–dec 2024"
    last_tool_result_summary: str = ""
```

### Write rules

| Field | Writer | Trigger |
|---|---|---|
| `goal` | dispatch node | once, at initialization |
| `step_index` | tool execution loop | after each tool call |
| `artifact_handles` | tool execution loop | when tool result contains an artifact |
| `sandbox_var_names` | tool execution loop | after each sandbox execution (pandas/plotly/value) |
| `tool_call_count` | tool execution loop | after each tool call |
| `current_plan` | planner_tool callback | only when planner_tool is invoked; full replace on re-plan |
| `completed_actions` | tool execution loop | after each tool call |
| `last_tool_result_summary` | tool execution loop | after each tool call, overwritten |

---

## Layer 2: `StructuredSessionMemory` (cross-turn, persisted)

Replaces `SessionMemory` (which was `notes: str`). Stored in `state.json` via `SessionStore`.

```python
# backend/sessions/session_memory.py

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class SessionArtifactRef:
    """
    Persisted lightweight reference to an artifact from a completed turn.
    Subset of ArtifactHandle — no runtime-only fields.
    """
    id: str
    name: str
    type: str
    turn_index: int             # which user turn (0-indexed) produced this artifact
    schema: dict[str, str] | None
    row_count: int | None
    summary: str | None


@dataclass
class StructuredSessionMemory:
    """
    Cross-turn persistent state. Replaces SessionMemory.notes: str.
    Backward compatible: notes field preserved, memory_tool continues to write here.
    """
    notes: str = ""                                      # free text, backward compat
    artifact_index: list[SessionArtifactRef] = field(default_factory=list)  # all artifacts, all turns

    # key_findings: significant subset of completed_actions after deterministic filter.
    # Two write paths:
    #   Path A — memory_tool (explicit, model-driven)
    #   Path B — finalize node (deterministic filter of completed_actions)
    # Cap: last 30 entries, oldest evicted.
    key_findings: list[str] = field(default_factory=list)

    turn_count: int = 0                                  # incremented at each finalize

    def is_empty(self) -> bool:
        return (
            not self.notes.strip()
            and not self.artifact_index
            and not self.key_findings
        )

    def build_block(self) -> str:
        """
        Compact context block for system prompt injection.
        Returns artifact refs and findings — NOT inline data.
        """
        parts: list[str] = []

        if self.notes.strip():
            parts.append(f"## Session notes\n{self.notes.strip()}")

        if self.key_findings:
            findings = "\n".join(f"- {f}" for f in self.key_findings[-10:])
            parts.append(f"## Key findings from this session\n{findings}")

        if self.artifact_index:
            lines: list[str] = []
            for a in self.artifact_index[-20:]:
                meta = f"{a.type}"
                if a.row_count is not None and a.schema:
                    meta += f", {a.row_count}×{len(a.schema)}"
                if a.summary:
                    meta += f", {a.summary}"
                lines.append(f"- {a.name} ({meta})")
            parts.append(f"## Artifacts from this session\n" + "\n".join(lines))

        return "\n\n".join(parts)
```

---

## Canonical Source Contract

```
ArtifactStore.artifacts     canonical source of truth
                            holds full data: DataFrame, Figure, scalar, etc.
                            mutable during a session

ArtifactHandle              read-only memory projection
                            created once at artifact creation time
                            id always matches an ArtifactStore.artifact.id
                            never written back to ArtifactStore

SessionArtifactRef          persisted subset of ArtifactHandle
                            upcast from handle at finalize; strip runtime fields
                            stored in state.json
```

**Stale handle policy:**

A handle is stale when `handle.id` is not found in `ArtifactStore`.

```python
def resolve_artifact(handle: ArtifactHandle, store: ArtifactStore) -> Artifact | None:
    """Returns None if handle is stale. ArtifactStore is always authoritative."""
    result = next((a for a in store.artifacts if a.id == handle.id), None)
    if result is None:
        logger.warning(
            "stale ArtifactHandle: id=%s name=%s step=%d tool=%s",
            handle.id, handle.name, handle.step_index, handle.tool_name,
        )
    return result
```

Consequences of stale handle:
1. The corresponding `ToolMessage` is **not masked** — original content is preserved as-is.
2. The `SessionArtifactRef` for this handle is **skipped** during prompt assembly (not listed).
3. A `WARNING` is logged. No exception, no silent failure.

---

## Observation Masking Policy

Masking replaces `ToolMessage.content` with `handle.masked_ref` in `_build_messages()`.

```python
OBSERVATION_MASK_KEEP_LAST_N = 3     # keep last N tool results at full content
OBSERVATION_MASK_MIN_STEPS = 4       # do not mask at all if step_index < 4
OBSERVATION_MASK_MIN_TOOLS = 3       # do not mask at all if tool_call_count < 3
```

**Policy by artifact type:**

| Type | Masking behaviour |
|---|---|
| `error` | Never masked — needed for recovery and retry logic |
| `value` | Mask immediately once outside KEEP_LAST_N (summary in masked_ref is sufficient) |
| `table` | Mask after KEEP_LAST_N |
| `plot` | Mask immediately once outside KEEP_LAST_N (data is visual, summary sufficient) |
| No artifact (tool with text only) | Mask after KEEP_LAST_N |

**Required test scenarios:**

| Scenario | Expected behaviour |
|---|---|
| 1–3 tool calls | No masking regardless of type |
| 4+ tool calls, all success | Mask all outside last 3 |
| Last tool call is error | Error ToolMessage not masked |
| Error in mid-sequence, then success | Error stays full; older successes masked normally |
| CSV multi-step (5+ steps) | Masked refs include CSV table schema |
| DB multi-step (5+ steps) | Masked refs include DB query result schema |
| planner_tool → 5-step plan, all complete | Plan visible in working_memory; masking does not lose step continuity |
| planner_tool → re-plan mid-run | current_plan fully replaced; completed_actions preserves original steps |

---

## `key_findings` Write Paths

### Path A — `memory_tool` (model-driven, unchanged)
The model calls `memory_tool("some finding")` explicitly.
This appends to `StructuredSessionMemory.notes` as before (backward compat).
Separately, significant `memory_tool` calls may also be appended to `key_findings`
by the tool implementation (opt-in, simple string copy).

### Path B — finalize node (deterministic, always runs)

```python
def _extract_findings_from_actions(
    actions: list[str],
    turn_index: int,
) -> list[str]:
    """
    Deterministic filter: retain actions that look like quantitative or artifact-producing steps.
    No LLM call. Simple heuristics:
    - Contains a number (digit sequence)
    - Contains an artifact name (→ symbol present)
    - Is not a pure infrastructure call (database_tool, get_tool_instructions)
    """
    SKIP_TOOLS = {"database_tool", "get_tool_instructions", "planner_tool"}
    findings = []
    for action in actions:
        tool = action.split("→")[0].strip().rstrip()
        if any(skip in tool for skip in SKIP_TOOLS):
            continue
        findings.append(f"[turn {turn_index}] {action}")
    return findings
```

Cap enforcement: `key_findings = (existing + new)[-30:]`

---

## Integration Points in `runner.py`

### `AgentGraphState` — new field

```python
class AgentGraphState(TypedDict, total=False):
    # ... existing fields ...
    working_memory: AnalysisWorkingMemory | None   # ADD — optional, backward compat
```

### `_build_tool_message_text()` — returns handle

**Current:** `(result: object) -> str`
**New:** `(result: object) -> tuple[str, ArtifactHandle | None]`

The handle is extracted from the artifact dict already present in `result`.
Handle creation is deterministic — no LLM call.

All callers must be updated to unpack the tuple.
If callers only need the text (e.g. fallback paths), `text, _ = _build_tool_message_text(result)`.

### Tool execution loop

After each tool call:
```python
text, handle = _build_tool_message_text(result)
if handle:
    working_memory.artifact_handles.append(handle)
    working_memory.last_tool_result_summary = handle.summary or handle.masked_ref
else:
    working_memory.last_tool_result_summary = text[:120]
action_line = f"{tool_name} → {handle.name if handle else text[:60]}"
working_memory.completed_actions.append(action_line)
working_memory.step_index += 1
working_memory.tool_call_count += 1
```

### `_build_messages()` — observation masking

After assembling recent messages, apply masking pass:
```python
# identify ToolMessages eligible for masking
tool_msg_handles: dict[int, ArtifactHandle] = {}  # message index → handle
# ... populate from working_memory.artifact_handles matched by step_index ...

for i, msg in enumerate(messages):
    if not isinstance(msg, ToolMessage):
        continue
    handle = tool_msg_handles.get(i)
    if handle is None:
        continue
    # check stale
    if resolve_artifact(handle, artifact_store) is None:
        continue  # stale: keep original content
    # check masking eligibility
    steps_ago = working_memory.step_index - handle.step_index
    if steps_ago < OBSERVATION_MASK_KEEP_LAST_N:
        continue
    if handle.type == "error":
        continue
    messages[i] = ToolMessage(content=handle.masked_ref, tool_call_id=msg.tool_call_id)
```

Replace `_history_artifact_summary()` with `structured_memory.build_block()` —
compact refs from `SessionArtifactRef` list instead of inline table previews.

### finalize node — flush to session store

```python
# flush artifact handles → SessionArtifactRef in structured_memory
for handle in working_memory.artifact_handles:
    if resolve_artifact(handle, artifact_store) is None:
        continue  # skip stale
    ref = SessionArtifactRef(
        id=handle.id,
        name=handle.name,
        type=handle.type,
        turn_index=structured_memory.turn_count,
        schema=handle.schema,
        row_count=handle.row_count,
        summary=handle.summary,
    )
    structured_memory.artifact_index.append(ref)

# extract findings from completed_actions (Path B)
new_findings = _extract_findings_from_actions(
    working_memory.completed_actions,
    turn_index=structured_memory.turn_count,
)
structured_memory.key_findings = (structured_memory.key_findings + new_findings)[-30:]
structured_memory.turn_count += 1

session_store.set_structured_memory(session_id, structured_memory)
```

---

## `SessionStore` changes

`session_memory: str` → `structured_memory: StructuredSessionMemory`

Migration (backward compat for existing sessions):
```python
# in _load_state()
old_notes = raw.get("session_memory", "")
structured_raw = raw.get("structured_memory")
if structured_raw:
    structured_memory = StructuredSessionMemory(**structured_raw)
else:
    # migrate from old format
    structured_memory = StructuredSessionMemory(notes=old_notes)
```

New `SessionStore` methods:
- `get_structured_memory(session_id) -> StructuredSessionMemory`
- `set_structured_memory(session_id, memory: StructuredSessionMemory) -> None`

Remove `set_session_memory()` and `append_session_memory()` after migration,
or keep as deprecated shims writing to `structured_memory.notes`.

---

## Staged Implementation Plan

### Stage 1 — New dataclasses, zero integration (risk: none)
- Create `backend/agent/working_memory.py` with `ArtifactHandle`, `AnalysisWorkingMemory`
- Update `backend/sessions/session_memory.py`: `SessionMemory` → `StructuredSessionMemory`, add `SessionArtifactRef`
- Add unit tests for `masked_ref` generation and `build_block()` output
- No other files touched

### Stage 2 — SessionStore migration (risk: very low)
- Add `structured_memory` field to `SessionState`
- Add migration in `_load_state()` for old sessions
- Add `get_structured_memory()` / `set_structured_memory()` methods
- Keep `session_memory: str` in serialization as deprecated fallback
- Test: existing session loads correctly with both old and new format

### Stage 3 — Handle creation in tool execution (risk: medium)
- Change `_build_tool_message_text()` to return `(str, ArtifactHandle | None)`
- Update all callers (unpack tuple)
- Add handle accumulation in the tool execution loop inside `run_query()`
- Add `AnalysisWorkingMemory` to `AgentGraphState` as optional field
- Test: `sql_tool`, `pandas_tool`, `plotly_tool`, `value_tool` all produce correct handles
- Test: CSV scenario, DB scenario, tool returning error

### Stage 4 — Finalize flush (risk: low)
- Add flush logic in finalize node: handles → `SessionArtifactRef` in structured_memory
- Add `_extract_findings_from_actions()` and Path B key_findings write
- Test: findings are accumulated correctly across 3+ turns; stale handles are skipped with warning

### Stage 5 — Observation masking (risk: medium, feature-flagged)
- Implement masking pass in `_build_messages()`
- Add `OBSERVATION_MASK_ENABLED` config flag (default `False` at merge, enable after validation)
- Replace `_history_artifact_summary()` with `structured_memory.build_block()`
- Run full test matrix from the masking policy table above
- Enable flag, validate in dev, then promote to default `True`

### Stage 6 — Cleanup (risk: low)
- Remove deprecated `session_memory: str` from serialization after one release cycle
- Remove `set_session_memory()` / `append_session_memory()` from `SessionStore`
- Add deprecation warnings in Stage 2–3 to ease the transition

---

## What Does Not Change

- Graph routing: `dispatch → agent → tools → finalize` stays identical
- Streaming callbacks (`AgentProgressCollector`, `LLMTextCollector`, `PhaseCollector`)
- Tool implementations (all tools in `backend/tools/impl/`)
- CSV runtime (`csv_session_runtime.py`) and DB runtime (`db_runtime_service.py`)
- `ArtifactStore` public API
- `memory_tool` (Path A continues to work unchanged)
- `planner_tool` output format (only the consumption of its output into `current_plan` changes)

---

## Evolution Path to Variant C

This design is explicitly a stepping stone. When ready to move to full variant C:

- `AnalysisWorkingMemory` grows into a full `AgentWorkingState` with richer plan tracking
- `StructuredSessionMemory` grows into a full `SessionKnowledgeBase` with vector retrieval
- `ArtifactHandle` → `ArtifactDescriptor` with richer lineage (depends_on, producing_query)
- `AgentGraphState` gets a proper checkpointing layer (LangGraph Checkpointer)

None of these require breaking changes to the schema introduced here —
they are additive extensions of the same contracts.
