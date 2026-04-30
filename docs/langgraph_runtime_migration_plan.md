# LangGraph Runtime Migration Plan

## Target

The backend should have one production agent runtime: the new LangGraph runtime
under `backend/agent_graph/`.

The legacy `backend/agent/runner.py` may be used during development as a
behavior reference, but it should not remain as a permanent production fallback.

## Non-negotiable Compatibility

- Keep the current frontend contract unchanged.
- Keep `/sessions/{session_id}/query` and `/sessions/{session_id}/query/stream`
  response shapes compatible.
- Preserve artifact payloads, SSE event names, persisted chat records, reasoning
  traces, working memory flushes and tool access policies.
- Do not rewrite tools unless their runtime boundary requires it.

## Phases

### 1. Baseline

- Run and record the current contract tests.
- Add graph-runtime tests that protect the new state shape and node topology.
- Treat existing behavior as the migration contract.

### 2. Serializable Graph State

- Keep LangGraph state JSON-like.
- Store only ids, messages, tool-call descriptors, artifact refs and compact
  metadata in state.
- Keep live Python objects in runtime context until they can be rehydrated from
  services:
  - DataFrames
  - callbacks
  - tools
  - sandboxes
  - DB/runtime services
  - runner/service instances

### 3. Context And Routing

- Move request normalization, quick routing, data-source detection, sandbox
  lookup and tool construction from `AgentRunner` into graph nodes/services.
- Preserve chat, summary and analysis routing behavior.

### 4. Tool Loop Decomposition

Replace the manual `_direct_tool_loop` with graph nodes:

- `planner`
- `llm`
- `route_tool_calls`
- `tools`
- `update_working_memory`
- `mask_observations`
- `decide_continue`
- `finalize`
- `review`

The first pass should preserve behavior 1:1 before adding new LangGraph
features.

### 5. Streaming And Artifacts

- Preserve existing SSE event contract first.
- Keep artifact-aware tool execution compatible with `ToolCollector` behavior.
- Only after parity, consider switching selected live updates to native
  LangGraph stream events.

### 6. Replace Entrypoints

- Route query and stream endpoints to `backend.agent_graph`.
- Remove legacy runtime selection from production code.
- Keep old code only temporarily if tests still need it as a fixture/reference.

### 7. Remove Legacy Runtime

- Delete or archive obsolete `AgentRunner` execution-loop code.
- Keep reusable components that are not runtime-specific:
  - prompts
  - LLM wrappers
  - callbacks if still used
  - working-memory models if still useful
  - tool registry and tools

### 8. LangGraph Features

Only after the replacement is stable:

- checkpoint/resume
- interrupt/human-in-the-loop
- replan/review loops
- parallel branches via `Send`
- multi-agent supervisor if product requirements justify it
