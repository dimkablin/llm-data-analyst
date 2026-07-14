# Frontend Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** turn the React frontend into small typed feature slices instead of page-sized files that mix API calls, state machines, storage, and rendering.

**Architecture:** keep the current React 18 + Vite stack. Split by existing product boundaries: chat, workspace board, workspace sources, account/admin, observability, shared API/types. Keep old import paths as thin re-export adapters during migration, then delete adapters when callers move.

**Tech Stack:** React 18, Vite, TypeScript, Node `node:test`, existing UI primitives, existing backend REST/SSE contracts.

---

## Audit Findings

Ranked by cut size and risk reduction.

1. `shrink:` `frontend/src/app/components/workspace/DashboardPanel.tsx` is a 2887-line container with 42 `useState`, 16 `useEffect`, source management, DB/RAG/OpenProject API workflows, board layout, drag/resize, export, dialogs, and rendering. Split into source hooks/panels plus board hooks/panels.

2. `shrink:` `frontend/src/app/hooks/useChatAgent.ts` is a 1076-line hook that owns session cache, streaming state, SSE event folding, persistence recovery, and private backend reasoning reconstruction. Backend must return a public `AssistantBlock[]` contract; frontend should render blocks.

3. `shrink:` `frontend/src/app/lib/backend-api.ts` has 76 exports across auth, sessions, data upload, DB, RAG, OpenProject, query streaming, reports, Phoenix, admin MCP, and skills. Split by API domain and keep `lib/backend-api.ts` as a temporary re-export.

4. `shrink:` `frontend/src/app/lib/backend-types.ts` has 70 exported contracts in one file. Split by domain: chat, artifacts, sessions, sources, settings, observability, admin. Keep `lib/backend-types.ts` as a temporary re-export.

5. `shrink:` `frontend/src/app/pages/Workspace.tsx` is 631 lines of route guard, session loading, localStorage keys, board pinning, source state, upload, settings, and page composition. Move storage/session/board logic into hooks; leave the page as composition.

6. `shrink:` `frontend/src/app/components/workspace/ChatPanel.tsx` is 734 lines mixing composer, message list, message bubble, artifact chips, source badge, and quick actions. Extract render-only pieces.

7. `shrink:` `frontend/src/app/components/account/ToolAccessSection.tsx` is 1158 lines mixing user tool toggles, skill editor, MCP server CRUD, dialogs, forms, and API calls. Split admin hooks and dialogs.

8. `shrink:` `frontend/src/app/pages/Phoenix.tsx` is 1425 lines with data loading, chart mapping, trace table, trace detail dialog, resizing, and theme state. Split observability API state from render components.

9. `delete:` `frontend/src/app/components/ui` keeps 44 shadcn wrappers, but only 11 are imported outside the UI folder: `badge`, `button`, `checkbox`, `dialog`, `dropdown-menu`, `input`, `label`, `resizable`, `select`, `switch`, `textarea`. Delete unused wrappers and their unused dependencies after import migration.

10. `delete:` zero-use frontend deps from `frontend/package.json`: `@emotion/react`, `@emotion/styled`, `@mui/icons-material`, `@mui/material`, `@popperjs/core`, `canvas-confetti`, `date-fns`, `react-dnd`, `react-dnd-html5-backend`, `react-popper`, `react-responsive-masonry`, `react-slick`.

11. `delete:` dependencies used only by unused UI wrappers can go with those wrappers: `@radix-ui/react-accordion`, `@radix-ui/react-alert-dialog`, `@radix-ui/react-aspect-ratio`, `@radix-ui/react-avatar`, `@radix-ui/react-collapsible`, `@radix-ui/react-context-menu`, `@radix-ui/react-hover-card`, `@radix-ui/react-menubar`, `@radix-ui/react-navigation-menu`, `@radix-ui/react-popover`, `@radix-ui/react-progress`, `@radix-ui/react-radio-group`, `@radix-ui/react-scroll-area`, `@radix-ui/react-separator`, `@radix-ui/react-slider`, `@radix-ui/react-tabs`, `@radix-ui/react-toggle`, `@radix-ui/react-toggle-group`, `@radix-ui/react-tooltip`, `cmdk`, `embla-carousel-react`, `input-otp`, `react-day-picker`, `react-hook-form`, `sonner`, `vaul`.

12. `shrink:` frontend contract tests are split between root Node tests and Python source-string tests. Move frontend behavioral tests under `frontend/tests`, add npm scripts, and replace source-string assertions with tests of extracted pure functions/contracts.

## Target File Structure

```text
frontend/src/app/api/
  http.ts
  auth.ts
  sessions.ts
  sources.ts
  query.ts
  reports.ts
  observability.ts
  admin.ts
  index.ts

frontend/src/app/types/
  artifacts.ts
  chat.ts
  sessions.ts
  sources.ts
  settings.ts
  observability.ts
  admin.ts
  index.ts

frontend/src/app/features/chat/
  useChatAgent.ts
  useChatSessions.ts
  useQueryStream.ts
  assistant-blocks.ts
  ChatPanel.tsx
  ChatComposer.tsx
  MessageList.tsx
  MessageBubble.tsx
  ArtifactChips.tsx

frontend/src/app/features/workspace/
  Workspace.tsx
  useWorkspaceSession.ts
  useBoardPins.ts
  workspace-storage.ts
  board/
    BoardPanel.tsx
    BoardToolbar.tsx
    ArtifactBoard.tsx
    ArtifactBoardCard.tsx
    OpenProjectReportView.tsx
    useArtifactBoardLayout.ts
  sources/
    SourcesPanel.tsx
    useSourcesController.ts
    DbSourcesPanel.tsx
    CsvSourcesPanel.tsx
    RagSourcesPanel.tsx
    OpenProjectSourcePanel.tsx
    ConnectionDialog.tsx
    UploadDialog.tsx

frontend/src/app/features/account/
  ToolAccessSection.tsx
  useToolAccess.ts
  SkillEditorDialog.tsx
  McpServerDialog.tsx

frontend/src/app/features/observability/
  Phoenix.tsx
  usePhoenixDashboard.ts
  PhoenixOverviewCards.tsx
  PhoenixTraceTable.tsx
  PhoenixTraceDialog.tsx
```

Temporary adapters stay until the route imports are migrated:

```text
frontend/src/app/lib/backend-api.ts
frontend/src/app/lib/backend-types.ts
frontend/src/app/hooks/useChatAgent.ts
frontend/src/app/pages/Workspace.tsx
frontend/src/app/pages/Phoenix.tsx
frontend/src/app/components/workspace/ChatPanel.tsx
frontend/src/app/components/workspace/DashboardPanel.tsx
frontend/src/app/components/account/ToolAccessSection.tsx
```

Each adapter should only re-export from `features/*`, `api/*`, or `types/*`.

## Non-Goals

- Do not add Zustand, Redux, React Query, OpenAPI codegen, Vitest, Playwright, or a new UI kit in this refactor.
- Do not redesign screens while splitting code.
- Do not change REST paths, SSE event names, auth behavior, or persisted session semantics except adding the public `AssistantBlock[]` field.
- Do not keep unused shadcn wrappers "just in case".

## Implementation Tasks

### Task 1: Add a Minimal Frontend Safety Net

**Files:**
- Modify: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Move: `tests/*.test.ts` -> `frontend/tests/*.test.ts`
- Modify: moved test imports from `../frontend/src/app/...` to `../src/app/...`

- [ ] **Step 1: Add scripts**

Add these scripts to `frontend/package.json`:

```json
{
  "scripts": {
    "build": "vite build",
    "dev": "vite",
    "preview": "vite preview",
    "test:contracts": "node --test --experimental-strip-types --experimental-specifier-resolution=node tests/*.test.ts",
    "typecheck": "tsc --noEmit",
    "check": "npm run typecheck && npm run test:contracts && npm run build"
  }
}
```

Add `typescript` to `devDependencies` because this TypeScript app currently builds without a typecheck command:

```json
{
  "devDependencies": {
    "typescript": "^5.9.0"
  }
}
```

- [ ] **Step 2: Add `frontend/tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "useDefineForClassFields": true,
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "allowJs": false,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "module": "ESNext",
    "moduleResolution": "Bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx"
  },
  "include": ["src", "tests"]
}
```

- [ ] **Step 3: Move frontend Node tests under `frontend/tests`**

For each moved test, update imports like this:

```ts
import { streamQuery } from "../src/app/lib/backend-api.ts";
import type { ArtifactPayload } from "../src/app/lib/backend-types.ts";
```

For tests that read files, replace path strings like this:

```ts
new URL("../src/app/components/workspace/ChatPanel.tsx", import.meta.url)
```

- [ ] **Step 4: Run the safety net**

Run:

```bash
cd frontend
npm run test:contracts
npm run typecheck
npm run build
```

Expected: all commands pass before moving production code.

### Task 2: Replace Frontend Source-String Tests with Behavioral Tests

**Files:**
- Modify: `frontend/tests/parallelToolDisplayContract.test.ts`
- Modify: `frontend/tests/messagePlotArtifactLayout.test.ts`
- Modify: `frontend/tests/contextUsageIndicator.test.ts`
- Modify: `tests/test_frontend_auth_guards.py`
- Modify: `tests/test_frontend_knowledge_base_ui.py`
- Modify: `tests/test_frontend_mcp_settings_contract.py`
- Create focused pure helpers only when they are already needed by production code.

- [ ] **Step 1: Keep Python frontend tests only for route/file layout**

Delete Python tests that prove frontend behavior by `Path(...).read_text()` and regex over `.tsx`. Replace them with TypeScript behavioral tests against exported helpers or typed payload builders.

- [ ] **Step 2: Extract pure helpers before testing**

Examples of helpers that should be tested instead of source text:

```ts
export function formatParallelToolLabel(count: number): string {
  return count > 1 ? `Параллельно: ${count}` : "";
}
```

```ts
export function canUseKnowledgeBase(sessionId: string, activeSourceType?: string | null): boolean {
  return Boolean(sessionId) && activeSourceType !== "rag";
}
```

```ts
export function shouldRedirectUnauthenticated(hasToken: boolean): boolean {
  return !hasToken;
}
```

- [ ] **Step 3: Write Node behavioral tests**

Use `node:test` and `node:assert/strict`, matching existing frontend contract tests:

```ts
import assert from "node:assert/strict";
import test from "node:test";

import { formatParallelToolLabel } from "../src/app/features/chat/assistant-blocks.ts";

test("parallel tool label is shown only for multiple running tools", () => {
  assert.equal(formatParallelToolLabel(1), "");
  assert.equal(formatParallelToolLabel(3), "Параллельно: 3");
});
```

- [ ] **Step 4: Verify**

Run:

```bash
cd frontend
npm run test:contracts
```

Expected: tests assert behavior, not source fragments.

### Task 3: Split API Client and Backend Types

**Files:**
- Create: `frontend/src/app/api/http.ts`
- Create: `frontend/src/app/api/auth.ts`
- Create: `frontend/src/app/api/sessions.ts`
- Create: `frontend/src/app/api/sources.ts`
- Create: `frontend/src/app/api/query.ts`
- Create: `frontend/src/app/api/reports.ts`
- Create: `frontend/src/app/api/observability.ts`
- Create: `frontend/src/app/api/admin.ts`
- Create: `frontend/src/app/api/index.ts`
- Create: `frontend/src/app/types/*.ts`
- Modify: `frontend/src/app/lib/backend-api.ts`
- Modify: `frontend/src/app/lib/backend-types.ts`

- [ ] **Step 1: Move shared HTTP code**

`frontend/src/app/api/http.ts` owns `API_BASE`, token storage, `authFetch`, and `assertOk`:

```ts
export const TOKEN_STORAGE_KEY = "llm_data_analyst_access_token";
export const AUTH_CHANGED_EVENT = "llm-backend-auth-changed";

export async function authFetch(path: string, init?: RequestInit): Promise<Response> {
  const headers = new Headers(init?.headers ?? {});
  const token = getStoredToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  return fetch(`${API_BASE}${path}`, { ...init, headers });
}
```

- [ ] **Step 2: Move endpoint groups without changing signatures**

Keep function names stable. Example:

```ts
// frontend/src/app/api/sources.ts
export async function bindRagSource(sessionId: string): Promise<SessionSourceState> {
  const response = await authFetch(`/sessions/${sessionId}/source/rag`, { method: "POST" });
  await assertOk(response);
  return (await response.json()) as SessionSourceState;
}
```

- [ ] **Step 3: Keep compatibility re-exports**

```ts
// frontend/src/app/lib/backend-api.ts
export * from "../api/index.ts";
```

```ts
// frontend/src/app/lib/backend-types.ts
export * from "../types/index.ts";
```

- [ ] **Step 4: Verify**

Run:

```bash
cd frontend
npm run test:contracts
npm run build
```

Expected: existing imports still work through re-exports.

### Task 4: Add Public Assistant Blocks Contract

**Files:**
- Modify: `backend/api/models.py`
- Modify: `backend/api/services/query_execution.py`
- Modify: `backend/sessions/session_store.py`
- Modify: `backend/api/routes/sessions.py`
- Modify: `frontend/src/app/types/chat.ts`
- Modify: `frontend/src/app/features/chat/assistant-blocks.ts`
- Modify: `frontend/src/app/features/chat/useChatAgent.ts`
- Test: `tests/test_query_execution_service.py`
- Test: `frontend/tests/assistantBlocks.test.ts`

- [ ] **Step 1: Add typed public block models to backend API**

Add public DTOs in `backend/api/models.py`:

```python
class AssistantBlock(BaseModel):
    type: str
    id: str
    content: str | None = None
    kind: str | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    input_summary: str | None = None
    input_preview: str | None = None
    input_code: str | None = None
    status: str | None = None
    started_at: int | None = None
    result_summary: str | None = None
    output_preview: str | None = None
    artifact_keys: list[str] | None = None
    tool_use_id: str | None = None
```

Add optional blocks:

```python
class QueryResponse(BaseModel):
    ...
    blocks: list[AssistantBlock] | None = None
```

Session history stays dict-backed, but assistant messages may include `"blocks"`.

- [ ] **Step 2: Persist blocks with assistant messages**

Extend `SessionStore.add_chat_message`:

```python
def add_chat_message(..., blocks: list[dict[str, Any]] | None = None) -> None:
    ...
    if blocks:
        payload["blocks"] = blocks
```

- [ ] **Step 3: Build blocks on the backend**

In `QueryExecutionService`, build public blocks from collector tool calls and reasoning steps at the same place where `reasoning_steps` and `tools` are already built. Persist `blocks` with the assistant message and return them in `QueryResponse`.

- [ ] **Step 4: Remove private reconstruction from frontend**

Delete `buildBlocksFromHistory()` from `useChatAgent.ts`. Hydration becomes:

```ts
const blocks = item.blocks?.length ? item.blocks : undefined;
```

Keep only a small normalizer in `assistant-blocks.ts` for missing IDs or legacy empty values.

- [ ] **Step 5: Verify**

Run:

```bash
poetry run pytest tests/test_query_execution_service.py tests/test_thinking_parser.py -q
cd frontend
npm run test:contracts
npm run build
```

Expected: reload renders persisted `AssistantBlock[]` without frontend mirroring `_build_reasoning_steps`.

### Task 5: Split `useChatAgent`

**Files:**
- Create: `frontend/src/app/features/chat/useChatSessions.ts`
- Create: `frontend/src/app/features/chat/useQueryStream.ts`
- Create: `frontend/src/app/features/chat/assistant-blocks.ts`
- Create: `frontend/src/app/features/chat/useChatAgent.ts`
- Modify: `frontend/src/app/hooks/useChatAgent.ts`
- Test: `frontend/tests/streamQueryFinal.test.ts`
- Test: `frontend/tests/abortToolHistory.test.ts`
- Test: `frontend/tests/assistantBlocks.test.ts`

- [ ] **Step 1: Move session slot state**

`useChatSessions.ts` owns only session maps, hydration, reset, replace, and artifact append.

- [ ] **Step 2: Move streaming state**

`useQueryStream.ts` owns only one active stream, callbacks, collected blocks, stop handling, and final/interrupted state.

- [ ] **Step 3: Compose in `features/chat/useChatAgent.ts`**

The public hook still returns the same shape used by `ChatAgentContext`.

- [ ] **Step 4: Keep old import path**

```ts
// frontend/src/app/hooks/useChatAgent.ts
export { useChatAgent } from "../features/chat/useChatAgent.ts";
```

- [ ] **Step 5: Verify**

Run:

```bash
cd frontend
npm run test:contracts
npm run build
```

Expected: no route/component import needs to change in this task.

### Task 6: Split Workspace Session and Board State

**Files:**
- Create: `frontend/src/app/features/workspace/workspace-storage.ts`
- Create: `frontend/src/app/features/workspace/useWorkspaceSession.ts`
- Create: `frontend/src/app/features/workspace/useBoardPins.ts`
- Create: `frontend/src/app/features/workspace/Workspace.tsx`
- Modify: `frontend/src/app/pages/Workspace.tsx`
- Test: `frontend/tests/workspaceStorage.test.ts`
- Test: `frontend/tests/boardArtifactSelection.test.ts`

- [ ] **Step 1: Move localStorage helpers**

`workspace-storage.ts` owns keys and parse helpers:

```ts
export function loadIdList(storage: Storage, storageKey: string, sessionId: string): string[] {
  const raw = storage.getItem(`${storageKey}_${sessionId}`);
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter((item): item is string => typeof item === "string") : [];
  } catch {
    return [];
  }
}
```

- [ ] **Step 2: Move session loading**

`useWorkspaceSession.ts` owns `createSession`, `getSession`, `listSessions`, active session storage, model profile, and source metadata hydration.

- [ ] **Step 3: Move board pinning**

`useBoardPins.ts` owns auto-pinning, manual pins, hidden artifact IDs, and text note artifacts.

- [ ] **Step 4: Reduce route page**

`frontend/src/app/pages/Workspace.tsx` becomes:

```ts
export { Workspace } from "../features/workspace/Workspace.tsx";
```

- [ ] **Step 5: Verify**

Run:

```bash
cd frontend
npm run test:contracts
npm run build
```

Expected: page behavior is unchanged, but route file is only an adapter.

### Task 7: Split Dashboard Board Rendering

**Files:**
- Create: `frontend/src/app/features/workspace/board/useArtifactBoardLayout.ts`
- Create: `frontend/src/app/features/workspace/board/BoardPanel.tsx`
- Create: `frontend/src/app/features/workspace/board/BoardToolbar.tsx`
- Create: `frontend/src/app/features/workspace/board/ArtifactBoard.tsx`
- Create: `frontend/src/app/features/workspace/board/ArtifactBoardCard.tsx`
- Create: `frontend/src/app/features/workspace/board/OpenProjectReportView.tsx`
- Modify: `frontend/src/app/components/workspace/DashboardPanel.tsx`
- Test: `frontend/tests/artifactBoardLayout.test.ts`

- [ ] **Step 1: Move pure board layout functions**

Move `computeBoardLayouts`, `estimateAutoHeight`, width/height clamps, turn header helpers, and layout persistence into `useArtifactBoardLayout.ts`.

- [ ] **Step 2: Test board layout behavior**

Behavior to assert:

```ts
assert.equal(clampWidthUnitsValue(99), 12);
assert.equal(clampWidthUnitsValue(1), 4);
```

Also assert that two 6-column artifacts share a row and a third starts a new row.

- [ ] **Step 3: Move render components**

`BoardPanel.tsx` composes toolbar, OpenProject report, and normal artifact board. `ArtifactBoardCard.tsx` owns resize handles and remove/rename UI only.

- [ ] **Step 4: Keep adapter**

`DashboardPanel.tsx` imports `BoardPanel` and `SourcesPanel`. It should no longer contain board math or OpenProject report rendering.

- [ ] **Step 5: Verify**

Run:

```bash
cd frontend
npm run test:contracts
npm run build
```

Expected: `DashboardPanel.tsx` loses board layout, resize, and export internals.

### Task 8: Split Workspace Source Management

**Files:**
- Create: `frontend/src/app/features/workspace/sources/useSourcesController.ts`
- Create: `frontend/src/app/features/workspace/sources/SourcesPanel.tsx`
- Create: `frontend/src/app/features/workspace/sources/DbSourcesPanel.tsx`
- Create: `frontend/src/app/features/workspace/sources/CsvSourcesPanel.tsx`
- Create: `frontend/src/app/features/workspace/sources/RagSourcesPanel.tsx`
- Create: `frontend/src/app/features/workspace/sources/OpenProjectSourcePanel.tsx`
- Create: `frontend/src/app/features/workspace/sources/ConnectionDialog.tsx`
- Create: `frontend/src/app/features/workspace/sources/UploadDialog.tsx`
- Modify: `frontend/src/app/components/workspace/DashboardPanel.tsx`
- Test: `frontend/tests/sourcePayloads.test.ts`

- [ ] **Step 1: Move source API workflows**

`useSourcesController.ts` owns:

```text
loadConnections
loadRagDocuments
handleDeleteRagDocument
handleConfirmUpload
handleSubmitConnection
handleDeleteConnection
handleTestConnection
loadSchemasForConnection
handleSelectSchema
handleBindConnection
handleClearSource
handleBindCsvSource
handleBindRagSource
handleBindOpenProjectSource
handleLoadOpenProjectProjects
```

- [ ] **Step 2: Move payload builders**

Export and test pure helpers:

```ts
export function buildOpenProjectPayload(form: OpenProjectFormState): OpenProjectSyncRequest {
  return {
    base_url: form.baseUrl.trim() || undefined,
    api_key: form.apiKey.trim() || undefined,
    project: form.project.trim() || undefined,
    days: Number(form.days || 30),
  };
}
```

- [ ] **Step 3: Move panels**

Each source panel receives state/actions from `useSourcesController` and performs rendering only.

- [ ] **Step 4: Verify**

Run:

```bash
cd frontend
npm run test:contracts
npm run build
```

Expected: `DashboardPanel.tsx` no longer imports source API functions directly.

### Task 9: Split Chat Rendering

**Files:**
- Create: `frontend/src/app/features/chat/ChatPanel.tsx`
- Create: `frontend/src/app/features/chat/ChatComposer.tsx`
- Create: `frontend/src/app/features/chat/MessageList.tsx`
- Create: `frontend/src/app/features/chat/MessageBubble.tsx`
- Create: `frontend/src/app/features/chat/ArtifactChips.tsx`
- Modify: `frontend/src/app/components/workspace/ChatPanel.tsx`
- Test: `frontend/tests/artifactIconMapping.test.ts`
- Test: `frontend/tests/messagePlotArtifactLayout.test.ts`

- [ ] **Step 1: Move composer**

`ChatComposer.tsx` owns input text, quick suggestions, submit, stop, retry, and upload button UI.

- [ ] **Step 2: Move message rendering**

`MessageBubble.tsx` owns one message. `MessageList.tsx` owns list ordering and scroll anchoring.

- [ ] **Step 3: Move artifact chip rendering**

`ArtifactChips.tsx` owns plot/table/value/json/note chips and pin actions.

- [ ] **Step 4: Keep adapter**

```ts
// frontend/src/app/components/workspace/ChatPanel.tsx
export { ChatPanel } from "../../features/chat/ChatPanel.tsx";
```

- [ ] **Step 5: Verify**

Run:

```bash
cd frontend
npm run test:contracts
npm run build
```

Expected: chat UI still receives the same props from `Workspace`.

### Task 10: Split Account Tool Access

**Files:**
- Create: `frontend/src/app/features/account/useToolAccess.ts`
- Create: `frontend/src/app/features/account/ToolAccessSection.tsx`
- Create: `frontend/src/app/features/account/SkillEditorDialog.tsx`
- Create: `frontend/src/app/features/account/McpServerDialog.tsx`
- Modify: `frontend/src/app/components/account/ToolAccessSection.tsx`
- Test: `frontend/tests/toolAccessPayloads.test.ts`

- [ ] **Step 1: Move API state**

`useToolAccess.ts` owns loading, refresh, toggle skill, toggle MCP server, toggle tool, export archive, create/update/delete MCP server.

- [ ] **Step 2: Move dialogs**

`SkillEditorDialog.tsx` and `McpServerDialog.tsx` own form state and submit payload creation. Export pure payload builders for tests.

- [ ] **Step 3: Keep adapter**

```ts
export { ToolAccessSection } from "../../features/account/ToolAccessSection.tsx";
```

- [ ] **Step 4: Verify**

Run:

```bash
cd frontend
npm run test:contracts
npm run build
```

Expected: account page imports do not change.

### Task 11: Split Phoenix Observability Page

**Files:**
- Create: `frontend/src/app/features/observability/usePhoenixDashboard.ts`
- Create: `frontend/src/app/features/observability/Phoenix.tsx`
- Create: `frontend/src/app/features/observability/PhoenixOverviewCards.tsx`
- Create: `frontend/src/app/features/observability/PhoenixTraceTable.tsx`
- Create: `frontend/src/app/features/observability/PhoenixTraceDialog.tsx`
- Modify: `frontend/src/app/pages/Phoenix.tsx`
- Test: `frontend/tests/phoenixUrl.test.ts`
- Test: `frontend/tests/phoenixTraceHistory.test.ts`

- [ ] **Step 1: Move data loading**

`usePhoenixDashboard.ts` owns overview, traces, trace detail, selected project, pagination, and loading/error state.

- [ ] **Step 2: Move render components**

`PhoenixTraceDialog.tsx` owns resize state. `PhoenixTraceTable.tsx` owns row rendering and selection. The page composes.

- [ ] **Step 3: Keep adapter**

```ts
export { Phoenix } from "../features/observability/Phoenix.tsx";
```

- [ ] **Step 4: Verify**

Run:

```bash
cd frontend
npm run test:contracts
npm run build
```

Expected: Phoenix route behavior stays unchanged.

### Task 12: Delete Unused UI Wrappers and Dependencies

**Files:**
- Delete unused files in `frontend/src/app/components/ui/`
- Modify: `frontend/package.json`
- Modify: `frontend/package-lock.json`

- [ ] **Step 1: Confirm used UI wrappers**

Run:

```bash
rg "../ui/|components/ui/" frontend/src/app --glob "!**/components/ui/**"
```

Expected used list:

```text
badge
button
checkbox
dialog
dropdown-menu
input
label
resizable
select
switch
textarea
```

- [ ] **Step 2: Delete unused UI wrappers**

Delete every other `frontend/src/app/components/ui/*.tsx`.

- [ ] **Step 3: Remove unused dependencies**

Run:

```bash
cd frontend
npm uninstall @emotion/react @emotion/styled @mui/icons-material @mui/material @popperjs/core canvas-confetti date-fns react-dnd react-dnd-html5-backend react-popper react-responsive-masonry react-slick
npm uninstall @radix-ui/react-accordion @radix-ui/react-alert-dialog @radix-ui/react-aspect-ratio @radix-ui/react-avatar @radix-ui/react-collapsible @radix-ui/react-context-menu @radix-ui/react-hover-card @radix-ui/react-menubar @radix-ui/react-navigation-menu @radix-ui/react-popover @radix-ui/react-progress @radix-ui/react-radio-group @radix-ui/react-scroll-area @radix-ui/react-separator @radix-ui/react-slider @radix-ui/react-tabs @radix-ui/react-toggle @radix-ui/react-toggle-group @radix-ui/react-tooltip cmdk embla-carousel-react input-otp react-day-picker react-hook-form sonner vaul
```

- [ ] **Step 4: Verify**

Run:

```bash
cd frontend
npm run check
```

Expected: build passes and `package-lock.json` shrinks.

### Task 13: Remove Temporary Adapters

**Files:**
- Modify imports in `frontend/src/app/**`
- Delete temporary adapter files only after no caller imports them.

- [ ] **Step 1: Move route imports to feature paths**

Update route imports to `features/*` and `api/*`.

- [ ] **Step 2: Confirm adapters are unused**

Run:

```bash
rg "lib/backend-api|lib/backend-types|hooks/useChatAgent|components/workspace/ChatPanel|components/workspace/DashboardPanel|components/account/ToolAccessSection" frontend/src/app
```

Expected: no matches except deliberate compatibility comments.

- [ ] **Step 3: Delete adapters**

Delete:

```text
frontend/src/app/lib/backend-api.ts
frontend/src/app/lib/backend-types.ts
frontend/src/app/hooks/useChatAgent.ts
frontend/src/app/components/workspace/ChatPanel.tsx
frontend/src/app/components/workspace/DashboardPanel.tsx
frontend/src/app/components/account/ToolAccessSection.tsx
```

- [ ] **Step 4: Verify**

Run:

```bash
cd frontend
npm run check
poetry run pytest tests/test_query_execution_service.py tests/api/test_session_think_filter.py -q
```

Expected: frontend check passes and backend block/session tests pass.

## Final Acceptance Criteria

- `DashboardPanel.tsx`, `useChatAgent.ts`, `backend-api.ts`, `backend-types.ts`, `Workspace.tsx`, `ChatPanel.tsx`, `ToolAccessSection.tsx`, and `Phoenix.tsx` are either focused files under 300-500 lines or deleted adapters.
- UI components do not contain API workflows.
- API modules do not contain render logic.
- Frontend renders backend `AssistantBlock[]`; it does not mirror private backend reasoning order.
- Frontend tests live under `frontend/tests` and assert behavior of exported functions/contracts.
- No source-string tests remain for frontend runtime behavior.
- Unused UI wrappers and zero-use dependencies are removed.
- `cd frontend && npm run check` passes.
- Relevant backend tests for session/query block contracts pass.

## Suggested Commit Order

1. `test(frontend): add frontend check scripts`
2. `refactor(frontend): split api client and types`
3. `feat(api): expose assistant blocks contract`
4. `refactor(frontend): split chat agent runtime hook`
5. `refactor(frontend): split workspace session and board state`
6. `refactor(frontend): split source management panels`
7. `refactor(frontend): split chat rendering`
8. `refactor(frontend): split account tool access`
9. `refactor(frontend): split phoenix observability`
10. `chore(frontend): remove unused ui wrappers and deps`
11. `chore(frontend): remove temporary adapters`
