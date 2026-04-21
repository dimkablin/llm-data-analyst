# Think Visibility & Streaming Toggle — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users control LLM streaming and per-kind thinking block visibility from the Settings panel, with preferences persisted to the DB and applied to both live streaming and history rendering.

**Architecture:** Add 5 boolean fields to `user_settings` SQLite table; propagate them through backend models and `_effective_runtime_settings`; filter `reasoning_steps` in the frontend before rendering using a dedicated `filterReasoningSteps()` utility; add a new SettingsPanel section.

**Tech Stack:** Python / SQLite / FastAPI / Pydantic (backend); React / TypeScript (frontend)

---

## File Map

| File | Change |
|------|--------|
| `backend/auth/auth_db.py` | +5 columns in migration, +5 fields in dataclass, update parse/get/update methods |
| `backend/api/models.py` | +5 fields in `UserSettingsResponse` + `UserSettingsUpdateRequest` |
| `backend/api/routes/query.py` | Add `llm_streaming` override in `_effective_runtime_settings()` |
| `frontend/src/app/lib/backend-types.ts` | +5 fields in `UserSettings` type |
| `frontend/src/app/lib/think-filter.ts` | **New file** — `filterReasoningSteps()` utility |
| `frontend/src/app/components/workspace/ChatPanel.tsx` | Import + apply `filterReasoningSteps()` to history rendering |
| `frontend/src/app/components/workspace/SettingsPanel.tsx` | New "Стриминг и thinking" section |
| `tests/test_user_think_settings.py` | **New file** — DB persistence tests |
| `tests/test_think_filter_util.ts` | **New file** — frontend filter unit tests (vitest) |

---

## Task 1: DB Migration — 5 New Columns

**Files:**
- Modify: `backend/auth/auth_db.py`
- Test: `tests/test_user_think_settings.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_user_think_settings.py`:

```python
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.auth import AuthDB


class UserThinkSettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = str(Path(self.tmpdir) / "app.db")
        self.auth_db = AuthDB(self.db_path, token_ttl_days=30)
        self.user = self.auth_db.create_user("alice_think", "secret", is_admin=False)

    def tearDown(self) -> None:
        del self.auth_db
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_new_settings_have_correct_defaults(self) -> None:
        settings = self.auth_db.get_user_settings(self.user.id)
        self.assertTrue(settings.llm_streaming)
        self.assertTrue(settings.show_thinking)
        self.assertTrue(settings.show_think_planning)
        self.assertTrue(settings.show_think_tool)
        self.assertTrue(settings.show_think_final)

    def test_update_and_persist_think_settings(self) -> None:
        updated = self.auth_db.update_user_settings(
            self.user.id,
            llm_streaming=False,
            show_thinking=True,
            show_think_planning=True,
            show_think_tool=False,
            show_think_final=False,
        )
        self.assertFalse(updated.llm_streaming)
        self.assertTrue(updated.show_thinking)
        self.assertTrue(updated.show_think_planning)
        self.assertFalse(updated.show_think_tool)
        self.assertFalse(updated.show_think_final)

        # Reload from DB to verify persistence
        reloaded = self.auth_db.get_user_settings(self.user.id)
        self.assertFalse(reloaded.llm_streaming)
        self.assertFalse(reloaded.show_think_tool)

    def test_partial_update_preserves_other_fields(self) -> None:
        self.auth_db.update_user_settings(self.user.id, show_think_final=False)
        settings = self.auth_db.get_user_settings(self.user.id)
        self.assertTrue(settings.llm_streaming)      # unchanged
        self.assertTrue(settings.show_thinking)       # unchanged
        self.assertFalse(settings.show_think_final)   # changed
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd C:\Users\dimka\Documents\PROJECTS\llm-data-analyst\llm-data-analyst-dev
python -m pytest tests/test_user_think_settings.py -v
```

Expected: `AttributeError: 'UserSettings' object has no attribute 'llm_streaming'`

- [ ] **Step 3: Add 5 fields to `UserSettings` dataclass**

In `backend/auth/auth_db.py`, modify the `UserSettings` dataclass (after line 46, before the closing):

```python
@dataclass(frozen=True)
class UserSettings:
    theme: str
    default_include_reasoning: bool
    default_answer_style: str
    analysis_depth: str
    llm_temperature_chat: float
    llm_temperature_tool: float
    llm_max_tokens_default: int
    llm_max_tokens_reasoning: int
    backend_query_timeout_sec: int
    agent_max_steps: int
    agent_step_timeout_sec: int
    agent_inner_recursion_limit: int
    agent_react_enabled: bool = False
    ui_scale: int = 100
    llm_streaming: bool = True
    show_thinking: bool = True
    show_think_planning: bool = True
    show_think_tool: bool = True
    show_think_final: bool = True
```

- [ ] **Step 4: Add migration for 5 new columns in `_ensure_user_settings_columns`**

At the end of the `_ensure_user_settings_columns` method in `backend/auth/auth_db.py`, append after the last `if "agent_react_enabled" not in existing_columns:` block:

```python
        if "llm_streaming" not in existing_columns:
            conn.execute(
                """
                ALTER TABLE user_settings
                ADD COLUMN llm_streaming INTEGER NOT NULL DEFAULT 1
                """
            )
        if "show_thinking" not in existing_columns:
            conn.execute(
                """
                ALTER TABLE user_settings
                ADD COLUMN show_thinking INTEGER NOT NULL DEFAULT 1
                """
            )
        if "show_think_planning" not in existing_columns:
            conn.execute(
                """
                ALTER TABLE user_settings
                ADD COLUMN show_think_planning INTEGER NOT NULL DEFAULT 1
                """
            )
        if "show_think_tool" not in existing_columns:
            conn.execute(
                """
                ALTER TABLE user_settings
                ADD COLUMN show_think_tool INTEGER NOT NULL DEFAULT 1
                """
            )
        if "show_think_final" not in existing_columns:
            conn.execute(
                """
                ALTER TABLE user_settings
                ADD COLUMN show_think_final INTEGER NOT NULL DEFAULT 1
                """
            )
```

- [ ] **Step 5: Update `_parse_user_settings` to read the 5 new columns**

In `_parse_user_settings` (around line 664), add to the `return UserSettings(...)` call the following fields after `ui_scale=...`:

```python
            llm_streaming=(
                bool(row["llm_streaming"]) if "llm_streaming" in row.keys() else True
            ),
            show_thinking=(
                bool(row["show_thinking"]) if "show_thinking" in row.keys() else True
            ),
            show_think_planning=(
                bool(row["show_think_planning"]) if "show_think_planning" in row.keys() else True
            ),
            show_think_tool=(
                bool(row["show_think_tool"]) if "show_think_tool" in row.keys() else True
            ),
            show_think_final=(
                bool(row["show_think_final"]) if "show_think_final" in row.keys() else True
            ),
```

- [ ] **Step 6: Update `get_user_settings` SELECT query**

In `get_user_settings` (around line 696), extend the SELECT:

```python
            row = conn.execute(
                """
                SELECT theme, default_include_reasoning, default_answer_style
                    , analysis_depth
                    , llm_temperature_chat, llm_temperature_tool
                    , llm_max_tokens_default, llm_max_tokens_reasoning
                    , backend_query_timeout_sec, agent_max_steps
                    , agent_step_timeout_sec, agent_inner_recursion_limit
                    , agent_react_enabled
                    , ui_scale
                    , llm_streaming, show_thinking
                    , show_think_planning, show_think_tool, show_think_final
                FROM user_settings
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
```

Also update the fallback `UserSettings(...)` returned when `row is None` to add the 5 new fields with defaults:

```python
        if row is None:
            return UserSettings(
                theme="dark",
                default_include_reasoning=True,
                default_answer_style="detailed",
                analysis_depth="light",
                llm_temperature_chat=0.7,
                llm_temperature_tool=0.5,
                llm_max_tokens_default=2048,
                llm_max_tokens_reasoning=4096,
                backend_query_timeout_sec=180,
                agent_max_steps=20,
                agent_step_timeout_sec=45,
                agent_inner_recursion_limit=6,
                agent_react_enabled=False,
                ui_scale=100,
                llm_streaming=True,
                show_thinking=True,
                show_think_planning=True,
                show_think_tool=True,
                show_think_final=True,
            )
```

- [ ] **Step 7: Update `update_user_settings` — signature, SELECT, logic, SQL UPDATE, return**

**Signature** — add 5 keyword-only params after `ui_scale`:

```python
    def update_user_settings(
        self,
        user_id: int,
        *,
        theme: str | None = None,
        default_include_reasoning: bool | None = None,
        default_answer_style: str | None = None,
        analysis_depth: str | None = None,
        llm_temperature_chat: float | None = None,
        llm_temperature_tool: float | None = None,
        llm_max_tokens_default: int | None = None,
        llm_max_tokens_reasoning: int | None = None,
        backend_query_timeout_sec: int | None = None,
        agent_max_steps: int | None = None,
        agent_step_timeout_sec: int | None = None,
        agent_inner_recursion_limit: int | None = None,
        agent_react_enabled: bool | None = None,
        ui_scale: int | None = None,
        llm_streaming: bool | None = None,
        show_thinking: bool | None = None,
        show_think_planning: bool | None = None,
        show_think_tool: bool | None = None,
        show_think_final: bool | None = None,
    ) -> UserSettings:
```

**SELECT inside `update_user_settings`** — extend to include the 5 new columns (same as `get_user_settings`):

```python
            current_row = conn.execute(
                """
                SELECT theme, default_include_reasoning, default_answer_style
                    , analysis_depth
                    , llm_temperature_chat, llm_temperature_tool
                    , llm_max_tokens_default, llm_max_tokens_reasoning
                    , backend_query_timeout_sec, agent_max_steps
                    , agent_step_timeout_sec, agent_inner_recursion_limit
                    , ui_scale
                    , llm_streaming, show_thinking
                    , show_think_planning, show_think_tool, show_think_final
                FROM user_settings
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
```

**Logic** — add 5 `next_*` variables after `next_ui_scale`:

```python
            next_llm_streaming = (
                bool(llm_streaming) if llm_streaming is not None else current.llm_streaming
            )
            next_show_thinking = (
                bool(show_thinking) if show_thinking is not None else current.show_thinking
            )
            next_show_think_planning = (
                bool(show_think_planning)
                if show_think_planning is not None
                else current.show_think_planning
            )
            next_show_think_tool = (
                bool(show_think_tool)
                if show_think_tool is not None
                else current.show_think_tool
            )
            next_show_think_final = (
                bool(show_think_final)
                if show_think_final is not None
                else current.show_think_final
            )
```

**SQL UPDATE** — extend SET and params tuple. Replace the existing `conn.execute("""UPDATE user_settings SET ...""", (...))` with:

```python
            conn.execute(
                """
                UPDATE user_settings
                SET theme = ?, default_include_reasoning = ?, default_answer_style = ?,
                    analysis_depth = ?,
                    llm_temperature_chat = ?, llm_temperature_tool = ?,
                    llm_max_tokens_default = ?, llm_max_tokens_reasoning = ?,
                    backend_query_timeout_sec = ?, agent_max_steps = ?,
                    agent_step_timeout_sec = ?, agent_inner_recursion_limit = ?,
                    agent_react_enabled = ?,
                    ui_scale = ?,
                    llm_streaming = ?, show_thinking = ?,
                    show_think_planning = ?, show_think_tool = ?, show_think_final = ?,
                    updated_at = ?
                WHERE user_id = ?
                """,
                (
                    next_theme,
                    1 if next_reasoning else 0,
                    next_answer_style,
                    next_analysis_depth,
                    next_llm_temperature_chat,
                    next_llm_temperature_tool,
                    next_llm_max_tokens_default,
                    next_llm_max_tokens_reasoning,
                    next_backend_query_timeout_sec,
                    next_agent_max_steps,
                    next_agent_step_timeout_sec,
                    next_agent_inner_recursion_limit,
                    1 if next_agent_react_enabled else 0,
                    next_ui_scale,
                    1 if next_llm_streaming else 0,
                    1 if next_show_thinking else 0,
                    1 if next_show_think_planning else 0,
                    1 if next_show_think_tool else 0,
                    1 if next_show_think_final else 0,
                    self._now_iso(),
                    user_id,
                ),
            )
```

**Return** — extend the returned `UserSettings(...)` at the end of `update_user_settings`:

```python
        return UserSettings(
            theme=next_theme,
            default_include_reasoning=next_reasoning,
            default_answer_style=next_answer_style,
            analysis_depth=next_analysis_depth,
            llm_temperature_chat=next_llm_temperature_chat,
            llm_temperature_tool=next_llm_temperature_tool,
            llm_max_tokens_default=next_llm_max_tokens_default,
            llm_max_tokens_reasoning=next_llm_max_tokens_reasoning,
            backend_query_timeout_sec=next_backend_query_timeout_sec,
            agent_max_steps=next_agent_max_steps,
            agent_step_timeout_sec=next_agent_step_timeout_sec,
            agent_inner_recursion_limit=next_agent_inner_recursion_limit,
            agent_react_enabled=next_agent_react_enabled,
            ui_scale=next_ui_scale,
            llm_streaming=next_llm_streaming,
            show_thinking=next_show_thinking,
            show_think_planning=next_show_think_planning,
            show_think_tool=next_show_think_tool,
            show_think_final=next_show_think_final,
        )
```

- [ ] **Step 8: Run tests to verify they pass**

```bash
python -m pytest tests/test_user_think_settings.py -v
```

Expected: 3 tests PASS

- [ ] **Step 9: Commit**

```bash
git add backend/auth/auth_db.py tests/test_user_think_settings.py
git commit -m "feat: add llm_streaming and show_think_* fields to user_settings DB"
```

---

## Task 2: API Models — Expose 5 New Fields

**Files:**
- Modify: `backend/api/models.py`

- [ ] **Step 1: Add 5 fields to `UserSettingsResponse`**

In `backend/api/models.py`, extend `UserSettingsResponse` after the `ui_scale` field:

```python
class UserSettingsResponse(BaseModel):
    theme: str = Field(default="dark", pattern="^(light|dark)$")
    default_include_reasoning: bool = True
    default_answer_style: str = Field(default="detailed", pattern="^(concise|detailed)$")
    analysis_depth: str = Field(default="light", pattern="^(light|medium|deep)$")
    llm_temperature_chat: float = Field(default=0.5, ge=0.0, le=2.0)
    llm_temperature_tool: float = Field(default=0.15, ge=0.0, le=2.0)
    llm_max_tokens_default: int = Field(default=1200, ge=128, le=32768)
    llm_max_tokens_reasoning: int = Field(default=2200, ge=128, le=32768)
    backend_query_timeout_sec: int = Field(default=180, ge=15, le=1800)
    agent_max_steps: int = Field(default=DEPTH_MAX_STEPS["light"], ge=2, le=_MAX_STEPS)
    agent_step_timeout_sec: int = Field(default=45, ge=5, le=600)
    agent_inner_recursion_limit: int = Field(default=14, ge=2, le=30)
    ui_scale: int = Field(default=100, ge=70, le=150)
    llm_streaming: bool = True
    show_thinking: bool = True
    show_think_planning: bool = True
    show_think_tool: bool = True
    show_think_final: bool = True
```

- [ ] **Step 2: Add 5 fields to `UserSettingsUpdateRequest`**

Extend `UserSettingsUpdateRequest` after `ui_scale`:

```python
class UserSettingsUpdateRequest(BaseModel):
    theme: str | None = Field(default=None, pattern="^(light|dark)$")
    default_include_reasoning: bool | None = None
    default_answer_style: str | None = Field(default=None, pattern="^(concise|detailed)$")
    analysis_depth: str | None = Field(default=None, pattern="^(light|medium|deep)$")
    llm_temperature_chat: float | None = Field(default=None, ge=0.0, le=2.0)
    llm_temperature_tool: float | None = Field(default=None, ge=0.0, le=2.0)
    llm_max_tokens_default: int | None = Field(default=None, ge=128, le=32768)
    llm_max_tokens_reasoning: int | None = Field(default=None, ge=128, le=32768)
    backend_query_timeout_sec: int | None = Field(default=None, ge=15, le=1800)
    agent_max_steps: int | None = Field(default=None, ge=2, le=_MAX_STEPS)
    agent_step_timeout_sec: int | None = Field(default=None, ge=5, le=600)
    agent_inner_recursion_limit: int | None = Field(default=None, ge=2, le=30)
    ui_scale: int | None = Field(default=None, ge=70, le=150)
    llm_streaming: bool | None = None
    show_thinking: bool | None = None
    show_think_planning: bool | None = None
    show_think_tool: bool | None = None
    show_think_final: bool | None = None
```

- [ ] **Step 3: Verify the PATCH endpoint passes new fields through to `update_user_settings`**

Open `backend/api/routes/auth.py` and find the `PATCH /auth/settings` handler (around line 128). It should call `_auth_db.update_user_settings(user.id, **payload.model_dump(exclude_none=True))`. Confirm the kwargs will now include the 5 new optional fields when provided — no code change needed if it already uses `**payload.model_dump(exclude_none=True)`.

- [ ] **Step 4: Run existing auth tests to confirm no regressions**

```bash
python -m pytest tests/ -k "settings" -v
```

Expected: all settings-related tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/api/models.py
git commit -m "feat: expose llm_streaming and show_think_* in settings API models"
```

---

## Task 3: Wire `llm_streaming` from User Settings into Runner

**Files:**
- Modify: `backend/api/routes/query.py:258-272`

- [ ] **Step 1: Add `llm_streaming` override in `_effective_runtime_settings`**

Find `_effective_runtime_settings` at line 258 in `backend/api/routes/query.py`. It currently ends with `agent_analysis_depth=depth`. Add `llm_streaming` override:

```python
def _effective_runtime_settings(user_id: int, *, analysis_depth_override: str | None = None):
    user_runtime = _auth_db.get_user_settings(user_id)
    depth = analysis_depth_override or user_runtime.analysis_depth
    return replace(
        _settings,
        llm_temperature_chat=user_runtime.llm_temperature_chat,
        llm_temperature_tool=user_runtime.llm_temperature_tool,
        llm_max_tokens_default=user_runtime.llm_max_tokens_default,
        llm_max_tokens_reasoning=user_runtime.llm_max_tokens_reasoning,
        backend_query_timeout_sec=user_runtime.backend_query_timeout_sec,
        agent_max_steps=user_runtime.agent_max_steps,
        agent_step_timeout_sec=user_runtime.agent_step_timeout_sec,
        agent_inner_recursion_limit=user_runtime.agent_inner_recursion_limit,
        agent_analysis_depth=depth,
        llm_streaming=user_runtime.llm_streaming,
    )
```

- [ ] **Step 2: Verify `llm_streaming` is a field in the `Settings` dataclass**

Open `backend/core/config.py` and confirm `llm_streaming: bool` exists (line ~150). If the field is named differently, match the exact name used in the `replace()` call elsewhere in the codebase.

- [ ] **Step 3: Run smoke test to confirm no import errors**

```bash
python -c "from backend.api.routes.query import _effective_runtime_settings; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add backend/api/routes/query.py
git commit -m "feat: override llm_streaming from user settings in runtime config"
```

---

## Task 4: Frontend Types

**Files:**
- Modify: `frontend/src/app/lib/backend-types.ts:160-175`

- [ ] **Step 1: Add 5 fields to `UserSettings` type**

Replace the `UserSettings` type definition:

```typescript
export type UserSettings = {
  theme: "light" | "dark";
  default_include_reasoning: boolean;
  default_answer_style: "concise" | "detailed";
  analysis_depth: AnalysisDepth;
  llm_temperature_chat: number;
  llm_temperature_tool: number;
  llm_max_tokens_default: number;
  llm_max_tokens_reasoning: number;
  backend_query_timeout_sec: number;
  agent_max_steps: number;
  agent_step_timeout_sec: number;
  agent_inner_recursion_limit: number;
  agent_react_enabled: boolean;
  ui_scale: number;
  llm_streaming: boolean;
  show_thinking: boolean;
  show_think_planning: boolean;
  show_think_tool: boolean;
  show_think_final: boolean;
};
```

- [ ] **Step 2: Run TypeScript type check**

```bash
cd frontend
npx tsc --noEmit
```

Expected: no new errors from this file

- [ ] **Step 3: Commit**

```bash
git add frontend/src/app/lib/backend-types.ts
git commit -m "feat: add llm_streaming and show_think_* to UserSettings frontend type"
```

---

## Task 5: `filterReasoningSteps` Utility

**Files:**
- Create: `frontend/src/app/lib/think-filter.ts`
- Test: (vitest) inline type checks via tsc

- [ ] **Step 1: Create the utility file**

Create `frontend/src/app/lib/think-filter.ts`:

```typescript
import type { PersistedReasoningStep, UserSettings } from "./backend-types";

type ThinkVisibility = Pick<
  UserSettings,
  "show_thinking" | "show_think_planning" | "show_think_tool" | "show_think_final"
>;

/**
 * Filter reasoning steps according to user visibility settings.
 * - If show_thinking is false, returns [].
 * - Otherwise filters by kind: planning/tool_synthesis/final_synthesis.
 * - "unknown" kind falls back to show_think_tool.
 */
export function filterReasoningSteps(
  steps: PersistedReasoningStep[] | null | undefined,
  settings: ThinkVisibility,
): PersistedReasoningStep[] {
  if (!steps || steps.length === 0) return [];
  if (!settings.show_thinking) return [];

  return steps.filter((step) => {
    switch (step.kind) {
      case "planning":
        return settings.show_think_planning;
      case "tool_synthesis":
        return settings.show_think_tool;
      case "final_synthesis":
        return settings.show_think_final;
      default:
        return settings.show_think_tool; // "unknown" → tool fallback
    }
  });
}
```

- [ ] **Step 2: Verify TypeScript compiles cleanly**

```bash
cd frontend
npx tsc --noEmit
```

Expected: no errors

- [ ] **Step 3: Commit**

```bash
git add frontend/src/app/lib/think-filter.ts
git commit -m "feat: add filterReasoningSteps utility for think block visibility"
```

---

## Task 6: Apply Filter in ChatPanel + Live Streaming Guard

**Files:**
- Modify: `frontend/src/app/components/workspace/ChatPanel.tsx`

- [ ] **Step 1: Understand how settings reach ChatPanel**

Open `frontend/src/app/components/workspace/ChatPanel.tsx` and find the component's props type. Identify whether `settings: UserSettings` is already a prop. If it is, skip to Step 3. If not, go to Step 2.

- [ ] **Step 2: Add `settings` prop to ChatPanel (if not already present)**

Find the `Props` type at the top of `ChatPanel.tsx` and add:

```typescript
type Props = {
  // ... existing props ...
  settings: UserSettings;
};
```

Update the function signature to destructure `settings`. Find every place `ChatPanel` is instantiated in the codebase and pass the settings object:

```bash
cd frontend
grep -r "ChatPanel" src --include="*.tsx" -l
```

Pass `settings={settings}` at each call site.

- [ ] **Step 3: Import `filterReasoningSteps` and `UserSettings`**

At the top of `ChatPanel.tsx`, add:

```typescript
import { filterReasoningSteps } from "../../lib/think-filter";
```

Confirm `UserSettings` is already imported from `backend-types`; if not, add it.

- [ ] **Step 4: Apply filter to history rendering of `reasoning_steps`**

Find the block around line 477 that renders `message.reasoning_steps`:

```tsx
{!isUser && message.reasoning_steps?.filter((s) => !s.tool_name).length
  ? message.reasoning_steps
      .filter((s) => !s.tool_name)
      .map((step) => (
        <ThinkingBlock key={`rs-${step.step_index}`} content={step.content} defaultCollapsed />
      ))
  : null}
```

Replace with:

```tsx
{!isUser && (() => {
  const filtered = filterReasoningSteps(
    message.reasoning_steps?.filter((s) => !s.tool_name) ?? [],
    settings,
  );
  return filtered.length > 0
    ? filtered.map((step) => (
        <ThinkingBlock key={`rs-${step.step_index}`} content={step.content} defaultCollapsed />
      ))
    : null;
})()}
```

- [ ] **Step 5: Guard live streaming thinking blocks with `show_thinking`**

In `ChatPanel.tsx` (or wherever the `onThinkingStart` / `onThinkingEnd` / `onReasoning` handlers are wired), wrap the live thinking display so it is a no-op when `settings.show_thinking` is false.

Find the handlers that set live thinking state (look for `liveThinking`, `LiveThinkingBlock`, or similar state setters). Wrap the setter:

```typescript
onThinkingEnd: (text) => {
  if (!settings.show_thinking) return;  // ← add this guard
  setLiveThinking(text);
},
onReasoning: (text, mode) => {
  if (!settings.show_thinking) return;  // ← add this guard
  // existing logic...
},
```

The exact variable names depend on what ChatPanel uses — check the useChatAgent hook integration in ChatPanel.

- [ ] **Step 6: Run TypeScript type check**

```bash
cd frontend
npx tsc --noEmit
```

Expected: no errors

- [ ] **Step 7: Commit**

```bash
git add frontend/src/app/components/workspace/ChatPanel.tsx
git commit -m "feat: apply filterReasoningSteps to history and guard live thinking by show_thinking"
```

---

## Task 7: SettingsPanel — New Section

**Files:**
- Modify: `frontend/src/app/components/workspace/SettingsPanel.tsx`

- [ ] **Step 1: Add streaming toggle and think visibility section**

In `SettingsPanel.tsx`, add a new `<SectionCard>` block after the existing `"Runtime агента"` section (after line 157, before `"Память сессии"`). The `Eye` icon from lucide-react is appropriate here.

First, add `Eye` and `Radio` to the existing lucide import at line 2:

```typescript
import { BookOpen, Brain, Cpu, Eye, Info, Loader2, Radio, RefreshCw, Settings, Sliders, X } from "lucide-react";
```

Then add the new section:

```tsx
<SectionCard title="Стриминг и thinking" icon={<Eye className="h-3.5 w-3.5" />}>
  <div className="grid gap-3">
    {/* Streaming toggle */}
    <label className="inline-flex items-center justify-between rounded-xl border border-border/40 bg-background/25 px-4 py-3">
      <div className="flex items-center gap-2">
        <Radio className="h-3.5 w-3.5 text-muted-foreground" />
        <span className="text-sm">Стримить ответ в реальном времени</span>
      </div>
      <input
        type="checkbox"
        checked={draft.llm_streaming}
        onChange={(e) => setDraft((prev) => ({ ...prev, llm_streaming: e.target.checked }))}
        className="h-4 w-4 accent-primary ml-3 shrink-0"
      />
    </label>

    {/* Master thinking toggle */}
    <label className="inline-flex items-center justify-between rounded-xl border border-border/40 bg-background/25 px-4 py-3">
      <span className="text-sm">Показывать thinking блоки</span>
      <input
        type="checkbox"
        checked={draft.show_thinking}
        onChange={(e) => setDraft((prev) => ({ ...prev, show_thinking: e.target.checked }))}
        className="h-4 w-4 accent-primary ml-3 shrink-0"
      />
    </label>

    {/* Per-kind sub-checkboxes — indented, disabled when master is off */}
    <div className={`ml-4 grid gap-2 transition-opacity ${draft.show_thinking ? "opacity-100" : "opacity-40"}`}>
      {(
        [
          { key: "show_think_planning",  label: "Планирование" },
          { key: "show_think_tool",      label: "Синтез инструментов" },
          { key: "show_think_final",     label: "Финальный вывод" },
        ] as const
      ).map(({ key, label }) => (
        <label
          key={key}
          className={`inline-flex items-center justify-between rounded-xl border border-border/30 bg-background/15 px-4 py-2.5 ${!draft.show_thinking ? "pointer-events-none" : ""}`}
        >
          <span className="text-[13px] text-muted-foreground">{label}</span>
          <input
            type="checkbox"
            checked={draft[key]}
            disabled={!draft.show_thinking}
            onChange={(e) => setDraft((prev) => ({ ...prev, [key]: e.target.checked }))}
            className="h-4 w-4 accent-primary ml-3 shrink-0"
          />
        </label>
      ))}
    </div>
  </div>
</SectionCard>
```

- [ ] **Step 2: Ensure `draft` TypeScript type includes the 5 new fields**

Since `draft` is typed as `UserSettings` (line 26: `const [draft, setDraft] = useState<UserSettings>(settings)`), and `UserSettings` now includes the 5 new fields from Task 4, this is automatically covered. Run tsc to confirm:

```bash
cd frontend
npx tsc --noEmit
```

Expected: no errors

- [ ] **Step 3: Commit**

```bash
git add frontend/src/app/components/workspace/SettingsPanel.tsx
git commit -m "feat: add streaming + thinking visibility section to SettingsPanel"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] `llm_streaming` DB column + dataclass + API + frontend + wired in runner → Tasks 1, 2, 3, 4, 7
- [x] `show_thinking` master + per-kind DB/API/types → Tasks 1, 2, 4
- [x] `filterReasoningSteps` utility → Task 5
- [x] Filter applied to ChatPanel history rendering → Task 6
- [x] Live streaming guarded by `show_thinking` → Task 6 Step 5
- [x] SettingsPanel new section → Task 7

**No placeholders:** All steps contain actual code.

**Type consistency:**
- `filterReasoningSteps` takes `PersistedReasoningStep[]` (same type used in ChatPanel `message.reasoning_steps`)
- `ThinkVisibility` is a `Pick<UserSettings, ...>` — consistent with the fields added in Task 4
- `show_think_planning | show_think_tool | show_think_final` naming is consistent across all 7 tasks

**Potential issue noted:** Task 6 Step 2 says "if `settings` is not already a prop in ChatPanel, add it". The exact call sites for ChatPanel are unknown without reading the full app. The plan includes the grep command to find them.
