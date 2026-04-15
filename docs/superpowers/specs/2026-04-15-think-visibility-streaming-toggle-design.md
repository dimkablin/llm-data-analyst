# Spec: Think Visibility & Streaming Toggle

**Date:** 2026-04-15  
**Status:** Approved

---

## Overview

Add user-controlled settings to:
1. Toggle LLM response streaming on/off
2. Show/hide individual groups of thinking blocks — both during live streaming and when viewing history loaded from the database

The data model follows **Approach B** (flat boolean flags per thinking kind). A future **Approach C** (preset buttons) can be layered on top of the frontend without any backend changes.

---

## Scope

### In scope
- 5 new `user_settings` fields: `llm_streaming`, `show_thinking`, `show_think_planning`, `show_think_tool`, `show_think_final`
- DB migration (ALTER TABLE)
- Backend: UserSettings dataclass + API models updated
- Backend: `llm_streaming` user preference wired into `query.py` → runner
- Frontend: `UserSettings` type updated
- Frontend: `filterReasoningSteps()` utility
- Frontend: new SettingsPanel section (streaming toggle + master thinking toggle + 3 sub-checkboxes)
- Frontend: live streaming respects `show_thinking` master flag

### Out of scope
- Preset buttons (Variant C) — frontend-only addition, deferred
- Per-message visibility overrides
- Changing how thinking blocks are stored in the DB

---

## Data Model

### DB migration

```sql
ALTER TABLE user_settings ADD COLUMN llm_streaming       INTEGER DEFAULT 1;
ALTER TABLE user_settings ADD COLUMN show_thinking        INTEGER DEFAULT 1;
ALTER TABLE user_settings ADD COLUMN show_think_planning  INTEGER DEFAULT 1;
ALTER TABLE user_settings ADD COLUMN show_think_tool      INTEGER DEFAULT 1;
ALTER TABLE user_settings ADD COLUMN show_think_final     INTEGER DEFAULT 1;
```

All defaults = `1` (true) — existing users retain full visibility.

### Backend dataclass (`backend/auth/auth_db.py` — `UserSettings`)

```python
llm_streaming: bool = True
show_thinking: bool = True
show_think_planning: bool = True
show_think_tool: bool = True
show_think_final: bool = True
```

### API models (`backend/api/models.py`)

Both `UserSettingsUpdateRequest` and `UserSettingsResponse` gain the same 5 optional fields (all `bool | None`). No special validation beyond type checking.

---

## Backend: Streaming Toggle

**File:** `backend/api/routes/query.py`

When constructing `RuntimeSettings` (or equivalent), read `user_settings.llm_streaming` and pass it to the runner instead of the global `config.llm_streaming`.

Priority: `user_settings.llm_streaming` overrides `config.llm_streaming` when a user is authenticated and has a stored preference.

---

## Frontend: Filtering Logic

### `kind` → settings flag mapping

| `PersistedReasoningStep.kind` | Flag |
|---|---|
| `"planning"` | `show_think_planning` |
| `"tool_synthesis"` | `show_think_tool` |
| `"final_synthesis"` | `show_think_final` |
| `"unknown"` | `show_think_tool` (fallback) |

### Utility: `frontend/src/app/lib/think-filter.ts`

```typescript
export function filterReasoningSteps(
  steps: PersistedReasoningStep[],
  settings: Pick<UserSettings,
    "show_thinking" | "show_think_planning" |
    "show_think_tool" | "show_think_final"
  >,
): PersistedReasoningStep[] {
  if (!settings.show_thinking) return [];
  return steps.filter((s) => {
    if (s.kind === "planning")        return settings.show_think_planning;
    if (s.kind === "tool_synthesis")  return settings.show_think_tool;
    if (s.kind === "final_synthesis") return settings.show_think_final;
    return settings.show_think_tool; // unknown → tool fallback
  });
}
```

Called in the single place where history messages render their `reasoning_steps`.

### Live streaming

If `show_thinking === false`: `onThinkingStart` / `onThinkingEnd` / `onReasoning` SSE events are ignored client-side. No backend change needed.

Per-kind filtering during live streaming is not applied (blocks arrive as a stream, kind is assigned post-hoc). Granular kind filtering applies only to history view.

---

## Frontend: SettingsPanel

New section added to `SettingsPanel.tsx` below existing agent parameters:

```
─── Стриминг ────────────────────────────────────────
  [toggle] Стримить ответ в реальном времени

─── Thinking блоки ──────────────────────────────────
  [toggle] Показывать thinking блоки           ← master

    [checkbox] Планирование                    ┐
    [checkbox] Синтез инструментов             ├ disabled (not hidden) when master = off
    [checkbox] Финальный вывод                 ┘
```

The three sub-checkboxes are visually `disabled` but remain visible when the master toggle is off — users can see the options exist without being able to interact with them.

Each toggle/checkbox calls `PATCH /auth/settings` immediately on change (same pattern as existing settings fields).

---

## Future: Preset Buttons (Variant C, frontend-only)

No DB or API changes required. Preset buttons map to combinations of existing flags:

| Preset | `show_thinking` | `planning` | `tool` | `final` |
|---|---|---|---|---|
| Скрыть | `false` | — | — | — |
| Краткие | `true` | `true` | `false` | `false` |
| Полные | `true` | `true` | `true` | `true` |

A preset button is highlighted when the current flag state matches its combination exactly. Custom (non-preset) combinations show no highlighted button.

```typescript
function applyPreset(preset: "none" | "brief" | "full") {
  const presets = {
    none:  { show_thinking: false, show_think_planning: false, show_think_tool: false, show_think_final: false },
    brief: { show_thinking: true,  show_think_planning: true,  show_think_tool: false, show_think_final: false },
    full:  { show_thinking: true,  show_think_planning: true,  show_think_tool: true,  show_think_final: true  },
  };
  updateSettings(presets[preset]);
}
```

---

## Files Changed

| File | Change |
|---|---|
| `backend/auth/auth_db.py` | DB migration + UserSettings dataclass +5 fields |
| `backend/api/models.py` | UserSettingsUpdateRequest + UserSettingsResponse +5 fields |
| `backend/api/routes/query.py` | Read `user_settings.llm_streaming`, pass to runner |
| `frontend/src/app/lib/backend-types.ts` | UserSettings type +5 fields |
| `frontend/src/app/lib/think-filter.ts` | New file: `filterReasoningSteps()` utility |
| `frontend/src/app/components/workspace/SettingsPanel.tsx` | New section: streaming + thinking toggles |
| `frontend/src/app/components/workspace/ChatPanel.tsx` | Apply `filterReasoningSteps()` before rendering `reasoning_steps` |
