# Frontend Refactor Next

This is deferred work. Do not change frontend source files as part of the
2026-06-29 backend cleanup.

## DashboardPanel split

Current issue:

- `frontend/src/app/components/workspace/DashboardPanel.tsx` owns source tabs,
  DB/RAG/OpenProject state, board layout, artifact rendering, export state, and
  several API workflows.

Next implementation should:

- Move source-management API calls and state into focused hooks/services.
- Move board layout persistence and resize logic into a board hook.
- Extract source panels for DB, RAG, and OpenProject.
- Keep the visual dashboard component focused on composition and rendering.

## Chat reasoning contract

Current issue:

- `frontend/src/app/hooks/useChatAgent.ts` reconstructs `AssistantBlock[]` from
  persisted history and mirrors backend private reasoning-step ordering.

Next implementation should:

- Make the backend return a public persisted `AssistantBlock[]` or equivalent
  stable step contract.
- Keep frontend logic as a renderer/filter for that contract.
- Remove duplicated backend reasoning/block mapping from the hook.
