# Quality And Refactor Roadmap

Goal: reduce regression risk before large architecture work. Keep each increment
small enough to review and verify independently.

## Increment 1: Baseline Gates

Status: started.

Scope:
- Add a CI quality stage before build/deploy.
- Expose a frontend baseline through `npm run verify`.
- Document that frontend contract tests need a TypeScript-aware runner before
  they can be part of the canonical Node 20 baseline.
- Run quality commands inside Docker so shell runners do not need host-level
  `poetry`/`npm`.
- Document the baseline commands.

Done when:
- `poetry check` passes.
- `poetry run pytest -m "not live and not e2e" -q` passes.
- `poetry run ruff check .` passes or known legacy violations are documented.
- `cd frontend && npm run verify` passes in CI.

## Increment 2: Frontend Surface Split

Status: started.

Scope:
- Split `DashboardPanel.tsx` by vertical area without changing props or behavior.
- Start with board layout helpers and source panels.
- Avoid changing streaming or session switching in this increment.

Completed slices:
- Board layout helpers moved to `frontend/src/app/components/workspace/board-layout.ts`.
- DB connection form helpers moved to
  `frontend/src/app/components/workspace/db-connection-form.ts`.
- RAG status normalization helpers moved to
  `frontend/src/app/components/workspace/rag-status.ts`.
- OpenProject form payload helpers moved to
  `frontend/src/app/components/workspace/openproject-form.ts`.

Done when:
- Frontend build passes.
- Existing frontend contract tests pass.
- No public API contract changes.

## Increment 3: Backend Wiring

Scope:
- Move service construction out of `backend/api/app.py` into an application
  container/factory.
- Keep route behavior and setup calls stable.
- Do not refactor query streaming in the same increment.

Done when:
- Offline backend tests pass.
- `tests/test_api_app_wiring.py` and route contract tests pass.

## Increment 4: Session Consistency

Scope:
- Add explicit session turn persistence helpers around chat messages, artifacts,
  context usage, and session metadata updates.
- Preserve current storage format.

Done when:
- Query execution tests cover success, fallback, interrupted stream, and repeated
  persistence calls.
- No storage migration is required.

## Increment 5: Query Execution Split

Scope:
- Split `QueryExecutionService` into focused collaborators:
  runtime preparation, non-stream execution, stream execution, persistence, and
  response/fallback building.
- Keep SSE event contracts stable.

Done when:
- Existing stream/query tests pass.
- One added contract test covers final event persistence and one covers
  interrupted stream persistence.

## Separate Reliability Tracks

Do separately from the refactor increments:
- Replace sandbox thread timeout with process-level hard kill.
- Replace Redis pickle cache serialization with a versioned safe format.
- Decide whether SQLite auth migrations need a dedicated migration layer.
