# Semantic Layer Implementation Report

Baseline:
- Branch: `feat/semantic_layer_1984`
- HEAD before work: `329901f`
- Commits over `origin/dev` before work: `329901f fix: harden plan-fact workflow`, `fd2646a feat: add plan-fact source workflow`
- Baseline semantic tests: `python -m pytest tests/test_semantic_catalog.py tests/test_semantic_generation.py tests/test_semantic_query.py` -> 33 passed, 1 warning
- Full backend non-live tests after implementation: `python -m pytest -m "not live"` -> 923 passed, 2 skipped, 7 deselected, 29 warnings
- Full backend including live tests: `python -m pytest` -> 4 failed, 922 passed, 6 skipped; 3 failures require Ollama at `localhost:11434`, 1 local plan-fact failure was fixed and rechecked.

## Этап 1. Базовый технический каркас

Статус: готово

Что реализовано:
- Connection-level semantic status returns `not_built` before explicit Build.
- DB source bind no longer starts heavy semantic build automatically.
- Existing session semantic endpoints remain compatibility wrappers.

Измененные файлы:
- `backend/api/routes/sources.py`
- `backend/api/routes/semantic_catalog.py`
- `backend/data_access/semantic_models.py`
- `frontend/src/app/components/workspace/SemanticCatalogBlock.tsx`

Какие новые файлы добавлены:
- `docs/semantic_layer_implementation_report.md`
- `tests/test_semantic_connection_level.py`

Назначение измененных и новых файлов:
- `sources.py`: keeps DB bind lightweight; technical catalog refresh remains, semantic Build is manual.
- `semantic_catalog.py`: exposes connection-level status/build/refresh/catalog endpoints.
- `semantic_models.py`: adds `not_built` status and connection-level catalog fields.
- `SemanticCatalogBlock.tsx`: shows manual Build for DB connection catalog.
- `tests/test_semantic_connection_level.py`: regression coverage for connection-level behavior and ACL.

Какие тесты выполнены:
- `python -m pytest tests/test_api_app_wiring.py tests/test_semantic_connection_level.py tests/test_semantic_catalog.py tests/test_semantic_generation.py tests/test_semantic_query.py` -> 39 passed, 2 warnings
- `python -m pytest -m "not live"` -> 923 passed, 2 skipped, 7 deselected, 29 warnings
- `python -m py_compile backend\data_access\semantic_catalog_store.py backend\data_access\semantic_catalog_service.py backend\data_access\semantic_context.py backend\api\routes\semantic_catalog.py backend\auth\auth_db.py backend\data_access\db_connections_service.py` -> passed

Что осталось нереализованным:
- Full live backend smoke against running FastAPI was not executed.

Как воспроизвести результат:
- Select a DB connection in UI, open semantic layer settings, observe `not_built`.
- Click `Сформировать семантический слой`; status becomes `indexing`, then `ready` or `failed`.

## Этап 2. Backend-модели

Статус: готово

Что реализовано:
- `SemanticCatalog` now carries `connection_id`.
- `not_built` status is part of the typed status contract.
- Unsafe derived metric formulas are rejected by service validation.

Измененные файлы:
- `backend/data_access/semantic_models.py`
- `backend/data_access/semantic_catalog_service.py`
- `frontend/src/app/lib/backend-types.ts`

Какие тесты выполнены:
- `tests/test_semantic_connection_level.py`
- Existing semantic model/catalog tests.

Что осталось нереализованным:
- Public model fields still retain legacy `version`, `overlay_version`, `published_version` for compatibility.

Как воспроизвести результат:
- Run `python -m pytest tests/test_semantic_connection_level.py`.

## Этап 3. PostgreSQL store

Статус: готово

Что реализовано:
- `SemanticCatalogPostgresStore` writes current catalog into normalized PostgreSQL tables:
  `semantic_catalogs`, `semantic_tables`, `semantic_columns`, `semantic_column_profiles`,
  `semantic_relationships`, `semantic_metrics`, `semantic_terms`,
  `semantic_refresh_jobs`, `semantic_audit_log`.
- Qdrant remains optional; PostgreSQL is the source of truth for catalog data.
- `SEMANTIC_METADATA_DATABASE_URL` and `SEMANTIC_METADATA_SCHEMA` configure the mandatory metadata PostgreSQL store.
- There is no file-store fallback: a missing DSN fails startup configuration.

Измененные файлы:
- `backend/data_access/semantic_catalog_store.py`
- `backend/core/config.py`
- `.env.example`
- `docker-compose.yaml`

Какие тесты выполнены:
- `py_compile` on store and semantic route/service modules.
- `tests/test_api_app_wiring.py` -> passed.
- `docker compose config` -> config rendered successfully; Docker warned about denied access to `C:\Users\dimka\.docker\config.json`.

Что осталось нереализованным:
- A real PostgreSQL write/read smoke was not executed in this turn.
- Generated drafts and overlays are persisted as JSONB documents in the same metadata PostgreSQL database.

Как воспроизвести результат:
- Run backend with `SEMANTIC_METADATA_DATABASE_URL=...` and optional `SEMANTIC_METADATA_SCHEMA=...`.
- Build a semantic catalog and inspect the listed `semantic_*` tables in PostgreSQL.

## Этап 4. Профилирование БД

Статус: готово

Что реализовано:
- Connection-level Build profiles external DB runtime via existing `DBAnalyticsHelper`.
- Existing profile limits remain driven by `SEMANTIC_PROFILE_*`.

Измененные файлы:
- `backend/data_access/semantic_catalog_service.py`

Какие тесты выполнены:
- `tests/test_semantic_connection_level.py` uses a fake DB helper to verify external DB build flow.

Что осталось нереализованным:
- Live PostgreSQL/ClickHouse profiling smoke was not executed.

Как воспроизвести результат:
- Call `POST /db-connections/{connection_id}/semantic-catalog/build`.

## Этап 5. SemanticCatalogService

Статус: готово

Что реализовано:
- Added `build_for_connection`, `load_for_connection`, `status_for_connection`.
- Added connection-level metric CRUD and search helpers.
- Catalog identity is stable per `connection_id`.

Измененные файлы:
- `backend/data_access/semantic_catalog_service.py`
- `backend/data_access/semantic_context.py`

Какие тесты выполнены:
- `tests/test_semantic_connection_level.py`
- Existing semantic catalog/context tests.

Что осталось нереализованным:
- Refresh job rows are schema-ready but not populated as a separate job lifecycle table beyond audit save.

Как воспроизвести результат:
- Build a catalog, then call `GET /db-connections/{connection_id}/semantic-catalog/status`.

## Этап 6. API semantic layer

Статус: готово

Что реализовано:
- Added connection-level endpoints:
  - `GET /db-connections/{connection_id}/semantic-catalog/status`
  - `GET /db-connections/{connection_id}/semantic-catalog`
  - `POST /db-connections/{connection_id}/semantic-catalog/build`
  - `POST /db-connections/{connection_id}/semantic-catalog/refresh`
  - `GET/POST/PATCH/DELETE /db-connections/{connection_id}/semantic-catalog/metrics`
  - `POST /db-connections/{connection_id}/semantic-catalog/search`
- Session endpoints remain compatibility wrappers.

Измененные файлы:
- `backend/api/routes/semantic_catalog.py`
- `backend/api/app.py`

Какие тесты выполнены:
- Targeted service tests; route-level TestClient coverage remains to add.

Что осталось нереализованным:
- Dedicated route tests for every new HTTP endpoint.
- Connection-level relationship/term write endpoints.

Как воспроизвести результат:
- Use the endpoints above with an authenticated user that has access to the DB connection.

## Этап 7. Metric layer

Статус: готово

Что реализовано:
- Connection-level metric create/update/delete persists through the catalog store.
- Dangerous SQL expressions are rejected.

Измененные файлы:
- `backend/data_access/semantic_catalog_service.py`
- `backend/api/routes/semantic_catalog.py`
- `frontend/src/app/lib/backend-api.ts`
- `frontend/src/app/components/workspace/SemanticCatalogBlock.tsx`

Какие тесты выполнены:
- `test_connection_metric_rejects_unsafe_formula`.

Что осталось нереализованным:
- Route-level metric CRUD tests.

Как воспроизвести результат:
- Build catalog, create metric through UI or `POST /db-connections/{connection_id}/semantic-catalog/metrics`.

## Этап 8. Semantic context для AI

Статус: готово

Что реализовано:
- Added `SemanticContextBuilder.build_from_catalog` for connection-level context formatting.
- Existing session semantic context tests continue to pass.

Измененные файлы:
- `backend/data_access/semantic_context.py`
- `backend/data_access/semantic_catalog_service.py`

Какие тесты выполнены:
- Existing `tests/test_semantic_catalog.py` semantic context tests.

Что осталось нереализованным:
- Live AI question over a two-user shared DB catalog was not executed.

Как воспроизвести результат:
- Build catalog, create metric with synonym, call semantic search/context for the same connection.

## Этап 9. Qdrant/search

Статус: готово

Что реализовано:
- Connection-level search tries Qdrant when enabled and falls back to lexical catalog search.
- Existing Qdrant/indexing tests remain green.

Измененные файлы:
- `backend/data_access/semantic_catalog_service.py`

Какие тесты выполнены:
- Existing semantic Qdrant/fallback tests.

Что осталось нереализованным:
- Live Qdrant enabled/disabled smoke was not executed.

Как воспроизвести результат:
- Call `POST /db-connections/{connection_id}/semantic-catalog/search` with Qdrant enabled and disabled.

## Этап 10. UI build/refresh/status

Статус: готово

Что реализовано:
- Semantic settings panel detects selected DB connection.
- UI uses connection-level status/catalog/build/refresh for DB sources.
- Build button appears before first build; Refresh is disabled while `not_built`.

Измененные файлы:
- `frontend/src/app/components/workspace/SettingsPanel.tsx`
- `frontend/src/app/components/workspace/SemanticCatalogBlock.tsx`
- `frontend/src/app/lib/backend-api.ts`
- `frontend/src/app/lib/backend-types.ts`

Какие тесты выполнены:
- `npm.cmd run build` from `frontend` -> passed, Vite chunk-size warning only.

Что осталось нереализованным:
- Browser screenshot/manual UI smoke was not executed.

Как воспроизвести результат:
- Select DB source, open Settings -> Semantic Layer, click Build, then Refresh.

## Этап 11. UI catalog и metrics

Статус: частично готово

Что реализовано:
- Existing catalog/table/column/metric UI is reused.
- Metric CRUD uses connection-level API when a DB connection is selected.

Измененные файлы:
- `frontend/src/app/components/workspace/SemanticCatalogBlock.tsx`
- `frontend/src/app/lib/backend-api.ts`

Какие тесты выполнены:
- `npm.cmd run build` from `frontend` -> passed.

Что осталось нереализованным:
- Connection-level relationship/term UI write APIs are still routed through session compatibility endpoints.
- No browser accessibility smoke was executed.

Как воспроизвести результат:
- Build catalog, open Metrics tab, create/edit/delete a metric.

## Этап 12. Полный E2E

Статус: частично готово

Что реализовано:
- Minimal ACL table `user_db_connection_access`.
- Owner can grant access; grantee can list/read the same connection; outsider cannot.
- Shared connection-level catalog is loadable independently of session ownership.

Измененные файлы:
- `backend/auth/auth_db.py`
- `backend/data_access/db_connections_service.py`
- `backend/api/routes/db_connections.py`
- `backend/api/models.py`
- `tests/test_semantic_connection_level.py`

Какие тесты выполнены:
- `test_db_connection_acl_shares_one_connection_without_global_visibility`.

Что осталось нереализованным:
- Full two-user UI/API/AI live E2E against real PostgreSQL and Qdrant was not executed.
- Failure scenarios are covered only by existing semantic tests and targeted unsafe formula test, not by a full E2E matrix.

Как воспроизвести результат:
- Owner creates DB connection.
- Owner calls `POST /db-connections/{connection_id}/access` with `{"user_id": <user_b_id>}`.
- User B lists DB connections and sees the shared connection.
- User B opens semantic layer and sees the connection-level catalog after build.

## Файлы

Файл: `.env.example`
Тип изменения: изменен
Назначение: documents new semantic metadata env aliases.
Связанный этап: 3
Как проверить: inspect env names.

Файл: `docker-compose.yaml`
Тип изменения: изменен
Назначение: passes `SEMANTIC_METADATA_*` to backend.
Связанный этап: 3
Как проверить: `docker compose config`.

Файл: `backend/api/app.py`
Тип изменения: изменен
Назначение: passes DB runtime service into semantic routes.
Связанный этап: 6
Как проверить: import/app startup.

Файл: `backend/api/models.py`
Тип изменения: изменен
Назначение: request model for granting DB connection access.
Связанный этап: 12
Как проверить: route model import.

Файл: `backend/api/routes/db_connections.py`
Тип изменения: изменен
Назначение: owner grants DB connection access to another user.
Связанный этап: 12
Как проверить: `POST /db-connections/{connection_id}/access`.

Файл: `backend/api/routes/semantic_catalog.py`
Тип изменения: изменен
Назначение: connection-level semantic catalog endpoints.
Связанный этап: 6
Как проверить: call connection semantic endpoints.

Файл: `backend/api/routes/sources.py`
Тип изменения: изменен
Назначение: avoids automatic semantic build on DB bind.
Связанный этап: 1
Как проверить: bind DB source and observe semantic status `not_built`.

Файл: `backend/auth/auth_db.py`
Тип изменения: изменен
Назначение: minimal DB connection ACL table and read logic.
Связанный этап: 12
Как проверить: `test_db_connection_acl_shares_one_connection_without_global_visibility`.

Файл: `backend/core/config.py`
Тип изменения: изменен
Назначение: configures mandatory PostgreSQL metadata storage through `SEMANTIC_METADATA_DATABASE_URL` and `SEMANTIC_METADATA_SCHEMA`.
Связанный этап: 3
Как проверить: instantiate settings with new env names.

Файл: `backend/data_access/db_connections_service.py`
Тип изменения: изменен
Назначение: service method to grant connection access.
Связанный этап: 12
Как проверить: DB connection route test/manual call.

Файл: `backend/data_access/csv_session_runtime.py`
Тип изменения: изменен
Назначение: normalizes nullable text/object values returned from DuckDB queries to `None` instead of pandas `NaN`.
Связанный этап: compatibility
Как проверить: `tests/test_planfact_source_service.py::test_planfact_service_content_is_carried_into_breakdowns`.

Файл: `backend/data_access/planfact_source_service.py`
Тип изменения: изменен
Назначение: keeps plan-fact service content nullable in derived article breakdowns.
Связанный этап: compatibility
Как проверить: `tests/test_planfact_source_service.py::test_planfact_service_content_is_carried_into_breakdowns`.

Файл: `backend/data_access/semantic_catalog_service.py`
Тип изменения: изменен
Назначение: connection-level build/load/status/metric/search flow and SQL metric validation.
Связанный этап: 4, 5, 7, 8, 9
Как проверить: `tests/test_semantic_connection_level.py`.

Файл: `backend/data_access/semantic_catalog_store.py`
Тип изменения: изменен
Назначение: normalized PostgreSQL semantic store schema and persistence.
Связанный этап: 3
Как проверить: build catalog with Postgres metadata DSN and inspect `semantic_*` tables.

Файл: `backend/data_access/semantic_context.py`
Тип изменения: изменен
Назначение: formats semantic context from a connection-level catalog.
Связанный этап: 8
Как проверить: semantic search/context tests.

Файл: `backend/data_access/semantic_models.py`
Тип изменения: изменен
Назначение: connection-level catalog fields and `not_built` status.
Связанный этап: 2
Как проверить: model tests.

Файл: `frontend/src/app/components/workspace/SemanticCatalogBlock.tsx`
Тип изменения: изменен
Назначение: UI uses connection-level semantic APIs for DB sources.
Связанный этап: 10, 11
Как проверить: frontend build and UI Build/Refresh flow.

Файл: `frontend/src/app/components/workspace/SettingsPanel.tsx`
Тип изменения: изменен
Назначение: resolves selected DB connection for semantic panel.
Связанный этап: 10
Как проверить: open settings with DB source selected.

Файл: `frontend/src/app/lib/backend-api.ts`
Тип изменения: изменен
Назначение: connection-level semantic API client functions.
Связанный этап: 10, 11
Как проверить: frontend build.

Файл: `frontend/src/app/lib/backend-types.ts`
Тип изменения: изменен
Назначение: adds `not_built` status type.
Связанный этап: 2, 10
Как проверить: frontend build.

Файл: `tests/test_semantic_connection_level.py`
Тип изменения: новый
Назначение: regression tests for connection-level catalog, unsafe metrics, and DB connection ACL.
Связанный этап: 1, 4, 7, 12
Как проверить: `python -m pytest tests/test_semantic_connection_level.py`.
