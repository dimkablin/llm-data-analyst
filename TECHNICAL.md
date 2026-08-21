# Техническое описание LLM Data Analyst

## 1. Архитектура агента

Оркестрация на LangGraph (`StateGraph`) — три узла:

```
START → dispatch ─┬─ (chat/summary) ──────────────► finalize → END
                  │
                  └─ (analysis) ──► agent ──────────► finalize → END
```

| Узел | Назначение |
|------|------------|
| `dispatch` | Детерминированный keyword pre-check (без LLM). Лёгкие bypass: `chat` (приветствия, вопросы о боте), `summary` (управленческая записка). Всё остальное → строит tools, sandbox, capability_context и передаёт в `agent` |
| `agent` | Единственный execution engine: нативный tool-calling через `bind_tools`. Получает полный system prompt (policy + skills + data context), гоняет tool loop до финального текста без вызовов |
| `finalize` | Перезаписывает ответ если нужно, запускает `review_tool` для проверки качества |

Ограничения:
- `AGENT_MAX_STEPS` — максимум tool-вызовов за один запрос (по умолчанию 32)
- `MAX_TOOLS_PER_CYCLE` — максимум tool-вызовов после одного ответа LLM (по умолчанию 4)
- `AGENT_STEP_TIMEOUT_SEC` — таймаут на один шаг (по умолчанию 45с)

При ошибке или таймауте — graceful fallback с текстовым ответом, без 5xx.

### Skills-система

Skills — markdown-инструкции в `skills/{name}/SKILL.md`, загружаются через `SkillRegistry` и инжектируются в system prompt агента. Позволяют расширять поведение агента без изменения кода.

### Глубина анализа

Три режима влияют на детальность system prompt и потребление токенов:

| Режим | Поведение |
|-------|-----------|
| `light` | Короткий plan prompt, минимум шагов (по умолчанию) |
| `medium` | Средняя детализация |
| `deep` | Развёрнутый план, больше токенов reasoning |

Настраивается через `AGENT_ANALYSIS_DEPTH` в `.env` или в UI (боковая панель настроек).

## 2. Модель и провайдер

Единый OpenAI-compatible endpoint:

- **Локально**: Ollama (`qwen3.5:...`)
- **Сервер**: vLLM, TGI или любой OpenAI-compatible API

Конфигурация из `.env`:

| Переменная | Назначение | Рекомендация (Qwen 3.5) |
|------------|------------|------------------------|
| `LLM_TEMPERATURE_CHAT` | Температура для chat-режима | 0.7 |
| `LLM_TEMPERATURE_TOOL` | Температура для tool-вызовов | 0.5 |
| `LLM_TOP_P` | Nucleus sampling | 0.95 |
| `LLM_TOP_K` | Top-K sampling | 20 |
| `LLM_PRESENCE_PENALTY` | Штраф за повторения | 1.5 |
| `LLM_MAX_TOKENS_DEFAULT` | Лимит токенов (chat) | 2048 |
| `LLM_MAX_TOKENS_REASONING` | Лимит токенов (reasoning) | 4096 |
| `LLM_ENABLE_THINKING` | Включить thinking-режим модели | true |

## 3. Инструменты и песочница

Инструменты регистрируются через `ToolRegistry` / `ToolCatalog` и подбираются в `dispatch` исходя из режима источника данных.

| Инструмент | Возвращает | Применение |
|------------|------------|------------|
| `PandasTool` | `pd.DataFrame` / `pd.Series` | Таблицы, агрегации, фильтрация |
| `PlotlyTool` | `plotly.graph_objects.Figure` | Графики и визуализации |
| `ValueTool` | `float` / `int` / `str` / `bool` | Скалярные метрики |
| `SQLTool` | результат запроса | SQL через DuckDB (CSV в памяти) |
| `DatabaseTool` | результат запроса | SQL через внешнее DB-подключение |
| `PlannerTool` | план / текст | LLM-планировщик (опциональный шаг) |
| `ReviewTool` | оценка / правки | Проверка качества ответа |
| `SearchTool` | результаты поиска | Веб-поиск |
| `RAGTool` | фрагменты | Retrieval-Augmented Generation |
| `ForecastTool` | прогноз | Временные ряды |
| `AnomalyPlanfactTool` | аномалии | План-факт анализ |
| `MemoryTool` | текст | Пользовательская память |
| `GetToolInstructionsTool` | markdown | Динамическая загрузка skill-инструкций |

Безопасность исполнения (code-executing tools):
- Код выполняется в отдельном подпроцессе (`forkserver` / `spawn`, не `fork`)
- Таймаут `TOOL_EXEC_TIMEOUT_SEC` (по умолчанию 25с)
- Ограниченные builtins и импорты
- Блокировка опасных паттернов (`open`, `eval`, `exec`, системные модули)
- Tool-level кеш: `TOOL_CACHE_SIZE` записей

## 4. Backend API

FastAPI с SSE-стримингом.

### Аутентификация

PostgreSQL `app_data`, JWT-токены с TTL (`AUTH_TOKEN_TTL_DAYS`).

| Эндпоинт | Описание |
|----------|----------|
| `POST /auth/register` | Регистрация нового пользователя |
| `POST /auth/login` | Авторизация, получение токена |
| `GET /auth/me` | Текущий пользователь |
| `POST /auth/change-password` | Смена пароля |
| `GET /auth/settings` | Настройки пользователя (температуры, токены, глубина анализа) |
| `PATCH /auth/settings` | Обновление настроек |
| `POST /auth/logout` | Выход |
| `GET /admin/users` | Список пользователей (admin) |
| `POST /admin/users` | Создание пользователя (admin) |
| `PATCH /admin/users/{id}` | Изменение роли/пароля (admin) |
| `DELETE /admin/users/{id}` | Удаление пользователя (admin) |

### Сессии и запросы

| Эндпоинт | Описание |
|----------|----------|
| `POST /sessions` | Создать сессию |
| `GET /sessions` | Список сессий пользователя |
| `GET /sessions/{id}` | Состояние сессии (история, артефакты, датасет) |
| `DELETE /sessions/{id}` | Удалить сессию |
| `PATCH /sessions/{id}/title` | Переименовать сессию |
| `POST /sessions/{id}/title/generate` | Автогенерация заголовка через LLM |
| `POST /sessions/{id}/data` | Загрузить CSV (лимит `BACKEND_MAX_DATASET_MB`) |
| `POST /sessions/{id}/query` | Синхронный запрос |
| `POST /sessions/{id}/query/stream` | SSE-стрим (события: `token`, `reasoning`, `reasoning_token`, `phase`, `phase_token`, `final`, `error`) |
| `POST /sessions/{id}/evaluate` | Запрос без записи в историю |
| `GET /runtime/model` | Информация о текущей модели |
| `GET /health` | Healthcheck |

### Кеширование

Query-level кеш (`AGENT_CACHE_ENABLED`, `AGENT_CACHE_SIZE`, `AGENT_CACHE_TTL_SEC`) — повторные идентичные запросы возвращают кешированный результат.

## 5. Хранение состояния

Постоянное состояние хранится в PostgreSQL: `app_data` содержит пользователей,
подключения, сессии, сообщения, артефакты, manifests, notebooks и загруженные файлы;
`semantic_metadata` содержит профили и семантические каталоги. Qdrant — производный индекс.

```
/tmp/llm-data-analyst/
└── sessions/               # временные DuckDB, Parquet и sandbox-файлы
```

| Аспект | Реализация |
|--------|------------|
| Формат runtime | Временные DuckDB и Parquet (pyarrow), восстанавливаемые из PostgreSQL |
| Кеш DataFrame | LRU на `OrderedDict`, максимум 20 записей |
| Конкурентность | Per-session `threading.Lock` через `_get_session_lock()` |
| TTL | Автоочистка сессий старше `BACKEND_SESSION_TTL_DAYS` (по умолчанию 7) |

## 6. Frontend

React + TypeScript + Vite.

### Архитектура

- Все API-запросы — относительные URL, проксируются через nginx (Docker) или Vite dev-proxy (локальная разработка)
- SSE-стриминг ответов с поддержкой abort
- Per-message панель активности: трейсинг фаз ReAct и reasoning для каждого сообщения
- Настройки в боковой панели: температуры, токены, таймауты, глубина анализа
- Дашборд артефактов с drag-and-drop
- Экспорт в HTML

### Безопасность

- Plotly iframe: экранирование `</script>` для защиты от XSS
- Phoenix iframe: `sandbox="allow-scripts allow-same-origin allow-forms"`

## 7. Инфраструктура (Docker)

### Архитектура

```
                    ┌─────────────────────────────────────┐
                    │        nginx (frontend)              │
  Browser ────────► │  :8603                               │
                    │  /auth, /admin, /sessions, /runtime  │──► backend:8000
                    │  /phoenix/                           │──► phoenix:6006
                    │  /*                                  │──► static SPA
                    └─────────────────────────────────────┘
```

- **Backend** — не выставлен наружу (`expose: 8000`), доступен только через nginx
- **Frontend** — единая точка входа (`FRONTEND_PORT:8603`)
- **Phoenix** — `PHOENIX_UI_PORT:6006` (UI) + `PHOENIX_GRPC_PORT:4317` (gRPC)

### Безопасность контейнера

- Backend запускается под `appuser` (non-root)
- nginx: заголовки `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`
- gzip-сжатие для статики
- `proxy_buffering off` для SSE-маршрутов
- `client_max_body_size 200m` для загрузки файлов
- `.env` в `.dockerignore` — не попадает в образ

### Healthcheck

| Сервис | Команда |
|--------|---------|
| Backend | `python3 -c "urllib.request.urlopen('http://localhost:8000/health')"` |
| Frontend | `curl -f http://localhost:8603/` |
| Phoenix | `python3 -c "urllib.request.urlopen('http://localhost:6006/healthz')"` |

Зависимости: phoenix → backend → frontend (каждый ждёт `service_healthy`).

## 8. Observability (Arize Phoenix)

- Backend при старте вызывает `phoenix.otel.register(...)` с `PHOENIX_COLLECTOR_ENDPOINT`
- LangChain инструментируется через `openinference.instrumentation.langchain`
- В Docker коллектор фиксирован: `http://phoenix:6006/v1/traces`
- При локальной разработке fallback: `http://{PHOENIX_HOST}:6006/v1/traces`

## 9. Основные модули

| Модуль | Назначение |
|--------|------------|
| `backend/api/app.py` | FastAPI app wiring, routes, SSE, timeout/fallback |
| `backend/agent/runner.py` | LangGraph-агент, узлы `dispatch / agent / finalize`, построение LLM |
| `backend/agent/callbacks.py` | LangChain callbacks: токены, reasoning, фазы, прогресс |
| `backend/agent/llm_client.py` | `ThinkingAwareChatOpenAI` — обёртка с поддержкой thinking-режима |
| `backend/agent/prompts.py` | System prompts: chat, execution, data-context |
| `backend/core/config.py` | `Settings` dataclass, чтение всех env-переменных |
| `backend/sessions/session_store.py` | Хранение сессий, DataFrame, артефактов |
| `backend/auth/auth_db.py` | SQLite auth, роли, пользовательские настройки |
| `backend/skills/registry.py` | `SkillRegistry` — загрузка и инжекция skills в system prompt |
| `backend/tools/impl/base_tool.py` | Песочница для исполнения кода |
| `backend/tools/impl/pandas_tool.py` | PandasTool |
| `backend/tools/impl/plotly_tool.py` | PlotlyTool |
| `backend/tools/impl/value_tool.py` | ValueTool |
| `backend/tools/impl/sql_tool.py` | SQLTool (DuckDB) |
| `backend/tools/impl/database_tool.py` | DatabaseTool (внешние БД) |
| `backend/tools/registry.py` | `ToolRegistry` — каталог и фабрика инструментов |
| `backend/tools/policy.py` | Политика доступа к инструментам |
| `backend/tools/sandbox_manager.py` | Управление пулом sandbox-процессов |
| `frontend/src/api.ts` | HTTP/SSE клиент |
| `frontend/src/hooks/useChatAgent.ts` | React hook для стриминга и состояния чата |
| `frontend/src/components/ChatPanel.tsx` | Основной UI чата с per-message активностью |
| `frontend/nginx.conf` | Reverse proxy: backend + Phoenix + SPA |
