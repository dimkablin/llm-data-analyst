# LLM Data Analyst

Интерактивная платформа для анализа данных на естественном языке.
Загрузите CSV, задайте вопрос — агент сам напишет код, построит графики и проверит результат.

---

## Возможности

- **Reason-Action агент** — цикл `think → act → evaluate → decide`, самокоррекция при ошибках, автоматический подбор инструментов
- **Три уровня анализа** — light / medium / deep, настраиваются в UI или через `.env`
- **Интерактивные графики** — Plotly с масштабированием, поворотом и экспортом
- **Активность по сообщениям** — трейсинг рассуждений и фаз ReAct для каждого ответа
- **Аутентификация** — регистрация, роли (admin/user), управление пользователями
- **Observability** — Arize Phoenix для трассинга LangChain/LangGraph
- **Экспорт** — HTML-отчёты с артефактами

---

## Технологический стек

| Компонент | Технология |
|-----------|------------|
| LLM | Qwen 3.5 через Ollama / vLLM (OpenAI-compatible API) |
| Агент | LangChain 1.0 + LangGraph |
| Backend | FastAPI, SSE-стриминг, SQLite (auth) |
| Frontend | React + TypeScript + Vite |
| Трассинг | Arize Phoenix |
| Деплой | Docker Compose, nginx |

---

## Быстрый старт (Docker)

```bash
cp .env.example .env
# Отредактируйте .env: LLM_MODEL_API_URL, пароль админа, порты

docker compose up --build -d
```

Единая точка входа — **frontend (nginx)**, который проксирует API-запросы к backend и Phoenix:

| Сервис | URL |
|--------|-----|
| Приложение | `http://localhost:${FRONTEND_PORT}` |
| Phoenix UI | `http://localhost:${FRONTEND_PORT}/phoenix/` |

Backend не выставлен наружу — доступен только через nginx во внутренней сети Docker.

### Порты

В `.env` задаются хостовые порты (внутренние фиксированы):

| Переменная | По умолчанию | Внутренний порт |
|------------|-------------|-----------------|
| `FRONTEND_PORT` | 8603 | 8603 |
| `PHOENIX_UI_PORT` | 6006 | 6006 |
| `PHOENIX_GRPC_PORT` | 4317 | 4317 |

### Зеркала (опционально)

Для сборки за корпоративным прокси — выставьте `USE_MIRROR=True` и укажите `PIP_INDEX_URL`, `NPM_REGISTRY` и т.д.

---

## Локальная разработка (без Docker)

```bash
# Backend
pip install -r backend/requirements.runtime.txt
uvicorn backend.app:app --host 0.0.0.0 --port ${BACKEND_PORT:-8000} --reload

# Frontend
cd frontend
npm install
npm run dev
```

Vite dev-сервер автоматически проксирует API-запросы к backend через настройки в `vite.config.ts`.

---

## API (основные эндпоинты)

| Метод | Путь | Описание |
|-------|------|----------|
| POST | `/auth/register` | Регистрация |
| POST | `/auth/login` | Авторизация |
| GET | `/auth/me` | Текущий пользователь |
| GET/PATCH | `/auth/settings` | Настройки пользователя (температуры, токены, глубина анализа) |
| POST | `/sessions` | Создать сессию |
| POST | `/sessions/{id}/data` | Загрузить CSV (лимит `BACKEND_MAX_DATASET_MB`) |
| POST | `/sessions/{id}/query/stream` | SSE-стрим: токены, reasoning, фазы, финальный payload |
| GET | `/sessions/{id}` | Состояние сессии |
| GET/POST/PATCH/DELETE | `/admin/users` | Управление пользователями (admin) |

---

## Конфигурация (.env)

Полный список переменных — в [.env.example](./.env.example).

Ключевые группы:
- **LLM** — провайдер, модель, URL, температуры, top_p/top_k, max_tokens, thinking mode
- **Agent** — лимит шагов, таймауты, кеш, глубина анализа, evaluate
- **Backend** — TTL сессий, лимит файлов, CORS, хранилище
- **Auth** — путь к БД, TTL токенов, admin-пароль
- **Phoenix** — вкл/выкл, проект, auto-instrument

---

## Архитектура

Подробности — в [TECHNICAL.md](./TECHNICAL.md).

```
Browser → nginx (frontend) → backend (FastAPI) → LLM (Ollama/vLLM)
                ↓                    ↓
           Phoenix UI          Phoenix Collector
```

---

## Сброс состояния Phoenix

```bash
./scripts/reset_phoenix_state.sh
```
