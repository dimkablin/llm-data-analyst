# Диаграмма 10 — Семантический слой

Презентационная схема: как физические данные превращаются в бизнес-контекст для AI-аналитика.

```mermaid
flowchart LR
    subgraph sources["Источники данных"]
        files["CSV и XLSX"]
        databases["PostgreSQL и ClickHouse"]
    end

    subgraph ingestion["Получение схемы и профиля"]
        duckdb["Session DuckDB runtime"]
        dbhelper["DBAnalyticsHelper"]
        profiler["Ограниченное профилирование"]
        snapshot["DataCatalogSnapshot"]
    end

    subgraph semantic["Построение семантического слоя"]
        catalogService["SemanticCatalogService"]
        aiGeneration["AI-генерация описаний и метрик"]
        validation["Валидация и публикация"]
    end

    subgraph storage["Хранение"]
        postgres[("PostgreSQL JSONB — источник истины")]
        qdrant[("Qdrant — векторный индекс")]
    end

    subgraph analytics["Работа AI-аналитика"]
        question["Вопрос пользователя"]
        context["SemanticContextBuilder — top-K контекст"]
        agent["LangGraph AI-аналитик"]
        sql["SQL и аналитический результат"]
    end

    subgraph management["Управление из UI"]
        panel["Панель семантического слоя"]
        actions["Статус, refresh, AI generation, CRUD метрик"]
    end

    files --> duckdb
    databases --> dbhelper
    duckdb --> profiler
    dbhelper --> profiler
    profiler --> snapshot
    snapshot --> catalogService
    aiGeneration --> catalogService
    catalogService --> validation
    validation -->|"Полный каталог"| postgres
    validation -.->|"Таблицы, колонки, метрики и термины"| qdrant

    question --> context
    postgres -->|"Метрики, связи и правила"| context
    qdrant -.->|"Semantic search"| context
    context --> agent
    agent --> sql

    panel --> actions
    actions -->|"Обновление и редактирование"| catalogService
    postgres -->|"Статус и содержимое"| panel

    classDef source fill:#E8F1FF,stroke:#3B82F6,color:#102A43
    classDef process fill:#EEF2FF,stroke:#6366F1,color:#1E1B4B
    classDef store fill:#ECFDF5,stroke:#10B981,color:#064E3B
    classDef ai fill:#FFF7ED,stroke:#F97316,color:#7C2D12
    classDef ui fill:#FDF2F8,stroke:#EC4899,color:#831843

    class files,databases source
    class duckdb,dbhelper,profiler,snapshot,catalogService,aiGeneration,validation process
    class postgres,qdrant store
    class question,context,agent,sql ai
    class panel,actions ui
```

## Как объяснять на презентации

1. Файл загружается в session DuckDB runtime, внешняя БД читается через `DBAnalyticsHelper`.
2. Профилирование ограничено sample, timeout и лимитами, поэтому не выполняет полный scan больших источников.
3. `SemanticCatalogService` объединяет физическую схему, эвристики, пользовательские правки и AI-генерацию.
4. Полный каталог хранится в PostgreSQL; Qdrant содержит только индекс для semantic search.
5. На вопрос пользователя `SemanticContextBuilder` выбирает top-K объектов, после чего AI строит SQL с учётом метрик, синонимов и связей.
