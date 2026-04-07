# Диаграмма 2 — Жизненный цикл запроса (Sequence Diagram)

Детальный поток от нажатия кнопки до получения ответа. Для разработчиков и технических заказчиков.

Два параллельных async task-а: `run_agent` выполняет граф агента в отдельном треде,
`emit_live_reasoning` опрашивает коллекторы и кладёт события в общую Queue.
`event_generator` читает Queue и отдаёт SSE клиенту.

```mermaid
sequenceDiagram
    actor User as 👤 Пользователь
    participant UI as React Frontend
    participant API as FastAPI /query/stream
    participant Queue as asyncio.Queue
    participant RunAgent as run_agent()<br/>(thread)
    participant EmitReasoning as emit_live_reasoning()<br/>(async)
    participant Graph as LangGraph Graph
    participant LLM as LLM Provider
    participant Tool as Tool + Sandbox

    User->>UI: Вводит вопрос, нажимает отправить
    UI->>API: POST /api/sessions/{id}/query/stream

    API->>API: Проверяет JWT, загружает сессию<br/>и DataFrame, строит коллбэки

    par Параллельные задачи
        API->>RunAgent: asyncio.create_task(run_agent())
    and
        API->>EmitReasoning: asyncio.create_task(emit_live_reasoning())
    end

    API-->>UI: event: start {"session_id": "..."}

    RunAgent->>Graph: graph.invoke(state)

    Note over Graph: dispatch_node<br/>Keyword pre-check

    alt Быстрый маршрут (чат / саммари)
        Graph->>LLM: Промпт без инструментов
        LLM-->>Graph: Ответ
        Graph->>Graph: finalize_node (rewrite if needed)
    else Анализ данных
        Graph->>Graph: build_tools + sandbox → agent_node

        loop _direct_tool_loop (до max_iterations)
            Graph->>LLM: invoke с bind_tools(tools) + история
            LLM-->>Graph: stream токенов + tool_calls

            Graph-->>Queue: token "..." (видимый текст)
            Graph-->>Queue: reasoning_token "..." (внутри <think>)
            Graph-->>Queue: thinking_start / thinking_end

            alt LLM вернул tool_calls
                Graph->>Tool: tool.invoke(code)
                Tool->>Tool: AST-валидация → sandbox.execute()<br/>→ validate_contract()
                alt Ошибка выполнения
                    Tool->>LLM: _fix_with_llm(code, error)
                    LLM-->>Tool: исправленный код
                    Tool->>Tool: повторная попытка
                end
                Tool-->>Graph: артефакт (table/plot/value)
                Graph-->>Queue: tool_start {tool_name, input_preview}
                Graph-->>Queue: tool_end {tool_name, status, artifact_keys}
                Graph-->>Queue: execution_graph snapshot
            else tool_calls пуст — выход из цикла
                Graph->>Graph: выход из _direct_tool_loop
            end
        end

        Graph->>Graph: finalize_node<br/>rewrite generic text → artifact-grounded summary<br/>ReviewTool quality check (опционально)
    end

    Graph-->>RunAgent: AgentResponse
    RunAgent->>API: сохраняет историю + артефакты в SessionStore
    RunAgent-->>Queue: ("final", response_dict)
    RunAgent-->>Queue: ("done", None)

    loop Опрос коллекторов (каждые 20мс)
        EmitReasoning->>Queue: phase events → ("phase", event)
        EmitReasoning->>Queue: tool events → ("reasoning", текст)
        EmitReasoning->>Queue: progress events → ("reasoning", текст)
    end
    EmitReasoning->>EmitReasoning: agent_finished → дренаж коллекторов

    loop event_generator читает Queue
        API-->>UI: event: token / reasoning_token
        API-->>UI: event: thinking_start / thinking_end
        API-->>UI: event: tool_start / tool_end
        API-->>UI: event: phase {id, phase, title, status}
        API-->>UI: event: execution_graph {nodes, edges}
        API-->>UI: event: reasoning (live tool output)
    end

    API-->>UI: event: final {text, artifacts, metrics}
    API-->>UI: event: done

    UI-->>User: Текст + Таблицы + Plotly-графики
```
