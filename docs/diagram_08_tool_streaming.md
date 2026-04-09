## Стриминг тул-вызовов на фронтенд

```mermaid
flowchart TD
    A([LLM вызывает tool]) --> B[LangChain callback\nToolCollector.on_tool_start]
    B --> B1["push_data: tool_name, input_preview,\ninput_summary, input_code"]
    B1 --> C[SSE queue\n_push_event tool_start]

    C --> D[event_generator\nasync SSE stream]
    D --> E[Браузер\nSSE Parser consumeSseLine]

    E --> F{META_TOOLS?}
    F -- get_tool_instructions --> G[skip — внутренний загрузчик скилов]
    F -- любой другой тул --> H[onToolStart handler]

    H --> H1["создаёт StreamToolCall\nstatus: running"]
    H1 --> H2[flushSync → React рендерит\nтул-карточку сразу]

    A2([Tool завершился]) --> B2[ToolCollector.on_tool_end]
    B2 --> B3["push_data: tool_name, status,\noutput_preview, artifact_keys"]
    B3 --> C2[SSE queue\n_push_event tool_end]
    C2 --> D
    D --> E2[onToolEnd handler]
    E2 --> E3["патчит StreamToolCall\nstatus: done / error\n+ output_preview"]
    E3 --> E4[React рендерит\nрезультат тул-карточки]

    subgraph "Live Reasoning текст (параллельно)"
        LR1[emit_live_reasoning task]
        LR1 --> LR2{tool_name?}
        LR2 -- planner_tool --> LR3["🗂 planner_tool составляет план: ...\n✅ план готов: {текст плана}"]
        LR2 -- review_tool --> LR4["🔍 review_tool проверяет ответ\n✅ проверка пройдена: ..."]
        LR2 -- get_tool_instructions --> LR5["📚 Загружаю инструкцию..."]
        LR2 -- остальные тулы --> LR6["### Live Tool #N\ntool_name запущен\n```code```"]
    end

    C --> LR1

    style G fill:#aaa,color:#fff
    style H2 fill:#5b9bd5,color:#fff
    style E4 fill:#4caf50,color:#fff
    style LR3 fill:#5b9bd5,color:#fff
    style LR4 fill:#5b9bd5,color:#fff
```

## Что видит пользователь для каждого тула

| Тул | Карточка в чате | input_summary | output_preview |
|-----|:--------------:|---------------|----------------|
| `planner_tool` | ✅ Да | вопрос пользователя | сгенерированный план |
| `review_tool` | ✅ Да | вопрос пользователя | результат проверки |
| `sql_tool` | ✅ Да | SQL-запрос | статус + артефакты |
| `pandas_tool` | ✅ Да | первая строка кода | статус + артефакты |
| `plotly_tool` | ✅ Да | первая строка кода | статус + артефакты |
| `get_tool_instructions` | ❌ Скрыт | — | — |

## Где были фильтры и что изменили

| Файл | Было | Стало |
|------|------|-------|
| `frontend/src/app/hooks/useChatAgent.ts:16` | `META_TOOLS = Set(["get_tool_instructions", "planner_tool", "review_tool"])` | `META_TOOLS = Set(["get_tool_instructions"])` |
| `frontend/src/app/hooks/useChatAgent.ts:24` | нет обработки `question` / `answer` | добавлены поля для planner / review |
| `backend/api/routes/query.py:562` | компактный emoji для planner+review | полноценный текст с вопросом и планом |
| `backend/agent/callbacks.py:261` | нет `question` / `answer` в `_build_input_summary` | добавлены |
