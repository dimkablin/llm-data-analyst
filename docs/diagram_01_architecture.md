# Диаграмма 1 — Общая архитектура системы

Высокоуровневый обзор компонентов для заказчиков и менеджеров.

```mermaid
graph TD
    User(["👤 Пользователь"])

    subgraph Browser["Браузер"]
        UI["React SPA\nЧат / Сессии / Графики / Настройки"]
    end

    subgraph Docker["Docker Compose"]
        Nginx["nginx\nПрокси + Статика\n:8603"]

        subgraph Backend["FastAPI Backend"]
            Auth["🔐 Auth\nJWT · SQLite\nпользователи · роли · настройки"]
            API["REST API + SSE Stream\n/sessions · /query/stream\n/data · /sources"]
            Agent["🤖 Agent-Centric Loop\ndispatch → agent → finalize"]

            subgraph Tools["🛠 Инструменты (Tool Catalog)"]
                BuiltIn["Встроенные\nPandas · Plotly · SQL\nDatabase · Value\nPlanner · Reviewer"]
                Integrations["Интеграции\nRAG · Search\nForecast · Anomaly"]
            end

            Sandbox["📦 Sandbox\nИзолированное выполнение кода\nAST-валидация · таймаут · LRU-кэш"]
            Sessions["💾 Session Store\nParquet · DataFrame LRU Cache\nистория чата · артефакты"]
        end

        Phoenix["🔭 Arize Phoenix\nLLM Observability\nтрейсы · спаны · OpenInference"]
    end

    subgraph LLM["LLM Provider (внешний)"]
        Model["Языковая модель\nOllama / vLLM / OpenAI-совместимый\ntool calling + streaming"]
    end

    User -->|"Вопрос / CSV / запрос к БД"| UI
    UI -->|"HTTPS"| Nginx
    Nginx -->|"/api/*"| API
    Nginx -->|"/phoenix/*"| Phoenix
    API --> Auth
    API --> Sessions
    API --> Agent
    Agent -->|"bind_tools + invoke"| Model
    Model -->|"токены + tool_calls (stream)"| Agent
    Agent --> Tools
    Tools --> Sandbox
    Agent -->|"SSE события через Queue"| API
    API -->|"SSE stream (text/event-stream)"| Nginx
    Nginx -->|"ответ + артефакты"| UI
    UI -->|"Текст · Таблицы · Plotly-графики"| User
    Agent -.->|"OpenInference трейсы"| Phoenix
```

Диаграмма состояний

```mermaid
flowchart TD
    CLIENT([Client HTTP])

    subgraph API["API Layer (FastAPI)"]
        EP_STREAM["POST /sessions/{id}/query/stream\nquery.py:823"]
        EP_QUERY["POST /sessions/{id}/query\nquery.py:795"]
        EP_EVAL["POST /sessions/{id}/evaluate\nquery.py:809"]
        EP_STREAM -->|"creates 2 async tasks\n+ asyncio.Queue"| RUN_AGENT
        EP_QUERY -->|"await _execute_query(persist=True)"| SYNC_RUN
        EP_EVAL -->|"await _execute_query(persist=False)"| SYNC_RUN
    end

    subgraph STREAM_TASKS["Streaming Tasks (concurrent)"]
        RUN_AGENT["run_agent()\nquery.py:898"]
        EMIT_REASONING["emit_live_reasoning()\nquery.py:1062\npoll phase_collector every 20ms"]
        EVENT_GEN["event_generator()\nquery.py:1093\nSSE: start/phase/execution_graph/final/done"]
        RUN_AGENT -->|"anyio.to_thread\nrun_query()"| RUN_QUERY
        EMIT_REASONING -->|"queue.put(phase/graph)"| EVENT_GEN
        RUN_AGENT -->|"queue.put(final/done)"| EVENT_GEN
    end

    subgraph SYNC_PATH["Sync Query Path"]
        SYNC_RUN["_execute_query()\nquery.py:600\nnew AgentRunner per call"]
        SYNC_RUN -->|"anyio.to_thread\nrun_query()"| RUN_QUERY
    end

    subgraph RUNNER["AgentRunner.run_query() runner.py:2536"]
        RUN_QUERY["run_query()\ncheck cache → graph.invoke()"]
        CACHE_CHECK{{"cache_allowed?\nsettings.agent_cache_enabled\n&& request_kind != 'stream'"}}
        RUN_QUERY --> CACHE_CHECK
        CACHE_CHECK -->|"HIT"| CACHE_RETURN["return cached AgentResponse"]
        CACHE_CHECK -->|"MISS → graph.invoke()"| GRAPH
    end

    subgraph GRAPH["LangGraph: StateGraph(AgentGraphState)"]
        DISPATCH["dispatch node\n_dispatch_node() :2039"]
        AGENT["agent node\n_agent_node() :2191"]
        FINALIZE["finalize node\n_finalize_node() :2337"]

        DISPATCH -->|"state.done == True\n(chat/summary route)"| FINALIZE
        DISPATCH -->|"state.done == False\n(analysis route)"| AGENT
        AGENT --> FINALIZE
        FINALIZE -->|"END"| DONE_STATE(["AgentResponse"])
    end

    subgraph DISPATCH_DETAIL["_dispatch_node internals"]
        QR{{"_quick_route(prompt)\n:1984"}}
        CHAT_BP["chat bypass\n:2070\ncalls self.chat()"]
        SUMMARY_BP["summary bypass\n:2092\ncalls _build_management_note()"]
        BUILD_CTX["build analysis context\n:2116\ntools, sandbox, capability_context"]

        QR -->|"== 'chat'"| CHAT_BP
        QR -->|"== 'summary'"| SUMMARY_BP
        QR -->|"None (analysis)"| BUILD_CTX
        CHAT_BP -->|"done=True"| FINALIZE
        SUMMARY_BP -->|"done=True"| FINALIZE
        BUILD_CTX -->|"done=False\n+ tools, sandbox"| AGENT
    end

    subgraph AGENT_DETAIL["_agent_node internals"]
        PRE_PLAN["pre-execute planner_tool\n:2222\nOUTSIDE iteration budget\nresult → system prompt suffix"]
        TOOL_LOOP["_direct_tool_loop()\n:1791\nmax_iterations = max_steps"]
        LOOP_ITER{{"tool_calls in response?"}}
        CALL_TOOL["invoke tool from tool_map\n:1917"]
        APPEND_MSGS["append response + ToolMessages"]
        BREAK_OUT["final_text extracted\nor artifacts_summary_text"]

        PRE_PLAN --> TOOL_LOOP
        TOOL_LOOP --> LOOP_ITER
        LOOP_ITER -->|"YES"| CALL_TOOL
        CALL_TOOL --> APPEND_MSGS
        APPEND_MSGS -->|"next iteration"| LOOP_ITER
        LOOP_ITER -->|"NO"| BREAK_OUT
        BREAK_OUT --> FINALIZE
    end

    subgraph TOOL_LOOP_SPECIAL["Special cases in tool loop"]
        EMPTY_AFTER_GTI["empty response after\nget_tool_instructions\n:1877\nnudge HumanMessage → continue"]
        MAX_REACHED["max_iterations reached\n:1923\nrecover from LLMTextCollector"]
        LLM_FAIL["_is_llm_transport_failure\n:1851\nreturn early, llm_unreachable=True"]
    end

    subgraph FINALIZE_DETAIL["_finalize_node internals"]
        LLM_UNREACH_EARLY{{"llm_unreachable\n&& response is None?"}}
        LLM_UNREACH_MID{{"response.llm_unreachable?"}}
        NO_RESP{{"response is None?"}}
        HAS_ARTIFACTS{{"response.artifacts?"}}
        REWRITE["_artifact_grounded_summary()\n:1546\nrewrite generic/plan text"]
        REVIEW["inline _ReviewTool().run()\n:2443\nnew LLM client each time"]
        FALLBACK_FINAL["_fallback_text() :1165"]

        LLM_UNREACH_EARLY -->|"YES → return LLM unavail"| DONE_STATE
        LLM_UNREACH_EARLY -->|"NO"| LLM_UNREACH_MID
        LLM_UNREACH_MID -->|"YES → return partial as-is"| DONE_STATE
        LLM_UNREACH_MID -->|"NO"| NO_RESP
        NO_RESP -->|"YES → fallback"| DONE_STATE
        NO_RESP -->|"NO"| HAS_ARTIFACTS
        HAS_ARTIFACTS -->|"should_rewrite check"| REWRITE
        REWRITE --> REVIEW
        REVIEW -->|"pass=false: append to reasoning"| DONE_STATE
        HAS_ARTIFACTS -->|"no artifacts"| REVIEW
        REVIEW -->|"empty text?"| FALLBACK_FINAL
    end

    subgraph SSE_EVENTS["SSE Event Types"]
        EV_START["start: session_id"]
        EV_PHASE["phase: think/act/finalize\n+ step_index, max_steps, status"]
        EV_GRAPH["execution_graph: DAG snapshot"]
        EV_FINAL["final: text+reasoning+artifacts+metrics"]
        EV_ERROR["error: SkillSelectionError detail\n⚠️ BUG: puts raw string not tuple"]
        EV_DONE["done: None (termination)"]
    end

    CLIENT --> API
    EVENT_GEN -->|SSE stream| CLIENT
    SYNC_PATH -->|"QueryResponse JSON"| CLIENT
```