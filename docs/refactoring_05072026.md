# Рефакторинг архитектуры агента

## Что было

Старый граф выполнения:

```mermaid
flowchart TD
    DEAD["🪦 DEAD — не подключено к графу\n_evaluate_node\n_decide_node\n_act_edge\n_decide_edge"]
    style DEAD fill:#3a1a1a,color:#ff6b6b,stroke:#ff6b6b,stroke-dasharray:5 5

    START --> think_node["think_node\nkeyword routing + tool building\n+ LLM-планировщик"]

    think_node -->|route == chat| finalize_node
    think_node -->|route == rag| rag_node["rag_node\nRAG service call"]
    think_node -->|route == summary| summary_node["summary_node\nLLM direct call"]
    think_node -->|route == None| act_node["act_node\nReAct engine\nИЛИ bind_tools\n⚠️ флаг agent_react_enabled"]

    rag_node --> finalize_node
    summary_node --> finalize_node
    act_node --> finalize_node

    finalize_node["finalize_node\ntext rewrite + review_tool"] --> END
```

- `_evaluate_node`, `_decide_node`, `_act_edge`, `_decide_edge` — мёртвый код,
  существовал как методы, но никогда не был подключён к скомпилированному графу
- `think_node` делал одновременно: keyword-routing, построение tools, вызов LLM-планировщика —
  три разные роли под одним именем
- `rag_node` и `summary_node` — отдельные узлы графа с дублирующей логикой
- Два конкурирующих execution engine: ReAct (`create_pandas_dataframe_agent`) и
  direct tool-calling (`bind_tools`) за флагом `agent_react_enabled`
- `agent_prompt` — мёртвый промпт (109 строк), не подключённый ни к одному узлу

## Что изменилось

| Было | Стало |
|------|-------|
| 7+ узлов графа, часть мёртвых | 3 узла: `dispatch → agent → finalize` |
| Два execution engine за флагом | Один: `bind_tools` / `_direct_tool_loop` |
| `agent_react_enabled` в config, БД, API | Флаг удалён из runtime полностью |
| System prompt собирался в 3 местах | Один метод `_build_execution_system_prompt` |
| `agent_prompt`, `_evaluate_node` и др. | Удалены вместе с 3 файлами legacy-агента |
| Skills-файлы не интегрированы в промпт | Инжектируются в system prompt через registry |

## Новая архитектура

```mermaid
flowchart TD
    START --> dispatch

    dispatch -->|chat / summary\nkeyword pre-check| finalize
    dispatch -->|analysis request| agent

    agent -->|load skill instructions| skills
    skills --> agent

    agent -->|call tool| tools
    tools -->|sandbox result / artifact| agent

    agent -->|no more tool calls| finalize
    finalize --> END

    skills["📚 Skills
    ───────────────
    planner · sql · database
    pandas · plotly · value
    search · forecast  · rag
    anomaly_planfact · review
    ───────────────
    💡 cohort_analysis"]

    tools["🛠 Tools
    ───────────────
    sql_tool · database_tool
    pandas_tool · value_tool
    plotly_tool · search_tool
    rag_tool · forecast_tool
    anomaly_planfact_tool
    planner_tool · review_tool"]

    style dispatch fill:#2d2d2d,color:#fff
    style agent fill:#1a3a5c,color:#fff
    style finalize fill:#2d2d2d,color:#fff
    style skills fill:#1a2d1a,color:#ccc,stroke:#4a7a4a
    style tools fill:#1a1a2d,color:#ccc,stroke:#4a4a7a
```

**`dispatch`** — детерминированный keyword pre-check (без LLM). Три лёгких bypass'а:
- `chat` — приветствия, вопросы о боте
- `summary` — управленческая записка по итогам сессии

Всё остальное → строит `tools`, `sandbox`, `capability_context` и передаёт в `agent`.

**`agent`** — единственный execution engine: нативный tool-calling через `bind_tools`.
Получает полный system prompt (policy + skills + data context) и гоняет tool loop
до тех пор, пока LLM не вернёт финальный текст без вызовов.

**`finalize`** — перезаписывает ответ если нужно, запускает `review_tool` для
проверки качества.

## Удалено

- `backend/agent/agent.py` — старый монолитный `Agent` класс (ReAct)
- `backend/agent/agent_callback.py` — legacy callback handlers
- `backend/agent/pandas_agent.py` — `create_pandas_dataframe_agent` и обвязка
- Мёртвые методы: `_evaluate_node`, `_decide_node`, `_act_edge`, `_decide_edge`,
  `_analysis_step`, `_rag_node`, `_summary_node`, `_think_system_prompt`
- Конфиг-флаги: `agent_react_enabled`, `agent_evaluate_enabled`, `agent_evaluate_max_tokens`
- Мёртвый промпт `agent_prompt` (109 строк)
- Null-паттерны `NoOpSkillFilter`, `NullSkillMatcher`, `NullSkillRanker`
