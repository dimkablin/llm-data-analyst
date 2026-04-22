# llm-data-mas — Phase 1 Design & Implementation Prompt

**Дата:** 2026-04-22
**Цель Phase 1:** Walking skeleton мультиагентной системы для data analysis — базовый каркас без бизнес-логики, с основными tools (Python sandbox + SQL + DuckDB) и полным MAS-графом.

---

## 1. Архитектурные решения (результат брейншторма)

| Решение | Выбор | Обоснование |
|---|---|---|
| Проект | Новый `llm-data-mas` рядом с `llm-data-analyst-dev` | Чистый старт, dev остаётся референсом |
| LLM провайдер | Ollama + vLLM (OpenAI-compatible) | Преемственность, локальные модели |
| Паттерн MAS | **Supervisor (Plan + Send) + Critic с re-planning** | One-shot plan для простых запросов, re-plan через critic для итеративных |
| Sub-agents | 3 типа: `data_agent`, `viz_agent`, `context_agent` | Специализация по функциональности |
| Параллелизм | LangGraph `Send` API (map-reduce) | Нативный, полная observability |
| Sub-agent реализация | `create_react_agent` из `langgraph.prebuilt` | Не писать свой ReAct-цикл |
| Critic | Hybrid: heuristics → LLM | Быстро для очевидных проблем |
| Retry стратегия | Два счётчика: `revision_count` (synthesis), `replan_count` (re-plan) | Revise для формулировки, replan для логики |
| Checkpointer | `PostgresSaver` | Production-ready |
| Artifact store | Filesystem + JSON (порт из dev) | Переиспользование существующего |
| Context management | `trim_messages` с `start_on="human"`, `end_on=("human","tool")` | Не рвёт tool_call пары |
| Skills | Tool `apply_skill(name)`, не system prompt override | Чёткий контракт |

---

## 2. Поток Plan + Send + Replan

```
START
  │
  ▼
plan_node ←──────────────────── (replan loop)
  │  (при replan: видит plan_history, subtask_results, critique.replan_reason)
  │
  ▼
route_subtasks → [Send × N независимых SubTask по depends_on]
  │
  ├─ data_agent_node ─┐
  ├─ viz_agent_node ──┤  (ReAct через create_react_agent)
  └─ context_agent_n ─┘
           │
           ▼
      synthesize_node ←─── (revise loop)
           │
           ▼
        critic_node (heuristics → LLM)
           │
   ┌───────┼──────────────────┐
   ok    revise_synthesis   replan
   │        (<1 раз)        (<1 раз)
   ▼        │                 │
  END   synthesize          plan
```

**Ключевой инвариант:** `artifacts` и успешные `subtask_results` сохраняются между replan. Supervisor не переделывает что уже работает — только добавляет новое.

---

## 3. State

```python
class GraphState(TypedDict):
    messages: list
    question: str
    plan: Plan | None
    plan_history: list[Plan]                        # все предыдущие планы
    subtask_results: dict[str, SubTaskResult]       # копится между replan
    artifacts: list[str]                            # artifact_ids, копится
    synthesis: Synthesis | None
    critique: Critique | None
    revision_count: int                             # synthesis revisions
    replan_count: int                               # re-plan iterations
```

```python
class SubTask(BaseModel):
    id: str
    agent: Literal["data_agent", "viz_agent", "context_agent"]
    question: str
    skill: str | None
    depends_on: list[str] = []

class Plan(BaseModel):
    subtasks: list[SubTask]                         # max 5
    reasoning: str

class SubTaskResult(BaseModel):
    artifact_ids: list[str]
    summary_text: str
    status: Literal["ok", "failed"]

class Synthesis(BaseModel):
    answer: str
    referenced_artifacts: list[str]

class Critique(BaseModel):
    ok: bool
    issues: list[str]
    required_fixes: list[str]
    action: Literal["ok", "revise_synthesis", "replan"]
    replan_reason: str | None = None
```

---

## 4. Implementation Prompt для кодового агента

> Промпт ниже следует подавать кодовому агенту для реализации Phase 1. Он работает поэтапно: одна фаза = одна остановка с отчётом.

---

### Роль
Senior Python-инженер. Пишешь новый проект с нуля на чистом LangGraph. Работаешь итеративно: одна фаза = одна остановка с отчётом. Не начинай следующую фазу без "ok".

### Контекст
- Источник логики: `C:\Users\dimka\Documents\PROJECTS\llm-data-analyst\llm-data-analyst-dev`
- Целевой проект: `C:\Users\dimka\Documents\PROJECTS\llm-data-analyst\llm-data-mas`
- Stack: Python 3.12, uv, LangGraph, Pydantic v2, Postgres (checkpointer)
- LLM: vLLM и Ollama (OpenAI-compatible, выбор через config)
- Artifact store: filesystem + JSON (портировать из dev)
- OS: Windows, PowerShell

### Архитектурные инварианты (нарушать нельзя)
1. Используем LangGraph как есть: `StateGraph`, `Send` API, `PostgresSaver`. Никаких самописных runtime.
2. Иерархия: `supervisor` → `[data_agent | viz_agent | context_agent]` → `critic` → opt. replan/revise.
3. Supervisor не делает ReAct — только structured output (`Plan`).
4. Sub-agents — полноценные ReAct через `create_react_agent` из `langgraph.prebuilt`.
5. Critic — гибрид: heuristics первыми, LLM-review если heuristics прошли.
6. Параллелизм sub-agents — `Send` API из плана supervisor.
7. Skills — **tools**, не system prompt override. Tool `apply_skill(skill_name)` возвращает playbook.
8. Artifacts не попадают в LLM-контекст целиком — только `artifact_id` + summary.
9. Один тонкий LLM client с переключением backend (vLLM/Ollama) через config. Никаких registry/router.
10. `subagents/*/agent.py` не импортирует `supervisor/` или `critic/`.
11. `tools/*` не импортирует LLM — tools детерминированы.
12. `mypy --strict` проходит на новом коде.
13. Пути через `pathlib.Path`, никаких строковых конкатенаций (Windows).
14. Свой ReAct-цикл не писать.
15. Для каждого sub-agent обязателен **adapter-node** в главном графе: принимает `GraphState`, формирует вход под `state_schema` sub-agent'а, вызывает `.ainvoke()`, извлекает результат в `SubTaskResult`.
16. `InjectedState` — ТОЛЬКО для сессионных данных (artifact_ids, промежуточные результаты). DB engine, artifact store, HTTP clients — через factory-замыкание. Connection strings и секреты в state НЕ попадают.
17. Компрессия истории — `trim_messages` из `langchain_core.messages` с обязательными `start_on="human"` и `end_on=("human","tool")`.
18. НЕ использовать `langgraph-supervisor` и `langgraph-swarm`. Наш supervisor кастомный (Plan + Send + replan).
19. **Replan loop**: critic может вернуть в `plan_node` (не только `synthesize_node`) через `action="replan"`. Максимум одна re-plan итерация.
20. Успешные `subtask_results` и `artifacts` сохраняются между replan — supervisor их переиспользует.

### Целевая структура

```
llm-data-mas/
  pyproject.toml
  .env.example
  README.md

  app/
    main.py
    graph.py
    config.py

    supervisor/
      node.py              # plan_node + synthesize_node
      prompts.py
      schema.py            # Plan, SubTask, Synthesis

    subagents/
      data_agent/
        agent.py           # create_react_agent
        prompts.py
        tools/
          sql.py
          catalog.py
          transform.py
          metrics.py
          duckdb.py
      viz_agent/
        agent.py
        prompts.py
        tools/chart.py
      context_agent/
        agent.py
        prompts.py
        tools/
          web.py
          rag.py           # заглушка

    critic/
      node.py
      heuristics.py
      prompts.py
      schema.py            # Critique

    skills/
      loader.py
      registry.py
      tool.py
      files/               # MD из dev

    state/
      schema.py            # GraphState + SubTaskResult
      checkpointer.py
      artifacts.py         # порт из dev

    llm/
      client.py
      structured.py

    db/
      readonly.py
      safety.py

  tests/
    unit/
    integration/
```

### Фазы

#### Фаза 0 — Инвентаризация dev-проекта (только чтение)
1. Прочитай dev-проект.
2. Составь список: tool-функции с сигнатурами, artifact store API, SQL safety правила, skills формат, config/ENV.
3. Таблица `src_path → target_module`.
4. Что НЕ переносим и почему.
5. СТОП. Жду "ok фаза 1".

#### Фаза 1 — Инициализация
1. `uv init`, Python 3.12.
2. Зависимости: `langgraph`, `langchain-core`, `pydantic`, `pydantic-settings`, `psycopg[binary]`, `langgraph-checkpoint-postgres`, `sqlalchemy`, `pandas`, `duckdb`, `plotly`, `httpx`, `openai`, `mypy`, `pytest`, `pytest-asyncio`, `ruff`.
3. Дерево `__init__.py`.
4. `.env.example`: `LLM_BACKEND`, `LLM_BASE_URL`, `LLM_MODEL`, `POSTGRES_DSN`, `ARTIFACTS_DIR`, `USER_DB_DSN`.
5. `config.py` через `pydantic-settings`.
6. README с командами.
7. Acceptance:
   - [ ] `uv sync` проходит
   - [ ] `uv run python -c "from app.config import settings; print(settings)"` работает
   - [ ] `ruff check`, `mypy app` зелёные
8. СТОП.

#### Фаза 2 — LLM client + structured output
1. `llm/client.py`: `get_client() -> OpenAI` (async), переключение vllm/ollama.
2. `llm/structured.py`: `call_structured[T](prompt, schema: type[T], max_retries=2) -> T`.
   - Первая попытка: `response_format={"type": "json_schema", ...}`.
   - Retry на parse fail с сообщением "твой ответ не валидный JSON: {error}".
   - Fallback: extract JSON из markdown code fences.
3. Unit-тесты с замоканным HTTP.
4. Acceptance: тесты зелёные, работает с обоими backend.
5. СТОП.

#### Фаза 3 — State + Checkpointer + Artifact Store
1. `state/schema.py`:
   ```python
   class GraphState(TypedDict):
       messages: list
       question: str
       plan: Plan | None
       plan_history: list[Plan]
       subtask_results: dict[str, SubTaskResult]
       artifacts: list[str]
       synthesis: Synthesis | None
       critique: Critique | None
       revision_count: int
       replan_count: int
   ```
   Reducer для `plan_history` — append (`operator.add`). Для `subtask_results` и `artifacts` — merge/append соответственно.
2. `state/checkpointer.py`: factory для `PostgresSaver`.
3. `state/artifacts.py`: порт из dev. Минимум: `save_artifact`, `load_artifact`, `get_summary`.
4. Acceptance:
   - [ ] Checkpointer поднимается на Postgres
   - [ ] Artifact round-trip тест
5. СТОП.

#### Фаза 4 — Tools + DB safety
1. `db/readonly.py`: read-only SQLAlchemy engine.
2. `db/safety.py`: валидация SQL (запрет DDL/DML, auto-LIMIT, таймаут).
3. Tools по файлам. Каждый возвращает `artifact_id` или компактный скаляр. `@tool` из langchain-core.
4. Порт логики из dev под новую artifact-сигнатуру.
5. Заглушки: `rag.py` возвращает "RAG service not available".
5a. Factory: `make_<tool>_tool(engine, artifact_store, ...) -> Tool`. Инфра — в замыкании.
5b. Для artifact_ids текущей сессии — `Annotated[SpecialistState, InjectedState]`.
6. Acceptance:
   - [ ] Unit happy path + edge case для каждого tool
   - [ ] SQL safety блокирует DDL/DML — тест
7. СТОП.

#### Фаза 5 — Sub-agents через create_react_agent
1. Для каждого sub-agent:
   - `prompts.py` — узкоспециализированный system prompt
   - `agent.py`:
     ```python
     def build_data_agent(llm, artifact_store, engine):
         tools = [make_sql_tool(...), make_catalog_tool(...), ...]
         return create_react_agent(
             model=llm,
             tools=tools,
             state_schema=DataAgentState,
             prompt=SYSTEM_PROMPT,
         )
     ```
   - State-schema: `messages`, `artifact_ids: list[str]`, `subtask: SubTask`.

2. Adapter-node в главном графе:
   ```python
   async def data_agent_node(state: GraphState) -> dict:
       subtask = pick_my_subtask(state, agent="data_agent")
       sub_state = {
           "messages": [HumanMessage(subtask.question)],
           "artifact_ids": [],
           "subtask": subtask,
       }
       sub_state["messages"] = trim_messages(
           sub_state["messages"],
           max_tokens=settings.subagent_max_tokens,
           token_counter=llm,
           strategy="last",
           include_system=True,
           start_on="human",
           end_on=("human", "tool"),
       )
       result = await agent.ainvoke(sub_state, config={"recursion_limit": 10})
       return {
           "subtask_results": {
               subtask.id: SubTaskResult(
                   artifact_ids=result["artifact_ids"],
                   summary_text=extract_final_text(result["messages"]),
                   status="ok",
               )
           }
       }
   ```

3. `recursion_limit` через `.ainvoke(..., config={"recursion_limit": 10})`.

4. Acceptance:
   - [ ] Каждый sub-agent через `create_react_agent`, свой ReAct не написан
   - [ ] Adapter-node изолирует `GraphState` от `state_schema` sub-agent'а
   - [ ] Integration-тест каждого sub-agent на заглушенной БД
   - [ ] Recursion limit срабатывает — тест на залипание
   - [ ] `trim_messages` не ломает tool_call/tool_result пары — тест
5. СТОП.

#### Фаза 6 — Skills
1. `skills/files/` — копия MD из dev.
2. `skills/registry.py` — `{name: SkillMeta(path, description, target_agents)}` из Фазы 0.
3. `skills/loader.py::load_skill(name) -> str`.
4. `skills/tool.py::make_apply_skill_tool(agent_name)` — factory, возвращает `@tool`, отдаёт только skills с `agent_name` в `target_agents`.
5. Supervisor получает `apply_skill` видящий ВСЕ skills.
6. Acceptance:
   - [ ] `apply_skill("auto_eda")` возвращает MD content
   - [ ] Фильтрация по target_agents работает
7. СТОП.

#### Фаза 7 — Supervisor (Plan + Synthesize + Replan)
1. `supervisor/schema.py`:
   ```python
   class SubTask(BaseModel):
       id: str
       agent: Literal["data_agent", "viz_agent", "context_agent"]
       question: str
       skill: str | None
       depends_on: list[str] = []

   class Plan(BaseModel):
       subtasks: list[SubTask]   # max 5
       reasoning: str

   class Synthesis(BaseModel):
       answer: str
       referenced_artifacts: list[str]
   ```
2. `supervisor/node.py`:
   - `plan_node(state)`:
     - Если `replan_count == 0`: обычный планинг на основе `question`.
     - Если `replan_count > 0` (replan): промпт включает `plan_history`, `subtask_results` (с summary артефактов), `critique.replan_reason`.
     - Supervisor **не дублирует** успешные SubTask из прошлых раундов.
     - Новые SubTask формулируются с конкретикой из `subtask_results` (пример: вместо "объясни аномалии" → "объясни падение -43% в регионе Север 15-17 марта").
     - Возвращает: `{"plan": new_plan, "plan_history": [new_plan], "replan_count": replan_count + 1}`.
   - `route_subtasks(state)` → `[Send(agent_node, state_for_subtask) for st in plan.subtasks]`. Учитывает `depends_on` — зависимые запускаются после завершения upstream.
   - `synthesize_node(state)` → structured → `Synthesis`. При `revision_count > 0` промпт включает `critique.required_fixes`.
3. Acceptance:
   - [ ] Для "сделай EDA" план: data_agent + viz_agent с правильным skill
   - [ ] Независимые SubTask запускаются параллельно (по таймстемпам)
   - [ ] При replan supervisor получает `plan_history` и `subtask_results` в промпт — тест формулировки SubTask с конкретикой
   - [ ] Успешные `subtask_results` из раунда 1 сохраняются после replan — тест
4. СТОП.

#### Фаза 8 — Critic (heuristics + LLM + replan routing)
1. `critic/heuristics.py`:
   ```python
   class HeuristicResult(BaseModel):
       passed: bool
       action_hint: Literal["ok", "revise_synthesis", "replan"]
       issues: list[str]

   def check_replan_needed(state) -> HeuristicResult:
       """Проблемы которые revise НЕ исправит."""
       if not state["artifacts"]:
           return HeuristicResult(False, "replan", ["no artifacts — plan failed"])
       if all(r.status == "failed" for r in state["subtask_results"].values()):
           return HeuristicResult(False, "replan", ["all subtasks failed"])
       for st_id, result in state["subtask_results"].items():
           subtask = find_subtask(state["plan"], st_id)
           if subtask.depends_on and result.summary_text.startswith("no data"):
               return HeuristicResult(False, "replan",
                   [f"subtask {st_id} failed due to upstream empty result"])
       return HeuristicResult(True, "ok", [])

   def check_synthesis_quality(state) -> HeuristicResult:
       """Проблемы которые revise исправит."""
       if not synthesis_mentions_artifacts(state):
           return HeuristicResult(False, "revise_synthesis", ["synthesis ignores artifacts"])
       # ...
   ```
2. `critic/node.py`:
   ```python
   async def critic_node(state):
       # 1. Replan-эвристики первыми (дорогие проблемы)
       replan_check = check_replan_needed(state)
       if not replan_check.passed:
           return {"critique": Critique(
               ok=False, action="replan",
               issues=replan_check.issues,
               replan_reason=generate_replan_reason(state, replan_check),
               required_fixes=[],
           )}
       # 2. Синтез-эвристики
       synth_check = check_synthesis_quality(state)
       if not synth_check.passed:
           return {"critique": Critique(
               ok=False, action="revise_synthesis",
               issues=synth_check.issues,
               required_fixes=[...],
           )}
       # 3. LLM review
       llm_critique = await call_structured(critic_prompt, Critique)
       return {"critique": llm_critique}
   ```
3. `critic/schema.py`:
   ```python
   class Critique(BaseModel):
       ok: bool
       issues: list[str]
       required_fixes: list[str]
       action: Literal["ok", "revise_synthesis", "replan"]
       replan_reason: str | None = None
   ```
4. Роутинг:
   ```python
   def route_after_critic(state):
       c = state["critique"]
       if c.action == "ok":
           return END
       if c.action == "revise_synthesis" and state["revision_count"] < 1:
           return "synthesize"
       if c.action == "replan" and state["replan_count"] < 1:
           return "plan"
       return END   # degraded
   ```
5. Acceptance:
   - [ ] Heuristics покрыты unit-тестами
   - [ ] `check_replan_needed` триггерится на `status == "failed"` и `depends_on` + empty upstream
   - [ ] Revision loop ограничен 1 итерацией
   - [ ] Replan loop ограничен 1 итерацией
   - [ ] После replan `artifacts` и успешные `subtask_results` сохранены
6. СТОП.

#### Фаза 9 — Сборка графа + main
1. `graph.py`:
   ```python
   builder = StateGraph(GraphState)
   builder.add_node("plan", plan_node)
   builder.add_node("data_agent", data_agent_node)
   builder.add_node("viz_agent", viz_agent_node)
   builder.add_node("context_agent", context_agent_node)
   builder.add_node("synthesize", synthesize_node)
   builder.add_node("critic", critic_node)

   builder.add_edge(START, "plan")
   builder.add_conditional_edges("plan", route_subtasks)
   builder.add_edge("data_agent", "synthesize")
   builder.add_edge("viz_agent", "synthesize")
   builder.add_edge("context_agent", "synthesize")
   builder.add_edge("synthesize", "critic")
   builder.add_conditional_edges("critic", route_after_critic)
   ```
2. `main.py`: CLI, стрим через `astream_events`.
3. Acceptance:
   - [ ] E2E "покажи структуру БД" (только data_agent)
   - [ ] E2E "сделай EDA таблицы X" (data + viz)
   - [ ] E2E "проанализируй аномалии продаж и найди объяснения" (replan triggers — context_agent получает конкретику из data_agent результата на втором раунде)
   - [ ] Checkpoint: прервать → resume → дойти до конца
4. СТОП. Демо.

### Anti-patterns (не делать)
- Свои абстракции поверх LangGraph — сначала спросить.
- Planner/executor/critic как отдельные runtime слои. Они ноды.
- Registry/router для LLM. Один client.
- Forecast/anomaly_detection — Phase 10+.
- Рефакторить dev-проект. Берём логику, переносим.
- Зависимости без подтверждения.
- Больше кода чем нужно для acceptance фазы.
- Свой ReAct-цикл.
- DB engine, connection strings, API keys в state. Factory при сборке tools.
- `langgraph-supervisor` / `langgraph-swarm`.
- `trim_messages(strategy="last")` без `start_on`/`end_on`.
- Перекладывать replan-решения только на LLM-critic. Heuristics — первый уровень классификации.
- Supervisor дублирует успешные SubTask при replan. Читать `subtask_results` перед генерацией плана.

### Формат отчёта после каждой фазы
1. Созданные/изменённые файлы с количеством строк.
2. `uv run ruff check`, `uv run mypy app`, `uv run pytest -q` — результаты.
3. Checklist acceptance с ✓/✗.
4. Что взял из dev и как адаптировал (фазы 3, 4, 6).
5. Вопросы, если неясно.
6. НЕ продолжать следующую фазу без "ok".

### Если что-то неясно
Не угадывай. Остановись и спроси. Особенно:
- Публичный API artifact store из dev.
- Формат MD-файлов skills и триггеры.
- Таблица/БД для integration-тестов.
- Как supervisor получает `plan_history` и `subtask_results` в промпт при replan (формат).

---

## 5. Что не входит в Phase 1 (на будущее)

- **Phase 2:** external tools (web search, полноценный RAG, forecast, anomaly detection)
- **Phase 3:** продвинутая skill-система (динамическое триггеривание, композиция skills)
- **Phase 4:** интеграция в frontend с визуализацией работы sub-agents и thinking перед вызовом tools

---

## 6. Ключевые проверки на E2E (фаза 9)

| Запрос | Ожидаемое поведение |
|---|---|
| "покажи структуру БД" | Один SubTask → data_agent → synthesize → ok |
| "сделай EDA таблицы X" | Два SubTask параллельно (data + viz) → synthesize → ok |
| "проанализируй аномалии продаж и найди объяснения" | Раунд 1: data_agent находит аномалию, context_agent падает на размытом depends_on вопросе → critic detects → **replan** с конкретикой → раунд 2: context_agent отрабатывает → synthesize → ok |
| Прерывание процесса | Resume через checkpoint → продолжение с последней ноды |
