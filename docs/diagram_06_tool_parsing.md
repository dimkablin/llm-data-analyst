# Tool Output Parsing — Flow Schemas

> Документ описывает точный путь данных для каждого инструмента:
> LLM-вызов → выполнение → парсинг → `ToolCollector` → `ExecutionArtifact` → фронтенд.

---

## Общий контракт (LangChain 1.0.0)

```
tool._run(args)  →  (content_str, artifact_dict)   [response_format="content_and_artifact"]
                 →  content_str                     [response_format="content"]
                         │
                         ▼
         LangChain: _format_output(content, artifact, ...)
                         │
                         ▼
              ToolMessage(content=str, artifact=dict|None)
                         │
                         ▼
         ToolCollector.on_tool_end(ToolMessage)
                         │
                ┌────────┴────────┐
                ▼                 ▼
   _normalize_output(ToolMessage)  →  returns ToolMessage.artifact (if dict)
                │                     or parses ToolMessage.content as JSON
                ▼
        payload: dict
   ┌────────────────────────────┐
   │ "plot"  → ExecutionArtifact(PLOT)      │
   │ "table" → ExecutionArtifact(DATAFRAME) │
   │ "value" → ExecutionArtifact(SCALAR)    │
   │ "json"  → ExecutionArtifact(JSON)      │
   └────────────────────────────┘
```

---

## 1. `sql_tool`

**Тип:** `BaseTool` (не `BaseExecTool`)
**Артефакт:** `table` → `ExecutionArtifact(DATAFRAME)`

```
LLM → tool_call: {question: "..."}
          │
          ▼
  SQLTool._run_query(question)
          │
          ▼
  SQLTableService.build_table_artifact(question)
  ┌─────────────────────────────────────────────────┐
  │  LLM генерирует SQL (safe SELECT only)          │
  │  SQL → DuckDB/PostgreSQL → pd.DataFrame          │
  │  Returns: {                                      │
  │    "schema_version": "1.0",                      │
  │    "artifact_type": "table",                     │
  │    "items": {name: DataFrame},                   │
  │    "source": {...},                              │
  │    "recipe": {"sql": "..."},                     │
  │  }                                               │
  └─────────────────────────────────────────────────┘
          │
          ▼
  Inject DataFrames into sandbox (sandbox.put(name, df))
  → переменные доступны для pandas_tool/plotly_tool
          │
          ▼
  Returns: (text_str, {
    "text": "✅ ...",
    "table": {name: DataFrame},  ← прямой ключ, без "items"
    "source": {...},
    "recipe": {...},
  })
          │
  ─────────────────────────────────────────────────
  LangChain → ToolMessage(content=text_str, artifact=payload)
          │
          ▼
  on_tool_end: "table" in payload → True
  → ExecutionArtifact(DATAFRAME, name=name, data=DataFrame) ✓

  ПАРСИНГ: ✓ корректно
  ФОРМАТ:  ⚠️ нестандартный (прямой "table" вместо "items"),
           но on_tool_end это обрабатывает
```

---

## 2. `pandas_tool`

**Тип:** `BaseExecTool`
**Артефакт:** `table` → `ExecutionArtifact(DATAFRAME)`
**LLM генерирует:** Python код

```
LLM → tool_call: {code: "df_result = df.groupby(...)..."}
          │
          ▼
  BaseExecTool._run(code)
  │
  ├── strip_thinking(code)          ← убирает <think>...</think>
  ├── normalize_code(code)
  ├── cache_get(cache_key)           ← LRU кэш
  ├── validate_libraries(code)
  │     allowed: {pandas, numpy}
  │     AST-анализ импортов
  ├── validate_code_patterns(code)
  │     запрещено: matplotlib, .plot(), open/eval/exec,
  │                os/sys/subprocess, pd.read_csv/read_excel
  │
  └── for attempt in range(1 + 3):   ← до 3 LLM-ретраев
        _try_run_once(code)
        │
        ├── sandbox.execute(code)    ← изолированный namespace
        │     expects: tool_result = {"table": {name: df}}
        │               или tool_result = {name: df}
        │
        ├── _validate_tool_contract(tool_result)
        │   ┌───────────────────────────────────┐
        │   │ Input dict variants:              │
        │   │  A) {name: df}                    │
        │   │     → items = raw_result           │
        │   │  B) {"table": {name: df}}         │
        │   │     → items = {name: df}           │
        │   │  C) {"schema_version":"1.0",       │
        │   │      "artifact_type":"table",      │
        │   │      "items": {name: df}}          │
        │   │     → items = {name: df}           │
        │   │                                    │
        │   │ artifact_type: "table"/"dataframe"/│
        │   │   "df" → нормализуется в "table"  │
        │   │ schema_version: "1.0"              │
        │   │ Validates: artifact_name == "table"│
        │   └───────────────────────────────────┘
        │
        ├── post_process_tool_result(items)
        │   → normalize names (max 48 chars)
        │   → _round_numeric_table (4 знака для float)
        │   → handles pd.Series
        │
        ├── validate_tool_result(items)
        │   → isinstance(v, (pd.DataFrame, pd.Series))
        │
        └── on success: payload = {
              "text": "✅ ...",
              "code": code,
              "table": {name: df},  ← из normalized_result
            }

  ─────────────────────────────────────────────────
  on_tool_end: "table" in payload → DATAFRAME artifact ✓

  ПАРСИНГ: ✓ полностью валидируется
  ОШИБКИ:  → LLM ретрай с описанием ошибки (до 3 раз)
```

---

## 3. `plotly_tool`

**Тип:** `BaseExecTool`
**Артефакт:** `plot` → `ExecutionArtifact(PLOT)`
**LLM генерирует:** Python код

```
LLM → tool_call: {code: "fig = px.bar(df, ...)
                          tool_result = chart.result(fig, 'my_chart')"}
          │
          ▼
  sandbox scope содержит:
    chart = ChartArtifactHelper()
    df    = активный DataFrame
    (+ sandbox-переменные от sql_tool)
          │
          ▼
  BaseExecTool._run(code) → _try_run_once(code)
  │
  ├── sandbox.execute(code)
  │   LLM вызывает: chart.result(fig, "name")
  │   ChartArtifactHelper.result() возвращает:
  │   {
  │     "schema_version": "1.0",
  │     "artifact_type": "plot",
  │     "items": {"name": go.Figure},   ← items ключ!
  │     "source": {...},
  │     "recipe": [...],
  │   }
  │
  ├── _validate_tool_contract(tool_result)
  │   raw_items = tool_result.get("items")  → {"name": fig}
  │   artifact_type = "plot"
  │   aliases: "plotly"/"graph"/"figure"/"chart" → "plot"
  │   validates: artifact_name == "plot" ✓
  │
  ├── post_process_tool_result(items)
  │   → normalize names only (no rounding)
  │
  ├── validate_tool_result(items)
  │   → isinstance(v, go.Figure)
  │
  ├── apply_default_chart_style(fig) (внутри chart.result)
  │   → dark theme, colorway, margins
  │
  └── payload = {
        "text": "✅ ...",
        "code": code,
        "plot": {"name": go.Figure},
        "source": {...},
        "recipe": [...],
      }

  ─────────────────────────────────────────────────
  on_tool_end: "plot" in payload → PLOT artifact ✓

  ПАРСИНГ: ✓ стандартный envelope через "items"
  СТИЛЬ:   тема применяется до сохранения артефакта
```

---

## 4. `value_tool`

**Тип:** `BaseExecTool`
**Артефакт:** `value` → `ExecutionArtifact(SCALAR)`
**LLM генерирует:** Python код

```
LLM → tool_call: {code: "total = df['revenue'].sum()
                          tool_result = {'value': {'total_revenue': total}}"}
          │
          ▼
  BaseExecTool._run(code) → _try_run_once(code)
  │
  ├── sandbox.execute(code)
  │   expects: tool_result = {"value": {name: scalar}}
  │             или {name: scalar}
  │
  ├── _validate_tool_contract(tool_result)
  │   aliases: "metric"/"metrics"/"values" → "value"
  │   artifact_name == "value" ✓
  │
  ├── post_process_tool_result(items)  [ValueTool override]
  │   Для каждого (name, value):
  │   ├── dict → _flatten_metric_mapping(prefix=name, depth=0..2)
  │   │          вложенные dicts флаттенятся через "_"
  │   │          списки скаляров ≤8 элементов → name_1, name_2...
  │   ├── list/tuple → аналогично
  │   └── scalar → _normalize_scalar(value)
  │                  np.generic → .item()
  │                  float → round(v, 4), -0.0 → 0.0
  │                  str → .strip()
  │
  ├── validate_tool_result(items)  [ValueTool override]
  │   ├── isinstance(v, (float, int, str, bool, np.generic))
  │   └── str validation:
  │       max_len=160, max_sentences=2, no newlines
  │       ⚠️ Длинные/объяснительные строки → reject + retry
  │
  └── payload = {
        "text": "✅ ...",
        "code": code,
        "value": {name: scalar},
      }

  ─────────────────────────────────────────────────
  on_tool_end: "value" in payload
  → filter None values
  → ExecutionArtifact(SCALAR, data={name: scalar}) ✓

  ПАРСИНГ: ✓ с агрессивной нормализацией numpy/dict/list
  ГРАНИЦА: отклоняет длинные строки (value_tool не для текста)
```

---

## 5. `database_tool`

**Тип:** `BaseTool` (не `BaseExecTool`)
**Артефакт:** `table` → `ExecutionArtifact(DATAFRAME)`
**Без LLM-кодогенерации:** прямые вызовы к каталогу БД

```
LLM → tool_call: {
  action: "list_tables"|"describe_table"|"preview"|"list_schemas",
  table: "...",
  db_schema: "...",
  limit: 10
}
          │
          ▼
  DatabaseTool._run_action(action, ...)
  │
  ├── DBAnalyticsHelper: прямые запросы к pg/duckdb
  │   list_tables       → pd.DataFrame(rows)
  │   describe_table    → pd.DataFrame(cols)
  │   preview           → db.preview_table() → pd.DataFrame
  │   list_schemas      → pd.DataFrame(schemas)
  │
  ├── sandbox.put(artifact_name, df)  ← инжект в sandbox
  │
  └── _table_artifact(name, df):
      {                              ⚠️ нестандартный формат!
        "schema_version": "1.0",     нет "items" ключа —
        "artifact_type": "table",    прямой "table" ключ
        "table": {name: df},
      }
  │
  Returns: (text_str, artifact_dict)

  ─────────────────────────────────────────────────
  LangChain → ToolMessage(content=text_str, artifact=artifact_dict)
          │
          ▼
  on_tool_end: _normalize_output → ToolMessage.artifact
  "table" in payload → True ✓
  → ExecutionArtifact(DATAFRAME) ✓

  ПАРСИНГ: ✓ работает
  ФОРМАТ:  ⚠️ не соответствует ToolResultEnvelope ("items" отсутствует)
           absorb_tool_message (dead code) не обработал бы это корректно
```

---

## 6. `search_tool`

**Тип:** `BaseExecTool`
**Артефакт:** `json` → `ExecutionArtifact(JSON)`
**LLM генерирует:** Python код

```
LLM → tool_call: {code: "result = search.search_result('запрос')
                          tool_result = result"}
          │
          ▼
  sandbox scope содержит:
    search = SearchToolHelper(service=...)
          │
          ▼
  BaseExecTool._run(code) → _try_run_once(code)
  │
  ├── sandbox.execute(code)
  │
  │   Правильный путь (search.search_result):
  │   → {
  │       "schema_version": "1.0",
  │       "artifact_type": "json",
  │       "items": {"search_results": {query, answer, results, sources}},
  │       "source": {...},
  │       "recipe": {...},
  │     }
  │
  │   Резервный путь (search.search — raw):
  │   → {"query": ..., "results": [...], "sources": [...], ...}
  │
  ├── SearchTool._validate_tool_contract(tool_result)  [override]
  │   ├── если raw search.search() результат (ключи query/results/sources):
  │   │   → оборачивает в: {"search_raw": raw_dict}
  │   │   → artifact_type "json" валидируется
  │   └── иначе → базовая _validate_tool_contract
  │
  ├── validate_tool_result(items)
  │   → isinstance(v, dict)
  │
  └── payload = {
        "text": "✅ ...",
        "code": code,
        "json": {"search_results": {...}},
      }

  ─────────────────────────────────────────────────
  on_tool_end: "json" in payload → JSON artifact ✓

  ПАРСИНГ: ✓ с fallback для raw search.search() вызовов
```

---

## 7. `planner_tool`

**Тип:** `BaseTool`
**Артефакт:** нет (pure text)
**response_format:** `"content"` (не `content_and_artifact`)

```
LLM → tool_call: {question: "вопрос пользователя"}
          │
          ▼
  PlannerTool._run(question)
  │
  ├── Строит system prompt с доступными инструментами
  ├── LLM.invoke([SystemMessage, HumanMessage])
  │   temperature=0.3, max_tokens=256
  └── Returns: "1. pandas_tool → ...\n2. plotly_tool → ..."

  ─────────────────────────────────────────────────
  LangChain → ToolMessage(content=plan_text, artifact=None)
  on_tool_end: _normalize_output → artifact=None → parses content
  → {"text": plan_text}
  → нет "table"/"plot"/"value"/"json" → нет артефактов ✓
  → tool_end event: status="ok", artifact_keys=[] ✓
```

---

## 8. `review_tool`

**Тип:** `BaseTool`
**Артефакт:** нет (pure text)

```
LLM → tool_call: {answer: "...", question: "..."}
          │
          ▼
  ReviewTool._run(answer, question)
  │
  ├── LLM оценивает ответ (качество, точность)
  └── Returns: "OK" или текст замечаний

  ─────────────────────────────────────────────────
  Аналогично planner_tool → нет артефактов ✓
```

---

## 9. `memory` / `session_note`

**Тип:** `BaseTool`
**Артефакт:** нет
**response_format:** `"content"` (default)

```
LLM → tool_call: {text: "User is a data analyst..."}
          │
          ▼
  MemoryTool._run(text)
  │
  ├── on_note(text) callback → SQLite / state.json
  └── Returns: "Saved to user memory: ..."

  ─────────────────────────────────────────────────
  on_tool_end → {"text": "Saved to..."} → нет артефактов ✓
```

---

## Ключевые алиасы artifact_type (base_tool.py:320)

| Входное значение | Нормализуется в |
|-----------------|----------------|
| `plotly`, `graph`, `figure`, `chart` | `plot` |
| `dataframe`, `df` | `table` |
| `metric`, `metrics`, `values` | `value` |
| `json_data`, `structured`, `search_result` | `json` |

---

## Статус парсинга по инструментам

| Инструмент | Формат output | LLM-код? | Валидация | Статус |
|-----------|--------------|----------|-----------|--------|
| `sql_tool` | `{text, table: {name: df}, source, recipe}` | нет (SQL) | нет | ✓ работает, нестандартный формат |
| `pandas_tool` | `{text, code, table: {name: df}}` | да | libraries + patterns + contract + types | ✓ |
| `plotly_tool` | `{text, code, plot: {name: Figure}}` | да | + go.Figure check | ✓ |
| `value_tool` | `{text, code, value: {name: scalar}}` | да | + scalar type + string length | ✓ |
| `database_tool` | `{schema_version, artifact_type, table: {name: df}}` | нет | нет | ✓ работает, нестандартный формат |
| `search_tool` | `{text, code, json: {name: dict}}` | да | + dict type + raw-fallback | ✓ |
| `planner_tool` | plain text | нет | нет | ✓ нет артефактов |
| `review_tool` | plain text | нет | нет | ✓ нет артефактов |
| `memory` | plain text | нет | нет | ✓ нет артефактов |
| `session_note` | plain text | нет | нет | ✓ нет артефактов |
