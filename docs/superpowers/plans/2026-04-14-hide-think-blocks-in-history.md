# Hide Think Blocks in History When LLM_SHOW_THINK=false

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Когда `LLM_SHOW_THINK=false`, think-блоки не должны попадать на фронт ни при стриминге, ни при загрузке истории после обновления страницы.

**Architecture:** Бэкенд хранит `reasoning_steps` и `pre_reasoning` в `chat_history` всегда (для возможной будущей переключаемости). При отдаче истории через `GET /sessions/{session_id}` — фильтруем данные о thinking из ответа, если `settings.llm_show_think=False`. Фронт получает уже чистые данные и не строит thinking-блоки.

**Tech Stack:** Python/FastAPI (backend), TypeScript/React (frontend — изменения не нужны)

---

## Контекст и корень проблемы

Стриминг: `TokenStreamCallbackHandler(show_think=settings.llm_show_think)` — при `False` thinking не стримится на клиент ✓

История (после рефреша):
- `ToolCollector.on_tool_start()` в `callbacks.py:340` вызывает `token_callback.take_pending_thinking()` и сохраняет результат в `event["pre_reasoning"]`
- `tool_collector.to_persisted_activities()` в `callbacks.py:536` сохраняет `pre_reasoning` в активности тулов
- В `query.py:1074` при персистировании сохраняются `reasoning_steps=[s.to_dict() for s in reasoning_steps]`
- `GET /sessions/{session_id}` в `sessions.py:250` возвращает `chat_history` с этими данными без фильтрации
- Фронт в `buildBlocksFromHistory()` создаёт `thinking`-блоки из `pre_reasoning` и orphan-`reasoning_steps`

---

## File Structure

**Изменяется только один файл:**
- Modify: `backend/api/routes/sessions.py` — добавить фильтрацию thinking-данных при `llm_show_think=False`

---

### Task 1: Добавить фильтрацию think-блоков в GET /sessions/{session_id}

**Files:**
- Modify: `backend/api/routes/sessions.py`

- [ ] **Step 1: Написать failing тест**

Создать файл `tests/api/test_session_think_filter.py`:

```python
"""Tests that think blocks are stripped from history when LLM_SHOW_THINK=False."""
import pytest
from unittest.mock import MagicMock, patch


CHAT_HISTORY_WITH_THINK = [
    {"role": "user", "content": "hello"},
    {
        "role": "ai",
        "content": "sure",
        "reasoning_steps": [
            {"step_index": 0, "kind": "final_synthesis", "content": "I think...", "tool_name": None}
        ],
        "tools": [
            {
                "tool_name": "python_repl",
                "pre_reasoning": "Let me think about this...",
                "input_summary": "x = 1",
                "status": "done",
            }
        ],
    },
]


def _make_state(chat_history):
    state = MagicMock()
    state.session_id = "test-session"
    state.chat_history = chat_history
    state.artifacts = []
    state.df_path = None
    state.dataset_name = None
    state.source_type = None
    state.source_ref_id = None
    state.source_label = None
    state.source_mode = None
    state.selected_skill_ids = []
    state.session_memory = ""
    return state


def test_think_blocks_stripped_when_show_think_false():
    """reasoning_steps and pre_reasoning must be absent when llm_show_think=False."""
    from backend.api.routes.sessions import _strip_thinking_from_history

    with patch("backend.core.config.settings") as mock_settings:
        mock_settings.llm_show_think = False
        result = _strip_thinking_from_history(CHAT_HISTORY_WITH_THINK)

    ai_message = result[1]
    assert "reasoning_steps" not in ai_message or ai_message.get("reasoning_steps") is None
    tools = ai_message.get("tools", [])
    for tool in tools:
        assert "pre_reasoning" not in tool or tool.get("pre_reasoning") is None


def test_think_blocks_preserved_when_show_think_true():
    """reasoning_steps and pre_reasoning must remain when llm_show_think=True."""
    from backend.api.routes.sessions import _strip_thinking_from_history

    result = _strip_thinking_from_history(CHAT_HISTORY_WITH_THINK)

    ai_message = result[1]
    assert ai_message.get("reasoning_steps") is not None
    tools = ai_message.get("tools", [])
    assert tools[0].get("pre_reasoning") == "Let me think about this..."


def test_user_messages_untouched():
    """User messages must not be modified."""
    from backend.api.routes.sessions import _strip_thinking_from_history

    result = _strip_thinking_from_history(CHAT_HISTORY_WITH_THINK)
    assert result[0]["content"] == "hello"
    assert result[0]["role"] == "user"
```

- [ ] **Step 2: Запустить тест — убедиться, что он падает**

```bash
cd C:/Users/dimka/Documents/PROJECTS/llm-data-analyst/llm-data-analyst-dev
python -m pytest tests/api/test_session_think_filter.py -v
```

Ожидаемый результат: `FAILED` — `ImportError: cannot import name '_strip_thinking_from_history'`

- [ ] **Step 3: Реализовать `_strip_thinking_from_history` и применить в `get_session`**

В файле `backend/api/routes/sessions.py`:

**3а. Добавить импорт settings** (в начало файла, рядом с другими импортами):

```python
from backend.core.config import settings
```

**3б. Добавить вспомогательную функцию** (после существующих хелперов, перед `@router.get("/sessions")`):

```python
def _strip_thinking_from_history(
    chat_history: list[dict],
) -> list[dict]:
    """Remove reasoning_steps and pre_reasoning from AI messages.

    Called when settings.llm_show_think=False so that think blocks
    are not exposed to the frontend on page refresh.
    Data is preserved in storage — only the API response is filtered.
    """
    result = []
    for message in chat_history:
        if str(message.get("role", "")).strip().lower() not in ("ai", "assistant"):
            result.append(message)
            continue
        filtered = {k: v for k, v in message.items() if k != "reasoning_steps"}
        tools = filtered.get("tools")
        if tools:
            filtered["tools"] = [
                {k: v for k, v in tool.items() if k != "pre_reasoning"}
                for tool in tools
            ]
        result.append(filtered)
    return result
```

**3в. Применить фильтр в `get_session`** — изменить строку с `chat_history=state.chat_history` в ответе:

```python
    chat_history = state.chat_history
    if not settings.llm_show_think:
        chat_history = _strip_thinking_from_history(chat_history)

    return SessionStateResponse(
        session_id=state.session_id,
        title=title,
        chat_history=chat_history,
        # ... остальные поля без изменений
    )
```

- [ ] **Step 4: Запустить тест — убедиться, что он проходит**

```bash
python -m pytest tests/api/test_session_think_filter.py -v
```

Ожидаемый результат: `3 passed`

- [ ] **Step 5: Запустить общий тест-сьют**

```bash
python -m pytest tests/ -v --tb=short -q
```

Ожидаемый результат: все тесты проходят (или число упавших не увеличилось по сравнению с базовым).

- [ ] **Step 6: Коммит**

```bash
git add backend/api/routes/sessions.py tests/api/test_session_think_filter.py
git commit -m "fix: strip think blocks from session history when LLM_SHOW_THINK=false

reasoning_steps and pre_reasoning are still persisted to storage
but filtered out of GET /sessions/{id} response when show_think=False,
matching streaming behavior."
```

---

## Self-Review

**Spec coverage:**
- Стриминг: не меняется, уже работает ✓
- История после рефреша: исправляется в Task 1 ✓
- Данные сохраняются в БД: да, фильтрация только на уровне ответа ✓
- Включение/выключение настройки: работает динамически (без перезаписи истории) ✓

**Placeholder scan:** нет TBD/TODO.

**Type consistency:** `_strip_thinking_from_history` принимает `list[dict]` и возвращает `list[dict]` — соответствует `state.chat_history: list[dict[str, Any]]` в `SessionState`.

**Граничные случаи:**
- Сообщение без `tools` — `filtered.get("tools")` вернёт `None`, блок пропустится ✓
- Сообщение без `reasoning_steps` — dict comprehension просто не найдёт ключ, ок ✓
- `role=user` — сообщение не трогается ✓
