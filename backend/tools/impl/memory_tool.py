"""Memory tools — let the agent persist observations.

Two tools:

* **memory** — saves long-term facts *about the user* (preferences,
  role, expertise) into SQLite via the user-memory consolidation pipeline.

* **session_note** — saves context *about the current analysis session*
  (data descriptions, key findings, intermediate conclusions) into the
  session's ``state.json``.

Neither tool is a sandboxed code-execution tool — both are direct
LangChain ``BaseTool`` instances backed by simple in-memory callbacks.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool

# ── User Memory Tool ──────────────────────────────────────────────────────────

_USER_MEMORY_DESCRIPTION = """\
Сохрани долгосрочное наблюдение о пользователе, которое будет полезно \
в будущих разговорах. Вызывай, когда делаешь значимый вывод о пользователе — \
его роль, домен, уровень экспертизы, аналитические цели, предпочтительный \
формат вывода, стиль общения или повторяющиеся паттерны в работе.

Записывай сжатую семантическую заметку — НЕ дословную цитату. Фиксируй \
*смысл*, а не *что было сказано*. Например, вместо \
«Пользователь сказал, что работает в ритейле» пиши «Пользователь — \
аналитик розничной торговли, фокус на продажах».

Правила:
- Делай вывод и обобщай: переводи наблюдения в устойчивые факты.
- Будь конкретным: «предпочитает bar-чарты Plotly с месячной гранулярностью» \
  лучше, чем «любит графики».
- Один атомарный факт на вызов; для нескольких фактов — несколько вызовов.
- НЕ сохраняй данные сессии (датасеты, схемы, результаты) \
  — для этого используй session_note.
- НЕ вызывай для воспроизведения истории — она уже есть в промпте.

Вход: 1–2 предложения с дистиллированным наблюдением.
Выход: подтверждение сохранения.

Пример: memory(«Пользователь — аналитик данных в ритейле. \
Предпочитает сжатые месячные агрегации и bar-чарты Plotly вместо таблиц.»)
"""


class MemoryTool(BaseTool):
    """Append a user-level note to the user memory buffer."""

    name: str = "memory"
    description: str = _USER_MEMORY_DESCRIPTION

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, on_note: Callable[[str], None], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_memo_cb", on_note)

    def _run(self, text: str, *args: Any, **_kwargs: Any) -> str:
        note = text.strip()
        if not note:
            return "Ничего не сохранено — заметка пустая."
        cb: Callable[[str], None] = object.__getattribute__(self, "_memo_cb")
        cb(note)
        return f"Сохранено в память пользователя: {note[:120]}{'…' if len(note) > 120 else ''}"

    async def _arun(self, text: str, *args: Any, **_kwargs: Any) -> str:
        return self._run(text)


# ── Session Note Tool ─────────────────────────────────────────────────────────

_SESSION_NOTE_DESCRIPTION = """\
Сохрани семантическое наблюдение о текущей аналитической сессии, \
чтобы на него можно было ссылаться в уточняющих вопросах без повторного анализа.

Используй для:
- *Смысла* загруженных данных (домен, гранулярность, временной диапазон, \
  бизнес-цель) — не просто названий колонок.
- Ключевых выводов и заключений: какие паттерны, аномалии или ответы \
  были найдены и *почему они важны*.
- Проблем качества данных, влияющих на интерпретацию (например, «В Q2 выручка \
  имеет 3% пропусков — вероятно, пробел в выгрузке, а не нулевые продажи»).
- Решений, принятых в ходе анализа (применённые фильтры, допущения).

Записывай сжатую семантическую заметку — НЕ дословную копию данных или \
вывода. Фиксируй *понимание*, а не сырые факты. Например, вместо перечисления \
колонок объясни, что представляет датасет: «Ежемесячные продажи ритейла \
по продукту и региону, 2023 — для анализа региональной выручки».

НЕ используй для фактов о пользователе (роль, предпочтения) \
— для этого используй memory.

Вход: 1–3 предложения с дистиллированным контекстом или выводом.
Выход: подтверждение сохранения.

Пример: session_note(«Датасет продаж охватывает янв–дек 2023 с выручкой \
по продукту и региону. В Q2 3% пропусков — вероятно, пробел в выгрузке, \
а не реальные нулевые продажи.»)
"""


class SessionNoteTool(BaseTool):
    """Append a session-level note to the session memory buffer."""

    name: str = "session_note"
    description: str = _SESSION_NOTE_DESCRIPTION

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, on_note: Callable[[str], None], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_memo_cb", on_note)

    def _run(self, text: str, *args: Any, **_kwargs: Any) -> str:
        note = text.strip()
        if not note:
            return "Ничего не сохранено — заметка пустая."
        cb: Callable[[str], None] = object.__getattribute__(self, "_memo_cb")
        cb(note)
        return f"Сохранено в заметки сессии: {note[:120]}{'…' if len(note) > 120 else ''}"

    async def _arun(self, text: str, *args: Any, **_kwargs: Any) -> str:
        return self._run(text)
