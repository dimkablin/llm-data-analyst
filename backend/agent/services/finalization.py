from __future__ import annotations

import pandas as pd


def fallback_text(
    prompt: str,
    df: pd.DataFrame | None = None,
    stop_reason: str | None = None,
) -> str:
    if df is not None:
        if stop_reason == "max_steps_reached":
            return (
                "Я выполнил несколько шагов анализа, но не получил надежный артефакт. "
                "Уточните запрос или сузьте задачу."
            )
        return (
            "Не удалось завершить анализ: агент не вернул финальный ответ по данным. "
            "Если до этого были tool-вызовы, их результаты и ошибки доступны в ходе выполнения."
        )

    if not prompt.strip():
        return "Я получил запрос, но не смог сформировать содержательный ответ."

    return (
        "Не удалось запустить расширенный аналитический режим для этого запроса. "
        "Попробуйте повторить запрос или уточнить задачу."
    )
