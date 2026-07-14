from __future__ import annotations

RECOVERY_TEXT_PREFIX = "Шаг анализа завершился с ограничением итераций модели"

LLM_UNAVAILABLE_USER_TEXT = (
    "Языковая модель сейчас недоступна: нет соединения с LLM-сервером или сработал таймаут. "
    "Проверьте, что Ollama (или другой провайдер) запущен и что "
    "LLM_MODEL_API_URL доступен из контейнера backend."
)
