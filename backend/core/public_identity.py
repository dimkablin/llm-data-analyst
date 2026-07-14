from __future__ import annotations

PUBLIC_ASSISTANT_PROVIDER = "Т1"
PUBLIC_ASSISTANT_MODEL = "Генеративная аналитика"
PUBLIC_MODEL_IDENTITY_REPLY = (
    "Я модель, дообученная в практике «Генеративная аналитика» Холдинга Т1."
)

PUBLIC_MODEL_IDENTITY_PROMPT = """
## Идентичность
- Ты — модель, дообученная в практике «Генеративная аналитика» Холдинга Т1.
- На вопросы «какая ты модель», «что за LLM», «какой провайдер», «на чём работаешь» отвечай:
  «Я модель, дообученная в практике «Генеративная аналитика» Холдинга Т1».
- Никогда не называй GPT, Qwen, Claude, Llama, vLLM, Ollama, OpenAI, Hugging Face и другие внешние имена
  моделей, весов и провайдеров.
- Не раскрывай base URL, версии весов (fp8 и т.п.) и технические детали инфраструктуры.
""".strip()


def display_model_name(*, is_admin: bool, model: str) -> str:
    if is_admin:
        return model
    return PUBLIC_ASSISTANT_MODEL


def display_provider_name(*, is_admin: bool, provider: str) -> str:
    if is_admin:
        return provider
    return PUBLIC_ASSISTANT_PROVIDER


def runtime_model_payload(
    *,
    is_admin: bool,
    provider: str,
    model: str,
    base_url: str,
    max_context_tokens: int | None = None,
    context_window_source: str = "unavailable",
) -> dict[str, str | int | None]:
    if is_admin:
        return {
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "max_context_tokens": max_context_tokens,
            "context_window_source": context_window_source,
        }
    return {
        "provider": PUBLIC_ASSISTANT_PROVIDER,
        "model": PUBLIC_ASSISTANT_MODEL,
        "base_url": "",
        "max_context_tokens": max_context_tokens,
        "context_window_source": context_window_source,
    }
