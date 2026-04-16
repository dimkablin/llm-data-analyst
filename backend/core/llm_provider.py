from __future__ import annotations

# NOTE: Ollama ≤ 0.20.7 limitation — the OpenAI-compat /v1/chat/completions endpoint
# ignores think:true/false in ALL payload forms tested (top-level field, options.think,
# /no_think prefix, chat_template_kwargs). Thinking control for Ollama is handled
# exclusively via make_reasoning_llm(provider="ollama", enable_thinking=...) which routes
# to ReasoningChatOllama using the native /api/chat endpoint (langchain-ollama ChatOllama).
# The methods below (build_extra_body) remain valid for vLLM and other OpenAI-compat
# providers; the Ollama path no longer calls them.

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class LLMProviderPolicy:
    """Описывает provider-specific поведение для управления thinking-режимом.

    Верхние слои (runner, tools, services) используют этот объект вместо
    прямых проверок вида ``if provider == "vllm"``.
    """

    # "chat_template_kwargs" — vLLM-specific: поле chat_template_kwargs в extra_body.
    # "none"                 — thinking toggle не поддерживается / не нужен.
    thinking_control_mode: Literal["chat_template_kwargs", "none"]

    # Diagnostic-only: vllm стриппит <think> server-side → orphaned </think>.
    # ThinkingOutputParser уже обрабатывает это генерически (коммит 34b408d).
    # Поле документирует поведение провайдера, не управляет runtime-логикой.
    may_emit_orphaned_think_close_tags: bool

    def build_extra_body(self, *, enable_thinking: bool) -> dict[str, Any]:
        """Возвращает provider-specific фрагмент extra_body для thinking-toggle.

        Возвращает пустой dict, если провайдер не поддерживает управление thinking.
        Caller: ``extra_body.update(policy.build_extra_body(enable_thinking=...))``.
        """
        if self.thinking_control_mode == "chat_template_kwargs":
            # vLLM: передаёт параметры Jinja-шаблона через chat_template_kwargs.
            return {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
        return {}


_POLICIES: dict[str, LLMProviderPolicy] = {
    # Ollama: thinking is controlled via ReasoningChatOllama(reasoning=...) which maps to
    # the native /api/chat "think" field. build_extra_body is not used for Ollama.
    "ollama": LLMProviderPolicy(
        thinking_control_mode="none",
        may_emit_orphaned_think_close_tags=False,
    ),
    "vllm": LLMProviderPolicy(
        thinking_control_mode="chat_template_kwargs",
        may_emit_orphaned_think_close_tags=True,
    ),
}

# Safe default для неизвестных провайдеров (LiteLLM-прокси и т.п.).
# thinking_control_mode="none" → build_extra_body() вернёт {} → no extra_body poisoning.
_DEFAULT_POLICY = LLMProviderPolicy(
    thinking_control_mode="none",
    may_emit_orphaned_think_close_tags=False,
)


def get_provider_policy(provider: str | None) -> LLMProviderPolicy:
    """Возвращает политику провайдера.

    None / "" / неизвестное имя → safe default (thinking_control_mode="none").
    """
    return _POLICIES.get(str(provider or "").strip().lower(), _DEFAULT_POLICY)
