from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class LLMProviderPolicy:
    """Описывает provider-specific поведение для управления thinking-режимом.

    Верхние слои (runner, tools, services) используют этот объект вместо
    прямых проверок вида ``if provider == "vllm"``.
    """

    # "chat_template_kwargs" — vLLM-specific: поле chat_template_kwargs в extra_body.
    # "ollama_think"         — Ollama-specific: top-level поле "think" в extra_body.
    # "none"                 — thinking toggle не поддерживается / не нужен.
    thinking_control_mode: Literal["chat_template_kwargs", "ollama_think", "none"]

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
        if self.thinking_control_mode == "ollama_think":
            # Ollama: top-level поле "think" в теле запроса (OpenAI-compatible API).
            return {"think": enable_thinking}
        return {}

    def get_thinking_message_prefix(self, *, enable_thinking: bool) -> str:
        """Возвращает префикс, который нужно вставить в первое human-сообщение.

        Для Qwen3 через Ollama параметр ``think`` в extra_body иногда игнорируется
        (зависит от версии Ollama и Modelfile).  Префикс ``/no_think`` / ``/think``
        — надёжный fallback, который модель обрабатывает на уровне токенизатора.
        Для vLLM и других провайдеров — пустая строка (управление идёт через кварги).
        """
        if self.thinking_control_mode == "ollama_think":
            return "/think\n" if enable_thinking else "/no_think\n"
        return ""


_POLICIES: dict[str, LLMProviderPolicy] = {
    "ollama": LLMProviderPolicy(
        thinking_control_mode="ollama_think",
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
