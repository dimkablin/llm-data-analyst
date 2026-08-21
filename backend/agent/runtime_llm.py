from __future__ import annotations

from typing import Literal

from backend.agent.llm_client import AnyReasoningLLM, make_reasoning_llm
from backend.core.config import Settings


def build_runtime_llm(
    settings: Settings,
    *,
    role: Literal["chat", "tool"],
    include_reasoning: bool,
    timeout_sec: int | None = None,
    max_tokens_override: int | None = None,
) -> AnyReasoningLLM:
    enable_thinking = settings.llm_enable_thinking and include_reasoning

    if enable_thinking:
        temperature = settings.llm_temperature_tool if role == "tool" else 1.0
        top_p = settings.llm_top_p
    else:
        temperature = (
            settings.llm_temperature_tool
            if role == "tool"
            else settings.llm_temperature_chat
        )
        top_p = 0.8

    max_tokens = max_tokens_override or settings.llm_max_tokens_default
    if max_tokens_override is None and include_reasoning:
        max_tokens = settings.llm_max_tokens_reasoning

    return make_reasoning_llm(
        provider=settings.llm_provider,
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        enable_thinking=enable_thinking,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=settings.llm_streaming,
        timeout=float(timeout_sec or settings.backend_query_timeout_sec),
        top_p=top_p,
        top_k=settings.llm_top_k,
        num_ctx=settings.llm_num_ctx,
        presence_penalty=settings.llm_presence_penalty,
        chat_template_kwargs_enabled=settings.llm_chat_template_kwargs_enabled,
    )
