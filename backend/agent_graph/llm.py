from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from backend.agent.llm_client import AnyReasoningLLM, make_reasoning_llm
from backend.agent.prompts import chat_system_prompt
from backend.core.config import Settings


@dataclass(slots=True)
class LlmFactory:
    """Creates configured chat models for graph nodes."""

    settings: Settings

    def build(
        self,
        *,
        role: Literal["chat", "tool"],
        include_reasoning: bool,
        timeout_sec: int | None = None,
        max_tokens_override: int | None = None,
    ) -> AnyReasoningLLM:
        enable_thinking = self.settings.llm_enable_thinking and include_reasoning

        if enable_thinking:
            temperature = 1.0
            top_p = self.settings.llm_top_p
        else:
            temperature = (
                self.settings.llm_temperature_tool
                if role == "tool"
                else self.settings.llm_temperature_chat
            )
            top_p = 0.8

        max_tokens = max_tokens_override or self.settings.llm_max_tokens_default
        if max_tokens_override is None and include_reasoning:
            max_tokens = self.settings.llm_max_tokens_reasoning

        return make_reasoning_llm(
            provider=self.settings.llm_provider,
            model=self.settings.llm_model,
            base_url=self.settings.llm_base_url,
            api_key=self.settings.llm_api_key,
            enable_thinking=enable_thinking,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=self.settings.llm_streaming_force or self.settings.llm_streaming,
            timeout=float(timeout_sec or self.settings.backend_query_timeout_sec),
            top_p=top_p,
            top_k=self.settings.llm_top_k,
            num_ctx=self.settings.llm_num_ctx,
            presence_penalty=self.settings.llm_presence_penalty,
            chat_template_kwargs_enabled=self.settings.llm_chat_template_kwargs_enabled,
        )


@dataclass(slots=True)
class MessageBuilder:
    """Builds LangChain messages from serializable graph request data."""

    settings: Settings
    user_memory: Any | None = None
    session_memory: Any | None = None

    def build_chat_messages(
        self,
        *,
        prompt: str,
        history: list[dict[str, Any]],
        use_history: bool,
        system_prompt_suffix: str = "",
    ) -> list[BaseMessage]:
        system_prompt = chat_system_prompt
        if system_prompt_suffix.strip():
            system_prompt = f"{system_prompt}\n\n{system_prompt_suffix.strip()}"
        return self.build_messages(
            prompt=prompt,
            history=history,
            use_history=use_history,
            system_prompt=system_prompt,
        )

    def build_messages(
        self,
        *,
        prompt: str,
        history: list[dict[str, Any]],
        use_history: bool,
        system_prompt: str | None = None,
    ) -> list[BaseMessage]:
        messages: list[BaseMessage] = []
        system_parts: list[str] = []

        if system_prompt:
            system_parts.append(system_prompt)

        memory_block = self._memory_block(self.user_memory)
        if memory_block:
            system_parts.append(memory_block)

        session_memory_block = self._memory_block(self.session_memory)
        if session_memory_block:
            system_parts.append(session_memory_block)

        recent: list[dict[str, Any]] = []
        if use_history and history:
            max_msgs = max(0, self.settings.agent_history_max_messages)
            recent = history[-max_msgs:] if max_msgs > 0 else []
            older = history[:-max_msgs] if max_msgs > 0 else history

            summary = self._history_summary(older)
            if summary:
                system_parts.append(summary)

        if system_parts:
            messages.append(SystemMessage(content="\n\n".join(system_parts)))

        for item in recent:
            role = item.get("role")
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))

        messages.append(HumanMessage(content=prompt))
        return messages

    @staticmethod
    def content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text:
                        parts.append(text)
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts)
        return str(content or "")

    @staticmethod
    def _memory_block(memory: Any | None) -> str:
        if memory is None or not hasattr(memory, "build_block"):
            return ""
        block = memory.build_block()
        return str(block or "").strip()

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        clean = str(text or "").strip()
        if len(clean) <= max_len:
            return clean
        return f"{clean[:max_len]}..."

    def _history_summary(self, older_history: list[dict[str, Any]]) -> str:
        if not older_history:
            return ""

        rows: list[str] = []
        for item in older_history[-8:]:
            role = str(item.get("role", "assistant"))
            content = self._truncate(str(item.get("content", "")), 140)
            if not content:
                continue
            marker = "U" if role == "user" else "A"
            rows.append(f"- {marker}: {content}")

        if not rows:
            return ""

        max_chars = max(200, self.settings.agent_history_summary_chars)
        summary = "Краткая сводка предыдущего диалога:\n" + "\n".join(rows)
        return self._truncate(summary, max_chars)


def build_runtime_metadata(trace_context: dict[str, Any] | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if not trace_context:
        return metadata

    session_id = trace_context.get("session_id")
    if isinstance(session_id, str) and session_id:
        metadata["session_id"] = session_id
        metadata["thread_id"] = session_id
        metadata["conversation_id"] = session_id

    user_id = trace_context.get("user_id")
    if user_id is not None:
        metadata["user_id"] = str(user_id)

    username = trace_context.get("username")
    if isinstance(username, str) and username:
        metadata["username"] = username

    request_kind = trace_context.get("request_kind")
    if isinstance(request_kind, str) and request_kind:
        metadata["request_kind"] = request_kind

    return metadata


def state_cache_signature(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()
