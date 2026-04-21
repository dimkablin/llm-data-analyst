from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from backend.agent.callbacks import ThinkingOutputParser


class ReasoningChatOpenAI(ChatOpenAI):
    """ChatOpenAI wrapper that strips ``<think>`` blocks from all response paths.

    Sanitization is applied at the provider boundary so that downstream
    consumers (sql_table_service, tool callers, etc.) always receive only
    the visible assistant content and never raw chain-of-thought.

    Covered paths
    -------------
    * ``invoke``   — sync, single response
    * ``ainvoke``  — async, single response
    * ``stream``   — sync generator of AIMessageChunk
    * ``astream``  — async generator of AIMessageChunk

    Reasoning text (everything inside ``<think>…</think>``) is stored in
    ``response.additional_kwargs["reasoning"]`` when present, so callers
    that need it can still access it without leaking into ``content``.

    Edge cases handled by :class:`ThinkingOutputParser`
    ---------------------------------------------------
    * Unclosed ``<think>`` — buffer discarded, no content leak
    * Multiple ``<think>`` blocks
    * Tags split across streaming chunks
    * Case-insensitive tags (``<THINK>``, ``</Think>``, …)
    * No ``<think>`` at all — content passes through unchanged
    """

    # ------------------------------------------------------------------
    # Ollama reasoning extraction — inject into additional_kwargs so
    # TokenStreamingCallback can emit thinking events via on_llm_new_token
    # and on_llm_end.
    # ------------------------------------------------------------------

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        gen_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if gen_chunk is None:
            return None
        try:
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                ollama_reasoning = delta.get("reasoning") or ""
                if ollama_reasoning:
                    gen_chunk.message.additional_kwargs["reasoning"] = ollama_reasoning
        except (AttributeError, IndexError, TypeError):
            pass
        return gen_chunk

    def _create_chat_result(
        self,
        response: dict | Any,
        generation_info: dict | None = None,
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        try:
            response_dict = (
                response if isinstance(response, dict) else response.model_dump()
            )
            ollama_reasoning = (
                response_dict.get("choices", [{}])[0]
                .get("message", {})
                .get("reasoning", "")
            ) or ""
            if ollama_reasoning and result.generations:
                ak = result.generations[0].message.additional_kwargs
                if "reasoning" not in ak:
                    ak["reasoning"] = ollama_reasoning
        except (AttributeError, IndexError, TypeError):
            pass
        return result

    # ------------------------------------------------------------------
    # Sync / async single-response paths
    # ------------------------------------------------------------------

    def invoke(
        self,
        input: Any,
        config: Any = None,
        **kwargs: Any,
    ) -> BaseMessage:
        response = super().invoke(input, config=config, **kwargs)
        return _sanitize_response(response)

    async def ainvoke(
        self,
        input: Any,
        config: Any = None,
        **kwargs: Any,
    ) -> BaseMessage:
        response = await super().ainvoke(input, config=config, **kwargs)
        return _sanitize_response(response)

    # ------------------------------------------------------------------
    # Streaming paths
    # ------------------------------------------------------------------

    def stream(
        self,
        input: Any,
        config: Any = None,
        **kwargs: Any,
    ) -> Iterator[BaseMessage]:
        """Yield sanitized chunks.

        Each non-final chunk carries only newly visible text.
        The final chunk always carries ``response_metadata`` and
        accumulated reasoning in ``additional_kwargs["reasoning"]``.
        No chunk is emitted twice.
        """
        parser = ThinkingOutputParser()
        prev: BaseMessage | None = None

        for chunk in super().stream(input, config=config, **kwargs):
            if prev is not None:
                content = prev.content if isinstance(prev.content, str) else ""
                visible, _ = parser.feed(content)
                if visible:
                    yield prev.model_copy(update={"content": visible})
            prev = chunk

        if prev is not None:
            yield _emit_final_chunk(prev, parser)

    async def astream(
        self,
        input: Any,
        config: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[BaseMessage]:
        """Async equivalent of :meth:`stream`."""
        parser = ThinkingOutputParser()
        prev: BaseMessage | None = None

        async for chunk in super().astream(input, config=config, **kwargs):
            if prev is not None:
                content = prev.content if isinstance(prev.content, str) else ""
                visible, _ = parser.feed(content)
                if visible:
                    yield prev.model_copy(update={"content": visible})
            prev = chunk

        if prev is not None:
            yield _emit_final_chunk(prev, parser)


# ------------------------------------------------------------------
# Helpers (module-level, not methods — easier to test and mock)
# ------------------------------------------------------------------

def _sanitize_response(response: BaseMessage) -> BaseMessage:
    """Strip ``<think>`` blocks from a complete (non-streaming) response.

    Uses :class:`ThinkingOutputParser` for correctness on all edge cases.
    Reasoning text is preserved in ``additional_kwargs["reasoning"]``.
    """
    content = response.content
    if not isinstance(content, str) or not content:
        return response

    parser = ThinkingOutputParser()
    parser.feed(content)
    parser.flush()

    stripped = parser.visible()
    reasoning = parser.reasoning()

    # Fast path: nothing to strip.
    if stripped == content.strip() and not reasoning:
        return response

    additional_kwargs: dict[str, Any] = {**response.additional_kwargs}
    if reasoning and "reasoning" not in additional_kwargs:
        additional_kwargs["reasoning"] = reasoning

    return response.model_copy(update={
        "content": stripped,
        "additional_kwargs": additional_kwargs,
    })


def _emit_final_chunk(chunk: BaseMessage, parser: ThinkingOutputParser) -> BaseMessage:
    """Process the last streaming chunk and return it with sanitized content.

    ``feed()`` + ``flush()`` are called here (not in the main loop) so that
    the content attributed to this chunk is *only* the new visible text from
    this specific chunk, never the accumulated total from previous chunks.
    Reasoning is stored in ``additional_kwargs["reasoning"]``.
    """
    content = chunk.content if isinstance(chunk.content, str) else ""
    vis_feed, _ = parser.feed(content) if content else ("", "")
    vis_flush, _ = parser.flush()
    new_visible = vis_feed + vis_flush

    reasoning = parser.reasoning()
    additional_kwargs: dict[str, Any] = {**chunk.additional_kwargs}
    if reasoning and "reasoning" not in additional_kwargs:
        additional_kwargs["reasoning"] = reasoning

    return chunk.model_copy(update={
        "content": new_visible,
        "additional_kwargs": additional_kwargs,
    })


def _remap_ollama_reasoning(msg: BaseMessage) -> BaseMessage:
    """Remap additional_kwargs['reasoning_content'] → ['reasoning'].

    ChatOllama stores thinking in 'reasoning_content'; our TokenStreamCallbackHandler
    and downstream consumers expect 'reasoning'.
    No-op if reasoning_content absent or reasoning already present.
    """
    ak = getattr(msg, "additional_kwargs", {}) or {}
    reasoning_content = ak.get("reasoning_content")
    if reasoning_content is not None and "reasoning" not in ak:
        return msg.model_copy(update={"additional_kwargs": {**ak, "reasoning": reasoning_content}})
    return msg


class ReasoningChatOllama(ChatOllama):
    """ChatOllama wrapper with unified interface matching ReasoningChatOpenAI.

    Uses native Ollama /api/chat endpoint so that reasoning=True/False is
    reliably honoured (Ollama ≤ 0.20.7 ignores think:false on the OpenAI-compat
    /v1/chat/completions endpoint in all payload variants tested).

    Behaviour:
      reasoning=True  → sends think:true, content is clean, thinking in
                        additional_kwargs["reasoning_content"] → remapped to ["reasoning"]
      reasoning=False → sends think:false, model does not think at all
      reasoning=None  → model default; <think> tags may appear in content

    All four LangChain response paths (invoke/ainvoke/stream/astream) remap
    reasoning_content → reasoning so TokenStreamCallbackHandler works unchanged.
    """

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> BaseMessage:
        return _remap_ollama_reasoning(super().invoke(input, config=config, **kwargs))

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> BaseMessage:
        return _remap_ollama_reasoning(await super().ainvoke(input, config=config, **kwargs))

    def stream(
        self, input: Any, config: Any = None, **kwargs: Any
    ) -> Iterator[BaseMessage]:
        for chunk in super().stream(input, config=config, **kwargs):
            yield _remap_ollama_reasoning(chunk)

    async def astream(
        self, input: Any, config: Any = None, **kwargs: Any
    ) -> AsyncIterator[BaseMessage]:
        async for chunk in super().astream(input, config=config, **kwargs):
            yield _remap_ollama_reasoning(chunk)


# Union type for type annotations across the codebase
AnyReasoningLLM = ReasoningChatOpenAI | ReasoningChatOllama


def make_reasoning_llm(
    *,
    provider: str | None,
    model: str,
    base_url: str,
    api_key: str | None = None,
    enable_thinking: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    streaming: bool = True,
    timeout: float = 120.0,
    top_p: float = 1.0,
    top_k: int = 0,
    num_ctx: int = 0,
    presence_penalty: float = 0.0,
    chat_template_kwargs_enabled: bool = False,
) -> AnyReasoningLLM:
    """Return a reasoning-capable LLM for the given provider.

    Routing:
      provider == "ollama"  → ReasoningChatOllama  (native /api/chat, reasoning= param)
      anything else         → ReasoningChatOpenAI  (OpenAI-compat /v1/chat/completions)

    Observed limitation (documented):
      Ollama ≤ 0.20.7 /v1/chat/completions ignores think:true/false in every
      payload form tested (top-level field, options.think, /no_think prefix,
      chat_template_kwargs). Native /api/chat via ChatOllama is the only reliable
      thinking-control path for Ollama.
    """
    provider_norm = (provider or "").strip().lower()

    if provider_norm == "ollama":
        ollama_kwargs: dict[str, Any] = {
            "model": model,
            "base_url": base_url,
            "reasoning": enable_thinking,   # True → think:true, False → think:false
            "temperature": temperature,
            "num_predict": max_tokens,
            "top_p": top_p,
            "streaming": streaming,
            # httpx timeout for native Ollama client
            "client_kwargs": {"timeout": timeout},
            "async_client_kwargs": {"timeout": timeout},
        }
        if top_k > 0:
            ollama_kwargs["top_k"] = top_k
        if num_ctx > 0:
            ollama_kwargs["num_ctx"] = num_ctx
        # presence_penalty not supported by Ollama native API — silently dropped
        return ReasoningChatOllama(**ollama_kwargs)

    # ── OpenAI-compat path (vLLM, LiteLLM, OpenAI, …) ──────────────────
    from backend.core.llm_provider import get_provider_policy  # avoid circular at module level

    extra_body: dict[str, Any] = {}
    if chat_template_kwargs_enabled:
        extra_body.update(
            get_provider_policy(provider).build_extra_body(enable_thinking=enable_thinking)
        )
    if top_k > 0:
        extra_body["top_k"] = top_k
    if num_ctx > 0:
        extra_body["num_ctx"] = num_ctx

    openai_kwargs: dict[str, Any] = {
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "streaming": streaming,
        "timeout": timeout,
    }
    if extra_body:
        openai_kwargs["extra_body"] = extra_body

    return ReasoningChatOpenAI(**openai_kwargs)
