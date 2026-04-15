from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
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
