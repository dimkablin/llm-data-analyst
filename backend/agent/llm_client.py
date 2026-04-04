from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from backend.agent.callbacks import THINKING_RE, extract_thinking, strip_thinking


class ThinkingAwareChatOpenAI(ChatOpenAI):
    """ChatOpenAI wrapper that extracts ``<think>`` blocks from sync ``invoke`` responses.

    * Strips ``<think>…</think>`` from ``response.content``.
    * Stores the extracted reasoning in ``response.additional_kwargs["reasoning"]``
      (only when the API has not already populated that field natively).

    Streaming responses are handled separately by
    :class:`~backend.agent.callbacks.TokenStreamCallbackHandler` and are not
    affected by this wrapper.
    """

    def invoke(
        self,
        input: Any,
        config: Any = None,
        **kwargs: Any,
    ) -> BaseMessage:
        response = super().invoke(input, config=config, **kwargs)
        return _extract_think_from_response(response)


def _extract_think_from_response(response: BaseMessage) -> BaseMessage:
    """Strip ``<think>`` from *content* and store reasoning in *additional_kwargs*."""
    content = response.content
    if not isinstance(content, str) or not THINKING_RE.search(content):
        return response

    reasoning = extract_thinking(content)
    stripped = strip_thinking(content)

    additional_kwargs: dict[str, Any] = {**response.additional_kwargs}
    if reasoning and "reasoning" not in additional_kwargs:
        additional_kwargs["reasoning"] = reasoning

    return response.model_copy(update={
        "content": stripped,
        "additional_kwargs": additional_kwargs,
    })
