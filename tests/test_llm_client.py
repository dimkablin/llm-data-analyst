"""Tests for make_reasoning_llm factory and ReasoningChatOllama adapter."""
from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from backend.agent.llm_client import (
    ReasoningChatOllama,
    ReasoningChatOpenAI,
    _remap_ollama_reasoning,
    make_reasoning_llm,
)

# ── _remap_ollama_reasoning ────────────────────────────────────────────────────

def test_remap_copies_reasoning_content_to_reasoning():
    msg = AIMessage(content="hello", additional_kwargs={"reasoning_content": "some thinking"})
    result = _remap_ollama_reasoning(msg)
    assert result.additional_kwargs["reasoning"] == "some thinking"
    assert result.additional_kwargs["reasoning_content"] == "some thinking"


def test_remap_noop_when_no_reasoning_content():
    msg = AIMessage(content="hello", additional_kwargs={})
    result = _remap_ollama_reasoning(msg)
    assert result is msg  # same object, no copy


def test_remap_noop_when_reasoning_already_present():
    msg = AIMessage(
        content="hello",
        additional_kwargs={"reasoning_content": "new", "reasoning": "existing"},
    )
    result = _remap_ollama_reasoning(msg)
    assert result.additional_kwargs["reasoning"] == "existing"  # not overwritten


# ── make_reasoning_llm routing ─────────────────────────────────────────────────

def test_factory_returns_ollama_for_ollama_provider():
    llm = make_reasoning_llm(
        provider="ollama",
        model="qwen3:14b",
        base_url="http://localhost:11434",
        enable_thinking=False,
    )
    assert isinstance(llm, ReasoningChatOllama)


def test_factory_returns_ollama_case_insensitive():
    llm = make_reasoning_llm(
        provider="Ollama",
        model="qwen3:14b",
        base_url="http://localhost:11434",
        enable_thinking=False,
    )
    assert isinstance(llm, ReasoningChatOllama)


def test_factory_returns_openai_for_vllm():
    llm = make_reasoning_llm(
        provider="vllm",
        model="qwen3:14b",
        base_url="http://localhost:8000",
        api_key="no-key",
        enable_thinking=False,
    )
    assert isinstance(llm, ReasoningChatOpenAI)


def test_factory_returns_openai_for_none_provider():
    llm = make_reasoning_llm(
        provider=None,
        model="gpt-4",
        base_url="https://api.openai.com",
        api_key="sk-test",
        enable_thinking=False,
    )
    assert isinstance(llm, ReasoningChatOpenAI)


def test_factory_ollama_sets_reasoning_false_when_disabled():
    llm = make_reasoning_llm(
        provider="ollama",
        model="qwen3:14b",
        base_url="http://localhost:11434",
        enable_thinking=False,
    )
    assert isinstance(llm, ReasoningChatOllama)
    assert llm.reasoning is False


def test_factory_ollama_sets_reasoning_true_when_enabled():
    llm = make_reasoning_llm(
        provider="ollama",
        model="qwen3:14b",
        base_url="http://localhost:11434",
        enable_thinking=True,
    )
    assert isinstance(llm, ReasoningChatOllama)
    assert llm.reasoning is True


def test_factory_ollama_drops_presence_penalty():
    # Should not raise even though ChatOllama doesn't support presence_penalty
    llm = make_reasoning_llm(
        provider="ollama",
        model="qwen3:14b",
        base_url="http://localhost:11434",
        presence_penalty=0.5,  # should be silently dropped
    )
    assert isinstance(llm, ReasoningChatOllama)


# ── ReasoningChatOllama stream remapping ──────────────────────────────────────

def test_reasoning_chat_ollama_stream_remaps_reasoning_content():
    """Verify that stream() yields chunks with reasoning_content → reasoning remapped."""
    chunk = AIMessageChunk(
        content="3",
        additional_kwargs={"reasoning_content": "let me think..."},
    )

    llm = ReasoningChatOllama(model="qwen3:14b", base_url="http://localhost:11434")
    with patch.object(type(llm).__bases__[0], "stream", return_value=iter([chunk])):
        results = list(llm.stream("test"))

    assert len(results) == 1
    assert results[0].additional_kwargs.get("reasoning") == "let me think..."


# ── Live tests (require running Ollama) ───────────────────────────────────────

@pytest.mark.live
def test_live_ollama_reasoning_false_no_thinking():
    """reasoning=False → model does not think, no reasoning in response."""
    llm = make_reasoning_llm(
        provider="ollama",
        model="qwen3:14b",
        base_url="http://localhost:11434",
        enable_thinking=False,
        streaming=False,
        max_tokens=200,
    )
    from langchain_core.messages import HumanMessage
    response = llm.invoke([HumanMessage(content="how many r in strawberry? one word")])
    ak = response.additional_kwargs
    reasoning = ak.get("reasoning", "") or ak.get("reasoning_content", "")
    assert not reasoning, f"Expected no reasoning, got: {reasoning[:100]}"
    assert response.content.strip()


@pytest.mark.live
def test_live_ollama_reasoning_true_separate_content():
    """reasoning=True → content clean, reasoning in additional_kwargs['reasoning']."""
    llm = make_reasoning_llm(
        provider="ollama",
        model="qwen3:14b",
        base_url="http://localhost:11434",
        enable_thinking=True,
        streaming=False,
        max_tokens=2000,
    )
    from langchain_core.messages import HumanMessage
    response = llm.invoke([HumanMessage(content="how many r in strawberry? one word")])
    assert response.content.strip()
    assert "<think>" not in response.content
    reasoning = response.additional_kwargs.get("reasoning", "")
    assert reasoning, "Expected reasoning in additional_kwargs['reasoning']"


@pytest.mark.live
def test_live_ollama_stream_works():
    """Streaming via ReasoningChatOllama yields chunks without error."""
    llm = make_reasoning_llm(
        provider="ollama",
        model="qwen3:14b",
        base_url="http://localhost:11434",
        enable_thinking=False,
        streaming=True,
        max_tokens=200,
    )
    from langchain_core.messages import HumanMessage
    chunks = list(llm.stream([HumanMessage(content="say 'hi'")]))
    assert chunks
    full_content = "".join(c.content for c in chunks if isinstance(c.content, str))
    assert full_content.strip()
