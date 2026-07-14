from __future__ import annotations

from importlib import import_module

__all__ = [
    "THINKING_RE",
    "AgentProgressCollector",
    "AgentResponse",
    "AgentRunRequest",
    "AgentRunResult",
    "AgentRunner",
    "AgentRuntimeEffects",
    "ContextUsageCollector",
    "LLMTextCollector",
    "PhaseCollector",
    "QueryCacheEntry",
    "TokenStreamCallbackHandler",
    "ToolCollector",
    "_is_llm_transport_failure",
    "extract_thinking",
    "strip_thinking",
]


def __getattr__(name: str):
    if name == "AgentRunner":
        module = import_module("backend.agent.runner")
        return getattr(module, name)
    if name == "_is_llm_transport_failure":
        module = import_module("backend.agent.tool_loop")
        return getattr(module, name)
    if name in {"AgentResponse", "AgentRuntimeEffects", "QueryCacheEntry"}:
        module = import_module("backend.agent.models")
        return getattr(module, name)
    if name in {"AgentRunRequest", "AgentRunResult"}:
        module = import_module("backend.agent.runtime_contracts")
        return getattr(module, name)
    if name in {
        "AgentProgressCollector",
        "ContextUsageCollector",
        "LLMTextCollector",
        "PhaseCollector",
        "TokenStreamCallbackHandler",
        "ToolCollector",
        "THINKING_RE",
        "extract_thinking",
        "strip_thinking",
    }:
        module = import_module("backend.agent.callbacks")
        return getattr(module, name)
    raise AttributeError(f"module 'backend.agent' has no attribute {name!r}")
