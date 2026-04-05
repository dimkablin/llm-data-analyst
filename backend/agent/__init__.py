from __future__ import annotations

from importlib import import_module

__all__ = [
    "THINKING_RE",
    "AgentProgressCollector",
    "AgentResponse",
    "AgentRunner",
    "LLMTextCollector",
    "PhaseCollector",
    "TokenStreamCallbackHandler",
    "ToolCollector",
    "_is_llm_transport_failure",
    "extract_thinking",
    "strip_thinking",
]


def __getattr__(name: str):
    if name in {"AgentResponse", "AgentRunner", "_is_llm_transport_failure"}:
        module = import_module("backend.agent.runner")
        return getattr(module, name)
    if name in {
        "AgentProgressCollector",
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
