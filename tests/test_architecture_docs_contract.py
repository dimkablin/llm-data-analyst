from __future__ import annotations

from pathlib import Path

CURRENT_ARCHITECTURE_DOCS = (
    Path("docs/diagram_01_architecture.md"),
    Path("docs/diagram_02_request_lifecycle.md"),
    Path("docs/diagram_03_agent_statemachine.md"),
    Path("docs/diagram_09_thinking_stream.md"),
)

PUBLIC_ARCHITECTURE_SURFACES = {
    Path("README.md"): (
        "QueryExecutionService",
        "AgentRunner",
        "prepare_context",
        "типизированные инструменты",
        "MCP",
    ),
    Path("frontend/src/app/pages/Platform.tsx"): (
        "Среда выполнения LangGraph",
        "QueryExecutionService",
        "типизированные инструменты",
        "доменные расширения MCP",
    ),
    Path("frontend/src/app/pages/Technical.tsx"): (
        "QueryExecutionService",
        "prepare_context",
        "типизированными инструментами",
        "MCP-адаптеры",
    ),
    Path("frontend/src/app/components/TechnicalDiagrams.tsx"): (
        "QueryExecutionService",
        "prepare_context -> agent -> finalize",
        "Типизированные инструменты + MCP",
    ),
}


def _current_architecture_docs_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in CURRENT_ARCHITECTURE_DOCS)


def test_current_architecture_docs_describe_prepare_context_runtime() -> None:
    text = _current_architecture_docs_text()

    required_terms = {
        "prepare_context",
        "AgentContextBuilder",
        "QueryExecutionService",
        "AgentRunner",
        "DomainExtensionRegistry",
        "typed tools",
        "MCP",
    }
    assert all(term in text for term in required_terms)


def test_current_architecture_docs_do_not_describe_removed_dispatch_shortcuts() -> None:
    lowered = _current_architecture_docs_text().lower()
    normalized = (
        lowered
        .replace("→", "->")
        .replace("\n", " ")
        .replace("  ", " ")
    )

    removed_runtime_terms = {
        "_quick_route",
        "dispatch_node",
        "_dispatch_node",
        "dispatch node",
        "dispatch -> agent -> finalize",
        "keyword pre-check",
        "chat bypass",
        "summary bypass",
        "runner.py:",
    }
    assert not any(term in normalized for term in removed_runtime_terms)


def test_public_architecture_surfaces_describe_current_runtime_boundary() -> None:
    removed_runtime_terms = {
        "dispatch -> agent -> finalize",
        "dispatch → agent → finalize",
        "dispatch_node",
        "_dispatch_node",
        "keyword pre-check",
        "chat bypass",
        "summary bypass",
    }

    for path, required_terms in PUBLIC_ARCHITECTURE_SURFACES.items():
        text = path.read_text(encoding="utf-8")
        normalized = text.lower().replace("→", "->")

        for term in required_terms:
            assert term in text, f"{path} must mention {term}"

        for term in removed_runtime_terms:
            assert term not in normalized, f"{path} must not describe removed runtime term {term}"
