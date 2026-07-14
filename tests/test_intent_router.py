"""Intent routing guardrails for the generic agent runtime.

The runtime must not use keyword-based shortcuts for chat, summary, report, or
domain flows. Summary/report generation are regular tools; the LLM decides when
to call them from the generic tool loop.
"""
from __future__ import annotations

from backend.agent.runner import AgentRunner
from backend.core import Settings
from backend.tools.context import ToolBuildContext
from backend.tools.registry import ToolRegistry


class IntentRouterContractTests:
    def test_runner_does_not_expose_keyword_quick_route(self) -> None:
        assert not hasattr(AgentRunner, "_quick_route")


class GenerationToolsContractTests:
    def test_summary_and_report_are_regular_registry_tools(self) -> None:
        registry = ToolRegistry.from_services()
        tools = registry.build_tools(ToolBuildContext(settings=Settings()))
        names = {getattr(tool, "name", "") for tool in tools}

        assert "generate_summary_tool" in names
        assert "generate_report_tool" in names
