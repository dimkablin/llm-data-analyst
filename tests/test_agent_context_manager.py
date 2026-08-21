from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pandas as pd
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from backend.agent.context_manager import (
    AgentContextBuilder,
    AgentContextManagerConfig,
    AgentContextRequest,
    ContextBudgetPolicy,
    ContextRetrievalPolicy,
    ContextRetrievalResult,
    RetrievedContextMessage,
)
from backend.agent.runner import AgentRunner
from backend.core.config import Settings
from backend.skills.registry import SkillRegistry
from backend.tools.context import ToolBuildContext
from backend.tools.registry import ToolRegistry


class _WorkingMemoryCaptureFactory:
    key = "capture_working_memory"

    def __init__(self) -> None:
        self.working_memory = None

    def is_available(self, _context: ToolBuildContext) -> bool:
        return True

    def build(self, context: ToolBuildContext) -> StructuredTool:
        self.working_memory = context.working_memory
        return StructuredTool.from_function(
            name=self.key,
            description="Capture the per-request working memory.",
            func=lambda: "ok",
        )


def _write_skill(tmp_path, folder: str, content: str) -> None:
    skill_dir = tmp_path / folder
    skill_dir.mkdir(exist_ok=True)
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")


def test_context_builder_prepares_agent_state_without_runner_private_access(tmp_path) -> None:
    runner = AgentRunner(
        settings=replace(
            Settings(),
            storage_dir=str(tmp_path),
            llm_warmup_enabled=False,
        )
    )
    builder = AgentContextBuilder(dependencies=runner.dependencies)
    state = {
        "prompt": "summarize current session",
        "df": None,
        "history": [],
        "trace_context": {"session_id": "ctx-session"},
        "session_source": {},
        "selected_skill_ids": [],
    }
    sandbox = MagicMock()
    sandbox.ensure_storage_dir.return_value = None

    with patch("backend.agent.context_manager.SandboxManager") as sandbox_manager:
        sandbox_manager.get_instance.return_value.get_or_create.return_value = sandbox
        prepared = builder.build(AgentContextRequest(state=state))

    assert isinstance(prepared, BaseModel)
    assert prepared.state_update["done"] is False
    assert prepared.state_update["prompt"] == "summarize current session"
    assert prepared.state_update["working_memory"].goal == "summarize current session"
    assert prepared.context_budget.strategy == "disabled"
    assert isinstance(prepared.retrieved_context, ContextRetrievalResult)
    assert prepared.retrieved_context.messages == []


def test_context_builder_binds_tools_to_the_same_per_request_working_memory(tmp_path) -> None:
    runner = AgentRunner(
        settings=replace(
            Settings(),
            storage_dir=str(tmp_path),
            llm_warmup_enabled=False,
        )
    )
    capture = _WorkingMemoryCaptureFactory()
    runner.dependencies.tool_registry = ToolRegistry([capture], capability_definitions=[])
    builder = AgentContextBuilder(dependencies=runner.dependencies)

    prepared = builder.build(
        AgentContextRequest(
            state={
                "prompt": "analyze current data",
                "df": None,
                "history": [],
                "trace_context": {"session_id": "ctx-working-memory"},
                "session_source": {},
                "selected_skill_ids": [],
            }
        )
    )

    assert capture.working_memory is prepared.state_update["working_memory"]


def test_context_builder_keeps_selected_skill_requirements_as_prompt_only(tmp_path) -> None:
    _write_skill(
        tmp_path,
        "cohort_analysis",
        """---
name: Cohort Analysis
description: Run cohort retention analysis
kind: analytical
triggers: cohort, retention
---
## Cohort Analysis
### Algorithm
Compute retention by cohort.
### Rules
Use loaded data only.
### Required tools
- pandas_tool
""",
    )
    runner = AgentRunner(
        settings=replace(
            Settings(),
            skills_dir=str(tmp_path),
            storage_dir=str(tmp_path),
            llm_warmup_enabled=False,
        ),
        allowed_tool_keys={"get_tool_instructions"},
    )
    builder = AgentContextBuilder(dependencies=runner.dependencies)
    state = {
        "prompt": "cohort analysis",
        "df": None,
        "history": [],
        "trace_context": {"session_id": "ctx-session"},
        "session_source": {},
        "selected_skill_ids": ["cohort_analysis"],
    }

    prepared = builder.build(AgentContextRequest(state=state))

    assert prepared.state_update["done"] is False
    assert prepared.state_update["stop_reason"] == ""
    assert "response" not in prepared.state_update
    assert "### Required tools" in runner.skill_registry.get("cohort_analysis").instructions_markdown
    tool_names = [tool.name for tool in prepared.state_update["tools"]]
    assert "get_tool_instructions" in tool_names
    assert "planner_tool" not in tool_names


def test_context_builder_does_not_apply_skill_contract_from_trigger_only(tmp_path) -> None:
    _write_skill(
        tmp_path,
        "cohort_analysis",
        """---
name: Cohort Analysis
description: Run cohort retention analysis
kind: analytical
triggers: cohort, retention
---
## Cohort Analysis
### Algorithm
Compute retention by cohort.
### Rules
Use loaded data only.
### Required tools
- pandas_tool
""",
    )
    runner = AgentRunner(
        settings=replace(
            Settings(),
            skills_dir=str(tmp_path),
            storage_dir=str(tmp_path),
            llm_warmup_enabled=False,
        ),
        allowed_tool_keys={"get_tool_instructions"},
    )
    builder = AgentContextBuilder(dependencies=runner.dependencies)
    state = {
        "prompt": "cohort retention analysis",
        "df": None,
        "history": [],
        "trace_context": {"session_id": "ctx-trigger-only"},
        "session_source": {},
        "selected_skill_ids": [],
    }
    sandbox = MagicMock()
    sandbox.ensure_storage_dir.return_value = None

    with patch("backend.agent.context_manager.SandboxManager") as sandbox_manager:
        sandbox_manager.get_instance.return_value.get_or_create.return_value = sandbox
        prepared = builder.build(AgentContextRequest(state=state))

    assert prepared.state_update["done"] is False
    assert prepared.state_update["stop_reason"] == ""
    assert "skill_execution_requirements" not in prepared.state_update


def test_context_builder_does_not_create_runtime_contract_for_selected_skill(tmp_path) -> None:
    _write_skill(
        tmp_path,
        "chart_analysis",
        """---
name: Chart Analysis
description: Analyze a dataframe and chart it
kind: analytical
triggers: chart
---
## Chart Analysis
### Algorithm
Choose the requested branch.
### Rules
Use active capabilities.
### Required capabilities
- chart
""",
    )
    runner = AgentRunner(
        settings=replace(
            Settings(),
            skills_dir=str(tmp_path),
            storage_dir=str(tmp_path),
            llm_warmup_enabled=False,
        ),
        allowed_tool_keys={"pandas_tool", "plotly_tool", "get_tool_instructions"},
    )
    builder = AgentContextBuilder(dependencies=runner.dependencies)
    state = {
        "prompt": "summarize the current rows",
        "df": pd.DataFrame({"value": [1, 2]}),
        "history": [],
        "trace_context": {"session_id": "ctx-selected-skill"},
        "session_source": {},
        "selected_skill_ids": ["chart_analysis"],
    }

    prepared = builder.build(AgentContextRequest(state=state))

    assert prepared.state_update["done"] is False
    assert "skill_execution_requirements" not in prepared.state_update
    assert "task_contract" not in prepared.state_update


def test_context_builder_exposes_only_active_capability_bindings(tmp_path) -> None:
    runner = AgentRunner(
        settings=replace(
            Settings(),
            storage_dir=str(tmp_path),
            llm_warmup_enabled=False,
        ),
        allowed_tool_keys={"pandas_tool", "plotly_tool", "get_tool_instructions"},
    )
    builder = AgentContextBuilder(dependencies=runner.dependencies)
    state = {
        "prompt": "Спрогнозируй продажи на 3 месяца",
        "df": pd.DataFrame({"month": ["2026-01"], "sales": [10]}),
        "history": [],
        "trace_context": {"session_id": "ctx-disabled-forecast"},
        "session_source": {},
        "selected_skill_ids": [],
    }

    prepared = builder.build(AgentContextRequest(state=state))

    assert prepared.state_update["done"] is False
    assert prepared.state_update["stop_reason"] == ""
    assert "task_contract" not in prepared.state_update
    prompt_block = prepared.state_update["capability_context"]["prompt_block"]
    assert "complete action surface for the current run" in prompt_block
    assert "`forecast_tool`" not in prompt_block
    assert "`anomaly_planfact_tool`" not in prompt_block
    assert prepared.state_update["capability_context"]["unavailable_capability_keys"] == ["forecast"]


def test_context_manager_rejects_incomplete_budget_policy() -> None:
    try:
        ContextBudgetPolicy(strategy="token_limit")
    except ValueError as exc:
        assert "max_context_tokens" in str(exc)
    else:
        raise AssertionError("token_limit policy must require max_context_tokens")


def test_context_manager_rejects_incomplete_vector_retrieval_policy() -> None:
    try:
        ContextRetrievalPolicy(strategy="vector_similarity")
    except ValueError as exc:
        assert "top_k" in str(exc)
    else:
        raise AssertionError("vector_similarity policy must require positive top_k")


def test_context_builder_records_future_budget_and_retrieval_contracts(tmp_path) -> None:
    runner = AgentRunner(
        settings=replace(
            Settings(),
            storage_dir=str(tmp_path),
            llm_warmup_enabled=False,
        )
    )
    builder = AgentContextBuilder(
        dependencies=runner.dependencies,
        context_config=AgentContextManagerConfig(
            budget_policy=ContextBudgetPolicy(
                strategy="token_limit",
                max_context_tokens=4096,
                reserved_response_tokens=512,
            ),
            retrieval_policy=ContextRetrievalPolicy(
                strategy="vector_similarity",
                top_k=3,
                min_score=0.25,
                provider="future-vector-store",
            ),
        ),
    )
    state = {
        "prompt": "compare this quarter with historical context",
        "df": None,
        "history": [{"role": "user", "content": "older question"}],
        "trace_context": {"session_id": "ctx-policy-session"},
        "session_source": {},
        "selected_skill_ids": [],
    }
    sandbox = MagicMock()
    sandbox.ensure_storage_dir.return_value = None

    with patch("backend.agent.context_manager.SandboxManager") as sandbox_manager:
        sandbox_manager.get_instance.return_value.get_or_create.return_value = sandbox
        prepared = builder.build(AgentContextRequest(state=state))

    assert prepared.context_budget.strategy == "token_limit"
    assert prepared.context_budget.status == "not_configured"
    assert prepared.context_budget.max_context_tokens == 4096
    assert prepared.context_budget.reserved_response_tokens == 512
    assert prepared.retrieved_context.strategy == "vector_similarity"
    assert prepared.retrieved_context.requested_top_k == 3
    assert prepared.retrieved_context.provider == "future-vector-store"
    assert prepared.retrieved_context.skipped_reason == "retriever_not_configured"
    assert prepared.state_update["context_budget"] == prepared.context_budget
    assert prepared.state_update["retrieved_context"] == prepared.retrieved_context


def test_context_builder_accepts_injected_history_retriever(tmp_path) -> None:
    class FakeRetriever:
        def retrieve(self, **kwargs):
            assert kwargs["top_k"] == 1
            return [
                RetrievedContextMessage(
                    role="assistant",
                    content="Relevant prior answer",
                    score=0.91,
                    source_index=2,
                )
            ]

    runner = AgentRunner(
        settings=replace(
            Settings(),
            storage_dir=str(tmp_path),
            llm_warmup_enabled=False,
        )
    )
    builder = AgentContextBuilder(
        dependencies=runner.dependencies,
        context_config=AgentContextManagerConfig(
            retrieval_policy=ContextRetrievalPolicy(
                strategy="vector_similarity",
                top_k=1,
            ),
        ),
    )
    builder.history_retriever.retriever = FakeRetriever()
    state = {
        "prompt": "use relevant context",
        "df": None,
        "history": [{"role": "user", "content": "older question"}],
        "trace_context": {"session_id": "ctx-retriever-session"},
        "session_source": {},
        "selected_skill_ids": [],
    }
    sandbox = MagicMock()
    sandbox.ensure_storage_dir.return_value = None

    with patch("backend.agent.context_manager.SandboxManager") as sandbox_manager:
        sandbox_manager.get_instance.return_value.get_or_create.return_value = sandbox
        prepared = builder.build(AgentContextRequest(state=state))

    assert prepared.retrieved_context.messages[0].content == "Relevant prior answer"
    assert prepared.retrieved_context.messages[0].score == 0.91


def test_retrieved_context_message_score_is_provider_neutral() -> None:
    message = RetrievedContextMessage(
        role="assistant",
        content="Prior context",
        score=-0.2,
        source_index=0,
    )

    assert message.score == -0.2


def test_skill_registry_has_no_runtime_execution_requirements_api(
    tmp_path,
) -> None:
    _write_skill(
        tmp_path,
        "cohort_analysis",
        """---
name: Cohort Analysis
description: Run cohort retention analysis
kind: analytical
triggers: cohort, retention
enabled_by_default: true
---
## Cohort Analysis
### Algorithm
Compute retention by cohort.
### Rules
Use loaded data only.
""",
    )
    registry = SkillRegistry.from_path(tmp_path)

    assert not hasattr(registry, "execution_requirements_for_selection")
