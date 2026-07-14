from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pandas as pd
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
from backend.agent.services.agent_prompt_context import AgentPromptContextBuilder
from backend.core.config import Settings
from backend.skills.registry import SkillRegistry


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


def test_context_builder_returns_permission_denied_state(tmp_path) -> None:
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

    assert prepared.state_update["done"] is True
    assert prepared.state_update["stop_reason"] == "skill_permission_denied"
    assert "response" in prepared.state_update


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
    assert prepared.state_update["skill_execution_requirements"] == []


def test_context_builder_prompts_agent_about_disabled_capability_tools(tmp_path) -> None:
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
    prompt_block = prepared.state_update["capability_context"]["prompt_block"]
    assert "If the user asks for an unavailable capability" in prompt_block
    assert "required tools: `forecast_tool`" in prompt_block
    assert "required tools: `anomaly_planfact_tool`" in prompt_block


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


def test_prompt_context_builder_has_no_matched_skills_hint_api(
    tmp_path,
) -> None:
    builder = AgentPromptContextBuilder(
        skill_registry=SkillRegistry.from_path(tmp_path),
        enabled_analytical_skill_ids=None,
    )

    assert not hasattr(builder, "matched_analytical_skills_hint")


def test_skill_registry_execution_requirements_do_not_accept_prompt_for_matching(
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

    assert registry.execution_requirements_for_prompt(selected_skill_ids=[]) == ()
