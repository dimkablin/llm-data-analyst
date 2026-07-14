from pathlib import Path

from pydantic import BaseModel

from backend.agent.dependencies import AgentRuntimeDependencies
from backend.agent.runner import AgentRunner


def test_agent_graph_modules_exist_after_split() -> None:
    expected = [
        "backend/agent/state.py",
        "backend/agent/graph/builder.py",
        "backend/agent/graph/routing.py",
        "backend/agent/graph/nodes/prepare_context.py",
        "backend/agent/graph/nodes/agent.py",
        "backend/agent/graph/nodes/finalize.py",
    ]

    missing = [path for path in expected if not Path(path).is_file()]

    assert missing == []
    assert not Path("backend/agent/graph/nodes/dispatch.py").exists()


def test_route_after_prepare_context_contract() -> None:
    from backend.agent.graph.routing import route_after_prepare_context

    assert route_after_prepare_context({"done": True}) == "finalize"
    assert route_after_prepare_context({"response": object()}) == "finalize"
    assert route_after_prepare_context({}) == "agent"


def test_build_query_graph_returns_compilable_graph() -> None:
    from backend.agent.graph.builder import build_query_graph

    runner = AgentRunner()
    graph = build_query_graph(runner.dependencies)

    assert graph.compile() is not None


def test_context_manager_exposes_pydantic_contracts() -> None:
    from backend.agent.context_manager import (
        AgentContextBuilder,
        AgentContextManagerConfig,
        AgentContextRequest,
        AgentPreparedContext,
        ContextBudget,
        ContextBudgetPolicy,
        ContextRetrievalPolicy,
        ContextRetrievalResult,
        RetrievedContextMessage,
    )

    for contract in (
        AgentContextBuilder,
        AgentContextManagerConfig,
        AgentContextRequest,
        AgentPreparedContext,
        ContextBudget,
        ContextBudgetPolicy,
        ContextRetrievalResult,
        ContextRetrievalPolicy,
        RetrievedContextMessage,
    ):
        assert issubclass(contract, BaseModel)


def test_runner_dependencies_expose_context_services() -> None:
    from backend.agent.context_manager import AgentContextBuilder
    from backend.agent.services.agent_prompt_context import AgentPromptContextBuilder

    runner = AgentRunner()

    assert isinstance(runner.dependencies.context_builder, AgentContextBuilder)
    assert isinstance(runner.dependencies.prompt_context_builder, AgentPromptContextBuilder)


def test_runner_exposes_pydantic_dependency_container() -> None:
    runner = AgentRunner()

    assert isinstance(runner.dependencies, AgentRuntimeDependencies)
    assert isinstance(runner.dependencies, BaseModel)
    assert runner.dependencies.tool_registry is runner._tool_registry
    assert runner.dependencies.forecast_service is None
    assert runner.dependencies.anomaly_planfact_service is None


def test_agent_runtime_public_contracts_are_pydantic_models() -> None:
    from backend.agent.runtime_contracts import AgentRunRequest, AgentRunResult

    assert issubclass(AgentRunRequest, BaseModel)
    assert issubclass(AgentRunResult, BaseModel)
    assert {
        "df",
        "prompt",
        "history",
        "use_history",
        "include_reasoning",
        "callbacks",
        "trace_context",
        "session_source",
        "selected_skill_ids",
    }.issubset(AgentRunRequest.model_fields)
    assert {"response"}.issubset(AgentRunResult.model_fields)


def test_agent_run_cache_hit_preserves_result_contract() -> None:
    from backend.agent.models import AgentResponse
    from backend.agent.runtime_contracts import AgentRunRequest, AgentRunResult

    runner = AgentRunner()
    request = AgentRunRequest(
        prompt="cached answer",
        history=[],
        use_history=False,
        include_reasoning=False,
        callbacks=[],
        trace_context={"request_kind": "query"},
        session_source={},
        selected_skill_ids=[],
    )
    cached = AgentResponse(
        final_text="cached response",
        reasoning=None,
        artifacts=[],
        route="analysis",
    )
    cache_key = runner._query_cache_key(
        df=request.df,
        prompt=request.prompt,
        history=request.history,
        use_history=request.use_history,
        include_reasoning=request.include_reasoning,
        selected_skill_ids=request.selected_skill_ids,
    )
    runner._cache_set(cache_key, cached)

    result = runner.run(request)

    assert isinstance(result, AgentRunResult)
    assert result.response.final_text == "cached response"


def test_dependency_container_accepts_runtime_service_adapters() -> None:
    class RagServiceAdapter:
        pass

    rag_service = RagServiceAdapter()

    runner = AgentRunner(rag_service=rag_service)

    assert runner.dependencies.rag_service is rag_service


def test_tool_loop_request_is_pydantic_contract() -> None:
    from langchain_core.messages import HumanMessage

    from backend.agent.tool_loop import ToolLoopRequest

    runner = AgentRunner()
    request = ToolLoopRequest(
        settings=runner.settings,
        include_reasoning=False,
        tools=[],
        callbacks=[],
        max_iterations=1,
        messages=[HumanMessage(content="hello")],
    )

    assert isinstance(request, BaseModel)
    assert request.settings is runner.settings
