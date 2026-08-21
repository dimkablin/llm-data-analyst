from __future__ import annotations

from backend.agent.graph.nodes.finalize import finalize_node
from backend.agent.models import AgentResponse


def test_finalize_preserves_answer_when_review_would_fail() -> None:
    response = AgentResponse(
        final_text="Revenue increased by 10% in Q2 after the pricing change.",
        reasoning="agent reasoning",
        artifacts=[],
        route="analysis",
        tool_calls=1,
        tool_names=["sql_tool"],
    )

    result = finalize_node(
        {
            "response": response,
            "prompt": "Analyze revenue change",
            "callbacks": [],
            "trace_context": {},
            "step_index": 1,
            "max_steps": 1,
        },
        object(),
    )

    finalized = result["response"]
    assert finalized.final_text == "Revenue increased by 10% in Q2 after the pricing change."
    assert finalized.reasoning == "agent reasoning"


def test_finalize_ignores_legacy_task_contract_requirements() -> None:
    response = AgentResponse(
        final_text="Revenue increased by 10% in Q2 after the pricing change.",
        reasoning="agent reasoning",
        artifacts=[],
        route="analysis",
        tool_calls=0,
        tool_names=[],
    )

    result = finalize_node(
        {
            "response": response,
            "prompt": "Explain revenue change",
            "task_contract": {
                "required_outputs": [
                    {"kind": "plot", "reason": "user asked for a plot"}
                ],
                "required_capabilities": ["chart"],
            },
            "callbacks": [],
            "trace_context": {},
            "step_index": 1,
            "max_steps": 1,
        },
        object(),
    )

    finalized = result["response"]
    assert finalized.final_text == "Revenue increased by 10% in Q2 after the pricing change."
    assert finalized.reasoning == "agent reasoning"


def test_finalize_ignores_legacy_skill_contract_requirements() -> None:
    response = AgentResponse(
        final_text="Revenue increased by 10% in Q2 after the pricing change.",
        reasoning="agent reasoning",
        artifacts=[],
        route="analysis",
        tool_calls=1,
        tool_names=["sql_tool"],
    )

    result = finalize_node(
        {
            "response": response,
            "prompt": "Analyze revenue change",
            "skill_execution_requirements": [
                {
                    "skill_id": "revenue_skill",
                    "required_tools": ["missing_tool"],
                    "required_artifacts": ["plot"],
                }
            ],
            "callbacks": [],
            "trace_context": {},
            "step_index": 1,
            "max_steps": 1,
        },
        object(),
    )

    finalized = result["response"]
    assert finalized.final_text == "Revenue increased by 10% in Q2 after the pricing change."
    assert finalized.reasoning == "agent reasoning"
