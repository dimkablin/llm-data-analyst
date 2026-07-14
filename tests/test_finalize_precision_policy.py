from __future__ import annotations

from backend.agent.contracts import AnalysisTaskContract, RequiredOutput
from backend.agent.graph.nodes.finalize import finalize_node
from backend.agent.models import AgentResponse
from backend.skills.contracts import SkillExecutionContract, SkillExecutionRequirement


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


def test_finalize_preserves_answer_when_task_contract_would_fail() -> None:
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
            "task_contract": AnalysisTaskContract(
                required_outputs=[
                    RequiredOutput(kind="brief", reason="user asked for explanation")
                ]
            ),
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


def test_finalize_preserves_answer_when_skill_contract_would_fail() -> None:
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
                SkillExecutionRequirement(
                    skill_id="revenue_skill",
                    execution_contract=SkillExecutionContract(
                        required_tools=("missing_tool",),
                    ),
                )
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
