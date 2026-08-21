from backend.agent.runner import AgentRunner
from backend.agent.services.message_builder import (
    ExecutionSystemPromptRequest,
    build_execution_context_messages,
)


def test_top_k_metric_candidates_are_visible_to_main_agent() -> None:
    semantic_prompt = """SEMANTIC DATA CONTEXT
top_k_metric_candidates:
- key=service_resolution_index; name=Service resolution index; score=0.91
- key=service_volume; name=Service volume; score=0.78
"""
    runner = AgentRunner()
    request = ExecutionSystemPromptRequest(
        settings=runner.settings,
        skill_registry=runner.skill_registry,
        enabled_analytical_skill_ids=runner.enabled_analytical_skill_ids,
        session_source={"semantic_context_prompt": semantic_prompt},
    )

    context = "\n".join(str(message.content) for message in build_execution_context_messages(request))

    assert "top_k_metric_candidates" in context
    assert "service_resolution_index" in context
    assert "service_volume" in context
