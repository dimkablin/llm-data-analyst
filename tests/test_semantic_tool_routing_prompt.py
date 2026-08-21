from types import SimpleNamespace

from backend.agent.graph.nodes.agent import _with_semantic_metric_footer
from backend.agent.runner import AgentRunner
from backend.agent.services.message_builder import (
    ExecutionSystemPromptRequest,
    build_execution_system_prompt,
)
from backend.tools.capabilities import build_runtime_capability_context


def test_runtime_capability_context_prompts_typed_semantic_resolution() -> None:
    context = build_runtime_capability_context(
        available_tool_keys=[
            "semantic_catalog_read_tool",
            "semantic_catalog_edit_tool",
            "sql_tool",
            "plotly_tool",
        ],
        has_dataframe=False,
        has_db_source=True,
    )

    prompt = context["prompt_block"]
    assert "semantic_catalog_read" in context["available_capability_keys"]
    assert "metadata absent from the injected context" in prompt
    assert "missing, incomplete, or ambiguous terms, formulas, and relationships" in prompt
    assert "does not calculate data" in prompt
    assert "never supply an executable business formula" in prompt


def test_execution_prompt_keeps_semantic_metadata_and_analysis_routes_separate() -> None:
    runner = AgentRunner()
    prompt = build_execution_system_prompt(
        ExecutionSystemPromptRequest(
            settings=runner.settings,
            skill_registry=runner.skill_registry,
            enabled_analytical_skill_ids=runner.enabled_analytical_skill_ids,
            capability_context={
                "source_mode": "db",
                "tool_descriptions": "",
                "available_tool_keys": [
                    "planner_tool",
                    "semantic_catalog_read_tool",
                    "semantic_catalog_edit_tool",
                    "semantic_catalog_generate_tool",
                    "sql_tool",
                    "plotly_tool",
                ],
            },
        )
    )

    assert "semantic metadata" in prompt
    assert "Raw-data requests route" in prompt
    assert "mode `semantic_query`" in prompt
    assert "synthesize conclusion, interpretation, caveats" in prompt
    assert "answer_ready" not in prompt


def test_agent_runner_passes_semantic_services_to_runtime_dependencies() -> None:
    catalog_service = object()
    generation_service = object()
    runner = AgentRunner(
        semantic_catalog_service=catalog_service,
        semantic_generation_service=generation_service,
    )

    assert runner.dependencies.semantic_catalog_service is catalog_service
    assert runner.dependencies.semantic_generation_service is generation_service


def _artifact_with_metric() -> SimpleNamespace:
    return SimpleNamespace(
        meta={
            "semantic_metric": {
                "metrics": [
                    {
                        "key": "service_resolution_index",
                        "name": "Service resolution index",
                        "formula": "SUM(resolved) / NULLIF(SUM(opened), 0) * 7.5",
                    }
                ]
            }
        }
    )


def test_semantic_metric_footer_uses_executed_artifact_provenance() -> None:
    result = _with_semantic_metric_footer(
        "Monthly values are shown in the table.",
        [_artifact_with_metric()],
    )

    assert result.endswith(
        "Semantic metric: Service resolution index; Formula: SUM(resolved) / NULLIF(SUM(opened), 0) * 7.5"
    )


def test_semantic_metric_footer_is_not_duplicated() -> None:
    text = "Service resolution index uses SUM(resolved) / NULLIF(SUM(opened), 0) * 7.5"

    assert _with_semantic_metric_footer(text, [_artifact_with_metric()]) == text
