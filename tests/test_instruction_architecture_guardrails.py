from __future__ import annotations

from pathlib import Path

from backend.instructions import InstructionKind, read_instruction_document
from backend.tools.catalog import KNOWN_TOOL_KEYS
from backend.tools.instructions import get_default_tool_instruction_registry

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_every_catalog_tool_has_top_level_tool_markdown_with_explicit_default() -> None:
    registry = get_default_tool_instruction_registry()
    loaded_tool_keys = {document.metadata.tool_key for document in registry.list_tools()}

    assert KNOWN_TOOL_KEYS <= loaded_tool_keys

    for tool_key in sorted(KNOWN_TOOL_KEYS):
        path = PROJECT_ROOT / "tools" / tool_key / "TOOL.md"
        assert path.exists(), f"{tool_key} must declare tools/{tool_key}/TOOL.md"
        text = path.read_text(encoding="utf-8")
        assert "\nenabled_by_default:" in text, f"{path} must declare enabled_by_default"


def test_project_skills_are_analytical_and_declare_explicit_default() -> None:
    skill_paths = sorted((PROJECT_ROOT / "skills").glob("*/SKILL.md"))
    assert skill_paths

    for path in skill_paths:
        document = read_instruction_document(
            path,
            default_id=path.parent.name,
            default_kind=InstructionKind.ANALYTICAL.value,
        )
        assert document.metadata.kind == InstructionKind.ANALYTICAL.value
        assert "\nenabled_by_default:" in path.read_text(encoding="utf-8")


def test_tool_prompts_are_not_defined_in_agent_prompt_module() -> None:
    import backend.agent.prompts as prompts

    forbidden_names = (
        "search_tool_prompt",
        "forecast_tool_prompt",
        "anomaly_planfact_tool_prompt",
        "pandas_tool_prompt",
        "plotly_tool_prompt",
        "value_tool_prompt",
    )
    for name in forbidden_names:
        assert not hasattr(prompts, name)


def test_general_analytics_workflow_is_prompt_only_not_runtime_context() -> None:
    assert not (PROJECT_ROOT / "backend" / "agent" / "workflow_context.py").exists()


def test_backend_docker_image_includes_tool_instruction_docs() -> None:
    source = (PROJECT_ROOT / "Dockerfile.backend").read_text(encoding="utf-8")

    assert "COPY tools ./tools" in source


def test_value_tool_is_not_exposed_as_callable_runtime_tool() -> None:
    from backend.agent.prompts import execution_agent_prompt
    from backend.tools.catalog import KNOWN_TOOL_KEYS
    from backend.tools.registry import ToolRegistry

    registry = get_default_tool_instruction_registry()
    registered_docs = {document.metadata.tool_key for document in registry.list_tools()}
    factory_keys = set(ToolRegistry.from_services()._factories)

    assert "value_tool" not in KNOWN_TOOL_KEYS
    assert "value_tool" not in registered_docs
    assert "value_tool" not in factory_keys
    assert "value_tool" not in execution_agent_prompt
