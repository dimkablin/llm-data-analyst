from __future__ import annotations

from pathlib import Path

import pytest

from backend.skills import SkillRegistry, SkillSelectionError, SkillValidationError
from backend.skills.models import Skill

# ---------------------------------------------------------------------------
# Task 1: Skill model two-level fields
# ---------------------------------------------------------------------------


def test_skill_core_markdown_field() -> None:
    skill = Skill(
        skill_id="my_skill",
        name="My Skill",
        description="Does things.",
        core_markdown="## API\nfoo() -> None",
        details_markdown=None,
        source_path="/fake/path",
    )
    assert skill.core_markdown == "## API\nfoo() -> None"
    assert skill.details_markdown is None
    assert skill.has_details is False


def test_skill_has_details_true_when_details_present() -> None:
    skill = Skill(
        skill_id="my_skill",
        name="My Skill",
        description="Does things.",
        core_markdown="## API\nfoo() -> None",
        details_markdown="## Examples\n```python\nfoo()\n```",
        source_path="/fake/path",
    )
    assert skill.has_details is True
    assert skill.details_markdown == "## Examples\n```python\nfoo()\n```"


def test_skill_instructions_markdown_backward_compat() -> None:
    skill = Skill(
        skill_id="my_skill",
        name="My Skill",
        description="Does things.",
        core_markdown="## API\nfoo() -> None",
        details_markdown=None,
        source_path="/fake/path",
    )
    assert skill.instructions_markdown == skill.core_markdown


def _write_skill(tmp_path: Path, folder: str, content: str) -> None:
    skill_dir = tmp_path / folder
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")


def _write_skill_with_details(
    tmp_path: Path, folder: str, skill_content: str, details_content: str
) -> None:
    skill_dir = tmp_path / folder
    skill_dir.mkdir(exist_ok=True)
    (skill_dir / "SKILL.md").write_text(skill_content, encoding="utf-8")
    (skill_dir / "DETAILS.md").write_text(details_content, encoding="utf-8")


# Minimal valid analytical skill content for use in tests that aren't about content
_VALID_ANALYTICAL = (
    "---\nname: {name}\ndescription: {desc}.\n---\n\n"
    "### Algorithm\n1. Step one → pandas_tool.\n\n"
    "### Rules\n- Rule one.\n"
)

# Minimal valid tool skill content
_VALID_TOOL = (
    "---\nname: {name}\ndescription: {desc}.\nkind: tool\ntool_key: {key}\n---\n\n"
    "### API\n{key}(x: int) -> None\n\n"
    "### Final result protocol\nLast expression must be tool_result.\n\n"
    "### Rules\n- Rule one.\n"
)


def test_registry_loads_markdown_skills_with_yaml_frontmatter(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "cohort_analysis",
        "---\n"
        "name: Cohort Analysis\n"
        "description: Analyze cohorts.\n"
        "triggers:\n"
        "  - retention\n"
        "  - ltv\n"
        "---\n\n"
        "### Algorithm\n\n"
        "Use pandas and plotly.\n"
        "```python\nprint('example')\n```\n\n"
        "### Rules\n- Always check retention.\n",
    )

    registry = SkillRegistry.from_path(tmp_path)
    skills = registry.list_skills()

    assert len(skills) == 1
    assert skills[0].skill_id == "cohort_analysis"
    assert skills[0].python_examples[0].code == "print('example')"
    assert "Algorithm" in skills[0].instructions_markdown


def test_registry_rejects_missing_frontmatter(tmp_path: Path) -> None:
    _write_skill(tmp_path, "broken", "## No frontmatter")

    with pytest.raises(SkillValidationError):
        SkillRegistry.from_path(tmp_path).load()


def test_registry_rejects_unknown_explicit_selection(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "cohort_analysis",
        _VALID_ANALYTICAL.format(name="Cohort Analysis", desc="Analyze cohorts"),
    )
    registry = SkillRegistry.from_path(tmp_path)

    with pytest.raises(SkillSelectionError):
        registry.resolve_selection(["missing_skill"])


def test_prompt_block_includes_only_explicitly_selected_skills(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "cohort_analysis",
        _VALID_ANALYTICAL.format(name="Cohort Analysis", desc="Analyze cohorts"),
    )
    _write_skill(
        tmp_path,
        "forecasting",
        _VALID_ANALYTICAL.format(name="Forecasting", desc="Forecast metrics"),
    )

    registry = SkillRegistry.from_path(tmp_path)
    prompt = registry.build_prompt_block(["forecasting"])

    assert "Forecasting" in prompt
    assert "Cohort Analysis" not in prompt
    assert "не выполнять напрямую" in prompt


# ---------------------------------------------------------------------------
# Task 2: Registry DETAILS.md loading + section lint
# ---------------------------------------------------------------------------


def test_registry_loads_details_when_present(tmp_path: Path) -> None:
    _write_skill_with_details(
        tmp_path,
        "my_skill",
        _VALID_ANALYTICAL.format(name="My Skill", desc="Does things"),
        "## Examples\n```python\nfoo()\n```\n",
    )
    skill = SkillRegistry.from_path(tmp_path).list_skills()[0]
    assert skill.has_details is True
    assert "Examples" in skill.details_markdown  # type: ignore[arg-type]


def test_registry_details_none_when_missing(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "my_skill",
        _VALID_ANALYTICAL.format(name="My Skill", desc="Does things"),
    )
    skill = SkillRegistry.from_path(tmp_path).list_skills()[0]
    assert skill.details_markdown is None
    assert skill.has_details is False


def test_registry_rejects_core_with_long_python_block(tmp_path: Path) -> None:
    long_block = "x = 1\n" * 10  # 10 lines > 5 limit
    _write_skill(
        tmp_path,
        "my_skill",
        f"---\nname: My Skill\ndescription: Does things.\n---\n\n"
        f"### Algorithm\n1. Step → pandas_tool.\n\n"
        f"### Rules\n- Rule.\n\n"
        f"```python\n{long_block}```\n",
    )
    with pytest.raises(SkillValidationError, match=r"DETAILS\.md"):
        SkillRegistry.from_path(tmp_path).load()


def test_registry_allows_core_with_short_python_block(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "my_skill",
        "---\nname: My Skill\ndescription: Does things.\n---\n\n"
        "### Algorithm\n1. Step → pandas_tool.\n\n"
        "### Rules\n- Rule.\n\n"
        "```python\nfoo(x: int) -> None\n```\n",
    )
    skills = SkillRegistry.from_path(tmp_path).list_skills()
    assert len(skills) == 1


def test_registry_rejects_tool_skill_missing_api_section(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "my_tool",
        "---\nname: My Tool\ndescription: Does things.\nkind: tool\ntool_key: my_tool\n---\n\n"
        "### Final result protocol\nLast expression must be tool_result.\n\n### Rules\n- Rule.\n",
    )
    with pytest.raises(SkillValidationError, match="### API"):
        SkillRegistry.from_path(tmp_path).load()


def test_registry_rejects_tool_skill_missing_final_result_section(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "my_tool",
        "---\nname: My Tool\ndescription: Does things.\nkind: tool\ntool_key: my_tool\n---\n\n"
        "### API\nfoo(x: int) -> None\n\n### Rules\n- Rule.\n",
    )
    with pytest.raises(SkillValidationError, match="Final result protocol"):
        SkillRegistry.from_path(tmp_path).load()


def test_registry_rejects_analytical_skill_missing_algorithm_section(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "my_skill",
        "---\nname: My Skill\ndescription: Does things.\n---\n\n### Rules\n- Rule.\n",
    )
    with pytest.raises(SkillValidationError, match="Algorithm"):
        SkillRegistry.from_path(tmp_path).load()


def test_registry_rejects_analytical_skill_missing_rules_section(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "my_skill",
        "---\nname: My Skill\ndescription: Does things.\n---\n\n### Algorithm\n1. Step → pandas_tool.\n",
    )
    with pytest.raises(SkillValidationError, match="Rules"):
        SkillRegistry.from_path(tmp_path).load()


def test_registry_accepts_russian_section_names(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "my_skill",
        "---\nname: My Skill\ndescription: Does things.\n---\n\n"
        "### Алгоритм\n1. Step → pandas_tool.\n\n### Правила\n- Rule.\n",
    )
    skills = SkillRegistry.from_path(tmp_path).list_skills()
    assert len(skills) == 1


def test_registry_existing_skills_backward_compat(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "cohort_analysis",
        _VALID_ANALYTICAL.format(name="Cohort Analysis", desc="Retention"),
    )
    skills = SkillRegistry.from_path(tmp_path).list_skills()
    assert len(skills) == 1
    assert skills[0].instructions_markdown == skills[0].core_markdown
