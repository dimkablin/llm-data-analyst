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
        "## Instructions\n\n"
        "Use pandas and plotly.\n"
        "```python\nprint('example')\n```\n",
    )

    registry = SkillRegistry.from_path(tmp_path)
    skills = registry.list_skills()

    assert len(skills) == 1
    assert skills[0].skill_id == "cohort_analysis"
    assert skills[0].python_examples[0].code == "print('example')"
    assert "Instructions" in skills[0].instructions_markdown


def test_registry_rejects_missing_frontmatter(tmp_path: Path) -> None:
    _write_skill(tmp_path, "broken", "## No frontmatter")

    with pytest.raises(SkillValidationError):
        SkillRegistry.from_path(tmp_path).load()


def test_registry_rejects_unknown_explicit_selection(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "cohort_analysis",
        "---\nname: Cohort Analysis\ndescription: Analyze cohorts.\n---\n\n## Instructions",
    )
    registry = SkillRegistry.from_path(tmp_path)

    with pytest.raises(SkillSelectionError):
        registry.resolve_selection(["missing_skill"])


def test_prompt_block_includes_only_explicitly_selected_skills(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "cohort_analysis",
        "---\nname: Cohort Analysis\ndescription: Analyze cohorts.\n---\n\n## Cohorts",
    )
    _write_skill(
        tmp_path,
        "forecasting",
        "---\nname: Forecasting\ndescription: Forecast metrics.\n---\n\n## Forecasting",
    )

    registry = SkillRegistry.from_path(tmp_path)
    prompt = registry.build_prompt_block(["forecasting"])

    assert "Forecasting" in prompt
    assert "Cohort Analysis" not in prompt
    assert "do not execute directly" in prompt
