from __future__ import annotations

from pathlib import Path

import pytest

from backend.skills import SkillRegistry, SkillSelectionError, SkillValidationError


def test_registry_loads_markdown_skills_with_yaml_frontmatter(tmp_path: Path) -> None:
    (tmp_path / "cohort.md").write_text(
        "---\n"
        "id: cohort_analysis\n"
        "name: Cohort Analysis\n"
        "description: Analyze cohorts.\n"
        "triggers:\n"
        "  - retention\n"
        "  - ltv\n"
        "---\n\n"
        "## Instructions\n\n"
        "Use pandas and plotly.\n"
        "```python\nprint('example')\n```\n",
        encoding="utf-8",
    )

    registry = SkillRegistry.from_path(tmp_path)
    skills = registry.list_skills()

    assert len(skills) == 1
    assert skills[0].skill_id == "cohort_analysis"
    assert skills[0].python_examples[0].code == "print('example')"
    assert "Instructions" in skills[0].instructions_markdown


def test_registry_rejects_missing_frontmatter(tmp_path: Path) -> None:
    (tmp_path / "broken.md").write_text("## No frontmatter", encoding="utf-8")

    with pytest.raises(SkillValidationError):
        SkillRegistry.from_path(tmp_path).load()


def test_registry_rejects_unknown_explicit_selection(tmp_path: Path) -> None:
    (tmp_path / "cohort.md").write_text(
        "---\nname: Cohort Analysis\ndescription: Analyze cohorts.\n---\n\n## Instructions",
        encoding="utf-8",
    )
    registry = SkillRegistry.from_path(tmp_path)

    with pytest.raises(SkillSelectionError):
        registry.resolve_selection(["missing_skill"])


def test_prompt_block_includes_only_explicitly_selected_skills(tmp_path: Path) -> None:
    (tmp_path / "cohort.md").write_text(
        "---\nid: cohort_analysis\nname: Cohort Analysis\ndescription: Analyze cohorts.\n---\n\n## Cohorts",
        encoding="utf-8",
    )
    (tmp_path / "forecast.md").write_text(
        "---\nid: forecasting\nname: Forecasting\ndescription: Forecast metrics.\n---\n\n## Forecasting",
        encoding="utf-8",
    )

    registry = SkillRegistry.from_path(tmp_path)
    prompt = registry.build_prompt_block(["forecasting"])

    assert "Forecasting" in prompt
    assert "Cohort Analysis" not in prompt
    assert "do not execute directly" in prompt
