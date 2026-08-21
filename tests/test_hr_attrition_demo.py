from __future__ import annotations

import importlib.util
from pathlib import Path

from backend.skills import SkillRegistry

ROOT = Path(__file__).parent.parent
DEMO_DIR = ROOT / "examples" / "hr_attrition_demo"


def _load_generator():
    path = DEMO_DIR / "generate_dataset.py"
    spec = importlib.util.spec_from_file_location("hr_attrition_demo_generator", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generator_creates_reproducible_demo_rows() -> None:
    generator = _load_generator()

    first = generator.generate_rows(row_count=3600, seed=20260720)
    second = generator.generate_rows(row_count=3600, seed=20260720)

    assert first == second
    assert len(first) == 3600
    assert len({row["employee_id"] for row in first}) == 3600
    assert "risk_score" not in first[0]

    historical = [row for row in first if row["attrition_90d"] != ""]
    current = [row for row in first if row["attrition_90d"] == ""]
    assert len(historical) == 2800
    assert len(current) == 800
    assert all(row["termination_date"] == "" for row in current)
    assert all(row["exit_reason"] == "" for row in current)

    termination_months = {row["termination_date"][:7] for row in historical if row["termination_date"]}
    assert len(termination_months) >= 24


def test_hr_attrition_skill_keeps_branch_capabilities_conditional() -> None:
    skill = SkillRegistry.from_path(ROOT / "skills").get("hr_attrition_analysis")

    assert skill.enabled_by_default is True
    assert not hasattr(skill, "execution_contract")
    assert "синтетическ" in skill.instructions_markdown.lower()
    assert "risk_score" in skill.instructions_markdown
    assert "Choose the branches requested by the user" in skill.instructions_markdown
    assert "one ordered monthly `dt, y` series" in skill.instructions_markdown
    assert "missing periods from source completeness" in skill.instructions_markdown
    assert '`targets=[{"name": "metric", "column": "y", "aggregation": "none"}]`' in (
        skill.instructions_markdown
    )
    assert "provider-published `plot.figure`" in skill.instructions_markdown
    assert "Never " not in skill.instructions_markdown
    assert "Do not " not in skill.instructions_markdown


def test_general_analytics_prepares_analysis_ready_data() -> None:
    skill = SkillRegistry.from_path(ROOT / "skills").get("general_analytics")
    text = skill.instructions_markdown

    assert "Data readiness" in text
    assert "row grain" in text
    assert "types, nulls, duplicates, and invalid values" in text
    assert "analysis-ready artifact" in text
