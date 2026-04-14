"""E2E тесты для новых скилов.

Проверяют каждый SKILL.md файл через реальный SkillRegistry:
- Загрузка без ошибок валидации
- Обязательные поля заполнены и непустые
- Triggers заданы
- Python-примеры компилируются без синтаксических ошибок
- Инструкции содержат ожидаемые разделы
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from backend.skills import SkillRegistry

# Корень проекта
PROJECT_ROOT = Path(__file__).parent.parent
SKILLS_DIR = PROJECT_ROOT / "skills"

# Скилы добавленные в проекте (папка → ожидаемые свойства)
NEW_SKILLS = [
    "csv_summarizer",
    "auto_eda",
    "ab_test_analysis",
    "root_cause_investigation",
    "data_quality_audit",
    "statistical_analysis",
    "time_series_analysis",
    "duckdb_analysis",
    "insight_synthesis",
    "cohort_analysis_advanced",
]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _load_registry() -> SkillRegistry:
    registry = SkillRegistry.from_path(SKILLS_DIR)
    registry.load()
    return registry


def _extract_python_blocks(markdown: str) -> list[str]:
    """Вернуть все ```python ... ``` блоки из markdown."""
    return re.findall(r"```python\s*\n(.*?)```", markdown, re.DOTALL | re.IGNORECASE)


def _skill_ids(registry: SkillRegistry) -> set[str]:
    return {s.skill_id for s in registry.list_skills()}


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def registry() -> SkillRegistry:
    return _load_registry()


# ─── Registry-level tests ────────────────────────────────────────────────────

class TestRegistryLoadsNewSkills:
    """Проверяет что все новые скилы загружаются без ошибок."""

    def test_skills_dir_exists(self) -> None:
        assert SKILLS_DIR.exists(), f"Папка skills/ не найдена: {SKILLS_DIR}"

    @pytest.mark.parametrize("skill_folder", NEW_SKILLS)
    def test_skill_folder_exists(self, skill_folder: str) -> None:
        skill_path = SKILLS_DIR / skill_folder
        assert skill_path.exists(), f"Папка скила не найдена: {skill_path}"

    @pytest.mark.parametrize("skill_folder", NEW_SKILLS)
    def test_skill_md_file_exists(self, skill_folder: str) -> None:
        skill_md = SKILLS_DIR / skill_folder / "SKILL.md"
        assert skill_md.exists(), f"SKILL.md не найден: {skill_md}"

    def test_all_new_skills_load_without_errors(self, registry: SkillRegistry) -> None:
        loaded_ids = _skill_ids(registry)
        for skill_folder in NEW_SKILLS:
            assert skill_folder in loaded_ids, (
                f"Скил '{skill_folder}' не загрузился. "
                f"Загруженные: {sorted(loaded_ids)}"
            )

    def test_total_new_skills_count(self, registry: SkillRegistry) -> None:
        loaded_ids = _skill_ids(registry)
        missing = [s for s in NEW_SKILLS if s not in loaded_ids]
        assert not missing, f"Не загрузились скилы: {missing}"


# ─── Per-skill validation ────────────────────────────────────────────────────

class TestSkillFrontmatterFields:
    """Проверяет обязательные поля frontmatter для каждого нового скила."""

    @pytest.mark.parametrize("skill_id", NEW_SKILLS)
    def test_name_is_non_empty(self, registry: SkillRegistry, skill_id: str) -> None:
        skill = registry.get(skill_id)
        assert skill is not None, f"Скил '{skill_id}' не найден"
        assert isinstance(skill.name, str) and skill.name.strip(), (
            f"Скил '{skill_id}': name пустое"
        )

    @pytest.mark.parametrize("skill_id", NEW_SKILLS)
    def test_description_is_meaningful(self, registry: SkillRegistry, skill_id: str) -> None:
        skill = registry.get(skill_id)
        assert skill is not None
        assert len(skill.description.strip()) >= 20, (
            f"Скил '{skill_id}': description слишком короткое ({len(skill.description)} символов)"
        )

    @pytest.mark.parametrize("skill_id", NEW_SKILLS)
    def test_triggers_are_non_empty(self, registry: SkillRegistry, skill_id: str) -> None:
        skill = registry.get(skill_id)
        assert skill is not None
        assert len(skill.triggers) >= 3, (
            f"Скил '{skill_id}': нужно минимум 3 trigger, найдено {len(skill.triggers)}"
        )

    @pytest.mark.parametrize("skill_id", NEW_SKILLS)
    def test_kind_is_analytical(self, registry: SkillRegistry, skill_id: str) -> None:
        skill = registry.get(skill_id)
        assert skill is not None
        assert skill.kind == "analytical", (
            f"Скил '{skill_id}': ожидался kind='analytical', получен '{skill.kind}'"
        )

    @pytest.mark.parametrize("skill_id", NEW_SKILLS)
    def test_instructions_markdown_is_non_empty(self, registry: SkillRegistry, skill_id: str) -> None:
        skill = registry.get(skill_id)
        assert skill is not None
        assert len(skill.instructions_markdown.strip()) >= 100, (
            f"Скил '{skill_id}': instructions_markdown слишком короткий"
        )


# ─── Python examples validation ──────────────────────────────────────────────

class TestSkillPythonExamples:
    """Проверяет синтаксис Python-примеров в каждом скиле."""

    @pytest.mark.parametrize("skill_id", NEW_SKILLS)
    def test_has_python_examples(self, registry: SkillRegistry, skill_id: str) -> None:
        skill = registry.get(skill_id)
        assert skill is not None
        # duckdb_analysis использует SQL блоки, Python опционален
        if skill_id == "duckdb_analysis":
            pytest.skip("duckdb_analysis использует SQL-блоки, Python необязателен")
        assert len(skill.python_examples) >= 1, (
            f"Скил '{skill_id}': нет Python-примеров"
        )

    @pytest.mark.parametrize("skill_id", NEW_SKILLS)
    def test_python_examples_have_valid_syntax(self, registry: SkillRegistry, skill_id: str) -> None:
        skill = registry.get(skill_id)
        assert skill is not None
        for i, example in enumerate(skill.python_examples):
            try:
                ast.parse(example.code)
            except SyntaxError as exc:
                pytest.fail(
                    f"Скил '{skill_id}', пример #{i+1}: синтаксическая ошибка Python:\n"
                    f"{exc}\n\nКод:\n{example.code[:300]}"
                )

    @pytest.mark.parametrize("skill_id", NEW_SKILLS)
    def test_python_examples_end_with_tool_result(self, registry: SkillRegistry, skill_id: str) -> None:
        """Каждый Python-пример должен заканчиваться на tool_result."""
        skill = registry.get(skill_id)
        assert skill is not None
        if skill_id in ("duckdb_analysis", "insight_synthesis"):
            pytest.skip(f"Скил '{skill_id}' использует шаблонный код без tool_result в каждом блоке")
        for i, example in enumerate(skill.python_examples):
            last_line = example.code.strip().splitlines()[-1].strip()
            assert last_line == "tool_result", (
                f"Скил '{skill_id}', пример #{i+1}: последняя строка должна быть 'tool_result', "
                f"получено: '{last_line}'"
            )


# ─── Instructions structure ───────────────────────────────────────────────────

class TestSkillInstructionsStructure:
    """Проверяет что инструкции содержат обязательные разделы."""

    REQUIRED_SECTIONS = {
        "csv_summarizer": ["Шаг", "Правила"],
        "auto_eda": ["Шаг", "Правила", "корреляц"],
        "ab_test_analysis": ["Шаг", "Правила", "t-test"],
        "root_cause_investigation": ["Шаг", "Правила", "waterfall"],
        "data_quality_audit": ["Шаг", "Правила", "severity"],
        "statistical_analysis": ["Правила", "регресс"],
        "time_series_analysis": ["Шаг", "Правила", "тренд"],
        "duckdb_analysis": ["Правила", "sql_tool", "read_csv"],
        "insight_synthesis": ["Правила", "инсайт"],
        "cohort_analysis_advanced": ["Шаг", "Правила", "retention"],
    }

    @pytest.mark.parametrize("skill_id", NEW_SKILLS)
    def test_required_sections_present(self, registry: SkillRegistry, skill_id: str) -> None:
        skill = registry.get(skill_id)
        assert skill is not None
        md = skill.instructions_markdown.lower()
        expected = self.REQUIRED_SECTIONS.get(skill_id, [])
        for section_keyword in expected:
            assert section_keyword.lower() in md, (
                f"Скил '{skill_id}': ожидаемый раздел/ключевое слово '{section_keyword}' "
                f"не найдено в instructions_markdown"
            )


# ─── Regression: existing skills still load ──────────────────────────────────

class TestExistingSkillsNotBroken:
    """Убеждаемся что новые скилы не сломали загрузку существующих."""

    EXISTING_SKILLS = [
        "cohort_analysis",
        "pandas_tool",
        "sql_tool",
        "plotly_tool",
        "value_tool",
        "forecast-tool",
    ]

    @pytest.mark.parametrize("skill_id", EXISTING_SKILLS)
    def test_existing_skill_still_loads(self, registry: SkillRegistry, skill_id: str) -> None:
        skill = registry.get(skill_id)
        assert skill is not None, f"Существующий скил '{skill_id}' пропал после добавления новых"
        assert skill.name, f"Существующий скил '{skill_id}': name пустое"
