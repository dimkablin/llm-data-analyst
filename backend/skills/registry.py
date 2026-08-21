from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from backend.instructions import InstructionMarkdownError, read_instruction_document
from backend.skills.models import (
    Skill,
    SkillExample,
    SkillSelectionError,
    SkillSummary,
    SkillValidationError,
)
from backend.skills.override_store import SkillOverride, SkillOverrideStore

logger = logging.getLogger(__name__)

_PYTHON_FENCE_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_SKILL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")
_DEFAULT_MAX_CORE_BYTES = 8 * 1024
_DEFAULT_MAX_DETAILS_BYTES = 64 * 1024


def _slugify_skill_id(raw: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")
    return slug.replace("-", "_")


def _extract_python_examples(markdown: str) -> tuple[SkillExample, ...]:
    examples: list[SkillExample] = []
    for match in _PYTHON_FENCE_RE.finditer(markdown):
        code = match.group(1).strip()
        if code:
            examples.append(SkillExample(language="python", code=code))
    return tuple(examples)


@dataclass
class SkillRegistry:
    skills_dir: Path
    # kept for backward compat; core cap is now _DEFAULT_MAX_CORE_BYTES
    max_skill_bytes: int = _DEFAULT_MAX_CORE_BYTES
    override_store: SkillOverrideStore | None = None
    _skills_by_id: dict[str, Skill] = field(default_factory=dict, init=False, repr=False)
    _loaded: bool = field(default=False, init=False, repr=False)

    @classmethod
    def from_path(cls, skills_dir: str | Path) -> SkillRegistry:
        return cls(skills_dir=Path(skills_dir))

    def load(self, *, force: bool = False) -> SkillRegistry:
        if self._loaded and not force:
            return self
        self._loaded = True
        self._skills_by_id = {}
        if not self.skills_dir.exists():
            logger.warning("Skills directory '%s' not found — no skills loaded.", self.skills_dir.resolve())
            return self
        if not self.skills_dir.is_dir():
            raise SkillValidationError(f"Skills path is not a directory: {self.skills_dir}")

        for skill_dir in sorted(self.skills_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            details_md = skill_dir / "DETAILS.md"
            skill = self._parse_skill_file(
                skill_md,
                details_path=details_md if details_md.exists() else None,
            )
            if skill.skill_id in self._skills_by_id:
                raise SkillValidationError(
                    f"Duplicate skill id '{skill.skill_id}' in {skill_dir.name}/SKILL.md."
                )
            self._skills_by_id[skill.skill_id] = skill

        self._apply_overrides()
        return self

    def reload(self) -> SkillRegistry:
        return self.load(force=True)

    def reload_skill(self, skill_id: str) -> Skill | None:
        self.load()
        skill_dir_candidates = [
            d for d in self.skills_dir.iterdir() if d.is_dir() and (d / "SKILL.md").exists()
        ]
        for skill_dir in skill_dir_candidates:
            skill_md = skill_dir / "SKILL.md"
            candidate = self._parse_skill_file(
                skill_md,
                details_path=(skill_dir / "DETAILS.md") if (skill_dir / "DETAILS.md").exists() else None,
            )
            if candidate.skill_id == skill_id:
                if self.override_store is not None:
                    override = self.override_store.get_override(skill_id)
                    if override is not None:
                        candidate = self._merge_override(candidate, override)
                self._skills_by_id[skill_id] = candidate
                return candidate
        self._skills_by_id.pop(skill_id, None)
        return None

    def _apply_overrides(self) -> None:
        if self.override_store is None:
            return
        overrides = self.override_store.get_all()
        if not overrides:
            return
        for skill_id, override in overrides.items():
            if skill_id not in self._skills_by_id:
                continue
            base = self._skills_by_id[skill_id]
            self._skills_by_id[skill_id] = self._merge_override(base, override)

    @staticmethod
    def _merge_override(base: Skill, override: SkillOverride) -> Skill:
        new_meta = dict(base.metadata)
        new_meta["overridden"] = True
        core_markdown = override.core_markdown if override.core_markdown is not None else base.core_markdown
        details_markdown = (
            override.details_markdown if override.details_markdown is not None else base.details_markdown
        )
        return Skill(
            skill_id=base.skill_id,
            name=override.name if override.name is not None else base.name,
            description=override.description if override.description is not None else base.description,
            core_markdown=core_markdown,
            details_markdown=details_markdown,
            source_path=base.source_path,
            triggers=override.triggers if override.triggers is not None else base.triggers,
            python_examples=base.python_examples,
            metadata=new_meta,
            kind=base.kind,
            tool_key=base.tool_key,
            enabled_by_default=base.enabled_by_default,
        )

    def list_skills(self) -> tuple[Skill, ...]:
        self.load()
        return tuple(self._skills_by_id.values())

    def list_summaries(self) -> tuple[SkillSummary, ...]:
        return tuple(
            SkillSummary(
                skill_id=skill.skill_id,
                name=skill.name,
                description=skill.description,
                triggers=skill.triggers,
                source_path=skill.source_path,
                enabled_by_default=skill.enabled_by_default,
            )
            for skill in self.list_skills()
        )

    def get(self, skill_id: str) -> Skill:
        self.load()
        normalized = str(skill_id or "").strip()
        if normalized not in self._skills_by_id:
            raise SkillSelectionError(f"Unknown skill id: {normalized}")
        return self._skills_by_id[normalized]

    def resolve_selection(self, selected_skill_ids: list[str] | tuple[str, ...] | None) -> tuple[Skill, ...]:
        self.load()
        if not selected_skill_ids:
            return ()
        resolved: list[Skill] = []
        seen: set[str] = set()
        for raw_skill_id in selected_skill_ids:
            normalized = str(raw_skill_id or "").strip()
            if not normalized:
                raise SkillSelectionError("Skill ids must be non-empty strings.")
            if normalized in seen:
                continue
            if normalized not in self._skills_by_id:
                raise SkillSelectionError(f"Unknown skill id: {normalized}")
            resolved.append(self._skills_by_id[normalized])
            seen.add(normalized)
        return tuple(resolved)

    def build_prompt_block(
        self,
        selected_skill_ids: list[str] | tuple[str, ...] | None,
    ) -> str:
        selected = self.resolve_selection(selected_skill_ids)
        if not selected:
            return ""

        lines = [
            "## Выбранные скилы",
            "",
            "Следующие markdown-скилы явно подключены runtime-кодом.",
            "Воспринимай их как инструкционный контекст, не как исполняемый код.",
            "Любой Python в примерах носит иллюстративный характер — выполнять нужно только через обычный путь tool-вызова, sandbox и policy.",  # noqa: E501
            "",
        ]
        for skill in selected:
            lines.append(f"### {skill.name} (`{skill.skill_id}`)")
            lines.append(f"Источник: {skill.source_name}")
            if skill.description:
                lines.append(f"Описание: {skill.description}")
            if skill.triggers:
                lines.append(f"Триггеры: {', '.join(skill.triggers)}")
            if skill.python_examples:
                lines.append(
                    f"Примеры Python: {len(skill.python_examples)} (только примеры; не выполнять напрямую)"
                )
            else:
                lines.append("Встроенные фрагменты кода, если есть, — только примеры; не выполнять напрямую.")
            lines.append("")
            lines.append(skill.instructions_markdown.strip())
            lines.append("")
        return "\n".join(lines).strip()

    def build_analytical_skills_brief_block(
        self,
        *,
        enabled_skill_ids: set[str] | frozenset[str] | None = None,
    ) -> str:
        """Analytical skills prompt section — brief list only.

        Skills are never auto-expanded inline. The agent can call
        ``get_tool_instructions(skill_id)`` when a specialized method is needed;
        the base workflow is already present in the execution prompt.

        - If *enabled_skill_ids* is provided, only those analytical skills are listed.
        """
        self.load()
        analytical_skills = [skill for skill in self._skills_by_id.values() if skill.kind == "analytical"]
        if enabled_skill_ids is not None:
            allow = {str(sid).strip() for sid in enabled_skill_ids if str(sid).strip()}
            analytical_skills = [s for s in analytical_skills if s.skill_id in allow]
        if not analytical_skills:
            return ""

        lines = [
            "## Аналитические скилы",
            "",
            "Базовый аналитический workflow уже есть в активном prompt. "
            "Вызывай `get_tool_instructions(skill_id)` только когда нужен специализированный "
            "метод или конкретная деталь, которой нет в текущем контексте. "
            "Не загружай `general_analytics` рутинно перед каждым SQL. "
            "Не синтезируй результаты без tool output.",
            "",
        ]
        for skill in analytical_skills:
            triggers_hint = ""
            if skill.triggers:
                sample = ", ".join(skill.triggers[:6])
                triggers_hint = f" | триггеры: {sample}"
            lines.append(f"- `{skill.skill_id}`: {skill.description}{triggers_hint}")

        return "\n".join(lines).strip()

    def _lint_core_markdown(self, path: Path, core: str, kind: str) -> None:
        """Validate core markdown structure and content limits."""
        for match in _PYTHON_FENCE_RE.finditer(core):
            lines = match.group(1).strip().splitlines()
            if len(lines) > 5:
                raise SkillValidationError(
                    f"{path.name}: core contains a Python block with {len(lines)} lines "
                    f"(max 5). Move code examples to DETAILS.md."
                )
        if kind == "tool":
            if "### API" not in core:
                raise SkillValidationError(f"{path.name}: tool skill missing '### API' section.")
            if "### Final result protocol" not in core:
                raise SkillValidationError(
                    f"{path.name}: tool skill missing '### Final result protocol' section."
                )
        else:  # analytical
            if not re.search(r"^### (Algorithm|Алгоритм)", core, re.MULTILINE):
                raise SkillValidationError(
                    f"{path.name}: analytical skill missing '### Algorithm' (or '### Алгоритм') section."
                )
            if not re.search(r"^### (Rules|Правила)", core, re.MULTILINE):
                raise SkillValidationError(
                    f"{path.name}: analytical skill missing '### Rules' (or '### Правила') section."
                )

    def _parse_skill_file(self, path: Path, details_path: Path | None = None) -> Skill:
        try:
            document = read_instruction_document(
                path,
                default_id=_slugify_skill_id(path.parent.name),
                default_kind="analytical",
                details_path=details_path,
                max_body_bytes=_DEFAULT_MAX_CORE_BYTES,
                max_details_bytes=_DEFAULT_MAX_DETAILS_BYTES,
            )
        except InstructionMarkdownError as exc:
            raise SkillValidationError(str(exc)) from exc

        instruction_metadata = document.metadata
        body = document.body
        kind = instruction_metadata.kind.value

        self._lint_core_markdown(path, body, kind)

        skill_id = instruction_metadata.id
        if not _SKILL_ID_RE.match(skill_id):
            raise SkillValidationError(
                f"Skill id '{skill_id}' in {path.name} must match {_SKILL_ID_RE.pattern}."
            )

        return Skill(
            skill_id=skill_id,
            name=instruction_metadata.name,
            description=instruction_metadata.description,
            core_markdown=body,
            details_markdown=document.details_markdown,
            source_path=str(path),
            triggers=instruction_metadata.triggers,
            python_examples=_extract_python_examples(body),
            metadata=instruction_metadata.extras,
            kind=kind,
            tool_key=instruction_metadata.tool_key,
            enabled_by_default=instruction_metadata.enabled_by_default,
        )
