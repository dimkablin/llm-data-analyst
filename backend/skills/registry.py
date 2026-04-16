from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from backend.skills.models import (
    Skill,
    SkillExample,
    SkillSelectionError,
    SkillSummary,
    SkillValidationError,
)

logger = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*(?:\n|$)", re.DOTALL)
_PYTHON_FENCE_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_SKILL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")
_DEFAULT_MAX_CORE_BYTES = 8 * 1024
_DEFAULT_MAX_DETAILS_BYTES = 64 * 1024


def _slugify_skill_id(raw: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")
    return slug.replace("-", "_")


def _normalize_triggers(raw: object) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        items = [item.strip().lower() for item in raw.split(",")]
    elif isinstance(raw, list):
        items = [str(item).strip().lower() for item in raw]
    else:
        raise SkillValidationError("Skill triggers must be a string or list of strings.")
    normalized = tuple(dict.fromkeys(item for item in items if item))
    return normalized


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
    _skills_by_id: dict[str, Skill] = field(default_factory=dict, init=False, repr=False)
    _loaded: bool = field(default=False, init=False, repr=False)

    @classmethod
    def from_path(cls, skills_dir: str | Path) -> SkillRegistry:
        return cls(skills_dir=Path(skills_dir))

    def load(self) -> SkillRegistry:
        if self._loaded:
            return self
        skills_by_id: dict[str, Skill] = {}
        if not self.skills_dir.exists():
            logger.warning("Skills directory '%s' not found — no skills loaded.", self.skills_dir.resolve())
            self._skills_by_id = skills_by_id
            self._loaded = True
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
            if skill.skill_id in skills_by_id:
                raise SkillValidationError(
                    f"Duplicate skill id '{skill.skill_id}' in {skill_dir.name}/SKILL.md."
                )
            skills_by_id[skill.skill_id] = skill
        self._skills_by_id = skills_by_id
        self._loaded = True
        return self

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

    def resolve_tool_skills(
        self,
        available_tool_keys: set[str] | frozenset[str],
    ) -> tuple[Skill, ...]:
        """Return tool skills whose tool_key is in *available_tool_keys*."""
        self.load()
        return tuple(
            skill
            for skill in self._skills_by_id.values()
            if skill.kind == "tool" and skill.tool_key in available_tool_keys
        )

    def build_analytical_skills_brief_block(
        self,
        *,
        enabled_skill_ids: set[str] | frozenset[str] | None = None,
        user_prompt: str | None = None,
    ) -> str:
        """Analytical skills prompt section — brief list only.

        Skills are never auto-expanded inline. The agent must always call
        ``get_tool_instructions(skill_id)`` to receive the full step-by-step
        algorithm as a tool result. This prevents the agent from reading code
        examples in the system prompt and hallucinating results instead of
        actually executing the tools.

        - If *enabled_skill_ids* is provided, only those analytical skills are listed.
        - *user_prompt* is accepted for API compatibility but no longer triggers inline expansion.
        """
        self.load()
        analytical_skills = [
            skill for skill in self._skills_by_id.values() if skill.kind == "analytical"
        ]
        if enabled_skill_ids is not None:
            allow = {str(sid).strip() for sid in enabled_skill_ids if str(sid).strip()}
            analytical_skills = [s for s in analytical_skills if s.skill_id in allow]
        if not analytical_skills:
            return ""

        lines = [
            "## Аналитические скилы",
            "",
            "ОБЯЗАТЕЛЬНО: если запрос совпадает с триггерами одного из методов ниже — "
            "сначала вызови `get_tool_instructions(skill_id)` и получи пошаговый алгоритм. "
            "Затем выполняй каждый шаг алгоритма вызовами tools. "
            "Не начинай анализ без инструкций. Не синтезируй результаты без tool output.",
            "",
        ]
        for skill in analytical_skills:
            triggers_hint = ""
            if skill.triggers:
                sample = ", ".join(skill.triggers[:6])
                triggers_hint = f" | триггеры: {sample}"
            lines.append(f"- `{skill.skill_id}`: {skill.description}{triggers_hint}")

        return "\n".join(lines).strip()

    def build_tool_skills_brief_block(
        self,
        available_tool_keys: set[str] | frozenset[str],
    ) -> str:
        """One-liner per tool + hint to call get_tool_instructions (deferred style).

        Keeps the base prompt compact.  The LLM fetches full instructions
        on demand via get_tool_instructions(tool_name) before first use.
        """
        tool_skills = self.resolve_tool_skills(available_tool_keys)
        if not tool_skills:
            return ""
        lines = [
            "## Инструменты (краткое описание)",
            "",
            "Для незнакомых или сложных инструментов вызывай `get_tool_instructions(tool_name)` "
            "чтобы получить полные инструкции и примеры кода.",
            "",
        ]
        for skill in tool_skills:
            triggers_hint = ""
            if skill.triggers:
                sample = ", ".join(skill.triggers[:5])
                triggers_hint = f" (триггеры: {sample})"
            lines.append(f"- `{skill.tool_key}`: {skill.description}{triggers_hint}")
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
                raise SkillValidationError(
                    f"{path.name}: tool skill missing '### API' section."
                )
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
            stat = path.stat()
        except OSError as exc:
            raise SkillValidationError(f"Failed to stat skill file {path}: {exc}") from exc
        if stat.st_size > _DEFAULT_MAX_CORE_BYTES:
            raise SkillValidationError(
                f"Skill file {path.name} exceeds max core size of {_DEFAULT_MAX_CORE_BYTES} bytes. "
                f"Move code examples to DETAILS.md."
            )
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise SkillValidationError(f"Failed to read skill file {path}: {exc}") from exc

        match = _FRONTMATTER_RE.match(text)
        if match is None:
            raise SkillValidationError(f"Skill file {path.name} must start with YAML frontmatter.")

        try:
            raw_frontmatter = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError as exc:
            raise SkillValidationError(f"Invalid YAML frontmatter in {path.name}: {exc}") from exc
        if not isinstance(raw_frontmatter, dict):
            raise SkillValidationError(f"Frontmatter in {path.name} must be a YAML mapping.")

        body = text[match.end() :].strip()
        if not body:
            raise SkillValidationError(f"Skill file {path.name} must contain markdown instructions.")

        # Parse kind early — needed for section presence lint
        kind = str(raw_frontmatter.get("kind", "analytical")).strip().lower()
        if kind not in ("analytical", "tool"):
            raise SkillValidationError(
                f"Skill kind must be 'analytical' or 'tool', got '{kind}' in {path.name}."
            )

        self._lint_core_markdown(path, body, kind)

        # Load optional DETAILS.md
        details_markdown: str | None = None
        if details_path is not None:
            try:
                details_stat = details_path.stat()
            except OSError as exc:
                raise SkillValidationError(f"Failed to stat {details_path}: {exc}") from exc
            if details_stat.st_size > _DEFAULT_MAX_DETAILS_BYTES:
                raise SkillValidationError(
                    f"Details file {details_path.name} exceeds max size"
                    f" of {_DEFAULT_MAX_DETAILS_BYTES} bytes."
                )
            try:
                details_markdown = details_path.read_text(encoding="utf-8").strip()
            except OSError as exc:
                raise SkillValidationError(f"Failed to read {details_path}: {exc}") from exc

        default_id = _slugify_skill_id(path.parent.name)
        skill_id = str(raw_frontmatter.get("id") or default_id).strip().lower()
        if not _SKILL_ID_RE.match(skill_id):
            raise SkillValidationError(
                f"Skill id '{skill_id}' in {path.name} must match {_SKILL_ID_RE.pattern}."
            )

        name = str(raw_frontmatter.get("name") or "").strip()
        description = str(raw_frontmatter.get("description") or "").strip()
        if not name:
            raise SkillValidationError(f"Skill file {path.name} is missing frontmatter field 'name'.")
        if not description:
            raise SkillValidationError(
                f"Skill file {path.name} is missing frontmatter field 'description'."
            )

        triggers = _normalize_triggers(raw_frontmatter.get("triggers"))
        tool_key = raw_frontmatter.get("tool_key")
        if tool_key is not None:
            tool_key = str(tool_key).strip()
        if kind == "tool" and not tool_key:
            raise SkillValidationError(
                f"Skill with kind='tool' must have a non-empty 'tool_key' in {path.name}."
            )
        metadata = {
            key: value
            for key, value in raw_frontmatter.items()
            if key not in {"id", "name", "description", "triggers", "kind", "tool_key"}
        }
        return Skill(
            skill_id=skill_id,
            name=name,
            description=description,
            core_markdown=body,
            details_markdown=details_markdown,
            source_path=str(path),
            triggers=triggers,
            python_examples=_extract_python_examples(body),
            metadata=metadata,
            kind=kind,
            tool_key=tool_key,
        )


