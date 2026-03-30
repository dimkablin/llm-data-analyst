from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_FIELD_RE = re.compile(r"^(\w+)\s*:\s*(.+)$", re.MULTILINE)


@dataclass
class Skill:
    name: str
    description: str
    triggers: list[str]
    body: str
    source: str  # file path, for debugging


@dataclass
class SkillLoader:
    """Loads Markdown skill files from a directory and formats them for injection
    into the LLM system prompt.

    Skill file format::

        ---
        name: Когортный анализ
        description: Анализ удержания пользователей по когортам
        triggers: когорт, retention, удержание, ltv
        ---

        ## Когортный анализ

        (workflow instructions using existing tools)

    Skills are injected into the system prompt as a reference section.
    The LLM reads them as documentation and uses them when relevant.
    """

    skills_dir: Path
    _skills: list[Skill] = field(default_factory=list, init=False, repr=False)
    _loaded: bool = field(default=False, init=False, repr=False)

    @classmethod
    def from_path(cls, skills_dir: str | Path) -> "SkillLoader":
        return cls(skills_dir=Path(skills_dir))

    def load(self) -> list[Skill]:
        if self._loaded:
            return self._skills
        self._loaded = True
        p = self.skills_dir
        if not p.exists() or not p.is_dir():
            return self._skills
        for md_file in sorted(p.glob("*.md")):
            skill = self._parse_skill_file(md_file)
            if skill:
                self._skills.append(skill)
        return self._skills

    def build_prompt_block(self, query: str | None = None) -> str:
        """Return a formatted block for injection into system prompt.

        If *query* is provided, only skills whose triggers match are included
        (plus any skill with no triggers — always-on). Falls back to all skills
        when nothing matches.
        """
        skills = self.load()
        if not skills:
            return ""

        selected = self._select(skills, query)
        if not selected:
            return ""

        lines = ["## Аналитические воркфлоу (Skills)", ""]
        for skill in selected:
            lines.append(f"### {skill.name}")
            if skill.description:
                lines.append(f"_{skill.description}_")
                lines.append("")
            lines.append(skill.body.strip())
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select(self, skills: list[Skill], query: str | None) -> list[Skill]:
        """Return skills relevant to *query*, or all if no match / no query."""
        if not query:
            return skills
        q = query.lower()
        matched = [
            s for s in skills
            if not s.triggers  # always-on (no triggers defined)
            or any(t in q for t in s.triggers)
        ]
        return matched if matched else skills

    @staticmethod
    def _parse_skill_file(path: Path) -> Skill | None:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return None

        name = path.stem.replace("_", " ").title()
        description = ""
        triggers: list[str] = []
        body = text

        m = _FRONTMATTER_RE.match(text)
        if m:
            frontmatter = m.group(1)
            body = text[m.end():]
            for field_match in _FIELD_RE.finditer(frontmatter):
                key, value = field_match.group(1).strip(), field_match.group(2).strip()
                if key == "name":
                    name = value
                elif key == "description":
                    description = value
                elif key == "triggers":
                    triggers = [t.strip().lower() for t in value.split(",") if t.strip()]

        body = body.strip()
        if not body:
            return None

        return Skill(
            name=name,
            description=description,
            triggers=triggers,
            body=body,
            source=str(path),
        )
