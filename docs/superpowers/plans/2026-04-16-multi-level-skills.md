# Multi-Level Skill Instructions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two-level skill retrieval (`core` / `extended`) — short `SKILL.md` (≤150 words, API + rules only) plus optional `DETAILS.md` (code examples, scenarios, error tables) — to eliminate context overflow from large skill instructions.

**Architecture:** `Skill` model gains `core_markdown` + `details_markdown` fields; `instructions_markdown` becomes a backward-compat property. Registry auto-loads `DETAILS.md` from each skill directory when present and validates core structure by kind. `get_tool_instructions` tool gains `details: bool = False` — core returned by default, DETAILS.md content on `details=True`. Hint about extended availability is injected by the tool layer at runtime, not baked into content files.

**Tech Stack:** Python 3.11+, pydantic v2, langchain-core, pytest, existing `SkillRegistry` / `Skill` / `GetToolInstructionsTool`

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `backend/skills/models.py` | Modify | Add `core_markdown`, `details_markdown`, `has_details` to `Skill` dataclass |
| `backend/skills/registry.py` | Modify | Load `DETAILS.md`, add `MAX_CORE_BYTES`, semantic lint |
| `backend/tools/impl/get_tool_instructions_tool.py` | Modify | Add `details: bool` param, hint injection, retrieval policy |
| `tests/test_skills_registry.py` | Modify | New tests for two-level loading and lint |
| `tests/test_get_tool_instructions_tool.py` | Create | Tool-level tests for `details: bool` semantics |
| `skills/*/SKILL.md` | Modify | Shorten to ≤150 words (11 skills + 3 borderline) |
| `skills/*/DETAILS.md` | Create | Heavy code examples moved out of SKILL.md |

---

## Task 1: Update `Skill` dataclass — two-level fields

**Files:**
- Modify: `backend/skills/models.py`
- Test: `tests/test_skills_registry.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_skills_registry.py`:

```python
from backend.skills.models import Skill


def test_skill_core_markdown_field(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "my_skill",
        "---\nname: My Skill\ndescription: Does things.\n---\n\n## API\nfoo() -> None\n",
    )
    skill = SkillRegistry.from_path(tmp_path).list_skills()[0]
    assert skill.core_markdown == "## API\nfoo() -> None"
    assert skill.details_markdown is None
    assert skill.has_details is False


def test_skill_instructions_markdown_backward_compat(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "my_skill",
        "---\nname: My Skill\ndescription: Does things.\n---\n\n## API\nfoo() -> None\n",
    )
    skill = SkillRegistry.from_path(tmp_path).list_skills()[0]
    assert skill.instructions_markdown == skill.core_markdown
```

- [ ] **Step 2: Run tests — expect failure**

```
pytest tests/test_skills_registry.py::test_skill_core_markdown_field tests/test_skills_registry.py::test_skill_instructions_markdown_backward_compat -v
```

Expected: `AttributeError` — `core_markdown` doesn't exist yet.

- [ ] **Step 3: Implement in `backend/skills/models.py`**

Replace the `Skill` dataclass. Full new content of `models.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class SkillError(Exception):
    """Base error for skill loading and selection."""


class SkillValidationError(SkillError):
    """Raised when a markdown skill file is malformed or unsafe to load."""


class SkillSelectionError(SkillError):
    """Raised when explicit runtime skill selection is invalid."""


@dataclass(frozen=True)
class SkillExample:
    language: str
    code: str


@dataclass(frozen=True)
class Skill:
    skill_id: str
    name: str
    description: str
    core_markdown: str            # from SKILL.md body — required
    details_markdown: str | None  # from DETAILS.md — optional
    source_path: str
    triggers: tuple[str, ...] = ()
    python_examples: tuple[SkillExample, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    kind: str = "analytical"
    tool_key: str | None = None

    @property
    def instructions_markdown(self) -> str:
        """Backward-compat alias → core_markdown."""
        return self.core_markdown

    @property
    def has_details(self) -> bool:
        return self.details_markdown is not None

    @property
    def source_name(self) -> str:
        return Path(self.source_path).name


@dataclass(frozen=True)
class SkillSummary:
    skill_id: str
    name: str
    description: str
    triggers: tuple[str, ...]
    source_path: str


@dataclass(frozen=True)
class SkillSelectionContext:
    query: str | None = None
    dataset_columns: tuple[str, ...] = ()
    source_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SkillFilter(Protocol):
    def filter(
        self,
        skills: tuple[Skill, ...],
        context: SkillSelectionContext,
    ) -> tuple[Skill, ...]: ...


class SkillMatcher(Protocol):
    def match(
        self,
        skills: tuple[Skill, ...],
        context: SkillSelectionContext,
    ) -> tuple[SkillSummary, ...]: ...


class SkillRanker(Protocol):
    def rank(
        self,
        skills: tuple[SkillSummary, ...],
        context: SkillSelectionContext,
    ) -> tuple[SkillSummary, ...]: ...


class SkillSelector(Protocol):
    def select(
        self,
        skills: tuple[Skill, ...],
        context: SkillSelectionContext,
    ) -> tuple[SkillSummary, ...]: ...
```

- [ ] **Step 4: Run tests — expect pass**

```
pytest tests/test_skills_registry.py::test_skill_core_markdown_field tests/test_skills_registry.py::test_skill_instructions_markdown_backward_compat -v
```

Expected: 2 passed. (The registry still creates `Skill(instructions_markdown=body)` which will fail — that's expected; fix in Task 2.)

- [ ] **Step 5: Commit**

```bash
git add backend/skills/models.py tests/test_skills_registry.py
git commit -m "feat(skills): add core_markdown/details_markdown/has_details to Skill model"
```

---

## Task 2: Update registry — load DETAILS.md + semantic lint

**Files:**
- Modify: `backend/skills/registry.py`
- Test: `tests/test_skills_registry.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_skills_registry.py`:

```python
def _write_skill_with_details(tmp_path: Path, folder: str, skill_content: str, details_content: str) -> None:
    skill_dir = tmp_path / folder
    skill_dir.mkdir(exist_ok=True)
    (skill_dir / "SKILL.md").write_text(skill_content, encoding="utf-8")
    (skill_dir / "DETAILS.md").write_text(details_content, encoding="utf-8")


def test_registry_loads_details_when_present(tmp_path: Path) -> None:
    _write_skill_with_details(
        tmp_path,
        "my_skill",
        "---\nname: My Skill\ndescription: Does things.\n---\n\n## API\nfoo() -> None\n",
        "## Examples\n```python\nfoo()\n```\n",
    )
    skill = SkillRegistry.from_path(tmp_path).list_skills()[0]
    assert skill.has_details is True
    assert "Examples" in skill.details_markdown


def test_registry_details_none_when_missing(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "my_skill",
        "---\nname: My Skill\ndescription: Does things.\n---\n\n## API\nfoo() -> None\n",
    )
    skill = SkillRegistry.from_path(tmp_path).list_skills()[0]
    assert skill.details_markdown is None


def test_registry_rejects_core_with_long_python_block(tmp_path: Path) -> None:
    long_block = "x = 1\n" * 10  # 10 lines > 5 limit
    _write_skill(
        tmp_path,
        "my_skill",
        f"---\nname: My Skill\ndescription: Does things.\n---\n\n## API\n```python\n{long_block}```\n",
    )
    with pytest.raises(SkillValidationError, match="DETAILS.md"):
        SkillRegistry.from_path(tmp_path).load()


def test_registry_allows_core_with_short_python_block(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "my_skill",
        "---\nname: My Skill\ndescription: Does things.\n---\n\n## API\n```python\nfoo(x: int) -> None\n```\n",
    )
    skills = SkillRegistry.from_path(tmp_path).list_skills()
    assert len(skills) == 1


def test_registry_existing_skills_still_load(tmp_path: Path) -> None:
    """Backward compat: existing SKILL.md files without DETAILS.md load fine."""
    _write_skill(
        tmp_path,
        "cohort_analysis",
        "---\nname: Cohort Analysis\ndescription: Retention.\n---\n\n### Algorithm\n1. Step one → pandas_tool.\n\n### Rules\n- Rule one.\n",
    )
    skills = SkillRegistry.from_path(tmp_path).list_skills()
    assert len(skills) == 1
    assert skills[0].instructions_markdown == skills[0].core_markdown


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
        "---\nname: My Skill\ndescription: Does things.\n---\n\n### Алгоритм\n1. Step → pandas_tool.\n\n### Правила\n- Rule.\n",
    )
    skills = SkillRegistry.from_path(tmp_path).list_skills()
    assert len(skills) == 1
```

- [ ] **Step 2: Run tests — expect failure**

```
pytest tests/test_skills_registry.py::test_registry_loads_details_when_present tests/test_skills_registry.py::test_registry_details_none_when_missing tests/test_skills_registry.py::test_registry_rejects_core_with_long_python_block tests/test_skills_registry.py::test_registry_allows_core_with_short_python_block tests/test_skills_registry.py::test_registry_existing_skills_still_load -v
```

Expected: failures due to `Skill()` constructor mismatch (`core_markdown` vs `instructions_markdown`).

- [ ] **Step 3: Implement in `backend/skills/registry.py`**

Key changes:
1. Replace `_DEFAULT_MAX_SKILL_BYTES` with separate `_DEFAULT_MAX_CORE_BYTES` and `_DEFAULT_MAX_DETAILS_BYTES`
2. Update `load()` to detect `DETAILS.md`
3. Update `_parse_skill_file()` signature and construction
4. Add `_lint_core_markdown()` function

Replace in `registry.py`:

```python
# After existing imports, change constants:
_DEFAULT_MAX_CORE_BYTES = 8 * 1024     # ~2 000 words — hard cap for core
_DEFAULT_MAX_DETAILS_BYTES = 64 * 1024  # existing cap, now for details
```

Replace `load()` method:

```python
def load(self) -> SkillRegistry:
    if self._loaded:
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
    return self
```

Add `_lint_core_markdown()` function (before `_parse_skill_file`):

```python
def _lint_core_markdown(path: Path, core: str, kind: str) -> None:
    """Validate core markdown structure and content limits."""
    # Code block size check
    for match in _PYTHON_FENCE_RE.finditer(core):
        lines = match.group(1).strip().splitlines()
        if len(lines) > 5:
            raise SkillValidationError(
                f"{path.name}: core contains a Python block with {len(lines)} lines "
                f"(max 5). Move code examples to DETAILS.md."
            )
    # Required section presence by kind
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
```

Update `_parse_skill_file()` signature and body — add `details_path` parameter, replace `Skill(instructions_markdown=body, ...)` with `Skill(core_markdown=body, details_markdown=..., ...)`, and add size validation for core + lint call:

```python
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

    body = text[match.end():].strip()
    if not body:
        raise SkillValidationError(f"Skill file {path.name} must contain markdown instructions.")

    # Parse kind early — needed for section presence lint
    kind = str(raw_frontmatter.get("kind", "analytical")).strip().lower()
    if kind not in ("analytical", "tool"):
        raise SkillValidationError(
            f"Skill kind must be 'analytical' or 'tool', got '{kind}' in {path.name}."
        )

    _lint_core_markdown(path, body, kind)

    # Load optional DETAILS.md
    details_markdown: str | None = None
    if details_path is not None:
        try:
            details_stat = details_path.stat()
        except OSError as exc:
            raise SkillValidationError(f"Failed to stat {details_path}: {exc}") from exc
        if details_stat.st_size > _DEFAULT_MAX_DETAILS_BYTES:
            raise SkillValidationError(
                f"Details file {details_path.name} exceeds max size of {_DEFAULT_MAX_DETAILS_BYTES} bytes."
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
```

Also remove the old `max_skill_bytes` field from `SkillRegistry` dataclass (it's replaced by module-level constants) or keep it as deprecated:

```python
@dataclass
class SkillRegistry:
    skills_dir: Path
    # max_skill_bytes kept for backward compat but no longer used; limits are now module constants
    max_skill_bytes: int = field(default=_DEFAULT_MAX_CORE_BYTES, repr=False)
    _skills_by_id: dict[str, Skill] = field(default_factory=dict, init=False, repr=False)
    _loaded: bool = field(default=False, init=False, repr=False)
```

- [ ] **Step 4: Run tests — expect pass**

```
pytest tests/test_skills_registry.py -v
```

Expected: all pass including existing tests.

- [ ] **Step 5: Verify real skills still load**

```
python -c "
from backend.skills import SkillRegistry
r = SkillRegistry.from_path('skills').load()
skills = r.list_skills()
print(f'Loaded {len(skills)} skills')
for s in skills:
    print(f'  {s.skill_id}: core={len(s.core_markdown)} chars, has_details={s.has_details}')
"
```

Expected: all 22 skills load, has_details=False for all (DETAILS.md not created yet).

- [ ] **Step 6: Commit**

```bash
git add backend/skills/registry.py backend/skills/models.py tests/test_skills_registry.py
git commit -m "feat(skills): registry loads DETAILS.md and validates core size/structure"
```

---

## Task 3: Update `get_tool_instructions` tool — `details: bool` + hint injection

**Files:**
- Modify: `backend/tools/impl/get_tool_instructions_tool.py`
- Create: `tests/test_get_tool_instructions_tool.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_get_tool_instructions_tool.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from backend.skills import SkillRegistry
from backend.tools.impl.get_tool_instructions_tool import GetToolInstructionsTool


def _make_registry(tmp_path: Path, with_details: bool = False) -> SkillRegistry:
    skill_dir = tmp_path / "my_tool"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: My Tool\ndescription: Does things.\nkind: tool\ntool_key: my_tool\n---\n\n"
        "## API\nfoo(x: int) -> None\n\n### Rules\n- Always set x > 0\n",
        encoding="utf-8",
    )
    if with_details:
        (skill_dir / "DETAILS.md").write_text(
            "## Examples\n```python\nfoo(1)\n```\n",
            encoding="utf-8",
        )
    return SkillRegistry.from_path(tmp_path).load()


def test_default_returns_core(tmp_path: Path) -> None:
    registry = _make_registry(tmp_path, with_details=True)
    tool = GetToolInstructionsTool(registry)
    result = tool._run(skill_id="my_tool")
    assert "API" in result
    assert "Examples" not in result  # details not included by default


def test_default_includes_hint_when_details_exist(tmp_path: Path) -> None:
    registry = _make_registry(tmp_path, with_details=True)
    tool = GetToolInstructionsTool(registry)
    result = tool._run(skill_id="my_tool")
    assert "details=True" in result
    assert "my_tool" in result


def test_no_hint_when_no_details(tmp_path: Path) -> None:
    registry = _make_registry(tmp_path, with_details=False)
    tool = GetToolInstructionsTool(registry)
    result = tool._run(skill_id="my_tool")
    assert "details=True" not in result


def test_details_true_returns_details(tmp_path: Path) -> None:
    registry = _make_registry(tmp_path, with_details=True)
    tool = GetToolInstructionsTool(registry)
    result = tool._run(skill_id="my_tool", details=True)
    assert "Examples" in result
    assert "API" not in result  # core not repeated


def test_details_true_missing_returns_graceful_fallback(tmp_path: Path) -> None:
    registry = _make_registry(tmp_path, with_details=False)
    tool = GetToolInstructionsTool(registry)
    result = tool._run(skill_id="my_tool", details=True)
    assert "not available" in result.lower()
    assert "API" not in result  # core NOT repeated


def test_unknown_skill_returns_available_list(tmp_path: Path) -> None:
    registry = _make_registry(tmp_path)
    tool = GetToolInstructionsTool(registry)
    result = tool._run(skill_id="nonexistent_tool")
    assert "my_tool" in result
```

- [ ] **Step 2: Run tests — expect failure**

```
pytest tests/test_get_tool_instructions_tool.py -v
```

Expected: failures — `_run` doesn't have `details` param yet.

- [ ] **Step 3: Implement in `backend/tools/impl/get_tool_instructions_tool.py`**

Full new content:

```python
"""On-demand tool instruction loader with two-level retrieval (core / details).

The agent calls get_tool_instructions(skill_id) to receive core instructions
(API signatures, behavioral rules). For code examples and scenarios the agent
calls get_tool_instructions(skill_id, details=True).

Retrieval policy is encoded in the tool description so the model follows
deterministic rules rather than guessing.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from backend.skills.models import SkillError

logger = logging.getLogger(__name__)

_EXTENDED_HINT_TEMPLATE = (
    "\n\n[Extended available: "
    "call get_tool_instructions('{skill_id}', details=True) "
    "for code scenarios, error patterns, and edge cases.]"
)

_RETRIEVAL_POLICY = (
    "Load skill instructions before using a tool or analytical method.\n"
    "\n"
    "RETRIEVAL POLICY:\n"
    "1. Before first use of an unfamiliar or non-trivial tool → call without details (default).\n"
    "2. Before complex scenarios (multiple charts, DB mode, JOIN chains, multi-step analysis) "
    "→ call with details=True.\n"
    "3. After a tool call fails → call with details=True before retrying.\n"
    "4. Never call get_tool_instructions for the same skill_id + details combination twice in one session.\n"
    "\n"
    "details=False (default): API signatures, behavioral rules, contract.\n"
    "details=True: code scenarios, error patterns, edge cases (on demand only)."
)


class _Input(BaseModel):
    skill_id: str = Field(
        description=(
            "ID of the skill or analytical method. "
            "Tools: 'plotly_tool', 'sql_tool', 'pandas_tool', 'database_tool', etc. "
            "Analytical methods: 'auto_eda', 'cohort_analysis', 'ab_test_analysis', etc."
        )
    )
    details: bool = Field(
        default=False,
        description=(
            "False (default) — return core: API signatures, behavioral rules, contract. "
            "True — return DETAILS.md: code scenarios, error patterns, edge cases."
        ),
    )


class GetToolInstructionsTool(BaseTool):
    """Returns skill instructions at the requested detail level."""

    name: str = "get_tool_instructions"
    description: str = _RETRIEVAL_POLICY
    args_schema: type[BaseModel] = _Input
    response_format: str = "content"

    _skill_registry: Any = PrivateAttr()

    def __init__(self, skill_registry: Any) -> None:
        super().__init__()
        self._skill_registry = skill_registry

    def _run(
        self,
        skill_id: str,
        details: bool = False,
    ) -> str:
        try:
            skill = self._skill_registry.get(str(skill_id).strip())
        except SkillError:
            return self._not_found_response(str(skill_id).strip())
        except Exception:
            logger.exception("Unexpected error looking up skill '%s'", skill_id)
            return self._not_found_response(str(skill_id).strip())

        if details:
            if not skill.has_details:
                return (
                    f"Extended instructions not available for '{skill.skill_id}'. "
                    f"Core instructions were already provided."
                )
            return skill.details_markdown

        # Core — inject hint if details are available
        content = skill.core_markdown
        if skill.has_details:
            content += _EXTENDED_HINT_TEMPLATE.format(skill_id=skill.skill_id)
        return content

    def _not_found_response(self, skill_id: str) -> str:
        all_skills = self._skill_registry.list_skills()
        tool_ids = sorted(s.skill_id for s in all_skills if s.kind == "tool")
        analytical_ids = sorted(s.skill_id for s in all_skills if s.kind == "analytical")
        parts = []
        if tool_ids:
            parts.append(f"tools: {', '.join(tool_ids)}")
        if analytical_ids:
            parts.append(f"analytical methods: {', '.join(analytical_ids)}")
        available_str = "; ".join(parts) or "none"
        return f"Skill '{skill_id}' not found. Available: {available_str}."
```

- [ ] **Step 4: Run tests — expect pass**

```
pytest tests/test_get_tool_instructions_tool.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Run full test suite**

```
pytest tests/test_skills_registry.py tests/test_get_tool_instructions_tool.py -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add backend/tools/impl/get_tool_instructions_tool.py tests/test_get_tool_instructions_tool.py
git commit -m "feat(skills): add details bool param to get_tool_instructions with retrieval policy"
```

---

## Task 4: Migrate `root_cause_investigation` (992 → ~130 words)

**Files:**
- Modify: `skills/root_cause_investigation/SKILL.md`
- Create: `skills/root_cause_investigation/DETAILS.md`

- [ ] **Step 1: Write new `SKILL.md`**

```markdown
---
name: Root Cause Investigation
description: Systematic metric change investigation — z-score validation, dimension drill-down, top contributors, hypothesis testing with chi-square, waterfall.
triggers: root cause, first cause, why dropped, why grew, what changed, drill down, segment contribution, metric change, investigation, decomposition, первопричина, почему упало, почему выросло, что изменилось, расследование
---

## Root Cause Investigation

Use when you need to understand WHY a metric changed. Validates the change is real (z-score), finds guilty segments via drill-down, tests hypotheses.

### Algorithm (4 steps)
1. **Change validation** → `pandas_tool`: z-score vs baseline. If |Z| < 1 → stop, change is noise.
2. **Dimension drill-down** → `pandas_tool`: delta contribution per segment, up to 3 categorical dims.
3. **Top contributors** → `pandas_tool`: rank all segments across dims, identify main culprit.
4. **Hypothesis testing + waterfall** → `pandas_tool` + `plotly_tool`: chi-square mix shift, volume change, per-unit quality. Waterfall chart for top dimension.

### Rules
- Start with z-score: |Z| < 1 → change is normal noise, don't hunt root causes
- Mix shift — ALWAYS validate with chi-square (p < 0.05)
- If one segment contributes > 80% → explicitly state "Main culprit: [segment]"
- Three hypothesis types: mix shift, volume, quality (metric per unit)
- If fewer than two periods in data → ask user to clarify what to compare
```

- [ ] **Step 2: Create `DETAILS.md`** — move all code from the current SKILL.md body

Migration starting point: copy the current `SKILL.md` body (everything after frontmatter) into `DETAILS.md` as-is. Clean up afterward if needed.

- [ ] **Step 3: Verify skill loads**

```
python -c "
from backend.skills import SkillRegistry
r = SkillRegistry.from_path('skills').load()
s = r.get('root_cause_investigation')
print('core words:', len(s.core_markdown.split()))
print('has_details:', s.has_details)
print('details words:', len(s.details_markdown.split()) if s.details_markdown else 0)
"
```

Expected: core ≤ 150 words, has_details=True.

- [ ] **Step 4: Commit**

```bash
git add skills/root_cause_investigation/SKILL.md skills/root_cause_investigation/DETAILS.md
git commit -m "refactor(skills): split root_cause_investigation into core + details"
```

---

## Task 5: Migrate `ab_test_analysis` (960 → ~130 words)

**Files:**
- Modify: `skills/ab_test_analysis/SKILL.md`
- Create: `skills/ab_test_analysis/DETAILS.md`

- [ ] **Step 1: Write new `SKILL.md`**

```markdown
---
name: A/B Test Analysis
description: Statistically rigorous A/B test analysis — SRM check, auto-selected test (proportion z-test / Welch t-test), power analysis (Cohen's h/d), Bonferroni correction, guardrail metrics.
triggers: a/b test, ab test, ab тест, control group, test group, statistical significance, conversion, experiment, hypothesis test, significance, srm, sample ratio mismatch, контрольная группа, тестовая группа, конверсия, эксперимент
---

## A/B Test Analysis

Checks SRM, computes statistical significance and power, gives SHIP / DO_NOT_SHIP / INCONCLUSIVE recommendation.

### Algorithm (3 steps)
1. **SRM + base metrics** → `pandas_tool`: chi-square split check, conversion rate per group with 95% CIs.
2. **Significance + power** → `pandas_tool`: auto-select test (binary → proportion z-test; continuous → Welch t-test), Cohen's h/d, achieved power. Run Bonferroni if multiple numeric columns.
3. **Visualization** → `plotly_tool`: conversion rate bars with CI + relative uplift bar.

### Recommendation logic
- `SHIP` ✅ requires: no SRM + p < 0.05 + power > 0.70 + diff > 0
- `DO_NOT_SHIP` ❌: SRM detected OR (p < 0.05 AND diff < 0)
- `INCONCLUSIVE` ⚠️: power < 0.70 OR p ≥ 0.05

### Rules
- Check SRM first — if `srm_detected=True`, results are unreliable
- Run Step 2.5 (Bonferroni) when there are multiple numeric columns
- Warn if n < 100 in either group — small sample
- SRM assumes 50/50 split; adjust `expected_ratio` if intended split differs
```

- [ ] **Step 2: Create `DETAILS.md`**

Migration starting point: copy the current `SKILL.md` body (everything after frontmatter) into `DETAILS.md` as-is. Clean up afterward if needed.

- [ ] **Step 3: Verify**

```
python -c "
from backend.skills import SkillRegistry
r = SkillRegistry.from_path('skills').load()
s = r.get('ab_test_analysis')
print('core words:', len(s.core_markdown.split()))
print('has_details:', s.has_details)
"
```

Expected: core ≤ 150 words, has_details=True.

- [ ] **Step 4: Commit**

```bash
git add skills/ab_test_analysis/SKILL.md skills/ab_test_analysis/DETAILS.md
git commit -m "refactor(skills): split ab_test_analysis into core + details"
```

---

## Task 6: Migrate `data_quality_audit` (919 → ~120 words)

**Files:**
- Modify: `skills/data_quality_audit/SKILL.md`
- Create: `skills/data_quality_audit/DETAILS.md`

- [ ] **Step 1: Write new `SKILL.md`**

```markdown
---
name: Data Quality Audit
description: Comprehensive data checks — duplicates, missing values, outliers, type mismatches, referential integrity.
triggers: data quality, duplicates, deduplication, missing values, outliers, data anomalies, data check, dq, audit, integrity, качество данных, дубли, пропуски, выбросы, проверка данных, целостность
---

## Data Quality Audit

Systematic dataset quality checks before analysis or when data issues are suspected.

### Algorithm (4 steps)
1. **DQ report per column** → `pandas_tool`: null_pct, unique count, outliers (3×IQR), numeric-as-object detection. Severity: `critical` / `warning` / `ok`.
2. **Duplicates** → `pandas_tool`: full-row dedup + key-based dedup on auto-detected ID candidates.
3. **Cross-column validation** → `pandas_tool`: date ordering (end < start), negative values in positive-only columns.
4. **Issue visualization** → `plotly_tool`: missing % by column + outlier counts.

### Rules
- Severity: `critical` → blocks analysis; `warning` → needs attention
- ALWAYS run Step 3 (cross-column) — often catches critical errors in dates and amounts
- Duplicates > 5% of rows → stop and warn the user before proceeding
- DO NOT fix data — diagnostics only; let the user decide
- Conclude explicitly: "Data is ready for analysis" or "Cleaning required"
- Object columns with > 80% numeric values → suggest `pd.to_numeric` conversion
```

- [ ] **Step 2: Create `DETAILS.md`**

Migration starting point: copy the current `SKILL.md` body (everything after frontmatter) into `DETAILS.md` as-is. Clean up afterward if needed.

- [ ] **Step 3: Verify + commit**

```
python -c "
from backend.skills import SkillRegistry
r = SkillRegistry.from_path('skills').load()
s = r.get('data_quality_audit')
print('core words:', len(s.core_markdown.split()))
print('has_details:', s.has_details)
"
```

```bash
git add skills/data_quality_audit/SKILL.md skills/data_quality_audit/DETAILS.md
git commit -m "refactor(skills): split data_quality_audit into core + details"
```

---

## Task 7: Migrate `auto_eda` (839 → ~125 words)

**Files:**
- Modify: `skills/auto_eda/SKILL.md`
- Create: `skills/auto_eda/DETAILS.md`

- [ ] **Step 1: Write new `SKILL.md`**

```markdown
---
name: Auto EDA
description: Systematic exploratory data analysis — distributions, correlations, outliers, type anomalies.
triggers: eda, exploratory analysis, data exploration, correlation, distribution, outliers, profiling, full analysis, разведочный анализ, исследование данных
---

## Auto EDA

Deep initial dataset analysis: distributions, correlations, outliers, type anomalies.

### Algorithm (5 steps)
1. **Numeric statistics** → `pandas_tool`: mean, median, std, skew, IQR outliers per numeric column.
2. **Correlation matrix** → `plotly_tool`: Pearson heatmap (`px.imshow`).
3. **Numeric distributions** → `plotly_tool`: box + histogram per column, top-6 by coefficient of variation.
4. **Categorical columns** → `pandas_tool`: cardinality, top value %, null %.
5. **Automated observations** → `pandas_tool`: flag missing > 10%, high correlation (|r| > 0.7), |skew| > 2, IQR outliers > 5%. Return as `observations_df` table.

### Rules
- Sample to 50k rows for Steps 1–3 when `len(df) > 50_000`
- Exclude ID-like columns (unique == nrows) from numeric analysis
- Step 3: prioritize columns by coefficient of variation, not column order
- Step 5: return computed `observations_df` table, not a text list
```

- [ ] **Step 2: Create `DETAILS.md`**

Migration starting point: copy the current `SKILL.md` body (everything after frontmatter) into `DETAILS.md` as-is. Clean up afterward if needed.

- [ ] **Step 3: Verify + commit**

```
python -c "
from backend.skills import SkillRegistry
r = SkillRegistry.from_path('skills').load()
s = r.get('auto_eda')
print('core words:', len(s.core_markdown.split()))
print('has_details:', s.has_details)
"
```

```bash
git add skills/auto_eda/SKILL.md skills/auto_eda/DETAILS.md
git commit -m "refactor(skills): split auto_eda into core + details"
```

---

## Task 8: Migrate `time_series_analysis` (804 → ~125 words)

**Files:**
- Modify: `skills/time_series_analysis/SKILL.md`
- Create: `skills/time_series_analysis/DETAILS.md`

- [ ] **Step 1: Write new `SKILL.md`**

```markdown
---
name: Time Series Analysis
description: Trend and seasonality decomposition, stationarity test, moving averages, anomaly detection, temporal patterns.
triggers: time series, trend, seasonality, dynamics, rolling, moving average, period, temporal, temporal dependency, временной ряд, тренд, сезонность, динамика, по времени, скользящее среднее
---

## Time Series Analysis

Data with a time component: trends, seasonality, stationarity, anomalies.

### Algorithm (4 steps)
1. **Preparation + moving averages** → `pandas_tool`: auto-detect date column, resample by auto-granularity, compute short/long MA.
2. **Stationarity (ADF)** → `pandas_tool`: Augmented Dickey-Fuller; rolling-variance fallback if statsmodels unavailable.
3. **Autocorrelation (ACF)** → `pandas_tool`: up to 20 lags, identify dominant lag.
4. **Decomposition + anomalies + visualization** → `pandas_tool` + `plotly_tool`: trend slope, seasonal variation %, Z-score anomalies, line chart.

### Rules
- Auto-granularity: ≤90 days → `D`, ≤2 years → `W`, else → `ME`
- MA windows adapt to frequency — don't hardcode values
- Anomaly threshold: Z > 2.5 if points < 30, else Z > 3.0
- < 12 points after resampling → skip ADF/ACF/decomposition, return chart with warning
- Not for forecasting — use `forecast_tool` / `anomaly_planfact_tool` instead
- Non-stationary (ADF p > 0.05) → trend/seasonality present; seasonal variation > 20% → series is seasonal
```

- [ ] **Step 2: Create `DETAILS.md`**

Migration starting point: copy the current `SKILL.md` body (everything after frontmatter) into `DETAILS.md` as-is. Clean up afterward if needed.

- [ ] **Step 3: Verify + commit**

```
python -c "
from backend.skills import SkillRegistry
r = SkillRegistry.from_path('skills').load()
s = r.get('time_series_analysis')
print('core words:', len(s.core_markdown.split()))
print('has_details:', s.has_details)
"
```

```bash
git add skills/time_series_analysis/SKILL.md skills/time_series_analysis/DETAILS.md
git commit -m "refactor(skills): split time_series_analysis into core + details"
```

---

## Task 9: Migrate `statistical_analysis` (789 → ~130 words)

**Files:**
- Modify: `skills/statistical_analysis/SKILL.md`
- Create: `skills/statistical_analysis/DETAILS.md`

- [ ] **Step 1: Write new `SKILL.md`**

```markdown
---
name: Statistical Analysis
description: Hypothesis testing, regression, ANOVA, correlations — full statistical toolkit with result interpretation.
triggers: statistics, hypothesis, regression, anova, correlation, pearson, spearman, t-test, chi-square, normality, linear dependence, statistical test, статистика, гипотеза, регрессия, корреляция, нормальность
---

## Statistical Analysis

Rigorous hypothesis testing, dependency modeling, and group comparison.

### Algorithm (order matters)
1. **Normality** → `pandas_tool`: Shapiro-Wilk (n ≤ 5000) or D'Agostino (n > 5000) per numeric column. Result determines parametric vs non-parametric branch.
2. **Correlation** → `pandas_tool`: Pearson + Spearman for all numeric pairs, ranked by |r|.
3. **One-Way ANOVA** → `pandas_tool`: if categorical + numeric columns present (min 5 rows per group).
4. **Chi-Square** → `pandas_tool`: categorical column pairs (min expected cell frequency ≥ 5).
5. **Linear regression** → `pandas_tool` + `plotly_tool`: auto-select highest-|r| pair; scatter + OLS line; residuals Q-Q plot.

### Rules
- Normality test first — determines which subsequent tests apply
- Regression: only for the pair with highest absolute correlation
- ALWAYS visualize regression residuals (Q-Q plot)
- R² < 0.3 → weak model, warn the user
- p > 0.05 = "no evidence of an effect", not "no effect"
```

- [ ] **Step 2: Create `DETAILS.md`**

Migration starting point: copy the current `SKILL.md` body (everything after frontmatter) into `DETAILS.md` as-is. Clean up afterward if needed.

- [ ] **Step 3: Verify + commit**

```
python -c "
from backend.skills import SkillRegistry
r = SkillRegistry.from_path('skills').load()
s = r.get('statistical_analysis')
print('core words:', len(s.core_markdown.split()))
print('has_details:', s.has_details)
"
```

```bash
git add skills/statistical_analysis/SKILL.md skills/statistical_analysis/DETAILS.md
git commit -m "refactor(skills): split statistical_analysis into core + details"
```

---

## Task 10: Migrate `insight_synthesis` (770 → ~135 words)

**Files:**
- Modify: `skills/insight_synthesis/SKILL.md`
- Create: `skills/insight_synthesis/DETAILS.md`

- [ ] **Step 1: Write new `SKILL.md`**

```markdown
---
name: Insight Synthesis
description: Transform analysis results into structured business insights using the So What / Why / Now What framework with impact prioritization.
triggers: insights, conclusions, analysis summary, what does this mean, business conclusion, results, recommendations, executive summary, what to do next, interpretation, инсайты, выводы, резюме анализа, рекомендации, интерпретация
---

## Insight Synthesis

Final step after completing analysis. Structures observations via So What / Why / Now What, prioritizes by impact×confidence, produces executive summary.

### Algorithm (3 steps)
1. **Key metrics** → `value_tool`: total/mean/median per numeric column; skip constants (CV ≤ 0.01); flag mean/median divergence > 20%.
2. **Programmatic insight extraction** → `pandas_tool`: auto-detect missing > 30%, extreme outliers (3×IQR, > 1%), dominant category (> 70%). Append manual insights from prior session steps (root_cause, ab_test, etc.). Sort by priority, keep top 5.
3. **Priority chart** → `plotly_tool`: horizontal bar, colored by Critical / Important / FYI.

### Rules
- Generate insights programmatically from real data — do NOT summarize from memory
- After auto_insights, manually append findings from prior session tool calls
- Maximum 5 insights — prioritize by impact × confidence
- Each insight: **So What** (numbers) + **Why** (hypothesis) + **Now What** (concrete action)
- Recommendations must be actionable with owner + timeline, not "conduct further analysis"
```

- [ ] **Step 2: Create `DETAILS.md`**

Migration starting point: copy the current `SKILL.md` body (everything after frontmatter) into `DETAILS.md` as-is. Clean up afterward if needed.

- [ ] **Step 3: Verify + commit**

```
python -c "
from backend.skills import SkillRegistry
r = SkillRegistry.from_path('skills').load()
s = r.get('insight_synthesis')
print('core words:', len(s.core_markdown.split()))
print('has_details:', s.has_details)
"
```

```bash
git add skills/insight_synthesis/SKILL.md skills/insight_synthesis/DETAILS.md
git commit -m "refactor(skills): split insight_synthesis into core + details"
```

---

## Task 11: Migrate `plotly_tool` (741 → ~115 words)

**Files:**
- Modify: `skills/plotly_tool/SKILL.md`
- Create: `skills/plotly_tool/DETAILS.md`

- [ ] **Step 1: Write new `SKILL.md`**

```markdown
---
name: Plotly Tool
description: Построение интерактивных графиков через Plotly. Единственный инструмент для визуализации — используй всегда когда нужен chart/plot/diagram.
kind: tool
tool_key: plotly_tool
triggers: график, графики, графика, диаграмм, диаграмма, визуализац, визуализация, plotly, chart, charts, plot, scatter, bar, line, pie, histogram, heatmap, столбчат, линейн
---

## plotly_tool — interactive charts

Entry: Python code. Exit: Plotly Figure wrapped via `chart.result(fig, artifact_name="...")`.

### API
```python
chart.result(fig: go.Figure, artifact_name: str) -> tool_result
db.query_dataframe(sql: str) -> pd.DataFrame  # if DB connected
```

### Scope
`px`, `go`, `chart`, `df`, `pd`, `np` always available.
`db`, `db_connection` — when DB session active.
All variables from prior tool calls available by name (see sandbox block in system prompt).

### Final result protocol
The last expression must be `tool_result` — the sandbox captures only the last expression.

```python
tool_result = chart.result(fig, artifact_name="chart_name")
tool_result
```

A print or assignment as the final line produces a silent empty result.

### Rules
- Never call `pd.read_csv()` — `df` is already in scope
- `len(df) > 5000` → `df.sample(5000, random_state=42)` first
- Always set `title` and axis `labels` for readability
- Never use matplotlib, seaborn, or `.plot()`
```

- [ ] **Step 2: Create `DETAILS.md`**

Migration starting point: copy the current `SKILL.md` body (everything after frontmatter) into `DETAILS.md` as-is. Clean up afterward if needed.

- [ ] **Step 3: Verify + commit**

```
python -c "
from backend.skills import SkillRegistry
r = SkillRegistry.from_path('skills').load()
s = r.get('plotly_tool')
print('core words:', len(s.core_markdown.split()))
print('has_details:', s.has_details)
"
```

```bash
git add skills/plotly_tool/SKILL.md skills/plotly_tool/DETAILS.md
git commit -m "refactor(skills): split plotly_tool into core + details"
```

---

## Task 12: Migrate `csv_summarizer` (679 → ~130 words)

**Files:**
- Modify: `skills/csv_summarizer/SKILL.md`
- Create: `skills/csv_summarizer/DETAILS.md`

- [ ] **Step 1: Write new `SKILL.md`**

```markdown
---
name: CSV Summarizer
description: Fast automatic dataset overview — types, missing values, statistics, top values, and basic visualizations.
triggers: overview, summary, describe dataset, show structure, what's in the file, initial analysis, csv summarizer, обзор, резюме, опиши датасет, покажи структуру, что в файле, первичный анализ
---

## CSV Summarizer — quick dataset overview

Use when a file was just uploaded or a quick initial analysis is needed (not deep EDA).

### Algorithm (5 steps)
1. **Size + sample** → `pandas_tool`: shape, memory MB, first 5 rows.
2. **Schema + types** → `pandas_tool`: dtype, null_pct, unique count, `likely_type` (ID / binary / low_cardinality / high_cardinality / datetime-like / numeric).
3. **Descriptive statistics** → `pandas_tool`: `df.describe(include="all")`.
4. **Top categorical values** → `pandas_tool`: top-3 per column; if > 5 cat columns, select top-5 by entropy.
5. **Visualizations** → `plotly_tool`: missing values bar chart + numeric histograms (up to 4 non-ID columns).

### Rules
- ALWAYS start with Step 1 — agent needs scale context before proceeding
- Dataset > 500 columns → show only Steps 1–2, then ask which blocks are needed
- Columns with unique == nrows → mark as `ID`, exclude from histograms
- Object columns with `likely_type == "datetime-like"` → warn to cast to datetime
```

- [ ] **Step 2: Create `DETAILS.md`**

Migration starting point: copy the current `SKILL.md` body (everything after frontmatter) into `DETAILS.md` as-is. Clean up afterward if needed.

- [ ] **Step 3: Verify + commit**

```
python -c "
from backend.skills import SkillRegistry
r = SkillRegistry.from_path('skills').load()
s = r.get('csv_summarizer')
print('core words:', len(s.core_markdown.split()))
print('has_details:', s.has_details)
"
```

```bash
git add skills/csv_summarizer/SKILL.md skills/csv_summarizer/DETAILS.md
git commit -m "refactor(skills): split csv_summarizer into core + details"
```

---

## Task 13: Migrate `cohort_analysis` (560 → ~120 words)

**Files:**
- Modify: `skills/cohort_analysis/SKILL.md`
- Create: `skills/cohort_analysis/DETAILS.md`

- [ ] **Step 1: Write new `SKILL.md`**

```markdown
---
name: Cohort Analysis
description: User retention and LTV analysis by cohorts (date of first event).
triggers: cohort, retention, ltv, churn, когорт, удержание, отток
---

## Cohort Analysis

User retention and LTV analysis grouped by cohorts (period of first event).

### Algorithm (3 steps)
1. **Cohorts + retention + LTV** → `pandas_tool`: auto-detect user/date columns, compute `cohort_month`, `period_number`, `retention_pct` matrix. Build `ltv_cumulative` if revenue column exists.
2. **Retention heatmap** → `plotly_tool`: `px.imshow` with Blues scale; Y-axis labels include cohort size `(n=X)`.
3. **Cohort comparison** → `pandas_tool`: rank cohorts by period-1 retention, surface best and worst.

### Rules
- ALWAYS auto-detect `user_col` and `date_col` from names and types — never hardcode
- ALWAYS auto-select granularity: range ≤ 90 days → `W`, else → `M`
- ALWAYS validate period_number=0 = 100% per cohort; warn if not
- ALWAYS show `(n=X)` in Y-axis labels
- Compute `nunique(user_col)` per cohort first, then divide by `cohort_sizes` — do not pre-aggregate
```

- [ ] **Step 2: Create `DETAILS.md`**

Migration starting point: copy the current `SKILL.md` body (everything after frontmatter) into `DETAILS.md` as-is. Clean up afterward if needed.

- [ ] **Step 3: Verify + commit**

```
python -c "
from backend.skills import SkillRegistry
r = SkillRegistry.from_path('skills').load()
s = r.get('cohort_analysis')
print('core words:', len(s.core_markdown.split()))
print('has_details:', s.has_details)
"
```

```bash
git add skills/cohort_analysis/SKILL.md skills/cohort_analysis/DETAILS.md
git commit -m "refactor(skills): split cohort_analysis into core + details"
```

---

## Task 14: Migrate `duckdb_analysis` (558 → ~130 words)

**Files:**
- Modify: `skills/duckdb_analysis/SKILL.md`
- Create: `skills/duckdb_analysis/DETAILS.md`

- [ ] **Step 1: Write new `SKILL.md`**

```markdown
---
name: DuckDB Large File Analysis
description: SQL analysis of large CSV and Parquet files via DuckDB — no in-memory loading, supports JOINs across multiple files.
triggers: duckdb, large file, large csv, parquet, sql on file, multiple files, join files, out of memory, large dataset, gb file, большой файл, несколько файлов, большой датасет
---

## DuckDB Large File Analysis

SQL analysis of large CSV/Parquet files via DuckDB — no in-memory loading.
In this project `sql_tool` uses DuckDB — queries run directly on files.

### Key functions (pass as question to `sql_tool`)
```sql
read_csv_auto('/path/file.csv')           -- auto-detects types, delimiter, encoding
read_parquet('/path/*.parquet')           -- glob patterns work
read_csv('/path', encoding='cp1251', delim=';')  -- explicit options
```

### Workflow
1. Locate file path — ask user if unknown; glob `/uploads/*.csv` if partially known
2. Explore schema: `DESCRIBE SELECT * FROM read_csv_auto(...)` + `LIMIT 10`
3. Aggregate/filter/join directly on files without loading into memory
4. After `sql_tool` → result is available as `df` in `pandas_tool`

### Rules
- ALWAYS use `read_csv_auto` — auto-detects everything
- ALWAYS add `LIMIT` when initially exploring files > 1 GB
- Never guess file paths — ask explicitly if unknown
- Cyrillic encoding: try auto_detect first; if garbled → `encoding='cp1251'`
- `PERCENTILE_CONT` is DuckDB-specific, not standard SQL
```

- [ ] **Step 2: Create `DETAILS.md`**

Migration starting point: copy the current `SKILL.md` body (everything after frontmatter) into `DETAILS.md` as-is. Clean up afterward if needed.

- [ ] **Step 3: Verify + commit**

```
python -c "
from backend.skills import SkillRegistry
r = SkillRegistry.from_path('skills').load()
s = r.get('duckdb_analysis')
print('core words:', len(s.core_markdown.split()))
print('has_details:', s.has_details)
"
```

```bash
git add skills/duckdb_analysis/SKILL.md skills/duckdb_analysis/DETAILS.md
git commit -m "refactor(skills): split duckdb_analysis into core + details"
```

---

## Task 15: Migrate `cohort_analysis_advanced` (464 → ~115 words)

**Files:**
- Modify: `skills/cohort_analysis_advanced/SKILL.md`
- Create: `skills/cohort_analysis_advanced/DETAILS.md`

- [ ] **Step 1: Write new `SKILL.md`**

```markdown
---
name: Когортный анализ (расширенный)
description: Расширенный когортный анализ — retention, LTV, revenue cohorts, сравнение когорт и визуализация heatmap.
triggers: когорт ltv, revenue cohort, ltv когорт, выручка по когортам, сравнение когорт, продвинутый когортный, cohort revenue, lifetime value
---

## Когортный анализ (расширенный)

Для детального когортного анализа с LTV, revenue-когортами и сравнением когорт.
Для базового retention используй `cohort_analysis`.

### Алгоритм (4 шага)
1. **Retention + LTV матрица** → `pandas_tool`: когорта = месяц первого события, `period_number`, `retention_pct`. Если есть `revenue` → строит `ltv_cumulative` и `ltv_per_user`.
2. **Retention heatmap** → `plotly_tool`: `px.imshow`, цветовая шкала Blues.
3. **LTV heatmap** → `plotly_tool`: `px.imshow`, цветовая шкала Greens. Если revenue нет → placeholder.
4. **Сравнение когорт** → `plotly_tool`: grouped bar по периодам 0, 1, 3, 6, 12.

### Правила
- Гранулярность по умолчанию — месяц (`M`); для молодых продуктов — неделя (`W`)
- Когорт > 24 → показывать только последние 12 для читаемости
- LTV монотонно растёт (кумулятивная сумма) — если убывает, ошибка в данных
- Нормальный retention P1: e-commerce 20–30%, SaaS 40–60%
```

- [ ] **Step 2: Create `DETAILS.md`**

Migration starting point: copy the current `SKILL.md` body (everything after frontmatter) into `DETAILS.md` as-is. Clean up afterward if needed.

- [ ] **Step 3: Verify + commit**

```
python -c "
from backend.skills import SkillRegistry
r = SkillRegistry.from_path('skills').load()
s = r.get('cohort_analysis_advanced')
print('core words:', len(s.core_markdown.split()))
print('has_details:', s.has_details)
"
```

```bash
git add skills/cohort_analysis_advanced/SKILL.md skills/cohort_analysis_advanced/DETAILS.md
git commit -m "refactor(skills): split cohort_analysis_advanced into core + details"
```

---

## Task 16: Migrate `sql_tool` (491 → ~140 words)

**Files:**
- Modify: `skills/sql_tool/SKILL.md`
- Create: `skills/sql_tool/DETAILS.md`

- [ ] **Step 1: Write new `SKILL.md`**

```markdown
---
name: SQL Tool
description: Аналитические запросы к БД и CSV в DuckDB. Возвращает табличный артефакт.
kind: tool
tool_key: sql_tool
triggers: sql, база данных, таблица, таблиц, запрос, query, database, db, выборка, джойн, join, агрегация, агрег, посчитай, сумм, средн, медиан, pivot, dataset, датасет, данных
---

## sql_tool — SQL queries

Entry: one `question` argument in natural language. Tool generates safe SELECT, returns table artifact.

### When to use
- Aggregations with GROUP BY, JOIN, subqueries, window functions
- Working with CSV loaded into DuckDB session
- When `database_tool` is too simple (no aggregation capability)

Prefer `database_tool` for: list tables, describe columns, preview rows.

### Question quality
- ✅ `"Average age by Age column in table titanic"`
- ✅ `"Top-5 categories by revenue sum in table sales"`
- ❌ `"Show data"` — unclear
- ❌ `"Analyze"` — too abstract

### Final result protocol
This tool returns a table artifact directly — no code execution, no `tool_result` needed.
The result variable name is stated in the tool response. Use that exact name in subsequent tools — do not invent `sql_dataset` or `data`.
For DB sessions: prefer `db.query_dataframe(sql)` inside `plotly_tool` directly (one call instead of two).

### Limits
- Read-only: INSERT / UPDATE / DELETE / DROP blocked
- Max 200 rows in result
```

- [ ] **Step 2: Create `DETAILS.md`**

Migration starting point: copy the current `SKILL.md` body (everything after frontmatter) into `DETAILS.md` as-is. Clean up afterward if needed.

- [ ] **Step 3: Verify + commit**

```
python -c "
from backend.skills import SkillRegistry
r = SkillRegistry.from_path('skills').load()
s = r.get('sql_tool')
print('core words:', len(s.core_markdown.split()))
print('has_details:', s.has_details)
"
```

```bash
git add skills/sql_tool/SKILL.md skills/sql_tool/DETAILS.md
git commit -m "refactor(skills): split sql_tool into core + details"
```

---

## Task 17: Migrate `database_tool` (447 → ~125 words)

**Files:**
- Modify: `skills/database_tool/SKILL.md`
- Create: `skills/database_tool/DETAILS.md`

- [ ] **Step 1: Write new `SKILL.md`**

```markdown
---
name: Database Tool
description: Быстрый просмотр структуры БД — таблицы, колонки, превью строк, схемы. Без генерации SQL.
kind: tool
tool_key: database_tool
triggers: таблицы, покажи таблицы, структура бд, колонки, схема, превью, первые строки, список таблиц, describe, list tables, show tables, какие таблицы, перечисли таблицы
---

## Database Tool — structure inspection

Light tool for structural DB operations. **Does not generate SQL** — calls catalog directly, fast.

### API
```python
database_tool(
    action: Literal["list_tables", "describe_table", "preview", "list_schemas"],
    table: str | None = None,      # required for describe_table, preview
    db_schema: str | None = None,  # use result of list_schemas, never guess
    limit: int = 10,               # max 50, preview only
)
```

### Mandatory order
⚠️ Never call `preview` or `describe_table` with a guessed table name.

1. Schema unknown → `list_schemas` → `list_tables(db_schema=<result>)`
2. Schema known → `list_tables` directly
3. Only after getting real table names → `preview` or `describe_table`

If `list_tables` returns empty → call `list_schemas` first, then retry.

### Final result protocol
This tool returns results directly to the agent — no code execution, no `tool_result` needed.
Use `describe_table` or `preview` results as context; pass table names to `sql_tool` for further querying.

### When NOT to use
- Complex queries (JOIN, GROUP BY, subqueries) → `sql_tool`
- Aggregation, filtering, computation → `sql_tool`
- CSV data → `pandas_tool`
```

- [ ] **Step 2: Create `DETAILS.md`**

Migration starting point: copy the current `SKILL.md` body (everything after frontmatter) into `DETAILS.md` as-is. Clean up afterward if needed.

- [ ] **Step 3: Verify + commit**

```
python -c "
from backend.skills import SkillRegistry
r = SkillRegistry.from_path('skills').load()
s = r.get('database_tool')
print('core words:', len(s.core_markdown.split()))
print('has_details:', s.has_details)
"
```

```bash
git add skills/database_tool/SKILL.md skills/database_tool/DETAILS.md
git commit -m "refactor(skills): split database_tool into core + details"
```

---

## Task 18: Final verification — all skills load + prompt size regression

**Files:**
- No code changes — verification only

- [ ] **Step 1: Verify all skills load**

```
python -c "
from backend.skills import SkillRegistry

r = SkillRegistry.from_path('skills').load()
skills = r.list_skills()
print(f'Total skills loaded: {len(skills)}')
print()

over_limit = []
for s in sorted(skills, key=lambda x: len(x.core_markdown.split()), reverse=True):
    words = len(s.core_markdown.split())
    flag = ' ⚠️ OVER LIMIT' if words > 200 else ''
    print(f'  {s.skill_id}: {words} words core, has_details={s.has_details}{flag}')
    if words > 200:
        over_limit.append(s.skill_id)

if over_limit:
    print(f'Skills over 200-word limit: {over_limit}')
else:
    print('All skills within 200-word limit ✅')
"
```

Expected: all 22 skills load, none over 200 words, 14 with has_details=True.

- [ ] **Step 2: Run full test suite**

```
pytest tests/test_skills_registry.py tests/test_get_tool_instructions_tool.py tests/test_skills_e2e.py tests/test_session_store_skills.py -v
```

Expected: all pass.

- [ ] **Step 3: Spot check retrieval with hint**

```python
from backend.skills import SkillRegistry
from backend.tools.impl.get_tool_instructions_tool import GetToolInstructionsTool

r = SkillRegistry.from_path('skills').load()
tool = GetToolInstructionsTool(r)

# Should include hint
result = tool._run(skill_id='plotly_tool')
assert "details=True" in result, "Hint missing"
print("Core with hint ✅")

# details=True should have code examples
details = tool._run(skill_id='plotly_tool', details=True)
assert '```python' in details, "No code in details"
print("Details has code ✅")

# Missing details → graceful fallback, no core repeat
for sid in ['rag_tool', 'review_tool', 'pandas_tool']:
    r2 = tool._run(skill_id=sid, details=True)
    assert 'not available' in r2.lower(), f"Bad fallback for {sid}"
print("Fallback for no-details skills ✅")
```

Run: `python -c "<paste above>"`

- [ ] **Step 4: Commit**

```bash
git add .
git commit -m "test(skills): final verification — all skills within limits, retrieval policy correct"
```

---

## Summary

After completing all tasks:

| Metric | Before | After |
|--------|--------|-------|
| Skills with >400-word core | 14 | 0 |
| Average core size (heavy skills) | ~750 words | ~125 words |
| Skills with DETAILS.md | 0 | 14 |
| Retrieval levels | 2 (empty / all) | 3 (brief / core / extended) |
| Hint injection location | content files | tool layer |
| Semantic lint | none | Python block > 5 lines → error |
