from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, PrivateAttr

from backend.instructions import (
    InstructionDocument,
    InstructionKind,
    InstructionMarkdownError,
    read_instruction_document,
)

_DEFAULT_MAX_TOOL_BYTES = 12 * 1024
_DEFAULT_MAX_TOOL_DETAILS_BYTES = 64 * 1024
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


class ToolInstructionError(ValueError):
    """Raised when tool instruction markdown is missing or invalid."""


class ToolInstructionRegistry(BaseModel):
    """Loads executable tool instructions from top-level tools/*/TOOL.md files."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tools_dir: Path
    max_tool_bytes: int = _DEFAULT_MAX_TOOL_BYTES
    max_details_bytes: int = _DEFAULT_MAX_TOOL_DETAILS_BYTES
    _documents_by_tool_key: dict[str, InstructionDocument] = PrivateAttr(default_factory=dict)
    _loaded: bool = PrivateAttr(default=False)

    @classmethod
    def from_path(cls, tools_dir: str | Path) -> ToolInstructionRegistry:
        return cls(tools_dir=Path(tools_dir))

    def load(self, *, force: bool = False) -> ToolInstructionRegistry:
        if self._loaded and not force:
            return self
        self._loaded = True
        self._documents_by_tool_key = {}

        if not self.tools_dir.exists():
            raise ToolInstructionError(
                f"Tools instruction path does not exist: {self.tools_dir}"
            )
        if not self.tools_dir.is_dir():
            raise ToolInstructionError(f"Tools instruction path is not a directory: {self.tools_dir}")

        for tool_dir in sorted(self.tools_dir.iterdir()):
            if not tool_dir.is_dir():
                continue
            tool_md = tool_dir / "TOOL.md"
            if not tool_md.exists():
                continue
            details_md = tool_dir / "DETAILS.md"
            document = self._parse_tool_file(
                tool_md,
                details_path=details_md if details_md.exists() else None,
            )
            tool_key = str(document.metadata.tool_key or document.metadata.id)
            if tool_key in self._documents_by_tool_key:
                raise ToolInstructionError(
                    f"Duplicate tool instruction key '{tool_key}' in {tool_dir.name}/TOOL.md."
                )
            self._documents_by_tool_key[tool_key] = document
        return self

    def reload(self) -> ToolInstructionRegistry:
        return self.load(force=True)

    def list_tools(self) -> tuple[InstructionDocument, ...]:
        self.load()
        return tuple(self._documents_by_tool_key.values())

    def get(self, tool_key: str) -> InstructionDocument:
        self.load()
        normalized = str(tool_key or "").strip()
        if normalized not in self._documents_by_tool_key:
            raise ToolInstructionError(f"Unknown tool instruction key: {normalized}")
        return self._documents_by_tool_key[normalized]

    def get_optional(self, tool_key: str) -> InstructionDocument | None:
        self.load()
        return self._documents_by_tool_key.get(str(tool_key or "").strip())

    def description(self, tool_key: str, fallback: str = "") -> str:
        document = self.get_optional(tool_key)
        if document is None:
            return fallback
        return document.metadata.description or fallback

    def section(self, tool_key: str, heading: str) -> str:
        document = self.get(tool_key)
        return extract_markdown_section(document.body, heading)

    def build_brief_block(self, available_tool_keys: set[str] | frozenset[str]) -> str:
        allowed = {str(tool_key).strip() for tool_key in available_tool_keys if str(tool_key).strip()}
        documents = [
            document
            for document in self.list_tools()
            if str(document.metadata.tool_key or document.metadata.id) in allowed
        ]
        if not documents:
            return ""

        lines = [
            "## Инструменты (краткое описание)",
            "",
            "Для незнакомых или сложных инструментов вызывай `get_tool_instructions(tool_name)` "
            "перед первым использованием.",
            "",
        ]
        for document in documents:
            tool_key = str(document.metadata.tool_key or document.metadata.id)
            triggers_hint = ""
            if document.metadata.triggers:
                sample = ", ".join(document.metadata.triggers[:5])
                triggers_hint = f" (triggers: {sample})"
            lines.append(f"- `{tool_key}`: {document.metadata.description}{triggers_hint}")
        return "\n".join(lines).strip()

    def _parse_tool_file(
        self,
        path: Path,
        *,
        details_path: Path | None = None,
    ) -> InstructionDocument:
        try:
            document = read_instruction_document(
                path,
                default_id=path.parent.name,
                default_kind=InstructionKind.TOOL.value,
                details_path=details_path,
                max_body_bytes=self.max_tool_bytes,
                max_details_bytes=self.max_details_bytes,
            )
        except InstructionMarkdownError as exc:
            raise ToolInstructionError(str(exc)) from exc

        if document.metadata.kind != InstructionKind.TOOL:
            raise ToolInstructionError(
                f"{path.name}: tool instruction must use kind='tool'."
            )
        if not document.metadata.tool_key:
            raise ToolInstructionError(
                f"{path.name}: tool instruction must declare tool_key."
            )
        return document


def default_tools_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "tools"


@lru_cache(maxsize=1)
def get_default_tool_instruction_registry() -> ToolInstructionRegistry:
    return ToolInstructionRegistry.from_path(default_tools_dir()).load()


def tool_description(tool_key: str, fallback: str = "") -> str:
    return get_default_tool_instruction_registry().description(tool_key, fallback=fallback)


def tool_section(tool_key: str, heading: str) -> str:
    return get_default_tool_instruction_registry().section(tool_key, heading)


def tool_section_text(tool_key: str, heading: str) -> str:
    return strip_single_code_fence(tool_section(tool_key, heading))


def strip_single_code_fence(text: str) -> str:
    raw = str(text or "").strip()
    match = re.fullmatch(r"```[a-zA-Z0-9_-]*\s*\n(.*?)\n```", raw, flags=re.DOTALL)
    if match is None:
        return raw
    return match.group(1).strip()


def extract_markdown_section(markdown: str, heading: str) -> str:
    expected = heading.strip().lower()
    lines = str(markdown or "").splitlines()
    start_index: int | None = None
    start_level = 0

    for index, line in enumerate(lines):
        match = _HEADING_RE.match(line)
        if match is None:
            continue
        title = match.group(2).strip().lower()
        if title == expected:
            start_index = index + 1
            start_level = len(match.group(1))
            break

    if start_index is None:
        raise ToolInstructionError(f"Markdown section '{heading}' not found.")

    end_index = len(lines)
    for index in range(start_index, len(lines)):
        match = _HEADING_RE.match(lines[index])
        if match is not None and len(match.group(1)) <= start_level:
            end_index = index
            break

    return "\n".join(lines[start_index:end_index]).strip()
