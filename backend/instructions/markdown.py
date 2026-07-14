from __future__ import annotations

import re
from pathlib import Path

import yaml

from backend.instructions.models import InstructionDocument, InstructionMetadata

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*(?:\n|$)", re.DOTALL)


class InstructionMarkdownError(ValueError):
    """Raised when an instruction markdown file cannot be parsed."""


def parse_instruction_markdown(
    text: str,
    *,
    source_path: str | Path,
    default_id: str,
    default_kind: str = "analytical",
    details_markdown: str | None = None,
) -> InstructionDocument:
    source = Path(source_path)
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        raise InstructionMarkdownError(
            f"Instruction file {source.name} must start with YAML frontmatter."
        )

    try:
        raw_frontmatter = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        raise InstructionMarkdownError(
            f"Invalid YAML frontmatter in {source.name}: {exc}"
        ) from exc
    if not isinstance(raw_frontmatter, dict):
        raise InstructionMarkdownError(
            f"Frontmatter in {source.name} must be a YAML mapping."
        )

    raw_frontmatter = {
        "id": default_id,
        "kind": default_kind,
        **raw_frontmatter,
    }
    body = text[match.end() :].strip()
    if not body:
        raise InstructionMarkdownError(
            f"Instruction file {source.name} must contain markdown instructions."
        )

    try:
        metadata = InstructionMetadata.model_validate(raw_frontmatter)
    except Exception as exc:
        raise InstructionMarkdownError(
            f"Invalid instruction metadata in {source.name}: {exc}"
        ) from exc

    return InstructionDocument(
        metadata=metadata,
        body=body,
        source_path=source,
        details_markdown=details_markdown,
    )


def read_instruction_document(
    path: str | Path,
    *,
    default_id: str,
    default_kind: str = "analytical",
    details_path: str | Path | None = None,
    max_body_bytes: int | None = None,
    max_details_bytes: int | None = None,
) -> InstructionDocument:
    source = Path(path)
    try:
        stat = source.stat()
    except OSError as exc:
        raise InstructionMarkdownError(f"Failed to stat instruction file {source}: {exc}") from exc
    if max_body_bytes is not None and stat.st_size > max_body_bytes:
        raise InstructionMarkdownError(
            f"Instruction file {source.name} exceeds max size of {max_body_bytes} bytes."
        )

    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise InstructionMarkdownError(f"Failed to read instruction file {source}: {exc}") from exc

    details_markdown: str | None = None
    if details_path is not None:
        details = Path(details_path)
        try:
            details_stat = details.stat()
        except OSError as exc:
            raise InstructionMarkdownError(f"Failed to stat {details}: {exc}") from exc
        if max_details_bytes is not None and details_stat.st_size > max_details_bytes:
            raise InstructionMarkdownError(
                f"Details file {details.name} exceeds max size of {max_details_bytes} bytes."
            )
        try:
            details_markdown = details.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise InstructionMarkdownError(f"Failed to read {details}: {exc}") from exc

    return parse_instruction_markdown(
        text,
        source_path=source,
        default_id=default_id,
        default_kind=default_kind,
        details_markdown=details_markdown,
    )
