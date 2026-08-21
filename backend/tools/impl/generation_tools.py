from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, ClassVar, Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from backend.agent.services.message_builder import (
    history_artifact_summary,
    truncate,
)
from backend.tools.instructions import tool_description

SummaryStatus = Literal["ok", "empty_context"]
ReportStatus = Literal[
    "ok",
    "missing_session_id",
    "session_not_found",
    "empty_history",
    "empty_artifacts",
    "error",
]


class GenerateSummaryToolArgs(BaseModel):
    focus: str = Field(
        default="",
        description=(
            "Optional summary focus requested by the user, for example executive, "
            "management, risks, or next steps."
        ),
    )
    max_history_items: int = Field(
        default=12,
        ge=1,
        le=50,
        description="Maximum number of recent chat messages to include.",
    )


class GenerateSummaryToolResult(BaseModel):
    status: SummaryStatus
    message: str
    summary_markdown: str
    history_items_used: int = 0
    artifact_count: int = 0


class GenerateSummaryTool(BaseTool):
    """Build a generic summary from current session context."""

    name: str = "generate_summary_tool"
    description: str = tool_description("generate_summary_tool")
    args_schema: type[BaseModel] = GenerateSummaryToolArgs
    response_format: str = "content"
    parallel_safe: ClassVar[bool] = True

    _history: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    _session_notes: str = PrivateAttr(default="")
    _artifact_summaries: list[str] = PrivateAttr(default_factory=list)

    def __init__(
        self,
        *,
        history: list[dict[str, Any]],
        session_notes: str = "",
        artifact_summaries: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._history = [dict(item) for item in history if isinstance(item, dict)]
        self._session_notes = str(session_notes or "").strip()
        self._artifact_summaries = [
            str(item).strip() for item in (artifact_summaries or []) if str(item or "").strip()
        ]

    def _run(self, focus: str = "", max_history_items: int = 12) -> str:
        result = self._build_summary(
            focus=str(focus or "").strip(),
            max_history_items=max_history_items,
        )
        return result.model_dump_json(indent=2)

    async def _arun(self, focus: str = "", max_history_items: int = 12) -> str:
        return self._run(focus=focus, max_history_items=max_history_items)

    def _build_summary(
        self,
        *,
        focus: str,
        max_history_items: int,
    ) -> GenerateSummaryToolResult:
        recent_history = self._recent_history(max_history_items)
        artifact_summaries = self._artifact_summaries
        has_context = bool(recent_history or self._session_notes or artifact_summaries)
        if not has_context:
            return GenerateSummaryToolResult(
                status="empty_context",
                message="No session context is available to summarize.",
                summary_markdown="",
            )

        summary = self._render_markdown(
            focus=focus or "session summary",
            recent_history=recent_history,
            artifact_summaries=artifact_summaries,
        )
        return GenerateSummaryToolResult(
            status="ok",
            message="Summary generated from session context.",
            summary_markdown=summary,
            history_items_used=len(recent_history),
            artifact_count=len(artifact_summaries),
        )

    def _recent_history(self, max_history_items: int) -> list[dict[str, str]]:
        limit = max(1, min(int(max_history_items), 50))
        recent: list[dict[str, str]] = []
        for item in self._history[-limit:]:
            content = truncate(str(item.get("content", "")).strip(), 700)
            if not content:
                continue
            recent.append(
                {
                    "role": str(item.get("role") or "assistant").strip() or "assistant",
                    "content": content,
                }
            )
        return recent

    def _render_markdown(
        self,
        *,
        focus: str,
        recent_history: list[dict[str, str]],
        artifact_summaries: list[str],
    ) -> str:
        lines: list[str] = [
            f"## Summary: {focus}",
            "",
        ]
        if self._session_notes:
            lines.extend(
                [
                    "### Session Notes",
                    self._session_notes,
                    "",
                ]
            )
        if recent_history:
            lines.append("### Recent Context")
            for item in recent_history:
                role = item["role"].title()
                lines.append(f"- **{role}:** {item['content']}")
            lines.append("")
        if artifact_summaries:
            lines.append("### Artifacts")
            lines.extend(f"- {summary}" for summary in artifact_summaries)
            lines.append("")
        return "\n".join(lines).strip()


class GenerateReportToolArgs(BaseModel):
    title: str = Field(
        default="",
        description="Optional report title requested by the user.",
    )


class GenerateReportToolResult(BaseModel):
    status: ReportStatus
    message: str
    download_url: str | None = None
    file_name: str | None = None


class GenerateReportTool(BaseTool):
    """Export the current persisted session as a downloadable report."""

    name: str = "generate_report_tool"
    description: str = tool_description("generate_report_tool")
    args_schema: type[BaseModel] = GenerateReportToolArgs
    response_format: str = "content"
    parallel_safe: ClassVar[bool] = False

    _session_id: str = PrivateAttr(default="")
    _user_id: int = PrivateAttr(default=0)
    _session_store: Any = PrivateAttr(default=None)
    _blob_store: Any = PrivateAttr(default=None)

    def __init__(
        self,
        *,
        session_id: str,
        user_id: int,
        session_store: Any,
        blob_store: Any,
    ) -> None:
        super().__init__()
        self._session_id = str(session_id or "").strip()
        self._user_id = int(user_id)
        self._session_store = session_store
        self._blob_store = blob_store

    def _run(self, title: str = "") -> str:
        del title
        result = self._build_report()
        if result.status == "error":
            raise RuntimeError(result.message)
        return result.model_dump_json(indent=2)

    async def _arun(self, title: str = "") -> str:
        return self._run(title=title)

    def _build_report(self) -> GenerateReportToolResult:
        if not self._session_id:
            return GenerateReportToolResult(
                status="missing_session_id",
                message="Cannot generate report: session_id is missing.",
            )

        try:
            from backend.auth.blob_store import BlobWrite
            from backend.services.report_export import build_report_docx

            if self._session_store is None or self._blob_store is None or self._user_id <= 0:
                raise RuntimeError("persistent report storage is unavailable")
            state = self._session_store.load_session(self._session_id)
            if state is None:
                return GenerateReportToolResult(
                    status="session_not_found",
                    message="Cannot generate report: session was not found.",
                )
            if not state.chat_history:
                return GenerateReportToolResult(
                    status="empty_history",
                    message="Cannot generate report: chat history is empty.",
                )
            if not state.artifacts:
                return GenerateReportToolResult(
                    status="empty_artifacts",
                    message="Cannot generate report: session has no artifacts.",
                )

            with tempfile.TemporaryDirectory(prefix="analyst-report-") as temp_dir:
                result = build_report_docx(
                    session_id=self._session_id,
                    chat_history=state.chat_history,
                    artifacts=state.artifacts,
                    output_dir=Path(temp_dir),
                    base_download_url="reports/download/",
                )
                content = Path(result.file_path).read_bytes()
            blob_id = self._blob_store.put_many(
                user_id=self._user_id,
                session_id=self._session_id,
                kind="report",
                items=[
                    BlobWrite(
                        logical_name=result.file_name,
                        media_type=(
                            "application/vnd.openxmlformats-officedocument."
                            "wordprocessingml.document"
                        ),
                        content=content,
                    )
                ],
            )[0]
            return GenerateReportToolResult(
                status="ok",
                message="Report generated.",
                download_url=f"reports/download/{blob_id}/{result.file_name}",
                file_name=result.file_name,
            )
        except Exception as exc:
            return GenerateReportToolResult(
                status="error",
                message=f"Cannot generate report: {exc}",
            )


def artifact_summaries_from_history(history: list[dict[str, Any]]) -> list[str]:
    summary = history_artifact_summary(history)
    if not summary:
        return []
    return [line.strip("- ").strip() for line in summary.splitlines() if line.strip()]
