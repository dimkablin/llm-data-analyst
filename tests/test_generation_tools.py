from __future__ import annotations

import json
import unittest

from backend.tools.impl.generation_tools import (
    GenerateReportTool,
    GenerateReportToolResult,
    GenerateSummaryTool,
    GenerateSummaryToolResult,
)


class GenerateSummaryToolTests(unittest.TestCase):
    def test_summary_tool_returns_typed_contextual_summary(self) -> None:
        tool = GenerateSummaryTool(
            history=[
                {"role": "user", "content": "Analyze revenue by region."},
                {
                    "role": "assistant",
                    "content": "North revenue grew by 12%; South declined by 4%.",
                },
            ],
            session_notes="Dataset covers 2024 regional sales.",
            artifact_summaries=["table: revenue_by_region"],
        )

        result = GenerateSummaryToolResult.model_validate_json(
            tool._run(focus="management summary")
        )

        self.assertEqual(result.status, "ok")
        self.assertIn("management summary", result.summary_markdown)
        self.assertIn("North revenue grew by 12%", result.summary_markdown)
        self.assertEqual(result.history_items_used, 2)
        self.assertEqual(result.artifact_count, 1)

    def test_summary_tool_returns_empty_context_status_without_sources(self) -> None:
        tool = GenerateSummaryTool(history=[], session_notes="", artifact_summaries=[])

        result = GenerateSummaryToolResult.model_validate_json(tool._run(focus=""))

        self.assertEqual(result.status, "empty_context")
        self.assertIn("No session context", result.message)


class GenerateReportToolTests(unittest.TestCase):
    def test_report_tool_requires_session_id(self) -> None:
        tool = GenerateReportTool(session_id="", storage_dir="storage", session_ttl_days=7)

        result = GenerateReportToolResult.model_validate_json(tool._run(title=""))

        self.assertEqual(result.status, "missing_session_id")
        self.assertIsNone(result.download_url)
        self.assertIn("session_id", result.message)

    def test_report_tool_returns_json_not_markdown(self) -> None:
        tool = GenerateReportTool(session_id="", storage_dir="storage", session_ttl_days=7)

        raw = tool._run(title="")
        payload = json.loads(raw)

        self.assertIsInstance(payload, dict)
        self.assertNotIn("[Download", raw)
