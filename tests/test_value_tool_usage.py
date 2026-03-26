from __future__ import annotations

from dataclasses import replace
import unittest

import pandas as pd

from backend.agent import AgentRunner
from backend.agent.tools.value_tool import ValueTool
from backend.core import Settings
from backend.tools import build_runtime_capability_context


class ValueToolUsageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame({"sales": [10, 20, 30]})
        self.tool = ValueTool(self.df, execution_timeout_sec=5.0, tool_cache_size=0)

    def test_short_value_like_output_passes(self) -> None:
        code = """
tool_result = {
    "schema_version": "1.0",
    "artifact_type": "value",
    "items": {
        "row_count": len(df),
        "status": "loaded"
    }
}
tool_result
"""

        text, payload = self.tool._run(code)

        self.assertIn("value_tool", text)
        self.assertEqual(payload["value"]["row_count"], 3)
        self.assertEqual(payload["value"]["status"], "loaded")

    def test_long_explanatory_string_is_rejected(self) -> None:
        code = """
tool_result = {
    "schema_version": "1.0",
    "artifact_type": "value",
    "items": {
        "analysis_result": "Analysis is not possible because the dataset does not contain fuel data. A valid answer would require external sources, macro indicators, taxes, exchange rates, logistics data, and market context."
    }
}
tool_result
"""

        text, payload = self.tool._run(code)

        self.assertIn("value_tool", text)
        self.assertIn("value-like", text)
        self.assertIsNone(payload["value"])

    def test_prompt_discourages_using_value_tool_instead_of_external_research(self) -> None:
        settings = replace(
            Settings(),
            agent_cache_enabled=False,
            llm_warmup_enabled=False,
        )
        runner = AgentRunner(settings)
        capability_context = build_runtime_capability_context(
            available_tool_keys={"value_tool", "search_tool"},
            has_dataframe=True,
            has_db_source=False,
        )

        prompt = runner._think_system_prompt(capability_context)

        self.assertIn("value_tool", prompt)
        self.assertIn("search_tool", prompt)
        self.assertTrue(
            "search_tool" in prompt or "internet" in prompt.lower(),
            msg="Prompt should warn not to use value_tool instead of web search",
        )


if __name__ == "__main__":
    unittest.main()
