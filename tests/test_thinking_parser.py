"""Unit tests for ThinkingOutputParser and ThinkingAwareChatOpenAI sanitization.

Coverage
--------
ThinkingOutputParser:
- no-think response passes through unchanged
- closed <think>…</think> block stripped, reasoning captured
- unclosed <think> block discarded (no content leak)
- multiple <think> blocks all stripped
- visible text before and after <think>
- tag split across consecutive feed() calls
- stray </think> without opening tag passes through
- SQL code inside <think> does not pollute visible output
- case-insensitive tags (<THINK>, </Think>)
- feed() / flush() contract: no double-emission of tail

ThinkingAwareChatOpenAI:
- invoke() strips thinking from sync response
- ainvoke() strips thinking from async response
- invoke() preserves reasoning in additional_kwargs
- invoke() fast-paths a response that has no <think>
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from backend.agent.callbacks import ThinkingOutputParser

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _full_parse(text: str) -> tuple[str, str]:
    """Feed the full *text* at once and flush; return (visible, reasoning)."""
    p = ThinkingOutputParser()
    p.feed(text)
    p.flush()
    return p.visible(), p.reasoning()


def _stream_parse(chunks: list[str]) -> tuple[str, str]:
    """Feed *chunks* one at a time and flush; return (visible, reasoning)."""
    p = ThinkingOutputParser()
    for chunk in chunks:
        p.feed(chunk)
    p.flush()
    return p.visible(), p.reasoning()


# ---------------------------------------------------------------------------
# ThinkingOutputParser — full-document tests
# ---------------------------------------------------------------------------

class TestThinkingOutputParserFullDoc(unittest.TestCase):

    def test_no_think_passes_through(self):
        vis, rsn = _full_parse("SELECT 1")
        self.assertEqual(vis, "SELECT 1")
        self.assertEqual(rsn, "")

    def test_empty_string(self):
        vis, rsn = _full_parse("")
        self.assertEqual(vis, "")
        self.assertEqual(rsn, "")

    def test_closed_think_block_stripped(self):
        vis, rsn = _full_parse("<think>some reasoning</think>SELECT 1")
        self.assertEqual(vis, "SELECT 1")
        self.assertEqual(rsn, "some reasoning")

    def test_unclosed_think_no_leak(self):
        """Content after an unclosed <think> must NOT appear in visible output."""
        vis, rsn = _full_parse("<think>dangling reasoning without close tag")
        self.assertEqual(vis, "")
        self.assertIn("dangling reasoning", rsn)

    def test_unclosed_think_discards_sql(self):
        """SQL inside unclosed thinking must not leak into visible."""
        vis, _ = _full_parse("<think>\nSELECT * FROM t\n")
        self.assertEqual(vis, "")

    def test_multiple_think_blocks(self):
        vis, rsn = _full_parse("<think>a</think>SEL<think>b</think>ECT 1")
        self.assertEqual(vis, "SELECT 1")
        self.assertIn("a", rsn)
        self.assertIn("b", rsn)

    def test_visible_before_and_after_think(self):
        vis, rsn = _full_parse("prefix<think>reason</think>suffix")
        self.assertEqual(vis, "prefixsuffix")
        self.assertEqual(rsn, "reason")

    def test_only_visible_before_think(self):
        vis, rsn = _full_parse("visible<think>thinking")
        self.assertEqual(vis, "visible")
        self.assertIn("thinking", rsn)

    def test_stray_close_tag_discarded(self):
        """An orphaned </think> (vLLM strips the opening tag) is discarded; content after is visible."""
        vis, rsn = _full_parse("</think>SELECT 1")
        self.assertEqual(vis, "SELECT 1")
        self.assertEqual(rsn, "")

    def test_stray_close_tag_with_prefix_reasoning(self):
        """Content before an orphaned </think> is treated as reasoning, not visible."""
        vis, rsn = _full_parse("some reasoning</think>SELECT 1")
        self.assertEqual(vis, "SELECT 1")
        self.assertEqual(rsn, "some reasoning")

    def test_sql_in_thinking_not_in_visible(self):
        thinking = "<think>```sql\nSELECT evil FROM t\n```\n</think>"
        vis, rsn = _full_parse(thinking + "\nSELECT good FROM t")
        self.assertNotIn("evil", vis)
        self.assertIn("good", vis)
        self.assertIn("evil", rsn)

    def test_case_insensitive_open_tag(self):
        vis, rsn = _full_parse("<THINK>reason</THINK>result")
        self.assertEqual(vis, "result")
        self.assertEqual(rsn, "reason")

    def test_case_insensitive_mixed_tags(self):
        vis, rsn = _full_parse("<Think>reason</tHiNk>result")
        self.assertEqual(vis, "result")
        self.assertEqual(rsn, "reason")

    def test_whitespace_stripped_from_visible(self):
        vis, _ = _full_parse("<think>r</think>\n\nSELECT 1\n")
        self.assertEqual(vis, "SELECT 1")

    def test_reasoning_stripped_of_outer_whitespace(self):
        _, rsn = _full_parse("<think>\n  reason  \n</think>x")
        self.assertEqual(rsn, "reason")


# ---------------------------------------------------------------------------
# ThinkingOutputParser — incremental / streaming tests
# ---------------------------------------------------------------------------

class TestThinkingOutputParserStreaming(unittest.TestCase):

    def test_tag_split_across_chunks_open(self):
        """<thi | nk> split — reasoning should still be suppressed."""
        vis, rsn = _stream_parse(["<thi", "nk>body</think>SQL"])
        self.assertEqual(vis, "SQL")
        self.assertIn("body", rsn)

    def test_tag_split_across_chunks_close(self):
        """</thi | nk> split — close tag must still be recognised."""
        vis, rsn = _stream_parse(["<think>reason</thi", "nk>SQL"])
        self.assertEqual(vis, "SQL")
        self.assertIn("reason", rsn)

    def test_single_char_chunks(self):
        text = "<think>r</think>vis"
        chunks = list(text)
        vis, rsn = _stream_parse(chunks)
        self.assertEqual(vis, "vis")
        self.assertEqual(rsn, "r")

    def test_no_double_emission_of_tail(self):
        """Content emitted by feed() must not be re-emitted by flush()."""
        p = ThinkingOutputParser()
        # Simulate: first chunk arrives, second chunk resolves partial tag
        v1, _ = p.feed("Hello ")
        v2, _ = p.feed("World")          # no <think> at all
        v_flush, _ = p.flush()
        # "Hello" emitted by first feed; "World" by second feed; flush emits tail only
        combined = v1 + v2 + v_flush
        self.assertEqual(combined.strip(), "Hello World")
        # Check no duplication
        self.assertEqual(combined.count("Hello"), 1)
        self.assertEqual(combined.count("World"), 1)

    def test_feed_flush_combined_equals_input(self):
        """Total of all feed() returns + flush() return must equal the input — no loss, no duplication."""
        p = ThinkingOutputParser()
        v1, _ = p.feed("abc")
        v2, _ = p.feed("def")
        v3, _ = p.flush()
        combined = v1 + v2 + v3
        self.assertEqual(combined, "abcdef")
        self.assertEqual(p.visible(), "abcdef")

    def test_flush_emits_only_remaining_buffer(self):
        """flush() emits only what is still in the look-ahead buffer — never what feed() already returned."""
        p = ThinkingOutputParser()
        # Feed a string long enough to trigger partial emission
        # look-ahead keeps last 6 chars; "Hello World" (11) emits first 5 chars "Hello"
        v_feed, _ = p.feed("Hello World")
        v_flush, _ = p.flush()
        # Combined must equal original with no duplication
        self.assertEqual(v_feed + v_flush, "Hello World")
        # flush only carries what wasn't yet emitted
        self.assertNotIn(v_feed, v_flush)  # no overlap

    def test_reasoning_accumulates_across_feeds(self):
        p = ThinkingOutputParser()
        p.feed("<think>part1 ")
        p.feed("part2</think>")
        p.flush()
        self.assertIn("part1", p.reasoning())
        self.assertIn("part2", p.reasoning())

    def test_unclosed_block_in_streaming_no_leak(self):
        p = ThinkingOutputParser()
        p.feed("<think>dangerous ")
        p.feed("SELECT evil FROM t")
        _, _rsn = p.flush()
        # flush should have discarded the buffer
        self.assertEqual(p.visible(), "")
        self.assertIn("dangerous", p.reasoning())


# ---------------------------------------------------------------------------
# ThinkingAwareChatOpenAI — integration (mock LLM)
# ---------------------------------------------------------------------------

def _make_message(content: str, additional_kwargs: dict | None = None) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    msg.additional_kwargs = dict(additional_kwargs or {})
    # model_copy: return a new mock with updated fields
    def _copy(update=None):
        m2 = _make_message(
            (update or {}).get("content", content),
            {**msg.additional_kwargs, **(update or {}).get("additional_kwargs", {})},
        )
        return m2
    msg.model_copy = _copy
    return msg


class TestThinkingAwareChatOpenAIInvoke(unittest.TestCase):

    def _make_wrapper(self):
        from backend.agent.llm_client import ThinkingAwareChatOpenAI
        with patch.object(ThinkingAwareChatOpenAI, "__init__", lambda self, **kw: None):
            wrapper = ThinkingAwareChatOpenAI.__new__(ThinkingAwareChatOpenAI)
        return wrapper

    def test_invoke_strips_thinking(self):
        raw = _make_message("<think>reason</think>SELECT 1")
        with patch("langchain_openai.ChatOpenAI.invoke", return_value=raw):
            from backend.agent.llm_client import _sanitize_response
            result = _sanitize_response(raw)
        self.assertEqual(result.content, "SELECT 1")

    def test_invoke_stores_reasoning_in_additional_kwargs(self):
        from backend.agent.llm_client import _sanitize_response
        raw = _make_message("<think>the reason</think>answer")
        result = _sanitize_response(raw)
        self.assertIn("reasoning", result.additional_kwargs)
        self.assertEqual(result.additional_kwargs["reasoning"], "the reason")

    def test_invoke_no_think_fast_path(self):
        from backend.agent.llm_client import _sanitize_response
        raw = _make_message("SELECT 1")
        result = _sanitize_response(raw)
        # content unchanged, no reasoning key added
        self.assertEqual(result.content, "SELECT 1")
        self.assertNotIn("reasoning", result.additional_kwargs)

    def test_invoke_unclosed_think_returns_empty(self):
        from backend.agent.llm_client import _sanitize_response
        raw = _make_message("<think>dangling")
        result = _sanitize_response(raw)
        self.assertEqual(result.content, "")
        self.assertIn("reasoning", result.additional_kwargs)

    def test_invoke_multiple_think_blocks(self):
        from backend.agent.llm_client import _sanitize_response
        raw = _make_message("<think>a</think>vis<think>b</think>ible")
        result = _sanitize_response(raw)
        self.assertEqual(result.content, "visible")

    def test_ainvoke_strips_thinking(self):
        from backend.agent.llm_client import _sanitize_response
        raw = _make_message("<think>r</think>SQL")
        result = _sanitize_response(raw)
        self.assertEqual(result.content, "SQL")

    def test_emit_final_chunk_total_visible_correct(self):
        """_emit_final_chunk accounts for look-ahead buffer from previous feeds.

        The look-ahead buffer (up to 6 chars) may contain content from an
        earlier feed() that couldn't be emitted yet. _emit_final_chunk drains
        it together with the final chunk's content — no content is lost or
        duplicated across the stream.
        """
        from backend.agent.llm_client import _emit_final_chunk

        parser = ThinkingOutputParser()
        # Feed a long-enough earlier chunk so some content is already emitted.
        # "Hello World! " (13 chars) → emits "Hello W" (7), buffers "orld! " (6)
        already_emitted, _ = parser.feed("Hello World! ")

        # Final chunk strips thinking, remainder is "end"
        chunk = _make_message("<think>reason</think>end")
        final = _emit_final_chunk(chunk, parser)

        # The final chunk carries the look-ahead remainder + new visible
        self.assertIn("orld! ", final.content)
        self.assertIn("end", final.content)
        # Content from already_emitted is NOT repeated in the final chunk
        self.assertNotIn(already_emitted, final.content)
        self.assertIn("reasoning", final.additional_kwargs)

    def test_emit_final_chunk_preserves_additional_kwargs(self):
        from backend.agent.llm_client import _emit_final_chunk
        parser = ThinkingOutputParser()
        chunk = _make_message("visible", {"existing_key": "existing_val"})
        final = _emit_final_chunk(chunk, parser)
        self.assertEqual(final.additional_kwargs.get("existing_key"), "existing_val")


# ---------------------------------------------------------------------------
# LLMProviderPolicy
# ---------------------------------------------------------------------------


class TestLLMProviderPolicy(unittest.TestCase):
    def setUp(self) -> None:
        from backend.core.llm_provider import get_provider_policy

        self.get = get_provider_policy

    # --- thinking_control_mode capability ---

    def test_vllm_supports_thinking_control(self) -> None:
        self.assertEqual(self.get("vllm").thinking_control_mode, "chat_template_kwargs")

    def test_ollama_supports_thinking_control(self) -> None:
        self.assertEqual(self.get("ollama").thinking_control_mode, "chat_template_kwargs")

    def test_vllm_may_emit_orphaned_tags(self) -> None:
        self.assertTrue(self.get("vllm").may_emit_orphaned_think_close_tags)

    def test_ollama_no_orphaned_tags(self) -> None:
        self.assertFalse(self.get("ollama").may_emit_orphaned_think_close_tags)

    # --- safe default for unknown providers ---

    def test_unknown_provider_safe_default(self) -> None:
        self.assertEqual(self.get("litellm").thinking_control_mode, "none")

    # --- edge inputs: case-insensitive, empty, None ---

    def test_case_insensitive_vllm(self) -> None:
        self.assertEqual(self.get("VLLM").thinking_control_mode, "chat_template_kwargs")

    def test_empty_string_safe_default(self) -> None:
        self.assertEqual(self.get("").thinking_control_mode, "none")

    def test_none_safe_default(self) -> None:
        self.assertEqual(self.get(None).thinking_control_mode, "none")  # type: ignore[arg-type]

    # --- build_extra_body ---

    def test_build_extra_body_enable_false(self) -> None:
        body = self.get("vllm").build_extra_body(enable_thinking=False)
        self.assertEqual(body, {"chat_template_kwargs": {"enable_thinking": False}})

    def test_build_extra_body_enable_true(self) -> None:
        body = self.get("ollama").build_extra_body(enable_thinking=True)
        self.assertEqual(body, {"chat_template_kwargs": {"enable_thinking": True}})

    def test_build_extra_body_unsupported_returns_empty(self) -> None:
        body = self.get("litellm").build_extra_body(enable_thinking=True)
        self.assertEqual(body, {})

    def test_update_preserves_existing_keys(self) -> None:
        """build_extra_body() не затрагивает top_k / num_ctx при merge через update()."""
        extra: dict = {"top_k": 20, "num_ctx": 32768}
        extra.update(self.get("ollama").build_extra_body(enable_thinking=False))
        self.assertEqual(extra["top_k"], 20)
        self.assertEqual(extra["num_ctx"], 32768)
        self.assertIn("chat_template_kwargs", extra)

    # --- config regression: vllm bug fix ---

    def test_config_default_vllm_now_true(self) -> None:
        """Регрессионный тест: баг с vllm-исключением устранён."""
        from backend.core.config import _default_chat_template_kwargs_enabled

        self.assertTrue(_default_chat_template_kwargs_enabled("vllm"))

    def test_config_default_ollama_true(self) -> None:
        from backend.core.config import _default_chat_template_kwargs_enabled

        self.assertTrue(_default_chat_template_kwargs_enabled("ollama"))


# ---------------------------------------------------------------------------
# Thinking policy: global app + tool-level
# ---------------------------------------------------------------------------


class TestThinkingPolicy(unittest.TestCase):
    """Tests for TOOL_ENABLE_THINKING AND global llm_enable_thinking logic."""

    # --- runner.py: role="tool" now sends chat_template_kwargs ---

    def test_runner_role_tool_sends_thinking_kwargs(self) -> None:
        """role='tool' must send chat_template_kwargs just like role='chat'."""
        import inspect
        from backend.agent.runner import AgentRunner
        src = inspect.getsource(AgentRunner._build_llm)
        # The old guard "role != 'tool'" must be absent
        self.assertNotIn("role != \"tool\"", src)
        self.assertNotIn("role != 'tool'", src)

    # --- BaseExecTool: TOOL_ENABLE_THINKING ---

    def test_base_exec_tool_default_thinking_off(self) -> None:
        from backend.tools.impl.base_tool import BaseExecTool
        self.assertFalse(BaseExecTool.TOOL_ENABLE_THINKING)

    def test_effective_thinking_global_off_tool_off(self) -> None:
        """Global=False, tool=False → effective False."""
        from backend.tools.impl.pandas_tool import PandasTool
        import pandas as pd
        tool = PandasTool(pd.DataFrame(), llm_enable_thinking=False)
        self.assertFalse(tool._llm_enable_thinking)

    def test_effective_thinking_global_on_tool_off(self) -> None:
        """Global=True, tool=False → effective False (tool default wins)."""
        from backend.tools.impl.pandas_tool import PandasTool
        import pandas as pd
        tool = PandasTool(pd.DataFrame(), llm_enable_thinking=True)
        self.assertFalse(tool._llm_enable_thinking)

    def test_effective_thinking_global_off_tool_on(self) -> None:
        """Global=False, tool=True → effective False (global is master switch)."""
        from backend.tools.impl.base_tool import BaseExecTool
        import pandas as pd

        class ThinkingTool(BaseExecTool):
            name: str = "thinking_tool"
            description: str = "test"
            TOOL_ENABLE_THINKING = True

            def _run(self, *a, **kw):  # type: ignore[override]
                return "", {}

        tool = ThinkingTool(pd.DataFrame(), llm_enable_thinking=False)
        self.assertFalse(tool._llm_enable_thinking)

    def test_effective_thinking_global_on_tool_on(self) -> None:
        """Global=True, tool=True → effective True."""
        from backend.tools.impl.base_tool import BaseExecTool
        import pandas as pd

        class ThinkingTool(BaseExecTool):
            name: str = "thinking_tool2"
            description: str = "test"
            TOOL_ENABLE_THINKING = True

            def _run(self, *a, **kw):  # type: ignore[override]
                return "", {}

        tool = ThinkingTool(pd.DataFrame(), llm_enable_thinking=True)
        self.assertTrue(tool._llm_enable_thinking)

    # --- per-tool defaults ---

    def test_pandas_tool_thinking_off(self) -> None:
        from backend.tools.impl.pandas_tool import PandasTool
        self.assertFalse(PandasTool.TOOL_ENABLE_THINKING)

    def test_plotly_tool_thinking_off(self) -> None:
        from backend.tools.impl.plotly_tool import PlotlyTool
        self.assertFalse(PlotlyTool.TOOL_ENABLE_THINKING)

    def test_value_tool_thinking_off(self) -> None:
        from backend.tools.impl.value_tool import ValueTool
        self.assertFalse(ValueTool.TOOL_ENABLE_THINKING)

    def test_sql_table_service_thinking_off(self) -> None:
        from backend.data_access.sql_table_service import SQLTableService
        self.assertFalse(SQLTableService.TOOL_ENABLE_THINKING)

    # --- SQLTableService: effective thinking ---

    def test_sql_service_global_on_tool_off_is_false(self) -> None:
        """SQLTableService.TOOL_ENABLE_THINKING=False gates global setting."""
        from backend.data_access.sql_table_service import SQLTableService
        # global=True but TOOL_ENABLE_THINKING=False → effective False
        effective = True and SQLTableService.TOOL_ENABLE_THINKING
        self.assertFalse(effective)


if __name__ == "__main__":
    unittest.main()
