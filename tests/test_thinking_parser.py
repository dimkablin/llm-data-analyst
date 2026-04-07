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


if __name__ == "__main__":
    unittest.main()
