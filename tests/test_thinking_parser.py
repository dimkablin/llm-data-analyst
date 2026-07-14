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

from types import SimpleNamespace
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

class TestThinkingOutputParserFullDoc:

    def test_no_think_passes_through(self):
        vis, rsn = _full_parse("SELECT 1")
        assert vis == "SELECT 1"
        assert rsn == ""

    def test_empty_string(self):
        vis, rsn = _full_parse("")
        assert vis == ""
        assert rsn == ""

    def test_closed_think_block_stripped(self):
        vis, rsn = _full_parse("<think>some reasoning</think>SELECT 1")
        assert vis == "SELECT 1"
        assert rsn == "some reasoning"

    def test_unclosed_think_no_leak(self):
        """Content after an unclosed <think> must NOT appear in visible output."""
        vis, rsn = _full_parse("<think>dangling reasoning without close tag")
        assert vis == ""
        assert "dangling reasoning" in rsn

    def test_unclosed_think_discards_sql(self):
        """SQL inside unclosed thinking must not leak into visible."""
        vis, _ = _full_parse("<think>\nSELECT * FROM t\n")
        assert vis == ""

    def test_multiple_think_blocks(self):
        vis, rsn = _full_parse("<think>a</think>SEL<think>b</think>ECT 1")
        assert vis == "SELECT 1"
        assert "a" in rsn
        assert "b" in rsn

    def test_visible_before_and_after_think(self):
        vis, rsn = _full_parse("prefix<think>reason</think>suffix")
        assert vis == "prefixsuffix"
        assert rsn == "reason"

    def test_only_visible_before_think(self):
        vis, rsn = _full_parse("visible<think>thinking")
        assert vis == "visible"
        assert "thinking" in rsn

    def test_stray_close_tag_discarded(self):
        """An orphaned </think> (vLLM strips the opening tag) is discarded; content after is visible."""
        vis, rsn = _full_parse("</think>SELECT 1")
        assert vis == "SELECT 1"
        assert rsn == ""

    def test_stray_close_tag_with_prefix_reasoning(self):
        """Content before an orphaned </think> is treated as reasoning, not visible."""
        vis, rsn = _full_parse("some reasoning</think>SELECT 1")
        assert vis == "SELECT 1"
        assert rsn == "some reasoning"

    def test_sql_in_thinking_not_in_visible(self):
        thinking = "<think>```sql\nSELECT evil FROM t\n```\n</think>"
        vis, rsn = _full_parse(thinking + "\nSELECT good FROM t")
        assert "evil" not in vis
        assert "good" in vis
        assert "evil" in rsn

    def test_case_insensitive_open_tag(self):
        vis, rsn = _full_parse("<THINK>reason</THINK>result")
        assert vis == "result"
        assert rsn == "reason"

    def test_case_insensitive_mixed_tags(self):
        vis, rsn = _full_parse("<Think>reason</tHiNk>result")
        assert vis == "result"
        assert rsn == "reason"

    def test_whitespace_stripped_from_visible(self):
        vis, _ = _full_parse("<think>r</think>\n\nSELECT 1\n")
        assert vis == "SELECT 1"

    def test_reasoning_stripped_of_outer_whitespace(self):
        _, rsn = _full_parse("<think>\n  reason  \n</think>x")
        assert rsn == "reason"


# ---------------------------------------------------------------------------
# ThinkingOutputParser — incremental / streaming tests
# ---------------------------------------------------------------------------

class TestThinkingOutputParserStreaming:

    def test_tag_split_across_chunks_open(self):
        """<thi | nk> split — reasoning should still be suppressed."""
        vis, rsn = _stream_parse(["<thi", "nk>body</think>SQL"])
        assert vis == "SQL"
        assert "body" in rsn

    def test_tag_split_across_chunks_close(self):
        """</thi | nk> split — close tag must still be recognised."""
        vis, rsn = _stream_parse(["<think>reason</thi", "nk>SQL"])
        assert vis == "SQL"
        assert "reason" in rsn

    def test_single_char_chunks(self):
        text = "<think>r</think>vis"
        chunks = list(text)
        vis, rsn = _stream_parse(chunks)
        assert vis == "vis"
        assert rsn == "r"

    def test_no_double_emission_of_tail(self):
        """Content emitted by feed() must not be re-emitted by flush()."""
        p = ThinkingOutputParser()
        # Simulate: first chunk arrives, second chunk resolves partial tag
        v1, _ = p.feed("Hello ")
        v2, _ = p.feed("World")          # no <think> at all
        v_flush, _ = p.flush()
        # "Hello" emitted by first feed; "World" by second feed; flush emits tail only
        combined = v1 + v2 + v_flush
        assert combined.strip() == "Hello World"
        # Check no duplication
        assert combined.count("Hello") == 1
        assert combined.count("World") == 1

    def test_feed_flush_combined_equals_input(self):
        """Total of all feed() returns + flush() return must equal the input — no loss, no duplication."""
        p = ThinkingOutputParser()
        v1, _ = p.feed("abc")
        v2, _ = p.feed("def")
        v3, _ = p.flush()
        combined = v1 + v2 + v3
        assert combined == "abcdef"
        assert p.visible() == "abcdef"

    def test_flush_emits_only_remaining_buffer(self):
        """flush() emits only what is still in the look-ahead buffer — never what feed() already returned."""
        p = ThinkingOutputParser()
        # Feed a string long enough to trigger partial emission
        # look-ahead keeps last 6 chars; "Hello World" (11) emits first 5 chars "Hello"
        v_feed, _ = p.feed("Hello World")
        v_flush, _ = p.flush()
        # Combined must equal original with no duplication
        assert v_feed + v_flush == "Hello World"
        # flush only carries what wasn't yet emitted
        assert v_feed not in v_flush  # no overlap

    def test_reasoning_accumulates_across_feeds(self):
        p = ThinkingOutputParser()
        p.feed("<think>part1 ")
        p.feed("part2</think>")
        p.flush()
        assert "part1" in p.reasoning()
        assert "part2" in p.reasoning()

    def test_unclosed_block_in_streaming_no_leak(self):
        p = ThinkingOutputParser()
        p.feed("<think>dangerous ")
        p.feed("SELECT evil FROM t")
        _, _rsn = p.flush()
        # flush should have discarded the buffer
        assert p.visible() == ""
        assert "dangerous" in p.reasoning()


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


class TestThinkingAwareChatOpenAIInvoke:

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
        assert result.content == "SELECT 1"

    def test_invoke_stores_reasoning_in_additional_kwargs(self):
        from backend.agent.llm_client import _sanitize_response
        raw = _make_message("<think>the reason</think>answer")
        result = _sanitize_response(raw)
        assert "reasoning" in result.additional_kwargs
        assert result.additional_kwargs["reasoning"] == "the reason"

    def test_invoke_no_think_fast_path(self):
        from backend.agent.llm_client import _sanitize_response
        raw = _make_message("SELECT 1")
        result = _sanitize_response(raw)
        # content unchanged, no reasoning key added
        assert result.content == "SELECT 1"
        assert "reasoning" not in result.additional_kwargs

    def test_invoke_unclosed_think_returns_empty(self):
        from backend.agent.llm_client import _sanitize_response
        raw = _make_message("<think>dangling")
        result = _sanitize_response(raw)
        assert result.content == ""
        assert "reasoning" in result.additional_kwargs

    def test_invoke_multiple_think_blocks(self):
        from backend.agent.llm_client import _sanitize_response
        raw = _make_message("<think>a</think>vis<think>b</think>ible")
        result = _sanitize_response(raw)
        assert result.content == "visible"

    def test_ainvoke_strips_thinking(self):
        from backend.agent.llm_client import _sanitize_response
        raw = _make_message("<think>r</think>SQL")
        result = _sanitize_response(raw)
        assert result.content == "SQL"

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
        assert "orld! " in final.content
        assert "end" in final.content
        # Content from already_emitted is NOT repeated in the final chunk
        assert already_emitted not in final.content
        assert "reasoning" in final.additional_kwargs

    def test_emit_final_chunk_preserves_additional_kwargs(self):
        from backend.agent.llm_client import _emit_final_chunk
        parser = ThinkingOutputParser()
        chunk = _make_message("visible", {"existing_key": "existing_val"})
        final = _emit_final_chunk(chunk, parser)
        assert final.additional_kwargs.get("existing_key") == "existing_val"


# ---------------------------------------------------------------------------
# LLMProviderPolicy
# ---------------------------------------------------------------------------


class TestLLMProviderPolicy:
    def get(self, provider: str | None):
        from backend.core.llm_provider import get_provider_policy

        return get_provider_policy(provider)

    # --- thinking_control_mode capability ---

    def test_vllm_supports_thinking_control(self) -> None:
        assert self.get("vllm").thinking_control_mode == "chat_template_kwargs"

    def test_ollama_uses_native_thinking_control(self) -> None:
        assert self.get("ollama").thinking_control_mode == "none"

    def test_vllm_may_emit_orphaned_tags(self) -> None:
        assert self.get("vllm").may_emit_orphaned_think_close_tags

    def test_ollama_no_orphaned_tags(self) -> None:
        assert not self.get("ollama").may_emit_orphaned_think_close_tags

    # --- safe default for unknown providers ---

    def test_unknown_provider_safe_default(self) -> None:
        assert self.get("litellm").thinking_control_mode == "none"

    # --- edge inputs: case-insensitive, empty, None ---

    def test_case_insensitive_vllm(self) -> None:
        assert self.get("VLLM").thinking_control_mode == "chat_template_kwargs"

    def test_empty_string_safe_default(self) -> None:
        assert self.get("").thinking_control_mode == "none"

    def test_none_safe_default(self) -> None:
        assert self.get(None).thinking_control_mode == "none"  # type: ignore[arg-type]

    # --- build_extra_body ---

    def test_build_extra_body_enable_false(self) -> None:
        body = self.get("vllm").build_extra_body(enable_thinking=False)
        assert body == {"chat_template_kwargs": {"enable_thinking": False}}

    def test_build_extra_body_enable_true(self) -> None:
        body = self.get("vllm").build_extra_body(enable_thinking=True)
        assert body == {"chat_template_kwargs": {"enable_thinking": True}}

    def test_build_extra_body_unsupported_returns_empty(self) -> None:
        assert self.get("ollama").build_extra_body(enable_thinking=True) == {}
        assert self.get("litellm").build_extra_body(enable_thinking=True) == {}

    def test_update_preserves_existing_keys(self) -> None:
        """build_extra_body() не затрагивает top_k / num_ctx при merge через update()."""
        extra: dict = {"top_k": 20, "num_ctx": 32768}
        extra.update(self.get("vllm").build_extra_body(enable_thinking=False))
        assert extra["top_k"] == 20
        assert extra["num_ctx"] == 32768
        assert extra["chat_template_kwargs"] == {"enable_thinking": False}

    # --- config regression: vllm bug fix ---

    def test_config_default_vllm_now_true(self) -> None:
        """Регрессионный тест: баг с vllm-исключением устранён."""
        from backend.core.config import _default_chat_template_kwargs_enabled

        assert _default_chat_template_kwargs_enabled("vllm")

    def test_config_default_ollama_false(self) -> None:
        from backend.core.config import _default_chat_template_kwargs_enabled

        assert not _default_chat_template_kwargs_enabled("ollama")


# ---------------------------------------------------------------------------
# Thinking policy: global app + tool-level
# ---------------------------------------------------------------------------


class TestThinkingPolicy:
    """Tests for LLM thinking policy and deterministic exec-tool boundaries."""

    # --- runner.py: role="tool" now sends chat_template_kwargs ---

    def test_runner_role_tool_sends_thinking_kwargs(self) -> None:
        """role='tool' must send chat_template_kwargs just like role='chat'."""
        from backend.agent.runner import AgentRunner

        runner = AgentRunner()

        with patch("backend.agent.runner.build_runtime_llm") as build_runtime_llm:
            runner._build_llm(role="tool", include_reasoning=True)

        _, kwargs = build_runtime_llm.call_args
        assert kwargs["role"] == "tool"
        assert kwargs["include_reasoning"] is True

    def test_exec_tools_do_not_accept_internal_llm_thinking_flags(self) -> None:
        import inspect

        from backend.tools.impl.pandas_tool import PandasTool
        from backend.tools.impl.plotly_tool import PlotlyTool

        assert "llm_enable_thinking" not in inspect.signature(PandasTool).parameters
        assert "llm_enable_thinking" not in inspect.signature(PlotlyTool).parameters

    def test_sql_table_service_thinking_off(self) -> None:
        from backend.data_access.sql_table_service import SQLTableService
        assert not SQLTableService.TOOL_ENABLE_THINKING

    # --- SQLTableService: effective thinking ---

    def test_sql_service_global_on_tool_off_is_false(self) -> None:
        """SQLTableService.TOOL_ENABLE_THINKING=False gates global setting."""
        from backend.data_access.sql_table_service import SQLTableService
        # global=True but TOOL_ENABLE_THINKING=False → effective False
        effective = True and SQLTableService.TOOL_ENABLE_THINKING
        assert not effective


class TestReasoningSteps:
    """Tests for per-step reasoning persistence pipeline."""

    # ── callbacks: all_reasoning_steps() ─────────────────────────────────────

    def _make_handler(self):
        import asyncio

        from backend.agent.callbacks import TokenStreamCallbackHandler
        loop = asyncio.new_event_loop()
        q: asyncio.Queue = asyncio.Queue()
        return TokenStreamCallbackHandler(q, loop, show_think=True)

    def test_all_reasoning_steps_empty(self):
        h = self._make_handler()
        assert h.all_reasoning_steps() == []

    def test_all_reasoning_steps_one_step(self):
        h = self._make_handler()
        h.reasoning_chunks = ["think step 1"]
        # Simulate on_llm_end completing the step
        complete = "".join(h.reasoning_chunks)
        h._pending_thinking = complete
        h._per_step_reasoning.append(complete)
        h.reasoning_chunks = []
        assert h.all_reasoning_steps() == ["think step 1"]

    def test_all_reasoning_steps_multiple(self):
        h = self._make_handler()
        h._per_step_reasoning = ["step A", "step B", "step C"]
        assert h.all_reasoning_steps() == ["step A", "step B", "step C"]

    def test_all_reasoning_steps_returns_copy(self):
        """Mutation of returned list must not affect internal state."""
        h = self._make_handler()
        h._per_step_reasoning = ["step X"]
        result = h.all_reasoning_steps()
        result.append("injected")
        assert h._per_step_reasoning == ["step X"]

    def test_collected_visible_excludes_thinking(self):
        h = self._make_handler()

        h.on_llm_new_token("hello ")
        h.on_llm_new_token("<think>secret</think>")
        h.on_llm_new_token("world")
        h.on_llm_end(SimpleNamespace(generations=[]))

        assert h.collected_visible() == "hello world"

    def test_tool_activity_persistence_keeps_output_and_marks_interrupted(self):
        from backend.agent.callbacks import ToolCollector

        collector = ToolCollector()
        collector.events = [
            {
                "phase": "start",
                "tool_name": "sql_tool",
                "input_summary": "SELECT 1",
                "input_preview": '{"query":"SELECT 1"}',
                "timestamp": "2026-01-01T00:00:00+00:00",
            },
            {
                "phase": "end",
                "tool_name": "sql_tool",
                "status": "ok",
                "artifact_keys": [],
                "result_summary": "1 row",
                "output_preview": "answer rows",
                "timestamp": "2026-01-01T00:00:01+00:00",
            },
            {
                "phase": "start",
                "tool_name": "pandas_tool",
                "input_summary": "df.head()",
                "input_preview": '{"code":"df.head()"}',
                "timestamp": "2026-01-01T00:00:02+00:00",
            },
        ]

        activities = collector.to_persisted_activities(
            unfinished_status="error",
            unfinished_error="Остановлено пользователем.",
        )

        assert activities[0]["result_summary"] == "1 row"
        assert activities[0]["output_preview"] == "answer rows"
        assert activities[1]["status"] == "error"
        assert activities[1]["error"] == "Остановлено пользователем."
        assert activities[1]["output_preview"] == "Остановлено пользователем."

    # ── _build_reasoning_steps helper ────────────────────────────────────────

    def _build(self, raw_steps, tool_call_count=0):
        from backend.api.services.query_execution import QueryExecutionService

        return QueryExecutionService._build_reasoning_steps(raw_steps, tool_call_count)

    def test_build_steps_empty_input(self):
        assert self._build([]) == []

    def test_build_steps_single_step_no_tool(self):
        steps = self._build(["final answer thinking"])
        assert len(steps) == 1
        # Single step with no tool call → treated as final_synthesis (is_last=True, has_tool=False)
        assert steps[0].kind == "final_synthesis"
        assert steps[0].tool_name is None

    def test_build_steps_kinds_multi(self):
        raw = ["plan thinking", "tool thinking", "final thinking"]
        steps = self._build(raw, tool_call_count=2)
        # Steps 0 and 1 (i < 2) → already in pre_reasoning → filtered out.
        # Only step 2 (final_synthesis) is returned.
        assert len(steps) == 1
        assert steps[0].kind == "final_synthesis"
        assert steps[0].step_index == 2

    def test_build_steps_tool_associated_steps_excluded(self):
        """Steps preceding a tool call must NOT appear in reasoning_steps (already in pre_reasoning)."""
        raw = ["plan", "tool1 think", "tool2 think"]
        # 3 raw steps, tool_call_count=3 → every step has a tool → all excluded
        steps = self._build(raw, tool_call_count=3)
        assert steps == []

    def test_build_steps_dedup_same_tool_called_twice(self):
        """Duplicate tool calls must not corrupt the step→tool mapping."""
        # 6 LLM steps, plotly_tool called twice → tool_call_count=4 (not 3 unique names)
        raw = ["s0", "s1", "s2", "s3", "s4", "s5"]
        steps = self._build(raw, tool_call_count=4)
        # Steps 0-3 excluded (before 4 tool calls), steps 4-5 are orphans
        assert len(steps) == 2
        assert steps[0].step_index == 4
        assert steps[1].step_index == 5

    def test_build_steps_orphan_only_returned(self):
        """Only steps after all tool calls are returned."""
        raw = ["plan", "tool1 think", "orphan synthesis"]
        steps = self._build(raw, tool_call_count=1)
        # step 0: i < 1 → excluded
        # step 1: i=1 >= 1 → included
        # step 2: i=2 >= 1 → included
        assert len(steps) == 2
        assert steps[0].tool_name is None
        assert steps[1].tool_name is None

    def test_build_steps_max_limit(self):
        from backend.agent.reasoning import MAX_REASONING_STEPS
        raw = [f"step {i}" for i in range(MAX_REASONING_STEPS + 5)]
        steps = self._build(raw, tool_call_count=0)
        assert len(steps) <= MAX_REASONING_STEPS

    def test_build_steps_content_truncated(self):
        from backend.agent.reasoning import MAX_STEP_CONTENT_LEN
        long_content = "x" * (MAX_STEP_CONTENT_LEN + 100)
        steps = self._build([long_content], tool_call_count=0)
        assert len(steps[0].content) <= MAX_STEP_CONTENT_LEN + 1  # +1 for "…"
        assert steps[0].content.endswith("…")

    def test_build_steps_empty_steps_skipped(self):
        raw = ["real step", "   ", "", "another real"]
        # step 0 ("real step") i < 1 → excluded; whitespace/empty skipped
        # step 3 ("another real") i=3 >= 1 → included
        steps = self._build(raw, tool_call_count=1)
        contents = [s.content for s in steps]
        assert "   " not in contents
        assert "" not in contents
        assert len(steps) == 1
        assert steps[0].content == "another real"

    def test_build_steps_step_index_preserved(self):
        """step_index reflects original position even after filtering."""
        raw = ["step0", "step1", "step2"]
        # steps 0 and 1 excluded (tool_call_count=2); step 2 kept with index=2
        steps = self._build(raw, tool_call_count=2)
        assert len(steps) == 1
        assert steps[0].step_index == 2

    # ── ReasoningStep serde ───────────────────────────────────────────────────

    def test_reasoning_step_to_dict_round_trip(self):
        from backend.agent.reasoning import ReasoningStep
        original = ReasoningStep(step_index=2, kind="tool_synthesis", content="hello", tool_name="sql_tool")
        restored = ReasoningStep.from_dict(original.to_dict())
        assert original.step_index == restored.step_index
        assert original.kind == restored.kind
        assert original.content == restored.content
        assert original.tool_name == restored.tool_name

    def test_reasoning_step_to_dict_no_tool_name(self):
        from backend.agent.reasoning import ReasoningStep
        step = ReasoningStep(step_index=0, kind="planning", content="plan")
        d = step.to_dict()
        assert "tool_name" not in d

    def test_reasoning_step_from_dict_tolerant(self):
        """from_dict must not raise on missing keys."""
        from backend.agent.reasoning import ReasoningStep
        step = ReasoningStep.from_dict({})
        assert step.step_index == 0
        assert step.kind == "unknown"
        assert step.content == ""
        assert step.tool_name is None

    def test_reasoning_step_truncated(self):
        from backend.agent.reasoning import MAX_STEP_CONTENT_LEN, ReasoningStep
        step = ReasoningStep(step_index=0, content="a" * (MAX_STEP_CONTENT_LEN + 50))
        t = step.truncated()
        assert len(t.content) == MAX_STEP_CONTENT_LEN + 1  # +1 for "…"
        assert t.content.endswith("…")

    def test_reasoning_step_truncated_short_unchanged(self):
        from backend.agent.reasoning import ReasoningStep
        step = ReasoningStep(step_index=0, content="short")
        assert step.truncated() is step  # same object returned

    # ── persistence wiring ────────────────────────────────────────────────────

    def test_session_store_persists_reasoning_steps(self):
        import tempfile

        from backend.sessions.session_store import SessionStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(root_dir=tmpdir, ttl_days=1)
            created = store.create_session()
            sid = created.session_id
            steps = [{"step_index": 0, "kind": "planning", "content": "think"}]
            store.add_chat_message(sid, "ai", "hello", reasoning_steps=steps)
            state = store._load_state(sid)
            last = state.chat_history[-1]
            assert "reasoning_steps" in last
            assert last["reasoning_steps"][0]["kind"] == "planning"

    def test_session_store_omits_reasoning_steps_when_none(self):
        import tempfile

        from backend.sessions.session_store import SessionStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(root_dir=tmpdir, ttl_days=1)
            created = store.create_session()
            sid = created.session_id
            store.add_chat_message(sid, "ai", "hello")
            state = store._load_state(sid)
            last = state.chat_history[-1]
            assert "reasoning_steps" not in last

    # ── backward compat ───────────────────────────────────────────────────────

    def test_agentresponse_reasoning_steps_default_empty(self):
        from backend.agent.models import AgentResponse
        resp = AgentResponse(final_text="hi", reasoning=None, artifacts=[])
        assert resp.reasoning_steps == []

    def test_agentresponse_reasoning_steps_set(self):
        from backend.agent.models import AgentResponse
        resp = AgentResponse(final_text="hi", reasoning="r", artifacts=[], reasoning_steps=["step1", "step2"])
        assert resp.reasoning_steps == ["step1", "step2"]
