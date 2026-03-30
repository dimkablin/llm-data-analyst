from __future__ import annotations

import asyncio
import unittest

from backend.api.routes import query


class _BaseStub:
    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        self.args = args
        self.kwargs = kwargs


class _LLMTextCollectorStub(_BaseStub):
    pass


class _ToolCollectorStub(_BaseStub):
    pass


class _AgentProgressCollectorStub(_BaseStub):
    pass


class _PhaseCollectorStub(_BaseStub):
    events: list[dict[str, str]]

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        super().__init__(*args, **kwargs)
        self.events = []


class _TokenStreamCallbackHandlerStub(_BaseStub):
    pass


class _PhaseTokenStreamHandlerStub(_BaseStub):
    pass


class StreamCallbackPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._originals = (
            query._LLMTextCollector,
            query._ToolCollector,
            query._AgentProgressCollector,
            query._PhaseCollector,
            query._TokenStreamCallbackHandler,
            query._PhaseTokenStreamHandler,
        )
        query._LLMTextCollector = _LLMTextCollectorStub
        query._ToolCollector = _ToolCollectorStub
        query._AgentProgressCollector = _AgentProgressCollectorStub
        query._PhaseCollector = _PhaseCollectorStub
        query._TokenStreamCallbackHandler = _TokenStreamCallbackHandlerStub
        query._PhaseTokenStreamHandler = _PhaseTokenStreamHandlerStub
        self.loop = asyncio.new_event_loop()

    def tearDown(self) -> None:
        (
            query._LLMTextCollector,
            query._ToolCollector,
            query._AgentProgressCollector,
            query._PhaseCollector,
            query._TokenStreamCallbackHandler,
            query._PhaseTokenStreamHandler,
        ) = self._originals
        self.loop.close()

    def test_build_stream_callbacks_excludes_phase_token_handler_without_reasoning(self) -> None:
        callbacks, token_collector, tool_collector, progress_collector, phase_collector = (
            query._build_stream_callbacks(
                queue=asyncio.Queue(),
                loop=self.loop,
                session_source={"source_type": "csv"},
                include_reasoning=False,
            )
        )

        self.assertIsInstance(token_collector, _TokenStreamCallbackHandlerStub)
        self.assertIsInstance(tool_collector, _ToolCollectorStub)
        self.assertIsInstance(progress_collector, _AgentProgressCollectorStub)
        self.assertIsInstance(phase_collector, _PhaseCollectorStub)
        self.assertFalse(any(isinstance(cb, _PhaseTokenStreamHandlerStub) for cb in callbacks))

    def test_build_stream_callbacks_includes_phase_token_handler_with_reasoning(self) -> None:
        callbacks, *_ = query._build_stream_callbacks(
            queue=asyncio.Queue(),
            loop=self.loop,
            session_source={"source_type": "csv"},
            include_reasoning=True,
        )

        self.assertTrue(any(isinstance(cb, _PhaseTokenStreamHandlerStub) for cb in callbacks))


if __name__ == "__main__":
    unittest.main()
