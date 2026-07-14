from pydantic import BaseModel

from backend.agent.models import AgentResponse, AgentRuntimeEffects, QueryCacheEntry
from backend.sessions.session_memory import StructuredSessionMemory


def test_agent_response_contract_is_pydantic_model_with_runtime_effects() -> None:
    session_memory = StructuredSessionMemory(notes="remember this")

    response = AgentResponse(
        final_text="done",
        reasoning=None,
        artifacts=[],
        runtime_effects=AgentRuntimeEffects(
            user_memory_notes=("u1",),
            session_memory_notes=("s1",),
            session_memory=session_memory,
        ),
    )

    assert isinstance(response, BaseModel)
    assert response.runtime_effects.session_memory is session_memory
    assert response.tool_names == []
    assert response.model_dump()["runtime_effects"]["user_memory_notes"] == ("u1",)


def test_query_cache_entry_wraps_agent_response_contract() -> None:
    response = AgentResponse(final_text="cached", reasoning=None, artifacts=[])

    entry = QueryCacheEntry(created_at=1.0, response=response)

    assert isinstance(entry, BaseModel)
    assert entry.response is response
