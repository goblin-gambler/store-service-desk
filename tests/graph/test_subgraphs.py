"""Tests for agent subgraph context isolation."""

from unittest.mock import patch

import pytest

from app.agents.base import BaseAgent
from app.graph.subgraphs import _filter_state, build_agent_subgraph
from app.models.state import AgentProcessResult, AgentState, make_agent_state
from tests._llm import DeterministicChatModel


class RecordingAgent(BaseAgent):
    """Mock agent that records the state passed to process()."""

    def __init__(self):
        super().__init__(name="recording", llm=DeterministicChatModel())
        self.last_state: AgentState | None = None
        self.process_result: AgentProcessResult = {
            "response": "ok",
            "updated_state": {},
        }

    async def process(self, state: AgentState) -> AgentProcessResult:
        self.last_state = state
        return self.process_result


@pytest.mark.asyncio
async def test_subgraph_filters_state_before_agent_process():
    agent = RecordingAgent()

    allowed_keys = ["question", "user_id", "thread_id", "history"]
    subgraph = build_agent_subgraph(agent, allowed_keys=allowed_keys)

    full_state = make_agent_state(
        question="hello",
        user_id=1,
        thread_id="t1",
        history=[],
        store_data={"status": "shipped"},
        product_data={"name": "iPhone"},
    )

    await subgraph.ainvoke(full_state)

    received_state = agent.last_state
    assert received_state is not None
    assert "question" in received_state
    assert "store_data" not in received_state
    assert "product_data" not in received_state


@pytest.mark.asyncio
async def test_subgraph_overlays_branch_question_for_agent():
    """并行分支的子问题只覆盖传给 Agent 的副本，不写回父图。"""
    agent = RecordingAgent()
    node = build_agent_subgraph(agent, allowed_keys=["question", "user_id"])

    state = make_agent_state(question="查下张江店的设备情况，顺便说说报损流程", user_id=1)
    state["_branch_question"] = "说说报损流程"  # type: ignore[typeddict-unknown-key]

    updates = await node(state)

    assert agent.last_state is not None
    assert agent.last_state["question"] == "说说报损流程"
    assert "question" not in updates
    assert "_branch_question" not in updates


@pytest.mark.asyncio
async def test_subgraph_without_allowed_keys_passes_full_state():
    agent = RecordingAgent()

    subgraph = build_agent_subgraph(agent)

    full_state = make_agent_state(
        question="hello",
        user_id=1,
        thread_id="t1",
        store_data={"status": "shipped"},
    )

    await subgraph.ainvoke(full_state)

    received_state = agent.last_state
    assert received_state is not None
    assert "store_data" in received_state


@pytest.mark.asyncio
async def test_subgraph_masks_large_updated_state():
    agent = RecordingAgent()
    agent.process_result = {
        "response": "ok",
        "updated_state": {"long_text": "x" * 1000},
    }

    subgraph = build_agent_subgraph(agent, allowed_keys=["question"])
    state = make_agent_state(question="hi")

    result = await subgraph.ainvoke(state)
    sub_answers = result.get("sub_answers", [])
    assert len(sub_answers) == 1
    updated = sub_answers[0]["updated_state"]
    assert updated.get("long_text", {}).get("_masked") is True


def test_filter_state_retains_only_allowed_keys():
    state = make_agent_state(
        question="q",
        user_id=1,
        thread_id="t1",
        store_data={"status": "shipped"},
    )
    filtered = _filter_state(state, ["question", "user_id"])
    assert set(filtered.keys()) == {"question", "user_id"}


def test_estimate_state_tokens_returns_non_negative():
    from app.graph.subgraphs import _estimate_state_tokens

    state = make_agent_state(question="hello", user_id=1)
    assert _estimate_state_tokens(state) >= 0


def test_estimate_state_tokens_increases_with_content():
    from app.graph.subgraphs import _estimate_state_tokens

    small = make_agent_state(question="hi", user_id=1)
    large = make_agent_state(question="hi" * 100, user_id=1)
    assert _estimate_state_tokens(large) > _estimate_state_tokens(small)


def test_get_agent_tools_returns_scoped_tools():
    from app.graph.subgraphs import get_agent_tools

    store_tools = get_agent_tools("store_agent")
    assert len(store_tools) > 0
    assert all("name" in t and "description" in t for t in store_tools)
    names = [t["name"] for t in store_tools]
    assert "store" in names


def test_get_agent_tools_returns_empty_for_unknown_agent():
    from app.graph.subgraphs import get_agent_tools

    assert get_agent_tools("unknown_agent") == []


@pytest.mark.asyncio
async def test_subgraph_records_context_tokens():

    agent = RecordingAgent()
    allowed_keys = ["question", "user_id"]
    subgraph = build_agent_subgraph(agent, allowed_keys=allowed_keys)

    full_state = make_agent_state(
        question="hello",
        user_id=1,
        thread_id="t1",
        store_data={"status": "shipped"},
    )

    with patch("app.graph.subgraphs.record_agent_context_tokens") as mock_record:
        await subgraph.ainvoke(full_state)
        mock_record.assert_called_once()
        call_kwargs = mock_record.call_args.kwargs
        assert call_kwargs["agent_name"] == agent.name
        assert call_kwargs["tokens"] >= 0
