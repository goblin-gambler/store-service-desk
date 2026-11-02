import pytest

from app.graph.workflow import (
    _AGENT_ALLOWED_KEYS,
    create_workflow,
    get_agent_tool_scope,
)
from tests._agents import DeterministicAgent


@pytest.mark.asyncio
async def test_workflow_compiles(redis_checkpointer):
    checkpointer = redis_checkpointer
    app_graph = create_workflow(
        router_agent=DeterministicAgent(),
        policy_agent=DeterministicAgent(),
        store_agent=DeterministicAgent(),
        supply_agent=DeterministicAgent(),
        settlement_agent=DeterministicAgent(),
        material_agent=DeterministicAgent(),
        staff_agent=DeterministicAgent(),
        evaluator=DeterministicAgent(),
        supervisor_agent=DeterministicAgent(),
        product_agent=DeterministicAgent(),
        complaint_agent=DeterministicAgent(),
        llm=DeterministicAgent(),
    ).compile(checkpointer=checkpointer)
    assert app_graph is not None


def test_agent_allowed_keys_are_granular():
    assert "store_agent" in _AGENT_ALLOWED_KEYS
    store_keys = _AGENT_ALLOWED_KEYS["store_agent"]
    assert "store_data" in store_keys
    assert "retrieval_result" in store_keys
    assert "loss_data" in store_keys
    assert "audit_level" in store_keys


def test_agent_allowed_keys_differ_by_agent():
    store_keys = set(_AGENT_ALLOWED_KEYS["store_agent"])
    product_keys = set(_AGENT_ALLOWED_KEYS["product"])
    assert "store_data" in store_keys
    assert "store_data" not in product_keys
    assert "product_data" in product_keys
    assert "product_data" not in store_keys


def test_get_agent_tool_scope_returns_tools():
    store_tools = get_agent_tool_scope("store_agent")
    assert len(store_tools) > 0
    assert "store" in store_tools
    assert "submit_loss_report" in store_tools


def test_get_agent_tool_scope_returns_empty_for_unknown():
    assert get_agent_tool_scope("unknown_agent") == []
