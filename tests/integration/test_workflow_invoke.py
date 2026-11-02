from typing import Any, cast

import pytest

from app.graph.workflow import create_workflow
from app.models.state import make_agent_state
from tests._agents import DeterministicAgent, DeterministicEvaluator, DeterministicSupervisor


def _supervisor_mock(next_agent: str):
    return DeterministicSupervisor(
        process_result={
            "response": "",
            "updated_state": {
                "next_agent": next_agent,
                "execution_mode": "serial",
                "pending_agent_results": [next_agent],
            },
        }
    )


def _workflow(router, agents: dict[str, Any], evaluator, supervisor, llm):
    kwargs: dict[str, Any] = {
        "router_agent": router,
        "policy_agent": DeterministicAgent(),
        "store_agent": DeterministicAgent(),
        "supply_agent": DeterministicAgent(),
        "settlement_agent": DeterministicAgent(),
        "material_agent": DeterministicAgent(),
        "staff_agent": DeterministicAgent(),
        "evaluator": evaluator,
        "supervisor_agent": supervisor,
        "llm": llm,
    }
    kwargs.update(agents)
    return create_workflow(**kwargs)


def _eval():
    return DeterministicEvaluator(
        evaluate_result={
            "confidence_score": 0.9,
            "confidence_signals": {},
            "needs_human_transfer": False,
            "transfer_reason": None,
            "audit_level": "none",
        }
    )


@pytest.mark.asyncio
async def test_workflow_store_query(deterministic_llm, redis_checkpointer):
    initial_state = make_agent_state(
        question="帮我查下张江店的设备情况",
        thread_id="1__test_store",
    )
    mock_router = DeterministicAgent(
        process_result={"response": "", "updated_state": {"next_agent": "store_agent"}}
    )
    mock_agent = DeterministicAgent(
        name="store_agent",
        process_result={
            "response": "张江店设备状态：2 台离线",
            "updated_state": {"store_data": {"store_code": "ST1001", "devices": 2}},
        },
    )
    workflow = _workflow(
        mock_router,
        {"store_agent": mock_agent},
        _eval(),
        _supervisor_mock("store_agent"),
        deterministic_llm,
    )
    app_graph = workflow.compile(checkpointer=redis_checkpointer)
    result = await app_graph.ainvoke(
        cast(Any, initial_state),
        config={"configurable": {"thread_id": initial_state["thread_id"]}},
    )
    assert "answer" in result


@pytest.mark.asyncio
async def test_workflow_policy_query(deterministic_llm, redis_checkpointer):
    initial_state = make_agent_state(
        question="报损金额多少需要区域督导审批？",
        thread_id="1__test_policy",
    )
    mock_router = DeterministicAgent(
        process_result={"response": "", "updated_state": {"next_agent": "policy_agent"}}
    )
    mock_agent = DeterministicAgent(
        name="policy_agent",
        process_result={
            "response": "超过 2000 元需区域督导审批 [来源: OPS-LOSS-2026]",
            "updated_state": {"retrieval_result": None},
        },
    )
    workflow = _workflow(
        mock_router,
        {"policy_agent": mock_agent},
        _eval(),
        _supervisor_mock("policy_agent"),
        deterministic_llm,
    )
    app_graph = workflow.compile(checkpointer=redis_checkpointer)
    result = await app_graph.ainvoke(
        cast(Any, initial_state),
        config={"configurable": {"thread_id": initial_state["thread_id"]}},
    )
    assert "2000" in result.get("answer", "")


@pytest.mark.asyncio
async def test_workflow_supply_query(deterministic_llm, redis_checkpointer):
    initial_state = make_agent_state(
        question="这周的配送单到哪了？",
        thread_id="1__test_supply",
    )
    mock_router = DeterministicAgent(
        process_result={"response": "", "updated_state": {"next_agent": "supply_agent"}}
    )
    mock_agent = DeterministicAgent(
        name="supply_agent",
        process_result={
            "response": "配送单 PO20261027001 状态：运输中",
            "updated_state": {"supply_data": {"po": "PO20261027001", "status": "运输中"}},
        },
    )
    workflow = _workflow(
        mock_router,
        {"supply_agent": mock_agent},
        _eval(),
        _supervisor_mock("supply_agent"),
        deterministic_llm,
    )
    app_graph = workflow.compile(checkpointer=redis_checkpointer)
    result = await app_graph.ainvoke(
        cast(Any, initial_state),
        config={"configurable": {"thread_id": initial_state["thread_id"]}},
    )
    assert "配送单" in result.get("answer", "")


@pytest.mark.asyncio
async def test_workflow_settlement_query(deterministic_llm, redis_checkpointer):
    initial_state = make_agent_state(
        question="今天日结差多少钱？",
        thread_id="1__test_settlement",
    )
    mock_router = DeterministicAgent(
        process_result={"response": "", "updated_state": {"next_agent": "settlement_agent"}}
    )
    mock_agent = DeterministicAgent(
        name="settlement_agent",
        process_result={
            "response": "日结差额：-¥120.00",
            "updated_state": {"settlement_data": {"diff": -120.0}},
        },
    )
    workflow = _workflow(
        mock_router,
        {"settlement_agent": mock_agent},
        _eval(),
        _supervisor_mock("settlement_agent"),
        deterministic_llm,
    )
    app_graph = workflow.compile(checkpointer=redis_checkpointer)
    result = await app_graph.ainvoke(
        cast(Any, initial_state),
        config={"configurable": {"thread_id": initial_state["thread_id"]}},
    )
    assert "120" in result.get("answer", "")


@pytest.mark.asyncio
async def test_workflow_material_query(deterministic_llm, redis_checkpointer):
    initial_state = make_agent_state(
        question="帮我申领 200 个纸杯",
        thread_id="1__test_material",
    )
    mock_router = DeterministicAgent(
        process_result={"response": "", "updated_state": {"next_agent": "material_agent"}}
    )
    mock_agent = DeterministicAgent(
        name="material_agent",
        process_result={
            "response": "物料单已生成，合计 ¥70.00",
            "updated_state": {"material_data": {"total": 70.0}},
        },
    )
    workflow = _workflow(
        mock_router,
        {"material_agent": mock_agent},
        _eval(),
        _supervisor_mock("material_agent"),
        deterministic_llm,
    )
    app_graph = workflow.compile(checkpointer=redis_checkpointer)
    result = await app_graph.ainvoke(
        cast(Any, initial_state),
        config={"configurable": {"thread_id": initial_state["thread_id"]}},
    )
    assert "¥70.00" in result.get("answer", "")


@pytest.mark.asyncio
async def test_workflow_parallel_multi_intent(deterministic_llm, redis_checkpointer):
    initial_state = make_agent_state(
        question="查下门店设备，顺便说说报损流程",
        thread_id="1__test_parallel",
        intent_result={"primary_intent": "STORE"},
        slots={"pending_intents": [{"primary_intent": "POLICY"}]},
    )
    mock_router = DeterministicAgent(
        process_result={"response": "", "updated_state": {"next_agent": "store_agent"}}
    )
    mock_store = DeterministicAgent(
        name="store_agent",
        process_result={
            "response": "张江店 2 台设备离线",
            "updated_state": {"store_data": {"devices": 2}},
        },
    )
    mock_policy = DeterministicAgent(
        name="policy_agent",
        process_result={
            "response": "报损需上传现场照片 [来源: OPS-LOSS-2026]",
            "updated_state": {"retrieval_result": None},
        },
    )
    supervisor = DeterministicSupervisor(
        process_result={
            "response": "",
            "updated_state": {
                "next_agent": "store_agent",
                "execution_mode": "parallel",
                "pending_agent_results": ["store_agent", "policy_agent"],
            },
        }
    )
    deterministic_llm.responses = [("整合员", "张江店 2 台设备离线，报损需上传现场照片")]

    workflow = _workflow(
        mock_router,
        {"store_agent": mock_store, "policy_agent": mock_policy},
        _eval(),
        supervisor,
        deterministic_llm,
    )
    app_graph = workflow.compile(checkpointer=redis_checkpointer)
    result = await app_graph.ainvoke(
        cast(Any, initial_state),
        config={"configurable": {"thread_id": initial_state["thread_id"]}},
    )
    sub_answers = result.get("sub_answers", [])
    agents = {sa["agent"] for sa in sub_answers}
    assert "store_agent" in agents
    assert "policy_agent" in agents
    assert result.get("answer")
