import pytest
from langgraph.types import Command

from app.graph.nodes import (
    build_evaluator_node,
    build_policy_node,
    build_router_node,
    build_store_node,
    build_synthesis_node,
    decider_node,
)
from app.models.state import make_agent_state
from tests._agents import (
    DeterministicEvaluator,
    DeterministicPolicyAgent,
    DeterministicRouterAgent,
    DeterministicStoreAgent,
)


@pytest.mark.asyncio
async def test_router_node_direct_response():
    mock_agent = DeterministicRouterAgent(
        process_result={
            "response": "您好！",
            "updated_state": {"next_agent": "supervisor", "iteration_count": 1},
        }
    )

    node = build_router_node(mock_agent)
    state = make_agent_state(
        question="你好",
        thread_id="t1",
        iteration_count=0,
    )
    result = await node(state)

    assert isinstance(result, Command)
    assert result.goto == "decider_node"
    assert result.update is not None
    assert result.update["answer"] == "您好！"
    assert result.update["iteration_count"] == 1


@pytest.mark.asyncio
async def test_router_node_routes_to_supervisor():
    mock_agent = DeterministicRouterAgent(
        process_result={
            "response": "",
            "updated_state": {"next_agent": "policy_agent", "iteration_count": 1},
        }
    )

    node = build_router_node(mock_agent)
    state = make_agent_state(
        question="报损怎么申请",
        thread_id="t2",
        iteration_count=0,
    )
    result = await node(state)

    assert isinstance(result, Command)
    assert result.goto == "memory_node"
    assert result.update is not None
    assert result.update["next_agent"] == "policy_agent"
    assert result.update["iteration_count"] == 1


@pytest.mark.asyncio
async def test_router_node_missing_next_agent():
    mock_agent = DeterministicRouterAgent(
        process_result={
            "response": "",
            "updated_state": {"needs_human_transfer": True},
        }
    )

    node = build_router_node(mock_agent)
    state = make_agent_state(
        question="xxx",
        thread_id="t5",
        iteration_count=0,
    )
    result = await node(state)

    assert isinstance(result, Command)
    assert result.goto == "memory_node"
    assert result.update is not None
    assert result.update["needs_human_transfer"] is True


@pytest.mark.asyncio
async def test_policy_node():
    mock_agent = DeterministicPolicyAgent(
        process_result={
            "response": "报损金额超过200元需店长审批",
            "updated_state": {"context": ["policy chunk"]},
        }
    )

    node = build_policy_node(mock_agent)
    state = make_agent_state(question="报损", thread_id="t6")
    result = await node(state)

    assert isinstance(result, Command)
    assert result.goto == "evaluator_node"
    assert result.update is not None
    assert result.update["answer"] == "报损金额超过200元需店长审批"
    assert result.update["context"] == ["policy chunk"]


@pytest.mark.asyncio
async def test_store_node():
    mock_agent = DeterministicStoreAgent(
        process_result={
            "response": "报损单已提交，等待店长审批",
            "updated_state": {"store_data": {"status": "pending"}},
        }
    )

    node = build_store_node(mock_agent)
    state = make_agent_state(question="报损", thread_id="t7")
    result = await node(state)

    assert isinstance(result, Command)
    assert result.goto == "evaluator_node"
    assert result.update is not None
    assert result.update["answer"] == "报损单已提交，等待店长审批"
    assert result.update["store_data"]["status"] == "pending"


@pytest.mark.asyncio
async def test_policy_node_returns_updates():
    mock_agent = DeterministicPolicyAgent(
        process_result={
            "response": "请转人工",
            "updated_state": {
                "needs_human_transfer": True,
                "transfer_reason": "policy_specialist_request",
            },
        }
    )

    node = build_policy_node(mock_agent)
    state = make_agent_state(question="投诉", thread_id="t8")
    result = await node(state)

    assert result.update is not None
    assert result.update["needs_human_transfer"] is True
    assert result.update["transfer_reason"] == "policy_specialist_request"


@pytest.mark.asyncio
async def test_evaluator_node_skips_when_needs_human():
    node = build_evaluator_node(DeterministicEvaluator())
    state = make_agent_state(
        question="转人工",
        answer="请转人工",
        iteration_count=1,
        needs_human_transfer=True,
    )
    result = await node(state)

    assert isinstance(result, Command)
    assert result.goto == "decider_node"
    assert result.update == {}


@pytest.mark.asyncio
async def test_evaluator_node_low_confidence_retries():
    eval_result = {
        "confidence_score": 0.2,
        "confidence_signals": {"rag": {"score": 0.2, "reason": "缺失"}},
        "needs_human_transfer": False,
        "transfer_reason": None,
        "audit_level": "auto",
    }
    mock_eval = DeterministicEvaluator(evaluate_result=eval_result)

    node = build_evaluator_node(mock_eval)
    state = make_agent_state(
        answer="不确定",
        question="怎么报损",
        iteration_count=1,
    )
    result = await node(state)

    assert isinstance(result, Command)
    assert result.goto == "router_node"
    assert result.update is not None
    assert result.update["confidence_score"] == 0.2
    assert "needs_human_transfer" not in result.update
    assert "audit_level" not in result.update
    assert "transfer_reason" not in result.update


@pytest.mark.asyncio
async def test_evaluator_node_retry_excludes_transfer_flags():
    """即使 evaluator 返回 needs_human_transfer=True，retry 更新中也不应包含该标志"""
    eval_result = {
        "confidence_score": 0.1,
        "confidence_signals": {"rag": {"score": 0.1, "reason": "缺失"}},
        "needs_human_transfer": True,
        "transfer_reason": "置信度不足",
        "audit_level": "manual",
    }
    mock_eval = DeterministicEvaluator(evaluate_result=eval_result)

    node = build_evaluator_node(mock_eval)
    state = make_agent_state(
        answer="不确定",
        question="怎么报损",
        iteration_count=1,
    )
    result = await node(state)

    assert result.goto == "router_node"
    assert result.update is not None
    assert result.update["confidence_score"] == 0.1
    assert "needs_human_transfer" not in result.update
    assert "audit_level" not in result.update
    assert "transfer_reason" not in result.update


@pytest.mark.asyncio
async def test_evaluator_node_high_confidence_goes_to_decider():
    eval_result = {
        "confidence_score": 0.85,
        "confidence_signals": {"rag": {"score": 0.9, "reason": "匹配"}},
        "needs_human_transfer": False,
        "transfer_reason": None,
        "audit_level": "none",
    }
    mock_eval = DeterministicEvaluator(evaluate_result=eval_result)

    node = build_evaluator_node(mock_eval)
    state = make_agent_state(
        answer="报损金额超过200元需店长审批",
        question="报损怎么申请",
        iteration_count=1,
    )
    result = await node(state)

    assert isinstance(result, Command)
    assert result.goto == "decider_node"
    assert result.update is not None
    assert result.update["confidence_score"] == 0.85


@pytest.mark.asyncio
async def test_evaluator_node_boundary_confidence_retries():
    eval_result = {
        "confidence_score": 0.3,
        "confidence_signals": {},
        "needs_human_transfer": False,
        "transfer_reason": None,
        "audit_level": "auto",
    }
    mock_eval = DeterministicEvaluator(evaluate_result=eval_result)

    node = build_evaluator_node(mock_eval)
    state = make_agent_state(
        answer="一般",
        question="问题",
        iteration_count=1,
    )
    result = await node(state)

    assert result.goto == "decider_node"


def test_decider_node_normal():
    state = make_agent_state(
        question="报损",
        answer="报损金额超过200元需店长审批",
        audit_level="none",
        confidence_score=0.85,
        confidence_signals={"rag": {"score": 0.9}},
        needs_human_transfer=False,
        transfer_reason=None,
    )
    result = decider_node(state)

    assert result["answer"] == "报损金额超过200元需店长审批"
    assert result["needs_human_transfer"] is False
    assert result["audit_level"] == "none"
    assert result["confidence_score"] == 0.85
    assert result["answer"] == "报损金额超过200元需店长审批"


def test_decider_node_transfer():
    state = make_agent_state(
        question="投诉",
        answer="请转人工",
        audit_level="manual",
        confidence_score=0.0,
        confidence_signals={},
        needs_human_transfer=True,
        transfer_reason="specialist_requested_transfer",
    )
    result = decider_node(state)

    assert result["needs_human_transfer"] is True
    assert result["transfer_reason"] == "specialist_requested_transfer"
    assert result["audit_level"] == "manual"


def test_decider_node_direct_response_from_router():
    """router 直接回答（问候/澄清）confidence_score 为 None 时不应误判为 missing_evaluation"""
    state = make_agent_state(
        question="你好",
        answer="您好！有什么可以帮您？",
        needs_human_transfer=False,
    )
    result = decider_node(state)

    assert result["needs_human_transfer"] is False
    assert result["transfer_reason"] is None
    assert result["audit_level"] == "auto"
    assert result["confidence_score"] == 1.0
    assert result["answer"] == "您好！有什么可以帮您？"


@pytest.mark.asyncio
async def test_synthesis_ignores_masked_answer_from_sub_answers():
    """长回答在 sub_answers 里会被脱敏成 dict，整合后不能覆盖最终 answer。"""
    from app.context.masking import mask_observation
    from tests._llm import DeterministicChatModel

    long_answer = "报损流程说明。" * 100
    masked = mask_observation({"answer": long_answer})
    assert isinstance(masked["answer"], dict)

    llm = DeterministicChatModel(responses=[("整合员", "设备离线，报损需上传现场照片")])
    node = build_synthesis_node(llm)
    state = make_agent_state(
        question="查设备顺便说说报损流程",
        execution_mode="parallel",
        iteration_count=0,
        sub_answers=[
            {
                "agent": "store_agent",
                "response": "设备离线",
                "updated_state": {"answer": "设备离线"},
                "iteration": 0,
            },
            {
                "agent": "policy_agent",
                "response": long_answer,
                "updated_state": masked,
                "iteration": 0,
            },
        ],
    )
    result = await node(state)
    assert result.update["answer"] == "设备离线，报损需上传现场照片"
    assert result.update["synthesized_answer"] == "设备离线，报损需上传现场照片"


@pytest.mark.asyncio
async def test_synthesis_single_answer_ignores_masked_answer():
    from app.context.masking import mask_observation
    from tests._llm import DeterministicChatModel

    long_answer = "报损流程说明。" * 100
    node = build_synthesis_node(DeterministicChatModel())
    state = make_agent_state(
        question="说说报损流程",
        iteration_count=0,
        sub_answers=[
            {
                "agent": "policy_agent",
                "response": long_answer,
                "updated_state": mask_observation({"answer": long_answer}),
                "iteration": 0,
            },
        ],
    )
    result = await node(state)
    assert result.update["answer"] == long_answer


@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_synthesis_node_with_real_llm(real_llm):
    node = build_synthesis_node(real_llm)
    state = make_agent_state(
        question="查门店设备顺便问下报损流程",
        sub_answers=[
            {
                "agent": "store_agent",
                "response": "张江店目前 2 台设备离线，已为您登记报修。",
                "iteration": 0,
            },
            {
                "agent": "policy_agent",
                "response": "我们支持 7 天无理由报损，请在收到商品后 7 天内申请。",
                "iteration": 0,
            },
        ],
        iteration_count=0,
    )
    result = await node(state)

    assert isinstance(result, Command)
    assert result.goto == "evaluator_node"
    assert result.update is not None
    assert result.update["answer"]
    assert isinstance(result.update["answer"], str)
    assert len(result.update["answer"]) > 0
