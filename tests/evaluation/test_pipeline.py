import json

import pytest
from langgraph.graph import END, START, StateGraph

from app.evaluation.metrics import (
    answer_correctness,
    intent_accuracy,
    rag_precision,
    slot_recall,
)
from app.evaluation.pipeline import EvaluationPipeline
from app.intent.service import IntentRecognitionService
from app.models.state import AgentState


def test_intent_accuracy_full_match():
    preds = ["STORE", "POLICY", "LOSS"]
    refs = ["STORE", "POLICY", "LOSS"]
    assert intent_accuracy(preds, refs) == pytest.approx(1.0)


def test_intent_accuracy_partial_match():
    preds = ["STORE", "POLICY", "LOSS"]
    refs = ["STORE", "POLICY", "OTHER"]
    assert intent_accuracy(preds, refs) == pytest.approx(2 / 3)


def test_intent_accuracy_empty():
    assert intent_accuracy([], []) == 0.0


def test_slot_recall_full():
    preds = [{"store_code": "ST1001"}, {"a": 1, "b": 2}]
    refs = [{"store_code": "ST1001"}, {"a": 1}]
    assert slot_recall(preds, refs) == pytest.approx(1.0)


def test_slot_recall_partial():
    preds = [{"store_code": "ST1001"}, {"a": 1}]
    refs = [{"store_code": "ST1001", "phone": "138"}, {"a": 1, "b": 2}]
    assert slot_recall(preds, refs) == pytest.approx(0.5)


def test_slot_recall_skips_empty_refs():
    preds = [{}, {"store_code": "ST1001"}]
    refs = [{}, {"store_code": "ST1001"}]
    assert slot_recall(preds, refs) == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_rag_precision_exact_match():
    chunks = ["报损审批政策: 350元由店长审批", "其他信息"]
    assert await rag_precision("报损审批政策", chunks) == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_rag_precision_token_match():
    chunks = ["关于门店配送时效的说明", "其他信息"]
    assert await rag_precision("配送时效", chunks) == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_rag_precision_no_match():
    chunks = ["不相关的内容", "其他信息"]
    assert await rag_precision("报损审批政策", chunks) == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_rag_precision_empty_chunks():
    assert await rag_precision("query", []) == 0.0


@pytest.mark.asyncio
async def test_answer_correctness_parses_score(deterministic_llm):
    deterministic_llm.responses = [("correctness", "0.85")]
    score = await answer_correctness("Q", "expected", "actual", deterministic_llm)
    assert score == pytest.approx(0.85)


@pytest.mark.asyncio
async def test_answer_correctness_clamps_score(deterministic_llm):
    deterministic_llm.responses = [("correctness", "1.5")]
    score = await answer_correctness("Q", "expected", "actual", deterministic_llm)
    assert score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_answer_correctness_empty_actual(deterministic_llm):
    score = await answer_correctness("Q", "expected", "", deterministic_llm)
    assert score == 0.0


def _build_success_graph():
    def _node(state: AgentState):
        question = state.get("question", "")
        return {
            "answer": "这是预期答案",
            # Include the question so the RAG heuristic can match its tokens.
            "retrieval_result": {"chunks": [f"关于{question}的政策说明"]},
        }

    workflow = StateGraph(AgentState)  # type: ignore
    workflow.add_node("policy_agent", _node)
    workflow.add_edge(START, "policy_agent")
    workflow.add_edge("policy_agent", END)
    return workflow.compile()


def _build_failing_graph():
    def _node(state: AgentState):
        raise Exception("graph failure")

    workflow = StateGraph(AgentState)  # type: ignore
    workflow.add_node("policy_agent", _node)
    workflow.add_edge(START, "policy_agent")
    workflow.add_edge("policy_agent", END)
    return workflow.compile()


async def _cleanup_intent_keys(redis_client):
    keys = []
    async for key in redis_client.scan_iter(match="intent:cache:*"):
        keys.append(key)
    async for key in redis_client.scan_iter(match="intent:session:*"):
        keys.append(key)
    if keys:
        await redis_client.delete(*keys)


@pytest.mark.asyncio
async def test_pipeline_run_full(tmp_path, deterministic_llm, redis_client):
    deterministic_llm.responses = [("correctness", "1.0"), ("consistency", "0.9")]

    dataset_path = tmp_path / "golden_store.jsonl"
    records = [
        {
            # Rule-based POLICY match, no LLM classification needed.
            "query": "请假怎么申请",
            "expected_intent": "POLICY",
            "expected_slots": {"matched_pattern": ""},
            "expected_answer_fragment": "预期",
            "expected_audit_level": "auto",
            "dimension": "leave_policy",
        },
        {
            # Rule-based STORE match via the ST#### store code pattern.
            "query": "ST1001 门店地址",
            "expected_intent": "STORE",
            "expected_slots": {"matched_pattern": ""},
            "expected_answer_fragment": "预期",
            "expected_audit_level": "auto",
            "dimension": "store_query",
        },
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8"
    )

    service = IntentRecognitionService(llm=deterministic_llm, redis_client=redis_client)
    pipeline = EvaluationPipeline(
        intent_service=service,
        llm=deterministic_llm,
        graph=_build_success_graph(),
    )

    results = await pipeline.run(str(dataset_path))

    assert results["intent_accuracy"] == pytest.approx(1.0)
    assert results["slot_recall"] == pytest.approx(1.0)
    assert results["rag_precision"] == pytest.approx(1.0)
    assert results["answer_correctness"] == pytest.approx(1.0)
    # Two records produce two agent turns, so the tone judge is actually invoked.
    assert results["tone_consistency"] == pytest.approx(0.9)
    assert results["total_records"] == 2

    await _cleanup_intent_keys(redis_client)


@pytest.mark.asyncio
async def test_pipeline_run_graph_failure_graceful(tmp_path, deterministic_llm, redis_client):
    dataset_path = tmp_path / "golden_store.jsonl"
    record = {
        # Rule-based POLICY match, so intent evaluation still succeeds.
        "query": "请假怎么申请",
        "expected_intent": "POLICY",
        # Not extracted by the rule classifier, so recall stays at 0.
        "expected_slots": {"store_code": "ST1001"},
        "expected_answer_fragment": "预期",
        "expected_audit_level": "auto",
        "dimension": "leave_policy",
    }
    dataset_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

    service = IntentRecognitionService(llm=deterministic_llm, redis_client=redis_client)
    pipeline = EvaluationPipeline(
        intent_service=service,
        llm=deterministic_llm,
        graph=_build_failing_graph(),
    )

    results = await pipeline.run(str(dataset_path))

    assert results["intent_accuracy"] == pytest.approx(1.0)
    assert results["slot_recall"] == pytest.approx(0.0)
    assert results["rag_precision"] == pytest.approx(0.0)
    assert results["answer_correctness"] == pytest.approx(0.0)
    # A failed graph yields an empty answer, and a single agent turn is below the
    # two-turn minimum, which the tone metric treats as perfectly consistent.
    assert results["tone_consistency"] == pytest.approx(1.0)
    assert results["total_records"] == 1

    await _cleanup_intent_keys(redis_client)


@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_real_llm_answer_correctness(real_llm):
    score = await answer_correctness("报损350元需要谁审批？", "店长审批", "店长审批", real_llm)
    assert 0.0 <= score <= 1.0


@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_real_llm_answer_correctness_low(real_llm):
    score = await answer_correctness("报损350元需要谁审批？", "店长审批", "区域督导审批", real_llm)
    assert 0.0 <= score <= 1.0
