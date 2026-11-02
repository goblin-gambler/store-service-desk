import json
from collections import Counter
from pathlib import Path

import pytest
from langgraph.graph import END, START, StateGraph

from app.evaluation.dataset import (
    GoldenDataset,
    GoldenRecord,
    load_golden_dataset,
    validate_dataset_dimensions,
)
from app.evaluation.pipeline import EvaluationPipeline
from app.intent.models import IntentCategory
from app.intent.service import IntentRecognitionService
from app.models.state import AgentState

GOLDEN_STORE_PATH = Path("data/golden_store_v1.jsonl")
VALID_INTENTS = {intent.value for intent in IntentCategory}


def _read_raw_golden_records(path: Path = GOLDEN_STORE_PATH) -> list[dict]:
    """Read the golden JSONL without going through the schema loader."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_golden_record_valid():
    record = GoldenRecord(
        query="昨天日结金额是多少",
        expected_intent="SETTLEMENT",
        expected_slots={"store_code": "ST1001"},
        expected_answer_fragment="日结",
        expected_audit_level="auto",
        dimension="settlement_query",
    )
    assert record.query == "昨天日结金额是多少"
    assert record.expected_intent == "SETTLEMENT"
    assert record.expected_slots == {"store_code": "ST1001"}
    assert record.expected_audit_level == "auto"


def test_golden_record_invalid_intent():
    with pytest.raises(ValueError, match="Invalid intent"):
        GoldenRecord(
            query="test",
            expected_intent="INVALID_INTENT",
            expected_slots={},
            expected_answer_fragment="",
            expected_audit_level="auto",
            dimension="test",
        )


def test_golden_record_rejects_legacy_intents():
    """The e-commerce intents (ORDER/AFTER_SALES/PAYMENT) are no longer valid."""
    for legacy_intent in ("ORDER", "AFTER_SALES", "PAYMENT"):
        with pytest.raises(ValueError, match="Invalid intent"):
            GoldenRecord(
                query="test",
                expected_intent=legacy_intent,
                expected_slots={},
                expected_answer_fragment="",
                expected_audit_level="auto",
                dimension="test",
            )


def test_golden_record_invalid_audit_level():
    with pytest.raises(ValueError, match="audit"):
        GoldenRecord(
            query="test",
            expected_intent="STORE",
            expected_slots={},
            expected_answer_fragment="",
            expected_audit_level="invalid",
            dimension="test",
        )


def test_golden_record_empty_query():
    with pytest.raises(ValueError):
        GoldenRecord(
            query="",
            expected_intent="STORE",
            expected_slots={},
            expected_answer_fragment="",
            expected_audit_level="auto",
            dimension="test",
        )


def test_load_golden_dataset_success(tmp_path):
    dataset_path = tmp_path / "test_dataset.jsonl"
    records = [
        {
            "query": "张江店今天营业吗",
            "expected_intent": "STORE",
            "expected_slots": {},
            "expected_answer_fragment": "营业",
            "expected_audit_level": "auto",
            "dimension": "store_query",
        },
        {
            "query": "报损 350 元需要谁审批",
            "expected_intent": "POLICY",
            "expected_slots": {},
            "expected_answer_fragment": "店长",
            "expected_audit_level": "medium",
            "dimension": "loss_policy",
        },
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8"
    )

    dataset = load_golden_dataset(dataset_path)
    assert dataset.total_records == 2
    assert dataset.source_path == str(dataset_path)
    assert dataset.records[0].query == "张江店今天营业吗"
    assert dataset.records[1].expected_intent == "POLICY"


def test_load_golden_dataset_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_golden_dataset("/nonexistent/path/dataset.jsonl")


def test_load_golden_dataset_invalid_json(tmp_path):
    dataset_path = tmp_path / "bad_dataset.jsonl"
    dataset_path.write_text("this is not json\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        load_golden_dataset(dataset_path)


def test_load_golden_dataset_invalid_record(tmp_path):
    dataset_path = tmp_path / "bad_dataset.jsonl"
    bad_record = {
        "query": "test",
        "expected_intent": "BAD_INTENT",
        "expected_slots": {},
        "expected_answer_fragment": "",
        "expected_audit_level": "auto",
        "dimension": "test",
    }
    dataset_path.write_text(json.dumps(bad_record, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Validation failed"):
        load_golden_dataset(dataset_path)


def test_load_golden_dataset_skips_empty_lines(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "query": "制冰机坏了，帮我报修",
                "expected_intent": "STORE",
                "expected_slots": {},
                "expected_answer_fragment": "报修",
                "expected_audit_level": "auto",
                "dimension": "repair_request",
            },
            ensure_ascii=False,
        )
        + "\n\n\n",
        encoding="utf-8",
    )

    dataset = load_golden_dataset(dataset_path)
    assert dataset.total_records == 1


def test_dimension_counts():
    records = [
        GoldenRecord(
            query="q1",
            expected_intent="STORE",
            expected_audit_level="auto",
            dimension="store_query",
        ),
        GoldenRecord(
            query="q2",
            expected_intent="STORE",
            expected_audit_level="auto",
            dimension="store_query",
        ),
        GoldenRecord(
            query="q3",
            expected_intent="POLICY",
            expected_audit_level="auto",
            dimension="loss_policy",
        ),
    ]
    dataset = GoldenDataset(records=records)
    assert dataset.dimension_counts == {"store_query": 2, "loss_policy": 1}


def test_filter_by_dimension():
    records = [
        GoldenRecord(
            query="q1",
            expected_intent="STORE",
            expected_audit_level="auto",
            dimension="store_query",
        ),
        GoldenRecord(
            query="q2",
            expected_intent="POLICY",
            expected_audit_level="auto",
            dimension="loss_policy",
        ),
    ]
    dataset = GoldenDataset(records=records)
    filtered = dataset.filter_by_dimension("store_query")
    assert len(filtered) == 1
    assert filtered[0].query == "q1"


def test_filter_by_intent():
    records = [
        GoldenRecord(
            query="q1",
            expected_intent="STORE",
            expected_audit_level="auto",
            dimension="store_query",
        ),
        GoldenRecord(
            query="q2",
            expected_intent="POLICY",
            expected_audit_level="auto",
            dimension="loss_policy",
        ),
    ]
    dataset = GoldenDataset(records=records)
    filtered = dataset.filter_by_intent("POLICY")
    assert len(filtered) == 1
    assert filtered[0].query == "q2"


def test_validate_dataset_dimensions_pass():
    """Expectations derived from the real file must satisfy the validator."""
    dataset = load_golden_dataset(GOLDEN_STORE_PATH)
    expectations = dataset.dimension_counts

    result = validate_dataset_dimensions(dataset, expected_dimensions=expectations)

    assert result["valid"] is True
    assert result["details"]["total"]["actual"] == dataset.total_records
    for dimension, count in expectations.items():
        assert result["details"][dimension]["actual"] == count
        assert result["details"][dimension]["valid"] is True


def test_validate_dataset_dimensions_custom_expectations():
    dataset = load_golden_dataset(GOLDEN_STORE_PATH)
    result = validate_dataset_dimensions(
        dataset, expected_dimensions={"store_query": dataset.total_records + 1}
    )

    assert result["valid"] is False
    assert result["details"]["store_query"]["valid"] is False
    assert result["details"]["store_query"]["actual"] == len(
        dataset.filter_by_dimension("store_query")
    )


def test_all_intents_valid():
    for intent in IntentCategory:
        record = GoldenRecord(
            query="test",
            expected_intent=intent.value,
            expected_slots={},
            expected_answer_fragment="",
            expected_audit_level="auto",
            dimension="test",
        )
        assert record.expected_intent == intent.value

    assert {intent.value for intent in IntentCategory} == VALID_INTENTS


def test_golden_store_v1_loads():
    raw_records = _read_raw_golden_records()
    dataset = load_golden_dataset(GOLDEN_STORE_PATH)

    assert dataset.total_records > 0
    assert dataset.total_records == len(raw_records)
    assert Path(dataset.source_path) == GOLDEN_STORE_PATH
    assert sum(dataset.dimension_counts.values()) == dataset.total_records
    assert len(dataset.dimension_counts) == len({r["dimension"] for r in raw_records})


def test_golden_store_v1_intent_distribution():
    raw_records = _read_raw_golden_records()
    expected_counts = Counter(record["expected_intent"] for record in raw_records)

    dataset = load_golden_dataset(GOLDEN_STORE_PATH)
    actual_counts = Counter(record.expected_intent for record in dataset.records)

    assert actual_counts == expected_counts
    assert set(actual_counts) == set(expected_counts)
    assert set(actual_counts) <= VALID_INTENTS
    assert sum(actual_counts.values()) == dataset.total_records
    for record in dataset.records:
        assert record.expected_intent in VALID_INTENTS


def test_golden_store_v1_audit_levels():
    raw_records = _read_raw_golden_records()
    expected_counts = Counter(record["expected_audit_level"] for record in raw_records)

    dataset = load_golden_dataset(GOLDEN_STORE_PATH)
    audit_counts = Counter(record.expected_audit_level for record in dataset.records)

    assert audit_counts == expected_counts
    assert set(audit_counts) <= {"auto", "medium", "manual"}
    assert sum(audit_counts.values()) == dataset.total_records
    assert audit_counts["auto"] > 0


def test_golden_store_v1_has_dimension_field():
    raw_records = _read_raw_golden_records()
    expected_dimensions = {record["dimension"] for record in raw_records}

    dataset = load_golden_dataset(GOLDEN_STORE_PATH)

    assert expected_dimensions
    for record in dataset.records:
        assert record.dimension
        assert record.dimension in expected_dimensions
    assert {record.dimension for record in dataset.records} == expected_dimensions


def _build_eval_graph():
    def _node(state: AgentState):
        return {
            "answer": "这是预期答案",
            "retrieval_result": {"chunks": ["政策A", "政策B"]},
        }

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
async def test_golden_store_pipeline_dry_run(deterministic_llm, redis_client):
    deterministic_llm.responses = [("correctness", "1.0"), ("consistency", "0.9")]
    deterministic_llm.structured = {
        "SafetyCheckResult": {
            "is_safe": True,
            "risk_level": "low",
            "risk_type": None,
            "reason": "safe",
        }
    }

    service = IntentRecognitionService(llm=deterministic_llm, redis_client=redis_client)
    pipeline = EvaluationPipeline(
        intent_service=service,
        llm=deterministic_llm,
        graph=_build_eval_graph(),
    )

    results = await pipeline.run(str(GOLDEN_STORE_PATH))

    assert results["total_records"] == len(_read_raw_golden_records())
    for metric in (
        "intent_accuracy",
        "slot_recall",
        "rag_precision",
        "answer_correctness",
        "token_efficiency",
        "tone_consistency",
    ):
        assert metric in results
        assert 0.0 <= results[metric] <= 1.0

    await _cleanup_intent_keys(redis_client)
