"""Tests for few_shot_loader module（门店域）."""

import json
from collections import Counter
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.intent.config import validate_tertiary_intent
from app.intent.few_shot_loader import (
    _cosine_similarity,
    _token_overlap_score,
    format_agent_examples_for_prompt,
    format_complaint_examples_for_prompt,
    format_intent_examples_for_prompt,
    load_agent_examples,
    load_complaint_examples,
    load_intent_examples,
    select_top_k_examples,
    select_top_k_examples_semantic,
)
from app.intent.models import IntentAction, IntentCategory

# 门店域 agent 类型（与 app.intent.few_shot_loader._AGENT_TYPES 保持一致）
AGENT_TYPES = [
    "store",
    "product",
    "material",
    "settlement",
    "supply",
    "staff",
    "policy",
    "complaint",
    "router",
]

INTENT_EXAMPLES_DIR = (
    Path(__file__).resolve().parent.parent.parent / "data" / "prompt_examples" / "intent"
)


# ========== Directory and Loading Tests ==========


class TestLoadAgentExamples:
    def test_load_intent_examples_returns_list(self):
        examples = load_intent_examples()
        assert isinstance(examples, list)
        assert examples, "intent few-shot examples should not be empty"
        for example in examples:
            assert "query" in example
            assert "primary_intent" in example
            # 标签必须落在门店域枚举内，避免旧业务标签污染模型
            primary = IntentCategory(example["primary_intent"])
            action = IntentAction(example["secondary_intent"])
            assert isinstance(primary, IntentCategory)
            assert isinstance(action, IntentAction)
            assert isinstance(example.get("slots", {}), dict)
            if example.get("tertiary_intent"):
                assert validate_tertiary_intent(
                    primary, action, example["tertiary_intent"]
                ), f"invalid tertiary intent in example: {example}"

    def test_load_intent_examples_cover_all_new_categories(self):
        examples = load_intent_examples()
        counts = Counter(example["primary_intent"] for example in examples)
        assert set(counts) == {category.value for category in IntentCategory}
        for category in IntentCategory:
            assert counts[category.value] >= 3, (
                f"Expected at least 3 examples for {category.value}, got {counts[category.value]}"
            )

    def test_load_complaint_examples_returns_list(self):
        examples = load_complaint_examples()
        assert isinstance(examples, list)
        assert examples
        first = examples[0]
        assert "query" in first
        has_category = any(
            key in first for key in ("complaint_category", "intent", "expected_intent")
        )
        assert has_category

    def test_load_agent_examples_all_types(self):
        for agent_type in AGENT_TYPES:
            examples = load_agent_examples(agent_type)
            assert isinstance(examples, list)
            if examples:
                assert "query" in examples[0]
                has_intent = any(
                    key in examples[0]
                    for key in (
                        "expected_intent",
                        "primary_intent",
                        "complaint_category",
                        "intent",
                    )
                )
                assert has_intent, f"Missing intent metadata for {agent_type}"

    def test_load_agent_examples_unknown_type_returns_empty(self):
        examples = load_agent_examples("nonexistent_agent")
        assert examples == []

    def test_load_agent_examples_case_insensitive(self):
        examples_upper = load_agent_examples("STORE")
        examples_lower = load_agent_examples("store")
        assert examples_upper == examples_lower

    def test_load_agent_examples_min_count(self):
        for agent_type in AGENT_TYPES:
            examples = load_agent_examples(agent_type)
            assert len(examples) >= 1, (
                f"Expected at least 1 example for {agent_type}, got {len(examples)}"
            )


# ========== Token Overlap Tests ==========


class TestTokenOverlapScore:
    def test_exact_match(self):
        score = _token_overlap_score("查询门店状态", "查询门店状态")
        assert score == 1.0

    def test_partial_overlap(self):
        score = _token_overlap_score("query store status", "query store detail")
        assert 0.0 < score < 1.0

    def test_chinese_partial_overlap(self):
        # 中文没有空格，必须靠字符 bigram 才能产生非零相似度
        score = _token_overlap_score("说说报损流程", "我上周提交的报损单审批到哪了")
        assert 0.0 < score < 1.0

    def test_no_overlap(self):
        score = _token_overlap_score("abcdef", "ghijkl")
        assert score == 0.0

    def test_empty_query(self):
        score = _token_overlap_score("", "查询门店")
        assert score == 0.0

    def test_empty_example(self):
        score = _token_overlap_score("查询门店", "")
        assert score == 0.0

    def test_case_insensitive(self):
        score1 = _token_overlap_score("Query Store", "query store")
        score2 = _token_overlap_score("query store", "query store")
        assert score1 == score2


# ========== Cosine Similarity Tests ==========


class TestCosineSimilarity:
    def test_identical_vectors(self):
        vec = [1.0, 2.0, 3.0]
        assert _cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        assert _cosine_similarity(vec1, vec2) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        vec1 = [1.0, 0.0]
        vec2 = [-1.0, 0.0]
        assert _cosine_similarity(vec1, vec2) == pytest.approx(-1.0)

    def test_zero_vector(self):
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 2.0, 3.0]
        assert _cosine_similarity(vec1, vec2) == 0.0

    def test_different_dimensions(self):
        vec1 = [1.0, 2.0]
        vec2 = [1.0, 2.0, 3.0]
        assert _cosine_similarity(vec1, vec2) == 0.0

    def test_empty_vectors(self):
        assert _cosine_similarity([], []) == 0.0


# ========== Select Top-K Tests ==========


class TestSelectTopKExamples:
    def test_returns_top_k(self):
        examples = [
            {"query": "查询门店状态"},
            {"query": "查询门店详情"},
            {"query": "报损申请"},
            {"query": "设备报修流程"},
        ]
        result = select_top_k_examples("查询门店状态", examples, k=2)
        assert len(result) <= 2
        assert all("查询" in ex["query"] for ex in result)

    def test_empty_examples(self):
        result = select_top_k_examples("查询门店", [], k=3)
        assert result == []

    def test_zero_score_filtered(self):
        examples = [
            {"query": "完全不同的内容"},
            {"query": "另一个不相关的话题"},
        ]
        result = select_top_k_examples("门店查询", examples, k=3)
        assert result == []

    def test_k_larger_than_examples(self):
        examples = [{"query": "查询门店"}]
        result = select_top_k_examples("查询门店", examples, k=5)
        assert len(result) == 1


# ========== Semantic Selection Tests ==========


class TestSelectTopKExamplesSemantic:
    @pytest.mark.asyncio
    async def test_fallback_to_token_overlap_on_embedding_failure(self):
        examples = [
            {"query": "查询门店状态"},
            {"query": "查询门店详情"},
            {"query": "报损申请"},
        ]
        with patch(
            "app.retrieval.embeddings.create_embedding_model",
            side_effect=RuntimeError("embedding failed"),
        ):
            result = await select_top_k_examples_semantic("查询门店状态", examples, k=2)
            assert len(result) <= 2

    @pytest.mark.asyncio
    async def test_uses_semantic_when_available(self):
        examples = [
            {"query": "查询门店状态"},
            {"query": "查询门店详情"},
            {"query": "报损申请"},
        ]
        mock_model = AsyncMock()
        mock_model.aembed_query.return_value = [1.0, 0.0, 0.0]
        mock_model.aembed_documents.return_value = [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
        ]

        with patch(
            "app.retrieval.embeddings.create_embedding_model",
            return_value=mock_model,
        ):
            result = await select_top_k_examples_semantic("查询门店状态", examples, k=2)
            assert len(result) <= 2
            mock_model.aembed_query.assert_called_once()
            mock_model.aembed_documents.assert_called_once()

    @pytest.mark.asyncio
    async def test_token_overlap_mode(self):
        examples = [
            {"query": "查询门店状态"},
            {"query": "查询门店详情"},
            {"query": "报损申请"},
        ]
        result = await select_top_k_examples_semantic(
            "查询门店状态", examples, k=2, use_semantic=False
        )
        assert len(result) <= 2
        assert all("查询" in ex["query"] for ex in result)


# ========== Format Tests ==========


class TestFormatIntentExamplesForPrompt:
    def test_empty_examples(self):
        result = format_intent_examples_for_prompt([])
        assert result == ""

    def test_formatted_output(self):
        examples = [
            {
                "query": "查门店设备状态",
                "primary_intent": "STORE",
                "secondary_intent": "QUERY",
                "tertiary_intent": None,
                "slots": {"store_code": "ST1001"},
                "reasoning": "用户想查门店设备",
            }
        ]
        result = format_intent_examples_for_prompt(examples)
        assert "查门店设备状态" in result
        assert "STORE" in result
        assert "QUERY" in result
        assert "ST1001" in result
        assert "用户想查门店设备" in result

    def test_with_tertiary_intent(self):
        examples = [
            {
                "query": "查门店设备状态",
                "primary_intent": "STORE",
                "secondary_intent": "QUERY",
                "tertiary_intent": "DEVICE_STATUS",
                "slots": {},
            }
        ]
        result = format_intent_examples_for_prompt(examples)
        assert "DEVICE_STATUS" in result


class TestFormatComplaintExamplesForPrompt:
    def test_empty_examples(self):
        result = format_complaint_examples_for_prompt([])
        assert result == ""

    def test_formatted_output(self):
        examples = [
            {
                "query": "顾客投诉喝到异物",
                "complaint_category": "FOOD_SAFETY",
                "urgency": "high",
                "expected_action": "立即建单并保留样品",
                "reasoning": "食品安全事件优先级最高",
            }
        ]
        result = format_complaint_examples_for_prompt(examples)
        assert "顾客投诉喝到异物" in result
        assert "FOOD_SAFETY" in result
        assert "high" in result
        assert "保留样品" in result


class TestFormatAgentExamplesForPrompt:
    def test_empty_examples(self):
        result = format_agent_examples_for_prompt("store", [])
        assert result == ""

    def test_store_examples(self):
        examples = [
            {
                "query": "ST1001 的设备正常吗",
                "expected_intent": "STORE",
                "expected_action": "QUERY",
                "slots": {"store_code": "ST1001"},
                "reasoning": "用户查询门店设备",
            }
        ]
        result = format_agent_examples_for_prompt("store", examples)
        assert "ST1001 的设备正常吗" in result
        assert "STORE" in result
        assert "QUERY" in result
        assert "ST1001" in result
        assert "门店与设备" in result

    def test_product_examples(self):
        examples = [
            {
                "query": "这款商品有货吗",
                "expected_intent": "PRODUCT",
                "expected_action": "QUERY",
                "reasoning": "用户询问库存",
            }
        ]
        result = format_agent_examples_for_prompt("product", examples)
        assert "商品查询" in result
        assert "这款商品有货吗" in result

    def test_unknown_agent_type(self):
        examples = [{"query": "test", "expected_intent": "TEST"}]
        result = format_agent_examples_for_prompt("unknown", examples)
        assert "test" in result
        assert "TEST" in result

    def test_complaint_category_and_urgency(self):
        examples = [
            {
                "query": "顾客投诉食品安全",
                "expected_intent": "COMPLAINT",
                "expected_action": "QUERY",
                "complaint_category": "FOOD_SAFETY",
                "urgency": "high",
                "reasoning": "食品安全投诉",
            }
        ]
        result = format_agent_examples_for_prompt("complaint", examples)
        assert "FOOD_SAFETY" in result
        assert "high" in result


# ========== Integration Tests ==========


class TestFewShotIntegration:
    def test_all_agent_example_files_exist(self):
        base_dir = Path(__file__).resolve().parent.parent.parent / "data" / "prompt_examples"
        for agent_type in AGENT_TYPES:
            agent_dir = base_dir / agent_type
            files = list(agent_dir.glob("*.jsonl"))
            assert files, f"Missing example files for agent: {agent_dir}"

    def test_all_agent_examples_valid_jsonl(self):
        base_dir = Path(__file__).resolve().parent.parent.parent / "data" / "prompt_examples"
        for agent_type in AGENT_TYPES:
            for file_path in (base_dir / agent_type).glob("*.jsonl"):
                with file_path.open("r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            assert "query" in data, f"Missing 'query' in {file_path}:{line_num}"
                            has_intent = any(
                                key in data
                                for key in (
                                    "expected_intent",
                                    "primary_intent",
                                    "complaint_category",
                                    "intent",
                                )
                            )
                            assert has_intent, f"Missing intent in {file_path}:{line_num}"
                        except json.JSONDecodeError as exc:
                            pytest.fail(f"Invalid JSON in {file_path}:{line_num}: {exc}")

    def test_example_count_per_agent(self):
        for agent_type in AGENT_TYPES:
            examples = load_agent_examples(agent_type)
            assert len(examples) >= 1, (
                f"Agent '{agent_type}' has only {len(examples)} examples, expected at least 1"
            )

    def test_intent_dir_contains_only_store_domain_examples(self):
        files = {path.name for path in INTENT_EXAMPLES_DIR.glob("*.jsonl")}
        assert files == {"general.jsonl"}, f"Unexpected legacy intent example files: {files}"
