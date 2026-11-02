"""意图分类器测试（门店域）

测试场景:
- Function Calling成功
- Function Calling无结果时降级到规则匹配
- 规则匹配（直接调用 _classify_with_rules）
- 三级意图验证（合法和非法）
- LLM异常时降级到规则匹配
"""

import pytest
from langchain_core.exceptions import LangChainException

from app.intent.classifier import IntentClassifier
from app.intent.models import IntentAction, IntentCategory, IntentResult


@pytest.fixture
def classifier(deterministic_llm):
    """创建带Deterministic LLM的分类器"""
    return IntentClassifier(llm=deterministic_llm)


# ========== Layer 1: Function Calling Tests ==========


@pytest.mark.asyncio
async def test_function_calling_success(classifier, deterministic_llm):
    """测试Function Calling成功场景"""
    deterministic_llm.tool_calls = [
        {
            "name": "classify_intent",
            "args": {
                "primary_intent": "STORE",
                "secondary_intent": "QUERY",
                "tertiary_intent": "DEVICE_STATUS",
                "confidence": 0.95,
                "slots": {"store_code": "ST1001", "device_type": "制冰机"},
            },
        }
    ]

    result = await classifier.classify("帮我看看最近的情况")

    assert result.primary_intent == IntentCategory.STORE
    assert result.secondary_intent == IntentAction.QUERY
    assert result.tertiary_intent == "DEVICE_STATUS"


@pytest.mark.asyncio
async def test_function_calling_no_tool_calls(classifier, deterministic_llm):
    """测试Function Calling无工具调用时降级到规则匹配"""
    deterministic_llm.tool_calls = []

    result = await classifier.classify("制冰机坏了，帮我报修")

    assert result.primary_intent == IntentCategory.STORE
    assert result.secondary_intent == IntentAction.APPLY


@pytest.mark.asyncio
async def test_function_calling_exception_graceful_fallback(classifier, deterministic_llm):
    """测试Function Calling异常时降级到规则匹配"""
    deterministic_llm.exception = LangChainException("Function calling failed")

    result = await classifier.classify("排班最多可以连上几天，怎么规定的")

    assert result.primary_intent == IntentCategory.POLICY
    assert result.secondary_intent == IntentAction.CONSULT


# ========== Layer 2: Rule Matching Tests ==========


def test_rule_matching_order_query(classifier):
    """测试规则匹配 - 门店信息查询"""
    result = classifier._classify_with_rules("ST1001 的门店地址是什么")

    assert result.primary_intent == IntentCategory.STORE
    assert result.secondary_intent == IntentAction.QUERY
    assert "matched_pattern" in result.slots


def test_rule_matching_after_sales_apply(classifier):
    """测试规则匹配 - 报损申请"""
    result = classifier._classify_with_rules("鲜奶过期了，帮我报损")

    assert result.primary_intent == IntentCategory.LOSS
    assert result.secondary_intent == IntentAction.APPLY


def test_rule_matching_policy_consult(classifier):
    """测试规则匹配 - 门店制度咨询"""
    result = classifier._classify_with_rules("排班最多可以连上几天，怎么规定的")

    assert result.primary_intent == IntentCategory.POLICY
    assert result.secondary_intent == IntentAction.CONSULT


def test_rule_matching_cart_add(classifier):
    """测试规则匹配 - 物料申领"""
    result = classifier._classify_with_rules("申领 2 箱纸杯和吸管")

    assert result.primary_intent == IntentCategory.MATERIAL
    assert result.secondary_intent == IntentAction.ADD
    assert result.slots.get("action") == "ADD"


def test_rule_matching_default_other(classifier):
    """测试规则匹配 - 无匹配时默认OTHER"""
    result = classifier._classify_with_rules("xyzabc123")

    assert result.primary_intent == IntentCategory.OTHER
    assert result.secondary_intent == IntentAction.CONSULT
    assert result.confidence == 0.3


def test_rule_matching_logistics_query(classifier):
    """测试规则匹配 - 配送/补货查询"""
    result = classifier._classify_with_rules("今天的补货到哪了")

    assert result.primary_intent == IntentCategory.SUPPLY
    assert result.secondary_intent == IntentAction.QUERY


def test_rule_matching_greeting(classifier):
    """测试规则匹配 - 问候语"""
    result = classifier._classify_with_rules("你好")

    assert result.primary_intent == IntentCategory.OTHER
    assert result.secondary_intent == IntentAction.CONSULT


# ========== Tertiary Intent Validation Tests ==========


@pytest.mark.asyncio
async def test_valid_tertiary_intent(classifier, deterministic_llm):
    """测试合法的三级意图"""
    deterministic_llm.tool_calls = [
        {
            "name": "classify_intent",
            "args": {
                "primary_intent": "LOSS",
                "secondary_intent": "APPLY",
                "tertiary_intent": "LOSS_DAMAGE",
                "confidence": 0.90,
                "slots": {},
            },
        }
    ]

    result = await classifier.classify("帮我处理一笔鲜奶损耗")

    assert result.tertiary_intent == "LOSS_DAMAGE"


@pytest.mark.asyncio
async def test_invalid_tertiary_intent_filtered(classifier, deterministic_llm):
    """测试非法的三级意图被过滤"""
    deterministic_llm.tool_calls = [
        {
            "name": "classify_intent",
            "args": {
                "primary_intent": "LOSS",
                "secondary_intent": "APPLY",
                "tertiary_intent": "INVALID_TERTIARY",
                "confidence": 0.90,
                "slots": {},
            },
        }
    ]

    result = await classifier.classify("帮我处理一笔鲜奶损耗")

    assert result.tertiary_intent is None


@pytest.mark.asyncio
async def test_none_tertiary_intent_allowed(classifier, deterministic_llm):
    """测试None三级意图是合法的"""
    deterministic_llm.tool_calls = [
        {
            "name": "classify_intent",
            "args": {
                "primary_intent": "STORE",
                "secondary_intent": "QUERY",
                "tertiary_intent": None,
                "confidence": 0.85,
                "slots": {"store_code": "ST1001"},
            },
        }
    ]

    result = await classifier.classify("帮我看看最近的情况")

    assert result.tertiary_intent is None


# ========== Intent Validation Tests ==========


def test_validate_intent_result_valid(classifier):
    """测试验证有效的意图结果"""
    result = IntentResult(
        primary_intent=IntentCategory.STORE,
        secondary_intent=IntentAction.QUERY,
        tertiary_intent="DEVICE_STATUS",
        confidence=0.85,
        slots={},
        raw_query="测试",
    )

    is_valid, error_msg = classifier.validate_intent_result(result)

    assert is_valid is True
    assert error_msg == ""


def test_validate_intent_result_invalid_primary(classifier):
    """测试验证无效的一级意图"""
    result = IntentResult.model_construct(
        primary_intent="INVALID",
        secondary_intent=IntentAction.QUERY,
        confidence=0.85,
        slots={},
        raw_query="测试",
    )

    is_valid, error_msg = classifier.validate_intent_result(result)

    assert is_valid is False
    assert "无效的一级意图" in error_msg


def test_validate_intent_result_invalid_tertiary(classifier):
    """测试验证无效的三级意图"""
    result = IntentResult(
        primary_intent=IntentCategory.LOSS,
        secondary_intent=IntentAction.APPLY,
        tertiary_intent="INVALID_TERTIARY",
        confidence=0.85,
        slots={},
        raw_query="测试",
    )

    is_valid, error_msg = classifier.validate_intent_result(result)

    assert is_valid is False
    assert "无效的三级意图" in error_msg


def test_validate_intent_result_invalid_confidence(classifier):
    """测试验证无效的置信度"""
    result = IntentResult(
        primary_intent=IntentCategory.STORE,
        secondary_intent=IntentAction.QUERY,
        confidence=1.5,
        slots={},
        raw_query="测试",
    )

    is_valid, error_msg = classifier.validate_intent_result(result)

    assert is_valid is False
    assert "置信度超出范围" in error_msg


# ========== Context Tests ==========


@pytest.mark.asyncio
async def test_classification_with_context(classifier, deterministic_llm):
    """测试带上下文的分类"""
    deterministic_llm.tool_calls = [
        {
            "name": "classify_intent",
            "args": {
                "primary_intent": "STORE",
                "secondary_intent": "QUERY",
                "tertiary_intent": "DEVICE_STATUS",
                "confidence": 0.90,
                "slots": {"store_code": "ST1001"},
            },
        }
    ]

    context = {
        "session_id": "sess_123",
        "history": "用户之前查询过门店设备",
        "user_info": {"store_code": "ST1001"},
    }

    result = await classifier.classify("那个设备现在怎么样", context=context)

    assert result.primary_intent == IntentCategory.STORE


# ========== Edge Case Tests ==========


def test_empty_query(classifier):
    """测试空输入直接走规则匹配"""
    result = classifier._classify_with_rules("")

    assert result.primary_intent == IntentCategory.OTHER


@pytest.mark.asyncio
async def test_very_long_query(classifier, deterministic_llm):
    """测试超长输入"""
    deterministic_llm.tool_calls = [
        {
            "name": "classify_intent",
            "args": {
                "primary_intent": "COMPLAINT",
                "secondary_intent": "CONSULT",
                "tertiary_intent": None,
                "confidence": 0.80,
                "slots": {"incident_type": "服务态度"},
            },
        }
    ]

    long_query = "问题" + "!" * 1000

    result = await classifier.classify(long_query)

    assert result is not None
    assert result.primary_intent == IntentCategory.COMPLAINT


def test_special_characters_query(classifier):
    """测试特殊字符输入直接走规则匹配"""
    result = classifier._classify_with_rules("门店!@#$%^&*()")

    assert result is not None
    assert isinstance(result, IntentResult)


def test_multiple_keywords_match(classifier):
    """测试多个关键词匹配"""
    result = classifier._classify_with_rules("制冰机坏了，帮我报修")

    assert result.primary_intent == IntentCategory.STORE
    assert result.secondary_intent == IntentAction.APPLY
    assert result.confidence == 0.8


# ========== Invalid Function Calling Result Tests ==========


@pytest.mark.asyncio
async def test_invalid_function_result_falls_back_to_rules(classifier, deterministic_llm):
    """测试Function Calling返回无效结果时降级到规则匹配"""
    deterministic_llm.tool_calls = [
        {
            "name": "classify_intent",
            "args": {
                "primary_intent": "STORE",
                "secondary_intent": "QUERY",
                "tertiary_intent": None,
                "confidence": 1.5,
                "slots": {},
            },
        }
    ]

    result = await classifier.classify("ST1001 的设备状态")

    assert result.primary_intent == IntentCategory.STORE
    assert result.secondary_intent == IntentAction.QUERY


# ========== Regression Tests for Critical LLM/Routing Bugs ==========


@pytest.mark.asyncio
async def test_dashscope_uses_auto_tool_choice(classifier, monkeypatch):
    """Regression: DashScope/Qwen rejects object-format tool_choice in thinking mode.
    Verify that tool_choice falls back to 'auto' for DashScope endpoints."""
    monkeypatch.setattr(
        "app.intent.classifier.settings.OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/v1"
    )
    monkeypatch.setattr("app.intent.classifier.settings.LLM_MODEL", "qwen-max")

    bound_tools_calls = []

    class _FakeLLM:
        async def ainvoke(self, messages, config=None):
            return type("Resp", (), {"tool_calls": []})()

        def bind_tools(self, tools, tool_choice):
            bound_tools_calls.append(tool_choice)
            return self

    classifier._fast_llm = _FakeLLM()
    await classifier._classify_with_function_calling("测试")
    assert bound_tools_calls == ["auto"]


@pytest.mark.asyncio
async def test_non_dashscope_uses_object_tool_choice(classifier, monkeypatch):
    """Verify non-DashScope endpoints still use object-format tool_choice."""
    monkeypatch.setattr(
        "app.intent.classifier.settings.OPENAI_BASE_URL", "https://api.openai.com/v1"
    )
    monkeypatch.setattr("app.intent.classifier.settings.LLM_MODEL", "gpt-4o")

    bound_tools_calls = []

    class _FakeLLM:
        async def ainvoke(self, messages, config=None):
            return type("Resp", (), {"tool_calls": []})()

        def bind_tools(self, tools, tool_choice):
            bound_tools_calls.append(tool_choice)
            return self

    classifier._fast_llm = _FakeLLM()
    await classifier._classify_with_function_calling("测试")
    assert bound_tools_calls == [{"type": "function", "function": {"name": "classify_intent"}}]


@pytest.fixture
def real_classifier(real_llm):
    return IntentClassifier(llm=real_llm)


@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_real_llm_classify_store_query(real_classifier):
    result = await real_classifier.classify("帮我查下 ST1001 的设备状态")
    assert isinstance(result.primary_intent, IntentCategory)
    assert result.confidence >= 0.0
    assert result.confidence <= 1.0
    assert result.raw_query == "帮我查下 ST1001 的设备状态"


@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_real_llm_classify_loss_apply(real_classifier):
    result = await real_classifier.classify("鲜奶过期了，我想申请报损")
    assert isinstance(result.primary_intent, IntentCategory)
    assert result.confidence >= 0.0
    assert result.confidence <= 1.0


@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_real_llm_classify_policy_consult(real_classifier):
    result = await real_classifier.classify("门店排班制度是怎么规定的？")
    assert isinstance(result.primary_intent, IntentCategory)
    assert result.confidence >= 0.0
    assert result.confidence <= 1.0


@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_real_llm_classify_with_context(real_classifier):
    context = {
        "session_id": "sess_123",
        "history": "用户之前查询过门店设备",
        "user_info": {"store_code": "ST1001"},
    }
    result = await real_classifier.classify("那个设备现在什么样", context=context)
    assert isinstance(result.primary_intent, IntentCategory)
    assert result.confidence >= 0.0
    assert result.confidence <= 1.0
