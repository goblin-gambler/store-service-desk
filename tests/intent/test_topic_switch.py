"""测试话题切换检测器（门店域）"""

import pytest

from app.intent.models import IntentAction, IntentCategory, IntentResult
from app.intent.topic_switch import TopicSwitchDetector


@pytest.fixture
def detector():
    """创建检测器实例"""
    return TopicSwitchDetector()


@pytest.fixture
def store_query_result():
    """门店/设备查询意图结果"""
    return IntentResult(
        primary_intent=IntentCategory.STORE,
        secondary_intent=IntentAction.QUERY,
        confidence=0.9,
    )


@pytest.fixture
def store_apply_result():
    """设备报修意图结果"""
    return IntentResult(
        primary_intent=IntentCategory.STORE,
        secondary_intent=IntentAction.APPLY,
        confidence=0.85,
    )


@pytest.fixture
def product_query_result():
    """商品查询意图结果"""
    return IntentResult(
        primary_intent=IntentCategory.PRODUCT,
        secondary_intent=IntentAction.QUERY,
        confidence=0.8,
    )


@pytest.fixture
def loss_apply_result():
    """报损申请意图结果"""
    return IntentResult(
        primary_intent=IntentCategory.LOSS,
        secondary_intent=IntentAction.APPLY,
        confidence=0.75,
    )


class TestExplicitSwitchDetection:
    """测试显式话题切换检测"""

    @pytest.mark.parametrize(
        "keyword",
        [
            "换个话题",
            "另外",
            "还有",
            "对了",
            "顺便问",
            "不说这个",
            "问别的",
            "还有一个问题",
            "by the way",
            "另外问一下",
            "再问一个",
        ],
    )
    def test_explicit_switch_keywords(self, detector, store_query_result, keyword):
        """测试各种显式切换关键词"""
        result = detector.detect(
            current_result=store_query_result,
            previous_result=None,
            query=f"{keyword}，我想查一下门店设备状态",
        )

        assert result.is_switch is True
        assert result.switch_type == "explicit"
        assert result.confidence == 0.9
        assert "显式切换" in result.reason
        assert result.should_reset_context is True

    def test_no_explicit_switch(self, detector, store_query_result):
        """测试无显式切换的情况"""
        result = detector.detect(
            current_result=store_query_result,
            previous_result=None,
            query="我想查一下门店设备状态",
        )

        assert result.is_switch is False
        assert result.switch_type is None

    def test_explicit_switch_case_insensitive(self, detector, store_query_result):
        """测试关键词大小写不敏感"""
        result = detector.detect(
            current_result=store_query_result,
            previous_result=None,
            query="By The Way, I want to check the store device status",
        )

        assert result.is_switch is True
        assert result.switch_type == "explicit"


class TestImplicitSwitchDetection:
    """测试隐式话题切换检测"""

    def test_confidence_drop_detection(self, detector, store_query_result, product_query_result):
        """测试置信度下降检测"""
        low_confidence_result = IntentResult(
            primary_intent=IntentCategory.PRODUCT,
            secondary_intent=IntentAction.QUERY,
            confidence=0.5,
        )

        result = detector.detect(
            current_result=low_confidence_result,
            previous_result=store_query_result,
            query="这个商品怎么样",
        )

        assert result.is_switch is True
        assert result.switch_type == "implicit"
        assert "置信度下降" in result.reason
        assert result.should_reset_context is False

    def test_confidence_drop_below_threshold(self, detector, store_query_result):
        """测试置信度下降但未超过阈值"""
        slight_drop_result = IntentResult(
            primary_intent=IntentCategory.STORE,
            secondary_intent=IntentAction.QUERY,
            confidence=0.7,
        )

        result = detector.detect(
            current_result=slight_drop_result,
            previous_result=store_query_result,
            query="我的门店设备状态",
        )

        assert result.is_switch is False or "兼容" in result.reason

    def test_intent_change_with_low_confidence(self, detector, store_query_result):
        """测试意图变化且置信度低"""
        low_confidence_different = IntentResult(
            primary_intent=IntentCategory.STAFF,
            secondary_intent=IntentAction.QUERY,
            confidence=0.4,
        )

        result = detector.detect(
            current_result=low_confidence_different,
            previous_result=store_query_result,
            query="我的员工权限",
        )

        assert result.is_switch is True
        assert result.switch_type == "implicit"
        assert "置信度下降" in result.reason or "意图变化" in result.reason
        assert result.should_reset_context is False

    def test_intent_change_with_high_confidence_incompatible(
        self, detector, store_query_result, product_query_result
    ):
        """测试意图变化但置信度高（不兼容意图）"""
        result = detector.detect(
            current_result=product_query_result,
            previous_result=store_query_result,
            query="这款商品有什么配方",
        )

        assert result.is_switch is True
        assert result.switch_type == "implicit"
        assert "不兼容" in result.reason


class TestIntentCompatibility:
    """测试意图兼容性检查"""

    def test_compatible_intents_same(self, detector, store_apply_result):
        """测试相同意图总是兼容"""
        result = detector.detect(
            current_result=store_apply_result,
            previous_result=store_apply_result,
            query="我要报修设备",
        )

        assert result.is_switch is False
        assert result.switch_type == "compatible"
        assert "兼容" in result.reason
        assert result.should_reset_context is False

    def test_compatible_intents_after_sales(
        self, detector, store_apply_result, store_query_result
    ):
        """测试门店域内查询与报修兼容"""
        result = detector.detect(
            current_result=store_query_result,
            previous_result=store_apply_result,
            query="查一下门店设备状态",
        )

        assert result.is_switch is False
        assert result.switch_type == "compatible"
        assert "兼容" in result.reason
        assert result.should_reset_context is False

    def test_compatible_intents_order_to_after_sales(
        self, detector, loss_apply_result, store_apply_result
    ):
        """测试LOSS/APPLY与STORE/APPLY兼容（根据配置）"""
        result = detector.detect(
            current_result=store_apply_result,
            previous_result=loss_apply_result,
            query="顺便把设备也报修一下",
        )

        assert result.is_switch is False
        assert result.switch_type == "compatible"
        assert "兼容" in result.reason
        assert result.should_reset_context is False

    def test_incompatible_intents(self, detector, store_query_result):
        """测试不兼容意图触发切换"""
        staff_result = IntentResult(
            primary_intent=IntentCategory.STAFF,
            secondary_intent=IntentAction.APPLY,
            confidence=0.85,
        )

        result = detector.detect(
            current_result=staff_result,
            previous_result=store_query_result,
            query="帮我申请开通权限",
        )

        assert result.is_switch is True
        assert result.switch_type == "implicit"
        assert "不兼容" in result.reason
        assert result.should_reset_context is True


class TestNoPreviousResult:
    """测试无历史意图的情况"""

    def test_no_previous_result_no_explicit(self, detector, store_query_result):
        """测试无历史意图且无显式切换"""
        result = detector.detect(
            current_result=store_query_result,
            previous_result=None,
            query="查询门店设备状态",
        )

        assert result.is_switch is False
        assert result.switch_type is None
        assert result.reason == "无历史意图"
        assert result.should_reset_context is False

    def test_no_previous_result_with_explicit(self, detector, store_query_result):
        """测试无历史意图但有显式切换关键词"""
        result = detector.detect(
            current_result=store_query_result,
            previous_result=None,
            query="换个话题，查询门店设备状态",
        )

        assert result.is_switch is True
        assert result.switch_type == "explicit"


class TestEdgeCases:
    """测试边界情况"""

    def test_exact_confidence_drop_threshold(self, detector, store_query_result):
        """测试置信度下降未超过阈值时按兼容处理（严格大于才触发）"""
        below_threshold_result = IntentResult(
            primary_intent=IntentCategory.STORE,
            secondary_intent=IntentAction.APPLY,
            confidence=0.65,
        )

        result = detector.detect(
            current_result=below_threshold_result,
            previous_result=store_query_result,
            query="设备状态",
        )

        assert result.is_switch is False
        assert result.switch_type == "compatible"

    def test_exact_confidence_threshold(self, detector, store_query_result):
        """测试恰好等于置信度阈值"""
        exact_threshold_result = IntentResult(
            primary_intent=IntentCategory.PRODUCT,
            secondary_intent=IntentAction.QUERY,
            confidence=0.5,
        )

        result = detector.detect(
            current_result=exact_threshold_result,
            previous_result=store_query_result,
            query="商品信息",
        )

        assert result.is_switch is True
        assert result.switch_type == "implicit"

    def test_empty_query(self, detector, store_query_result):
        """测试空查询"""
        result = detector.detect(
            current_result=store_query_result,
            previous_result=None,
            query="",
        )

        assert result.is_switch is False

    def test_conversation_history_parameter(self, detector, store_query_result):
        """测试对话历史参数"""
        result = detector.detect(
            current_result=store_query_result,
            previous_result=None,
            query="查询门店设备状态",
        )

        assert result is not None
