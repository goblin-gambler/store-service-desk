"""多意图处理器测试（门店域）"""

from __future__ import annotations

import pytest

from app.intent.models import IntentAction, IntentCategory, IntentResult
from app.intent.multi_intent import (
    MultiIntentProcessor,
    MultiIntentResult,
    are_independent,
)


class DeterministicClassifier:
    """确定性分类器测试替身，按顺序返回预设的 IntentResult。"""

    def __init__(self, results: list[IntentResult] | IntentResult | None = None) -> None:
        if results is None:
            results = []
        elif isinstance(results, IntentResult):
            results = [results]
        self._results = results
        self._call_index = 0
        self._calls: list[tuple[str, dict | None]] = []

    async def classify(self, query: str, context: dict | None = None) -> IntentResult:
        self._calls.append((query, context))
        if self._call_index < len(self._results):
            result = self._results[self._call_index]
            self._call_index += 1
            return result
        return IntentResult(
            primary_intent=IntentCategory.OTHER,
            secondary_intent=IntentAction.CONSULT,
            confidence=0.5,
            raw_query=query,
        )


class FailingClassifier:
    """总是抛出异常的分类器测试替身。"""

    def __init__(self, exception: Exception) -> None:
        self._exception = exception

    async def classify(self, query: str, context: dict | None = None) -> IntentResult:
        raise self._exception


class PartialFailingClassifier:
    def __init__(self, success_result: IntentResult, fail_after: int = 1) -> None:
        self._success_result = success_result
        self._fail_after = fail_after
        self._call_count = 0
        self._calls: list[tuple[str, dict | None]] = []

    async def classify(self, query: str, context: dict | None = None) -> IntentResult:
        self._calls.append((query, context))
        self._call_count += 1
        if self._call_count > self._fail_after:
            raise ValueError("第二个segment失败")
        return self._success_result


class TestMultiIntentResult:
    """测试 MultiIntentResult 数据类"""

    def test_default_initialization(self) -> None:
        """测试默认初始化"""
        result = MultiIntentResult(is_multi_intent=False)
        assert result.is_multi_intent is False
        assert result.sub_intents == []
        assert result.shared_slots == {}
        assert result.execution_order == []

    def test_full_initialization(self) -> None:
        """测试完整初始化"""
        intent_result = IntentResult(
            primary_intent=IntentCategory.STORE,
            secondary_intent=IntentAction.QUERY,
            confidence=0.9,
            slots={"store_code": "ST1001"},
        )
        result = MultiIntentResult(
            is_multi_intent=True,
            sub_intents=[intent_result],
            shared_slots={"store_code": "ST1001"},
            execution_order=[0],
        )
        assert result.is_multi_intent is True
        assert len(result.sub_intents) == 1
        assert result.shared_slots == {"store_code": "ST1001"}
        assert result.execution_order == [0]


class TestMultiIntentProcessorInitialization:
    """测试 MultiIntentProcessor 初始化"""

    def test_init_with_classifier(self) -> None:
        """测试带分类器初始化"""
        classifier = DeterministicClassifier()
        processor = MultiIntentProcessor(classifier=classifier)
        assert processor.classifier is classifier
        assert processor.mode == "cascade"

    def test_init_with_single_mode(self) -> None:
        """测试single模式初始化"""
        classifier = DeterministicClassifier()
        processor = MultiIntentProcessor(classifier=classifier, mode="single")
        assert processor.mode == "single"

    def test_init_with_cascade_mode(self) -> None:
        """测试cascade模式初始化"""
        classifier = DeterministicClassifier()
        processor = MultiIntentProcessor(classifier=classifier, mode="cascade")
        assert processor.mode == "cascade"


class TestMultiIntentProcessorSplitQuery:
    """测试查询拆分功能"""

    def test_single_intent_no_separator(self) -> None:
        """测试单意图无分隔符"""
        processor = MultiIntentProcessor(classifier=DeterministicClassifier())
        segments = processor._split_query("查询门店信息")
        assert segments == ["查询门店信息"]

    def test_split_with_separator_bianshun(self) -> None:
        """测试使用'顺便'分隔"""
        processor = MultiIntentProcessor(classifier=DeterministicClassifier())
        segments = processor._split_query("查询门店信息顺便报损申请")
        assert segments == ["查询门店信息", "报损申请"]

    def test_split_with_separator_haiyou(self) -> None:
        """测试使用'还有'分隔"""
        processor = MultiIntentProcessor(classifier=DeterministicClassifier())
        segments = processor._split_query("查询门店信息还有报损申请")
        assert segments == ["查询门店信息", "报损申请"]

    def test_split_with_separator_lingwai(self) -> None:
        """测试使用'另外'分隔"""
        processor = MultiIntentProcessor(classifier=DeterministicClassifier())
        segments = processor._split_query("查询门店信息另外报损申请")
        assert segments == ["查询门店信息", "报损申请"]

    def test_split_with_separator_yiji(self) -> None:
        """测试使用'以及'分隔"""
        processor = MultiIntentProcessor(classifier=DeterministicClassifier())
        segments = processor._split_query("查询门店信息以及报损申请")
        assert segments == ["查询门店信息", "报损申请"]

    def test_split_with_separator_he(self) -> None:
        """测试'和'不再作为分隔符（避免正常单意图被误拆分）"""
        processor = MultiIntentProcessor(classifier=DeterministicClassifier())
        segments = processor._split_query("查询门店信息和报损申请")
        assert segments == ["查询门店信息和报损申请"]

    def test_split_with_punctuation_comma(self) -> None:
        """测试使用'，然后'分隔"""
        processor = MultiIntentProcessor(classifier=DeterministicClassifier())
        segments = processor._split_query("查询门店信息，然后报损申请")
        assert segments == ["查询门店信息", "报损申请"]

    def test_split_with_punctuation_period_lingwai(self) -> None:
        """测试使用'。另外'分隔"""
        processor = MultiIntentProcessor(classifier=DeterministicClassifier())
        segments = processor._split_query("查询门店信息。另外报损申请")
        assert segments == ["查询门店信息", "报损申请"]

    def test_split_with_punctuation_period_haiyou(self) -> None:
        """测试使用'。还有'分隔"""
        processor = MultiIntentProcessor(classifier=DeterministicClassifier())
        segments = processor._split_query("查询门店信息。还有报损申请")
        assert segments == ["查询门店信息", "报损申请"]

    def test_split_with_semicolon(self) -> None:
        """测试使用分号分隔"""
        processor = MultiIntentProcessor(classifier=DeterministicClassifier())
        segments = processor._split_query("查询门店信息;报损申请")
        assert segments == ["查询门店信息", "报损申请"]

    def test_split_with_fullwidth_semicolon(self) -> None:
        """测试使用全角分号分隔"""
        processor = MultiIntentProcessor(classifier=DeterministicClassifier())
        segments = processor._split_query("查询门店信息；报损申请")
        assert segments == ["查询门店信息", "报损申请"]

    def test_split_multiple_separators(self) -> None:
        """测试多个分隔符"""
        processor = MultiIntentProcessor(classifier=DeterministicClassifier())
        segments = processor._split_query("查询门店信息顺便报损申请还有查看配送进度")
        assert segments == ["查询门店信息", "报损申请", "查看配送进度"]

    def test_split_with_whitespace(self) -> None:
        """测试带空白的分隔"""
        processor = MultiIntentProcessor(classifier=DeterministicClassifier())
        segments = processor._split_query("  查询门店信息  顺便  报损申请  ")
        assert segments == ["查询门店信息", "报损申请"]


class TestMultiIntentProcessorExtractSharedSlots:
    """测试共享槽位提取功能"""

    def test_extract_shared_slots_empty(self) -> None:
        """测试空意图列表"""
        processor = MultiIntentProcessor(classifier=DeterministicClassifier())
        shared = processor._extract_shared_slots([])
        assert shared == {}

    def test_extract_shared_slots_single_intent(self) -> None:
        """测试单意图槽位提取"""
        processor = MultiIntentProcessor(classifier=DeterministicClassifier())
        intent = IntentResult(
            primary_intent=IntentCategory.STORE,
            secondary_intent=IntentAction.QUERY,
            slots={"store_code": "ST1001", "employee_no": "E1002"},
        )
        shared = processor._extract_shared_slots([intent])
        assert shared == {"store_code": "ST1001", "employee_no": "E1002"}

    def test_extract_shared_slots_multiple_intents(self) -> None:
        """测试多意图槽位合并"""
        processor = MultiIntentProcessor(classifier=DeterministicClassifier())
        intent1 = IntentResult(
            primary_intent=IntentCategory.STORE,
            secondary_intent=IntentAction.QUERY,
            slots={"store_code": "ST1001", "employee_no": "E1002"},
        )
        intent2 = IntentResult(
            primary_intent=IntentCategory.LOSS,
            secondary_intent=IntentAction.APPLY,
            slots={"reason_category": "破损", "store_code": "ST1002"},
        )
        shared = processor._extract_shared_slots([intent1, intent2])
        assert shared == {
            "store_code": "ST1001",
            "employee_no": "E1002",
            "reason_category": "破损",
        }

    def test_extract_shared_slots_none_slots(self) -> None:
        """测试槽位为None的情况"""
        processor = MultiIntentProcessor(classifier=DeterministicClassifier())
        intent1 = IntentResult(
            primary_intent=IntentCategory.STORE,
            secondary_intent=IntentAction.QUERY,
            slots={},
        )
        intent2 = IntentResult(
            primary_intent=IntentCategory.LOSS,
            secondary_intent=IntentAction.APPLY,
            slots={"reason_category": "破损"},
        )
        shared = processor._extract_shared_slots([intent1, intent2])
        assert shared == {"reason_category": "破损"}


class TestMultiIntentProcessorProcess:
    """测试主处理流程"""

    @pytest.mark.asyncio
    async def test_process_single_intent_with_classifier(self) -> None:
        """测试有分类器时单意图处理"""
        classifier = DeterministicClassifier(
            IntentResult(
                primary_intent=IntentCategory.STORE,
                secondary_intent=IntentAction.QUERY,
                confidence=0.9,
                slots={"store_code": "ST1001"},
            )
        )

        processor = MultiIntentProcessor(classifier=classifier)
        result = await processor.process("查询门店信息")

        assert result.is_multi_intent is False
        assert len(result.sub_intents) == 1
        assert result.sub_intents[0].primary_intent == IntentCategory.STORE
        assert result.shared_slots == {"store_code": "ST1001"}
        assert result.execution_order == [0]
        assert classifier._calls == [("查询门店信息", None)]

    @pytest.mark.asyncio
    async def test_process_multi_intent_with_classifier_cascade_mode(self) -> None:
        """测试有分类器时多意图处理（cascade模式）"""
        classifier = DeterministicClassifier(
            [
                IntentResult(
                    primary_intent=IntentCategory.STORE,
                    secondary_intent=IntentAction.QUERY,
                    confidence=0.9,
                    slots={"store_code": "ST1001"},
                ),
                IntentResult(
                    primary_intent=IntentCategory.LOSS,
                    secondary_intent=IntentAction.APPLY,
                    confidence=0.85,
                    slots={"reason_category": "破损"},
                ),
            ]
        )

        processor = MultiIntentProcessor(classifier=classifier, mode="cascade")
        result = await processor.process("查询门店信息顺便报损申请")

        assert result.is_multi_intent is True
        assert len(result.sub_intents) == 2
        assert result.sub_intents[0].primary_intent == IntentCategory.STORE
        assert result.sub_intents[1].primary_intent == IntentCategory.LOSS
        assert result.shared_slots == {"store_code": "ST1001", "reason_category": "破损"}
        assert result.execution_order == [0, 1]
        assert classifier._call_index == 2

    @pytest.mark.asyncio
    async def test_process_multi_intent_single_mode(self) -> None:
        """测试single模式只返回置信度最高的意图"""
        classifier = DeterministicClassifier(
            [
                IntentResult(
                    primary_intent=IntentCategory.STORE,
                    secondary_intent=IntentAction.QUERY,
                    confidence=0.7,
                    slots={"store_code": "ST1001"},
                ),
                IntentResult(
                    primary_intent=IntentCategory.LOSS,
                    secondary_intent=IntentAction.APPLY,
                    confidence=0.9,
                    slots={"reason_category": "过期/临期"},
                ),
            ]
        )

        processor = MultiIntentProcessor(classifier=classifier, mode="single")
        result = await processor.process("查询门店信息顺便报损申请")

        assert result.is_multi_intent is True
        assert len(result.sub_intents) == 1
        assert result.sub_intents[0].primary_intent == IntentCategory.LOSS
        assert result.sub_intents[0].confidence == 0.9
        assert result.execution_order == [0]

    @pytest.mark.asyncio
    async def test_process_multi_intent_max_limit(self) -> None:
        """测试最多2个意图限制"""
        classifier = DeterministicClassifier(
            [
                IntentResult(
                    primary_intent=IntentCategory.STORE,
                    secondary_intent=IntentAction.QUERY,
                    confidence=0.9,
                    slots={},
                ),
                IntentResult(
                    primary_intent=IntentCategory.LOSS,
                    secondary_intent=IntentAction.APPLY,
                    confidence=0.85,
                    slots={},
                ),
            ]
        )

        processor = MultiIntentProcessor(classifier=classifier)
        result = await processor.process("查询门店信息顺便报损申请还有查看配送进度")

        assert result.is_multi_intent is True
        assert len(result.sub_intents) == 2
        assert classifier._call_index == 2

    @pytest.mark.asyncio
    async def test_process_with_conversation_history(self) -> None:
        """测试带对话历史的处理"""
        classifier = DeterministicClassifier(
            IntentResult(
                primary_intent=IntentCategory.STORE,
                secondary_intent=IntentAction.QUERY,
                confidence=0.9,
                slots={},
            )
        )

        processor = MultiIntentProcessor(classifier=classifier)
        history = [{"role": "user", "content": "之前的对话"}]
        result = await processor.process("查询门店信息", conversation_history=history)

        assert result.is_multi_intent is False
        assert len(result.sub_intents) == 1
        assert result.sub_intents[0].primary_intent == IntentCategory.STORE
        assert classifier._calls == [("查询门店信息", {"history": history})]

    @pytest.mark.asyncio
    async def test_process_classifier_exception_single_intent(self) -> None:
        """测试单意图时分类器异常直接抛出"""
        classifier = FailingClassifier(ValueError("分类器错误"))

        processor = MultiIntentProcessor(classifier=classifier)
        with pytest.raises(ValueError, match="分类器错误"):
            await processor.process("查询门店信息")

    @pytest.mark.asyncio
    async def test_process_classifier_exception_multi_intent(self) -> None:
        """测试多意图时分类器异常直接抛出"""
        classifier = FailingClassifier(ValueError("分类器错误"))

        processor = MultiIntentProcessor(classifier=classifier)
        with pytest.raises(ValueError, match="分类器错误"):
            await processor.process("查询门店信息顺便报损申请")

    @pytest.mark.asyncio
    async def test_process_classifier_partial_exception(self) -> None:
        success_result = IntentResult(
            primary_intent=IntentCategory.STORE,
            secondary_intent=IntentAction.QUERY,
            confidence=0.9,
            slots={"store_code": "ST1001"},
        )
        classifier = PartialFailingClassifier(success_result=success_result, fail_after=1)
        processor = MultiIntentProcessor(classifier=classifier)
        with pytest.raises(ValueError, match="第二个segment失败"):
            await processor.process("查询门店信息顺便报损申请")


class TestLLMFallback:
    @pytest.mark.asyncio
    async def test_llm_fallback_when_rules_uncertain(self):
        from tests._llm import DeterministicChatModel

        llm = DeterministicChatModel(
            structured={"IndependenceCheck": {"are_independent": True, "reason": "测试"}}
        )
        classifier = DeterministicClassifier(
            [
                IntentResult(
                    primary_intent=IntentCategory.STORE,
                    secondary_intent=IntentAction.QUERY,
                    confidence=0.9,
                    slots={},
                ),
                IntentResult(
                    primary_intent=IntentCategory.COMPLAINT,
                    secondary_intent=IntentAction.QUERY,
                    confidence=0.85,
                    slots={},
                ),
            ]
        )
        processor = MultiIntentProcessor(classifier=classifier, llm=llm)
        result = await processor.process("查询门店信息顺便投诉一下")
        assert result.is_multi_intent is True
        assert result.are_independent is True

    @pytest.mark.asyncio
    async def test_llm_fallback_logs_decision_when_db_session_provided(self, db_session):
        from sqlmodel import select

        from app.models.multi_intent_log import MultiIntentDecisionLog
        from tests._llm import DeterministicChatModel

        llm = DeterministicChatModel(
            structured={"IndependenceCheck": {"are_independent": False, "reason": "投诉与门店相关"}}
        )
        classifier = DeterministicClassifier(
            [
                IntentResult(
                    primary_intent=IntentCategory.STORE,
                    secondary_intent=IntentAction.QUERY,
                    confidence=0.9,
                    slots={},
                ),
                IntentResult(
                    primary_intent=IntentCategory.COMPLAINT,
                    secondary_intent=IntentAction.QUERY,
                    confidence=0.85,
                    slots={},
                ),
            ]
        )
        processor = MultiIntentProcessor(classifier=classifier, llm=llm)
        result = await processor.process("查询门店信息顺便投诉一下", db_session=db_session)
        assert result.are_independent is False

        logs_result = await db_session.exec(select(MultiIntentDecisionLog))
        logs = logs_result.all()
        assert len(logs) == 1
        assert logs[0].intent_a == "STORE"
        assert logs[0].intent_b == "COMPLAINT"
        assert logs[0].llm_result is False
        assert logs[0].rule_based_result is False

    @pytest.mark.asyncio
    async def test_llm_fallback_graceful_on_llm_failure(self):
        from tests._llm import DeterministicChatModel

        llm = DeterministicChatModel(exception=ValueError("模型失败"))
        classifier = DeterministicClassifier(
            [
                IntentResult(
                    primary_intent=IntentCategory.STORE,
                    secondary_intent=IntentAction.QUERY,
                    confidence=0.9,
                    slots={},
                ),
                IntentResult(
                    primary_intent=IntentCategory.COMPLAINT,
                    secondary_intent=IntentAction.QUERY,
                    confidence=0.85,
                    slots={},
                ),
            ]
        )
        processor = MultiIntentProcessor(classifier=classifier, llm=llm)
        result = await processor.process("查询门店信息顺便投诉一下")
        assert result.is_multi_intent is True
        assert result.are_independent is False


class TestAreIndependent:
    def test_store_policy_independent(self):
        assert are_independent("STORE", "POLICY") is True
        assert are_independent("POLICY", "STORE") is True

    def test_loss_store_dependent(self):
        assert are_independent("LOSS", "STORE") is False
        assert are_independent("STORE", "LOSS") is False

    def test_same_intent_dependent(self):
        assert are_independent("STORE", "STORE") is False

    def test_unknown_pair_defaults_to_dependent(self):
        assert are_independent("MYSTERY", "POLICY") is False


class TestRealLLM:
    @pytest.fixture
    def real_multi_intent_processor(self, real_llm):
        from app.intent.classifier import IntentClassifier

        classifier = IntentClassifier(llm=real_llm)
        return MultiIntentProcessor(classifier=classifier)

    @pytest.mark.requires_llm
    @pytest.mark.asyncio
    async def test_real_llm_process_single_intent(self, real_multi_intent_processor):
        result = await real_multi_intent_processor.process("查询门店 ST1001 的设备状态")
        assert result.is_multi_intent is False
        assert len(result.sub_intents) == 1
        assert isinstance(result.sub_intents[0].primary_intent, IntentCategory)

    @pytest.mark.requires_llm
    @pytest.mark.asyncio
    async def test_real_llm_process_multi_intent(self, real_multi_intent_processor):
        result = await real_multi_intent_processor.process("查询门店信息顺便投诉一下")
        assert result.is_multi_intent is True
        assert len(result.sub_intents) >= 1
        assert isinstance(result.sub_intents[0].primary_intent, IntentCategory)
