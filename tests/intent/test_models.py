"""测试数据模型"""

from app.intent.models import (
    ClarificationState,
    IntentAction,
    IntentCategory,
    IntentResult,
    Slot,
    SlotPriority,
)


class TestIntentCategory:
    def test_enum_values(self):
        """测试所有10个门店域IntentCategory值"""
        expected_values = {
            "STORE",
            "LOSS",
            "POLICY",
            "SUPPLY",
            "SETTLEMENT",
            "PRODUCT",
            "MATERIAL",
            "COMPLAINT",
            "STAFF",
            "OTHER",
        }
        actual_values = {e.value for e in IntentCategory}
        assert actual_values == expected_values

    def test_enum_comparison(self):
        assert IntentCategory.STORE == IntentCategory.STORE
        assert IntentCategory.STORE != IntentCategory.POLICY


class TestIntentAction:
    def test_enum_values(self):
        """测试所有8个IntentAction值"""
        expected_values = {
            "QUERY",
            "APPLY",
            "MODIFY",
            "CANCEL",
            "CONSULT",
            "ADD",
            "REMOVE",
            "COMPARE",
        }
        actual_values = {e.value for e in IntentAction}
        assert actual_values == expected_values


class TestSlotPriority:
    def test_enum_values(self):
        assert SlotPriority.P0 == "P0"
        assert SlotPriority.P1 == "P1"
        assert SlotPriority.P2 == "P2"


class TestSlot:
    def test_slot_creation(self):
        slot = Slot(
            name="store_code",
            description="门店编号",
            priority=SlotPriority.P1,
            required=True,
            extractor="store_code_extractor",
        )
        assert slot.name == "store_code"
        assert slot.priority == SlotPriority.P1


class TestIntentResult:
    def test_intent_result_creation(self):
        result = IntentResult(
            primary_intent=IntentCategory.STORE,
            secondary_intent=IntentAction.QUERY,
            confidence=0.95,
            slots={"store_code": "ST1001"},
        )
        assert result.primary_intent == IntentCategory.STORE
        assert result.confidence == 0.95

    def test_model_dump(self):
        result = IntentResult(
            primary_intent=IntentCategory.STORE,
            secondary_intent=IntentAction.QUERY,
            tertiary_intent="DEVICE_STATUS",
            confidence=0.95,
            slots={"store_code": "ST1001"},
            missing_slots=["device_type"],
            needs_clarification=True,
            clarification_question="请问要查询哪台设备？",
            raw_query="查一下门店设备状态",
        )
        data = result.model_dump()
        assert data["primary_intent"] == "STORE"
        assert data["secondary_intent"] == "QUERY"
        assert data["tertiary_intent"] == "DEVICE_STATUS"
        assert data["confidence"] == 0.95
        assert data["needs_clarification"] is True


class TestClarificationState:
    def test_initial_state(self):
        state = ClarificationState(session_id="test_session")
        assert state.session_id == "test_session"
        assert state.clarification_round == 0
        assert state.can_continue_clarification() is True

    def test_increment_round(self):
        state = ClarificationState(session_id="test_session")
        state.increment_round()
        assert state.clarification_round == 1
        assert state.can_continue_clarification() is True

    def test_max_rounds_reached(self):
        state = ClarificationState(
            session_id="test_session",
            max_clarification_rounds=2,
            clarification_round=2,
        )
        assert state.can_continue_clarification() is False
