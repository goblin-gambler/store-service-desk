"""槽位验证器测试（门店域）"""

import pytest

from app.intent.models import IntentAction, IntentCategory, IntentResult
from app.intent.slot_validator import SlotValidationResult, SlotValidator


class TestSlotValidationResult:
    """测试SlotValidationResult数据类"""

    def test_default_values(self) -> None:
        """测试默认值"""
        result = SlotValidationResult(is_complete=True)
        assert result.is_complete is True
        assert result.missing_slots == []
        assert result.missing_p0_slots == []
        assert result.missing_p1_slots == []
        assert result.missing_p2_slots == []
        assert result.filled_slots == []
        assert result.suggestions == {}

    def test_custom_values(self) -> None:
        """测试自定义值"""
        result = SlotValidationResult(
            is_complete=False,
            missing_slots=["store_code"],
            missing_p0_slots=[],
            missing_p1_slots=["store_code"],
            missing_p2_slots=["device_code"],
            filled_slots=["device_type"],
            suggestions={"device_type": ["POS 收银机", "制冰机"]},
        )
        assert result.is_complete is False
        assert result.missing_slots == ["store_code"]
        assert result.missing_p0_slots == []
        assert result.missing_p1_slots == ["store_code"]
        assert result.missing_p2_slots == ["device_code"]
        assert result.filled_slots == ["device_type"]
        assert result.suggestions == {"device_type": ["POS 收银机", "制冰机"]}

    def test_repr_complete(self) -> None:
        """测试__repr__方法 - 完整状态"""
        result = SlotValidationResult(is_complete=True)
        assert "SlotValidationResult" in repr(result)
        assert "is_complete=True" in repr(result)

    def test_repr_incomplete(self) -> None:
        """测试__repr__方法 - 不完整状态"""
        result = SlotValidationResult(
            is_complete=False,
            missing_slots=["store_code", "device_type"],
        )
        repr_str = repr(result)
        assert "is_complete=False" in repr_str
        assert "SlotValidationResult" in repr_str


class TestSlotValidator:
    """测试SlotValidator类"""

    @pytest.fixture
    def validator(self) -> SlotValidator:
        """创建验证器实例"""
        return SlotValidator()

    def test_init(self, validator: SlotValidator) -> None:
        """测试初始化"""
        assert validator is not None
        assert isinstance(validator.SLOT_SUGGESTIONS, dict)

    def test_validate_complete_slots(self, validator: SlotValidator) -> None:
        """测试报损申请槽位完整的情况（P0 为空，P1 全部填充）"""
        result = IntentResult(
            primary_intent=IntentCategory.LOSS,
            secondary_intent=IntentAction.APPLY,
            slots={
                "store_code": "ST1001",
                "amount": 350,
                "reason_category": "破损",
            },
        )
        validation_result = validator.validate(result)
        assert validation_result.is_complete is True
        assert validation_result.missing_p0_slots == []
        assert validation_result.missing_p1_slots == []
        # P2（evidence/device_code）缺失不影响完成度，但会进入 missing_slots
        assert validation_result.missing_p2_slots == ["evidence", "device_code"]
        assert validation_result.missing_slots == ["evidence", "device_code"]
        assert validation_result.filled_slots == ["store_code", "amount", "reason_category"]

    def test_validate_missing_p0_slot(self, validator: SlotValidator) -> None:
        """测试缺失 P1 槽位（门店报修未提供设备类型与问题描述）"""
        result = IntentResult(
            primary_intent=IntentCategory.STORE,
            secondary_intent=IntentAction.APPLY,
            slots={
                "store_code": "ST1001",
                # 缺少 device_type / issue
            },
        )
        validation_result = validator.validate(result)
        # 门店域 P0 为空，缺 P1 仍视为可处理（由 Agent 追问）
        assert validation_result.is_complete is True
        assert validation_result.missing_p0_slots == []
        assert "device_type" in validation_result.missing_p1_slots
        assert "issue" in validation_result.missing_p1_slots
        assert "device_type" in validation_result.missing_slots

    def test_validate_multiple_missing_p0(self, validator: SlotValidator) -> None:
        """测试缺失多个 P1 槽位"""
        result = IntentResult(
            primary_intent=IntentCategory.STORE,
            secondary_intent=IntentAction.APPLY,
            slots={},
        )
        validation_result = validator.validate(result)
        assert validation_result.is_complete is True
        assert validation_result.missing_p0_slots == []
        assert validation_result.missing_p1_slots == ["store_code", "device_type", "issue"]

    def test_validate_with_none_and_empty_values(self, validator: SlotValidator) -> None:
        """测试None和空字符串值"""
        result = IntentResult(
            primary_intent=IntentCategory.STORE,
            secondary_intent=IntentAction.APPLY,
            slots={
                "store_code": None,
                "device_type": "",
            },
        )
        validation_result = validator.validate(result)
        assert validation_result.is_complete is True
        assert validation_result.missing_p0_slots == []
        assert "store_code" in validation_result.missing_p1_slots
        assert "device_type" in validation_result.missing_p1_slots
        assert "issue" in validation_result.missing_p1_slots
        assert validation_result.filled_slots == []

    def test_validate_priority_separation(self, validator: SlotValidator) -> None:
        """测试优先级分离：P1 完整、P2 缺失"""
        result = IntentResult(
            primary_intent=IntentCategory.LOSS,
            secondary_intent=IntentAction.APPLY,
            slots={
                "store_code": "ST1001",
                "amount": 350,
                "reason_category": "过期/临期",
                # P2 槽位未填充
            },
        )
        validation_result = validator.validate(result)
        assert validation_result.is_complete is True  # P0 为空
        assert validation_result.missing_p0_slots == []
        assert validation_result.missing_p1_slots == []
        assert "evidence" in validation_result.missing_p2_slots
        assert "device_code" in validation_result.missing_p2_slots

    def test_get_next_missing_slot_p0_priority(self, validator: SlotValidator) -> None:
        """测试获取下一个缺失槽位（P0优先）"""
        validation_result = SlotValidationResult(
            is_complete=False,
            missing_p0_slots=["store_code"],
            missing_p1_slots=["device_type"],
            missing_p2_slots=["urgency"],
        )
        next_slot = validator.get_next_missing_slot(validation_result)
        assert next_slot == "store_code"

    def test_get_next_missing_slot_p1_when_p0_complete(self, validator: SlotValidator) -> None:
        """测试P0完整时获取P1槽位"""
        validation_result = SlotValidationResult(
            is_complete=True,
            missing_p0_slots=[],
            missing_p1_slots=["device_type", "issue"],
            missing_p2_slots=[],
        )
        next_slot = validator.get_next_missing_slot(validation_result)
        assert next_slot in ["device_type", "issue"]

    def test_get_next_missing_slot_no_missing(self, validator: SlotValidator) -> None:
        """测试没有缺失槽位"""
        validation_result = SlotValidationResult(
            is_complete=True,
            missing_p0_slots=[],
            missing_p1_slots=[],
            missing_p2_slots=[],
        )
        next_slot = validator.get_next_missing_slot(validation_result)
        assert next_slot is None

    def test_validate_with_suggestions(self, validator: SlotValidator) -> None:
        """测试验证时生成推荐值"""
        result = IntentResult(
            primary_intent=IntentCategory.STORE,
            secondary_intent=IntentAction.QUERY,
            slots={},
        )
        validation_result = validator.validate(result)
        assert "store_code" in validation_result.suggestions
        assert "ST1001" in validation_result.suggestions["store_code"]
        assert "device_type" in validation_result.suggestions

    def test_validate_different_intent(self, validator: SlotValidator) -> None:
        """测试不同意图组合的验证"""
        # STORE/QUERY：P0 为空，缺 device_type 属于 P1
        result = IntentResult(
            primary_intent=IntentCategory.STORE,
            secondary_intent=IntentAction.QUERY,
            slots={"store_code": "ST1001"},
        )
        validation_result = validator.validate(result)
        assert validation_result.is_complete is True
        assert "device_type" in validation_result.missing_p1_slots

        # PRODUCT/QUERY：缺 specification 属于 P1
        result = IntentResult(
            primary_intent=IntentCategory.PRODUCT,
            secondary_intent=IntentAction.QUERY,
            slots={"product_name": "燕麦拿铁"},
        )
        validation_result = validator.validate(result)
        assert validation_result.is_complete is True
        assert "specification" in validation_result.missing_p1_slots

    def test_validate_empty_config(self, validator: SlotValidator) -> None:
        """测试没有配置的情况"""
        result = IntentResult(
            primary_intent=IntentCategory.OTHER,
            secondary_intent=IntentAction.QUERY,
            slots={},
        )
        validation_result = validator.validate(result)
        assert validation_result.is_complete is True  # 没有配置，视为完整
        assert validation_result.missing_slots == []
