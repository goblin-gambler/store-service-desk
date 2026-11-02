from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.intent.config import SLOT_PRIORITY_CONFIG
from app.intent.models import IntentResult


class SlotValidationResult(BaseModel):
    is_complete: bool
    missing_slots: list[str] = Field(default_factory=list)
    missing_p0_slots: list[str] = Field(default_factory=list)
    missing_p1_slots: list[str] = Field(default_factory=list)
    missing_p2_slots: list[str] = Field(default_factory=list)
    filled_slots: list[str] = Field(default_factory=list)
    suggestions: dict[str, list[str]] = Field(default_factory=dict)


class SlotValidator:
    SLOT_SUGGESTIONS: dict[str, list[str]] = {
        "store_code": ["ST1001", "ST1002", "ST1003", "ST1004"],
        "amount": ["报损金额（元）"],
        "reason_category": ["破损", "过期/临期", "设备故障", "其他"],
        "device_type": ["POS 收银机", "制冰机", "标签打印机", "冷藏柜"],
        "delivery_sn": ["DL20261027001"],
        "material_name": ["纸杯", "吸管", "包装盒", "手套"],
        "policy_topic": ["排班规则", "请假制度", "报损制度", "物料额度"],
        "compare_aspect": ["价格", "规格", "配方", "保质期"],
    }

    def validate(self, result: IntentResult) -> SlotValidationResult:
        primary_key = result.primary_intent.value
        secondary_key = result.secondary_intent.value
        slots = result.slots
        config = SLOT_PRIORITY_CONFIG.get(primary_key, {}).get(secondary_key)
        if config is None:
            return SlotValidationResult(is_complete=True)

        p0_missing, p1_missing, p2_missing, filled = self._check_priority_slots(slots or {}, config)
        missing_slots = p0_missing + p1_missing + p2_missing
        suggestions = {
            slot: self.SLOT_SUGGESTIONS[slot]
            for slot in missing_slots
            if slot in self.SLOT_SUGGESTIONS
        }

        return SlotValidationResult(
            is_complete=len(p0_missing) == 0,
            missing_slots=missing_slots,
            missing_p0_slots=p0_missing,
            missing_p1_slots=p1_missing,
            missing_p2_slots=p2_missing,
            filled_slots=filled,
            suggestions=suggestions,
        )

    def get_next_missing_slot(self, result: SlotValidationResult) -> str | None:
        all_missing = result.missing_p0_slots + result.missing_p1_slots + result.missing_p2_slots
        return all_missing[0] if all_missing else None

    def _check_priority_slots(
        self, slots: dict[str, Any], config: dict[str, list[str]]
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        p0_missing = [s for s in config.get("P0", []) if self._is_empty(slots.get(s))]
        p1_missing = [s for s in config.get("P1", []) if self._is_empty(slots.get(s))]
        p2_missing = [s for s in config.get("P2", []) if self._is_empty(slots.get(s))]
        filled_slots = [
            s
            for s in (config.get("P0", []) + config.get("P1", []) + config.get("P2", []))
            if not self._is_empty(slots.get(s))
        ]
        return p0_missing, p1_missing, p2_missing, filled_slots

    def _is_empty(self, value: Any) -> bool:
        return value is None or value == ""
