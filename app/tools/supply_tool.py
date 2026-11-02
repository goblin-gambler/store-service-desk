"""配送与补货工具。

数据来自 WMS / 冷链监控的 mock 快照，按门店编号查询。
"""

from __future__ import annotations

import re
from typing import Any

from app.models.state import AgentState
from app.tools.base import BaseTool, ToolResult
from app.tools.store_data import (
    default_store_code,
    deliveries_for,
    extract_store_code,
    get_store,
)

_DELIVERY_PATTERN = re.compile(r"(DL\d{8,})", re.IGNORECASE)


class SupplyTool(BaseTool):
    name = "supply"
    description = "查询补货单、配送进度与冷链温度异常"

    async def execute(self, state: AgentState, **kwargs) -> ToolResult:
        _ = kwargs
        question = state.get("question", "")
        slots = state.get("slots") or {}
        combined = f"{question} {' '.join(str(v) for v in slots.values())}"

        delivery_sn = None
        matched = _DELIVERY_PATTERN.search(combined)
        if matched:
            delivery_sn = matched.group(1).upper()

        store_code = extract_store_code(question, slots) or default_store_code(state.get("user_id"))
        if delivery_sn:
            store_code = store_code or self._store_of_delivery(delivery_sn)
        if not store_code:
            return ToolResult(
                output={
                    "type": "error",
                    "reason": "missing_store",
                    "message": "请提供门店编号（如 ST1001）或补货单号（如 DL20261027001）。",
                }
            )

        store = get_store(store_code)
        if store is None:
            return ToolResult(
                output={"type": "error", "reason": "store_not_found", "store_code": store_code}
            )

        deliveries = deliveries_for(store_code)
        if delivery_sn:
            deliveries = [d for d in deliveries if d["delivery_sn"] == delivery_sn]
        elif any(k in question for k in ("异常", "温度", "冷链", "坏了", "变质")):
            deliveries = [d for d in deliveries if d["status"] == "异常"]

        return ToolResult(
            output={
                "type": "delivery",
                "store": store,
                "deliveries": deliveries,
                "delivery_sn": delivery_sn,
            }
        )

    @staticmethod
    def _store_of_delivery(delivery_sn: str) -> str | None:
        from app.tools.store_data import load_store_data

        for delivery in load_store_data()["deliveries"]:
            if delivery["delivery_sn"] == delivery_sn:
                return delivery["store_code"]
        return None
