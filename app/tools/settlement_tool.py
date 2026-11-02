"""结算与对账工具。

演示版返回近几日门店日结快照；差账会给出原因字段，方便 Agent 解释而不是编造。
"""

from __future__ import annotations

import re
from typing import Any

from app.models.state import AgentState
from app.tools.base import BaseTool, ToolResult
from app.tools.store_data import (
    default_store_code,
    extract_store_code,
    get_store,
    settlements_for,
)

_DATE_PATTERN = re.compile(r"(20\d{2}-\d{1,2}-\d{1,2})")


class SettlementTool(BaseTool):
    name = "settlement"
    description = "查询门店日结、对账差异与发票状态"

    async def execute(self, state: AgentState, **kwargs) -> ToolResult:
        _ = kwargs
        question = state.get("question", "")
        slots = state.get("slots") or {}
        store_code = extract_store_code(question, slots) or default_store_code(state.get("user_id"))
        if not store_code:
            return ToolResult(
                output={
                    "type": "error",
                    "reason": "missing_store",
                    "message": "对账需要门店编号（如 ST1001），请补充。",
                }
            )

        store = get_store(store_code)
        if store is None:
            return ToolResult(
                output={"type": "error", "reason": "store_not_found", "store_code": store_code}
            )

        rows: list[dict[str, Any]] = settlements_for(store_code)
        date_match = _DATE_PATTERN.search(question)
        if date_match:
            target = date_match.group(1)
            rows = [r for r in rows if r["settle_date"] == target]

        invoice_question = "发票" in question
        return ToolResult(
            output={
                "type": "settlement",
                "store": store,
                "settlements": rows,
                "invoice_question": invoice_question,
            }
        )
