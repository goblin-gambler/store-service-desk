"""员工与权限工具。

演示版把员工档案放在 mock JSON；账号本身仍来自 users 表。
权限变更属于高风险动作：工具只返回"需总部审批"，不直接改权限。
"""

from __future__ import annotations

import re
from typing import Any

from app.models.state import AgentState
from app.tools.base import BaseTool, ToolResult
from app.tools.store_data import extract_store_code, find_staff, get_store, load_store_data

_EMPLOYEE_PATTERN = re.compile(r"(E\d{4})", re.IGNORECASE)


class StaffTool(BaseTool):
    name = "staff"
    description = "查询员工档案、门店角色、权限与培训记录"

    async def execute(self, state: AgentState, **kwargs) -> ToolResult:
        _ = kwargs
        question = state.get("question", "")
        slots = state.get("slots") or {}
        combined = f"{question} {' '.join(str(v) for v in slots.values())}"

        employee = None
        matched = _EMPLOYEE_PATTERN.search(combined)
        if matched:
            employee = find_staff(matched.group(1).upper())

        store_code = extract_store_code(question, slots)
        staff_list: list[dict[str, Any]] = []
        if employee is None and store_code:
            staff_list = [s for s in load_store_data()["staff"] if s["store_code"] == store_code]
        if employee is None and not staff_list:
            # 没有工号/门店时，回退到与当前账号绑定的示例员工
            fallback_no = f"E{1001 + (int(state.get('user_id') or 1) - 1) % 4}"
            employee = find_staff(fallback_no)

        high_risk = any(k in question for k in ("开通权限", "改权限", "开账号", "删账号", "授权"))
        store = get_store(employee["store_code"]) if employee else (get_store(store_code) if store_code else None)

        return ToolResult(
            output={
                "type": "staff",
                "employee": employee,
                "staff": staff_list,
                "store": store,
                "requires_hq_approval": high_risk,
            }
        )
