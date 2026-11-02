from langchain_core.language_models.chat_models import BaseChatModel

from app.agents.base import BaseAgent
from app.intent.few_shot_loader import load_agent_examples
from app.models.state import AgentProcessResult, AgentState
from app.tools.registry import ToolRegistry

SETTLEMENT_SYSTEM_PROMPT = """你是连锁零售门店的「结算与对账」专家。

职责：
1. 查询门店日结、差异金额与差异原因
2. 解释对账差异的可能来源（POS 离线、券核销、现金未入账），但不得编造具体流水
3. 发票问题只说明流程归属，不承诺开票时间

规则：
1. 金额必须来自工具返回，精确到分
2. 差异原因未知时，明确说"需要财务核对"，不要猜
3. 语气专业、克制"""


class SettlementAgent(BaseAgent):
    def __init__(self, tool_registry: ToolRegistry, llm: BaseChatModel):
        super().__init__(
            name="settlement_agent", llm=llm, system_prompt=SETTLEMENT_SYSTEM_PROMPT
        )
        self.tool_registry = tool_registry
        self._few_shot_examples = load_agent_examples("settlement")

    async def process(self, state: AgentState) -> AgentProcessResult:
        await self._load_config()
        override = await self._resolve_experiment_prompt(state)
        if override:
            self._dynamic_system_prompt = override

        tool_result = await self.tool_registry.execute("settlement", state)
        output = tool_result.output

        if output.get("type") == "error":
            if output.get("reason") == "missing_store":
                text = output.get("message", "对账需要门店编号（如 ST1001）。")
            else:
                text = f"未找到门店 {output.get('store_code')}，请确认门店编号。"
            return {"response": text, "updated_state": {"settlement_data": None}}

        rows = output.get("settlements", [])
        store = output.get("store") or {}
        if not rows:
            return {
                "response": (
                    f"{store.get('name', '该门店')} 没有查询到对应日期的日结记录。"
                    "可以不带日期查询最近两天，或联系财务核对结算批次。"
                ),
                "updated_state": {"settlement_data": output},
            }

        lines = [f"💰 {store.get('name', '')} 日结信息："]
        for row in rows:
            diff = float(row["diff_amount"])
            diff_text = f"差异 ¥{diff:+.2f}" if diff else "已平账"
            lines.append(f"• {row['settle_date']} | 应结 ¥{row['expected_amount']:.2f}")
            lines.append(f"  实结 ¥{row['actual_amount']:.2f} | {diff_text}")
            lines.append(f"  状态: {row['status']}")
            if row.get("diff_reason"):
                lines.append(f"  差异原因: {row['diff_reason']}")
            else:
                lines.append("  差异原因: 无差异")

        if any(float(r["diff_amount"]) != 0 for r in rows):
            lines.append("建议：差异单请当日提交财务核对，超过 48 小时需要区域督导签字。")
        if output.get("invoice_question"):
            lines.append("发票说明：门店发票由总部财务统一开具，具体进度以财务系统为准。")

        return {"response": "\n".join(lines), "updated_state": {"settlement_data": output}}
