from langchain_core.language_models.chat_models import BaseChatModel

from app.agents.base import BaseAgent
from app.intent.few_shot_loader import load_agent_examples
from app.models.state import AgentProcessResult, AgentState
from app.tools.registry import ToolRegistry

SUPPLY_SYSTEM_PROMPT = """你是连锁零售门店的「配送与补货」专家。

职责：
1. 查询补货单、物料配送的进度与预计到达时间
2. 发现冷链温度异常时，明确提示品控复核状态，不要自行判断能否收货
3. 对未到货、少货、货损问题给出下一步动作（联系仓配 / 提报异常）

规则：
1. 配送数据必须来自工具返回，严禁编造单号或时间
2. 冷链异常属于高风险信息，必须原样转述温度与处理状态
3. 语气务实、简洁"""


class SupplyAgent(BaseAgent):
    def __init__(self, tool_registry: ToolRegistry, llm: BaseChatModel):
        super().__init__(name="supply_agent", llm=llm, system_prompt=SUPPLY_SYSTEM_PROMPT)
        self.tool_registry = tool_registry
        self._few_shot_examples = load_agent_examples("supply")

    async def process(self, state: AgentState) -> AgentProcessResult:
        await self._load_config()
        override = await self._resolve_experiment_prompt(state)
        if override:
            self._dynamic_system_prompt = override

        tool_result = await self.tool_registry.execute("supply", state)
        output = tool_result.output

        if output.get("type") == "error":
            if output.get("reason") == "missing_store":
                text = output.get("message", "请提供门店编号（如 ST1001）。")
            else:
                text = f"未找到门店 {output.get('store_code')}，请确认门店编号。"
            return {"response": text, "updated_state": {"supply_data": None}}

        deliveries = output.get("deliveries", [])
        store = output.get("store") or {}
        if not deliveries:
            return {
                "response": (
                    f"{store.get('name', '该门店')} 近期没有匹配的配送单。"
                    "如需查询历史配送，请提供补货单号（如 DL20261027001）。"
                ),
                "updated_state": {"supply_data": output},
            }

        lines = [f"🚚 {store.get('name', '')} 配送信息："]
        for delivery in deliveries:
            lines.append(
                f"• {delivery['delivery_sn']} | {delivery['type']} | {delivery['status']}"
            )
            lines.append(f"  货物: {delivery['items']}")
            lines.append(f"  承运: {delivery['carrier']} | 温度: {delivery['temperature']}")
            lines.append(f"  最新: {delivery['latest_update']}")
            lines.append(f"  预计到达: {delivery['eta']}")
            if delivery["status"] == "异常":
                lines.append("  ⚠️ 该单存在异常，收货前请等待品控复核结论。")

        return {"response": "\n".join(lines), "updated_state": {"supply_data": output}}
