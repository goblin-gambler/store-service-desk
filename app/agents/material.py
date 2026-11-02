from langchain_core.language_models.chat_models import BaseChatModel

from app.agents.base import BaseAgent
from app.intent.few_shot_loader import load_agent_examples
from app.models.state import AgentProcessResult, AgentState
from app.tools.registry import ToolRegistry

MATERIAL_SYSTEM_PROMPT = """你是连锁零售门店的「物料申领」专家。

职责：
1. 查询、添加、修改、移除申领单中的物料
2. 每次操作后说明当前申领总额、月度额度与剩余额度

规则：
1. 物料、价格、额度必须来自工具返回，严禁编造
2. 超出月度额度时，明确告知需要区域督导审批，不能承诺一定通过
3. 申领单只是草稿，提交后才会生成正式领用单
4. 语气简洁、面向店员"""


class MaterialAgent(BaseAgent):
    def __init__(self, tool_registry: ToolRegistry, llm: BaseChatModel):
        super().__init__(name="material_agent", llm=llm, system_prompt=MATERIAL_SYSTEM_PROMPT)
        self.tool_registry = tool_registry
        self._few_shot_examples = load_agent_examples("material")

    async def process(self, state: AgentState) -> AgentProcessResult:
        await self._load_config()
        override = await self._resolve_experiment_prompt(state)
        if override:
            self._dynamic_system_prompt = override

        tool_result = await self.tool_registry.execute("material", state)
        output = tool_result.output
        status = output.get("status")
        action = output.get("action")

        if status == "error":
            response_text = f"操作未完成：{output.get('reason', '未知错误')}"
        elif action == "QUERY":
            items = output.get("items", [])
            if not items:
                response_text = (
                    "当前申领单是空的。可以直接说“申领 2 箱纸杯、1 箱吸管”，我来帮您添加。"
                )
            else:
                lines = [f"🧾 物料申领单（共 {len(items)} 项，合计 ¥{output.get('total', 0):.2f}）："]
                for item in items:
                    lines.append(
                        f"• {item['name']} x {item['quantity']} = ¥{item['subtotal']:.2f}"
                    )
                lines.append(
                    f"月度额度 ¥{output.get('quota', 0):.0f}，剩余 ¥{output.get('remaining', 0):.2f}"
                )
                response_text = "\n".join(lines)
        elif action == "ADD":
            response_text = (
                f"已将 {output.get('name')} x {output.get('quantity')} 加入申领单，"
                f"当前合计 ¥{output.get('total', 0):.2f}，剩余额度 ¥{output.get('remaining', 0):.2f}。"
            )
        elif action == "REMOVE":
            response_text = (
                f"已移除 {output.get('name')}，当前合计 ¥{output.get('total', 0):.2f}。"
            )
        elif action == "MODIFY":
            response_text = (
                f"已将 {output.get('name')} 数量改为 {output.get('quantity')}，"
                f"当前合计 ¥{output.get('total', 0):.2f}。"
            )
        else:
            response_text = "申领单操作已完成。"

        return {"response": response_text, "updated_state": {"material_data": output}}
