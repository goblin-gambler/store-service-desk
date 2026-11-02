import logging

from langchain_core.language_models.chat_models import BaseChatModel

from app.agents.base import BaseAgent
from app.intent.few_shot_loader import load_agent_examples
from app.models.state import AgentProcessResult, AgentState
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

STAFF_SYSTEM_PROMPT = """你是连锁零售门店的「员工与权限」专家。

职责：
1. 查询员工档案、门店角色、权限清单与培训记录
2. 解释权限申请流程：门店内权限由店长审批，跨店/系统权限由总部审批

规则：
1. 员工信息必须来自工具返回，严禁编造工号、权限或培训记录
2. 权限变更一律不能在线直接生效，必须走审批；不得承诺开通时间
3. 不透露手机号、身份证等敏感信息，只展示脱敏信息
4. 语气专业、克制"""


class StaffAgent(BaseAgent):
    def __init__(self, tool_registry: ToolRegistry, llm: BaseChatModel):
        super().__init__(name="staff_agent", llm=llm, system_prompt=STAFF_SYSTEM_PROMPT)
        self.tool_registry = tool_registry
        self._few_shot_examples = load_agent_examples("staff")

    async def process(self, state: AgentState) -> AgentProcessResult:
        await self._load_config()
        override = await self._resolve_experiment_prompt(state)
        if override:
            self._dynamic_system_prompt = override

        tool_result = await self.tool_registry.execute("staff", state)
        output = tool_result.output
        employee = output.get("employee")
        staff_list = output.get("staff") or []

        lines: list[str] = []
        if employee:
            lines.append(f"👤 {employee['name']}（{employee['employee_no']}）")
            lines.append(f"门店: {output.get('store', {}).get('name', employee['store_code'])}")
            lines.append(f"角色: {employee['role']}")
            lines.append(f"状态: {employee['status']}")
            lines.append(f"权限: {', '.join(employee['permissions'])}")
            training = employee.get("training") or []
            lines.append(f"培训记录: {', '.join(training) if training else '暂无'}")
        elif staff_list:
            lines.append("当前门店员工：")
            for staff in staff_list:
                lines.append(f"• {staff['employee_no']} {staff['name']}（{staff['role']}）")
        else:
            lines.append("未查询到员工档案，请提供工号（如 E1001）。")

        if output.get("requires_hq_approval"):
            lines.append(
                "⚠️ 权限变更属于高风险操作：需由店长发起、总部运营审批，系统不能直接修改。"
            )

        return {"response": "\n".join(lines), "updated_state": {"staff_data": output}}
