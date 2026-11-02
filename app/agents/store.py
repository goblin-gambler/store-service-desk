import logging

from langchain_core.language_models.chat_models import BaseChatModel

from app.agents.base import BaseAgent
from app.intent.few_shot_loader import load_agent_examples
from app.models.state import AgentProcessResult, AgentState
from app.services.loss_service import LossService
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

STORE_SYSTEM_PROMPT = """你是连锁零售门店的「门店与设备」专家。

职责：
1. 查询门店基础信息（营业时间、地址、店长、营业状态）
2. 查询设备状态（POS、制冰机、标签打印机、冷藏柜等）
3. 受理设备报修，生成报修工单并告知上门时间

规则：
1. 门店、设备、工单数据必须来自工具返回，严禁编造
2. 报修前先说明已发现的异常设备，让用户确认是否针对该设备报修
3. 语气简洁、务实，面向店长和店员，不使用电商话术"""

class StoreAgent(BaseAgent):
    """门店与设备专家（含报损受理）"""

    def __init__(
        self,
        loss_service: LossService,
        tool_registry: ToolRegistry,
        llm: BaseChatModel,
    ):
        super().__init__(name="store_agent", llm=llm, system_prompt=STORE_SYSTEM_PROMPT)
        self.loss_service = loss_service
        self.tool_registry = tool_registry
        self._few_shot_examples = load_agent_examples("store")

    async def process(self, state: AgentState) -> AgentProcessResult:
        await self._load_config()
        override = await self._resolve_experiment_prompt(state)
        if override:
            self._dynamic_system_prompt = override

        question = state.get("question", "")
        user_id = state.get("user_id")
        if user_id is None:
            return {
                "response": "抱歉，无法识别员工身份，请重新登录。",
                "updated_state": {"store_data": None},
            }

        intent_result = state.get("intent_result") or {}
        primary = self._intent_value(intent_result.get("primary_intent"))

        if primary == "LOSS":
            if self._is_loss_query(question):
                return await self._handle_loss_query(question, user_id)
            return await self._handle_loss_report(state, question, user_id)

        tool_result = await self.tool_registry.execute("store", state)
        return self._format_tool_output(tool_result.output)

    @staticmethod
    def _intent_value(value: object) -> str:
        if value is None:
            return ""
        return value.value if hasattr(value, "value") else str(value)

    @staticmethod
    def _is_loss_query(question: str) -> bool:
        query_markers = ("进度", "状态", "查到哪", "审批到哪", "我的报损", "报损记录")
        return any(marker in question for marker in query_markers) and not any(
            marker in question for marker in ("提交", "申请", "报一个", "登记")
        )

    async def _handle_loss_report(
        self, state: AgentState, question: str, user_id: int
    ) -> AgentProcessResult:
        return await self.loss_service.submit_loss_report(
            question=question,
            user_id=user_id,
            thread_id=state.get("thread_id", ""),
            slots=state.get("slots") or {},
        )

    async def _handle_loss_query(self, question: str, user_id: int) -> AgentProcessResult:
        reports = await self.loss_service.list_recent_reports(user_id, limit=3)
        if not reports:
            return {
                "response": "您当前没有报损记录。如需提交，请说明门店、金额和原因，例如：张江店打翻 3 箱鲜奶，损失约 350 元。",
                "updated_state": {"loss_data": None},
            }
        lines = ["🧾 最近的报损单："]
        for report in reports:
            lines.append(
                f"• {report['report_sn']} | {report['store_name']} | ¥{report['amount']:.2f} | "
                f"{report['reason']} | {self._status_label(report)}"
            )
        return {"response": "\n".join(lines), "updated_state": {"loss_data": reports[0]}}

    @staticmethod
    def _status_label(report: dict) -> str:
        status_map = {
            "PENDING": "待审批",
            "APPROVED": "已通过",
            "REJECTED": "已驳回",
            "COMPLETED": "已完成",
            "CANCELLED": "已取消",
        }
        level_map = {
            "AUTO": "自动通过",
            "STORE_MANAGER": "店长审批",
            "REGIONAL": "区域督导审批",
            "HQ": "总部审批",
        }
        status = status_map.get(report["status"], report["status"])
        level = level_map.get(report["approval_level"], report["approval_level"])
        return f"{status}（{level}）"

    def _format_tool_output(self, output: dict) -> AgentProcessResult:
        output_type = output.get("type")

        if output_type == "error":
            if output.get("reason") == "missing_store":
                text = output.get("message", "请提供门店编号（如 ST1001）。")
            elif output.get("reason") == "store_not_found":
                text = f"未找到门店 {output.get('store_code')}，请确认门店编号是否正确。"
            else:
                text = "报修服务暂时不可用，请稍后重试或直接联系区域运维。"
            return {"response": text, "updated_state": {"store_data": None}}

        if output_type == "store_info":
            store = output["store"]
            lines = [
                f"🏪 {store['name']}（{store['store_code']}）",
                f"区域: {store['region']}",
                f"地址: {store['address']}",
                f"营业时间: {store['opening_hours']}",
                f"店长: {store['manager']}（{store['manager_phone']}）",
                f"营业状态: {store['status']}",
            ]
            devices = output.get("devices", [])
            abnormal = [d for d in devices if d["status"] != "ONLINE"]
            if abnormal:
                lines.append("设备异常:")
                for device in abnormal:
                    lines.append(
                        f"  - {device['device_type']} {device['device_code']}："
                        f"{device['note']}（{device['last_report']}）"
                    )
            else:
                lines.append("设备状态: 全部在线")
            return {"response": "\n".join(lines), "updated_state": {"store_data": output}}

        if output_type == "device_status":
            store = output["store"]
            devices = output.get("devices", [])
            lines = [f"🔧 {store['name']} 设备状态："]
            status_map = {
                "ONLINE": "在线",
                "OFFLINE": "离线",
                "ALARM": "报警",
                "MAINTENANCE": "维修中",
            }
            for device in devices:
                lines.append(
                    f"• {device['device_type']} {device['device_code']}"
                    f"（{device['model']}）: {status_map.get(device['status'], device['status'])}"
                    f" — {device['note']}"
                )
            if any(d["status"] != "ONLINE" for d in devices):
                lines.append("如需报修，请回复“帮我报修”，我会生成工单并派工程师上门。")
            return {"response": "\n".join(lines), "updated_state": {"store_data": output}}

        if output_type == "repair_ticket":
            text = (
                "🛠️ 报修工单已创建\n"
                f"工单号: {output['ticket_sn']}\n"
                f"门店: {output['store_name']}（{output['store_code']}）\n"
                f"涉及设备: {', '.join(output['device_codes'])}\n"
                f"紧急程度: {output['urgency']}\n"
                f"处理人: {output['engineer']}\n"
                f"预计: {output['eta']}"
            )
            return {"response": text, "updated_state": {"store_data": output}}

        if output_type == "ticket_status":
            text = (
                f"🔧 工单 {output['ticket_sn']} 当前状态: {output['status']}\n"
                f"处理人: {output['engineer']}（{output['eta']}）\n"
                f"问题描述: {output['issue']}"
            )
            return {"response": text, "updated_state": {"store_data": output}}

        if output_type == "ticket_not_found":
            return {
                "response": f"未找到工单 {output['ticket_sn']}，请确认工单号是否正确。",
                "updated_state": {"store_data": output},
            }

        return {"response": "已查询门店信息。", "updated_state": {"store_data": output}}
