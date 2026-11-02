import json
import logging

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError

from app.agents.base import BaseAgent
from app.intent.few_shot_loader import load_agent_examples
from app.models.state import AgentProcessResult, AgentState
from app.tools.complaint_tool import ComplaintTool

logger = logging.getLogger(__name__)


class ComplaintClassification(BaseModel):
    category: str = Field(description="投诉类别: food_safety, service, environment, supply, other")
    urgency: str = Field(description="紧急程度: low, medium, high")
    summary: str = Field(description="投诉摘要，用于生成工单标题/描述")
    expected_resolution: str = Field(
        description="期望处理方式: apology, compensation, replacement, investigation"
    )
    empathetic_response: str = Field(description="对顾客的同理心回复，包含工单号占位符 {ticket_id}")


_COMPLAINT_SYSTEM_PROMPT = """你是连锁零售门店的顾客投诉处理专家。

规则：
1. 认真倾听顾客的不满，表达理解和同理心
2. 主动了解投诉的具体原因和期望的处理方式
3. 涉及食品安全（异物、变质、过期）的投诉一律按最高紧急度处理，并提示门店保留样品
4. 如果问题超出门店权限，安抚顾客并告知会上报区域督导处理
5. 语气真诚、专业、耐心

输出要求：
- 请分析投诉并返回 JSON 格式：
{
  "category": "food_safety | service | environment | supply | other",
  "urgency": "low | medium | high",
  "summary": "投诉摘要",
  "expected_resolution": "apology | compensation | replacement | investigation",
  "empathetic_response": "对顾客的温暖回复，可包含 {ticket_id} 占位符"
}"""


class ComplaintAgent(BaseAgent):
    def __init__(self, llm: BaseChatModel):
        super().__init__(name="complaint", llm=llm, system_prompt=_COMPLAINT_SYSTEM_PROMPT)
        self._tool: ComplaintTool = ComplaintTool()
        self._few_shot_examples = load_agent_examples("complaint")

    async def process(self, state: AgentState) -> AgentProcessResult:
        await self._load_config()
        override = await self._resolve_experiment_prompt(state)
        if override:
            self._dynamic_system_prompt = override
        question = state.get("question", "")
        user_id = state.get("user_id", 0)
        thread_id = state.get("thread_id", "")

        # Fast-path: skip LLM for common complaint patterns to ensure <2s response
        classification = self._classify_with_rules(question)

        if classification.category == "other":
            try:
                messages = [
                    {
                        "role": "system",
                        "content": self._dynamic_system_prompt or self.system_prompt,
                    },
                    {"role": "user", "content": question},
                ]
                response = await self.llm.ainvoke(messages)
                if hasattr(response, "content") and isinstance(response.content, str):
                    raw_output = response.content
                else:
                    raw_output = str(response)
                classification = self._parse_classification(raw_output)
            except (ConnectionError, OSError, RuntimeError):
                logger.exception("LLM classification failed, using defaults")

        try:
            ticket = await self._tool.create_ticket(
                user_id=user_id,
                thread_id=thread_id,
                category=classification.category,
                urgency=classification.urgency,
                description=classification.summary,
                expected_resolution=classification.expected_resolution,
            )
            ticket_id = ticket.get("ticket_id", "N/A")
            response_text = classification.empathetic_response.replace(
                "{ticket_id}", str(ticket_id)
            )
        except (SQLAlchemyError, ConnectionError, OSError, RuntimeError):
            logger.exception("Failed to create complaint ticket")
            response_text = (
                "非常抱歉给您带来不好的体验，我们已经记录了您的问题，门店团队会尽快与您联系处理。"
            )

        return {
            "response": response_text,
            "updated_state": {"answer": response_text},
        }

    def _classify_with_rules(self, question: str) -> ComplaintClassification:
        """Fast rule-based classification to avoid LLM latency."""
        q = question.lower()
        if any(k in q for k in ("食品安全", "异物", "变质", "过期", "拉肚子", "食物中毒", "发霉")):
            return ComplaintClassification(
                category="food_safety",
                urgency="high",
                summary=question,
                expected_resolution="investigation",
                empathetic_response="非常抱歉出现食品安全相关的问题，我们已按最高优先级创建工单 #{ticket_id}，请保留样品与购物凭证，区域品控会在 2 小时内联系您。",
            )
        if any(k in q for k in ("环境", "卫生", "脏", "异味", "排队", "吵闹")):
            return ComplaintClassification(
                category="environment",
                urgency="medium",
                summary=question,
                expected_resolution="apology",
                empathetic_response="非常抱歉门店环境让您感到不适，我们已记录（工单号：{ticket_id}），店长会尽快整改。",
            )
        if any(k in q for k in ("补货", "缺货", "物料", "断货", "配送")):
            return ComplaintClassification(
                category="supply",
                urgency="medium",
                summary=question,
                expected_resolution="replacement",
                empathetic_response="非常抱歉因为缺货/配送问题影响了门店运营，我们已记录（工单号：{ticket_id}），会尽快协调补货。",
            )
        if any(k in q for k in ("店员", "态度", "服务", "服务态度")):
            return ComplaintClassification(
                category="service",
                urgency="medium",
                summary=question,
                expected_resolution="apology",
                empathetic_response="非常抱歉我们的服务没有让您满意，我们已经记录了您的反馈（工单号：{ticket_id}），店长会尽快跟进处理。",
            )
        return ComplaintClassification(
            category="other",
            urgency="medium",
            summary=question,
            expected_resolution="apology",
            empathetic_response="非常抱歉给您带来不好的体验，我们已经记录了您的问题（工单号：{ticket_id}），客服团队会尽快与您联系处理。",
        )

    def _parse_classification(self, raw: str) -> ComplaintClassification:
        try:
            json_str = raw
            if "```json" in raw:
                json_str = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                json_str = raw.split("```")[1].split("```")[0].strip()
            data = json.loads(json_str)
            return ComplaintClassification(**data)
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse complaint classification JSON, using defaults")
            return ComplaintClassification(
                category="other",
                urgency="medium",
                summary=raw,
                expected_resolution="apology",
                empathetic_response=raw,
            )
