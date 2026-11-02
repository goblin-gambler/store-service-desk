"""意图分类器"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from langchain_core.exceptions import LangChainException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import settings
from app.core.llm_factory import create_openai_llm, maybe_add_cache_control
from app.core.tracing import build_llm_config
from app.intent.config import validate_tertiary_intent
from app.intent.few_shot_loader import (
    format_intent_examples_for_prompt,
    load_intent_examples,
    select_top_k_examples,
)
from app.intent.models import IntentAction, IntentCategory, IntentResult

logger = logging.getLogger(__name__)


class IntentClassifier:
    """意图分类器"""

    supports_multi_intent: bool = True
    RULE_MATCH_CONFIDENCE = 0.8
    DEFAULT_FALLBACK_CONFIDENCE = 0.3
    FUNCTION_CALLING_THRESHOLD = settings.FUNCTION_CALLING_THRESHOLD

    SYSTEM_PROMPT = """你是连锁零售门店服务台的意图识别专家。用户是店员、店长或区域督导，
他们在处理门店运营问题。请分析输入，识别意图并提取槽位。

一级意图(primary_intent):
- STORE: 门店信息、设备状态、设备报修
- LOSS: 报损申请与报损进度
- POLICY: 运营政策（排班、请假、考勤、报损制度、员工手册）
- SUPPLY: 配送、补货、冷链异常
- SETTLEMENT: 日结、对账差异、发票
- PRODUCT: 商品、配方、保质期、下架与临期
- MATERIAL: 物料申领单（纸杯、吸管、包装等）
- COMPLAINT: 顾客投诉、差评、食品安全事件
- STAFF: 员工档案、账号、权限、培训
- OTHER: 问候、闲聊、无法归类

二级意图(secondary_intent): QUERY, APPLY, MODIFY, CANCEL, CONSULT, ADD, REMOVE, COMPARE
三级意图(tertiary_intent): 可选，具体场景

槽位提取重点：store_code（ST1001 形式）、amount（报损金额）、employee_no（E1001 形式）、
device_type、delivery_sn（DL 开头）、material_name、quantity。

请输出JSON: primary_intent, secondary_intent, tertiary_intent, confidence, slots
"""

    INTENT_FUNCTION_SCHEMA = {
        "name": "classify_intent",
        "description": "分析用户输入，识别连锁零售门店服务台意图并提取槽位",
        "parameters": {
            "type": "object",
            "properties": {
                "primary_intent": {
                    "type": "string",
                    "enum": [c.value for c in IntentCategory],
                    "description": "一级意图",
                },
                "secondary_intent": {
                    "type": "string",
                    "enum": [a.value for a in IntentAction],
                    "description": "二级意图",
                },
                "tertiary_intent": {"type": "string", "description": "三级意图"},
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "置信度",
                },
                "slots": {"type": "object", "description": "槽位"},
            },
            "required": ["primary_intent", "secondary_intent", "confidence", "slots"],
        },
    }

    RULE_PATTERNS: dict[tuple[str, str], list[str]] = {
        ("LOSS", "APPLY"): [
            r"(报损|报费|损耗|报废).*(申请|提交|登记|报一下|上报)",
            r"(打翻|摔坏|摔了|破损|压坏|过期|变质).*(报损|申请|登记)",
            r"损失.*(\d+).*(元|块)",
            r"报损.*(\d+).*(元|块)",
        ],
        ("LOSS", "QUERY"): [
            r"报损.*(进度|状态|到哪|记录)",
            r"报损.*审批.*(进度|结果|到哪)",
            r"我的报损",
            r"LS\d+",
        ],
        ("STORE", "APPLY"): [
            r"(报修|维修|派人|上门|修一下|修修)",
            r"(设备|机器|pos|制冰机|打印机|冷柜).*(坏|故障|报警|离线|不能用)",
        ],
        ("STORE", "QUERY"): [
            r"^ST\d{4}$",
            r"ST\d{4}.*(信息|地址|营业|店长|电话|状态)",
            r"门店.*(信息|地址|营业时间|店长|电话)",
            r"(设备|机器|pos|制冰机|打印机|冷柜).*(状态|报警|故障|离线|在吗)",
        ],
        ("POLICY", "CONSULT"): [
            r"(排班|请假|调休|考勤|报销|员工手册|制度|政策|规定).*(怎么|如何|多久|多少|能不能|可以吗|流程|要求)",
            r"(报损|物料申领).*(制度|规范|标准|额度)",
            r"(报损|物料|结算|设备).*(多少|超过|阈值|标准|层级).*(审批|批准)",
            r"(审批层级|审批权限|谁审批|哪级审批)",
            r"(迟到|旷工|年假|病假).*(怎么|如何|规定)",
        ],
        ("SUPPLY", "QUERY"): [
            r"(补货|配送|到货|送货|冷链).*(进度|到哪|什么时候|状态|温度|异常)",
            r"DL\d+",
            r"(什么时候|多久).*(到货|送达|补货)",
        ],
        ("SETTLEMENT", "QUERY"): [
            r"(结算|对账|日结|营业额|账目).*(查|看|多少|差异|状态|怎么)",
            r"(差异|差账|少钱|多钱).*(怎么|为什么|原因)",
            r"发票.*(开|进度|状态|怎么)",
        ],
        ("PRODUCT", "QUERY"): [
            r"(商品|产品|配方|配料|保质期|下架|临期).*(是什么|多少|多久|怎么|有没有|查|在售|下架)",
            r"(这个|这款).*(配方|保质期|规格|过敏原|价格)",
            r"(还有|有没有).*(在售|库存|货)",
        ],
        ("PRODUCT", "COMPARE"): [r"(对比|比较|哪个好|区别)", r"和.*(比|区别)"],
        ("MATERIAL", "QUERY"): [r"申领单.*(看|查|有什么|内容|状态)", r"我的申领", r"物料.*额度"],
        ("MATERIAL", "ADD"): [
            r"(申领|领用|申请).*(物料|纸杯|吸管|包装|耗材|手套|围裙)",
            r"(加|添加|再来).*(纸杯|吸管|物料)",
        ],
        ("MATERIAL", "REMOVE"): [r"(删|移除|去掉|取消).*(申领|物料|纸杯|吸管)"],
        ("STAFF", "APPLY"): [
            r"(开通|申请|变更|修改|关闭|注销).*(权限|账号)",
            r"(权限|账号).*(申请|变更|开通)",
        ],
        ("STAFF", "QUERY"): [
            r"(员工|店员|店长|工号|账号|权限|培训).*(信息|查|记录|什么|谁|有哪些)",
            r"E\d{4}",
            r"我的.*(权限|培训|角色)",
        ],
        ("COMPLAINT", "QUERY"): [
            r"(顾客|客人|客户).*(投诉|差评|不满意|举报)",
            r"(食品安全|异物|变质|吃坏|拉肚子|发霉)",
            r"(服务态度|店员态度).*(差|不好|投诉)",
            r"(投诉|举报|差评)",
        ],
        ("OTHER", "CONSULT"): [r"^(你好|您好|hi|hello|在吗|有人吗)", r"谢谢|再见|拜拜"],
    }

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self._fast_llm = llm if llm is not None else create_openai_llm(timeout=15.0, max_retries=1)
        self._compiled_rules = {
            key: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for key, patterns in self.RULE_PATTERNS.items()
        }
        self._few_shot_examples = load_intent_examples()

    async def classify(
        self, query: str, context: dict[str, Any] | None = None, min_confidence: float = 0.5
    ) -> IntentResult:
        result = self._classify_with_rules(query)
        if result.confidence >= self.FUNCTION_CALLING_THRESHOLD:
            return self._finalize_result(result, query, min_confidence)

        llm_result = await self._classify_with_function_calling(query, context)
        if llm_result is not None and llm_result.confidence >= self.FUNCTION_CALLING_THRESHOLD:
            return self._finalize_result(llm_result, query, min_confidence)

        return self._finalize_result(result, query, min_confidence)

    def _finalize_result(
        self, result: IntentResult, query: str, min_confidence: float
    ) -> IntentResult:
        is_valid, _ = self._validate_result(result)
        if not is_valid:
            result = self._classify_with_rules(query)

        if result.confidence < min_confidence:
            result.needs_clarification = True
            result.clarification_question = self._generate_clarification_question(result)

        return result

    def _validate_result(self, result: IntentResult) -> tuple[bool, str]:
        if not isinstance(result.primary_intent, IntentCategory):
            return False, f"无效的一级意图: {result.primary_intent}"
        if not isinstance(result.secondary_intent, IntentAction):
            return False, f"无效的二级意图: {result.secondary_intent}"
        if result.tertiary_intent and not validate_tertiary_intent(
            result.primary_intent, result.secondary_intent, result.tertiary_intent
        ):
            return False, f"无效的三级意图: {result.tertiary_intent}"
        if not 0 <= result.confidence <= 1:
            return False, f"置信度超出范围: {result.confidence}"
        return True, ""

    validate_intent_result = _validate_result

    async def _classify_with_function_calling(
        self, query: str, context: dict[str, Any] | None = None
    ) -> IntentResult | None:
        messages = self._create_messages(query, context)
        tool_choice: Any = {"type": "function", "function": {"name": "classify_intent"}}
        if "dashscope" in settings.OPENAI_BASE_URL.lower() or settings.LLM_MODEL.startswith("qwen"):
            tool_choice = "auto"
        llm_with_tools = self._fast_llm.bind_tools(
            [self.INTENT_FUNCTION_SCHEMA],
            tool_choice=tool_choice,  # type: ignore
        )
        config = build_llm_config(
            agent_name="intent_classifier",
            tags=["intent", "internal"],
        )
        try:
            response = await asyncio.wait_for(
                llm_with_tools.ainvoke(messages, config=config),
                timeout=5.0,
            )
        except TimeoutError:
            logger.warning("Intent classification LLM call timed out after 5s")
            return None
        except (LangChainException, ConnectionError, OSError) as exc:
            logger.warning("Intent classification LLM call failed: %s", exc)
            return None
        tool_calls = response.tool_calls or []
        if not tool_calls:
            return None
        tool_call = tool_calls[0]
        result_data = tool_call.get("args", {})
        try:
            return self._parse_result(result_data, query)
        except (KeyError, ValueError, AttributeError) as exc:
            logger.warning(f"Failed to parse LLM intent result: {exc}")
            return None

    def _classify_with_rules(self, query: str) -> IntentResult:
        query_lower = query.lower()
        for (primary, secondary), patterns in self._compiled_rules.items():
            for pattern in patterns:
                if pattern.search(query_lower):
                    slots: dict[str, Any] = {"matched_pattern": pattern.pattern}
                    if primary == "MATERIAL" and secondary in ("ADD", "REMOVE", "MODIFY", "QUERY"):
                        slots["action"] = secondary
                    return IntentResult(
                        primary_intent=IntentCategory[primary],
                        secondary_intent=IntentAction[secondary],
                        confidence=self.RULE_MATCH_CONFIDENCE,
                        slots=slots,
                        raw_query=query,
                    )
        return IntentResult(
            primary_intent=IntentCategory.OTHER,
            secondary_intent=IntentAction.CONSULT,
            confidence=self.DEFAULT_FALLBACK_CONFIDENCE,
            raw_query=query,
        )

    def _create_messages(self, query: str, context: dict[str, Any] | None = None) -> list:
        user_content = query
        if self._few_shot_examples:
            top_examples = select_top_k_examples(query, self._few_shot_examples, k=3)
            if top_examples:
                user_content = format_intent_examples_for_prompt(top_examples) + "\n" + query
        messages: list = [HumanMessage(content=user_content)]
        if context:
            context_str = "\n".join(f"{k}: {v}" for k, v in context.items())
            messages.insert(0, SystemMessage(content=self.SYSTEM_PROMPT + "\n" + context_str))
        else:
            messages.insert(0, SystemMessage(content=self.SYSTEM_PROMPT))
        return maybe_add_cache_control(messages)

    def _parse_result(self, data: dict[str, Any], query: str) -> IntentResult:
        primary = data.get("primary_intent", "OTHER")
        secondary = data.get("secondary_intent", "CONSULT")
        tertiary = data.get("tertiary_intent")
        confidence = data.get("confidence", 0.5)
        slots = data.get("slots", {})
        primary_intent = IntentCategory[primary]
        secondary_intent = IntentAction[secondary]
        if tertiary and not validate_tertiary_intent(primary_intent, secondary_intent, tertiary):
            tertiary = None
        return IntentResult(
            primary_intent=primary_intent,
            secondary_intent=secondary_intent,
            tertiary_intent=tertiary,
            confidence=confidence,
            slots=slots,
            raw_query=query,
        )

    def _generate_clarification_question(self, result: IntentResult) -> str:
        clarification_map = {
            ("STORE", "QUERY"): "请问您要查询哪家门店？可以提供门店编号（如 ST1001）。",
            ("STORE", "APPLY"): "请问是哪家门店、哪台设备需要报修？",
            ("LOSS", "APPLY"): "请说明门店、报损金额和原因，例如：张江店打翻 3 箱鲜奶，损失约 350 元。",
            ("LOSS", "QUERY"): "请问您想查询哪一张报损单？可以提供报损单号（LS 开头）。",
            ("PRODUCT", "QUERY"): "请问您想了解哪款商品的配方、保质期还是在售状态？",
            ("POLICY", "CONSULT"): "请问您想了解哪方面的门店制度？",
            ("MATERIAL", "ADD"): "请问您要申领什么物料、数量是多少？",
            ("SUPPLY", "QUERY"): "请问您要查哪家门店的补货进度？可以提供门店编号或补货单号。",
            ("SETTLEMENT", "QUERY"): "请问您要核对哪家门店、哪一天的日结？",
        }
        return clarification_map.get(
            (result.primary_intent.value, result.secondary_intent.value),
            "抱歉，我没有完全理解您的意思，能否再说详细一些？",
        )
