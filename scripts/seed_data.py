# scripts/seed_data.py
"""写入门店服务台的演示账号、Agent 配置与路由规则（幂等）。

用法::

    PYTHONUTF8=1 PYTHONPATH=. python scripts/seed_data.py
"""

import asyncio
import os
import sys
from typing import Any

sys.path.append(os.getcwd())

from sqlmodel import select

from app.core.database import async_session_maker
from app.models.memory import AgentConfig, RoutingRule
from app.models.user import User

_DEFAULT_AGENT_CONFIGS: list[dict[str, Any]] = [
    {
        "agent_name": "policy_agent",
        "system_prompt": (
            "你是连锁零售门店的运营政策咨询专家。\n\n"
            "规则：\n1. 只能依据检索到的制度文档回答，严禁编造\n"
            "2. 检索不到时明确说明“制度库暂未收录”，并建议联系区域督导\n"
            "3. 引用制度编号或条款名称\n4. 语气专业、克制"
        ),
        "confidence_threshold": 0.7,
        "max_retries": 2,
    },
    {
        "agent_name": "store_agent",
        "system_prompt": (
            "你是连锁零售门店的「门店与设备」专家，并负责受理报损。\n\n"
            "规则：\n1. 门店/设备/工单数据必须来自工具\n"
            "2. 报损审批层级由系统按金额计算，严禁自行承诺“已经通过”\n"
            "3. 报修前先复述异常设备，请用户确认\n4. 语气务实简洁"
        ),
        "confidence_threshold": 0.7,
        "max_retries": 2,
    },
    {
        "agent_name": "supply_agent",
        "system_prompt": (
            "你是连锁零售门店的配送与补货专家。\n\n"
            "规则：\n1. 配送单、冷链温度、到店时间必须来自工具\n"
            "2. 冷链异常必须标注风险并建议立即隔离商品\n"
            "3. 不承诺具体到店时间以外的补偿\n4. 语气简洁"
        ),
        "confidence_threshold": 0.7,
        "max_retries": 2,
    },
    {
        "agent_name": "settlement_agent",
        "system_prompt": (
            "你是连锁零售门店的结算与对账专家。\n\n"
            "规则：\n1. 日结金额、差异、发票状态必须来自工具\n"
            "2. 有差异时说明差异金额与可能原因，建议门店复核\n"
            "3. 严禁编造账目数据\n4. 语气严谨"
        ),
        "confidence_threshold": 0.7,
        "max_retries": 2,
    },
    {
        "agent_name": "material_agent",
        "system_prompt": (
            "你是门店物料申领助手。\n\n"
            "规则：\n1. 申领单内容以工具返回为准\n"
            "2. 超出月度额度时明确提示，不得擅自修改额度\n"
            "3. 删除/清空操作前必须复述待删除内容\n4. 语气简洁"
        ),
        "confidence_threshold": 0.7,
        "max_retries": 2,
    },
    {
        "agent_name": "staff_agent",
        "system_prompt": (
            "你是门店员工与权限专家。\n\n"
            "规则：\n1. 员工档案、权限、培训记录必须来自工具\n"
            "2. 权限变更不能在线直接生效，必须走审批\n"
            "3. 只展示脱敏信息\n4. 语气专业"
        ),
        "confidence_threshold": 0.7,
        "max_retries": 2,
    },
    {
        "agent_name": "product",
        "system_prompt": (
            "你是门店商品与配方专家。\n\n"
            "规则：\n1. 商品、价格、配方、保质期必须基于商品目录\n"
            "2. 目录未标注的内容必须说“目录未标注”\n"
            "3. 严禁编造配方与过敏原信息\n4. 语气简洁"
        ),
        "confidence_threshold": 0.7,
        "max_retries": 2,
    },
    {
        "agent_name": "complaint",
        "system_prompt": (
            "你是门店顾客投诉受理专家。\n\n"
            "规则：\n1. 食品安全类投诉最高优先级，立即建议封存商品并上报\n"
            "2. 记录投诉要素：时间、门店、商品、诉求\n"
            "3. 不承诺赔偿金额，说明后续处理流程\n4. 语气诚恳"
        ),
        "confidence_threshold": 0.7,
        "max_retries": 2,
    },
]

_DEFAULT_ROUTING_RULES: list[dict[str, Any]] = [
    {"intent_category": "STORE", "target_agent": "store_agent", "priority": 10},
    {"intent_category": "LOSS", "target_agent": "store_agent", "priority": 10},
    {"intent_category": "POLICY", "target_agent": "policy_agent", "priority": 10},
    {"intent_category": "SUPPLY", "target_agent": "supply_agent", "priority": 10},
    {"intent_category": "SETTLEMENT", "target_agent": "settlement_agent", "priority": 10},
    {"intent_category": "PRODUCT", "target_agent": "product", "priority": 10},
    {"intent_category": "MATERIAL", "target_agent": "material_agent", "priority": 10},
    {"intent_category": "COMPLAINT", "target_agent": "complaint", "priority": 10},
    {"intent_category": "STAFF", "target_agent": "staff_agent", "priority": 10},
    {"intent_category": "OTHER", "target_agent": "policy_agent", "priority": 10},
]

_DEMO_USERS: list[dict[str, Any]] = [
    {
        "username": "store1001",
        "email": "store1001@example.com",
        "full_name": "上海张江店 店长",
        "password": "store123",
    },
    {
        "username": "store1002",
        "email": "store1002@example.com",
        "full_name": "北京中关村店 店长",
        "password": "store123",
    },
    {
        "username": "store1003",
        "email": "store1003@example.com",
        "full_name": "广州天河店 店长",
        "password": "store123",
    },
    {
        "username": "store1004",
        "email": "store1004@example.com",
        "full_name": "深圳福田店 店员",
        "password": "store123",
    },
]


async def seed_data() -> None:
    async with async_session_maker() as session:
        for demo in _DEMO_USERS:
            result = await session.exec(select(User).where(User.username == demo["username"]))
            if result.first():
                continue
            print(f"🌱 Creating store user {demo['username']}...")
            session.add(
                User(
                    username=demo["username"],
                    email=demo["email"],
                    full_name=demo["full_name"],
                    password_hash=User.hash_password(demo["password"]),
                )
            )
            await session.flush()

        result = await session.exec(select(User).where(User.username == "admin"))
        if not result.first():
            print("🌱 Creating admin user...")
            session.add(
                User(
                    username="admin",
                    email="admin@example.com",
                    full_name="区域督导",
                    password_hash=User.hash_password("admin123"),
                    is_admin=True,
                )
            )
            await session.flush()

        for cfg in _DEFAULT_AGENT_CONFIGS:
            result = await session.exec(
                select(AgentConfig).where(AgentConfig.agent_name == cfg["agent_name"])
            )
            if result.one_or_none():
                continue
            print(f"🤖 Creating agent config for {cfg['agent_name']}...")
            session.add(
                AgentConfig(
                    agent_name=cfg["agent_name"],
                    system_prompt=cfg["system_prompt"],
                    confidence_threshold=cfg["confidence_threshold"],
                    max_retries=cfg["max_retries"],
                    enabled=True,
                )
            )
            await session.flush()

        for rule_data in _DEFAULT_ROUTING_RULES:
            result = await session.exec(
                select(RoutingRule).where(
                    RoutingRule.intent_category == rule_data["intent_category"]
                )
            )
            if result.one_or_none():
                continue
            print(f"📋 Creating routing rule for {rule_data['intent_category']}...")
            session.add(
                RoutingRule(
                    intent_category=rule_data["intent_category"],
                    target_agent=rule_data["target_agent"],
                    priority=rule_data["priority"],
                )
            )
            await session.flush()

        await session.commit()
        print("✅ Seed data completed.")


if __name__ == "__main__":
    asyncio.run(seed_data())
