"""物料申领工具。

申领单暂存在 Redis（与购物车同构），提交后由门店物料系统生成正式领用单。
月度额度写死在配置里：超过额度不自动放行，而是提示走区域督导审批。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import redis.asyncio as aioredis
from redis.exceptions import RedisError

from app.core.config import settings
from app.models.state import AgentState
from app.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

MONTHLY_QUOTA = 5000.0

_MATERIAL_KEYWORDS = (
    "纸杯",
    "杯盖",
    "吸管",
    "打包袋",
    "标签纸",
    "清洁用品",
    "工服",
    "手套",
    "纸巾",
    "搅拌棒",
)
_QUANTITY_PATTERN = re.compile(r"(\d+)\s*(箱|个|包|件|套|卷|条)")

# 演示版物料单价（元/最小申领单位），真实项目应查物料主数据
_MATERIAL_PRICES = {
    "纸杯": 35.0,
    "杯盖": 28.0,
    "吸管": 18.0,
    "打包袋": 22.0,
    "标签纸": 15.0,
    "清洁用品": 60.0,
    "工服": 120.0,
    "手套": 25.0,
    "纸巾": 30.0,
    "搅拌棒": 12.0,
}


def _extract_material_name(question: str) -> str | None:
    for keyword in _MATERIAL_KEYWORDS:
        if keyword in question:
            return keyword
    return None


def _extract_quantity(question: str) -> int | None:
    matched = _QUANTITY_PATTERN.search(question)
    return int(matched.group(1)) if matched else None


class MaterialTool(BaseTool):
    name = "material"
    description = "物料申领单的查询、添加、修改与删除"

    def __init__(self, redis_client: aioredis.Redis | None = None, key_prefix: str = ""):
        self._redis = redis_client
        self._key_prefix = key_prefix

    async def _get_redis(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        return self._redis

    async def execute(self, state: AgentState, **kwargs) -> ToolResult:
        slots = state.get("slots") or {}
        question = state.get("question", "")
        user_id = state.get("user_id")
        action = (slots.get("action") or kwargs.get("action") or "QUERY").upper()
        name = (
            slots.get("material_name")
            or slots.get("product_name")
            or kwargs.get("name")
            or _extract_material_name(question)
        )
        quantity = int(
            slots.get("quantity")
            or kwargs.get("quantity")
            or _extract_quantity(question)
            or 1
        )
        unit_price = float(
            slots.get("price")
            or kwargs.get("price")
            or _MATERIAL_PRICES.get(name or "", 0)
        )

        redis = await self._get_redis()
        key = f"{self._key_prefix}material_request:{user_id}"

        try:
            request = await self._load(redis, key)

            if action == "QUERY":
                return ToolResult(output={"type": "material", "action": "QUERY", **request})

            if action == "ADD":
                if not name:
                    return ToolResult(
                        output={"type": "material", "status": "error", "reason": "缺少物料名称"}
                    )
                merged = False
                for item in request["items"]:
                    if item["name"] == name:
                        item["quantity"] += quantity
                        item["subtotal"] = round(item["unit_price"] * item["quantity"], 2)
                        merged = True
                        break
                if not merged:
                    request["items"].append(
                        {
                            "name": name,
                            "quantity": quantity,
                            "unit_price": unit_price,
                            "subtotal": round(unit_price * quantity, 2),
                        }
                    )
                return await self._save_and_return(redis, key, request, action, name, quantity)

            if action == "REMOVE":
                removed = False
                new_items = []
                for item in request["items"]:
                    if name and item["name"] == name:
                        removed = True
                        continue
                    new_items.append(item)
                request["items"] = new_items
                if not removed:
                    return ToolResult(
                        output={
                            "type": "material",
                            "status": "error",
                            "reason": f"申领单中没有 {name}",
                        }
                    )
                return await self._save_and_return(redis, key, request, action, name, 0)

            if action == "MODIFY":
                for item in request["items"]:
                    if name and item["name"] == name:
                        item["quantity"] = quantity
                        item["subtotal"] = round(item["unit_price"] * quantity, 2)
                        break
                else:
                    return ToolResult(
                        output={
                            "type": "material",
                            "status": "error",
                            "reason": f"申领单中没有 {name}",
                        }
                    )
                return await self._save_and_return(redis, key, request, action, name, quantity)

            return ToolResult(
                output={
                    "type": "material",
                    "status": "error",
                    "reason": f"不支持的操作: {action}",
                }
            )
        except (RedisError, ConnectionError, OSError, json.JSONDecodeError):
            logger.exception("Material tool execution failed")
            return ToolResult(
                output={
                    "type": "material",
                    "status": "error",
                    "reason": "申领服务暂时不可用，请稍后重试",
                }
            )

    async def _load(self, redis: aioredis.Redis, key: str) -> dict[str, Any]:
        raw = await redis.get(key)
        if raw:
            return json.loads(raw)
        return {"user_id": key.rsplit(":", 1)[-1], "items": [], "total": 0.0}

    async def _save_and_return(
        self,
        redis: aioredis.Redis,
        key: str,
        request: dict[str, Any],
        action: str,
        name: str | None,
        quantity: int,
    ) -> ToolResult:
        request["total"] = round(sum(i["subtotal"] for i in request["items"]), 2)
        request["quota"] = MONTHLY_QUOTA
        request["remaining"] = round(MONTHLY_QUOTA - request["total"], 2)
        if request["total"] > MONTHLY_QUOTA:
            return ToolResult(
                output={
                    "type": "material",
                    "status": "error",
                    "reason": (
                        f"本月物料额度 ¥{MONTHLY_QUOTA:.0f} 已超支，"
                        "需区域督导审批后方可提交"
                    ),
                    **request,
                }
            )
        await redis.setex(key, 86400, json.dumps(request, ensure_ascii=False))
        return ToolResult(
            output={
                "type": "material",
                "action": action,
                "name": name,
                "quantity": quantity,
                **request,
            }
        )
