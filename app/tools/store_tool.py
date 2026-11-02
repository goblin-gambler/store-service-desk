"""门店与设备工具。

- 门店信息：来自门店主数据（mock JSON）
- 设备状态：来自设备监控（mock JSON）
- 报修工单：写入 Redis，供本次会话内查询状态
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import timedelta
from typing import Any

import redis.asyncio as aioredis
from redis.exceptions import RedisError

from app.core.config import settings
from app.core.utils import utc_now
from app.models.state import AgentState
from app.tools.base import BaseTool, ToolResult
from app.tools.store_data import (
    default_store_code,
    devices_for,
    extract_store_code,
    get_store,
)

logger = logging.getLogger(__name__)

_TICKET_PATTERN = re.compile(r"(RP\d{8}[A-Z0-9]{4})", re.IGNORECASE)
_DEVICE_HINTS = {
    "POS": ["pos", "收银", "收银机", "扫码"],
    "制冰机": ["制冰", "冰块"],
    "标签打印机": ["打印机", "标签", "小票"],
    "冷藏柜": ["冷柜", "冰箱", "冷藏"],
}


class StoreTool(BaseTool):
    name = "store"
    description = "查询门店信息、设备状态，创建/查询报修工单"

    def __init__(self, redis_client: aioredis.Redis | None = None, key_prefix: str = ""):
        self._redis = redis_client
        self._key_prefix = key_prefix

    async def _get_redis(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        return self._redis

    async def execute(self, state: AgentState, **kwargs) -> ToolResult:
        _ = kwargs
        question = state.get("question", "")
        slots = state.get("slots") or {}
        store_code = extract_store_code(question, slots) or default_store_code(state.get("user_id"))

        if not store_code:
            return ToolResult(
                output={
                    "type": "error",
                    "reason": "missing_store",
                    "message": "请提供门店编号（如 ST1001）或门店名称。",
                }
            )

        store = get_store(store_code)
        if store is None:
            return ToolResult(
                output={"type": "error", "reason": "store_not_found", "store_code": store_code}
            )

        ticket_match = _TICKET_PATTERN.search(question)
        if ticket_match:
            return await self._ticket_status(ticket_match.group(1).upper(), store)

        if any(k in question for k in ("报修", "维修", "派人", "上门", "修一下", "坏了")):
            return await self._create_repair_ticket(question, store_code, store)

        devices = self._filter_devices(question, devices_for(store_code))
        if self._is_device_question(question):
            return ToolResult(
                output={"type": "device_status", "store": store, "devices": devices}
            )

        return ToolResult(
            output={
                "type": "store_info",
                "store": store,
                "devices": devices_for(store_code),
            }
        )

    @staticmethod
    def _is_device_question(question: str) -> bool:
        q = question.lower()
        return any(
            kw in q
            for kws in _DEVICE_HINTS.values()
            for kw in kws
        ) or any(k in question for k in ("设备", "报警", "故障", "离线"))

    @staticmethod
    def _filter_devices(question: str, devices: list[dict[str, Any]]) -> list[dict[str, Any]]:
        q = question.lower()
        for device_type, hints in _DEVICE_HINTS.items():
            if any(hint in q for hint in hints):
                matched = [d for d in devices if device_type in d["device_type"]]
                if matched:
                    return matched
        return devices

    async def _create_repair_ticket(
        self, question: str, store_code: str, store: dict[str, Any]
    ) -> ToolResult:
        ticket_sn = f"RP{utc_now():%Y%m%d}{uuid.uuid4().hex[:4].upper()}"
        devices = devices_for(store_code)
        abnormal = [d for d in devices if d["status"] != "ONLINE"]

        payload = {
            "ticket_sn": ticket_sn,
            "store_code": store_code,
            "store_name": store["name"],
            "issue": question,
            "device_codes": [d["device_code"] for d in abnormal] or ["未指定设备"],
            "urgency": "high" if any(k in question for k in ("停业", "无法收银", "全部")) else "medium",
            "status": "已派单",
            "engineer": "区域运维-张工",
            "created_at": utc_now().strftime("%Y-%m-%d %H:%M"),
            "eta": "4 小时内上门",
        }
        try:
            redis = await self._get_redis()
            await redis.setex(
                f"{self._key_prefix}repair_ticket:{ticket_sn}",
                int(timedelta(days=30).total_seconds()),
                json.dumps(payload, ensure_ascii=False),
            )
            await redis.rpush(f"{self._key_prefix}repair_tickets:{store_code}", ticket_sn)
            await redis.expire(
                f"{self._key_prefix}repair_tickets:{store_code}",
                int(timedelta(days=30).total_seconds()),
            )
        except (RedisError, ConnectionError, OSError):
            logger.exception("Failed to persist repair ticket")
            return ToolResult(
                output={"type": "error", "reason": "repair_service_unavailable"}
            )

        return ToolResult(output={"type": "repair_ticket", **payload})

    async def _ticket_status(self, ticket_sn: str, store: dict[str, Any]) -> ToolResult:
        try:
            redis = await self._get_redis()
            raw = await redis.get(f"{self._key_prefix}repair_ticket:{ticket_sn}")
        except (RedisError, ConnectionError, OSError):
            logger.exception("Failed to read repair ticket")
            raw = None

        if not raw:
            return ToolResult(
                output={
                    "type": "ticket_not_found",
                    "ticket_sn": ticket_sn,
                    "store": store,
                }
            )
        return ToolResult(output={"type": "ticket_status", **json.loads(raw)})
