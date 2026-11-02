"""门店域 mock 数据加载。

真实项目里这些数据来自门店主数据 / 设备平台 / WMS / 结算系统。
演示版统一放在 data/store_mock.json，工具层只依赖这里返回的结构。
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "store_mock.json"

STORE_CODE_PATTERN = re.compile(r"(ST\d{4})", re.IGNORECASE)
_STORE_NAME_HINTS = {"张江": "ST1001", "中关村": "ST1002", "天河": "ST1003", "福田": "ST1004"}

REQUIRED_SECTIONS = ("stores", "devices", "deliveries", "settlements", "staff")


@lru_cache(maxsize=1)
def load_store_data() -> dict[str, Any]:
    raw = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
    for section in REQUIRED_SECTIONS:
        raw.setdefault(section, [])
    raw["stores_by_code"] = {store["store_code"]: store for store in raw["stores"]}
    return raw


def extract_store_code(text: str, slots: dict[str, Any] | None = None) -> str | None:
    """从问题或槽位中识别门店编号。"""
    slots = slots or {}
    for key in ("store_code", "store"):
        value = slots.get(key)
        if isinstance(value, str):
            matched = STORE_CODE_PATTERN.search(value)
            if matched:
                return matched.group(1).upper()
    matched = STORE_CODE_PATTERN.search(text)
    if matched:
        return matched.group(1).upper()
    for hint, code in _STORE_NAME_HINTS.items():
        if hint in text:
            return code
    return None


def default_store_code(user_id: int | None) -> str | None:
    """演示版按登录账号映射默认门店：user 1-4 -> ST1001-ST1004。

    真实项目应查员工-门店绑定表，而不是按 ID 取模。
    """
    if user_id is None:
        return None
    code = f"ST{1001 + (int(user_id) - 1) % 4}"
    return code if get_store(code) else None


def get_store(store_code: str | None) -> dict[str, Any] | None:
    if not store_code:
        return None
    return load_store_data()["stores_by_code"].get(store_code.upper())


def devices_for(store_code: str) -> list[dict[str, Any]]:
    return [d for d in load_store_data()["devices"] if d["store_code"] == store_code]


def deliveries_for(store_code: str) -> list[dict[str, Any]]:
    return [d for d in load_store_data()["deliveries"] if d["store_code"] == store_code]


def settlements_for(store_code: str) -> list[dict[str, Any]]:
    return [s for s in load_store_data()["settlements"] if s["store_code"] == store_code]


def staff_for(store_code: str) -> list[dict[str, Any]]:
    return [s for s in load_store_data()["staff"] if s["store_code"] == store_code]


def find_staff(employee_no: str | None) -> dict[str, Any] | None:
    if not employee_no:
        return None
    target = employee_no.upper()
    return next(
        (s for s in load_store_data()["staff"] if s["employee_no"].upper() == target), None
    )
