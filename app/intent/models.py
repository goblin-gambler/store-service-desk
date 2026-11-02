"""意图识别数据模型"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.core.utils import utc_now


class IntentCategory(str, Enum):
    """一级意图：业务域（连锁零售门店服务台）"""

    STORE = "STORE"  # 门店信息与设备报修
    LOSS = "LOSS"  # 报损申请与进度
    POLICY = "POLICY"  # 运营政策（排班/请假/报损制度）
    SUPPLY = "SUPPLY"  # 配送与补货
    SETTLEMENT = "SETTLEMENT"  # 结算与对账
    PRODUCT = "PRODUCT"  # 商品与配方
    MATERIAL = "MATERIAL"  # 物料申领
    COMPLAINT = "COMPLAINT"  # 顾客投诉与舆情
    STAFF = "STAFF"  # 员工与权限
    OTHER = "OTHER"


class IntentAction(str, Enum):
    """二级意图：动作类型"""

    QUERY = "QUERY"
    APPLY = "APPLY"
    MODIFY = "MODIFY"
    CANCEL = "CANCEL"
    CONSULT = "CONSULT"
    ADD = "ADD"
    REMOVE = "REMOVE"
    COMPARE = "COMPARE"


class SlotPriority(str, Enum):
    """槽位优先级"""

    P0 = "P0"  # 必须
    P1 = "P1"  # 重要
    P2 = "P2"  # 可选


class Slot(BaseModel):
    """槽位定义"""

    name: str
    description: str
    priority: SlotPriority
    required: bool = True
    extractor: str | None = None  # 提取器名称


class IntentResult(BaseModel):
    """意图识别结果"""

    primary_intent: IntentCategory
    secondary_intent: IntentAction
    tertiary_intent: str | None = None
    confidence: float = 0.0
    slots: dict[str, Any] | None = Field(default_factory=dict)
    missing_slots: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: str | None = None
    raw_query: str = ""


class ClarificationState(BaseModel):
    """澄清状态"""

    session_id: str
    current_intent: IntentResult | None = None
    clarification_round: int = 0
    max_clarification_rounds: int = 3
    asked_slots: list[str] = Field(default_factory=list)
    collected_slots: dict[str, Any] = Field(default_factory=dict)
    pending_slot: str | None = None
    user_refused_slots: list[str] = Field(default_factory=list)
    clarification_history: list[dict] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def can_continue_clarification(self) -> bool:
        return self.clarification_round < self.max_clarification_rounds

    def increment_round(self) -> None:
        self.clarification_round += 1
        self.updated_at = utc_now()
