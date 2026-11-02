# app/models/loss.py
"""门店报损单模型。

报损单是门店服务台的高风险动作：金额分级决定审批人，
系统只负责建单、分流和留痕，最终结论由人做出。
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum

from sqlalchemy import Column, DateTime, Numeric, String, Text, text
from sqlmodel import Field, SQLModel

from app.core.utils import utc_now


class LossStatus(str, Enum):
    """报损单状态"""

    PENDING = "PENDING"  # 待审批
    APPROVED = "APPROVED"  # 已通过
    REJECTED = "REJECTED"  # 已驳回
    COMPLETED = "COMPLETED"  # 已完成（已核销）
    CANCELLED = "CANCELLED"  # 已取消


class LossReason(str, Enum):
    """报损原因分类"""

    DAMAGE = "DAMAGE"  # 破损/摔坏
    EXPIRED = "EXPIRED"  # 过期/临期
    EQUIPMENT = "EQUIPMENT"  # 设备故障导致
    OTHER = "OTHER"  # 其他


class ApprovalLevel(str, Enum):
    """审批层级"""

    AUTO = "AUTO"  # 自动通过（小额）
    STORE_MANAGER = "STORE_MANAGER"  # 店长
    REGIONAL = "REGIONAL"  # 区域督导
    HQ = "HQ"  # 总部运营


class LossReport(SQLModel, table=True):
    """门店报损单"""

    __tablename__ = "loss_reports"

    id: int | None = Field(default=None, primary_key=True)
    report_sn: str = Field(unique=True, index=True, max_length=32)

    store_code: str = Field(index=True, max_length=16)
    store_name: str = Field(default="", max_length=64)
    user_id: int = Field(foreign_key="users.id", index=True, description="上报人")

    amount: Decimal = Field(
        sa_column=Column(Numeric(precision=10, scale=2)), description="报损金额"
    )
    reason_category: str = Field(
        default=LossReason.OTHER.value, sa_column=Column(String, nullable=True)
    )
    reason_detail: str = Field(sa_column=Column(Text, nullable=False))

    status: str = Field(
        default=LossStatus.PENDING.value, sa_column=Column(String, index=True, nullable=False)
    )
    approval_level: str = Field(
        default=ApprovalLevel.STORE_MANAGER.value,
        sa_column=Column(String, index=True, nullable=False),
    )

    approver_note: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    reviewed_by: int | None = Field(default=None, foreign_key="users.id")
    reviewed_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )

    created_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(
            DateTime(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP")
        ),
    )
    updated_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(
            DateTime(timezone=True),
            nullable=False,
            server_default=text("CURRENT_TIMESTAMP"),
            onupdate=text("CURRENT_TIMESTAMP"),
        ),
    )

    model_config = {"use_enum_values": True}
