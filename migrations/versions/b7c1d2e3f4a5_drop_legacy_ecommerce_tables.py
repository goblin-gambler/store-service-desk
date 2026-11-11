"""drop legacy ecommerce tables and audit columns

Revision ID: b7c1d2e3f4a5
Revises: eda726e63e1d
Create Date: 2026-11-11

门店服务台已移除订单与退款域，对应的 ``orders``、``refund_applications``
以及审计表上的关联字段不再被任何模型引用。旧库升级时清理掉这些残留对象。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b7c1d2e3f4a5"
down_revision: str | None = "eda726e63e1d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    # 先删引用 orders 的表，避免外键约束阻塞。
    if "refund_applications" in tables:
        op.drop_table("refund_applications")
    if "orders" in tables:
        op.drop_table("orders")

    if "audit_logs" in tables:
        columns = {column["name"] for column in inspector.get_columns("audit_logs")}
        for name in ("refund_application_id", "order_id"):
            if name in columns:
                op.drop_column("audit_logs", name)

    if "complaint_tickets" in tables:
        columns = {column["name"] for column in inspector.get_columns("complaint_tickets")}
        if "order_sn" in columns:
            op.drop_column("complaint_tickets", "order_sn")


def downgrade() -> None:
    # 旧电商业务表已无模型与入口，无法从门店域数据恢复，故不提供回滚。
    pass
