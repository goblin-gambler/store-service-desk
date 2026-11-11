# scripts/seed_large_data.py
"""批量生成门店演示数据：用户、报损单、审计单与消息卡片。

脚本只清理业务流水（审计、消息、报损），不会删除用户，避免破坏其他关联表。
"""

import asyncio
import os
import random
import sys
import uuid
from datetime import datetime, timedelta
from decimal import Decimal

sys.path.append(os.getcwd())

from sqlmodel import delete, select

from app.core.database import async_session_maker
from app.models.audit import AuditAction, AuditLog, AuditTriggerType, RiskLevel
from app.models.loss import ApprovalLevel, LossReason, LossReport, LossStatus
from app.models.message import MessageCard, MessageStatus, MessageType
from app.models.user import User

# --- 配置参数 ---
USER_COUNT = 200
TOTAL_REPORTS = 500

STORES = [
    ("ST1001", "上海张江店"),
    ("ST1002", "北京中关村店"),
    ("ST1003", "广州天河店"),
    ("ST1004", "深圳福田店"),
]

REASONS = [
    (LossReason.DAMAGE, ["搬运时摔碎", "顾客碰倒", "冷柜倾倒"]),
    (LossReason.EXPIRED, ["临期未售完", "保质期管理疏漏", "批次过期"]),
    (LossReason.EQUIPMENT, ["制冰机故障", "冷柜断电", "封口机损坏"]),
    (LossReason.OTHER, ["盘点差异", "账实不符", "其他损耗"]),
]

AUTO_APPROVE_AMOUNT = 200.0
REGIONAL_APPROVAL_AMOUNT = 2000.0


def _approval_level(amount: float) -> ApprovalLevel:
    if amount < AUTO_APPROVE_AMOUNT:
        return ApprovalLevel.AUTO
    if amount <= REGIONAL_APPROVAL_AMOUNT:
        return ApprovalLevel.STORE_MANAGER
    return ApprovalLevel.REGIONAL


def _random_amount() -> float:
    tier = random.choices(["auto", "manager", "regional"], weights=[0.4, 0.45, 0.15])[0]
    if tier == "auto":
        return round(random.uniform(10, 199), 2)
    if tier == "manager":
        return round(random.uniform(200, 2000), 2)
    return round(random.uniform(2000.01, 50000), 2)


async def seed_large_data() -> None:
    async with async_session_maker() as session:
        print("清理旧业务流水...")
        await session.exec(delete(AuditLog))
        await session.exec(delete(MessageCard))
        await session.exec(delete(LossReport))
        await session.commit()

        print(f"准备 {USER_COUNT} 个用户...")
        result = await session.exec(select(User).order_by(User.id).limit(USER_COUNT))
        users = list(result.all())
        existing = {u.username for u in users}
        for i in range(1, USER_COUNT + 1):
            username = f"bulk_{i:03d}"
            if username in existing:
                continue
            user = User(
                username=username,
                email=f"{username}@example.com",
                full_name=f"批量员工_{i:03d}",
                password_hash=User.hash_password(f"bulkpass{i:03d}"),
            )
            session.add(user)
            users.append(user)
        await session.flush()
        users = [u for u in users if u.id is not None]
        if not users:
            raise RuntimeError("没有可用用户，请先执行 scripts/seed_data.py")

        print(f"生成 {TOTAL_REPORTS} 条报损单...")
        now = datetime.now()
        for i in range(1, TOTAL_REPORTS + 1):
            user = random.choice(users)
            store_code, store_name = random.choice(STORES)
            reason, details = random.choice(REASONS)
            amount = _random_amount()
            level = _approval_level(amount)
            auto = level == ApprovalLevel.AUTO
            created_at = now - timedelta(days=random.randint(0, 30), minutes=random.randint(0, 1439))

            report = LossReport(
                report_sn=f"LS{created_at:%Y%m%d}{i:05d}",
                store_code=store_code,
                store_name=store_name,
                user_id=user.id or 0,
                amount=Decimal(str(amount)),
                reason_category=reason.value,
                reason_detail=random.choice(details),
                status=(LossStatus.APPROVED if auto else LossStatus.PENDING).value,
                approval_level=level.value,
                approver_note="金额低于自动通过阈值，系统自动核销" if auto else None,
                reviewed_at=created_at if auto else None,
                created_at=created_at,
            )
            session.add(report)
            await session.flush()
            await session.refresh(report)

            if auto:
                continue

            thread_id = f"{user.id}__bulk_{uuid.uuid4().hex[:8]}"
            audit = AuditLog(
                thread_id=thread_id,
                user_id=user.id or 0,
                trigger_reason=f"报损金额 ¥{amount:.2f} 触发{level.value}",
                risk_level=(
                    RiskLevel.HIGH if level == ApprovalLevel.REGIONAL else RiskLevel.MEDIUM
                ),
                action=AuditAction.PENDING,
                trigger_type=AuditTriggerType.RISK,
                context_snapshot={
                    "subject_type": "loss_report",
                    "loss_report_id": report.id,
                    "loss_report_sn": report.report_sn,
                    "store_code": store_code,
                    "store_name": store_name,
                    "amount": amount,
                    "approval_level": level.value,
                },
            )
            session.add(audit)
            await session.flush()
            await session.refresh(audit)

            session.add(
                MessageCard(
                    thread_id=thread_id,
                    message_type=MessageType.SYSTEM,
                    status=MessageStatus.SENT,
                    content={
                        "type": "admin_notification",
                        "audit_log_id": audit.id,
                        "risk_level": audit.risk_level,
                        "message": f"新的{audit.risk_level}风险审核任务",
                    },
                    sender_type="system",
                    receiver_id=None,
                )
            )

            if i % 100 == 0:
                await session.commit()
                print(f"  已写入 {i} / {TOTAL_REPORTS}")

        await session.commit()
        print("批量门店演示数据生成完成。")


if __name__ == "__main__":
    confirm = input("该脚本会清空审计、消息与报损流水，是否继续？(y/n): ")
    if confirm.lower() == "y":
        asyncio.run(seed_large_data())
    else:
        print("操作已取消。")
