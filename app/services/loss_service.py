"""门店报损服务。

核心原则：金额分级审批写死在代码里，不交给模型判断。
- 单笔 < 200 元：自动通过
- 200 ~ 2000 元：挂起，店长审批
- > 2000 元：挂起，区域督导审批

系统只做建单、分流、留痕；"通过/驳回"永远由人决定。
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import desc, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.config import settings
from app.core.database import async_session_maker
from app.core.utils import utc_now
from app.models.audit import AuditAction, AuditLog, AuditTriggerType, RiskLevel
from app.models.loss import ApprovalLevel, LossReason, LossReport, LossStatus
from app.models.state import AgentProcessResult
from app.tools.store_data import default_store_code, extract_store_code, get_store

logger = logging.getLogger(__name__)

_AMOUNT_PATTERN = re.compile(r"(?:¥|￥)?\s*(\d+(?:\.\d{1,2})?)\s*(?:元|块|rmb)?", re.IGNORECASE)

APPROVAL_LABELS = {
    ApprovalLevel.AUTO.value: "自动通过",
    ApprovalLevel.STORE_MANAGER.value: "店长审批",
    ApprovalLevel.REGIONAL.value: "区域督导审批",
    ApprovalLevel.HQ.value: "总部审批",
}

STATUS_LABELS = {
    LossStatus.PENDING.value: "待审批",
    LossStatus.APPROVED.value: "已通过",
    LossStatus.REJECTED.value: "已驳回",
    LossStatus.COMPLETED.value: "已完成",
    LossStatus.CANCELLED.value: "已取消",
}

REASON_LABELS = {
    LossReason.DAMAGE.value: "破损",
    LossReason.EXPIRED.value: "过期/临期",
    LossReason.EQUIPMENT.value: "设备故障",
    LossReason.OTHER.value: "其他",
}


def extract_amount(text: str, slots: dict[str, Any] | None = None) -> float | None:
    """从问题或槽位中识别报损金额。"""
    slots = slots or {}
    for key in ("amount", "loss_amount"):
        value = slots.get(key)
        if isinstance(value, (int, float, Decimal)):
            return float(value)
        if isinstance(value, str):
            matched = _AMOUNT_PATTERN.search(value)
            if matched:
                return float(matched.group(1))
    # 优先取带"元/块/¥"的金额，避免把"3 箱"误当成金额
    for matched in re.finditer(r"(?:¥|￥)\s*(\d+(?:\.\d{1,2})?)|(\d+(?:\.\d{1,2})?)\s*(?:元|块)", text):
        value = matched.group(1) or matched.group(2)
        if value:
            return float(value)
    return None


def classify_loss_reason(text: str) -> LossReason:
    if any(k in text for k in ("过期", "临期", "保质期", "变质")):
        return LossReason.EXPIRED
    if any(k in text for k in ("设备", "机器", "故障", "跳闸", "断电", "制冰机", "冷柜")):
        return LossReason.EQUIPMENT
    if any(k in text for k in ("破损", "摔", "打翻", "漏", "碎", "压坏")):
        return LossReason.DAMAGE
    return LossReason.OTHER


def resolve_approval_level(amount: float) -> ApprovalLevel:
    if amount < settings.LOSS_AUTO_APPROVE_AMOUNT:
        return ApprovalLevel.AUTO
    if amount <= settings.LOSS_REGIONAL_APPROVAL_AMOUNT:
        return ApprovalLevel.STORE_MANAGER
    return ApprovalLevel.REGIONAL


def _generate_report_sn() -> str:
    return f"LS{utc_now():%Y%m%d}{uuid.uuid4().hex[:4].upper()}"


class LossService:
    """报损单的建单、查询与审批留痕。"""

    async def submit_loss_report(
        self,
        question: str,
        user_id: int,
        thread_id: str = "",
        slots: dict[str, Any] | None = None,
        session: AsyncSession | None = None,
    ) -> AgentProcessResult:
        try:
            if session is None:
                async with async_session_maker() as own_session:
                    return await self._submit(
                        question, user_id, thread_id, slots or {}, own_session
                    )
            return await self._submit(question, user_id, thread_id, slots or {}, session)
        except SQLAlchemyError:
            logger.exception("[LossService] Database error while submitting loss report")
            return {
                "response": "报损单提交失败（数据库暂时不可用），请稍后重试或联系店长手工登记。",
                "updated_state": {"loss_data": None, "loss_flow_active": False},
            }

    async def _submit(
        self,
        question: str,
        user_id: int,
        thread_id: str,
        slots: dict[str, Any],
        session: AsyncSession,
    ) -> AgentProcessResult:
        store_code = extract_store_code(question, slots) or default_store_code(user_id)
        if not store_code:
            return {
                "response": (
                    "提交报损需要先确认门店。请提供门店编号（如 ST1001），"
                    "或直接说明门店名称（如“张江店”）。"
                ),
                "updated_state": {"loss_flow_active": False},
            }

        store_name = self._store_name(store_code)
        if store_name is None:
            return {
                "response": (
                    f"未找到门店 {store_code}。当前系统支持的门店："
                    "ST1001（上海张江店）、ST1002（北京中关村店）、"
                    "ST1003（广州天河店）、ST1004（深圳福田店）。"
                ),
                "updated_state": {"loss_flow_active": False},
            }

        amount = extract_amount(question, slots)
        if amount is None or amount <= 0:
            return {
                "response": (
                    "请补充本次报损的预估金额（单位：元），例如“损失约 350 元”。"
                    "金额决定审批层级：200 元以下自动通过，200–2000 元由店长审批，"
                    "2000 元以上需区域督导审批。"
                ),
                "updated_state": {"loss_flow_active": False},
            }

        if amount > 100000:
            return {
                "response": "单笔报损金额超过 10 万元，超出线上流程范围，请联系区域督导走专项审批。",
                "updated_state": {"loss_flow_active": False},
            }

        reason = classify_loss_reason(question)
        approval_level = resolve_approval_level(amount)
        auto_approved = approval_level == ApprovalLevel.AUTO
        status = LossStatus.APPROVED if auto_approved else LossStatus.PENDING

        report = LossReport(
            report_sn=_generate_report_sn(),
            store_code=store_code,
            store_name=store_name,
            user_id=user_id,
            amount=Decimal(str(round(amount, 2))),
            reason_category=reason.value,
            reason_detail=question,
            status=status.value,
            approval_level=approval_level.value,
            approver_note="金额低于自动通过阈值，系统自动核销" if auto_approved else None,
            reviewed_at=utc_now() if auto_approved else None,
        )
        session.add(report)
        await session.flush()
        await session.refresh(report)

        audit_id: int | None = None
        if not auto_approved:
            audit = AuditLog(
                user_id=user_id,
                thread_id=thread_id,
                action=AuditAction.PENDING,
                trigger_type=AuditTriggerType.RISK,
                risk_level=(
                    RiskLevel.HIGH
                    if approval_level == ApprovalLevel.REGIONAL
                    else RiskLevel.MEDIUM
                ),
                trigger_reason=(
                    f"报损金额 ¥{amount:.2f} 触发{APPROVAL_LABELS[approval_level.value]}"
                ),
                context_snapshot={
                    "subject_type": "loss_report",
                    "question": question,
                    "loss_report_id": report.id,
                    "loss_report_sn": report.report_sn,
                    "store_code": store_code,
                    "store_name": store_name,
                    "amount": amount,
                    "approval_level": approval_level.value,
                },
            )
            session.add(audit)
            await session.flush()
            await session.refresh(audit)
            audit_id = audit.id

        await session.commit()

        if audit_id is not None:
            self._notify_admin(audit_id)

        loss_data = {
            "report_id": report.id,
            "report_sn": report.report_sn,
            "store_code": store_code,
            "store_name": store_name,
            "amount": float(report.amount),
            "reason": REASON_LABELS[reason.value],
            "status": status.value,
            "approval_level": approval_level.value,
            "audit_id": audit_id,
        }
        return {
            "response": self._format_submit_response(loss_data),
            "updated_state": {"loss_data": loss_data, "loss_flow_active": True},
        }

    async def list_recent_reports(self, user_id: int, limit: int = 3) -> list[dict[str, Any]]:
        async with async_session_maker() as session:
            stmt = (
                select(LossReport)
                .where(LossReport.user_id == user_id)
                .order_by(desc(LossReport.created_at))
                .limit(limit)
            )
            result = await session.exec(stmt)
            return [self._serialize(report) for report in result.all()]

    async def get_by_sn(self, report_sn: str, user_id: int) -> dict[str, Any] | None:
        async with async_session_maker() as session:
            stmt = select(LossReport).where(
                LossReport.report_sn == report_sn, LossReport.user_id == user_id
            )
            result = await session.exec(stmt)
            report = result.one_or_none()
            return self._serialize(report) if report else None

    @staticmethod
    def _serialize(report: LossReport) -> dict[str, Any]:
        return {
            "report_id": report.id,
            "report_sn": report.report_sn,
            "store_code": report.store_code,
            "store_name": report.store_name,
            "amount": float(report.amount),
            "reason": REASON_LABELS.get(report.reason_category, report.reason_category),
            "status": report.status,
            "approval_level": report.approval_level,
            "approver_note": report.approver_note,
            "created_at": report.created_at.strftime("%Y-%m-%d %H:%M")
            if isinstance(report.created_at, datetime)
            else str(report.created_at),
        }

    @staticmethod
    def _store_name(store_code: str) -> str | None:
        store = get_store(store_code)
        return store["name"] if store else None

    @staticmethod
    def _notify_admin(audit_id: int) -> None:
        try:
            from app.tasks.refund_tasks import notify_admin_audit

            notify_admin_audit.delay(audit_id)
        except Exception:  # noqa: BLE001 - 通知失败不能阻塞业务主流程
            logger.warning("Failed to enqueue admin audit notification for audit_id=%s", audit_id)

    @staticmethod
    def _format_submit_response(data: dict[str, Any]) -> str:
        level = data["approval_level"]
        if level == ApprovalLevel.AUTO.value:
            status_line = "状态: 已自动通过（金额低于 200 元，无需人工审批）"
        elif level == ApprovalLevel.STORE_MANAGER.value:
            status_line = "状态: 待店长审批（200–2000 元），审批结果会同步到会话"
        else:
            status_line = "状态: 待区域督导审批（超过 2000 元），审批结果会同步到会话"
        return (
            "🧾 报损单已提交\n"
            f"单号: {data['report_sn']}\n"
            f"门店: {data['store_name']}（{data['store_code']}）\n"
            f"金额: ¥{data['amount']:.2f}\n"
            f"原因: {data['reason']}\n"
            f"{status_line}\n"
            "提示: 请保留实物照片，审批人可能会要求补充现场凭证。"
        )
