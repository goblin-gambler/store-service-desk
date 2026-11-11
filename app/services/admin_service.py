import logging
from collections.abc import Sequence

from sqlmodel import desc, func, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.utils import utc_now
from app.models.audit import AuditAction, AuditLog, AuditTriggerType
from app.models.loss import LossReport, LossStatus
from app.models.message import MessageCard, MessageStatus, MessageType
from app.models.user import User
from app.schemas.admin import AdminDecisionResponse, AuditTask, TaskStatsResponse
from app.websocket.manager import ConnectionManager

logger = logging.getLogger(__name__)


class AuditNotFoundError(Exception): ...


class AuditAlreadyProcessedError(Exception): ...


class AdminService:
    def __init__(self, manager: ConnectionManager | None = None):
        self.manager = manager

    async def get_pending_tasks(
        self, session: AsyncSession, risk_level: str | None = None
    ) -> list[AuditTask]:
        """获取待审核任务列表"""
        stmt = (
            select(AuditLog)
            .where(AuditLog.action == AuditAction.PENDING)
            .order_by(desc(AuditLog.created_at))
        )

        if risk_level:
            stmt = stmt.where(AuditLog.risk_level == risk_level)

        result = await session.exec(stmt)
        audit_logs = result.all()

        return _build_audit_tasks(audit_logs)

    async def get_confidence_pending_tasks(self, session: AsyncSession) -> list[AuditTask]:
        """获取置信度触发的待审核任务"""
        stmt = (
            select(AuditLog)
            .where(
                AuditLog.action == AuditAction.PENDING,
                AuditLog.trigger_type == AuditTriggerType.CONFIDENCE,
            )
            .order_by(desc(AuditLog.created_at))
        )

        result = await session.exec(stmt)
        audit_logs = result.all()

        tasks = []
        for log in audit_logs:
            confidence_meta = log.confidence_metadata or {}
            tasks.append(
                _build_audit_task(
                    log,
                    trigger_reason=f"置信度不足: {confidence_meta.get('confidence_score', 0):.2f}",
                )
            )
        return tasks

    async def get_all_pending_tasks(self, session: AsyncSession) -> TaskStatsResponse:
        """获取所有待审核任务（风险 + 置信度 + 手动）"""
        risk_count = await _count_pending_by_trigger(session, AuditTriggerType.RISK)
        conf_count = await _count_pending_by_trigger(session, AuditTriggerType.CONFIDENCE)
        manual_count = await _count_pending_by_trigger(session, AuditTriggerType.MANUAL)

        return TaskStatsResponse(
            risk_tasks=risk_count,
            confidence_tasks=conf_count,
            manual_tasks=manual_count,
            total=risk_count + conf_count + manual_count,
        )

    async def process_admin_decision(
        self,
        session: AsyncSession,
        audit_log_id: int,
        action: str,
        admin_comment: str | None,
        current_admin_id: int,
    ) -> AdminDecisionResponse:
        """处理管理员决策"""
        result = await session.exec(
            select(AuditLog).where(AuditLog.id == audit_log_id).with_for_update()
        )
        audit_log = result.one_or_none()

        if not audit_log:
            raise AuditNotFoundError()

        if audit_log.action != AuditAction.PENDING:
            raise AuditAlreadyProcessedError()

        action_enum = AuditAction.APPROVE if action == "APPROVE" else AuditAction.REJECT
        audit_log.action = action_enum
        audit_log.admin_id = current_admin_id
        audit_log.admin_comment = admin_comment
        audit_log.reviewed_at = utc_now()

        session.add(audit_log)

        user_result = await session.exec(select(User).where(User.id == audit_log.user_id))
        user = user_result.one_or_none()  # noqa: F841 - 保留以备后续通知扩展

        # 报损单审批：审批结论只落在 LossReport 上，不触发任何资金动作
        loss_report_id = (audit_log.context_snapshot or {}).get("loss_report_id")
        loss_sn: str | None = None
        if loss_report_id:
            loss_result = await session.exec(
                select(LossReport).where(LossReport.id == loss_report_id)
            )
            loss_report = loss_result.one_or_none()

            if loss_report:
                loss_sn = loss_report.report_sn
                loss_report.status = (
                    LossStatus.APPROVED
                    if action_enum == AuditAction.APPROVE
                    else LossStatus.REJECTED
                )
                loss_report.approver_note = admin_comment
                loss_report.reviewed_by = current_admin_id
                loss_report.reviewed_at = utc_now()
                session.add(loss_report)

        status_message = (
            f" 报损单 {loss_sn} 已通过审批"
            if action_enum == AuditAction.APPROVE
            else f" 报损单 {loss_sn or ''} 未通过审批: {admin_comment}"
        )

        message_card = MessageCard(
            thread_id=audit_log.thread_id,
            message_type=MessageType.AUDIT_CARD,
            status=MessageStatus.SENT,
            content={
                "card_type": "audit_result",
                "action": action,
                "message": status_message,
                "admin_comment": admin_comment,
                "timestamp": utc_now().isoformat(),
            },
            sender_type="admin",
            sender_id=current_admin_id,
            receiver_id=audit_log.user_id,
        )
        session.add(message_card)

        await session.commit()

        if self.manager is not None:
            try:
                await self.manager.notify_status_change(
                    thread_id=audit_log.thread_id,
                    status=action,
                    data={
                        "message": status_message,
                        "admin_comment": admin_comment,
                    },
                )
            except (RuntimeError, ConnectionError):
                logger.exception("Failed to notify status change via WebSocket")

        return AdminDecisionResponse(
            success=True,
            message=f"审核决策已提交:  {action}",
            audit_log_id=audit_log_id,
            action=action,
        )


def _build_audit_task(log: AuditLog, trigger_reason: str | None = None) -> AuditTask:
    if log.id is None:
        raise RuntimeError("Audit log ID is missing after creation")
    return AuditTask(
        audit_log_id=log.id,
        thread_id=log.thread_id,
        user_id=log.user_id,
        trigger_reason=trigger_reason if trigger_reason is not None else log.trigger_reason,
        risk_level=log.risk_level,
        context_snapshot=log.context_snapshot,
        created_at=log.created_at.isoformat(),
    )


def _build_audit_tasks(audit_logs: Sequence[AuditLog]) -> list[AuditTask]:
    return [_build_audit_task(log) for log in audit_logs]


async def _count_pending_by_trigger(session: AsyncSession, trigger_type: AuditTriggerType) -> int:
    stmt = (
        select(func.count())
        .select_from(AuditLog)
        .where(AuditLog.action == AuditAction.PENDING, AuditLog.trigger_type == trigger_type)
    )
    result = await session.exec(stmt)
    return result.one()
