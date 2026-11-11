"""审批通知异步任务。

门店域保留的 Celery 任务：`store.notify_admin_audit`。
"""

import logging
from typing import Any

from sqlmodel import Session, select

from app.celery_app import celery_app
from app.core.database import sync_session_maker
from app.core.utils import utc_now
from app.models.audit import AuditLog
from app.models.message import MessageCard, MessageStatus, MessageType

logger = logging.getLogger(__name__)


def _notify_admin_audit_body(self, audit_log_id: int, session: Session) -> dict[str, Any]:
    result = session.exec(select(AuditLog).where(AuditLog.id == audit_log_id))
    audit_log = result.one_or_none()

    if not audit_log:
        exc = ValueError(f"Audit log {audit_log_id} not found")
        try:
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError as _err:
            raise exc from _err

    logger.info("  [Notify] 通知管理员审核任务:")
    logger.info(f"  - 风险等级: {audit_log.risk_level}")
    logger.info(f"  - 触发原因: {audit_log.trigger_reason}")
    logger.info(f"  - 用户ID: {audit_log.user_id}")

    message = MessageCard(
        thread_id=audit_log.thread_id,
        message_type=MessageType.SYSTEM,
        status=MessageStatus.SENT,
        content={
            "type": "admin_notification",
            "audit_log_id": audit_log_id,
            "risk_level": audit_log.risk_level,
            "message": f"新的{audit_log.risk_level}风险审核任务",
        },
        sender_type="system",
        receiver_id=None,
    )
    session.add(message)
    session.commit()

    return {
        "status": "success",
        "audit_log_id": audit_log_id,
        "notified_at": utc_now().isoformat(),
    }


@celery_app.task(bind=True, name="store.notify_admin_audit", max_retries=2)
def notify_admin_audit(self, audit_log_id: int, session=None) -> dict[str, Any]:
    if session is None:
        with sync_session_maker() as session:
            return _notify_admin_audit_body(self, audit_log_id, session)
    return _notify_admin_audit_body(self, audit_log_id, session)
