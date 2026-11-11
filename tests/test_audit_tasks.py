import uuid

import pytest
from sqlmodel import select

from app.models.audit import AuditAction, AuditLog, AuditTriggerType, RiskLevel
from app.models.message import MessageCard, MessageType
from app.models.user import User
from app.tasks.audit_tasks import notify_admin_audit


class TestNotifyAdminAudit:
    def test_notify_admin_audit_success(self, db_sync_session):
        user = User(
            username=f"audit_user_{uuid.uuid4().hex[:8]}",
            password_hash=User.hash_password("testpass"),
            email=f"{uuid.uuid4().hex[:8]}@test.com",
            full_name="Test User",
        )
        db_sync_session.add(user)
        db_sync_session.commit()
        db_sync_session.refresh(user)
        assert user.id is not None

        log = AuditLog(
            thread_id=f"thread-{uuid.uuid4().hex[:8]}",
            user_id=user.id,
            trigger_reason="金额过大",
            risk_level=RiskLevel.HIGH,
            action=AuditAction.PENDING,
            trigger_type=AuditTriggerType.RISK,
            context_snapshot={},
        )
        db_sync_session.add(log)
        db_sync_session.commit()
        db_sync_session.refresh(log)

        result = notify_admin_audit.run(log.id, session=db_sync_session)

        assert result["status"] == "success"
        assert result["audit_log_id"] == log.id

        message = db_sync_session.exec(
            select(MessageCard).where(MessageCard.thread_id == log.thread_id)
        ).one_or_none()
        assert message is not None
        assert message.message_type == MessageType.SYSTEM
        assert message.content["type"] == "admin_notification"
        assert message.content["risk_level"] == RiskLevel.HIGH


def test_notify_admin_audit_not_found(db_sync_session):
    with pytest.raises(ValueError, match="Audit log 999 not found"):
        notify_admin_audit.run(999, session=db_sync_session)
