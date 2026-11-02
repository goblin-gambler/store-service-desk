import uuid
from decimal import Decimal

import pytest
from sqlmodel import Session, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.database import async_engine, sync_engine
from app.models.audit import AuditAction, AuditLog, AuditTriggerType, RiskLevel
from app.models.loss import LossReport, LossStatus
from app.models.message import MessageCard, MessageType
from app.models.user import User
from app.schemas.admin import TaskStatsResponse
from app.services.admin_service import AdminService, AuditAlreadyProcessedError, AuditNotFoundError
from app.websocket.manager import ConnectionManager


def _create_committed_user():
    with sync_engine.connect() as conn:
        session = Session(bind=conn)
        try:
            user = User(
                username=f"admin_test_user_{uuid.uuid4().hex[:8]}",
                password_hash=User.hash_password("testpass"),
                email=f"{uuid.uuid4().hex[:8]}@test.com",
                full_name="Test User",
                phone="13800138000",
            )
            session.add(user)
            session.commit()
            session.refresh(user)
            assert user.id is not None
            return user
        finally:
            session.close()


def _create_committed_admin_user():
    with sync_engine.connect() as conn:
        session = Session(bind=conn)
        try:
            user = User(
                username=f"admin_test_admin_{uuid.uuid4().hex[:8]}",
                password_hash=User.hash_password("adminpass"),
                email=f"{uuid.uuid4().hex[:8]}@admin.com",
                full_name="Test Admin",
                phone="13800138001",
                is_admin=True,
                is_active=True,
            )
            session.add(user)
            session.commit()
            session.refresh(user)
            assert user.id is not None
            return user
        finally:
            session.close()


def _create_committed_loss_report(user_id: int):
    with sync_engine.connect() as conn:
        session = Session(bind=conn)
        try:
            report = LossReport(
                report_sn=f"LS{uuid.uuid4().hex[:12].upper()}",
                store_code="ST1001",
                store_name="Test Store",
                user_id=user_id,
                amount=Decimal("350.00"),
                reason_category="DAMAGE",
                reason_detail="Test loss report",
                approval_level="STORE_MANAGER",
                status=LossStatus.PENDING.value,
            )
            session.add(report)
            session.commit()
            session.refresh(report)
            assert report.id is not None
            return report
        finally:
            session.close()


class TestProcessAdminDecision:
    @pytest.mark.asyncio
    async def test_approve_with_loss_report(self):
        user = _create_committed_user()
        admin = _create_committed_admin_user()
        loss_report = _create_committed_loss_report(user.id)

        async with async_engine.connect() as conn:
            session = AsyncSession(bind=conn, expire_on_commit=False)
            try:
                audit_log = AuditLog(
                    thread_id=f"{user.id}__thread",
                    user_id=user.id,
                    action=AuditAction.PENDING,
                    trigger_type=AuditTriggerType.RISK,
                    risk_level=RiskLevel.HIGH,
                    trigger_reason="risk",
                    context_snapshot={
                        "subject_type": "loss_report",
                        "loss_report_id": loss_report.id,
                        "loss_report_sn": loss_report.report_sn,
                    },
                )
                session.add(audit_log)
                await session.flush()
                await session.refresh(audit_log)
                assert audit_log.id is not None

                manager = ConnectionManager()
                service = AdminService(manager=manager)
                result = await service.process_admin_decision(
                    session,
                    audit_log_id=audit_log.id,
                    action="APPROVE",
                    admin_comment="Approved",
                    current_admin_id=admin.id,
                )

                assert result.success is True
                assert result.action == "APPROVE"
                assert result.audit_log_id == audit_log.id
                assert "审核决策已提交" in result.message

                await session.refresh(audit_log)
                assert audit_log.action == AuditAction.APPROVE
                assert audit_log.admin_id == admin.id
                assert audit_log.admin_comment == "Approved"
                assert audit_log.reviewed_at is not None

                loss_result = await session.exec(
                    select(LossReport).where(LossReport.id == loss_report.id)
                )
                db_loss = loss_result.one_or_none()
                assert db_loss is not None
                assert db_loss.status == LossStatus.APPROVED.value
                assert db_loss.reviewed_by == admin.id
                assert db_loss.reviewed_at is not None
                assert db_loss.approver_note == "Approved"

                message_result = await session.exec(
                    select(MessageCard).where(
                        MessageCard.thread_id == audit_log.thread_id,
                        MessageCard.message_type == MessageType.AUDIT_CARD,
                    )
                )
                message = message_result.one_or_none()
                assert message is not None
                assert message.content["card_type"] == "audit_result"
                assert message.content["action"] == "APPROVE"
                assert loss_report.report_sn in message.content["message"]
            finally:
                await session.close()

    @pytest.mark.asyncio
    async def test_reject_with_loss_report(self, db_session):
        user = _create_committed_user()
        admin = _create_committed_admin_user()
        loss_report = _create_committed_loss_report(user.id)

        audit_log = AuditLog(
            thread_id=f"{user.id}__thread",
            user_id=user.id,
            action=AuditAction.PENDING,
            trigger_type=AuditTriggerType.RISK,
            risk_level=RiskLevel.HIGH,
            trigger_reason="risk",
            context_snapshot={
                "subject_type": "loss_report",
                "loss_report_id": loss_report.id,
                "loss_report_sn": loss_report.report_sn,
            },
        )
        db_session.add(audit_log)
        await db_session.flush()
        await db_session.refresh(audit_log)
        assert audit_log.id is not None

        manager = ConnectionManager()
        service = AdminService(manager=manager)
        result = await service.process_admin_decision(
            db_session,
            audit_log_id=audit_log.id,
            action="REJECT",
            admin_comment="Rejected",
            current_admin_id=admin.id,
        )

        assert result.success is True
        assert result.action == "REJECT"

        await db_session.refresh(audit_log)
        assert audit_log.action == AuditAction.REJECT
        assert audit_log.admin_id == admin.id
        assert audit_log.admin_comment == "Rejected"

        loss_result = await db_session.exec(
            select(LossReport).where(LossReport.id == loss_report.id)
        )
        db_loss = loss_result.one_or_none()
        assert db_loss is not None
        assert db_loss.status == LossStatus.REJECTED.value
        assert db_loss.reviewed_by == admin.id
        assert db_loss.approver_note == "Rejected"

    @pytest.mark.asyncio
    async def test_404_for_missing_audit_log(self, db_session):
        admin = _create_committed_admin_user()
        service = AdminService(manager=None)
        with pytest.raises(AuditNotFoundError):
            await service.process_admin_decision(
                db_session,
                audit_log_id=999999,
                action="APPROVE",
                admin_comment=None,
                current_admin_id=admin.id,
            )

    @pytest.mark.asyncio
    async def test_400_for_already_processed(self, db_session):
        admin = _create_committed_admin_user()
        user = User(
            username=f"admin_test_user_{uuid.uuid4().hex[:8]}",
            password_hash=User.hash_password("testpass"),
            email=f"{uuid.uuid4().hex[:8]}@test.com",
            full_name="Test User",
        )
        db_session.add(user)
        await db_session.flush()
        await db_session.refresh(user)
        assert user.id is not None

        audit_log = AuditLog(
            thread_id=f"{user.id}__thread",
            user_id=user.id,
            action=AuditAction.APPROVE,
            trigger_type=AuditTriggerType.RISK,
            risk_level=RiskLevel.HIGH,
            trigger_reason="risk",
            context_snapshot={},
        )
        db_session.add(audit_log)
        await db_session.flush()
        await db_session.refresh(audit_log)
        assert audit_log.id is not None

        service = AdminService(manager=None)
        with pytest.raises(AuditAlreadyProcessedError):
            await service.process_admin_decision(
                db_session,
                audit_log_id=audit_log.id,
                action="REJECT",
                admin_comment=None,
                current_admin_id=admin.id,
            )


class TestQueryMethods:
    @pytest.mark.asyncio
    async def test_get_pending_tasks_without_filter(self, db_session):
        service = AdminService(manager=None)
        before_tasks = await service.get_pending_tasks(db_session)

        user1 = User(
            username=f"admin_test_user_{uuid.uuid4().hex[:8]}",
            password_hash=User.hash_password("testpass"),
            email=f"{uuid.uuid4().hex[:8]}@test.com",
            full_name="Test User 1",
        )
        user2 = User(
            username=f"admin_test_user_{uuid.uuid4().hex[:8]}",
            password_hash=User.hash_password("testpass"),
            email=f"{uuid.uuid4().hex[:8]}@test.com",
            full_name="Test User 2",
        )
        db_session.add(user1)
        db_session.add(user2)
        await db_session.flush()
        await db_session.refresh(user1)
        assert user1.id is not None
        await db_session.refresh(user2)
        assert user2.id is not None

        log1 = AuditLog(
            thread_id=f"{user1.id}__t1",
            user_id=user1.id,
            action=AuditAction.PENDING,
            trigger_type=AuditTriggerType.RISK,
            risk_level=RiskLevel.HIGH,
            trigger_reason="reason1",
            context_snapshot={},
        )
        log2 = AuditLog(
            thread_id=f"{user2.id}__t2",
            user_id=user2.id,
            action=AuditAction.PENDING,
            trigger_type=AuditTriggerType.RISK,
            risk_level=RiskLevel.MEDIUM,
            trigger_reason="reason2",
            context_snapshot={},
        )
        db_session.add(log1)
        db_session.add(log2)
        await db_session.flush()

        after_tasks = await service.get_pending_tasks(db_session)

        assert len(after_tasks) == len(before_tasks) + 2
        ids = {t.audit_log_id for t in after_tasks}
        assert log1.id in ids
        assert log2.id in ids

    @pytest.mark.asyncio
    async def test_get_pending_tasks_with_risk_level_filter(self, db_session):
        service = AdminService(manager=None)
        before_tasks = await service.get_pending_tasks(db_session, risk_level="HIGH")

        user = User(
            username=f"admin_test_user_{uuid.uuid4().hex[:8]}",
            password_hash=User.hash_password("testpass"),
            email=f"{uuid.uuid4().hex[:8]}@test.com",
            full_name="Test User",
        )
        db_session.add(user)
        await db_session.flush()
        await db_session.refresh(user)
        assert user.id is not None

        log = AuditLog(
            thread_id=f"{user.id}__t1",
            user_id=user.id,
            action=AuditAction.PENDING,
            trigger_type=AuditTriggerType.RISK,
            risk_level=RiskLevel.HIGH,
            trigger_reason="reason",
            context_snapshot={},
        )
        db_session.add(log)
        await db_session.flush()

        after_tasks = await service.get_pending_tasks(db_session, risk_level="HIGH")

        assert len(after_tasks) == len(before_tasks) + 1
        ids = {t.audit_log_id for t in after_tasks}
        assert log.id in ids
        assert any(t.risk_level == "HIGH" for t in after_tasks)

    @pytest.mark.asyncio
    async def test_get_confidence_pending_tasks(self, db_session):
        service = AdminService(manager=None)
        before_tasks = await service.get_confidence_pending_tasks(db_session)

        user = User(
            username=f"admin_test_user_{uuid.uuid4().hex[:8]}",
            password_hash=User.hash_password("testpass"),
            email=f"{uuid.uuid4().hex[:8]}@test.com",
            full_name="Test User",
        )
        db_session.add(user)
        await db_session.flush()
        await db_session.refresh(user)
        assert user.id is not None

        log = AuditLog(
            thread_id=f"{user.id}__t1",
            user_id=user.id,
            action=AuditAction.PENDING,
            trigger_type=AuditTriggerType.CONFIDENCE,
            risk_level=RiskLevel.LOW,
            trigger_reason="reason",
            confidence_metadata={"confidence_score": 0.45},
            context_snapshot={},
        )
        db_session.add(log)
        await db_session.flush()

        after_tasks = await service.get_confidence_pending_tasks(db_session)

        assert len(after_tasks) == len(before_tasks) + 1
        ids = {t.audit_log_id for t in after_tasks}
        assert log.id in ids
        assert any("0.45" in t.trigger_reason for t in after_tasks)

    @pytest.mark.asyncio
    async def test_get_all_pending_tasks(self, db_session):
        service = AdminService(manager=None)
        before_stats = await service.get_all_pending_tasks(db_session)

        user = User(
            username=f"admin_test_user_{uuid.uuid4().hex[:8]}",
            password_hash=User.hash_password("testpass"),
            email=f"{uuid.uuid4().hex[:8]}@test.com",
            full_name="Test User",
        )
        db_session.add(user)
        await db_session.flush()
        await db_session.refresh(user)
        assert user.id is not None

        for i in range(2):
            db_session.add(
                AuditLog(
                    thread_id=f"{user.id}__risk_{i}",
                    user_id=user.id,
                    action=AuditAction.PENDING,
                    trigger_type=AuditTriggerType.RISK,
                    risk_level=RiskLevel.HIGH,
                    trigger_reason="risk",
                    context_snapshot={},
                )
            )
        for i in range(3):
            db_session.add(
                AuditLog(
                    thread_id=f"{user.id}__conf_{i}",
                    user_id=user.id,
                    action=AuditAction.PENDING,
                    trigger_type=AuditTriggerType.CONFIDENCE,
                    risk_level=RiskLevel.LOW,
                    trigger_reason="confidence",
                    confidence_metadata={"confidence_score": 0.4},
                    context_snapshot={},
                )
            )
        db_session.add(
            AuditLog(
                thread_id=f"{user.id}__manual",
                user_id=user.id,
                action=AuditAction.PENDING,
                trigger_type=AuditTriggerType.MANUAL,
                risk_level=RiskLevel.MEDIUM,
                trigger_reason="manual",
                context_snapshot={},
            )
        )
        await db_session.flush()

        after_stats = await service.get_all_pending_tasks(db_session)

        assert isinstance(after_stats, TaskStatsResponse)
        assert after_stats.risk_tasks == before_stats.risk_tasks + 2
        assert after_stats.confidence_tasks == before_stats.confidence_tasks + 3
        assert after_stats.manual_tasks == before_stats.manual_tasks + 1
        assert after_stats.total == before_stats.total + 6
