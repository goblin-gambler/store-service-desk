# app/core/database.py
import sys
from collections.abc import AsyncGenerator, Generator

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from sqlmodel import Session
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.config import settings

_async_connect_args = {
    "timeout": settings.DB_CONNECT_TIMEOUT,
    "server_settings": {"application_name": "store-service-desk", "jit": "off"},
}

_sync_connect_args = {
    "connect_timeout": settings.DB_CONNECT_TIMEOUT,
    "options": "-c application_name=store-service-desk -c jit=off",
}

# Celery 任务通过 async_to_sync 在各自的事件循环里跑；池化连接一旦跨事件循环
# 复用，asyncpg 会抛 "'NoneType' object has no attribute 'send'"。worker 进程
# 里改用 NullPool，每次任务新建连接，用完即关。
_in_celery_worker = "celery" in " ".join(sys.argv[:2]).lower()

_async_pool_kwargs = (
    {"poolclass": NullPool}
    if _in_celery_worker
    else {
        "pool_pre_ping": True,
        "pool_size": settings.DB_POOL_SIZE,
        "max_overflow": settings.DB_MAX_OVERFLOW,
        "pool_recycle": settings.DB_POOL_RECYCLE,
        "pool_timeout": settings.DB_POOL_TIMEOUT,
    }
)

# 异步引擎与 Session 工厂（FastAPI / Agent 使用）
async_engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    future=True,
    connect_args=_async_connect_args,
    **_async_pool_kwargs,
)
async_session_maker = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

# 同步引擎与 Session 工厂（Celery 任务使用）
sync_engine = create_engine(
    settings.SYNC_DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_recycle=settings.DB_POOL_RECYCLE,
    pool_timeout=settings.DB_POOL_TIMEOUT,
    connect_args=_sync_connect_args,
)
sync_session_maker = sessionmaker(sync_engine, class_=Session, expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session


def get_sync_session() -> Generator[Session, None, None]:
    with sync_session_maker() as session:
        yield session


# engine and session factory are ready for use
