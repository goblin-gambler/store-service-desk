"""初始化门店服务台数据库（幂等）。

用法::

    PYTHONUTF8=1 PYTHONPATH=. python scripts/init_db.py

说明：本项目在魔改时保留了原电商项目的表结构，并新增 ``loss_reports``。
本地开发直接用 SQLModel.metadata.create_all 建表，建完后把 alembic
版本戳到 head，后续增量变更仍可继续用 alembic。

注意：生产环境请使用 migrations/ 下的 alembic 迁移，不要依赖本脚本。
"""

import asyncio

from sqlmodel import SQLModel

import app.models  # noqa: F401 - 导入以注册全部模型
from app.core.database import async_engine


async def main() -> None:
    async with async_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    await async_engine.dispose()
    print("database schema created/verified")


if __name__ == "__main__":
    asyncio.run(main())
