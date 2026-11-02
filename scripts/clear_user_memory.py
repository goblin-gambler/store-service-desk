"""清理某个用户的长期记忆。

删除范围：
1. Qdrant `conversation_memory` 中该用户的对话向量；
2. PostgreSQL `user_facts`（结构化事实）与 `interaction_summaries`（会话摘要）。

用法：
    PYTHONUTF8=1 PYTHONPATH=. .venv/Scripts/python.exe scripts/clear_user_memory.py --user-id 1 --dry-run
    PYTHONUTF8=1 PYTHONPATH=. .venv/Scripts/python.exe scripts/clear_user_memory.py --user-id 1 --yes

适用场景：测试数据污染了记忆（例如早期错误回答被事实抽取后，后续提问会反复召回）。
"""

from __future__ import annotations

import argparse
import asyncio

from qdrant_client import AsyncQdrantClient, models
from sqlmodel import delete, func, select

from app.core.config import settings
from app.core.database import async_session_maker
from app.models.memory import InteractionSummary, UserFact

COLLECTION = "conversation_memory"


async def _count_vectors(client: AsyncQdrantClient, user_id: int) -> int:
    if not await client.collection_exists(COLLECTION):
        return 0
    result = await client.count(
        collection_name=COLLECTION,
        count_filter=models.Filter(
            must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
        ),
        exact=True,
    )
    return int(result.count)


async def _delete_vectors(client: AsyncQdrantClient, user_id: int) -> None:
    if not await client.collection_exists(COLLECTION):
        return
    await client.delete(
        collection_name=COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))
                ]
            )
        ),
        wait=True,
    )


async def main(user_id: int, dry_run: bool) -> None:
    client = AsyncQdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY.get_secret_value(),
        timeout=settings.QDRANT_TIMEOUT,
    )
    try:
        vector_count = await _count_vectors(client, user_id)

        async with async_session_maker() as session:
            fact_count = (
                await session.exec(
                    select(func.count()).select_from(UserFact).where(UserFact.user_id == user_id)
                )
            ).one()
            summary_count = (
                await session.exec(
                    select(func.count())
                    .select_from(InteractionSummary)
                    .where(InteractionSummary.user_id == user_id)
                )
            ).one()

            print(
                f"user_id={user_id} 待清理：向量 {vector_count} 条，"
                f"事实 {fact_count} 条，摘要 {summary_count} 条"
            )
            if dry_run:
                print("dry-run：未删除任何数据")
                return

            await _delete_vectors(client, user_id)
            await session.exec(delete(UserFact).where(UserFact.user_id == user_id))
            await session.exec(
                delete(InteractionSummary).where(InteractionSummary.user_id == user_id)
            )
            await session.commit()

        remaining = await _count_vectors(client, user_id)
        print(f"已清理，剩余向量 {remaining} 条")
    finally:
        await client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="清理指定用户的长期记忆")
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true", help="只统计不删除")
    parser.add_argument("--yes", action="store_true", help="跳过确认直接删除")
    args = parser.parse_args()

    if not args.dry_run and not args.yes:
        print("删除不可恢复，确认请加 --yes（先跑 --dry-run 看看数量）")
        raise SystemExit(1)

    asyncio.run(main(args.user_id, args.dry_run))
