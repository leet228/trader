from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import Depends, FastAPI
from pydantic import BaseModel
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.config import get_settings
from shared.db import SessionLocal, get_session
from shared.logger import configure_logging, logger
from shared.models import NewsEvent
from .fetcher import fetch_rss, fetch_newsapi

configure_logging()
settings = get_settings()

app = FastAPI(title="News Service", version="0.1.0")

SessionDep = Annotated[AsyncSession, Depends(get_session)]
redis_client: Redis | None = None


class HealthOut(BaseModel):
    status: str
    env: str


async def ingest_loop() -> None:
    while True:
        try:
            await poll_and_store()
        except Exception as exc:  # noqa: BLE001
            logger.exception("news poll failed", error=str(exc))
        await asyncio.sleep(settings.news_poll_seconds)


async def poll_and_store() -> None:
    assert redis_client
    async with SessionLocal() as session:
        sources = settings.rss_urls + [
            "https://api.gdeltproject.org/api/v2/summary/summary?format=json"
        ]
        # NewsAPI (if key provided)
        if settings.newsapi_key:
            try:
                items = await fetch_newsapi(settings.newsapi_key, settings.newsapi_query)
                logger.info("news fetched", source="newsapi", count=len(items))
                await _store_items(session, items, source="newsapi")
            except Exception as exc:  # noqa: BLE001
                logger.exception("newsapi fetch failed", error=str(exc))
        for url in sources:
            try:
                items = await fetch_rss(url)
                logger.info("news fetched", source=url, count=len(items))
                await _store_items(session, items, source=url)
            except Exception as exc:  # noqa: BLE001
                logger.exception("news fetch failed", source=url, error=str(exc))
                continue


async def _store_items(session: AsyncSession, items: list[dict], source: str) -> None:
    for item in items:
        try:
            if await _exists(session, item["id"]):
                continue
            ev = NewsEvent(
                id=item["id"],
                ts_received=item["ts"],
                source=source,
                headline=item["headline"],
                body=None,
                url=item["url"],
                raw_json={"pub": item.get("raw_pub")},
            )
            session.add(ev)
            await session.commit()
            await redis_client.xadd(
                settings.redis_stream_news_events,
                {
                    "id": ev.id,
                    "ts_received": ev.ts_received.isoformat(),
                    "source": ev.source,
                    "headline": ev.headline,
                    "url": ev.url or "",
                },
                maxlen=settings.redis_xadd_maxlen,
                approximate=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("news store/publish failed", source=source, error=str(exc))
            await session.rollback()


async def _exists(session: AsyncSession, news_id: str) -> bool:
    res = await session.execute(select(NewsEvent.id).where(NewsEvent.id == news_id))
    return res.scalar_one_or_none() is not None


@app.on_event("startup")
async def on_startup() -> None:
    logger.info("news_service started", env=settings.app_env)
    global redis_client
    redis_client = Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        password=settings.redis_password,
        ssl=settings.redis_ssl,
        decode_responses=True,
    )
    asyncio.create_task(ingest_loop())


@app.get("/health", response_model=HealthOut)
async def health() -> HealthOut:
    return HealthOut(status="ok", env=settings.app_env)

