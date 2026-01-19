from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import Depends, FastAPI
from pydantic import BaseModel
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from shared.config import get_settings
from shared.db import SessionLocal, get_session
from shared.logger import configure_logging, logger
from shared.models import NewsScore as NewsScoreModel
from shared.schemas import EventType, NewsScore
from shared.notify import send_telegram_message

configure_logging()
settings = get_settings()

app = FastAPI(title="AI/NLP Service", version="0.1.0")

SessionDep = Annotated[AsyncSession, Depends(get_session)]
redis_client: Redis | None = None


class HealthOut(BaseModel):
    status: str
    env: str


KEYWORDS_NEG = ["ban", "hack", "sanction", "lawsuit", "fraud", "insolvency", "default", "down", "layoff"]
KEYWORDS_POS = ["approval", "etf", "partnership", "upgrade", "bullish", "record high", "support"]


def score_headline(headline: str) -> NewsScore:
    text = headline.lower()
    neg_hits = [k for k in KEYWORDS_NEG if k in text]
    pos_hits = [k for k in KEYWORDS_POS if k in text]
    bias = 0.0
    conf = 0.3
    if neg_hits and not pos_hits:
        bias = -0.7
        conf = 0.65
    elif pos_hits and not neg_hits:
        bias = 0.6
        conf = 0.6
    elif pos_hits and neg_hits:
        bias = 0.0
        conf = 0.2
    return NewsScore(
        news_id="",
        ts_scored=datetime.now(timezone.utc),
        event_type=EventType.other,
        news_bias=bias,
        news_confidence=conf,
        horizon_minutes=60,
        reason_tags=neg_hits + pos_hits,
        model_used="rules",
    )


async def process_news(data: dict, session: AsyncSession) -> None:
    news_id = data["id"]
    headline = data.get("headline", "")
    ns = score_headline(headline)
    ns.news_id = news_id
    model = NewsScoreModel(
        news_id=news_id,
        ts_scored=ns.ts_scored,
        event_type=ns.event_type,
        news_bias=ns.news_bias,
        news_confidence=ns.news_confidence,
        horizon_minutes=ns.horizon_minutes,
        reason_tags=ns.reason_tags,
        model_used=ns.model_used,
    )
    session.add(model)
    await session.commit()
    if redis_client:
        await redis_client.xadd(
            settings.redis_stream_news_scores,
            {
                "news_id": news_id,
                "ts_scored": ns.ts_scored.isoformat(),
                "news_bias": ns.news_bias,
                "news_confidence": ns.news_confidence,
                "event_type": ns.event_type.value,
                "symbol": data.get("symbol", "GLOBAL"),
            },
            maxlen=settings.redis_xadd_maxlen,
            approximate=True,
        )
        # block mean-reversion for negative, confident news
        if ns.news_bias < -0.4 and ns.news_confidence >= 0.6:
            ttl = settings.meanrev_block_news_minutes * 60
            await redis_client.setex("meanrev_block:GLOBAL", ttl, "1")
        if abs(ns.news_bias) >= 0.5 and ns.news_confidence >= 0.6:
            await send_telegram_message(
                f"News alert: bias {ns.news_bias:.2f} conf {ns.news_confidence:.2f}\n"
                f"{headline}"
            )


async def news_consumer_loop() -> None:
    assert redis_client
    stream = settings.redis_stream_news_events
    group = "ai_nlp_service"
    consumer = f"ai_nlp_{datetime.now().timestamp()}"
    while True:
        try:
            try:
                await redis_client.xgroup_create(stream, group, id="0", mkstream=True)
            except Exception:
                pass
            msgs = await redis_client.xreadgroup(
                group, consumer, streams={stream: ">"}, count=50, block=5000
            )
            if not msgs:
                continue
            async with SessionLocal() as session:
                for _, entries in msgs:
                    for msg_id, data in entries:
                        try:
                            await process_news(data, session)
                            await redis_client.xack(stream, group, msg_id)
                        except Exception as exc:  # noqa: BLE001
                            logger.exception("ai_nlp process failed", error=str(exc), msg_id=msg_id)
                            await session.rollback()
        except Exception as exc:  # noqa: BLE001
            logger.exception("ai_nlp consumer loop error", error=str(exc))
            await asyncio.sleep(1)


@app.on_event("startup")
async def on_startup() -> None:
    logger.info("ai_nlp_service started", env=settings.app_env)
    global redis_client
    redis_client = Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        password=settings.redis_password,
        ssl=settings.redis_ssl,
        decode_responses=True,
    )
    asyncio.create_task(news_consumer_loop())


@app.get("/health", response_model=HealthOut)
async def health() -> HealthOut:
    return HealthOut(status="ok", env=settings.app_env)

