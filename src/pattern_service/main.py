from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import Depends, FastAPI
from pydantic import BaseModel
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.config import get_settings
from shared.db import get_session
from shared.logger import configure_logging, logger
from shared.models import MarketFeatures, PatternSignal as PatternSignalModel
from shared.patterns import detect_patterns
from shared.schemas import PatternSignal, Regime, Timeframe

configure_logging()
settings = get_settings()

app = FastAPI(title="Pattern Service", version="0.1.0")

SessionDep = Annotated[AsyncSession, Depends(get_session)]
redis_client: Redis | None = None


class HealthOut(BaseModel):
    status: str
    env: str


async def process_feature(feature: dict, session: AsyncSession) -> None:
    symbol = feature["symbol"]
    timeframe = Timeframe(feature["timeframe"])
    ts = datetime.fromisoformat(feature["ts"])
    ema20 = _f(feature.get("ema20"))
    ema50 = _f(feature.get("ema50"))
    ema200 = _f(feature.get("ema200"))
    rsi = _f(feature.get("rsi"))
    atr = _f(feature.get("atr_pct"))
    vol = _f(feature.get("vol"))
    regime_val = Regime(feature.get("regime", "UNKNOWN"))

    # placeholder: block mean-reversion if recent high vol
    meanrev_blocked = vol is not None and vol > 0.01
    if redis_client:
        news_flag = await redis_client.get(f"meanrev_block:{symbol}")
        global_flag = await redis_client.get("meanrev_block:GLOBAL")
        if news_flag or global_flag:
            meanrev_blocked = True

    result = detect_patterns(
        symbol=symbol,
        timeframe=timeframe,
        close=_f(feature.get("close")) or _fetch_last_close(symbol, session),
        ema20=ema20,
        ema50=ema50,
        ema200=ema200,
        rsi=rsi,
        atr_pct=atr,
        vol=vol,
        regime_val=regime_val,
        meanrev_blocked=meanrev_blocked,
    )
    if result is None:
        return
    ps = PatternSignal(
        ts=ts,
        symbol=symbol,
        timeframe=timeframe,
        market_bias=result.market_bias,
        market_confidence=result.market_confidence,
        market_setup=result.market_setup,
        setup_name=result.setup_name,
        disabled_reason="meanrev_blocked" if meanrev_blocked and result.market_setup.value == "mean_revert" else None,
    )
    await persist_and_publish(ps, session)


def _f(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


async def _fetch_last_close(symbol: str, session: AsyncSession) -> float | None:
    res = await session.execute(
        select(MarketFeatures.close).where(MarketFeatures.symbol == symbol).order_by(MarketFeatures.ts.desc()).limit(1)
    )
    row = res.first()
    return row[0] if row else None


async def persist_and_publish(ps: PatternSignal, session: AsyncSession) -> None:
    model = PatternSignalModel(
        ts=ps.ts,
        symbol=ps.symbol,
        timeframe=ps.timeframe,
        market_bias=ps.market_bias,
        market_confidence=ps.market_confidence,
        market_setup=ps.market_setup,
        setup_name=ps.setup_name,
    )
    session.add(model)
    await session.commit()
    if redis_client:
        await redis_client.xadd(
            settings.redis_stream_patterns,
            {
                "ts": ps.ts.isoformat(),
                "symbol": ps.symbol,
                "timeframe": ps.timeframe.value,
                "market_bias": ps.market_bias,
                "market_confidence": ps.market_confidence,
                "market_setup": ps.market_setup.value,
                "setup_name": ps.setup_name or "",
            },
            maxlen=settings.redis_xadd_maxlen,
            approximate=True,
        )


async def redis_consumer_loop() -> None:
    assert redis_client
    stream = settings.redis_stream_features
    group = "pattern_service"
    consumer = f"pattern_{datetime.now().timestamp()}"
    try:
        await redis_client.xgroup_create(stream, group, id="0", mkstream=True)
    except Exception:
        pass
    while True:
        msgs = await redis_client.xreadgroup(group, consumer, streams={stream: ">"}, count=50, block=5000)
        if not msgs:
            continue
        async with get_session() as session_gen:
            async for session in session_gen:
                for _, entries in msgs:
                    for msg_id, data in entries:
                        try:
                            await process_feature(data, session)
                            await redis_client.xack(stream, group, msg_id)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("pattern processing failed", error=str(exc), msg_id=msg_id)


@app.on_event("startup")
async def on_startup() -> None:
    logger.info("pattern_service started", env=settings.app_env)
    global redis_client
    redis_client = Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        password=settings.redis_password,
        ssl=settings.redis_ssl,
        decode_responses=True,
    )
    asyncio.create_task(redis_consumer_loop())


@app.get("/health", response_model=HealthOut)
async def health() -> HealthOut:
    return HealthOut(status="ok", env=settings.app_env)

