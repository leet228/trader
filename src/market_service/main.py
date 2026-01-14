from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Annotated

import aiohttp
from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.config import get_settings
from shared.db import SessionLocal, get_session
from shared.features import RollingWindow, atr_pct, ema, realized_vol, regime, rsi
from shared.logger import configure_logging, logger
from shared.models import MarketBar, MarketFeatures
from shared.schemas import Regime, Timeframe

configure_logging()
settings = get_settings()

app = FastAPI(title="Market Service", version="0.1.0")

SessionDep = Annotated[AsyncSession, Depends(get_session)]


@dataclass
class SymbolState:
    last_price: float | None = None
    bid: float | None = None
    ask: float | None = None
    bid_qty: float | None = None
    ask_qty: float | None = None
    last_bar_ts: datetime | None = None
    closes: RollingWindow = RollingWindow(400)
    highs: RollingWindow = RollingWindow(400)
    lows: RollingWindow = RollingWindow(400)
    returns: RollingWindow = RollingWindow(400)
    ema200_prev: float | None = None


symbol_states: dict[str, SymbolState] = defaultdict(SymbolState)
redis_client: Redis | None = None


class CandleAgg:
    def __init__(self, tf_minutes: int):
        self.tf_minutes = tf_minutes
        self.start: datetime | None = None
        self.open: float | None = None
        self.high: float | None = None
        self.low: float | None = None
        self.close: float | None = None
        self.volume: float = 0.0

    def _bucket_start(self, ts: datetime) -> datetime:
        minute_bucket = (ts.minute // self.tf_minutes) * self.tf_minutes
        return ts.replace(minute=minute_bucket, second=0, microsecond=0)

    def update(self, ts: datetime, open_: float, high: float, low: float, close: float, volume: float):
        bucket = self._bucket_start(ts)
        completed = None
        if self.start is not None and bucket != self.start:
            completed = {
                "ts": self.start,
                "open": self.open,
                "high": self.high,
                "low": self.low,
                "close": self.close,
                "volume": self.volume,
            }
            self._reset()
        if self.start is None:
            self.start = bucket
            self.open = open_
            self.high = high
            self.low = low
            self.close = close
            self.volume = volume
        else:
            self.high = max(self.high or high, high)
            self.low = min(self.low or low, low)
            self.close = close
            self.volume += volume
        return completed

    def _reset(self) -> None:
        self.start = None
        self.open = None
        self.high = None
        self.low = None
        self.close = None
        self.volume = 0.0


agg_state: dict[int, dict[str, CandleAgg]] = defaultdict(lambda: defaultdict(lambda: CandleAgg(1)))


def _normalize_timeframe(tf: Timeframe | str) -> Timeframe:
    """Ensure timeframe is one of our enums; accept Bybit-style strings."""
    if isinstance(tf, Timeframe):
        return tf
    tf_norm = {
        "1": "1m",
        "1m": "1m",
        "m1": "1m",
        "5": "5m",
        "5m": "5m",
        "m5": "5m",
        "15": "15m",
        "15m": "15m",
        "m15": "15m",
        "60": "1h",
        "1h": "1h",
        "m60": "1h",
    }.get(str(tf), "1m")
    return {
        "1m": Timeframe.m1,
        "5m": Timeframe.m5,
        "15m": Timeframe.m15,
        "1h": Timeframe.h1,
    }.get(tf_norm, Timeframe.m1)


async def save_bar_and_features(
    session: AsyncSession,
    symbol: str,
    timeframe: Timeframe | str,
    ts: datetime,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float,
) -> None:
    timeframe_enum = _normalize_timeframe(timeframe)
    if not isinstance(timeframe_enum, Timeframe):
        logger.warning(f"timeframe_normalize_fallback raw={timeframe} normalized={timeframe_enum}")
        timeframe_enum = Timeframe.m1
    else:
        logger.info(f"timeframe_normalized tf={timeframe_enum.value} raw={timeframe}")
    # persist bar
    bar = MarketBar(
        ts=ts,
        symbol=symbol,
        timeframe=timeframe_enum.value,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )
    session.add(bar)

    state = symbol_states[symbol]
    state.closes.append(close)
    state.highs.append(high)
    state.lows.append(low)
    if state.last_price:
        state.returns.append((close - state.last_price) / state.last_price)
    state.last_price = close
    state.last_bar_ts = ts

    ema20 = ema(state.closes.to_list(), 20)
    ema50 = ema(state.closes.to_list(), 50)
    ema200 = ema(state.closes.to_list(), 200)
    rsi_val = rsi(state.closes.to_list(), 14)
    atr_val = atr_pct(state.highs.to_list(), state.lows.to_list(), state.closes.to_list(), 14)
    vol_val = realized_vol(state.returns.to_list())
    reg = regime(ema200, state.ema200_prev)
    state.ema200_prev = ema200
    spread_pct = None
    if state.bid and state.ask and state.ask > 0 and state.bid > 0:
        mid = 0.5 * (state.bid + state.ask)
        if mid > 0:
            spread_pct = (state.ask - state.bid) / mid

    features = MarketFeatures(
        ts=ts,
        symbol=symbol,
        timeframe=timeframe,
        atr_pct=atr_val,
        ema20=ema20,
        ema50=ema50,
        ema200=ema200,
        rsi=rsi_val,
        returns=state.returns.to_list()[-1] if len(state.returns) else None,
        vol=vol_val,
        spread=spread_pct,
        regime=Regime(reg),
    )
    session.add(features)
    await publish_to_redis(
        bars_payload={
            "ts": ts.isoformat(),
            "symbol": symbol,
            "timeframe": timeframe.value,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        features_payload={
            "ts": ts.isoformat(),
            "symbol": symbol,
            "timeframe": timeframe.value,
            "atr_pct": atr_val,
            "ema20": ema20,
            "ema50": ema50,
            "ema200": ema200,
            "rsi": rsi_val,
            "returns": state.returns.to_list()[-1] if len(state.returns) else None,
            "vol": vol_val,
            "regime": reg,
        },
    )


def _clean_payload(payload: dict) -> dict:
    cleaned = {}
    for k, v in payload.items():
        if v is None:
            continue
        if isinstance(v, (dict, list)):
            cleaned[k] = str(v)
        else:
            cleaned[k] = v
    return cleaned


async def publish_to_redis(bars_payload: dict, features_payload: dict) -> None:
    if redis_client is None:
        return
    try:
        await redis_client.xadd(
            settings.redis_stream_bars,
            _clean_payload(bars_payload),
            maxlen=settings.redis_xadd_maxlen,
            approximate=True,
        )
        await redis_client.xadd(
            settings.redis_stream_features,
            _clean_payload(features_payload),
            maxlen=settings.redis_xadd_maxlen,
            approximate=True,
        )
    except Exception as exc:  # noqa: BLE001 - log any redis issue
        logger.warning(f"failed to publish to redis: {exc}")


async def bybit_ws_loop() -> None:
    url = settings.bybit_ws_url
    subs = []
    for symbol in settings.symbols:
        subs.append({"op": "subscribe", "args": [f"kline.{settings.timeframe}.{symbol}"]})
        subs.append({"op": "subscribe", "args": [f"orderbook.1.{symbol}"]})
    backoff = 1
    while True:
        try:
            logger.info("connecting to bybit ws", url=url, symbols=settings.symbols)
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(url, heartbeat=20) as ws:
                    for sub in subs:
                        await ws.send_json(sub)
                    backoff = 1
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await handle_ws_message(msg.json())
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            raise RuntimeError(f"ws error {msg}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("ws reconnecting", error=str(exc), backoff=backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


async def handle_ws_message(payload: dict) -> None:
    if "topic" not in payload or "data" not in payload:
        return
    topic = payload["topic"]
    if topic.startswith("kline."):
        parts = topic.split(".")
        if len(parts) != 3:
            return
        timeframe_raw = parts[1]
        symbol = parts[2]
        # Normalize Bybit raw tf (can come as "m1", "1", "1m")
        timeframe = _normalize_timeframe(timeframe_raw)
        data = payload["data"]
        # Bybit returns list of bars
        bars = data if isinstance(data, list) else [data]
        async with SessionLocal() as db_session:
            for bar in bars:
                ts_ms = int(bar.get("start", bar.get("t", 0)))
                ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                lag_ms = (datetime.now(timezone.utc) - ts).total_seconds() * 1000
                if lag_ms > settings.ws_lag_warn_ms:
                    logger.warning("ws lag high", symbol=symbol, lag_ms=int(lag_ms), ts=ts.isoformat())
                open_ = float(bar["open"])
                high = float(bar["high"])
                low = float(bar["low"])
                close = float(bar["close"])
                volume = float(bar.get("volume", bar.get("turnover", 0)))
                await save_bar_and_features(
                    db_session,
                    symbol=symbol,
                    timeframe=timeframe,
                    ts=ts,
                    open_=open_,
                    high=high,
                    low=low,
                    close=close,
                    volume=volume,
                )
                await aggregate_and_save(db_session, symbol, ts, open_, high, low, close, volume)
            await db_session.commit()
    elif topic.startswith("orderbook."):
        parts = topic.split(".")
        if len(parts) != 3:
            return
        symbol = parts[2]
        data = payload["data"]
        if isinstance(data, list):
            data = data[0]
        bids = data.get("b", [])
        asks = data.get("a", [])
        if bids:
            symbol_states[symbol].bid = float(bids[0][0])
            if len(bids[0]) > 1:
                symbol_states[symbol].bid_qty = float(bids[0][1])
        if asks:
            symbol_states[symbol].ask = float(asks[0][0])
            if len(asks[0]) > 1:
                symbol_states[symbol].ask_qty = float(asks[0][1])
        if redis_client:
            await redis_client.xadd(
                settings.redis_stream_orderbook,
                {
                    "symbol": symbol,
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "bid": symbol_states[symbol].bid or 0.0,
                    "ask": symbol_states[symbol].ask or 0.0,
                    "bid_qty": symbol_states[symbol].bid_qty or 0.0,
                    "ask_qty": symbol_states[symbol].ask_qty or 0.0,
                },
                maxlen=settings.redis_xadd_maxlen,
                approximate=True,
            )


async def aggregate_and_save(
    db_session: AsyncSession,
    symbol: str,
    ts: datetime,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float,
) -> None:
    # build aggregated candles for configured timeframes (excluding 1m which is already saved)
    for tf in settings.agg_timeframes:
        if tf == 1:
            continue
        agg = agg_state[tf][symbol]
        completed = agg.update(ts, open_, high, low, close, volume)
        if completed:
            tf_enum = {
                5: Timeframe.m5,
                15: Timeframe.m15,
                60: Timeframe.h1,
            }.get(tf)
            if tf_enum is None:
                continue
            await save_bar_and_features(
                db_session,
                symbol=symbol,
                timeframe=tf_enum,
                ts=completed["ts"],
                open_=completed["open"],
                high=completed["high"],
                low=completed["low"],
                close=completed["close"],
                volume=completed["volume"],
            )


class HealthOut(BaseModel):
    status: str
    env: str
    symbols: list[str]


class StatusOut(BaseModel):
    symbol: str
    last_price: float | None
    last_bar_ts: datetime | None
    ema200: float | None = Field(None, description="latest ema200")
    regime: str | None
    atr_pct: float | None
    vol: float | None


@app.on_event("startup")
async def on_startup() -> None:
    logger.info("market_service started", env=settings.app_env)
    global redis_client
    redis_client = Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        password=settings.redis_password,
        ssl=settings.redis_ssl,
        decode_responses=True,
    )
    asyncio.create_task(bybit_ws_loop())


@app.get("/health", response_model=HealthOut)
async def health() -> HealthOut:
    return HealthOut(status="ok", env=settings.app_env, symbols=settings.symbols)


@app.get("/status", response_model=list[StatusOut])
async def status(session: SessionDep) -> list[StatusOut]:
    out: list[StatusOut] = []
    for symbol, state in symbol_states.items():
        # fetch latest ema200 from DB if available
        ema200_val: float | None = None
        reg: str | None = None
        atr_val: float | None = None
        vol_val: float | None = None
        res = await session.execute(
            select(MarketFeatures.ema200, MarketFeatures.regime, MarketFeatures.atr_pct, MarketFeatures.vol)
            .where(MarketFeatures.symbol == symbol)
            .order_by(MarketFeatures.ts.desc())
            .limit(1)
        )
        row = res.first()
        if row:
            ema200_val = row[0]
            reg = row[1].value if row[1] else None
            atr_val = row[2]
            vol_val = row[3]
        out.append(
            StatusOut(
                symbol=symbol,
                last_price=state.last_price,
                last_bar_ts=state.last_bar_ts,
                ema200=ema200_val,
                regime=reg,
                atr_pct=atr_val,
                vol=vol_val,
            )
        )
    return out


