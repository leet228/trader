from __future__ import annotations

import asyncio
import glob
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated

import shap
from fastapi import Depends, FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from redis.asyncio import Redis
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.config import get_settings
from shared.db import SessionLocal, get_session
from shared.logger import configure_logging, logger
from shared.models import (
    BotState,
    Decision as DecisionModel,
    MarketBar,
    MarketFeatures,
    ModelPrediction as ModelPredictionModel,
    Trade as TradeModel,
    TradePlan as TradePlanModel,
)
from shared.notify import send_telegram_message
from shared.schemas import DecisionSide, PatternSignal

configure_logging()
settings = get_settings()

app = FastAPI(title="Trader Service", version="0.1.0")

SessionDep = Annotated[AsyncSession, Depends(get_session)]
redis_client: Redis | None = None
news_cache: dict[str, dict] = {}
model_artifact: dict | None = None
use_orderbook: bool = True
SIGNAL_DEDUPE_TTL_SECONDS = 7 * 24 * 60 * 60


class HealthOut(BaseModel):
    status: str
    env: str


class StatusOut(BaseModel):
    open_trades: int
    last_decision_ts: datetime | None


async def get_bot_state(session: AsyncSession) -> BotState:
    res = await session.execute(select(BotState).limit(1))
    state = res.scalar_one_or_none()
    if state is None:
        state = BotState()
        state.daily_max_loss_pct = 5.0
        state.max_trades_per_day = 0
        state.cooldown_after_losses_min = 10 / 60  # 10 seconds
        state.equity_usd = settings.base_equity_usd
        session.add(state)
        await session.commit()
        await session.refresh(state)
        return state
    updated = False
    if state.daily_max_loss_pct != 5.0:
        state.daily_max_loss_pct = 5.0
        updated = True
    if state.max_trades_per_day != 0:
        state.max_trades_per_day = 0
        updated = True
    if updated:
        await session.commit()
        await session.refresh(state)
    return state


async def handle_signal(data: dict, session: AsyncSession) -> None:
    ps = PatternSignal(
        ts=datetime.fromisoformat(data["ts"]),
        symbol=data["symbol"],
        timeframe=data["timeframe"],
        market_bias=float(data["market_bias"]),
        market_confidence=float(data["market_confidence"]),
        market_setup=data["market_setup"],
        setup_name=data.get("setup_name") or None,
    )
    # basic guard
    if ps.market_confidence < 0.4:
        logger.info(
            f"skip signal: low market_confidence={ps.market_confidence:.3f} bias={ps.market_bias:.3f} "
            f"symbol={ps.symbol} tf={ps.timeframe}"
        )
        return
    # ML prediction
    pred_side, pred_conf, p_long, p_short, model_version, contrib = await _predict_ml(ps, session)
    logger.info(
        "ml prediction "
        f"symbol={ps.symbol} tf={ps.timeframe} side={pred_side} "
        f"pred_conf={pred_conf:.3f} p_long={p_long:.3f} p_short={p_short:.3f} "
        f"threshold={settings.model_confidence_threshold:.3f} "
        f"market_conf={ps.market_confidence:.3f} market_bias={ps.market_bias:.3f}"
    )
    if pred_side == DecisionSide.hold or pred_conf < settings.model_confidence_threshold:
        logger.info(
            "skip signal: model low confidence or hold "
            f"symbol={ps.symbol} tf={ps.timeframe} pred_conf={pred_conf:.3f} "
            f"market_conf={ps.market_confidence:.3f} market_bias={ps.market_bias:.3f} "
            f"threshold={settings.model_confidence_threshold:.3f} side={pred_side}"
        )
        return
    side = pred_side

    # news conflict filter
    news = news_cache.get(ps.symbol)
    if news:
        if news["conf"] >= 0.8 and abs(news["bias"]) >= 0.4:
            if news["bias"] * ps.market_bias < 0:  # conflict
                logger.info("skip due to news conflict", symbol=ps.symbol, news_bias=news["bias"])
                return
            # mild boost or keep as is; we just mark used_news
    used_news = news is not None

    state = await get_bot_state(session)
    if not await _risk_ok(state, session):
        return

    price, atr_pct, spread_pct, vol_val = await _latest_price_and_stats(ps.symbol, ps.timeframe, session)
    if price is None:
        return
    signal_ts = data.get("ts") or ps.ts.isoformat()
    tf_key = ps.timeframe.value if hasattr(ps.timeframe, "value") else str(ps.timeframe)
    if redis_client:
        dedupe_key = f"signal_trade:{ps.symbol}:{tf_key}:{signal_ts}"
        was_set = await redis_client.set(
            dedupe_key,
            "1",
            nx=True,
            ex=SIGNAL_DEDUPE_TTL_SECONDS,
        )
        if not was_set:
            logger.info(
                "skip duplicate signal trade",
                key=dedupe_key,
                symbol=ps.symbol,
                timeframe=tf_key,
                ts=signal_ts,
            )
            return
    else:
        res = await session.execute(
            select(DecisionModel.decision_id).where(
                DecisionModel.ts == ps.ts,
                DecisionModel.symbol == ps.symbol,
            )
        )
        if res.scalar_one_or_none():
            logger.info("skip duplicate signal trade (db)", symbol=ps.symbol, ts=ps.ts)
            return
    stop_pct = max(0.006, 1.5 * atr_pct) if atr_pct is not None else 0.01
    # dynamic risk: base from state, scaled by model confidence up to 5%
    risk_pct = min(0.05, max(state.risk_per_trade_pct / 100, pred_conf * 0.05))
    equity = state.equity_usd
    risk_usd = equity * risk_pct
    notional = risk_usd / stop_pct
    notional = min(notional, equity * state.max_leverage)
    qty = notional / price
    leverage = min(state.max_leverage, max(1.0, notional / equity))

    spread_base = settings.spread_bps / 10000
    spread_dyn = max(spread_base, (spread_pct or 0))
    slip = settings.slippage_bps / 10000 + (vol_val or 0) * settings.slippage_vol_mult / 10000
    if side == DecisionSide.long:
        entry_price = price * (1 + spread_dyn + slip)
        stop_price = entry_price * (1 - stop_pct)
        take_profit_price = entry_price * (1 + stop_pct * 2)
    else:
        entry_price = price * (1 - spread_dyn - slip)
        stop_price = entry_price * (1 + stop_pct)
        take_profit_price = entry_price * (1 - stop_pct * 2)

    decision_id = str(uuid.uuid4())
    trade_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    decision_enum = side if isinstance(side, DecisionSide) else DecisionSide(side)
    decision = DecisionModel(
        decision_id=decision_id,
        ts=now,
        symbol=ps.symbol,
        decision=decision_enum,
        used_market=True,
        used_news=used_news,
        used_ml=True,
        thresholds={"market_conf_min": 0.4},
        rule_version="phase4-market",
        decision_reason=contrib or ("news_conflict" if news and news["bias"] * ps.market_bias < 0 else None),
        model_version=model_version,
    )
    session.add(decision)
    session.add(
        TradePlanModel(
            decision_id=decision_id,
            entry_type="market",
            entry_price_ref=entry_price,
            stop_price=stop_price,
            take_profit_price=take_profit_price,
            notional_usd=notional,
            leverage=leverage,
            risk_usd=risk_usd,
            constraints_snapshot={"atr_pct": atr_pct},
        )
    )
    entry_fee = notional * settings.commission_rate
    session.add(
        TradeModel(
            trade_id=trade_id,
            decision_id=decision_id,
            open_ts=now,
            side=decision_enum,
            qty=qty,
            notional=notional,
            entry_px=entry_price,
            fees_est=entry_fee,
            close_reason="open",
        )
    )
    session.add(
        ModelPredictionModel(
            ts=now,
            symbol=ps.symbol,
            p_long=p_long,
            p_short=p_short,
            confidence=pred_conf,
            model_version=model_version,
            top_features=model_artifact.get("top_features", []),
        )
    )
    await session.commit()
    logger.info(
        "opened paper trade",
        symbol=ps.symbol,
        side=side.value,
        price=price,
        qty=qty,
        notional=notional,
        stop=stop_price,
        tp=take_profit_price,
    )
    await send_telegram_message(
        f"Opened {side.value} {ps.symbol}\n"
        f"Entry: {entry_price:.4f}\n"
        f"Stop: {stop_price:.4f} | TP: {take_profit_price:.4f}\n"
        f"Notional: {notional:.2f} USD, Qty: {qty:.6f}\n"
        f"Model: {model_version}, Conf: {pred_conf:.2f}\n"
        f"Top: {model_artifact.get('top_features', [])}"
    )


async def _latest_price_and_stats(symbol: str, timeframe: str, session: AsyncSession):
    res = await session.execute(
        select(MarketBar.close, MarketFeatures.atr_pct, MarketFeatures.spread, MarketFeatures.vol)
        .join(
            MarketFeatures,
            (MarketFeatures.ts == MarketBar.ts)
            & (MarketFeatures.symbol == MarketBar.symbol)
            & (MarketFeatures.timeframe == MarketBar.timeframe),
        )
        .where(MarketBar.symbol == symbol, MarketBar.timeframe == timeframe)
        .order_by(MarketBar.ts.desc())
        .limit(1)
    )
    row = res.first()
    if not row:
        return None, None, None, None
    close, atr, spread, vol_val = row
    spread_pct = spread if spread is not None else None
    return close, atr, spread_pct, vol_val


async def redis_consumer_loop() -> None:
    assert redis_client
    stream = settings.redis_stream_patterns
    group = "trader_service"
    consumer = f"trader_{uuid.uuid4()}"
    while True:
        try:
            await redis_client.xgroup_create(stream, group, id="0", mkstream=True)
        except Exception:
            pass
        try:
            msgs = await redis_client.xreadgroup(group, consumer, streams={stream: ">"}, count=20, block=5000)
        except Exception as exc:
            logger.warning("trader pattern consume xreadgroup failed", error=str(exc))
            await asyncio.sleep(1)
            continue
        if not msgs:
            continue
        async with SessionLocal() as session:
            for _, entries in msgs:
                for msg_id, data in entries:
                    try:
                        await handle_signal(data, session)
                        await session.commit()
                        await redis_client.xack(stream, group, msg_id)
                    except Exception as exc:  # noqa: BLE001
                        await session.rollback()
                        logger.exception(
                            "trader handle_signal failed",
                            error=str(exc),
                            msg_id=msg_id,
                            data_keys=list(data.keys()),
                        )


async def news_consumer_loop() -> None:
    assert redis_client
    stream = settings.redis_stream_news_scores
    group = "trader_news"
    consumer = f"trader_news_{uuid.uuid4()}"
    while True:
        try:
            await redis_client.xgroup_create(stream, group, id="0", mkstream=True)
        except Exception:
            pass
        try:
            msgs = await redis_client.xreadgroup(group, consumer, streams={stream: ">"}, count=50, block=5000)
        except Exception as exc:
            logger.warning("trader news consume xreadgroup failed", error=str(exc))
            await asyncio.sleep(1)
            continue
        if not msgs:
            continue
        for _, entries in msgs:
            for msg_id, data in entries:
                try:
                    symbol = data.get("symbol") or "GLOBAL"
                    news_cache[symbol] = {
                        "bias": float(data.get("news_bias", 0)),
                        "conf": float(data.get("news_confidence", 0)),
                        "ts": data.get("ts_scored"),
                    }
                    await redis_client.xack(stream, group, msg_id)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("trader news consume failed", error=str(exc), msg_id=msg_id)


def _load_model() -> None:
    global model_artifact
    paths = sorted(glob.glob("data/models/model_*.joblib"))
    if not paths:
        logger.warning("no model artifacts found")
        return
    latest = paths[-1]
    try:
        model_artifact = joblib.load(latest)
        logger.info("model loaded", path=latest)
    except Exception as exc:  # noqa: BLE001
        logger.error("failed to load model", error=str(exc))


def _reload_model() -> str:
    global model_artifact
    model_artifact = None
    _load_model()
    return "loaded" if model_artifact else "not_found"


async def _predict_ml(ps: PatternSignal, session: AsyncSession):
    if not model_artifact:
        return DecisionSide.hold, 0.0, 0.0, 0.0, "", ""
    feat = await _latest_features(ps.symbol, ps.timeframe, session)
    if feat is None:
        return DecisionSide.hold, 0.0, 0.0, 0.0, "", ""
    features = model_artifact["features"]
    # Build row with named features to satisfy sklearn feature_names
    row = {name: float(feat.get(name, 0.0) or 0.0) for name in features}
    row["market_bias"] = float(ps.market_bias)
    row["market_confidence"] = float(ps.market_confidence)
    x_df = pd.DataFrame([row], columns=features)
    model = model_artifact["model"]
    classes = model_artifact["classes"]
    proba = model.predict_proba(x_df)[0]
    idx_max = int(np.argmax(proba))
    conf = float(proba[idx_max])
    class_list = list(classes)
    p_long = float(proba[class_list.index("long")] if "long" in class_list else 0.0)
    p_short = float(proba[class_list.index("short")] if "short" in class_list else 0.0)
    side = DecisionSide.hold
    if conf >= settings.model_confidence_threshold:
        label = classes[idx_max]
        if label == "long":
            side = DecisionSide.long
        elif label == "short":
            side = DecisionSide.short

    contrib_text = ""
    try:
        feat_names = ["atr_pct", "ema20", "ema50", "ema200", "rsi", "returns", "vol", "spread", "market_bias", "market_confidence", "news_bias", "news_confidence"]
        if model_artifact.get("model_type") == "gbm" and settings.enable_shap:
            base = model.calibrated_classifiers_[0].base_estimator if hasattr(model, "calibrated_classifiers_") else model
            explainer = shap.Explainer(base)
            sv = explainer(x_df)
            vals = sv.values[0]
            top_idx = np.argsort(np.abs(vals))[::-1][:3]
            contrib_pairs = [(feat_names[i], float(vals[i])) for i in top_idx]
            contrib_text = " | ".join(f"{n}:{v:.3f}" for n, v in contrib_pairs)
        else:
            if hasattr(model, "calibrated_classifiers_"):
                base = model.calibrated_classifiers_[0].base_estimator
            else:
                base = model
            if hasattr(base, "coef_"):
                coefs = base.coef_
                class_list = list(classes)
                idx = class_list.index("long") if "long" in class_list else idx_max
                feats_arr = x_df.to_numpy()[0]
                prod = coefs[idx] * feats_arr
                top_idx = np.argsort(np.abs(prod))[::-1][:3]
                contrib_pairs = [(feat_names[i], float(prod[i])) for i in top_idx]
                contrib_text = " | ".join(f"{n}:{v:.3f}" for n, v in contrib_pairs)
    except Exception:
        contrib_text = ""

    return side, conf, p_long, p_short, str(model_artifact.get("version", "")), contrib_text


async def _latest_features(symbol: str, timeframe: str, session: AsyncSession) -> dict | None:
    res = await session.execute(
        select(MarketFeatures, MarketBar.close)
        .join(
            MarketBar,
            (MarketBar.ts == MarketFeatures.ts)
            & (MarketBar.symbol == MarketFeatures.symbol)
            & (MarketBar.timeframe == MarketFeatures.timeframe),
        )
        .where(MarketFeatures.symbol == symbol, MarketFeatures.timeframe == timeframe)
        .order_by(MarketFeatures.ts.desc())
        .limit(1)
    )
    row = res.first()
    if not row:
        return None
    mf, close = row
    news_raw = news_cache.get(symbol) or {}
    news_conf = float(news_raw.get("conf", 0.0))
    news_bias = float(news_raw.get("bias", 0.0))
    # feed news into model only if it passes threshold (conf>=0.8 and |bias|>=0.4)
    if news_conf >= 0.8 and abs(news_bias) >= 0.4:
        news = {"bias": news_bias, "conf": news_conf}
    else:
        news = {"bias": 0.0, "conf": 0.0}
    return {
        "atr_pct": mf.atr_pct,
        "ema20": mf.ema20,
        "ema50": mf.ema50,
        "ema200": mf.ema200,
        "rsi": mf.rsi,
        "returns": mf.returns,
        "vol": mf.vol,
        "spread": mf.spread,
        "news_bias": news.get("bias", 0.0),
        "news_confidence": news.get("conf", 0.0),
        "close": close,
    }


async def bar_consumer_loop() -> None:
    assert redis_client
    stream = settings.redis_stream_orderbook if use_orderbook else settings.redis_stream_bars
    group = "trader_orderbook" if use_orderbook else "trader_bars"
    consumer = f"{group}_{uuid.uuid4()}"
    while True:
        try:
            await redis_client.xgroup_create(stream, group, id="0", mkstream=True)
        except Exception:
            pass
        try:
            msgs = await redis_client.xreadgroup(group, consumer, streams={stream: ">"}, count=50, block=5000)
        except Exception as exc:
            logger.warning("trader bar/orderbook consume xreadgroup failed", error=str(exc))
            await asyncio.sleep(1)
            continue
        if not msgs:
            continue
        async with SessionLocal() as session:
            for _, entries in msgs:
                for msg_id, data in entries:
                    try:
                        if use_orderbook:
                            await handle_orderbook(data, session)
                        else:
                            await handle_bar(data, session)
                        await session.commit()
                        await redis_client.xack(stream, group, msg_id)
                    except Exception as exc:  # noqa: BLE001
                        await session.rollback()
                        logger.warning("trader handle_stream failed", error=str(exc), msg_id=msg_id)


async def handle_bar(data: dict, session: AsyncSession) -> None:
    symbol = data["symbol"]
    close = float(data["close"])
    high = float(data["high"])
    low = float(data["low"])
    ts = datetime.fromisoformat(data["ts"])
    # get spread/vol for dynamic slip
    res = await session.execute(
        select(MarketFeatures.spread, MarketFeatures.vol)
        .where(MarketFeatures.symbol == symbol)
        .order_by(MarketFeatures.ts.desc())
        .limit(1)
    )
    row = res.first()
    spread_val = row[0] if row else None
    vol_val = row[1] if row else None
    spread = max(settings.spread_bps / 10000, (spread_val or 0))
    slippage = settings.slippage_bps / 10000 + (vol_val or 0) * settings.slippage_vol_mult / 10000

    open_trades = await session.execute(
        select(TradeModel).where(and_(TradeModel.decision_id == DecisionModel.decision_id, TradeModel.close_ts == None))  # type: ignore  # noqa: E711
    )
    trades = open_trades.scalars().all()
    for trade in trades:
        # ensure symbol match via decision
        dec = await session.get(DecisionModel, trade.decision_id)
        if dec is None or dec.symbol != symbol:
            continue
        # max position time
        state = await get_bot_state(session)
        if state.max_position_time_min and ts - trade.open_ts > timedelta(minutes=state.max_position_time_min):
            exit_px = close * (1 - slippage) if trade.side == DecisionSide.long else close * (1 + slippage)
            await _close_trade(session, trade, dec, exit_px, "timeout")
            continue
        # stop/tp check
        plan = await session.execute(
            select(TradePlanModel).where(TradePlanModel.decision_id == trade.decision_id)
        )
        plan_obj = plan.scalar_one_or_none()
        if plan_obj is None:
            continue
        if trade.side == DecisionSide.long:
            if low <= (plan_obj.stop_price or 0):
                exit_px = (plan_obj.stop_price or close) * (1 - slippage - spread)
                await _close_trade(session, trade, dec, exit_px, "stop")
                continue
            if high >= (plan_obj.take_profit_price or 1e9):
                exit_px = (plan_obj.take_profit_price or close) * (1 - slippage - spread)
                await _close_trade(session, trade, dec, exit_px, "tp")
        else:
            if high >= (plan_obj.stop_price or 0):
                exit_px = (plan_obj.stop_price or close) * (1 + slippage + spread)
                await _close_trade(session, trade, dec, exit_px, "stop")
                continue
            if low <= (plan_obj.take_profit_price or -1e9):
                exit_px = (plan_obj.take_profit_price or close) * (1 + slippage + spread)
                await _close_trade(session, trade, dec, exit_px, "tp")
    await session.commit()


async def handle_orderbook(data: dict, session: AsyncSession) -> None:
    symbol = data["symbol"]
    bid = float(data.get("bid", 0))
    ask = float(data.get("ask", 0))
    bid_qty = float(data.get("bid_qty", 0))
    ask_qty = float(data.get("ask_qty", 0))
    ts = datetime.fromisoformat(data["ts"])

    open_trades = await session.execute(
        select(TradeModel).where(and_(TradeModel.decision_id == DecisionModel.decision_id, TradeModel.close_ts == None))  # type: ignore  # noqa: E711
    )
    trades = open_trades.scalars().all()
    for trade in trades:
        dec = await session.get(DecisionModel, trade.decision_id)
        if dec is None or dec.symbol != symbol:
            continue
        state = await get_bot_state(session)
        plan = await session.execute(
            select(TradePlanModel).where(TradePlanModel.decision_id == trade.decision_id)
        )
        plan_obj = plan.scalar_one_or_none()
        if plan_obj is None:
            continue
        spread = max(settings.spread_bps / 10000, (ask - bid) / max(ask, bid, 1e-9))
        slippage = settings.slippage_bps / 10000
        qty = trade.qty or 0
        if trade.side == DecisionSide.long:
            if plan_obj.stop_price and bid <= plan_obj.stop_price:
                exit_px = plan_obj.stop_price * (1 - slippage - spread)
                await _close_trade(session, trade, dec, exit_px, "stop")
                continue
            if plan_obj.take_profit_price and ask >= plan_obj.take_profit_price:
                exit_px = plan_obj.take_profit_price * (1 - slippage - spread)
                await _close_trade(session, trade, dec, exit_px, "tp")
        else:
            if plan_obj.stop_price and ask >= plan_obj.stop_price:
                exit_px = plan_obj.stop_price * (1 + slippage + spread)
                await _close_trade(session, trade, dec, exit_px, "stop")
                continue
            if plan_obj.take_profit_price and bid <= plan_obj.take_profit_price:
                exit_px = plan_obj.take_profit_price * (1 + slippage + spread)
                await _close_trade(session, trade, dec, exit_px, "tp")
        # timeout check
        if state.max_position_time_min and ts - trade.open_ts > timedelta(minutes=state.max_position_time_min):
            mid = 0.5 * (bid + ask) if (bid and ask) else (trade.entry_px)
            exit_px = mid * (1 - slippage - spread) if trade.side == DecisionSide.long else mid * (1 + slippage + spread)
            await _close_trade(session, trade, dec, exit_px, "timeout")

    await session.commit()


async def _close_trade(session: AsyncSession, trade: TradeModel, dec: DecisionModel, exit_px: float, reason: str) -> None:
    side = dec.decision
    qty = trade.qty or 0
    entry_px = trade.entry_px
    gross_pnl = (exit_px - entry_px) * qty if side == DecisionSide.long else (entry_px - exit_px) * qty
    exit_fee = (trade.notional or 0) * settings.commission_rate
    pnl = gross_pnl - (trade.fees_est or 0) - exit_fee
    trade.exit_px = exit_px
    trade.close_ts = datetime.now(timezone.utc)
    trade.pnl = pnl
    trade.close_reason = reason
    trade.fees_est = (trade.fees_est or 0) + exit_fee
    # update equity
    state = await get_bot_state(session)
    state.equity_usd = state.equity_usd + pnl
    # cooldown after two losses
    if pnl < 0:
        await _maybe_cooldown(session)
    await _check_daily_loss(session)
    await session.commit()
    await send_telegram_message(
        f"Closed {dec.decision.value} {dec.symbol} [{reason}]\n"
        f"Entry: {entry_px:.4f} Exit: {exit_px:.4f}\n"
        f"PnL: {pnl:.2f} USD"
    )


@app.on_event("startup")
async def on_startup() -> None:
    logger.info("trader_service started", env=settings.app_env)
    global redis_client
    redis_client = Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        password=settings.redis_password,
        ssl=settings.redis_ssl,
        decode_responses=True,
    )
    asyncio.create_task(redis_consumer_loop())
    asyncio.create_task(bar_consumer_loop())
    asyncio.create_task(news_consumer_loop())
    _load_model()


@app.get("/health", response_model=HealthOut)
async def health() -> HealthOut:
    return HealthOut(status="ok", env=settings.app_env)


@app.post("/reload_model")
async def reload_model() -> dict[str, str]:
    status = _reload_model()
    return {"status": status}


@app.get("/status", response_model=StatusOut)
async def status(session: SessionDep) -> StatusOut:
    res = await session.execute(select(func.count(TradeModel.id)))
    count = res.scalar() or 0
    res2 = await session.execute(select(func.max(DecisionModel.ts)))
    last_decision = res2.scalar()
    return StatusOut(open_trades=count, last_decision_ts=last_decision)


async def _risk_ok(state: BotState, session: AsyncSession) -> bool:
    now = datetime.now(timezone.utc)
    if state.halt:
        return False
    if state.paused_until and state.paused_until > now:
        return False
    # daily loss check
    if await _daily_loss_exceeded(session, state):
        state.halt = True
        await session.commit()
        await send_telegram_message(
            "Halt triggered: daily loss limit exceeded",
            chat_id=settings.alert_daily_limit_channel,
        )
        return False
    # max trades per day (0 = unlimited)
    if state.max_trades_per_day > 0:
        start_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        res = await session.execute(
            select(func.count(TradeModel.id)).where(TradeModel.open_ts >= start_day)
        )
        trades_today = res.scalar() or 0
        if trades_today >= state.max_trades_per_day:
            return False
    return True


async def _daily_loss_exceeded(session: AsyncSession, state: BotState) -> bool:
    now = datetime.now(timezone.utc)
    start_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    res = await session.execute(
        select(func.sum(TradeModel.pnl)).where(
            and_(TradeModel.close_ts != None, TradeModel.close_ts >= start_day)  # type: ignore  # noqa: E711
        )
    )
    pnl_today = res.scalar() or 0.0
    limit = -settings.base_equity_usd * (state.daily_max_loss_pct / 100)
    return pnl_today <= limit


async def _maybe_cooldown(session: AsyncSession) -> None:
    res = await session.execute(
        select(TradeModel).where(TradeModel.close_ts != None).order_by(TradeModel.close_ts.desc()).limit(2)  # type: ignore  # noqa: E711
    )
    trades = res.scalars().all()
    if len(trades) < 2:
        return
    if all((t.pnl or 0) < 0 for t in trades):
        state = await get_bot_state(session)
        state.paused_until = datetime.now(timezone.utc) + timedelta(minutes=state.cooldown_after_losses_min)
        await session.commit()
        await send_telegram_message(
            f"Cooldown engaged for {state.cooldown_after_losses_min} minutes after losses",
            chat_id=settings.alert_errors_channel,
        )


async def _check_daily_loss(session: AsyncSession) -> None:
    state = await get_bot_state(session)
    if await _daily_loss_exceeded(session, state):
        state.halt = True
        await session.commit()

