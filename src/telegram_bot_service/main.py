from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from typing import Annotated, Awaitable, Callable

import aiohttp
from aiogram import BaseMiddleware
from aiogram import Bot, Dispatcher
from aiogram.client.bot import DefaultBotProperties
from aiogram.types import BotCommand, ReplyKeyboardMarkup, KeyboardButton
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import Message
from fastapi import Depends, FastAPI
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.config import get_settings
from shared.db import SessionLocal, get_session
from shared.logger import configure_logging, logger
from redis.asyncio import Redis
from shared.models import BotState, Decision, MarketBar, MarketFeatures, ModelPrediction, NewsScore, PatternSignal, Trade
from shared.notify import send_telegram_message
from shared.schemas import Timeframe

configure_logging()
settings = get_settings()

app = FastAPI(title="Telegram Bot Service", version="0.1.0")

SessionDep = Annotated[AsyncSession, Depends(get_session)]
bot: Bot | None = None
dp: Dispatcher | None = None
redis_client: Redis | None = None


class BotStateOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    mode: str
    paused_until: datetime | None
    halt: bool
    risk_per_trade_pct: float
    daily_max_loss_pct: float
    max_leverage: float
    max_trades_per_day: int
    cooldown_after_losses_min: int
    max_position_time_min: int
    last_error: str | None
    updated_at: datetime


class PauseRequest(BaseModel):
    minutes: int = Field(default=60, ge=1, le=1440)


async def _get_or_create_state(session: AsyncSession) -> BotState:
    res = await session.execute(select(BotState).limit(1))
    state = res.scalar_one_or_none()
    if state is None:
        state = BotState()
        state.daily_max_loss_pct = 5.0
        state.max_trades_per_day = 0
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


def _admin_only(handler: Callable[[Message, AsyncSession], Awaitable[None]]):
    async def wrapper(message: Message, session: AsyncSession) -> None:
        if settings.telegram_admin_id and message.from_user and message.from_user.id != settings.telegram_admin_id:
            await message.answer("Not authorized.")
            return
        await handler(message, session)

    return wrapper


class DBSessionMiddleware(BaseMiddleware):
    async def __call__(self, handler, event, data):
        async with SessionLocal() as session:
            data["session"] = session
            return await handler(event, data)


async def _status_text(session: AsyncSession) -> str:
    state = await _get_or_create_state(session)
    res_trades = await session.execute(select(func.count(Trade.id)).where(Trade.close_ts == None))  # type: ignore  # noqa: E711
    open_trades = res_trades.scalar() or 0
    res_pnl = await session.execute(select(func.sum(Trade.pnl)))
    pnl = res_pnl.scalar() or 0.0
    # use plain text, no markdown to avoid parse errors
    return (
        f"Mode: {state.mode}\n"
        f"Halt: {state.halt}, Paused_until: {state.paused_until}\n"
        f"Equity: {state.equity_usd:.2f} USD\n"
        f"Risk/trade: {state.risk_per_trade_pct}% | Daily max loss: {state.daily_max_loss_pct}%\n"
        f"Leverage max: {state.max_leverage} | Trades open: {open_trades}\n"
        f"PnL total: {pnl:.2f} USD"
    )


async def _signals_text(session: AsyncSession) -> str:
    res = await session.execute(select(PatternSignal).order_by(PatternSignal.ts.desc()).limit(1))
    ps = res.scalar_one_or_none()
    if not ps:
        return "No signals yet."
    tf_str = ps.timeframe.value if hasattr(ps.timeframe, "value") else str(ps.timeframe)
    setup_str = ps.market_setup.value if hasattr(ps.market_setup, "value") else str(ps.market_setup)
    return (
        f"Last signal {ps.symbol} {tf_str}\n"
        f"Setup: {setup_str} | Bias: {ps.market_bias:.2f} | Conf: {ps.market_confidence:.2f}\n"
        f"Name: {ps.setup_name}"
    )


async def _news_text(session: AsyncSession) -> str:
    res = await session.execute(select(NewsScore).order_by(NewsScore.ts_scored.desc()).limit(1))
    ns = res.scalar_one_or_none()
    if not ns:
        return "No news scores yet."
    return (
        f"Last news: bias {ns.news_bias:.2f} conf {ns.news_confidence:.2f}\n"
        f"Type: {ns.event_type.value}, horizon: {ns.horizon_minutes}m"
    )


async def _model_text(session: AsyncSession) -> str:
    res = await session.execute(select(ModelPrediction).order_by(ModelPrediction.ts.desc()).limit(1))
    mp = res.scalar_one_or_none()
    version = mp.model_version if mp else "n/a"
    conf_line = (
        f"p_long {mp.p_long:.2f} p_short {mp.p_short:.2f} conf {mp.confidence:.2f}"
        if mp
        else "No predictions yet"
    )
    # best effort read meta
    meta_text = ""
    best_text = ""
    best_found = False
    try:
        from glob import glob
        import json
        paths = sorted(glob("data/models/model_*.json"))
        best_path = "data/models/best_model.json"
        if os.path.exists(best_path):
            with open(best_path) as f:
                best = json.load(f)
            best_found = True
            best_text = (
                f"Current model: {best.get('version','n/a')} ({best.get('model_type','')}) "
                f"precision@0.7: {best.get('metrics',{}).get('precision_conf_ge_0.7',0):.3f} "
                f"n_conf: {best.get('metrics',{}).get('n_conf_ge_0.7',0)}"
            )
        if paths:
            with open(paths[-1]) as f:
                meta = json.load(f)
            metrics = meta.get("metrics", {})
            meta_text = (
                f"Model {meta.get('version', version)} ({meta.get('model_type','')})\n"
                f"precision@0.7: {metrics.get('precision_conf_ge_0.7', 0):.3f}, "
                f"n_conf: {metrics.get('n_conf_ge_0.7', 0)}, test_size: {metrics.get('test_size', 0)}\n"
                f"Top features: {', '.join(meta.get('top_features', []) or [])}"
            )
    except Exception:
        meta_text = ""
    parts = [p for p in [best_text, meta_text] if p]
    if not parts and not best_found:
        parts.append("No model artifacts found")
    parts.append(conf_line)
    return "\n".join(parts)


async def _db_info_text(session: AsyncSession) -> str:
    counts = {}
    tables = [
        ("pattern_signals", PatternSignal),
        ("market_bars", MarketBar),
        ("market_features", MarketFeatures),
        ("trades", Trade),
        ("decisions", Decision),
        ("model_predictions", ModelPrediction),
        ("news_scores", NewsScore),
    ]
    for name, model in tables:
        res = await session.execute(select(func.count()).select_from(model))
        counts[name] = res.scalar() or 0
    lines = [f"{k}: {v}" for k, v in counts.items()]
    return "DB counts:\n" + "\n".join(lines)


async def _redis_info_text() -> str:
    if not redis_client:
        return "Redis not configured"
    try:
        info = await redis_client.info(section="memory")
        dbsize = await redis_client.dbsize()
        streams = {
            "bars": settings.redis_stream_bars,
            "features": settings.redis_stream_features,
            "patterns": settings.redis_stream_patterns,
            "orderbook": settings.redis_stream_orderbook,
            "news_events": settings.redis_stream_news_events,
            "news_scores": settings.redis_stream_news_scores,
        }
        lens = {}
        for k, stream in streams.items():
            try:
                lens[k] = await redis_client.xlen(stream)
            except Exception:
                lens[k] = -1
        lines = [
            f"dbsize: {dbsize}",
            f"used_memory_human: {info.get('used_memory_human', 'n/a')}",
            "streams:",
        ]
        lines.extend([f"- {k}: {v}" for k, v in lens.items()])
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        return f"Redis info error: {exc}"


async def _tf_counts_text(session: AsyncSession) -> str:
    res_bars = await session.execute(
        select(MarketBar.timeframe, func.count()).group_by(MarketBar.timeframe)
    )
    res_feats = await session.execute(
        select(MarketFeatures.timeframe, func.count()).group_by(MarketFeatures.timeframe)
    )
    res_pat = await session.execute(
        select(PatternSignal.timeframe, func.count()).group_by(PatternSignal.timeframe)
    )
    lines = ["Bars:"]
    for tf, cnt in res_bars:
        lines.append(f"  {tf}: {cnt}")
    lines.append("Features:")
    for tf, cnt in res_feats:
        lines.append(f"  {tf}: {cnt}")
    lines.append("Patterns:")
    for tf, cnt in res_pat:
        lines.append(f"  {tf}: {cnt}")
    return "\n".join(lines)


async def _set_bot_commands(bot: Bot) -> None:
    cmds = [
        BotCommand(command="help", description="Список команд"),
        BotCommand(command="status", description="Состояние бота/стратегии"),
        BotCommand(command="tfinfo", description="Статистика по timeframe"),
        BotCommand(command="dbinfo", description="Строки в БД"),
        BotCommand(command="redisinfo", description="Стримы Redis"),
        BotCommand(command="signals", description="Последний сигнал"),
        BotCommand(command="news", description="Последняя новость"),
        BotCommand(command="model", description="Модель/прогноз"),
        BotCommand(command="report", description="PnL день/неделя"),
        BotCommand(command="trades", description="Последние сделки"),
        BotCommand(command="why", description="Причина последнего решения"),
        BotCommand(command="reset_equity", description="Сброс equity до базового (админ)"),
        BotCommand(command="train", description="Обучить модель (админ)"),
        BotCommand(command="reload_model", description="Перезагрузить модель (админ)"),
        BotCommand(command="menu", description="Клавиатура с командами"),
    ]
    await bot.set_my_commands(cmds)


def _menu_keyboard() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton(text="/help"), KeyboardButton(text="/status"), KeyboardButton(text="/tfinfo")],
        [KeyboardButton(text="/dbinfo"), KeyboardButton(text="/redisinfo"), KeyboardButton(text="/signals")],
        [KeyboardButton(text="/news"), KeyboardButton(text="/model"), KeyboardButton(text="/report")],
        [KeyboardButton(text="/trades"), KeyboardButton(text="/why")],
        [KeyboardButton(text="/train"), KeyboardButton(text="/reload_model")],
        [KeyboardButton(text="/reset_equity")],
    ]
    return ReplyKeyboardMarkup(keyboard=rows, resize_keyboard=True)


async def _call_trainer_api(timeframe: Timeframe, horizon_minutes: int, model_type: str) -> str:
    url = settings.trainer_service_url.rstrip("/") + "/train"
    payload = {
        "timeframe": timeframe.value,
        "horizon_minutes": horizon_minutes,
        "model_type": model_type,
    }
    async with aiohttp.ClientSession() as client:
        try:
            async with client.post(url, json=payload, timeout=180) as resp:
                body = await resp.text()
                if resp.status != 200:
                    return f"Train failed ({resp.status}): {body[:500]}"
                data = await resp.json()
        except Exception as exc:  # noqa: BLE001
            return f"Train request error: {exc}"
    metrics = data.get("metrics", {})
    version = data.get("model_version") or "n/a"
    top = data.get("top_features") or []
    metrics_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
    return (
        f"Train ok. version={version}\n"
        f"model_type={model_type} tf={timeframe.value} horizon={horizon_minutes}m\n"
        f"metrics: {metrics_str}\n"
        f"top: {', '.join(top)}"
    )


async def _call_trader_reload() -> str:
    url = settings.trader_service_url.rstrip("/") + "/reload_model"
    async with aiohttp.ClientSession() as client:
        try:
            async with client.post(url, timeout=30) as resp:
                body = await resp.text()
                if resp.status != 200:
                    return f"Reload failed ({resp.status}): {body[:300]}"
                data = await resp.json()
        except Exception as exc:  # noqa: BLE001
            return f"Reload request error: {exc}"
    return f"Reload status: {data.get('status', 'unknown')}"


async def _pnl_between(session: AsyncSession, start: datetime, end: datetime) -> float:
    res = await session.execute(
        select(func.sum(Trade.pnl)).where(
            Trade.close_ts != None,  # type: ignore  # noqa: E711
            Trade.close_ts >= start,
            Trade.close_ts <= end,
        )
    )
    return float(res.scalar() or 0.0)


def register_handlers(dp: Dispatcher) -> None:
    help_text = (
        "Available commands:\n"
        "/help - this help\n"
        "/status - bot state & equity\n"
        "/pause - pause trading for 60m\n"
        "/resume - resume trading\n"
        "/kill - halt trading\n"
        "/risk - risk & limits\n"
        "/signals - last pattern signal\n"
        "/news - last news score\n"
        "/model - last model prediction/meta\n"
        "/dbinfo - DB row counts (patterns/bars/features/trades/decisions/...)\n"
        "/tfinfo - DB counts grouped by timeframe\n"
        "/redisinfo - Redis stats & stream lengths\n"
        "/report - PnL day/week\n"
        "/trades - last 10 trades\n"
        "/why - last decision explanation\n"
        "/reset_equity - reset equity to base (admin)\n"
        "/train [tf] [horizon] [model] - train model (admin)\n"
        "/reload_model - reload best model in trader (admin)"
    )

    @dp.message(Command("help"))
    async def cmd_help(message: Message, session: AsyncSession) -> None:  # session injected for consistency
        await message.answer(help_text)

    @dp.message(Command("status"))
    async def cmd_status(message: Message, session: AsyncSession) -> None:
        await message.answer(await _status_text(session))

    @dp.message(Command("pause"))
    async def cmd_pause(message: Message, session: AsyncSession) -> None:
        state = await _get_or_create_state(session)
        state.paused_until = datetime.now(timezone.utc) + timedelta(minutes=60)
        await session.commit()
        await message.answer("Paused for 60 minutes")

    @dp.message(Command("resume"))
    async def cmd_resume(message: Message, session: AsyncSession) -> None:
        state = await _get_or_create_state(session)
        state.paused_until = None
        state.halt = False
        await session.commit()
        await message.answer("Resumed")

    @dp.message(Command("kill"))
    async def cmd_kill(message: Message, session: AsyncSession) -> None:
        state = await _get_or_create_state(session)
        state.halt = True
        await session.commit()
        await message.answer("Halt set")

    @dp.message(Command("risk"))
    async def cmd_risk(message: Message, session: AsyncSession) -> None:
        state = await _get_or_create_state(session)
        max_trades_text = "unlimited" if state.max_trades_per_day <= 0 else str(state.max_trades_per_day)
        await message.answer(
            f"Risk/trade: {state.risk_per_trade_pct}%\n"
            f"Daily max loss: {state.daily_max_loss_pct}%\n"
            f"Max leverage: {state.max_leverage}\n"
            f"Max trades/day: {max_trades_text}"
        )

    @dp.message(Command("signals"))
    async def cmd_signals(message: Message, session: AsyncSession) -> None:
        await message.answer(await _signals_text(session))

    @dp.message(Command("news"))
    async def cmd_news(message: Message, session: AsyncSession) -> None:
        await message.answer(await _news_text(session))

    @dp.message(Command("model"))
    async def cmd_model(message: Message, session: AsyncSession) -> None:
        await message.answer(await _model_text(session))

    @dp.message(Command("dbinfo"))
    async def cmd_dbinfo(message: Message, session: AsyncSession) -> None:
        await message.answer(await _db_info_text(session))

    @dp.message(Command("tfinfo"))
    async def cmd_tfinfo(message: Message, session: AsyncSession) -> None:
        await message.answer(await _tf_counts_text(session))

    @dp.message(Command("redisinfo"))
    async def cmd_redisinfo(message: Message, session: AsyncSession) -> None:
        await message.answer(await _redis_info_text())

    @dp.message(Command("menu"))
    async def cmd_menu(message: Message, session: AsyncSession) -> None:
        await message.answer("Команды на клавиатуре", reply_markup=_menu_keyboard())

    @dp.message(Command("report"))
    async def cmd_report(message: Message, session: AsyncSession) -> None:
        now = datetime.now(timezone.utc)
        start_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        start_week = start_day - timedelta(days=7)
        pnl_day = await _pnl_between(session, start_day, now)
        pnl_week = await _pnl_between(session, start_week, now)
        await message.answer(
            f"PnL day: {pnl_day:.2f} USD\n"
            f"PnL week: {pnl_week:.2f} USD"
        )

    @dp.message(Command("trades"))
    async def cmd_trades(message: Message, session: AsyncSession) -> None:
        res = await session.execute(
            select(Trade)
            .order_by(Trade.open_ts.desc())
            .limit(10)
        )
        trades = res.scalars().all()
        if not trades:
            await message.answer("No trades yet.")
            return
        lines = []
        for t in trades:
            lines.append(
                f"{t.trade_id[:6]} {t.side.value} qty {t.qty or 0:.4f} "
                f"entry {t.entry_px:.4f} exit {t.exit_px or 0:.4f} pnl {t.pnl or 0:.2f} reason {t.close_reason}"
            )
        await message.answer("\n".join(lines))

    @dp.message(Command("why"))
    async def cmd_why(message: Message, session: AsyncSession) -> None:
        res = await session.execute(
            select(Decision, ModelPrediction)
            .join(ModelPrediction, ModelPrediction.ts == Decision.ts, isouter=True)
            .order_by(Decision.ts.desc())
            .limit(1)
        )
        row = res.first()
        if not row:
            await message.answer("No decisions yet.")
            return
        dec, mp = row
        txt = (
            f"Decision: {dec.decision.value} {dec.symbol}\n"
            f"Reason: {dec.decision_reason}\n"
            f"Used ML: {dec.used_ml} | Model: {dec.model_version}\n"
        )
        if mp:
            txt += (
                f"p_long {mp.p_long:.2f} p_short {mp.p_short:.2f} conf {mp.confidence:.2f}\n"
                f"Top: {', '.join(mp.top_features or [])}"
            )
        await message.answer(txt)

    @dp.message(Command("train"))
    @_admin_only
    async def cmd_train(message: Message, session: AsyncSession) -> None:
        parts = (message.text or "").split()
        tf_raw = parts[1] if len(parts) > 1 else Timeframe.m5.value
        try:
            tf = Timeframe(tf_raw)
        except ValueError:
            await message.answer("Bad timeframe. Use 1m / 5m / 15m / 1h")
            return
        try:
            horizon = int(parts[2]) if len(parts) > 2 else 30
        except ValueError:
            await message.answer("Bad horizon. Provide integer minutes, e.g. 30")
            return
        model_type = parts[3] if len(parts) > 3 else "gbm"
        await message.answer(f"Training started: tf={tf.value} horizon={horizon}m model={model_type}")
        result = await _call_trainer_api(tf, horizon, model_type)
        await message.answer(result)

    @dp.message(Command("reload_model"))
    @_admin_only
    async def cmd_reload_model(message: Message, session: AsyncSession) -> None:
        await message.answer("Reloading model in trader_service...")
        result = await _call_trader_reload()
        await message.answer(result)

    @dp.message(Command("reset_equity"))
    @_admin_only
    async def cmd_reset_equity(message: Message, session: AsyncSession) -> None:
        state = await _get_or_create_state(session)
        state.equity_usd = settings.base_equity_usd
        await session.commit()
        await session.refresh(state)
        await message.answer(f"Equity reset to {state.equity_usd:.2f} USD")


async def report_loop() -> None:
    while True:
        try:
            async with SessionLocal() as session:
                now = datetime.now(timezone.utc)
                start_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
                start_week = start_day - timedelta(days=7)
                pnl_day = await _pnl_between(session, start_day, now)
                pnl_week = await _pnl_between(session, start_week, now)
                await send_telegram_message(
                    f"Auto report\nPnL day: {pnl_day:.2f} USD\nPnL week: {pnl_week:.2f} USD"
                )
            # sleep until next report hour UTC
            now = datetime.now(timezone.utc)
            target = now.replace(hour=settings.report_hour_utc, minute=0, second=0, microsecond=0)
            if target <= now:
                target += timedelta(days=1)
            await asyncio.sleep((target - now).total_seconds())
        except Exception as exc:  # noqa: BLE001
            logger.warning("report loop failed", error=str(exc))
            await asyncio.sleep(300)


async def aiogram_polling() -> None:
    assert bot and dp
    await dp.start_polling(bot)


@app.on_event("startup")
async def on_startup() -> None:
    logger.info("telegram_bot_service started", env=settings.app_env)
    global bot, dp, redis_client
    if not settings.telegram_bot_token:
        logger.warning("telegram bot token not set; bot disabled")
        return
    redis_client = Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        password=settings.redis_password,
        ssl=settings.redis_ssl,
        decode_responses=True,
    )
    bot = Bot(
        token=settings.telegram_bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()
    dp.message.middleware(DBSessionMiddleware())
    register_handlers(dp)
    await _set_bot_commands(bot)
    if settings.telegram_polling:
        asyncio.create_task(aiogram_polling())
    asyncio.create_task(report_loop())


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "env": settings.app_env}


@app.get("/status", response_model=BotStateOut)
async def status(session: SessionDep) -> BotStateOut:
    state = await _get_or_create_state(session)
    return BotStateOut.model_validate(state)


@app.post("/pause", response_model=BotStateOut)
async def pause(req: PauseRequest, session: SessionDep) -> BotStateOut:
    state = await _get_or_create_state(session)
    state.paused_until = datetime.now(timezone.utc) + timedelta(minutes=req.minutes)
    await session.commit()
    await session.refresh(state)
    logger.info("bot paused", minutes=req.minutes)
    return BotStateOut.model_validate(state)


@app.post("/resume", response_model=BotStateOut)
async def resume(session: SessionDep) -> BotStateOut:
    state = await _get_or_create_state(session)
    state.paused_until = None
    state.halt = False
    await session.commit()
    await session.refresh(state)
    logger.info("bot resumed")
    return BotStateOut.model_validate(state)


@app.post("/kill", response_model=BotStateOut)
async def kill(session: SessionDep) -> BotStateOut:
    state = await _get_or_create_state(session)
    state.halt = True
    await session.commit()
    await session.refresh(state)
    logger.warning("bot halted via kill")
    return BotStateOut.model_validate(state)


