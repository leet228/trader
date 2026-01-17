from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Annotated, Awaitable, Callable

from aiogram import Bot, Dispatcher
from aiogram import BaseMiddleware
from aiogram.client.bot import DefaultBotProperties
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
from shared.models import BotState, Decision, ModelPrediction, NewsScore, PatternSignal, Trade
from shared.notify import send_telegram_message

configure_logging()
settings = get_settings()

app = FastAPI(title="Telegram Bot Service", version="0.1.0")

SessionDep = Annotated[AsyncSession, Depends(get_session)]
bot: Bot | None = None
dp: Dispatcher | None = None


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
        session.add(state)
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
    return (
        f"Last signal {ps.symbol} {ps.timeframe.value}\n"
        f"Setup: {ps.market_setup.value} | Bias: {ps.market_bias:.2f} | Conf: {ps.market_confidence:.2f}\n"
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
    try:
        from glob import glob
        import json
        paths = sorted(glob("data/models/model_*.json"))
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
    return f"{meta_text}\n{conf_line}"


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
    @dp.message(Command("status"))
    async def cmd_status(message: Message, session: SessionDep) -> None:
        await message.answer(await _status_text(session))

    @dp.message(Command("pause"))
    async def cmd_pause(message: Message, session: SessionDep) -> None:
        state = await _get_or_create_state(session)
        state.paused_until = datetime.now(timezone.utc) + timedelta(minutes=60)
        await session.commit()
        await message.answer("Paused for 60 minutes")

    @dp.message(Command("resume"))
    async def cmd_resume(message: Message, session: SessionDep) -> None:
        state = await _get_or_create_state(session)
        state.paused_until = None
        state.halt = False
        await session.commit()
        await message.answer("Resumed")

    @dp.message(Command("kill"))
    async def cmd_kill(message: Message, session: SessionDep) -> None:
        state = await _get_or_create_state(session)
        state.halt = True
        await session.commit()
        await message.answer("Halt set")

    @dp.message(Command("risk"))
    async def cmd_risk(message: Message, session: SessionDep) -> None:
        state = await _get_or_create_state(session)
        await message.answer(
            f"Risk/trade: {state.risk_per_trade_pct}%\n"
            f"Daily max loss: {state.daily_max_loss_pct}%\n"
            f"Max leverage: {state.max_leverage}\n"
            f"Max trades/day: {state.max_trades_per_day}"
        )

    @dp.message(Command("signals"))
    async def cmd_signals(message: Message, session: SessionDep) -> None:
        await message.answer(await _signals_text(session))

    @dp.message(Command("news"))
    async def cmd_news(message: Message, session: SessionDep) -> None:
        await message.answer(await _news_text(session))

    @dp.message(Command("model"))
    async def cmd_model(message: Message, session: SessionDep) -> None:
        await message.answer(await _model_text(session))

    @dp.message(Command("report"))
    async def cmd_report(message: Message, session: SessionDep) -> None:
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
    async def cmd_trades(message: Message, session: SessionDep) -> None:
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
    async def cmd_why(message: Message, session: SessionDep) -> None:
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


async def _pnl_between(session: AsyncSession, start: datetime, end: datetime) -> float:
    res = await session.execute(
        select(func.sum(Trade.pnl)).where(
            Trade.close_ts != None,  # type: ignore  # noqa: E711
            Trade.close_ts >= start,
            Trade.close_ts <= end,
        )
    )
    return float(res.scalar() or 0.0)


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
    global bot, dp
    if not settings.telegram_bot_token:
        logger.warning("telegram bot token not set; bot disabled")
        return
    bot = Bot(
        token=settings.telegram_bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()
    dp.message.middleware(DBSessionMiddleware())
    register_handlers(dp)
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


