from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base
from .schemas import DecisionSide, EventType, MarketSetup, Regime, Timeframe


def _enum_values(enum_cls):
    return [member.value for member in enum_cls]


DECISION_SIDE_ENUM = Enum(
    DecisionSide,
    values_callable=_enum_values,
    name="decisionside",
)

MARKET_SETUP_ENUM = Enum(
    MarketSetup,
    values_callable=_enum_values,
    name="marketsetup",
)

EVENT_TYPE_ENUM = Enum(
    EventType,
    values_callable=_enum_values,
    name="eventtype",
)


class MarketBar(Base):
    __tablename__ = "market_bars"
    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "ts", name="uq_market_bars_symbol_tf_ts"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(16))
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float, index=True)
    volume: Mapped[float] = mapped_column(Float)


class MarketFeatures(Base):
    __tablename__ = "market_features"
    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "ts", name="uq_market_features_symbol_tf_ts"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(16))
    atr_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    ema20: Mapped[float | None] = mapped_column(Float, nullable=True)
    ema50: Mapped[float | None] = mapped_column(Float, nullable=True)
    ema200: Mapped[float | None] = mapped_column(Float, nullable=True)
    rsi: Mapped[float | None] = mapped_column(Float, nullable=True)
    returns: Mapped[float | None] = mapped_column(Float, nullable=True)
    vol: Mapped[float | None] = mapped_column(Float, nullable=True)
    spread: Mapped[float | None] = mapped_column(Float, nullable=True)
    # store regime as plain text to avoid enum mismatch during ingestion
    regime: Mapped[str] = mapped_column(String(16), default=Regime.unknown.value, nullable=False)


class PatternSignal(Base):
    __tablename__ = "pattern_signals"
    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "ts", name="uq_pattern_symbol_tf_ts"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(16))
    market_bias: Mapped[float] = mapped_column(Float)
    market_confidence: Mapped[float] = mapped_column(Float)
    market_setup: Mapped[MarketSetup] = mapped_column(MARKET_SETUP_ENUM)
    setup_name: Mapped[str | None] = mapped_column(String(64), nullable=True)


class NewsEvent(Base):
    __tablename__ = "news_events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ts_received: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    source: Mapped[str] = mapped_column(String(64))
    headline: Mapped[str] = mapped_column(Text)
    body: Mapped[str | None] = mapped_column(Text, nullable=True)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    scores: Mapped[list["NewsScore"]] = relationship(back_populates="news", cascade="all,delete")


class NewsScore(Base):
    __tablename__ = "news_scores"
    __table_args__ = (UniqueConstraint("news_id", "ts_scored", name="uq_news_scores_news_ts"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    news_id: Mapped[str] = mapped_column(String(64), ForeignKey("news_events.id"), index=True)
    ts_scored: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    event_type: Mapped[EventType] = mapped_column(EVENT_TYPE_ENUM)
    news_bias: Mapped[float] = mapped_column(Float)
    news_confidence: Mapped[float] = mapped_column(Float)
    horizon_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reason_tags: Mapped[list[str]] = mapped_column(ARRAY(String(64)), default=list)
    model_used: Mapped[str] = mapped_column(String(16), default="rules")

    news: Mapped[NewsEvent] = relationship(back_populates="scores")


class ModelPrediction(Base):
    __tablename__ = "model_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    p_long: Mapped[float] = mapped_column(Float)
    p_short: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float, index=True)
    model_version: Mapped[str] = mapped_column(String(64))
    top_features: Mapped[list[str]] = mapped_column(ARRAY(String(64)), default=list)


class Decision(Base):
    __tablename__ = "decisions"

    decision_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    decision: Mapped[DecisionSide] = mapped_column(DECISION_SIDE_ENUM)
    decision_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    used_market: Mapped[bool] = mapped_column(Boolean, default=False)
    used_news: Mapped[bool] = mapped_column(Boolean, default=False)
    used_ml: Mapped[bool] = mapped_column(Boolean, default=False)
    thresholds: Mapped[dict[str, float]] = mapped_column(JSON, default=dict)
    rule_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)

    trade_plan: Mapped["TradePlan"] = relationship(back_populates="decision", uselist=False)
    trades: Mapped[list["Trade"]] = relationship(back_populates="decision")


class TradePlan(Base):
    __tablename__ = "trade_plans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    decision_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("decisions.decision_id"), unique=True, index=True
    )
    entry_type: Mapped[str] = mapped_column(String(16))
    entry_price_ref: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    take_profit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    notional_usd: Mapped[float] = mapped_column(Float)
    leverage: Mapped[float | None] = mapped_column(Float, nullable=True)
    risk_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    constraints_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    decision: Mapped[Decision] = relationship(back_populates="trade_plan")


class Trade(Base):
    __tablename__ = "trades"
    __table_args__ = (UniqueConstraint("trade_id", name="uq_trades_trade_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_id: Mapped[str] = mapped_column(String(64), unique=True)
    decision_id: Mapped[str] = mapped_column(String(64), ForeignKey("decisions.decision_id"))
    open_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    close_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    side: Mapped[DecisionSide] = mapped_column(DECISION_SIDE_ENUM)
    qty: Mapped[float | None] = mapped_column(Float, nullable=True)
    notional: Mapped[float | None] = mapped_column(Float, nullable=True)
    entry_px: Mapped[float] = mapped_column(Float)
    exit_px: Mapped[float | None] = mapped_column(Float, nullable=True)
    fees_est: Mapped[float | None] = mapped_column(Float, nullable=True)
    slippage_est: Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_adverse: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_favorable: Mapped[float | None] = mapped_column(Float, nullable=True)
    close_reason: Mapped[str] = mapped_column(
        String(16), default="open"
    )  # stop/tp/manual/timeout/open

    decision: Mapped[Decision] = relationship(back_populates="trades")


class BotState(Base):
    __tablename__ = "bot_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    mode: Mapped[str] = mapped_column(String(16), default="paper")  # paper/demo
    paused_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    halt: Mapped[bool] = mapped_column(Boolean, default=False)
    risk_per_trade_pct: Mapped[float] = mapped_column(Float, default=0.5)
    daily_max_loss_pct: Mapped[float] = mapped_column(Float, default=2.0)
    max_leverage: Mapped[float] = mapped_column(Float, default=10.0)
    max_trades_per_day: Mapped[int] = mapped_column(Integer, default=10)
    cooldown_after_losses_min: Mapped[float] = mapped_column(Float, default=60.0)
    max_position_time_min: Mapped[int] = mapped_column(Integer, default=360)  # 6 hours
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    equity_usd: Mapped[float] = mapped_column(Float, default=10000.0)

