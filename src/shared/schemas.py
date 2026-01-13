from datetime import datetime
from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, Field


class Timeframe(str, Enum):
    m1 = "1m"
    m5 = "5m"
    m15 = "15m"
    h1 = "1h"


class MarketSetup(str, Enum):
    trend_follow = "trend_follow"
    breakout = "breakout"
    mean_revert = "mean_revert"
    none = "none"


class EventType(str, Enum):
    earnings = "earnings"
    macro = "macro"
    company = "company"
    regulation = "regulation"
    other = "other"


class DecisionSide(str, Enum):
    long = "LONG"
    short = "SHORT"
    hold = "HOLD"


class Regime(str, Enum):
    trend_up = "TREND_UP"
    trend_down = "TREND_DOWN"
    range = "RANGE"
    unknown = "UNKNOWN"


class MarketBar(BaseModel):
    ts: datetime
    symbol: str
    timeframe: Timeframe
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketFeatures(BaseModel):
    ts: datetime
    symbol: str
    timeframe: Timeframe
    atr_pct: float | None = None
    ema20: float | None = None
    ema50: float | None = None
    ema200: float | None = None
    rsi: float | None = None
    returns: float | None = None
    vol: float | None = None
    spread: float | None = None
    regime: Regime = Regime.unknown


class PatternSignal(BaseModel):
    ts: datetime
    symbol: str
    timeframe: Timeframe
    market_bias: float = Field(..., ge=-1.0, le=1.0)
    market_confidence: float = Field(..., ge=0.0, le=1.0)
    market_setup: MarketSetup
    setup_name: str | None = None
    disabled_reason: str | None = None


class NewsEvent(BaseModel):
    id: str
    ts_received: datetime
    source: str
    headline: str
    body: str | None = None
    url: str | None = None
    raw_json: dict | None = None


class NewsScore(BaseModel):
    news_id: str
    ts_scored: datetime
    event_type: EventType
    news_bias: float = Field(..., ge=-1.0, le=1.0)
    news_confidence: float = Field(..., ge=0.0, le=1.0)
    horizon_minutes: int | None = None
    reason_tags: list[str] = Field(default_factory=list)
    model_used: Literal["rules", "llm"] = "rules"


class ModelPrediction(BaseModel):
    ts: datetime
    symbol: str
    p_long: float = Field(..., ge=0.0, le=1.0)
    p_short: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    top_features: list[str] = Field(default_factory=list)


class Decision(BaseModel):
    decision_id: str
    ts: datetime
    symbol: str
    decision: DecisionSide
    decision_reason: str | None = None
    used_market: bool = False
    used_news: bool = False
    used_ml: bool = False
    thresholds: dict[str, float] = Field(default_factory=dict)
    rule_version: str | None = None
    model_version: str | None = None


class TradePlan(BaseModel):
    decision_id: str
    entry_type: Literal["market", "limit"]
    entry_price_ref: float | None = None
    stop_price: float | None = None
    take_profit_price: float | None = None
    notional_usd: float
    leverage: float | None = None
    risk_usd: float | None = None
    constraints_snapshot: dict | None = None


class Trade(BaseModel):
    trade_id: str
    decision_id: str
    open_ts: datetime
    close_ts: Optional[datetime] = None
    side: DecisionSide
    qty: float | None = None
    notional: float | None = None
    entry_px: float
    exit_px: float | None = None
    fees_est: float | None = None
    slippage_est: float | None = None
    pnl: float | None = None
    pnl_pct: float | None = None
    max_adverse: float | None = None
    max_favorable: float | None = None
    close_reason: Literal["stop", "tp", "manual", "timeout", "open"] = "open"

