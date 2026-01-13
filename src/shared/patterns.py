from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from shared.schemas import MarketSetup, PatternSignal, Regime, Timeframe


@dataclass
class PatternResult:
    market_bias: float
    market_confidence: float
    market_setup: MarketSetup
    setup_name: str | None = None


def detect_patterns(
    *,
    symbol: str,
    timeframe: Timeframe,
    close: float,
    ema20: float | None,
    ema50: float | None,
    ema200: float | None,
    rsi: float | None,
    atr_pct: float | None,
    vol: float | None,
    regime_val: Regime,
    last_breakout_range: float | None = None,
    meanrev_blocked: bool = False,
) -> Optional[PatternResult]:
    """Heuristic detectors for trend-follow, breakout, mean-revert."""
    # Trend-follow: price vs EMA200 filter + pullback toward EMA20/50
    if ema200 is not None:
        # uptrend
        if close > ema200:
            bias = 0.7
            conf = 0.55
            if ema50 and close > ema50 and ema20 and close > ema20:
                conf = 0.68
            return PatternResult(
                market_bias=bias,
                market_confidence=conf,
                market_setup=MarketSetup.trend_follow,
                setup_name="trend_follow_up",
            )
        # downtrend
        if close < ema200:
            bias = -0.7
            conf = 0.55
            if ema50 and close < ema50 and ema20 and close < ema20:
                conf = 0.68
            return PatternResult(
                market_bias=bias,
                market_confidence=conf,
                market_setup=MarketSetup.trend_follow,
                setup_name="trend_follow_down",
            )

    # Breakout: flat regime with low vol/ATR and optional range width
    if regime_val == Regime.range and vol is not None and atr_pct is not None:
        if vol < 0.0025 and atr_pct < 0.01:
            bias = 0.5
            conf = 0.55
            if last_breakout_range:
                conf = min(0.8, 0.55 + min(last_breakout_range, 0.02))
            return PatternResult(
                market_bias=bias,
                market_confidence=conf,
                market_setup=MarketSetup.breakout,
                setup_name="breakout_flat_vol_low",
            )

    # Mean reversion: only in RANGE, RSI extremes, not blocked
    if not meanrev_blocked and regime_val == Regime.range and rsi is not None:
        if rsi < 30:
            return PatternResult(
                market_bias=0.45,
                market_confidence=0.5,
                market_setup=MarketSetup.mean_revert,
                setup_name="meanrev_rsi_oversold",
            )
        if rsi > 70:
            return PatternResult(
                market_bias=-0.45,
                market_confidence=0.5,
                market_setup=MarketSetup.mean_revert,
                setup_name="meanrev_rsi_overbought",
            )
    return None

