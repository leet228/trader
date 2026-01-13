from collections import deque
from typing import Deque, Iterable

import numpy as np


def _ema(values: Iterable[float], period: int) -> float | None:
    arr = np.fromiter(values, dtype=float)
    if arr.size < period:
        return None
    # numpy EMA via recursive formula
    alpha = 2 / (period + 1)
    ema_val = arr[0]
    for v in arr[1:]:
        ema_val = alpha * v + (1 - alpha) * ema_val
    return float(ema_val)


def ema(values: Iterable[float], period: int) -> float | None:
    return _ema(values, period)


def rsi(values: Iterable[float], period: int = 14) -> float | None:
    arr = np.fromiter(values, dtype=float)
    if arr.size <= period:
        return None
    deltas = np.diff(arr)
    ups = np.where(deltas > 0, deltas, 0.0)
    downs = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(ups[-period:])
    avg_loss = np.mean(downs[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def atr_pct(highs: Iterable[float], lows: Iterable[float], closes: Iterable[float], period: int = 14) -> float | None:
    highs_arr = np.fromiter(highs, dtype=float)
    lows_arr = np.fromiter(lows, dtype=float)
    closes_arr = np.fromiter(closes, dtype=float)
    if highs_arr.size <= period:
        return None
    trs = []
    for i in range(1, highs_arr.size):
        tr = max(
            highs_arr[i] - lows_arr[i],
            abs(highs_arr[i] - closes_arr[i - 1]),
            abs(lows_arr[i] - closes_arr[i - 1]),
        )
        trs.append(tr)
    atr = np.mean(trs[-period:])
    if closes_arr.size == 0:
        return None
    return float(atr / closes_arr[-1]) if closes_arr[-1] != 0 else None


def realized_vol(returns: Iterable[float]) -> float | None:
    arr = np.fromiter(returns, dtype=float)
    if arr.size == 0:
        return None
    return float(np.std(arr))


def regime(ema200: float | None, ema200_prev: float | None, slope_eps: float = 1e-4) -> str:
    if ema200 is None or ema200_prev is None:
        return "UNKNOWN"
    slope = ema200 - ema200_prev
    if abs(slope) < slope_eps:
        return "RANGE"
    return "TREND_UP" if slope > 0 else "TREND_DOWN"


class RollingWindow:
    """Fixed-size rolling window for numeric values."""

    def __init__(self, maxlen: int):
        self.values: Deque[float] = deque(maxlen=maxlen)

    def append(self, value: float) -> None:
        self.values.append(value)

    def to_list(self) -> list[float]:
        return list(self.values)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.values)

