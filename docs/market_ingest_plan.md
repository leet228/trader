# Market Ingest & Features Plan (Phase 2)

## Data flow
- Source: Bybit WebSocket (ticker/agg trades/book TBA), fallback REST for backfill.
- Topics: `kline.1m` primary; aggregate to 5m/15m/1h in-process.
- Publish: Redis Streams `market_bars` and `market_features`.
- Storage: insert bars/features into Postgres asynchronously (batched).

## In-memory state (market_service)
- Latest bid/ask/last for each symbol.
- Rolling windows for ATR, RSI, EMA20/50/200, realized vol, returns.
- Regime detection inputs (slope of EMA200, range vs trend).

## Feature set (initial)
- Returns: 1m, 5m, 15m log returns.
- Volatility: rolling std / realized vol (5m, 15m).
- ATR_pct: ATR / close.
- EMA: 20/50/200; slopes for 50/200.
- RSI: 14-period on 1m or 5m.
- Spread: if book is present; else null.
- Regime: RANGE if |slope(EMA200)| < eps and price within band; TREND_UP/DOWN otherwise.

## Persistence cadence
- Every 1m: bar + features row.
- Every 5m/15m: aggregated bar/feature derived from 1m buffer.
- Backpressure: queue to Redis, worker flushes to Postgres in batches (e.g., 100 msgs).

## Pattern_service expectations
- Input: `market_features` stream.
- Output: `pattern_signals` with market_bias/confidence/setup_name.
- Trend-follow: price vs EMA200 filter, pullback to EMA20/50 + minor breakout confirm.
- Breakout: low vol + range; breakout candle on volume/impulse.
- Mean-revert: only in RANGE; RSI extreme + re-entry, disabled if fresh negative news.

## Health/observability
- /health returns latest ts per symbol/timeframe.
- Metrics: feature calc latency, Redis backlog size, WS reconnects count.

