from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_env: str = "local"
    log_level: str = "INFO"

    bybit_ws_url: str = "wss://stream.bybit.com/v5/public/linear"
    symbols: list[str] = ["BTCUSDT"]
    timeframe: str = "1"  # Bybit kline interval (1=1m)
    agg_timeframes: list[int] = [1, 5, 15, 60]  # in minutes
    ws_lag_warn_ms: int = 2000

    redis_stream_bars: str = "market_bars"
    redis_stream_features: str = "market_features"
    redis_stream_patterns: str = "pattern_signals"
    redis_stream_orderbook: str = "orderbook_updates"
    redis_xadd_maxlen: int = 5000

    meanrev_block_news_minutes: int = 30  # placeholder, integrate with news later

    base_equity_usd: float = 10000.0
    commission_rate: float = 0.0006  # 6 bps per side
    slippage_bps: float = 5  # 5 bps
    spread_bps: float = 5  # 5 bps
    slippage_vol_mult: float = 10.0  # extra bps per 1.0 vol
    orderbook_enable: bool = True

    enable_shap: bool = False

    alert_daily_limit_channel: str | None = None  # telegram chat id or email placeholder
    alert_errors_channel: str | None = None

    redis_stream_news_events: str = "news_events_stream"
    redis_stream_news_scores: str = "news_scores_stream"

    rss_urls: list[str] = [
        "https://news.google.com/rss/search?q=crypto",
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
    ]
    newsapi_key: str | None = None
    newsapi_query: str = (
        "crypto OR bitcoin OR ethereum OR blockchain OR defi OR web3 OR btc OR eth "
        "OR binance OR coinbase OR sec OR etf OR regulation OR fed OR inflation OR rates "
        "OR stocks OR equities OR nasdaq OR sp500 OR macro OR recession OR war OR politics OR sanctions"
    )
    news_poll_seconds: int = 300

    report_hour_utc: int = 15  # 18:00 MSK (~UTC+3)
    postgres_user: str = "trader"
    postgres_password: str = "trader"
    postgres_db: str = "trader"
    postgres_host: str = "postgres"
    postgres_port: int = 5432

    redis_host: str = "redis"
    redis_port: int = 6379
    redis_password: str | None = None
    redis_ssl: bool = False

    trainer_service_url: str = "http://trainer_service:8006"
    trader_service_url: str = "http://trader_service:8005"

    telegram_bot_token: str | None = None
    telegram_admin_id: int | None = None
    telegram_polling: bool = True

    model_confidence_threshold: float = 0.65

    class Config:
        env_prefix = ""
        case_sensitive = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]

