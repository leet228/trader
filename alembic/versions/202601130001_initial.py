"""initial schema

Revision ID: 202601130001
Revises:
Create Date: 2026-01-13 00:01:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "202601130001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    timeframe_enum = sa.Enum("1m", "5m", "15m", "1h", name="timeframe")
    regime_enum = sa.Enum("TREND_UP", "TREND_DOWN", "RANGE", "UNKNOWN", name="regime")
    market_setup_enum = sa.Enum(
        "trend_follow", "breakout", "mean_revert", "none", name="marketsetup"
    )
    event_type_enum = sa.Enum("earnings", "macro", "company", "regulation", "other", name="eventtype")
    decision_side_enum = sa.Enum("LONG", "SHORT", "HOLD", name="decisionside")

    op.create_table(
        "market_bars",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("symbol", sa.String(length=32), nullable=False, index=True),
        sa.Column("timeframe", timeframe_enum, nullable=False),
        sa.Column("open", sa.Float(), nullable=False),
        sa.Column("high", sa.Float(), nullable=False),
        sa.Column("low", sa.Float(), nullable=False),
        sa.Column("close", sa.Float(), nullable=False, index=True),
        sa.Column("volume", sa.Float(), nullable=False),
        sa.UniqueConstraint("symbol", "timeframe", "ts", name="uq_market_bars_symbol_tf_ts"),
    )

    op.create_table(
        "market_features",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("symbol", sa.String(length=32), nullable=False, index=True),
        sa.Column("timeframe", timeframe_enum, nullable=False),
        sa.Column("atr_pct", sa.Float(), nullable=True),
        sa.Column("ema20", sa.Float(), nullable=True),
        sa.Column("ema50", sa.Float(), nullable=True),
        sa.Column("ema200", sa.Float(), nullable=True),
        sa.Column("rsi", sa.Float(), nullable=True),
        sa.Column("returns", sa.Float(), nullable=True),
        sa.Column("vol", sa.Float(), nullable=True),
        sa.Column("spread", sa.Float(), nullable=True),
        sa.Column("regime", regime_enum, nullable=False, server_default="UNKNOWN"),
        sa.UniqueConstraint("symbol", "timeframe", "ts", name="uq_market_features_symbol_tf_ts"),
    )

    op.create_table(
        "news_events",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("ts_received", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("headline", sa.Text(), nullable=False),
        sa.Column("body", sa.Text(), nullable=True),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column("raw_json", postgresql.JSONB(), nullable=True),
    )

    op.create_table(
        "model_predictions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("symbol", sa.String(length=32), nullable=False, index=True),
        sa.Column("p_long", sa.Float(), nullable=False),
        sa.Column("p_short", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False, index=True),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("top_features", postgresql.ARRAY(sa.String(length=64)), server_default="{}", nullable=False),
    )

    op.create_table(
        "decisions",
        sa.Column("decision_id", sa.String(length=64), primary_key=True),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("symbol", sa.String(length=32), nullable=False, index=True),
        sa.Column("decision", decision_side_enum, nullable=False),
        sa.Column("decision_reason", sa.Text(), nullable=True),
        sa.Column("used_market", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("used_news", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("used_ml", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("thresholds", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("rule_version", sa.String(length=64), nullable=True),
        sa.Column("model_version", sa.String(length=64), nullable=True),
    )

    op.create_table(
        "pattern_signals",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("symbol", sa.String(length=32), nullable=False, index=True),
        sa.Column("timeframe", timeframe_enum, nullable=False),
        sa.Column("market_bias", sa.Float(), nullable=False),
        sa.Column("market_confidence", sa.Float(), nullable=False),
        sa.Column("market_setup", market_setup_enum, nullable=False),
        sa.Column("setup_name", sa.String(length=64), nullable=True),
        sa.UniqueConstraint("symbol", "timeframe", "ts", name="uq_pattern_symbol_tf_ts"),
    )

    op.create_table(
        "bot_state",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("mode", sa.String(length=16), nullable=False, server_default="paper"),
        sa.Column("paused_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("halt", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("risk_per_trade_pct", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("daily_max_loss_pct", sa.Float(), nullable=False, server_default="2.0"),
        sa.Column("max_leverage", sa.Float(), nullable=False, server_default="10.0"),
        sa.Column("max_trades_per_day", sa.Integer(), nullable=False, server_default="10"),
        sa.Column("cooldown_after_losses_min", sa.Float(), nullable=False, server_default="60"),
        sa.Column("max_position_time_min", sa.Integer(), nullable=False, server_default="360"),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.Column("equity_usd", sa.Float(), nullable=False, server_default="10000.0"),
    )

    op.create_table(
        "news_scores",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("news_id", sa.String(length=64), sa.ForeignKey("news_events.id"), index=True, nullable=False),
        sa.Column("ts_scored", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("event_type", event_type_enum, nullable=False),
        sa.Column("news_bias", sa.Float(), nullable=False),
        sa.Column("news_confidence", sa.Float(), nullable=False),
        sa.Column("horizon_minutes", sa.Integer(), nullable=True),
        sa.Column("reason_tags", postgresql.ARRAY(sa.String(length=64)), server_default="{}", nullable=False),
        sa.Column("model_used", sa.String(length=16), nullable=False, server_default="rules"),
        sa.UniqueConstraint("news_id", "ts_scored", name="uq_news_scores_news_ts"),
    )

    op.create_table(
        "trade_plans",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("decision_id", sa.String(length=64), sa.ForeignKey("decisions.decision_id"), nullable=False, unique=True, index=True),
        sa.Column("entry_type", sa.String(length=16), nullable=False),
        sa.Column("entry_price_ref", sa.Float(), nullable=True),
        sa.Column("stop_price", sa.Float(), nullable=True),
        sa.Column("take_profit_price", sa.Float(), nullable=True),
        sa.Column("notional_usd", sa.Float(), nullable=False),
        sa.Column("leverage", sa.Float(), nullable=True),
        sa.Column("risk_usd", sa.Float(), nullable=True),
        sa.Column("constraints_snapshot", postgresql.JSONB(), nullable=True),
    )

    op.create_table(
        "trades",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("trade_id", sa.String(length=64), nullable=False, unique=True),
        sa.Column("decision_id", sa.String(length=64), sa.ForeignKey("decisions.decision_id"), nullable=False),
        sa.Column("open_ts", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("close_ts", sa.DateTime(timezone=True), nullable=True),
        sa.Column("side", decision_side_enum, nullable=False),
        sa.Column("qty", sa.Float(), nullable=True),
        sa.Column("notional", sa.Float(), nullable=True),
        sa.Column("entry_px", sa.Float(), nullable=False),
        sa.Column("exit_px", sa.Float(), nullable=True),
        sa.Column("fees_est", sa.Float(), nullable=True),
        sa.Column("slippage_est", sa.Float(), nullable=True),
        sa.Column("pnl", sa.Float(), nullable=True),
        sa.Column("pnl_pct", sa.Float(), nullable=True),
        sa.Column("max_adverse", sa.Float(), nullable=True),
        sa.Column("max_favorable", sa.Float(), nullable=True),
        sa.Column("close_reason", sa.String(length=16), nullable=False, server_default="open"),
        sa.UniqueConstraint("trade_id", name="uq_trades_trade_id"),
    )


def downgrade():
    op.drop_table("trades")
    op.drop_table("trade_plans")
    op.drop_table("news_scores")
    op.drop_table("bot_state")
    op.drop_table("pattern_signals")
    op.drop_table("decisions")
    op.drop_table("model_predictions")
    op.drop_table("news_events")
    op.drop_table("market_features")
    op.drop_table("market_bars")
    sa.Enum(name="timeframe").drop(op.get_bind(), checkfirst=False)
    sa.Enum(name="regime").drop(op.get_bind(), checkfirst=False)
    sa.Enum(name="marketsetup").drop(op.get_bind(), checkfirst=False)
    sa.Enum(name="eventtype").drop(op.get_bind(), checkfirst=False)
    sa.Enum(name="decisionside").drop(op.get_bind(), checkfirst=False)

