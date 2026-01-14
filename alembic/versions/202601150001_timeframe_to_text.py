"""convert timeframe enum columns to text

Revision ID: 202601150001
Revises: 202601130001
Create Date: 2026-01-15 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "202601150001"
down_revision = "202601130001"
branch_labels = None
depends_on = None

timeframe_enum = sa.Enum("1m", "5m", "15m", "1h", name="timeframe")


def upgrade():
    # Convert timeframe columns from enum to text/varchar for flexibility
    for table in ("market_bars", "market_features", "pattern_signals"):
        op.alter_column(
            table,
            "timeframe",
            existing_type=timeframe_enum,
            type_=sa.String(length=16),
            existing_nullable=False,
            postgresql_using="timeframe::text",
        )

    # Drop the obsolete enum type to avoid schema drift on fresh installs
    op.execute("DROP TYPE IF EXISTS timeframe;")


def downgrade():
    # Recreate enum type and cast columns back
    timeframe_enum.create(op.get_bind(), checkfirst=True)
    for table in ("market_bars", "market_features", "pattern_signals"):
        op.alter_column(
            table,
            "timeframe",
            existing_type=sa.String(length=16),
            type_=timeframe_enum,
            existing_nullable=False,
            postgresql_using="timeframe::timeframe",
        )
