"""convert regime enum to text

Revision ID: 202601150002
Revises: 202601150001
Create Date: 2026-01-15 00:30:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "202601150002"
down_revision = "202601150001"
branch_labels = None
depends_on = None

regime_enum = sa.Enum("TREND_UP", "TREND_DOWN", "RANGE", "UNKNOWN", name="regime")


def upgrade():
    # drop default that references enum before converting
    op.execute("ALTER TABLE market_features ALTER COLUMN regime DROP DEFAULT;")

    # convert column to text
    op.execute("ALTER TABLE market_features ALTER COLUMN regime TYPE TEXT USING regime::text;")
    # set a new text default after conversion
    op.execute("ALTER TABLE market_features ALTER COLUMN regime SET DEFAULT 'UNKNOWN';")

    # drop enum type with cascade to remove any lingering deps
    op.execute("DROP TYPE IF EXISTS regime CASCADE;")


def downgrade():
    regime_enum.create(op.get_bind(), checkfirst=True)
    op.alter_column(
        "market_features",
        "regime",
        existing_type=sa.String(length=16),
        type_=regime_enum,
        existing_nullable=False,
        postgresql_using="regime::regime",
        server_default="UNKNOWN",
    )
