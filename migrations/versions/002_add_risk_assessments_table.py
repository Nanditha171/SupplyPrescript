"""002_add_risk_assessments_table

Revision ID: 002_add_risk_assessments_table
Revises: 001_initial_database_schema
Create Date: 2026-10-13

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "002_add_risk_assessments_table"
down_revision: Union[str, None] = "001_initial_database_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "risk_assessments",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("event_id", sa.Integer(), nullable=False),
        sa.Column("prediction_id", sa.Integer(), nullable=True),
        sa.Column("delay_risk_score", sa.Float(), nullable=False),
        sa.Column("supplier_risk_score", sa.Float(), nullable=False),
        sa.Column("inventory_risk_score", sa.Float(), nullable=False),
        sa.Column("capacity_risk_score", sa.Float(), nullable=False),
        sa.Column("service_risk_score", sa.Float(), nullable=True),
        sa.Column("composite_risk_score", sa.Float(), nullable=False),
        sa.Column("risk_level", sa.String(length=20), nullable=False),
        sa.Column("risk_reasons", sa.Text(), nullable=False),
        sa.Column("configuration_version", sa.String(length=50), nullable=False),
        sa.Column("assessment_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["event_id"], ["supply_events.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["prediction_id"], ["predictions.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "delay_risk_score >= 0.0 AND delay_risk_score <= 1.0",
            name="chk_risk_delay_score_range",
        ),
        sa.CheckConstraint(
            "supplier_risk_score >= 0.0 AND supplier_risk_score <= 1.0",
            name="chk_risk_supplier_score_range",
        ),
        sa.CheckConstraint(
            "inventory_risk_score >= 0.0 AND inventory_risk_score <= 1.0",
            name="chk_risk_inventory_score_range",
        ),
        sa.CheckConstraint(
            "capacity_risk_score >= 0.0 AND capacity_risk_score <= 1.0",
            name="chk_risk_capacity_score_range",
        ),
        sa.CheckConstraint(
            "service_risk_score IS NULL OR (service_risk_score >= 0.0 AND service_risk_score <= 1.0)",
            name="chk_risk_service_score_range",
        ),
        sa.CheckConstraint(
            "composite_risk_score >= 0.0 AND composite_risk_score <= 1.0",
            name="chk_risk_composite_score_range",
        ),
        sa.CheckConstraint(
            "risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')",
            name="chk_risk_level_enum",
        ),
    )
    op.create_index("ix_risk_assessments_event_id", "risk_assessments", ["event_id"], unique=False)
    op.create_index("ix_risk_assessments_prediction_id", "risk_assessments", ["prediction_id"], unique=False)
    op.create_index("ix_risk_assessments_risk_level", "risk_assessments", ["risk_level"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_risk_assessments_risk_level", table_name="risk_assessments")
    op.drop_index("ix_risk_assessments_prediction_id", table_name="risk_assessments")
    op.drop_index("ix_risk_assessments_event_id", table_name="risk_assessments")
    op.drop_table("risk_assessments")
