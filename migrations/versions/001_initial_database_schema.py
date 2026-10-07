"""001_initial_database_schema

Revision ID: 001_initial_database_schema
Revises: None
Create Date: 2026-10-07

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "001_initial_database_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. products
    op.create_table(
        "products",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("product_code", sa.String(length=50), nullable=False),
        sa.Column("product_name", sa.String(length=255), nullable=False),
        sa.Column("product_category", sa.String(length=100), nullable=False),
        sa.Column("unit_cost", sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column("criticality_level", sa.String(length=20), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint("unit_cost > 0", name="chk_products_unit_cost_positive"),
        sa.CheckConstraint(
            "criticality_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')",
            name="chk_products_criticality_level",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("product_code"),
    )
    op.create_index("ix_products_product_code", "products", ["product_code"], unique=True)
    op.create_index("ix_products_product_category", "products", ["product_category"], unique=False)
    op.create_index("ix_products_criticality_level", "products", ["criticality_level"], unique=False)

    # 2. suppliers
    op.create_table(
        "suppliers",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("supplier_code", sa.String(length=50), nullable=False),
        sa.Column("supplier_name", sa.String(length=255), nullable=False),
        sa.Column("supplier_country", sa.String(length=100), nullable=False),
        sa.Column("reliability_score", sa.Float(), nullable=False),
        sa.Column("supplier_capacity_units", sa.Integer(), nullable=False),
        sa.Column("average_lead_time_days", sa.Float(), nullable=False),
        sa.Column("risk_level", sa.String(length=20), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "reliability_score >= 0.0 AND reliability_score <= 1.0",
            name="chk_suppliers_reliability_range",
        ),
        sa.CheckConstraint("supplier_capacity_units > 0", name="chk_suppliers_capacity_positive"),
        sa.CheckConstraint("average_lead_time_days > 0", name="chk_suppliers_lead_time_positive"),
        sa.CheckConstraint(
            "risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')",
            name="chk_suppliers_risk_level",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("supplier_code"),
    )
    op.create_index("ix_suppliers_supplier_code", "suppliers", ["supplier_code"], unique=True)
    op.create_index("ix_suppliers_supplier_country", "suppliers", ["supplier_country"], unique=False)
    op.create_index("ix_suppliers_risk_level", "suppliers", ["risk_level"], unique=False)

    # 3. inventory
    op.create_table(
        "inventory",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("product_id", sa.Integer(), nullable=False),
        sa.Column("warehouse_code", sa.String(length=50), nullable=False),
        sa.Column("inventory_units", sa.Integer(), nullable=False),
        sa.Column("reserved_units", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("available_units", sa.Integer(), nullable=False),
        sa.Column("reorder_point", sa.Integer(), nullable=False),
        sa.Column("safety_stock_units", sa.Integer(), nullable=False),
        sa.Column("inventory_date", sa.Date(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["product_id"], ["products.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("product_id", "warehouse_code", "inventory_date", name="uq_inventory_snapshot"),
        sa.CheckConstraint("inventory_units >= 0", name="chk_inv_units_non_negative"),
        sa.CheckConstraint("reserved_units >= 0", name="chk_inv_reserved_non_negative"),
        sa.CheckConstraint("reserved_units <= inventory_units", name="chk_inv_reserved_lte_total"),
        sa.CheckConstraint("available_units >= 0", name="chk_inv_available_non_negative"),
        sa.CheckConstraint("reorder_point >= 0", name="chk_inv_reorder_point_non_negative"),
        sa.CheckConstraint("safety_stock_units >= 0", name="chk_inv_safety_stock_non_negative"),
        sa.CheckConstraint("available_units = inventory_units - reserved_units", name="chk_inv_available_calc"),
    )
    op.create_index("ix_inventory_product_id", "inventory", ["product_id"], unique=False)
    op.create_index("ix_inventory_warehouse_code", "inventory", ["warehouse_code"], unique=False)
    op.create_index("ix_inventory_inventory_date", "inventory", ["inventory_date"], unique=False)

    # 4. supply_events
    op.create_table(
        "supply_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("event_id", sa.String(length=100), nullable=False),
        sa.Column("product_id", sa.Integer(), nullable=False),
        sa.Column("supplier_id", sa.Integer(), nullable=False),
        sa.Column("event_date", sa.Date(), nullable=False),
        sa.Column("demand_units", sa.Integer(), nullable=False),
        sa.Column("supplier_capacity_units", sa.Integer(), nullable=False),
        sa.Column("planned_lead_time_days", sa.Float(), nullable=False),
        sa.Column("actual_lead_time_days", sa.Float(), nullable=True),
        sa.Column("delay_occurred", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("delay_days", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("transport_cost", sa.Numeric(precision=12, scale=2), nullable=False, server_default="0.00"),
        sa.Column("unit_cost", sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column("secondary_supplier_premium", sa.Numeric(precision=12, scale=2), nullable=False, server_default="0.00"),
        sa.Column("air_freight_cost", sa.Numeric(precision=12, scale=2), nullable=False, server_default="0.00"),
        sa.Column("service_risk_score", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["product_id"], ["products.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["supplier_id"], ["suppliers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("event_id"),
        sa.CheckConstraint("demand_units >= 0", name="chk_supply_events_demand_non_negative"),
        sa.CheckConstraint("supplier_capacity_units >= 0", name="chk_supply_events_capacity_non_negative"),
        sa.CheckConstraint("planned_lead_time_days >= 0", name="chk_supply_events_planned_lead_time_non_negative"),
        sa.CheckConstraint("actual_lead_time_days IS NULL OR actual_lead_time_days >= 0", name="chk_supply_events_actual_lead_time_non_negative"),
        sa.CheckConstraint("delay_days >= 0", name="chk_supply_events_delay_days_non_negative"),
        sa.CheckConstraint("transport_cost >= 0", name="chk_supply_events_transport_cost_non_negative"),
        sa.CheckConstraint("unit_cost >= 0", name="chk_supply_events_unit_cost_non_negative"),
        sa.CheckConstraint("secondary_supplier_premium >= 0", name="chk_supply_events_secondary_premium_non_negative"),
        sa.CheckConstraint("air_freight_cost >= 0", name="chk_supply_events_air_freight_cost_non_negative"),
        sa.CheckConstraint("service_risk_score >= 0.0 AND service_risk_score <= 1.0", name="chk_supply_events_service_risk_score_range"),
    )
    op.create_index("ix_supply_events_event_id", "supply_events", ["event_id"], unique=True)
    op.create_index("ix_supply_events_product_id", "supply_events", ["product_id"], unique=False)
    op.create_index("ix_supply_events_supplier_id", "supply_events", ["supplier_id"], unique=False)
    op.create_index("ix_supply_events_event_date", "supply_events", ["event_date"], unique=False)
    op.create_index("ix_supply_events_delay_occurred", "supply_events", ["delay_occurred"], unique=False)

    # 5. predictions
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("event_id", sa.Integer(), nullable=False),
        sa.Column("model_name", sa.String(length=150), nullable=False),
        sa.Column("model_version", sa.String(length=50), nullable=False),
        sa.Column("prediction_type", sa.String(length=50), nullable=False),
        sa.Column("delay_probability", sa.Float(), nullable=False),
        sa.Column("predicted_delay_days", sa.Float(), nullable=False),
        sa.Column("risk_score", sa.Float(), nullable=False),
        sa.Column("risk_level", sa.String(length=20), nullable=False),
        sa.Column(
            "prediction_timestamp",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("prediction_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["event_id"], ["supply_events.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint("delay_probability >= 0.0 AND delay_probability <= 1.0", name="chk_pred_delay_prob_range"),
        sa.CheckConstraint("predicted_delay_days >= 0", name="chk_pred_delay_days_non_negative"),
        sa.CheckConstraint("risk_score >= 0.0 AND risk_score <= 1.0", name="chk_pred_risk_score_range"),
        sa.CheckConstraint("prediction_type IN ('DELAY', 'DEMAND', 'SUPPLY_RISK')", name="chk_pred_type_enum"),
        sa.CheckConstraint("risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')", name="chk_pred_risk_level_enum"),
    )
    op.create_index("ix_predictions_event_id", "predictions", ["event_id"], unique=False)

    # 6. optimization_runs
    op.create_table(
        "optimization_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("event_id", sa.Integer(), nullable=False),
        sa.Column("prediction_id", sa.Integer(), nullable=False),
        sa.Column("optimization_run_id", sa.String(length=100), nullable=False),
        sa.Column("objective_type", sa.String(length=50), nullable=False),
        sa.Column("optimization_status", sa.String(length=50), nullable=False),
        sa.Column("total_cost_baseline", sa.Numeric(precision=14, scale=2), nullable=False, server_default="0.00"),
        sa.Column("recommended_cost", sa.Numeric(precision=14, scale=2), nullable=True),
        sa.Column("service_level_target", sa.Float(), nullable=False, server_default="0.95"),
        sa.Column("risk_weight", sa.Float(), nullable=False, server_default="0.33"),
        sa.Column("cost_weight", sa.Float(), nullable=False, server_default="0.34"),
        sa.Column("time_weight", sa.Float(), nullable=False, server_default="0.33"),
        sa.Column("solver_name", sa.String(length=100), nullable=False, server_default="PULP_CBC"),
        sa.Column("solver_status", sa.String(length=50), nullable=False),
        sa.Column("execution_time_ms", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["event_id"], ["supply_events.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["prediction_id"], ["predictions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("optimization_run_id"),
        sa.CheckConstraint("total_cost_baseline >= 0", name="chk_opt_total_cost_baseline_non_negative"),
        sa.CheckConstraint("recommended_cost IS NULL OR recommended_cost >= 0", name="chk_opt_rec_cost_non_negative"),
        sa.CheckConstraint("service_level_target >= 0.0 AND service_level_target <= 1.0", name="chk_opt_service_level_target_range"),
        sa.CheckConstraint("risk_weight >= 0", name="chk_opt_risk_weight_non_negative"),
        sa.CheckConstraint("cost_weight >= 0", name="chk_opt_cost_weight_non_negative"),
        sa.CheckConstraint("time_weight >= 0", name="chk_opt_time_weight_non_negative"),
        sa.CheckConstraint("execution_time_ms >= 0", name="chk_opt_exec_time_non_negative"),
        sa.CheckConstraint("objective_type IN ('MIN_COST', 'MIN_RISK', 'MIN_DELAY', 'BALANCED')", name="chk_opt_objective_type_enum"),
        sa.CheckConstraint("optimization_status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED')", name="chk_opt_status_enum"),
        sa.CheckConstraint("solver_status IN ('OPTIMAL', 'FEASIBLE', 'INFEASIBLE', 'UNBOUNDED', 'ERROR')", name="chk_opt_solver_status_enum"),
    )
    op.create_index("ix_optimization_runs_event_id", "optimization_runs", ["event_id"], unique=False)
    op.create_index("ix_optimization_runs_prediction_id", "optimization_runs", ["prediction_id"], unique=False)
    op.create_index("ix_optimization_runs_optimization_run_id", "optimization_runs", ["optimization_run_id"], unique=True)

    # 7. recommendations
    op.create_table(
        "recommendations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("optimization_run_id", sa.Integer(), nullable=False),
        sa.Column("recommendation_code", sa.String(length=50), nullable=False),
        sa.Column("strategy_type", sa.String(length=50), nullable=False),
        sa.Column("strategy_description", sa.Text(), nullable=False),
        sa.Column("estimated_cost", sa.Numeric(precision=14, scale=2), nullable=False),
        sa.Column("estimated_delay_days", sa.Float(), nullable=False),
        sa.Column("estimated_risk", sa.Float(), nullable=False),
        sa.Column("service_level", sa.Float(), nullable=False),
        sa.Column("cost_difference", sa.Numeric(precision=14, scale=2), nullable=False, server_default="0.00"),
        sa.Column("risk_difference", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("is_recommended", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["optimization_run_id"], ["optimization_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("optimization_run_id", "recommendation_code", name="uq_run_recommendation_code"),
        sa.CheckConstraint("estimated_cost >= 0", name="chk_rec_estimated_cost_non_negative"),
        sa.CheckConstraint("estimated_delay_days >= 0", name="chk_rec_estimated_delay_non_negative"),
        sa.CheckConstraint("estimated_risk >= 0.0 AND estimated_risk <= 1.0", name="chk_rec_estimated_risk_range"),
        sa.CheckConstraint("service_level >= 0.0 AND service_level <= 1.0", name="chk_rec_service_level_range"),
        sa.CheckConstraint("rank >= 1", name="chk_rec_rank_positive"),
        sa.CheckConstraint(
            "strategy_type IN ('AIR_FREIGHT', 'SECONDARY_SUPPLIER', 'PRIMARY_SUPPLIER', 'DELAY_LAUNCH', 'EXPEDITE', 'PARTIAL_ALLOCATION')",
            name="chk_rec_strategy_type_enum",
        ),
    )
    op.create_index("ix_recommendations_optimization_run_id", "recommendations", ["optimization_run_id"], unique=False)

    # 8. decisions
    op.create_table(
        "decisions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("decision_id", sa.String(length=100), nullable=False),
        sa.Column("optimization_run_id", sa.Integer(), nullable=False),
        sa.Column("recommendation_id", sa.Integer(), nullable=False),
        sa.Column("decision_maker", sa.String(length=150), nullable=False),
        sa.Column("decision_status", sa.String(length=50), nullable=False),
        sa.Column("decision_reason", sa.Text(), nullable=True),
        sa.Column(
            "selected_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("execution_status", sa.String(length=50), nullable=False, server_default="NOT_STARTED"),
        sa.Column("executed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["optimization_run_id"], ["optimization_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["recommendation_id"], ["recommendations.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("decision_id"),
        sa.CheckConstraint(
            "decision_status IN ('PENDING', 'APPROVED', 'REJECTED', 'OVERRIDDEN', 'CANCELLED')",
            name="chk_decision_status_enum",
        ),
        sa.CheckConstraint(
            "execution_status IN ('NOT_STARTED', 'IN_PROGRESS', 'COMPLETED', 'FAILED')",
            name="chk_decision_execution_status_enum",
        ),
    )
    op.create_index("ix_decisions_decision_id", "decisions", ["decision_id"], unique=True)
    op.create_index("ix_decisions_optimization_run_id", "decisions", ["optimization_run_id"], unique=False)
    op.create_index("ix_decisions_recommendation_id", "decisions", ["recommendation_id"], unique=False)

    # 9. outcomes
    op.create_table(
        "outcomes",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("decision_id", sa.Integer(), nullable=False),
        sa.Column("actual_cost", sa.Numeric(precision=14, scale=2), nullable=False),
        sa.Column("actual_delay_days", sa.Float(), nullable=False),
        sa.Column("actual_service_level", sa.Float(), nullable=False),
        sa.Column("actual_risk", sa.Float(), nullable=False),
        sa.Column("actual_quantity_delivered", sa.Integer(), nullable=False),
        sa.Column("cost_variance", sa.Numeric(precision=14, scale=2), nullable=False),
        sa.Column("delay_variance", sa.Float(), nullable=False),
        sa.Column("service_level_variance", sa.Float(), nullable=False),
        sa.Column("outcome_status", sa.String(length=50), nullable=False),
        sa.Column("evaluation_date", sa.Date(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["decision_id"], ["decisions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("decision_id"),
        sa.CheckConstraint("actual_cost >= 0", name="chk_outcomes_actual_cost_non_negative"),
        sa.CheckConstraint("actual_delay_days >= 0", name="chk_outcomes_actual_delay_non_negative"),
        sa.CheckConstraint("actual_service_level >= 0.0 AND actual_service_level <= 1.0", name="chk_outcomes_actual_service_level_range"),
        sa.CheckConstraint("actual_risk >= 0.0 AND actual_risk <= 1.0", name="chk_outcomes_actual_risk_range"),
        sa.CheckConstraint("actual_quantity_delivered >= 0", name="chk_outcomes_actual_qty_non_negative"),
        sa.CheckConstraint("outcome_status IN ('SUCCESS', 'PARTIAL_SUCCESS', 'FAILED', 'CANCELLED')", name="chk_outcomes_status_enum"),
    )
    op.create_index("ix_outcomes_decision_id", "outcomes", ["decision_id"], unique=True)

    # 10. feedback
    op.create_table(
        "feedback",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("outcome_id", sa.Integer(), nullable=False),
        sa.Column("optimization_run_id", sa.Integer(), nullable=False),
        sa.Column("feedback_type", sa.String(length=50), nullable=False),
        sa.Column("cost_error", sa.Numeric(precision=14, scale=2), nullable=False, server_default="0.00"),
        sa.Column("delay_error", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("risk_error", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("prediction_error", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("weight_adjustment", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("learning_signal", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["outcome_id"], ["outcomes.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["optimization_run_id"], ["optimization_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "feedback_type IN ('COST_VARIANCE', 'DELAY_VARIANCE', 'PREDICTION_ERROR', 'OPTIMIZATION_ERROR', 'SERVICE_LEVEL_ERROR')",
            name="chk_feedback_type_enum",
        ),
    )
    op.create_index("ix_feedback_outcome_id", "feedback", ["outcome_id"], unique=False)
    op.create_index("ix_feedback_optimization_run_id", "feedback", ["optimization_run_id"], unique=False)


def downgrade() -> None:
    op.drop_table("feedback")
    op.drop_table("outcomes")
    op.drop_table("decisions")
    op.drop_table("recommendations")
    op.drop_table("optimization_runs")
    op.drop_table("predictions")
    op.drop_table("supply_events")
    op.drop_table("inventory")
    op.drop_table("suppliers")
    op.drop_table("products")
