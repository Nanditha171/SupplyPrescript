"""SQLAlchemy 2.x ORM Models for Supply Prescript.

Implements all 10 core domain entities for closed-loop prescriptive analytics:
Predict -> Prescribe -> Decide -> Execute -> Measure -> Learn
"""

from datetime import datetime, date
from decimal import Decimal
from typing import List, Optional

from sqlalchemy import (
    Integer,
    String,
    Text,
    Numeric,
    Float,
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    CheckConstraint,
    UniqueConstraint,
    Index,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base


class Product(Base):
    """Represents a product/SKU managed in the supply chain."""

    __tablename__ = "products"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_code: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    product_name: Mapped[str] = mapped_column(String(255), nullable=False)
    product_category: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    unit_cost: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    criticality_level: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (
        CheckConstraint("unit_cost > 0", name="chk_products_unit_cost_positive"),
        CheckConstraint(
            "criticality_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')",
            name="chk_products_criticality_level",
        ),
    )

    # Relationships
    inventory: Mapped[List["Inventory"]] = relationship(
        "Inventory", back_populates="product", cascade="all, delete-orphan"
    )
    supply_events: Mapped[List["SupplyEvent"]] = relationship(
        "SupplyEvent", back_populates="product", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Product(id={self.id}, code='{self.product_code}', name='{self.product_name}')>"


class Supplier(Base):
    """Represents a primary or secondary supplier providing materials or goods."""

    __tablename__ = "suppliers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    supplier_code: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    supplier_name: Mapped[str] = mapped_column(String(255), nullable=False)
    supplier_country: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    reliability_score: Mapped[float] = mapped_column(Float, nullable=False)
    supplier_capacity_units: Mapped[int] = mapped_column(Integer, nullable=False)
    average_lead_time_days: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (
        CheckConstraint(
            "reliability_score >= 0.0 AND reliability_score <= 1.0",
            name="chk_suppliers_reliability_range",
        ),
        CheckConstraint("supplier_capacity_units > 0", name="chk_suppliers_capacity_positive"),
        CheckConstraint("average_lead_time_days > 0", name="chk_suppliers_lead_time_positive"),
        CheckConstraint(
            "risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')",
            name="chk_suppliers_risk_level",
        ),
    )

    # Relationships
    supply_events: Mapped[List["SupplyEvent"]] = relationship(
        "SupplyEvent", back_populates="supplier", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Supplier(id={self.id}, code='{self.supplier_code}', name='{self.supplier_name}')>"


class Inventory(Base):
    """Represents warehouse stock snapshots for a specific product and date."""

    __tablename__ = "inventory"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("products.id", ondelete="CASCADE"), nullable=False, index=True
    )
    warehouse_code: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    inventory_units: Mapped[int] = mapped_column(Integer, nullable=False)
    reserved_units: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    available_units: Mapped[int] = mapped_column(Integer, nullable=False)
    reorder_point: Mapped[int] = mapped_column(Integer, nullable=False)
    safety_stock_units: Mapped[int] = mapped_column(Integer, nullable=False)
    inventory_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("product_id", "warehouse_code", "inventory_date", name="uq_inventory_snapshot"),
        CheckConstraint("inventory_units >= 0", name="chk_inv_units_non_negative"),
        CheckConstraint("reserved_units >= 0", name="chk_inv_reserved_non_negative"),
        CheckConstraint("reserved_units <= inventory_units", name="chk_inv_reserved_lte_total"),
        CheckConstraint("available_units >= 0", name="chk_inv_available_non_negative"),
        CheckConstraint("reorder_point >= 0", name="chk_inv_reorder_point_non_negative"),
        CheckConstraint("safety_stock_units >= 0", name="chk_inv_safety_stock_non_negative"),
        CheckConstraint(
            "available_units = inventory_units - reserved_units",
            name="chk_inv_available_equals_total_minus_reserved",
        ),
    )

    # Relationships
    product: Mapped["Product"] = relationship("Product", back_populates="inventory")

    def __repr__(self) -> str:
        return f"<Inventory(id={self.id}, product_id={self.product_id}, warehouse='{self.warehouse_code}', available={self.available_units})>"


class SupplyEvent(Base):
    """Represents operational supply-chain events and shipment records."""

    __tablename__ = "supply_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    product_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("products.id", ondelete="CASCADE"), nullable=False, index=True
    )
    supplier_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("suppliers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    event_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    demand_units: Mapped[int] = mapped_column(Integer, nullable=False)
    supplier_capacity_units: Mapped[int] = mapped_column(Integer, nullable=False)
    planned_lead_time_days: Mapped[float] = mapped_column(Float, nullable=False)
    actual_lead_time_days: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    delay_occurred: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    delay_days: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    transport_cost: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False, default=Decimal("0.00"))
    unit_cost: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    secondary_supplier_premium: Mapped[Decimal] = mapped_column(
        Numeric(12, 2), nullable=False, default=Decimal("0.00")
    )
    air_freight_cost: Mapped[Decimal] = mapped_column(
        Numeric(12, 2), nullable=False, default=Decimal("0.00")
    )
    service_risk_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        CheckConstraint("demand_units >= 0", name="chk_supply_events_demand_non_negative"),
        CheckConstraint("supplier_capacity_units >= 0", name="chk_supply_events_capacity_non_negative"),
        CheckConstraint("planned_lead_time_days >= 0", name="chk_supply_events_planned_lead_time_non_negative"),
        CheckConstraint("actual_lead_time_days IS NULL OR actual_lead_time_days >= 0", name="chk_supply_events_actual_lead_time_non_negative"),
        CheckConstraint("delay_days >= 0", name="chk_supply_events_delay_days_non_negative"),
        CheckConstraint("transport_cost >= 0", name="chk_supply_events_transport_cost_non_negative"),
        CheckConstraint("unit_cost >= 0", name="chk_supply_events_unit_cost_non_negative"),
        CheckConstraint("secondary_supplier_premium >= 0", name="chk_supply_events_secondary_premium_non_negative"),
        CheckConstraint("air_freight_cost >= 0", name="chk_supply_events_air_freight_cost_non_negative"),
        CheckConstraint("service_risk_score >= 0.0 AND service_risk_score <= 1.0", name="chk_supply_events_service_risk_score_range"),
    )

    # Relationships
    product: Mapped["Product"] = relationship("Product", back_populates="supply_events")
    supplier: Mapped["Supplier"] = relationship("Supplier", back_populates="supply_events")
    predictions: Mapped[List["Prediction"]] = relationship(
        "Prediction", back_populates="event", cascade="all, delete-orphan"
    )
    optimization_runs: Mapped[List["OptimizationRun"]] = relationship(
        "OptimizationRun", back_populates="event", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<SupplyEvent(id={self.id}, event_id='{self.event_id}', date={self.event_date})>"


class Prediction(Base):
    """Represents machine-learning inference outputs for supply events."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("supply_events.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model_name: Mapped[str] = mapped_column(String(150), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    prediction_type: Mapped[str] = mapped_column(String(50), nullable=False)
    delay_probability: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_delay_days: Mapped[float] = mapped_column(Float, nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(20), nullable=False)
    prediction_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    prediction_expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        CheckConstraint("delay_probability >= 0.0 AND delay_probability <= 1.0", name="chk_pred_delay_prob_range"),
        CheckConstraint("predicted_delay_days >= 0", name="chk_pred_delay_days_non_negative"),
        CheckConstraint("risk_score >= 0.0 AND risk_score <= 1.0", name="chk_pred_risk_score_range"),
        CheckConstraint("prediction_type IN ('DELAY', 'DEMAND', 'SUPPLY_RISK')", name="chk_pred_type_enum"),
        CheckConstraint("risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')", name="chk_pred_risk_level_enum"),
    )

    # Relationships
    event: Mapped["SupplyEvent"] = relationship("SupplyEvent", back_populates="predictions")
    optimization_runs: Mapped[List["OptimizationRun"]] = relationship(
        "OptimizationRun", back_populates="prediction", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Prediction(id={self.id}, model='{self.model_name}', prob={self.delay_probability:.2f}, risk='{self.risk_level}')>"


class OptimizationRun(Base):
    """Represents execution records of the prescriptive optimization solver."""

    __tablename__ = "optimization_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("supply_events.id", ondelete="CASCADE"), nullable=False, index=True
    )
    prediction_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    optimization_run_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    objective_type: Mapped[str] = mapped_column(String(50), nullable=False)
    optimization_status: Mapped[str] = mapped_column(String(50), nullable=False)
    total_cost_baseline: Mapped[Decimal] = mapped_column(
        Numeric(14, 2), nullable=False, default=Decimal("0.00")
    )
    recommended_cost: Mapped[Optional[Decimal]] = mapped_column(Numeric(14, 2), nullable=True)
    service_level_target: Mapped[float] = mapped_column(Float, nullable=False, default=0.95)
    risk_weight: Mapped[float] = mapped_column(Float, nullable=False, default=0.33)
    cost_weight: Mapped[float] = mapped_column(Float, nullable=False, default=0.34)
    time_weight: Mapped[float] = mapped_column(Float, nullable=False, default=0.33)
    solver_name: Mapped[str] = mapped_column(String(100), nullable=False, default="PULP_CBC")
    solver_status: Mapped[str] = mapped_column(String(50), nullable=False)
    execution_time_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        CheckConstraint("total_cost_baseline >= 0", name="chk_opt_total_cost_baseline_non_negative"),
        CheckConstraint("recommended_cost IS NULL OR recommended_cost >= 0", name="chk_opt_rec_cost_non_negative"),
        CheckConstraint("service_level_target >= 0.0 AND service_level_target <= 1.0", name="chk_opt_service_level_target_range"),
        CheckConstraint("risk_weight >= 0", name="chk_opt_risk_weight_non_negative"),
        CheckConstraint("cost_weight >= 0", name="chk_opt_cost_weight_non_negative"),
        CheckConstraint("time_weight >= 0", name="chk_opt_time_weight_non_negative"),
        CheckConstraint("execution_time_ms >= 0", name="chk_opt_exec_time_non_negative"),
        CheckConstraint("objective_type IN ('MIN_COST', 'MIN_RISK', 'MIN_DELAY', 'BALANCED')", name="chk_opt_objective_type_enum"),
        CheckConstraint("optimization_status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED')", name="chk_opt_status_enum"),
        CheckConstraint("solver_status IN ('OPTIMAL', 'FEASIBLE', 'INFEASIBLE', 'UNBOUNDED', 'ERROR')", name="chk_opt_solver_status_enum"),
    )

    # Relationships
    event: Mapped["SupplyEvent"] = relationship("SupplyEvent", back_populates="optimization_runs")
    prediction: Mapped["Prediction"] = relationship("Prediction", back_populates="optimization_runs")
    recommendations: Mapped[List["Recommendation"]] = relationship(
        "Recommendation", back_populates="optimization_run", cascade="all, delete-orphan"
    )
    decisions: Mapped[List["Decision"]] = relationship(
        "Decision", back_populates="optimization_run", cascade="all, delete-orphan"
    )
    feedback_records: Mapped[List["Feedback"]] = relationship(
        "Feedback", back_populates="optimization_run", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<OptimizationRun(id={self.id}, run_id='{self.optimization_run_id}', status='{self.optimization_status}')>"


class Recommendation(Base):
    """Represents alternative action strategies generated by the optimizer."""

    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    optimization_run_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("optimization_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    recommendation_code: Mapped[str] = mapped_column(String(50), nullable=False)
    strategy_type: Mapped[str] = mapped_column(String(50), nullable=False)
    strategy_description: Mapped[str] = mapped_column(Text, nullable=False)
    estimated_cost: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    estimated_delay_days: Mapped[float] = mapped_column(Float, nullable=False)
    estimated_risk: Mapped[float] = mapped_column(Float, nullable=False)
    service_level: Mapped[float] = mapped_column(Float, nullable=False)
    cost_difference: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False, default=Decimal("0.00"))
    risk_difference: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    is_recommended: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("optimization_run_id", "recommendation_code", name="uq_run_recommendation_code"),
        CheckConstraint("estimated_cost >= 0", name="chk_rec_estimated_cost_non_negative"),
        CheckConstraint("estimated_delay_days >= 0", name="chk_rec_estimated_delay_non_negative"),
        CheckConstraint("estimated_risk >= 0.0 AND estimated_risk <= 1.0", name="chk_rec_estimated_risk_range"),
        CheckConstraint("service_level >= 0.0 AND service_level <= 1.0", name="chk_rec_service_level_range"),
        CheckConstraint("rank >= 1", name="chk_rec_rank_positive"),
        CheckConstraint(
            "strategy_type IN ('AIR_FREIGHT', 'SECONDARY_SUPPLIER', 'PRIMARY_SUPPLIER', 'DELAY_LAUNCH', 'EXPEDITE', 'PARTIAL_ALLOCATION')",
            name="chk_rec_strategy_type_enum",
        ),
    )

    # Relationships
    optimization_run: Mapped["OptimizationRun"] = relationship("OptimizationRun", back_populates="recommendations")
    decisions: Mapped[List["Decision"]] = relationship("Decision", back_populates="recommendation")

    def __repr__(self) -> str:
        return f"<Recommendation(id={self.id}, code='{self.recommendation_code}', strategy='{self.strategy_type}', cost={self.estimated_cost})>"


class Decision(Base):
    """Represents a human manager's selection/override of a recommendation."""

    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    decision_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    optimization_run_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("optimization_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    recommendation_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("recommendations.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    decision_maker: Mapped[str] = mapped_column(String(150), nullable=False)
    decision_status: Mapped[str] = mapped_column(String(50), nullable=False)
    decision_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    selected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    execution_status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="NOT_STARTED"
    )
    executed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (
        CheckConstraint(
            "decision_status IN ('PENDING', 'APPROVED', 'REJECTED', 'OVERRIDDEN', 'CANCELLED')",
            name="chk_decision_status_enum",
        ),
        CheckConstraint(
            "execution_status IN ('NOT_STARTED', 'IN_PROGRESS', 'COMPLETED', 'FAILED')",
            name="chk_decision_execution_status_enum",
        ),
    )

    # Relationships
    optimization_run: Mapped["OptimizationRun"] = relationship("OptimizationRun", back_populates="decisions")
    recommendation: Mapped["Recommendation"] = relationship("Recommendation", back_populates="decisions")
    outcome: Mapped[Optional["Outcome"]] = relationship(
        "Outcome", back_populates="decision", uselist=False, cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Decision(id={self.id}, decision_id='{self.decision_id}', maker='{self.decision_maker}', status='{self.decision_status}')>"


class Outcome(Base):
    """Represents actual post-execution operational results and variances."""

    __tablename__ = "outcomes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    decision_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("decisions.id", ondelete="CASCADE"), unique=True, nullable=False, index=True
    )
    actual_cost: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    actual_delay_days: Mapped[float] = mapped_column(Float, nullable=False)
    actual_service_level: Mapped[float] = mapped_column(Float, nullable=False)
    actual_risk: Mapped[float] = mapped_column(Float, nullable=False)
    actual_quantity_delivered: Mapped[int] = mapped_column(Integer, nullable=False)
    cost_variance: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    delay_variance: Mapped[float] = mapped_column(Float, nullable=False)
    service_level_variance: Mapped[float] = mapped_column(Float, nullable=False)
    outcome_status: Mapped[str] = mapped_column(String(50), nullable=False)
    evaluation_date: Mapped[date] = mapped_column(Date, nullable=False)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        CheckConstraint("actual_cost >= 0", name="chk_outcomes_actual_cost_non_negative"),
        CheckConstraint("actual_delay_days >= 0", name="chk_outcomes_actual_delay_non_negative"),
        CheckConstraint(
            "actual_service_level >= 0.0 AND actual_service_level <= 1.0",
            name="chk_outcomes_actual_service_level_range",
        ),
        CheckConstraint("actual_risk >= 0.0 AND actual_risk <= 1.0", name="chk_outcomes_actual_risk_range"),
        CheckConstraint(
            "actual_quantity_delivered >= 0",
            name="chk_outcomes_actual_qty_non_negative",
        ),
        CheckConstraint(
            "outcome_status IN ('SUCCESS', 'PARTIAL_SUCCESS', 'FAILED', 'CANCELLED')",
            name="chk_outcomes_status_enum",
        ),
    )

    # Relationships
    decision: Mapped["Decision"] = relationship("Decision", back_populates="outcome")
    feedback_records: Mapped[List["Feedback"]] = relationship(
        "Feedback", back_populates="outcome", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Outcome(id={self.id}, decision_id={self.decision_id}, cost_var={self.cost_variance}, status='{self.outcome_status}')>"


class Feedback(Base):
    """Closes the prescriptive learning loop by recording variance error signals for future runs."""

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    outcome_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("outcomes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    optimization_run_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("optimization_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    feedback_type: Mapped[str] = mapped_column(String(50), nullable=False)
    cost_error: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False, default=Decimal("0.00"))
    delay_error: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    risk_error: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    prediction_error: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    weight_adjustment: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    learning_signal: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        CheckConstraint(
            "feedback_type IN ('COST_VARIANCE', 'DELAY_VARIANCE', 'PREDICTION_ERROR', 'OPTIMIZATION_ERROR', 'SERVICE_LEVEL_ERROR')",
            name="chk_feedback_type_enum",
        ),
    )

    # Relationships
    outcome: Mapped["Outcome"] = relationship("Outcome", back_populates="feedback_records")
    optimization_run: Mapped["OptimizationRun"] = relationship("OptimizationRun", back_populates="feedback_records")

    def __repr__(self) -> str:
        return f"<Feedback(id={self.id}, type='{self.feedback_type}', cost_error={self.cost_error})>"
