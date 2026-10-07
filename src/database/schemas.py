"""Pydantic V2 Schemas for Supply Prescript.

Provides data validation and serialization for API requests/responses across all 10 domain entities.
"""

from datetime import datetime, date
from decimal import Decimal
from typing import Optional, List, Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator


# ==========================================
# 1. Product Schemas
# ==========================================

CriticalityLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class ProductBase(BaseModel):
    product_code: str = Field(..., max_length=50, description="Unique product SKU/Code")
    product_name: str = Field(..., max_length=255, description="Product Name")
    product_category: str = Field(..., max_length=100, description="Category classification")
    unit_cost: Decimal = Field(..., gt=0, description="Unit cost, must be positive")
    criticality_level: CriticalityLevel = Field(..., description="Criticality level")


class ProductCreate(ProductBase):
    pass


class ProductUpdate(BaseModel):
    product_name: Optional[str] = Field(None, max_length=255)
    product_category: Optional[str] = Field(None, max_length=100)
    unit_cost: Optional[Decimal] = Field(None, gt=0)
    criticality_level: Optional[CriticalityLevel] = None


class ProductResponse(ProductBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ==========================================
# 2. Supplier Schemas
# ==========================================

RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class SupplierBase(BaseModel):
    supplier_code: str = Field(..., max_length=50, description="Unique supplier code")
    supplier_name: str = Field(..., max_length=255, description="Supplier company name")
    supplier_country: str = Field(..., max_length=100, description="Country of operations")
    reliability_score: float = Field(..., ge=0.0, le=1.0, description="Reliability score between 0.0 and 1.0")
    supplier_capacity_units: int = Field(..., gt=0, description="Maximum capacity in units")
    average_lead_time_days: float = Field(..., gt=0.0, description="Average lead time in days")
    risk_level: RiskLevel = Field(..., description="Assessed risk level")


class SupplierCreate(SupplierBase):
    pass


class SupplierUpdate(BaseModel):
    supplier_name: Optional[str] = Field(None, max_length=255)
    supplier_country: Optional[str] = Field(None, max_length=100)
    reliability_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    supplier_capacity_units: Optional[int] = Field(None, gt=0)
    average_lead_time_days: Optional[float] = Field(None, gt=0.0)
    risk_level: Optional[RiskLevel] = None


class SupplierResponse(SupplierBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ==========================================
# 3. Inventory Schemas
# ==========================================

class InventoryBase(BaseModel):
    product_id: int = Field(..., description="Associated product ID")
    warehouse_code: str = Field(..., max_length=50, description="Warehouse facility identifier")
    inventory_units: int = Field(..., ge=0, description="Total physical inventory units")
    reserved_units: int = Field(default=0, ge=0, description="Units allocated or reserved")
    available_units: int = Field(..., ge=0, description="Available unreserved units")
    reorder_point: int = Field(..., ge=0, description="Threshold triggering reorder")
    safety_stock_units: int = Field(..., ge=0, description="Minimum buffer stock units")
    inventory_date: date = Field(..., description="Date of inventory snapshot")

    @model_validator(mode="after")
    def validate_inventory_math(self):
        if self.reserved_units > self.inventory_units:
            raise ValueError("reserved_units cannot exceed inventory_units")
        if self.available_units != (self.inventory_units - self.reserved_units):
            raise ValueError("available_units must equal (inventory_units - reserved_units)")
        return self


class InventoryCreate(InventoryBase):
    pass


class InventoryUpdate(BaseModel):
    inventory_units: Optional[int] = Field(None, ge=0)
    reserved_units: Optional[int] = Field(None, ge=0)
    available_units: Optional[int] = Field(None, ge=0)
    reorder_point: Optional[int] = Field(None, ge=0)
    safety_stock_units: Optional[int] = Field(None, ge=0)
    inventory_date: Optional[date] = None


class InventoryResponse(InventoryBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ==========================================
# 4. Supply Event Schemas
# ==========================================

class SupplyEventBase(BaseModel):
    event_id: str = Field(..., max_length=100, description="Unique supply chain event identifier")
    product_id: int = Field(..., description="Referenced Product ID")
    supplier_id: int = Field(..., description="Referenced Supplier ID")
    event_date: date = Field(..., description="Operational event date")
    demand_units: int = Field(..., ge=0, description="Order demand in units")
    supplier_capacity_units: int = Field(..., ge=0, description="Capacity available at event time")
    planned_lead_time_days: float = Field(..., ge=0.0, description="Contracted/planned lead time")
    actual_lead_time_days: Optional[float] = Field(None, ge=0.0, description="Actual observed lead time")
    delay_occurred: bool = Field(default=False, description="Whether delivery delay happened")
    delay_days: float = Field(default=0.0, ge=0.0, description="Observed delay in days")
    transport_cost: Decimal = Field(default=Decimal("0.00"), ge=0, description="Baseline shipping cost")
    unit_cost: Decimal = Field(..., ge=0, description="Agreed unit cost")
    secondary_supplier_premium: Decimal = Field(
        default=Decimal("0.00"), ge=0, description="Cost surcharge for secondary supplier"
    )
    air_freight_cost: Decimal = Field(
        default=Decimal("0.00"), ge=0, description="Expedited air shipping cost"
    )
    service_risk_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Composite operational risk score"
    )


class SupplyEventCreate(SupplyEventBase):
    pass


class SupplyEventResponse(SupplyEventBase):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ==========================================
# 5. Prediction Schemas
# ==========================================

PredictionType = Literal["DELAY", "DEMAND", "SUPPLY_RISK"]


class PredictionBase(BaseModel):
    event_id: int = Field(..., description="Referenced Supply Event ID")
    model_name: str = Field(..., max_length=150, description="Name of predictive model")
    model_version: str = Field(..., max_length=50, description="Model release version")
    prediction_type: PredictionType = Field(..., description="Prediction category")
    delay_probability: float = Field(..., ge=0.0, le=1.0, description="Predicted delay probability")
    predicted_delay_days: float = Field(..., ge=0.0, description="Predicted delay duration in days")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Calculated composite risk score")
    risk_level: RiskLevel = Field(..., description="Risk category classification")
    prediction_timestamp: Optional[datetime] = None
    prediction_expires_at: Optional[datetime] = None


class PredictionCreate(PredictionBase):
    pass


class PredictionResponse(PredictionBase):
    id: int
    prediction_timestamp: datetime
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ==========================================
# 6. Optimization Run Schemas
# ==========================================

ObjectiveType = Literal["MIN_COST", "MIN_RISK", "MIN_DELAY", "BALANCED"]
OptimizationStatus = Literal["PENDING", "RUNNING", "COMPLETED", "FAILED"]
SolverStatus = Literal["OPTIMAL", "FEASIBLE", "INFEASIBLE", "UNBOUNDED", "ERROR"]


class OptimizationRunBase(BaseModel):
    event_id: int = Field(..., description="Referenced Supply Event ID")
    prediction_id: int = Field(..., description="Referenced Prediction ID")
    optimization_run_id: str = Field(..., max_length=100, description="Unique execution ID")
    objective_type: ObjectiveType = Field(..., description="Primary optimization goal")
    optimization_status: OptimizationStatus = Field(..., description="Lifecycle status of optimization")
    total_cost_baseline: Decimal = Field(default=Decimal("0.00"), ge=0, description="Baseline status quo cost")
    recommended_cost: Optional[Decimal] = Field(None, ge=0, description="Best solution total cost")
    service_level_target: float = Field(default=0.95, ge=0.0, le=1.0, description="Target SLA fraction")
    risk_weight: float = Field(default=0.33, ge=0.0, description="Objective weight for risk")
    cost_weight: float = Field(default=0.34, ge=0.0, description="Objective weight for cost")
    time_weight: float = Field(default=0.33, ge=0.0, description="Objective weight for time")
    solver_name: str = Field(default="PULP_CBC", max_length=100, description="Solver library used")
    solver_status: SolverStatus = Field(..., description="Status returned by solver")
    execution_time_ms: float = Field(default=0.0, ge=0.0, description="Solver runtime in milliseconds")


class OptimizationRunCreate(OptimizationRunBase):
    pass


class OptimizationRunResponse(OptimizationRunBase):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ==========================================
# 7. Recommendation Schemas
# ==========================================

StrategyType = Literal[
    "AIR_FREIGHT",
    "SECONDARY_SUPPLIER",
    "PRIMARY_SUPPLIER",
    "DELAY_LAUNCH",
    "EXPEDITE",
    "PARTIAL_ALLOCATION",
]


class RecommendationBase(BaseModel):
    optimization_run_id: int = Field(..., description="Associated optimization run ID")
    recommendation_code: str = Field(..., max_length=50, description="Code like ALT_A, ALT_B")
    strategy_type: StrategyType = Field(..., description="Prescriptive action category")
    strategy_description: str = Field(..., description="Detailed description of intervention")
    estimated_cost: Decimal = Field(..., ge=0, description="Expected operational cost")
    estimated_delay_days: float = Field(..., ge=0.0, description="Expected remaining delay")
    estimated_risk: float = Field(..., ge=0.0, le=1.0, description="Expected residual risk")
    service_level: float = Field(..., ge=0.0, le=1.0, description="Expected service level")
    cost_difference: Decimal = Field(default=Decimal("0.00"), description="Cost delta vs baseline")
    risk_difference: float = Field(default=0.0, description="Risk delta vs baseline")
    rank: int = Field(..., ge=1, description="Recommendation priority rank")
    is_recommended: bool = Field(default=False, description="Whether solver flagged this as top option")


class RecommendationCreate(RecommendationBase):
    pass


class RecommendationResponse(RecommendationBase):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ==========================================
# 8. Decision Schemas
# ==========================================

DecisionStatus = Literal["PENDING", "APPROVED", "REJECTED", "OVERRIDDEN", "CANCELLED"]
ExecutionStatus = Literal["NOT_STARTED", "IN_PROGRESS", "COMPLETED", "FAILED"]


class DecisionBase(BaseModel):
    decision_id: str = Field(..., max_length=100, description="Unique human decision tracking code")
    optimization_run_id: int = Field(..., description="Referenced Optimization Run ID")
    recommendation_id: int = Field(..., description="Chosen Recommendation ID")
    decision_maker: str = Field(..., max_length=150, description="Name or role of manager")
    decision_status: DecisionStatus = Field(..., description="State of decision approval")
    decision_reason: Optional[str] = Field(None, description="Manager rationale or override justification")
    selected_at: Optional[datetime] = None
    execution_status: ExecutionStatus = Field(default="NOT_STARTED", description="Execution tracking status")
    executed_at: Optional[datetime] = None


class DecisionCreate(DecisionBase):
    pass


class DecisionStatusUpdate(BaseModel):
    decision_status: Optional[DecisionStatus] = None
    execution_status: Optional[ExecutionStatus] = None
    decision_reason: Optional[str] = None
    executed_at: Optional[datetime] = None


class DecisionResponse(DecisionBase):
    id: int
    selected_at: datetime
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ==========================================
# 9. Outcome Schemas
# ==========================================

OutcomeStatus = Literal["SUCCESS", "PARTIAL_SUCCESS", "FAILED", "CANCELLED"]


class OutcomeBase(BaseModel):
    decision_id: int = Field(..., description="Referenced Decision ID")
    actual_cost: Decimal = Field(..., ge=0, description="Actual realized total cost")
    actual_delay_days: float = Field(..., ge=0.0, description="Actual observed delay in days")
    actual_service_level: float = Field(..., ge=0.0, le=1.0, description="Actual realized SLA")
    actual_risk: float = Field(..., ge=0.0, le=1.0, description="Actual observed disruption risk")
    actual_quantity_delivered: int = Field(..., ge=0, description="Actual units delivered")
    cost_variance: Decimal = Field(..., description="actual_cost - estimated_cost")
    delay_variance: float = Field(..., description="actual_delay_days - estimated_delay_days")
    service_level_variance: float = Field(..., description="actual_service_level - estimated_service_level")
    outcome_status: OutcomeStatus = Field(..., description="Evaluation outcome status")
    evaluation_date: date = Field(..., description="Date outcome was evaluated")
    notes: Optional[str] = Field(None, description="Operational outcome post-mortem notes")


class OutcomeCreate(OutcomeBase):
    pass


class OutcomeResponse(OutcomeBase):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ==========================================
# 10. Feedback Schemas
# ==========================================

FeedbackType = Literal[
    "COST_VARIANCE",
    "DELAY_VARIANCE",
    "PREDICTION_ERROR",
    "OPTIMIZATION_ERROR",
    "SERVICE_LEVEL_ERROR",
]


class FeedbackBase(BaseModel):
    outcome_id: int = Field(..., description="Referenced Outcome ID")
    optimization_run_id: int = Field(..., description="Referenced Optimization Run ID")
    feedback_type: FeedbackType = Field(..., description="Category of error signal")
    cost_error: Decimal = Field(default=Decimal("0.00"), description="Cost estimation error")
    delay_error: float = Field(default=0.0, description="Delay estimation error in days")
    risk_error: float = Field(default=0.0, description="Risk score variance")
    prediction_error: float = Field(default=0.0, description="Predictive model forecast error")
    weight_adjustment: float = Field(default=0.0, description="Recommended adjustment to optimization weights")
    learning_signal: Optional[str] = Field(None, description="Heuristic adjustment advice for next optimization")


class FeedbackCreate(FeedbackBase):
    pass


class FeedbackResponse(FeedbackBase):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
