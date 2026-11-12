"""Data structures and parser for optimization solver inputs and outputs."""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Any
from src.optimization.cost_calculator import CostBreakdown


@dataclass
class OptimizationInput:
    """Operational parameters provided to the prescriptive solver."""

    event_id: str
    product_code: str
    demand_units: int
    available_inventory: int
    unit_cost: float
    base_transport_cost: float
    primary_capacity: int
    secondary_capacity: int
    air_freight_capacity: int
    predicted_delay_days: float
    delay_probability: float
    composite_risk_score: float
    secondary_premium_rate: float = 0.10
    air_freight_base_cost: Optional[float] = None
    service_level_target: float = 0.95
    objective_type: str = "MIN_COST"  # MIN_COST, MIN_RISK, MIN_DELAY, BALANCED
    cost_weight: float = 1.00
    risk_weight: float = 0.00
    time_weight: float = 0.00
    learned_multipliers: Dict[str, float] = field(default_factory=dict)
    allow_unmet_demand: bool = True
    allow_launch_delay: bool = True
    forced_strategy: Optional[str] = None  # To evaluate specific alternatives directly


@dataclass
class OptimizationOutput:
    """Prescriptive optimization solution output with full audit metadata."""

    optimization_run_id: str
    event_id: str
    solver_status: str  # OPTIMAL, FEASIBLE, INFEASIBLE, UNBOUNDED, NOT_SOLVED, ERROR
    objective_value: float
    estimated_total_cost: float
    primary_supplier_quantity: int
    secondary_supplier_quantity: int
    air_freight_quantity: int
    unmet_demand: int
    is_launch_delayed: bool
    selected_strategies: List[str]
    cost_breakdown: CostBreakdown
    constraint_status: Dict[str, Any]
    solver_execution_time: float
    service_level_achieved: float
    estimated_delivery_days: float
    estimated_risk: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "optimization_run_id": self.optimization_run_id,
            "event_id": self.event_id,
            "solver_status": self.solver_status,
            "objective_value": round(self.objective_value, 4),
            "estimated_total_cost": round(self.estimated_total_cost, 2),
            "primary_supplier_quantity": self.primary_supplier_quantity,
            "secondary_supplier_quantity": self.secondary_supplier_quantity,
            "air_freight_quantity": self.air_freight_quantity,
            "unmet_demand": self.unmet_demand,
            "is_launch_delayed": self.is_launch_delayed,
            "selected_strategies": self.selected_strategies,
            "cost_breakdown": self.cost_breakdown.to_dict(),
            "constraint_status": self.constraint_status,
            "solver_execution_time": round(self.solver_execution_time, 4),
            "service_level_achieved": round(self.service_level_achieved, 4),
            "estimated_delivery_days": round(self.estimated_delivery_days, 1),
            "estimated_risk": round(self.estimated_risk, 4),
        }

