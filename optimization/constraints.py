"""
Supply Prescript - Prescriptive Optimization Constraints & Business Rules
Author: Prasanna (ML & Optimization Lead)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class OptimizationConstraints:
    """
    Hard and soft business constraints for prescriptive supply chain optimization.
    """
    max_budget: float = 25000.0                # Hard budget limit ($)
    max_delivery_days: float = 12.0            # Hard delivery deadline SLA (days)
    max_acceptable_risk: str = "HIGH"          # Risk tolerance threshold
    penalty_cost_per_day_late: float = 850.0   # SLA penalty cost per delayed day ($/day)
    min_inventory_buffer: int = 50             # Minimum safety buffer needed (units)
    allow_split_shipments: bool = True         # Allow multi-supplier split allocation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_budget": self.max_budget,
            "max_delivery_days": self.max_delivery_days,
            "max_acceptable_risk": self.max_acceptable_risk,
            "penalty_cost_per_day_late": self.penalty_cost_per_day_late,
            "min_inventory_buffer": self.min_inventory_buffer,
            "allow_split_shipments": self.allow_split_shipments,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "OptimizationConstraints":
        if not data:
            return cls()
        return cls(
            max_budget=float(data.get("max_budget", 25000.0)),
            max_delivery_days=float(data.get("max_delivery_days", 12.0)),
            max_acceptable_risk=str(data.get("max_acceptable_risk", "HIGH")),
            penalty_cost_per_day_late=float(data.get("penalty_cost_per_day_late", 850.0)),
            min_inventory_buffer=int(data.get("min_inventory_buffer", 50)),
            allow_split_shipments=bool(data.get("allow_split_shipments", True)),
        )
