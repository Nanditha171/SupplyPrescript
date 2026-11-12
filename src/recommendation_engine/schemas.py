"""Schemas and data models for Recommendation Engine (Day 11)."""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import List, Dict, Any, Optional


@dataclass
class RecommendationOption:
    """A business-actionable supply chain recovery strategy alternative."""

    recommendation_code: str  # ALT_A, ALT_B, etc.
    strategy_type: str  # AIR_FREIGHT, SECONDARY_SUPPLIER, PRIMARY_SUPPLIER, DELAY_LAUNCH, PARTIAL_ALLOCATION
    strategy_description: str
    estimated_cost: float
    estimated_delay_days: float
    estimated_risk: float
    service_level: float
    rank: int
    is_recommended: bool
    advantages: List[str]
    disadvantages: List[str]
    reasoning: str
    cost_difference: float = 0.0  # vs baseline or vs recommended
    risk_difference: float = 0.0
    feasibility_status: str = "FEASIBLE"
    allocated_quantities: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommendation_code": self.recommendation_code,
            "strategy_type": self.strategy_type,
            "strategy_description": self.strategy_description,
            "estimated_cost": round(self.estimated_cost, 2),
            "estimated_delay_days": round(self.estimated_delay_days, 1),
            "estimated_risk": round(self.estimated_risk, 4),
            "service_level": round(self.service_level, 4),
            "rank": self.rank,
            "is_recommended": self.is_recommended,
            "advantages": self.advantages,
            "disadvantages": self.disadvantages,
            "reasoning": self.reasoning,
            "cost_difference": round(self.cost_difference, 2),
            "risk_difference": round(self.risk_difference, 4),
            "feasibility_status": self.feasibility_status,
            "allocated_quantities": self.allocated_quantities,
        }


@dataclass
class RecommendationComparison:
    """Tabular and structured multi-alternative decision matrix."""

    optimization_run_id: str
    event_id: str
    ranking_mode: str
    recommendations: List[RecommendationOption]
    comparison_table: List[Dict[str, Any]]
    recommended_strategy_code: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "optimization_run_id": self.optimization_run_id,
            "event_id": self.event_id,
            "ranking_mode": self.ranking_mode,
            "recommended_strategy_code": self.recommended_strategy_code,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "comparison_table": self.comparison_table,
        }

