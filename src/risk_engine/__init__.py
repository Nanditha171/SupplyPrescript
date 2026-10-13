"""Supply Prescript Risk Engine Module.

Closed-loop prescriptive analytics:
Shipment Data -> Predictive Probability -> Operational Risk Assessment ->
Risk Classification -> Risk Alerts -> Optimization Input.
"""

from src.risk_engine.config import RiskEngineConfig, load_risk_config
from src.risk_engine.probability_risk import classify_delay_probability, compute_delay_risk_score
from src.risk_engine.operational_risk import (
    calculate_supplier_risk,
    calculate_inventory_risk,
    calculate_capacity_risk,
    calculate_service_risk,
    compute_past_supplier_delay_rates,
)
from src.risk_engine.risk_scoring import (
    calculate_composite_risk_score,
    generate_risk_reasons,
    assess_single_shipment,
)
from src.risk_engine.alerts import RiskAlertEngine, generate_risk_alert
from src.risk_engine.schemas import (
    RiskAssessmentInput,
    RiskAssessmentResult,
    RiskAlert,
    RiskEngineSummary,
)

__all__ = [
    "RiskEngineConfig",
    "load_risk_config",
    "classify_delay_probability",
    "compute_delay_risk_score",
    "calculate_supplier_risk",
    "calculate_inventory_risk",
    "calculate_capacity_risk",
    "calculate_service_risk",
    "compute_past_supplier_delay_rates",
    "calculate_composite_risk_score",
    "generate_risk_reasons",
    "assess_single_shipment",
    "RiskAlertEngine",
    "generate_risk_alert",
    "RiskAssessmentInput",
    "RiskAssessmentResult",
    "RiskAlert",
    "RiskEngineSummary",
]
