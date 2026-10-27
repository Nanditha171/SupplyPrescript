"""
Supply Prescript - Core API Integration Interfaces
Exported clean functions for Trisha (Backend) and Nanditha (Frontend).
Author: Prasanna (ML & Optimization Lead)
"""

import os
import sys
from typing import Dict, Any, List, Optional

# Ensure root directory is on PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ml.predictor import predict_delay as _predict_delay, get_model_metrics as _get_model_metrics
from optimization.optimizer import optimize_actions as _optimize_actions
from optimization.validator import audit_option as _audit_option
from optimization.constraints import OptimizationConstraints
from evaluation.evaluator import evaluate_decision as _evaluate_decision
from evaluation.feedback import (
    record_closed_loop_event as _record_closed_loop_event,
    get_feedback_summary as _get_feedback_summary,
    retrain_model_with_feedback as _retrain_model_with_feedback,
    load_history as _load_history,
)


def predict_delay(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Contract Function 1: Machine Learning Delay Prediction.
    
    Args:
        input_data: Dict with fields:
            - supplier_name: str
            - product_category: str
            - shipping_mode: str
            - lead_time_days: float
            - shipping_cost: float
            - inventory_level: int
            - order_quantity: int
            - supplier_delay_rate: float (0.0 to 1.0)
            - route_distance_km: float
            - weather_risk_index: float (0.0 to 1.0)
            - congestion_index: float (0.0 to 1.0)
            
    Returns:
        Dict with:
            - is_delayed: int (0 or 1)
            - delay_probability: float (0.0 to 1.0)
            - delay_probability_pct: float (0.0 to 100.0)
            - predicted_delay_days: float
            - expected_total_days: float
            - risk_level: str ("LOW", "MEDIUM", "HIGH", "CRITICAL")
            - confidence_score: float
            - top_risk_drivers: List[Dict]
    """
    return _predict_delay(input_data)


def optimize_actions(
    input_data: Dict[str, Any],
    prediction: Dict[str, Any],
    constraints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Contract Function 2: Prescriptive Action Optimization.
    
    Args:
        input_data: Dict containing shipment data (shipping_cost, lead_time_days, order_quantity, etc.)
        prediction: Dict containing prediction result from predict_delay()
        constraints: Optional Dict with max_budget, max_delivery_days, penalty_cost_per_day_late
        
    Returns:
        Dict with:
            - solver_status: str ("Optimal", "INFEASIBLE")
            - is_optimal: bool
            - recommended_option_id: str ("OPTION_A", "OPTION_B", "OPTION_C")
            - recommended_option: Dict
            - options: List of 3 option Dicts (Cost, Delivery Time, Constraints, Trade-offs)
            - why_explanation: Dict (Summary of constraints and rationale)
            - tradeoff_comparison: List[Dict]
    """
    return _optimize_actions(input_data, prediction, constraints)


def validate_recommendation(
    option: Dict[str, Any],
    constraints: Optional[Dict[str, Any]] = None,
    order_quantity: int = 100,
) -> Dict[str, Any]:
    """
    Contract Function 3: Constraint Audit Validation for an Option.
    
    Args:
        option: Dict containing cost, delivery_days, capacity
        constraints: Optional Dict with max_budget, max_delivery_days
        order_quantity: int
        
    Returns:
        Dict with:
            - status: str ("FEASIBLE" or "INFEASIBLE")
            - is_feasible: bool
            - constraint_checks: Dict[str, bool]
            - violations: List[str]
            - audit_summary: str
    """
    constraint_obj = OptimizationConstraints.from_dict(constraints)
    return _audit_option(option, constraint_obj, order_quantity=order_quantity)


def evaluate_decision(
    prediction: Dict[str, Any],
    decision: Dict[str, Any],
    actual_outcome: Dict[str, Any],
    penalty_per_day_late: float = 850.0,
) -> Dict[str, Any]:
    """
    Contract Function 4: Closed-Loop Decision Outcome Evaluation.
    
    Args:
        prediction: Dict from predict_delay()
        decision: Dict of the executed action (cost, delivery_days, name)
        actual_outcome: Dict with actual_cost, actual_delivery_days, notes
        penalty_per_day_late: float
        
    Returns:
        Dict with:
            - cost_variance: float ($)
            - cost_variance_pct: float (%)
            - delivery_variance_days: float
            - penalties_avoided: float ($)
            - mitigation_roi_pct: float (%)
            - qualitative_outcome: str
            - outcome_category: str
    """
    return _evaluate_decision(prediction, decision, actual_outcome, penalty_per_day_late=penalty_per_day_late)


def log_and_evaluate_cycle(
    shipment_data: Dict[str, Any],
    prediction: Dict[str, Any],
    decision: Dict[str, Any],
    actual_outcome: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Full closed-loop cycle evaluator and database recorder.
    """
    eval_result = evaluate_decision(prediction, decision, actual_outcome)
    record = _record_closed_loop_event(shipment_data, prediction, decision, actual_outcome, eval_result)
    return {
        "evaluation": eval_result,
        "event_record": record,
    }


def get_model_metrics() -> Dict[str, Any]:
    """Returns model performance metrics (Accuracy, Precision, Recall, F1, MAE, R2)."""
    return _get_model_metrics()


def get_history() -> List[Dict[str, Any]]:
    """Returns historical decision logs."""
    return _load_history()


def get_feedback_analytics() -> Dict[str, Any]:
    """Returns aggregated ROI and variance analytics."""
    return _get_feedback_summary()


def retrain_model_with_feedback() -> Dict[str, Any]:
    """Triggers retraining of the XGBoost models using closed-loop outcome feedback."""
    return _retrain_model_with_feedback()
