"""
Supply Prescript - Closed-Loop Evaluation Module
Evaluates real execution outcomes against model predictions and optimization targets.
Author: Prasanna (ML & Optimization Lead)
"""

from typing import Dict, Any, Optional
import numpy as np


def evaluate_decision(
    prediction: Dict[str, Any],
    decision: Dict[str, Any],
    actual_outcome: Dict[str, Any],
    penalty_per_day_late: float = 850.0,
) -> Dict[str, Any]:
    """
    Computes closed-loop evaluation metrics comparing the executed decision against the real-world outcome.
    
    Args:
        prediction: The initial predictive model output (predicted_delay_days, delay_probability).
        decision: The chosen prescriptive action (cost, delivery_days, id, name).
        actual_outcome: The recorded actual results (actual_cost, actual_delivery_days, notes).
        penalty_per_day_late: Dollar SLA penalty per day past scheduled delivery.
        
    Returns:
        Comprehensive evaluation dictionary containing:
        - cost_variance ($ and %)
        - delivery_variance (days)
        - prediction_error (days)
        - penalties_avoided ($)
        - mitigation_roi (%)
        - qualitative_outcome (str summary)
        - outcome_category ("SUCCESS_OPTIMAL", "COST_OVERRUN", "DELAY_OVERRUN", "CRITICAL_MISS")
    """
    pred_cost = float(decision.get("cost", 0.0))
    pred_delivery = float(decision.get("delivery_days", 0.0))
    initial_predicted_delay = float(prediction.get("predicted_delay_days", 0.0))
    
    actual_cost = float(actual_outcome.get("actual_cost", pred_cost))
    actual_delivery = float(actual_outcome.get("actual_delivery_days", pred_delivery))
    notes = str(actual_outcome.get("notes", ""))

    # Variances
    cost_variance = round(actual_cost - pred_cost, 2)
    cost_variance_pct = round((cost_variance / (pred_cost + 1e-5)) * 100, 1)

    delivery_variance = round(actual_delivery - pred_delivery, 1)
    
    # Baseline delay vs actual delay saved
    baseline_unmitigated_days = initial_predicted_delay + float(prediction.get("input_summary", {}).get("lead_time_days", 14.0))
    days_saved = round(max(0.0, baseline_unmitigated_days - actual_delivery), 1)

    # Financial penalties avoided
    penalties_prevented = round(days_saved * penalty_per_day_late, 2)
    net_financial_benefit = round(penalties_prevented - actual_cost, 2)

    # Mitigation ROI = (Penalties Avoided - Actual Cost) / Actual Cost * 100
    mitigation_roi_pct = round((net_financial_benefit / (actual_cost + 1e-5)) * 100, 1)

    # Determine qualitative outcome category
    if cost_variance <= 0 and delivery_variance <= 0:
        outcome_category = "SUCCESS_HIGH_ROI"
        qualitative_outcome = "Outstanding Execution: Completed under budget and on/ahead of schedule."
    elif cost_variance > 0 and delivery_variance <= 0:
        outcome_category = "COST_OVERRUN_ON_TIME"
        qualitative_outcome = f"Higher than expected cost (+${cost_variance:,.2f}), but delivered on schedule."
    elif cost_variance <= 0 and delivery_variance > 0:
        outcome_category = "DELAY_OVERRUN_UNDER_BUDGET"
        qualitative_outcome = f"Delivered under budget, but experienced minor delay (+{delivery_variance:.1f} days)."
    else:
        outcome_category = "UNFAVORABLE_OVERRUN"
        qualitative_outcome = f"Exceeded both cost (+${cost_variance:,.2f}) and delivery time (+{delivery_variance:.1f} days)."

    return {
        "predicted_cost": pred_cost,
        "actual_cost": actual_cost,
        "cost_variance": cost_variance,
        "cost_variance_pct": cost_variance_pct,
        "predicted_delivery_days": pred_delivery,
        "actual_delivery_days": actual_delivery,
        "delivery_variance_days": delivery_variance,
        "baseline_unmitigated_days": baseline_unmitigated_days,
        "actual_days_saved": days_saved,
        "penalties_avoided": penalties_prevented,
        "net_financial_benefit": net_financial_benefit,
        "mitigation_roi_pct": mitigation_roi_pct,
        "qualitative_outcome": qualitative_outcome,
        "outcome_category": outcome_category,
        "decision_name": decision.get("name", "Unknown Action"),
        "notes": notes,
    }


if __name__ == "__main__":
    sample_pred = {"predicted_delay_days": 10.0, "input_summary": {"lead_time_days": 14.0}}
    sample_dec = {"id": "OPTION_A", "name": "Expedited Air Freight", "cost": 15000.0, "delivery_days": 3.0}
    sample_act = {"actual_cost": 18000.0, "actual_delivery_days": 4.0, "notes": "Port gate fee + 1 day customs inspection"}

    res = evaluate_decision(sample_pred, sample_dec, sample_act)
    print("--- CLOSED-LOOP EVALUATION RESULT ---")
    print(f"Predicted Cost: ${res['predicted_cost']} | Actual Cost: ${res['actual_cost']} (Variance: ${res['cost_variance']:+,.2f})")
    print(f"Predicted Delivery: {res['predicted_delivery_days']}d | Actual: {res['actual_delivery_days']}d (Variance: {res['delivery_variance_days']:+.1f}d)")
    print(f"Outcome: {res['qualitative_outcome']}")
    print(f"Mitigation ROI: {res['mitigation_roi_pct']}%")
