"""Performance classification rules and operational KPI calculators (Day 14)."""

from typing import Dict, Any


def classify_outcome_status(
    actual_service_level: float,
    delay_variance: float,
    cost_variance: float,
    estimated_cost: float,
    is_cancelled: bool = False,
) -> str:
    """Classify the operational success status of an executed recovery strategy.

    Rules:
        - CANCELLED: The strategy was officially aborted or cancelled prior to completion.
        - FAILED: Severe service disruption (SLA < 80%), severe delay (> 5 days overrun),
                  or catastrophic cost overrun (> 50%).
        - SUCCESS: High service fulfillment (SLA >= 95%), negligible delay (<= 0.5 days),
                   and controlled cost expenditure (overrun <= 25%).
        - PARTIAL_SUCCESS: Acceptable recovery meeting intermediate thresholds.

    Returns:
        One of 'SUCCESS', 'PARTIAL_SUCCESS', 'FAILED', 'CANCELLED'.
    """
    if is_cancelled:
        return "CANCELLED"

    cost_overrun_pct = (cost_variance / estimated_cost) if estimated_cost > 0 else 0.0

    # Failure criteria
    if actual_service_level < 0.80 or delay_variance > 5.0 or cost_overrun_pct > 0.50:
        return "FAILED"

    # Success criteria
    if actual_service_level >= 0.95 and delay_variance <= 0.5 and cost_overrun_pct <= 0.25:
        return "SUCCESS"

    return "PARTIAL_SUCCESS"

