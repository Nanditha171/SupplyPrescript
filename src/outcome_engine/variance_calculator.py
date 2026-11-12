"""Variance calculation for supply chain post-execution outcomes (Day 14)."""

from typing import Dict, Any, Tuple


def calculate_variances(
    actual_cost: float,
    estimated_cost: float,
    actual_delay_days: float,
    estimated_delay_days: float,
    actual_service_level: float,
    estimated_service_level: float,
) -> Dict[str, Any]:
    """Calculate operational and financial variances with directionality semantics.

    Directionality:
        - Cost variance: Negative is favorable (spent less than estimated).
        - Delay variance: Negative is favorable (faster turnaround than estimated).
        - Service-level variance: Positive is favorable (higher SLA than estimated).

    Returns:
        Dictionary of raw variances, percentages, and favorability flags.
    """
    cost_var = actual_cost - estimated_cost
    cost_var_pct = (cost_var / estimated_cost) * 100.0 if estimated_cost > 0 else 0.0
    is_cost_favorable = cost_var <= 0.0

    delay_var = actual_delay_days - estimated_delay_days
    is_delay_favorable = delay_var <= 0.0

    service_var = actual_service_level - estimated_service_level
    is_service_favorable = service_var >= 0.0

    return {
        "cost_variance": round(cost_var, 2),
        "cost_variance_pct": round(cost_var_pct, 2),
        "is_cost_favorable": is_cost_favorable,
        "delay_variance": round(delay_var, 2),
        "is_delay_favorable": is_delay_favorable,
        "service_level_variance": round(service_var, 4),
        "is_service_favorable": is_service_favorable,
    }

