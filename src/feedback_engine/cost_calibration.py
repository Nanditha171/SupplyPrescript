"""Cost multiplier calibration using Exponentially Weighted Moving Average (EWMA) (Day 15)."""

from typing import List, Tuple
from src.feedback_engine.learning_rules import clamp_multiplier


def compute_ewma_multiplier(
    historical_ratios: List[float],
    alpha: float = 0.20,
    initial_multiplier: float = 1.0,
    lower_bound: float = 0.70,
    upper_bound: float = 1.30,
) -> Tuple[float, bool]:
    """Compute calibrated cost multiplier via EWMA across valid historical observations.

    Equation:
        mu_t = alpha * ratio_t + (1 - alpha) * mu_{t-1}

    Args:
        historical_ratios: Chronological list of valid actual/estimated cost ratios.
        alpha: Smoothing parameter (learning rate), e.g. 0.20.
        initial_multiplier: Baseline multiplier before observations (1.0).
        lower_bound: Minimum permitted multiplier (0.70).
        upper_bound: Maximum permitted multiplier (1.30).

    Returns:
        Tuple of (clamped_calibrated_multiplier, was_clamped)
    """
    if not historical_ratios:
        return initial_multiplier, False

    current_mu = initial_multiplier
    for ratio in historical_ratios:
        current_mu = (alpha * ratio) + ((1.0 - alpha) * current_mu)

    clamped_val, was_clamped = clamp_multiplier(current_mu, lower_bound, upper_bound)
    return round(clamped_val, 4), was_clamped

