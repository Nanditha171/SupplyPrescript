"""Learning safeguards and integrity rules for prescriptive feedback calibration (Day 15)."""

from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class FeedbackSafeguardError(Exception):
    """Raised when feedback violates operational learning safeguards."""
    pass


def validate_learning_candidate(
    actual_cost: float,
    estimated_cost: float,
    outcome_status: str,
) -> Tuple[bool, str]:
    """Validate whether an outcome is suitable to update optimization parameters.

    Checks:
        1. Positive estimated cost and non-negative actual cost.
        2. Outcome not cancelled.
        3. Outlier ratio sanity check: 0.20 <= (actual / estimated) <= 3.00.

    Returns:
        Tuple of (is_valid, reason)
    """
    if outcome_status == "CANCELLED":
        return False, "Cancelled outcomes cannot be used for parameter calibration."

    if estimated_cost <= 0:
        return False, f"Estimated cost must be positive (got {estimated_cost})."

    if actual_cost < 0:
        return False, f"Actual cost cannot be negative (got {actual_cost})."

    ratio = actual_cost / estimated_cost
    if ratio < 0.20 or ratio > 3.00:
        return False, f"Cost ratio {ratio:.2f} flagged as statistical outlier (permitted range [0.20, 3.00])."

    return True, "Passed sanity checks."


def clamp_multiplier(
    multiplier: float,
    lower_bound: float = 0.70,
    upper_bound: float = 1.30,
) -> Tuple[float, bool]:
    """Ensure learned cost multiplier stays strictly within configured safety bounds (±30%).

    Returns:
        Tuple of (clamped_multiplier, was_clamped)
    """
    if multiplier < lower_bound:
        return lower_bound, True
    if multiplier > upper_bound:
        return upper_bound, True
    return multiplier, False

