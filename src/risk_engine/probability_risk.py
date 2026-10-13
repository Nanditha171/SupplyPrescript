"""Probability-based risk classification and thresholding.

Differentiates:
1. Operational Risk Boundaries (LOW, MEDIUM, HIGH, CRITICAL)
2. Validated Decision Classification Threshold (predicts whether shipment is delayed)
"""

import math
from typing import Optional, Union
from src.risk_engine.config import DelayProbabilityConfig


def validate_probability(probability: Union[float, int]) -> float:
    """Validate that probability is numeric, finite, and in [0.0, 1.0].

    Args:
        probability: Probability value to validate.

    Returns:
        float: Validated float value.

    Raises:
        TypeError: If probability is not a float or integer.
        ValueError: If probability is NaN, infinite, or outside [0.0, 1.0].
    """
    if not isinstance(probability, (float, int)) or isinstance(probability, bool):
        raise TypeError(
            f"Delay probability must be a numeric float or int, got {type(probability).__name__}: {probability}"
        )

    prob_float = float(probability)

    if math.isnan(prob_float) or math.isinf(prob_float):
        raise ValueError(f"Delay probability must be finite, got: {prob_float}")

    if not (0.0 <= prob_float <= 1.0):
        raise ValueError(
            f"Delay probability must be between 0.0 and 1.0 inclusive, got: {prob_float}"
        )

    return prob_float


def classify_delay_probability(
    probability: float,
    thresholds: Optional[DelayProbabilityConfig] = None,
) -> str:
    """Classify predicted delay probability into operational risk level bands.

    Default Boundaries:
    - LOW: probability below 0.30
    - MEDIUM: probability from 0.30 to below 0.60
    - HIGH: probability from 0.60 to below 0.80
    - CRITICAL: probability of 0.80 or higher

    Args:
        probability: Predicted probability between 0.0 and 1.0.
        thresholds: Configurable delay probability boundaries.

    Returns:
        Risk level string: 'LOW', 'MEDIUM', 'HIGH', or 'CRITICAL'.

    Raises:
        TypeError: If probability is not numeric.
        ValueError: If probability is NaN, Inf, or out of [0.0, 1.0].
    """
    prob = validate_probability(probability)

    low_thresh = thresholds.low_threshold if thresholds else 0.30
    med_thresh = thresholds.medium_threshold if thresholds else 0.60
    high_thresh = thresholds.high_threshold if thresholds else 0.80

    if prob < low_thresh:
        return "LOW"
    elif prob < med_thresh:
        return "MEDIUM"
    elif prob < high_thresh:
        return "HIGH"
    else:
        return "CRITICAL"


def compute_delay_risk_score(probability: float) -> float:
    """Normalize predicted delay probability into a normalized risk score [0.0, 1.0].

    Args:
        probability: Validated delay probability.

    Returns:
        float: Normalized delay risk score rounded to 4 decimals.
    """
    prob = validate_probability(probability)
    return round(prob, 4)


def is_predicted_delayed(
    probability: float,
    classification_threshold: float = 0.65,
) -> bool:
    """Evaluate whether shipment is classified as delayed under approved threshold.

    Note: Distinct from risk bands; separates binary decision from risk severity.

    Args:
        probability: Predicted delay probability.
        classification_threshold: Approved operating decision threshold (Day 7: 0.65).

    Returns:
        bool: True if probability >= classification_threshold, False otherwise.
    """
    prob = validate_probability(probability)
    if not (0.0 <= classification_threshold <= 1.0):
        raise ValueError(
            f"Classification threshold must be in [0.0, 1.0], got {classification_threshold}"
        )
    return prob >= classification_threshold
