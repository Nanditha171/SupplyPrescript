"""Unit tests for probability-based risk classification and thresholding."""

import pytest
from src.risk_engine.config import DelayProbabilityConfig
from src.risk_engine.probability_risk import (
    classify_delay_probability,
    compute_delay_risk_score,
    is_predicted_delayed,
    validate_probability,
)


def test_classify_delay_probability_boundaries():
    """Verify exact boundary classifications under default thresholds."""
    # LOW: [0.00, 0.30)
    assert classify_delay_probability(0.0) == "LOW"
    assert classify_delay_probability(0.15) == "LOW"
    assert classify_delay_probability(0.2999) == "LOW"

    # MEDIUM: [0.30, 0.60)
    assert classify_delay_probability(0.30) == "MEDIUM"
    assert classify_delay_probability(0.45) == "MEDIUM"
    assert classify_delay_probability(0.5999) == "MEDIUM"

    # HIGH: [0.60, 0.80)
    assert classify_delay_probability(0.60) == "HIGH"
    assert classify_delay_probability(0.70) == "HIGH"
    assert classify_delay_probability(0.7999) == "HIGH"

    # CRITICAL: [0.80, 1.00]
    assert classify_delay_probability(0.80) == "CRITICAL"
    assert classify_delay_probability(0.95) == "CRITICAL"
    assert classify_delay_probability(1.0) == "CRITICAL"


def test_custom_probability_thresholds():
    """Verify classification with custom threshold configuration."""
    cfg = DelayProbabilityConfig(
        low_threshold=0.20,
        medium_threshold=0.50,
        high_threshold=0.75,
        classification_threshold=0.65,
    )
    assert classify_delay_probability(0.19, thresholds=cfg) == "LOW"
    assert classify_delay_probability(0.20, thresholds=cfg) == "MEDIUM"
    assert classify_delay_probability(0.49, thresholds=cfg) == "MEDIUM"
    assert classify_delay_probability(0.50, thresholds=cfg) == "HIGH"
    assert classify_delay_probability(0.74, thresholds=cfg) == "HIGH"
    assert classify_delay_probability(0.75, thresholds=cfg) == "CRITICAL"


def test_invalid_and_missing_probabilities():
    """Verify rejection of NaN, Inf, out-of-bounds, and non-numeric probabilities."""
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        validate_probability(-0.01)

    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        validate_probability(1.01)

    with pytest.raises(ValueError, match="finite"):
        validate_probability(float("nan"))

    with pytest.raises(ValueError, match="finite"):
        validate_probability(float("inf"))

    with pytest.raises(TypeError, match="numeric"):
        validate_probability("0.5")  # type: ignore

    with pytest.raises(TypeError, match="numeric"):
        validate_probability(None)  # type: ignore

    with pytest.raises(TypeError, match="numeric"):
        validate_probability(True)  # boolean should not masquerade as numeric 1.0


def test_compute_delay_risk_score():
    """Verify delay risk score normalization."""
    assert compute_delay_risk_score(0.0) == 0.0
    assert compute_delay_risk_score(0.78542) == 0.7854
    assert compute_delay_risk_score(1.0) == 1.0


def test_is_predicted_delayed_threshold():
    """Verify binary delay classification against validated Day 7 threshold."""
    # Day 7 recommended threshold = 0.65
    assert is_predicted_delayed(0.64, classification_threshold=0.65) is False
    assert is_predicted_delayed(0.65, classification_threshold=0.65) is True
    assert is_predicted_delayed(0.85, classification_threshold=0.65) is True
    assert is_predicted_delayed(0.20, classification_threshold=0.65) is False
