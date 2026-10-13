"""Unit tests for composite risk score calculation, weight validation, and reason generation."""

import pytest
from src.risk_engine.config import RiskEngineConfig, ScoringWeightsConfig, load_risk_config
from src.risk_engine.risk_scoring import (
    assess_single_shipment,
    calculate_composite_risk_score,
    generate_risk_reasons,
    map_score_to_risk_level,
)
from src.risk_engine.schemas import RiskAssessmentInput


@pytest.fixture
def default_cfg():
    return load_risk_config()


def test_composite_score_calculation(default_cfg):
    """Verify transparent weighted calculation across all dimensions."""
    # delay: 0.40 * 0.80 = 0.32
    # supplier: 0.20 * 0.50 = 0.10
    # inventory: 0.20 * 0.60 = 0.12
    # capacity: 0.10 * 0.40 = 0.04
    # service: 0.10 * 0.70 = 0.07
    # total = 0.65
    score, unavail = calculate_composite_risk_score(
        delay_risk_score=0.80,
        supplier_risk_score=0.50,
        inventory_risk_score=0.60,
        capacity_risk_score=0.40,
        service_risk_score=0.70,
        config=default_cfg,
    )
    assert score == 0.65
    assert unavail == []


def test_weight_validation_and_no_silent_normalization():
    """Verify that invalid weights summing != 1.0 fail loudly without silent normalization."""
    # Sum is 0.90 -> must raise ValueError
    with pytest.raises(ValueError, match="Scoring weights must sum to 1.0"):
        ScoringWeightsConfig(
            delay_weight=0.30,
            supplier_weight=0.20,
            inventory_weight=0.20,
            capacity_weight=0.10,
            service_weight=0.10,
        )

    # Sum is 1.20 -> must raise ValueError
    with pytest.raises(ValueError, match="Scoring weights must sum to 1.0"):
        ScoringWeightsConfig(
            delay_weight=0.50,
            supplier_weight=0.30,
            inventory_weight=0.20,
            capacity_weight=0.10,
            service_weight=0.10,
        )


def test_missing_dimension_renormalization_policy(default_cfg):
    """Verify dynamic weight renormalization when a dimension (service) is unavailable."""
    # Default policy is 'renormalize'
    # Available weights: 0.40 + 0.20 + 0.20 + 0.10 = 0.90
    # Weighted sum: 0.4*0.5 + 0.2*0.5 + 0.2*0.5 + 0.1*0.5 = 0.5 * 0.90 = 0.45
    # Composite score = 0.45 / 0.90 = 0.50
    score, unavail = calculate_composite_risk_score(
        delay_risk_score=0.50,
        supplier_risk_score=0.50,
        inventory_risk_score=0.50,
        capacity_risk_score=0.50,
        service_risk_score=None,
        config=default_cfg,
    )
    assert score == 0.50
    assert unavail == ["service"]


def test_missing_dimension_require_all_policy(default_cfg):
    """Verify that require_all policy raises ValueError when a dimension is missing."""
    cfg_copy = default_cfg.model_copy(deep=True)
    cfg_copy.scoring.missing_dimension_policy = "require_all"

    with pytest.raises(ValueError, match="require_all.*failed"):
        calculate_composite_risk_score(
            delay_risk_score=0.50,
            supplier_risk_score=0.50,
            inventory_risk_score=0.50,
            capacity_risk_score=0.50,
            service_risk_score=None,
            config=cfg_copy,
        )


def test_all_dimensions_missing_fails(default_cfg):
    """Verify that failure occurs if zero usable dimensions remain."""
    with pytest.raises(ValueError, match="No usable risk dimensions remain"):
        calculate_composite_risk_score(
            delay_risk_score=None,  # type: ignore
            supplier_risk_score=None,  # type: ignore
            inventory_risk_score=None,  # type: ignore
            capacity_risk_score=None,  # type: ignore
            service_risk_score=None,
            config=default_cfg,
        )


def test_map_score_to_risk_level(default_cfg):
    """Verify risk-level band assignments."""
    assert map_score_to_risk_level(0.15, default_cfg) == "LOW"
    assert map_score_to_risk_level(0.2999, default_cfg) == "LOW"
    assert map_score_to_risk_level(0.30, default_cfg) == "MEDIUM"
    assert map_score_to_risk_level(0.5999, default_cfg) == "MEDIUM"
    assert map_score_to_risk_level(0.60, default_cfg) == "HIGH"
    assert map_score_to_risk_level(0.7999, default_cfg) == "HIGH"
    assert map_score_to_risk_level(0.80, default_cfg) == "CRITICAL"
    assert map_score_to_risk_level(1.00, default_cfg) == "CRITICAL"


def test_generate_risk_reasons_triggers(default_cfg):
    """Verify that explicit threshold breaches trigger appropriate reason codes."""
    codes, descs = generate_risk_reasons(
        delay_prob=0.88,
        delay_risk_score=0.88,
        supplier_reliability=0.65,
        historical_delay_rate=0.35,
        coverage_ratio=0.30,
        shortage_flag=1,
        capacity_utilization=1.05,
        product_criticality="CRITICAL",
        config=default_cfg,
    )
    assert "CRITICAL_DELAY_PROBABILITY" in codes
    assert "CRITICAL_SUPPLIER_RELIABILITY" in codes
    assert "HIGH_HISTORICAL_SUPPLIER_DELAYS" in codes
    assert "CRITICAL_INVENTORY_SHORTAGE" in codes
    assert "CRITICAL_CAPACITY_UTILIZATION" in codes
    assert "CRITICAL_PRODUCT_EXPOSURE" in codes
    assert len(codes) == len(descs)


def test_assess_single_shipment(default_cfg):
    """Verify single shipment end-to-end assessment."""
    inp = RiskAssessmentInput(
        event_id="EVT_TEST_001",
        delay_probability=0.82,
        supplier_reliability=0.90,
        historical_delay_rate=0.10,
        inventory_coverage_ratio=0.45,
        inventory_units=450,
        demand_units=1000,
        shortage_flag=1,
        supplier_capacity_units=1200,
        capacity_utilization=0.83,
        product_criticality="HIGH",
    )
    res = assess_single_shipment(inp, config=default_cfg)

    assert res.event_id == "EVT_TEST_001"
    assert res.predicted_delay is True  # 0.82 >= 0.65 threshold
    assert 0.0 <= res.composite_risk_score <= 1.0
    assert res.risk_level in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert len(res.risk_reasons) > 0

    # Optimization engine contract check
    contract = res.to_optimization_contract()
    assert contract["event_id"] == "EVT_TEST_001"
    assert "risk_score" in contract
    assert "model_version" in contract
