"""Unit tests for Risk Alert Engine and deduplication handling."""

import pytest
from src.risk_engine.alerts import RiskAlertEngine, generate_risk_alert
from src.risk_engine.config import load_risk_config
from src.risk_engine.schemas import RiskAssessmentResult


@pytest.fixture
def default_cfg():
    return load_risk_config()


def create_sample_assessment(
    event_id: str,
    score: float,
    level: str,
    reasons: list = None,
) -> RiskAssessmentResult:
    return RiskAssessmentResult(
        event_id=event_id,
        delay_probability=score,
        predicted_delay=score >= 0.65,
        delay_risk_score=score,
        supplier_risk_score=0.20,
        inventory_risk_score=score,
        capacity_risk_score=0.30,
        service_risk_score=None,
        composite_risk_score=score,
        risk_level=level,
        risk_reasons=reasons or ["TEST_REASON"],
        model_version="1.0.0",
        configuration_version="1.0.0",
    )


def test_alert_severity_generation(default_cfg):
    """Verify appropriate alert types and messages based on severity."""
    # Critical Alert
    crit_res = create_sample_assessment("EVT_CRIT", 0.85, "CRITICAL")
    crit_alert = generate_risk_alert(crit_res, default_cfg)
    assert crit_alert is not None
    assert crit_alert.alert_type == "PRIORITY_DISRUPTION_ALERT"
    assert crit_alert.severity == "CRITICAL"
    assert "PRIORITY ALERT" in crit_alert.alert_message

    # High Alert
    high_res = create_sample_assessment("EVT_HIGH", 0.68, "HIGH")
    high_alert = generate_risk_alert(high_res, default_cfg)
    assert high_alert is not None
    assert high_alert.alert_type == "WARNING_DISRUPTION_ALERT"
    assert high_alert.severity == "HIGH"
    assert "WARNING ALERT" in high_alert.alert_message

    # Medium Alert
    med_res = create_sample_assessment("EVT_MED", 0.45, "MEDIUM")
    med_alert = generate_risk_alert(med_res, default_cfg)
    assert med_alert is not None
    assert med_alert.alert_type == "MONITORING_NOTICE"
    assert med_alert.severity == "MEDIUM"

    # Low Risk -> No urgent alert by default
    low_res = create_sample_assessment("EVT_LOW", 0.15, "LOW")
    low_alert = generate_risk_alert(low_res, default_cfg)
    assert low_alert is None


def test_duplicate_alert_handling(default_cfg):
    """Verify that repeated assessments with the same event_id and type are deduplicated."""
    engine = RiskAlertEngine(config=default_cfg)

    res1 = create_sample_assessment("EVT_001", 0.85, "CRITICAL")
    res2 = create_sample_assessment("EVT_002", 0.70, "HIGH")

    # First run
    alerts_run1 = engine.process_assessments([res1, res2])
    assert len(alerts_run1) == 2

    # Second run with exact same assessments -> should produce 0 new alerts
    alerts_run2 = engine.process_assessments([res1, res2])
    assert len(alerts_run2) == 0

    # Total accumulated history counts
    assert engine.alert_counts["CRITICAL"] == 1
    assert engine.alert_counts["HIGH"] == 1
