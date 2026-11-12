"""Tests for Closed-Loop Feedback & Learning Engine (Day 15)."""

from datetime import date
from decimal import Decimal
import pytest

from src.database.models import (
    Product,
    Supplier,
    SupplyEvent,
    Prediction,
    OptimizationRun,
    Recommendation,
    Decision,
    Outcome,
)
from src.feedback_engine.learning_rules import validate_learning_candidate, clamp_multiplier
from src.feedback_engine.cost_calibration import compute_ewma_multiplier
from src.feedback_engine.feedback_service import FeedbackService


def test_clamp_multiplier():
    val, clamped = clamp_multiplier(1.45, 0.70, 1.30)
    assert val == 1.30
    assert clamped is True

    val2, clamped2 = clamp_multiplier(0.55, 0.70, 1.30)
    assert val2 == 0.70
    assert clamped2 is True

    val3, clamped3 = clamp_multiplier(1.15, 0.70, 1.30)
    assert val3 == 1.15
    assert clamped3 is False


def test_outlier_validation():
    # Normal: $18,000 / $15,000 = 1.20 -> valid
    ok, _ = validate_learning_candidate(18000.0, 15000.0, "SUCCESS")
    assert ok is True

    # Outlier: $80,000 / $15,000 = 5.33 -> rejected
    ok2, msg2 = validate_learning_candidate(80000.0, 15000.0, "SUCCESS")
    assert ok2 is False
    assert "outlier" in msg2.lower()


def test_compute_ewma_multiplier():
    ratios = [1.20, 1.20, 1.20]
    mult, _ = compute_ewma_multiplier(ratios, alpha=0.20, initial_multiplier=1.0)
    # 1.0 -> 0.2*1.2 + 0.8*1.0 = 1.04
    # 1.04 -> 0.2*1.2 + 0.8*1.04 = 1.072
    # 1.072 -> 0.2*1.2 + 0.8*1.072 = 1.0976
    assert 1.05 < mult < 1.15


@pytest.fixture
def feedback_setup(db_session):
    p = Product(product_code="PROD_CHIP", product_name="Microchip", product_category="Tech", unit_cost=Decimal("125.50"), criticality_level="CRITICAL")
    s = Supplier(supplier_code="SUP_TAIWAN_2", supplier_name="Taiwan Semi", supplier_country="Taiwan", reliability_score=0.9, supplier_capacity_units=10000, average_lead_time_days=10.0, risk_level="LOW")
    db_session.add_all([p, s])
    db_session.commit()

    ev = SupplyEvent(
        event_id="EVT-FB-001", product_id=p.id, supplier_id=s.id, event_date=date.today(),
        demand_units=1000, supplier_capacity_units=10000, planned_lead_time_days=10.0, unit_cost=Decimal("125.50")
    )
    db_session.add(ev)
    db_session.commit()

    pred = Prediction(
        event_id=ev.id, model_name="model", model_version="1.0", prediction_type="DELAY",
        delay_probability=0.8, predicted_delay_days=14.0, risk_score=0.8, risk_level="HIGH"
    )
    db_session.add(pred)
    db_session.commit()

    run = OptimizationRun(
        event_id=ev.id, prediction_id=pred.id, optimization_run_id="OPT-FB-001",
        objective_type="MIN_COST", optimization_status="COMPLETED", solver_status="OPTIMAL"
    )
    db_session.add(run)
    db_session.commit()

    rec = Recommendation(
        optimization_run_id=run.id, recommendation_code="ALT_A", strategy_type="AIR_FREIGHT",
        strategy_description="Air freight", estimated_cost=Decimal("15000.00"), estimated_delay_days=2.0,
        estimated_risk=0.08, service_level=1.0, rank=1, is_recommended=True
    )
    db_session.add(rec)
    db_session.commit()

    dec = Decision(
        decision_id="DEC-FB-001", optimization_run_id=run.id, recommendation_id=rec.id,
        decision_maker="Manager", decision_status="APPROVED", execution_status="COMPLETED"
    )
    db_session.add(dec)
    db_session.commit()

    out = Outcome(
        decision_id=dec.id, actual_cost=Decimal("18000.00"), actual_delay_days=2.0,
        actual_service_level=1.0, actual_risk=0.08, actual_quantity_delivered=1000,
        cost_variance=Decimal("3000.00"), delay_variance=0.0, service_level_variance=0.0,
        outcome_status="SUCCESS", evaluation_date=date.today()
    )
    db_session.add(out)
    db_session.commit()

    return out


def test_feedback_minimum_observation_safeguard(db_session, feedback_setup):
    out = feedback_setup
    svc = FeedbackService()

    # With only 1 observation and min_observations=3, multiplier must remain 1.0 (baseline)
    fb = svc.process_outcome_feedback(db_session, out.id, min_observations=3)
    assert fb.id is not None
    assert float(fb.cost_error) == 3000.0
    assert fb.weight_adjustment == 0.0
    assert "1.00" in fb.learning_signal
    assert "retaining baseline" in fb.learning_signal.lower()


def test_feedback_learning_active_when_min_observations_met(db_session, feedback_setup):
    out = feedback_setup
    svc = FeedbackService()

    # Setting min_observations=1 to simulate threshold being met
    fb = svc.process_outcome_feedback(db_session, out.id, min_observations=1)
    assert fb.id is not None
    assert float(fb.cost_error) == 3000.0
    assert fb.weight_adjustment > 0.0  # Learned upward adjustment for air freight
    assert "AIR_FREIGHT" in fb.learning_signal

