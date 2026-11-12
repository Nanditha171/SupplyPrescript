"""Tests for Outcome Evaluation Engine (Day 14)."""

from datetime import date
from decimal import Decimal
import pytest

from src.database.models import OptimizationRun, Recommendation, Decision, SupplyEvent, Product, Supplier, Prediction
from src.outcome_engine.outcome_service import OutcomeService, OutcomeEvaluationError
from src.outcome_engine.schemas import OutcomeRecordRequest
from src.outcome_engine.variance_calculator import calculate_variances
from src.outcome_engine.performance_metrics import classify_outcome_status


@pytest.fixture
def executed_decision_setup(db_session):
    p = Product(product_code="PROD_MICROCHIP", product_name="Microchip X1", product_category="Electronics", unit_cost=Decimal("125.50"), criticality_level="CRITICAL")
    s = Supplier(supplier_code="SUP_TAIWAN", supplier_name="Pacific Silicon Ltd", supplier_country="Taiwan", reliability_score=0.88, supplier_capacity_units=15000, average_lead_time_days=14.0, risk_level="HIGH")
    db_session.add_all([p, s])
    db_session.commit()

    ev = SupplyEvent(
        event_id="EVT-001", product_id=p.id, supplier_id=s.id, event_date=date.today(),
        demand_units=1000, supplier_capacity_units=15000, planned_lead_time_days=14.0, unit_cost=Decimal("125.50")
    )
    db_session.add(ev)
    db_session.commit()

    pred = Prediction(
        event_id=ev.id, model_name="delay_classifier", model_version="1.0.0", prediction_type="DELAY",
        delay_probability=0.87, predicted_delay_days=14.0, risk_score=0.82, risk_level="CRITICAL"
    )
    db_session.add(pred)
    db_session.commit()

    run = OptimizationRun(
        event_id=ev.id, prediction_id=pred.id, optimization_run_id="OPT-RUN-001",
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
        decision_id="DEC-001", optimization_run_id=run.id, recommendation_id=rec.id,
        decision_maker="Global Operations VP", decision_status="APPROVED", execution_status="COMPLETED"
    )
    db_session.add(dec)
    db_session.commit()

    return dec, rec


def test_calculate_variances_core_scenario():
    """Core demonstration scenario: Estimated $15k, Actual $18k -> Cost Variance +$3,000."""
    res = calculate_variances(
        actual_cost=18000.0,
        estimated_cost=15000.0,
        actual_delay_days=2.0,
        estimated_delay_days=2.0,
        actual_service_level=1.0,
        estimated_service_level=1.0,
    )
    assert res["cost_variance"] == 3000.0
    assert res["cost_variance_pct"] == 20.0
    assert res["is_cost_favorable"] is False  # Higher cost is unfavorable
    assert res["delay_variance"] == 0.0
    assert res["is_delay_favorable"] is True


def test_record_outcome_success(db_session, executed_decision_setup):
    dec, rec = executed_decision_setup
    svc = OutcomeService()

    req = OutcomeRecordRequest(
        decision_id=dec.id,
        actual_cost=18000.0,
        actual_delay_days=2.0,
        actual_service_level=1.0,
        actual_risk=0.08,
        actual_quantity_delivered=1000,
        notes="Shipment delivered via expedited charter with $3,000 surcharge.",
    )

    outcome = svc.record_outcome(db_session, req)
    assert outcome.id is not None
    assert float(outcome.cost_variance) == 3000.0
    assert outcome.delay_variance == 0.0
    assert outcome.outcome_status == "SUCCESS"


def test_cannot_record_outcome_for_unexecuted_decision(db_session, executed_decision_setup):
    dec, rec = executed_decision_setup
    dec.execution_status = "IN_PROGRESS"
    db_session.commit()

    svc = OutcomeService()
    req = OutcomeRecordRequest(
        decision_id=dec.id,
        actual_cost=18000.0,
        actual_delay_days=2.0,
        actual_service_level=1.0,
        actual_risk=0.08,
        actual_quantity_delivered=1000,
    )
    with pytest.raises(OutcomeEvaluationError):
        svc.record_outcome(db_session, req)


def test_prevent_duplicate_outcome(db_session, executed_decision_setup):
    dec, rec = executed_decision_setup
    svc = OutcomeService()
    req = OutcomeRecordRequest(
        decision_id=dec.id,
        actual_cost=18000.0,
        actual_delay_days=2.0,
        actual_service_level=1.0,
        actual_risk=0.08,
        actual_quantity_delivered=1000,
    )
    svc.record_outcome(db_session, req)

    with pytest.raises(OutcomeEvaluationError):
        svc.record_outcome(db_session, req)

