"""Tests for Decision Management & Auditing (Day 13)."""

from decimal import Decimal
import pytest
from src.database.models import OptimizationRun, Recommendation, SupplyEvent, Product, Supplier, Prediction
from src.decision_management.decision_service import DecisionService
from src.decision_management.schemas import DecisionCreateRequest, DecisionUpdateRequest
from src.decision_management.decision_validation import DecisionValidationError


@pytest.fixture
def test_setup(db_session):
    p = Product(product_code="TEST_P", product_name="Test Product", product_category="Tech", unit_cost=Decimal("100.00"), criticality_level="HIGH")
    s = Supplier(supplier_code="TEST_S", supplier_name="Test Supplier", supplier_country="US", reliability_score=0.9, supplier_capacity_units=1000, average_lead_time_days=5.0, risk_level="LOW")
    db_session.add_all([p, s])
    db_session.commit()

    ev = SupplyEvent(
        event_id="TEST_EVT_DEC", product_id=p.id, supplier_id=s.id, event_date=p.created_at.date(),
        demand_units=500, supplier_capacity_units=1000, planned_lead_time_days=5.0, unit_cost=Decimal("100.00")
    )
    db_session.add(ev)
    db_session.commit()

    pred = Prediction(
        event_id=ev.id, model_name="test_model", model_version="1.0", prediction_type="DELAY",
        delay_probability=0.8, predicted_delay_days=10.0, risk_score=0.75, risk_level="HIGH"
    )
    db_session.add(pred)
    db_session.commit()

    run = OptimizationRun(
        event_id=ev.id, prediction_id=pred.id, optimization_run_id="RUN_DEC_TEST",
        objective_type="MIN_COST", optimization_status="COMPLETED", solver_status="OPTIMAL"
    )
    db_session.add(run)
    db_session.commit()

    rec1 = Recommendation(
        optimization_run_id=run.id, recommendation_code="ALT_A", strategy_type="AIR_FREIGHT",
        strategy_description="Air freight", estimated_cost=Decimal("15000.00"), estimated_delay_days=2.0,
        estimated_risk=0.08, service_level=1.0, rank=1, is_recommended=True
    )
    rec2 = Recommendation(
        optimization_run_id=run.id, recommendation_code="ALT_B", strategy_type="SECONDARY_SUPPLIER",
        strategy_description="Secondary", estimated_cost=Decimal("25000.00"), estimated_delay_days=7.0,
        estimated_risk=0.15, service_level=1.0, rank=2, is_recommended=False
    )
    db_session.add_all([rec1, rec2])
    db_session.commit()

    return run, rec1, rec2


def test_create_approved_decision(db_session, test_setup):
    run, rec1, rec2 = test_setup
    svc = DecisionService()

    req = DecisionCreateRequest(
        optimization_run_id=run.id,
        recommendation_id=rec1.id,
        decision_maker="Supply Chain Director",
        decision_status="APPROVED",
    )
    dec = svc.record_decision(db_session, req)
    assert dec.id is not None
    assert dec.decision_status == "APPROVED"
    assert dec.execution_status == "NOT_STARTED"


def test_create_override_without_reason_fails(db_session, test_setup):
    run, rec1, rec2 = test_setup
    svc = DecisionService()

    req = DecisionCreateRequest(
        optimization_run_id=run.id,
        recommendation_id=rec2.id,  # Non-recommended
        decision_maker="Manager",
        decision_status="OVERRIDDEN",
        decision_reason="",  # Missing required reason
    )
    with pytest.raises(DecisionValidationError):
        svc.record_decision(db_session, req)


def test_execution_status_lifecycle(db_session, test_setup):
    run, rec1, rec2 = test_setup
    svc = DecisionService()

    req = DecisionCreateRequest(
        optimization_run_id=run.id,
        recommendation_id=rec1.id,
        decision_maker="Logistics Manager",
        decision_status="APPROVED",
    )
    dec = svc.record_decision(db_session, req)

    # Step 1: Start execution
    up1 = DecisionUpdateRequest(execution_status="IN_PROGRESS")
    dec = svc.update_decision(db_session, dec.id, up1)
    assert dec.execution_status == "IN_PROGRESS"

    # Step 2: Complete execution
    up2 = DecisionUpdateRequest(execution_status="COMPLETED")
    dec = svc.update_decision(db_session, dec.id, up2)
    assert dec.execution_status == "COMPLETED"
    assert dec.executed_at is not None

