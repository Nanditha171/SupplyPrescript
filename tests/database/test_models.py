"""Test all 10 domain models creation, column mappings, and representations."""

from datetime import date, datetime, timezone
from decimal import Decimal
from src.database.models import (
    Product,
    Supplier,
    Inventory,
    SupplyEvent,
    Prediction,
    OptimizationRun,
    Recommendation,
    Decision,
    Outcome,
    Feedback,
)


def test_product_model_creation(db_session):
    prod = Product(
        product_code="TEST_P001",
        product_name="Microchip X1",
        product_category="Electronics",
        unit_cost=Decimal("125.50"),
        criticality_level="CRITICAL",
    )
    db_session.add(prod)
    db_session.commit()
    db_session.refresh(prod)

    assert prod.id is not None
    assert prod.product_code == "TEST_P001"
    assert "Microchip X1" in repr(prod)


def test_supplier_model_creation(db_session):
    sup = Supplier(
        supplier_code="TEST_SUP001",
        supplier_name="Pacific Silicon",
        supplier_country="Taiwan",
        reliability_score=0.88,
        supplier_capacity_units=15000,
        average_lead_time_days=14.0,
        risk_level="HIGH",
    )
    db_session.add(sup)
    db_session.commit()
    db_session.refresh(sup)

    assert sup.id is not None
    assert sup.supplier_code == "TEST_SUP001"
    assert "Pacific Silicon" in repr(sup)


def test_inventory_model_creation(db_session):
    prod = Product(
        product_code="TEST_P002",
        product_name="Battery Module",
        product_category="Energy",
        unit_cost=Decimal("85.00"),
        criticality_level="CRITICAL",
    )
    db_session.add(prod)
    db_session.commit()

    inv = Inventory(
        product_id=prod.id,
        warehouse_code="WH-NORTH-01",
        inventory_units=2500,
        reserved_units=500,
        available_units=2000,
        reorder_point=1200,
        safety_stock_units=600,
        inventory_date=date(2026, 10, 1),
    )
    db_session.add(inv)
    db_session.commit()
    db_session.refresh(inv)

    assert inv.id is not None
    assert inv.available_units == 2000
    assert "WH-NORTH-01" in repr(inv)


def test_full_operational_entities_flow(db_session):
    """Test creation of SupplyEvent, Prediction, OptimizationRun, Recommendation, Decision, Outcome, Feedback."""
    # 1. Product & Supplier
    prod = Product(
        product_code="P_DEMO",
        product_name="Demo Part",
        product_category="Test",
        unit_cost=Decimal("100.00"),
        criticality_level="HIGH",
    )
    sup = Supplier(
        supplier_code="S_DEMO",
        supplier_name="Demo Supplier",
        supplier_country="Germany",
        reliability_score=0.95,
        supplier_capacity_units=10000,
        average_lead_time_days=10.0,
        risk_level="LOW",
    )
    db_session.add_all([prod, sup])
    db_session.commit()

    # 2. Supply Event
    event = SupplyEvent(
        event_id="EVT_DEMO_01",
        product_id=prod.id,
        supplier_id=sup.id,
        event_date=date(2026, 9, 15),
        demand_units=3000,
        supplier_capacity_units=15000,
        planned_lead_time_days=14.0,
        actual_lead_time_days=16.0,
        delay_occurred=True,
        delay_days=2.0,
        transport_cost=Decimal("4500.00"),
        unit_cost=Decimal("100.00"),
        secondary_supplier_premium=Decimal("10000.00"),
        air_freight_cost=Decimal("15000.00"),
        service_risk_score=0.82,
    )
    db_session.add(event)
    db_session.commit()

    # 3. Prediction
    pred = Prediction(
        event_id=event.id,
        model_name="XGBoost Delay Classifier",
        model_version="v1.0",
        prediction_type="DELAY",
        delay_probability=0.87,
        predicted_delay_days=14.0,
        risk_score=0.82,
        risk_level="HIGH",
        prediction_timestamp=datetime.now(timezone.utc),
    )
    db_session.add(pred)
    db_session.commit()

    # 4. Optimization Run
    opt = OptimizationRun(
        event_id=event.id,
        prediction_id=pred.id,
        optimization_run_id="OPT_DEMO_01",
        objective_type="BALANCED",
        optimization_status="COMPLETED",
        total_cost_baseline=Decimal("381000.00"),
        recommended_cost=Decimal("396000.00"),
        service_level_target=0.98,
        risk_weight=0.40,
        cost_weight=0.40,
        time_weight=0.20,
        solver_name="PULP_CBC",
        solver_status="OPTIMAL",
        execution_time_ms=142.3,
    )
    db_session.add(opt)
    db_session.commit()

    # 5. Recommendation
    rec = Recommendation(
        optimization_run_id=opt.id,
        recommendation_code="ALT_A",
        strategy_type="AIR_FREIGHT",
        strategy_description="Expedite via Air Freight",
        estimated_cost=Decimal("15000.00"),
        estimated_delay_days=2.0,
        estimated_risk=0.08,
        service_level=0.99,
        cost_difference=Decimal("15000.00"),
        risk_difference=-0.74,
        rank=1,
        is_recommended=True,
    )
    db_session.add(rec)
    db_session.commit()

    # 6. Decision
    decision = Decision(
        decision_id="DEC_DEMO_01",
        optimization_run_id=opt.id,
        recommendation_id=rec.id,
        decision_maker="Manager John",
        decision_status="APPROVED",
        decision_reason="Approve air freight for critical assembly",
        execution_status="COMPLETED",
        executed_at=datetime.now(timezone.utc),
    )
    db_session.add(decision)
    db_session.commit()

    # 7. Outcome
    outcome = Outcome(
        decision_id=decision.id,
        actual_cost=Decimal("18000.00"),
        actual_delay_days=2.0,
        actual_service_level=0.99,
        actual_risk=0.05,
        actual_quantity_delivered=3000,
        cost_variance=Decimal("3000.00"),
        delay_variance=0.0,
        service_level_variance=0.0,
        outcome_status="SUCCESS",
        evaluation_date=date(2026, 9, 18),
        notes="Delivered with slight fuel surcharge cost variance.",
    )
    db_session.add(outcome)
    db_session.commit()

    # 8. Feedback
    fb = Feedback(
        outcome_id=outcome.id,
        optimization_run_id=opt.id,
        feedback_type="COST_VARIANCE",
        cost_error=Decimal("3000.00"),
        delay_error=0.0,
        risk_error=-0.03,
        prediction_error=0.0,
        weight_adjustment=0.05,
        learning_signal="Add fuel surcharge buffer to air freight cost model.",
    )
    db_session.add(fb)
    db_session.commit()

    # Assertions
    assert event.id is not None
    assert pred.id is not None
    assert opt.id is not None
    assert rec.id is not None
    assert decision.id is not None
    assert outcome.id is not None
    assert fb.id is not None
    assert fb.cost_error == Decimal("3000.00")
