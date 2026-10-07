"""Test database ORM relationships and foreign key cascades."""

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


def test_relationships_and_cascades(db_session):
    # Setup hierarchy
    prod = Product(
        product_code="REL_P1",
        product_name="Relational Test Part",
        product_category="Hardware",
        unit_cost=Decimal("75.00"),
        criticality_level="HIGH",
    )
    sup = Supplier(
        supplier_code="REL_S1",
        supplier_name="Relational Supplier",
        supplier_country="Japan",
        reliability_score=0.92,
        supplier_capacity_units=5000,
        average_lead_time_days=8.0,
        risk_level="MEDIUM",
    )
    db_session.add_all([prod, sup])
    db_session.commit()

    inv = Inventory(
        product_id=prod.id,
        warehouse_code="WH_REL",
        inventory_units=500,
        reserved_units=100,
        available_units=400,
        reorder_point=200,
        safety_stock_units=100,
        inventory_date=date.today(),
    )
    event = SupplyEvent(
        event_id="EVT_REL_01",
        product_id=prod.id,
        supplier_id=sup.id,
        event_date=date.today(),
        demand_units=500,
        supplier_capacity_units=2000,
        planned_lead_time_days=8.0,
        unit_cost=Decimal("75.00"),
    )
    db_session.add_all([inv, event])
    db_session.commit()

    pred = Prediction(
        event_id=event.id,
        model_name="RelModel",
        model_version="v1",
        prediction_type="DELAY",
        delay_probability=0.75,
        predicted_delay_days=6.0,
        risk_score=0.7,
        risk_level="HIGH",
    )
    db_session.add(pred)
    db_session.commit()

    opt = OptimizationRun(
        event_id=event.id,
        prediction_id=pred.id,
        optimization_run_id="OPT_REL_01",
        objective_type="BALANCED",
        optimization_status="COMPLETED",
        total_cost_baseline=Decimal("50000.00"),
        solver_name="PULP_CBC",
        solver_status="OPTIMAL",
    )
    db_session.add(opt)
    db_session.commit()

    rec = Recommendation(
        optimization_run_id=opt.id,
        recommendation_code="ALT_A",
        strategy_type="AIR_FREIGHT",
        strategy_description="Air freight option",
        estimated_cost=Decimal("5000.00"),
        estimated_delay_days=1.0,
        estimated_risk=0.1,
        service_level=0.98,
        rank=1,
        is_recommended=True,
    )
    db_session.add(rec)
    db_session.commit()

    decision = Decision(
        decision_id="DEC_REL_01",
        optimization_run_id=opt.id,
        recommendation_id=rec.id,
        decision_maker="Manager Jane",
        decision_status="APPROVED",
    )
    db_session.add(decision)
    db_session.commit()

    outcome = Outcome(
        decision_id=decision.id,
        actual_cost=Decimal("5500.00"),
        actual_delay_days=1.0,
        actual_service_level=0.98,
        actual_risk=0.1,
        actual_quantity_delivered=500,
        cost_variance=Decimal("500.00"),
        delay_variance=0.0,
        service_level_variance=0.0,
        outcome_status="SUCCESS",
        evaluation_date=date.today(),
    )
    db_session.add(outcome)
    db_session.commit()

    fb = Feedback(
        outcome_id=outcome.id,
        optimization_run_id=opt.id,
        feedback_type="COST_VARIANCE",
        cost_error=Decimal("500.00"),
    )
    db_session.add(fb)
    db_session.commit()

    # Forward navigation
    assert len(prod.inventory) == 1
    assert len(prod.supply_events) == 1
    assert len(event.predictions) == 1
    assert len(event.optimization_runs) == 1
    assert len(opt.recommendations) == 1
    assert len(opt.decisions) == 1
    assert decision.outcome is not None
    assert len(outcome.feedback_records) == 1

    # Backward navigation
    assert fb.outcome.decision.recommendation.strategy_type == "AIR_FREIGHT"
    assert fb.outcome.decision.optimization_run.event.product.product_code == "REL_P1"

    # Test cascade delete from Product
    db_session.delete(prod)
    db_session.commit()

    # All related records should be cascade deleted
    assert db_session.get(Inventory, inv.id) is None
    assert db_session.get(SupplyEvent, event.id) is None
    assert db_session.get(Prediction, pred.id) is None
    assert db_session.get(OptimizationRun, opt.id) is None
    assert db_session.get(Recommendation, rec.id) is None
    assert db_session.get(Decision, decision.id) is None
    assert db_session.get(Outcome, outcome.id) is None
    assert db_session.get(Feedback, fb.id) is None
