"""Test all CRUD operations and closed-loop traceability."""

from datetime import date, datetime, timezone
from decimal import Decimal

from src.database import crud
from src.database.schemas import (
    ProductCreate,
    ProductUpdate,
    SupplierCreate,
    SupplierUpdate,
    InventoryCreate,
    InventoryUpdate,
    SupplyEventCreate,
    PredictionCreate,
    OptimizationRunCreate,
    RecommendationCreate,
    DecisionCreate,
    DecisionStatusUpdate,
    OutcomeCreate,
    FeedbackCreate,
)


def test_product_crud(db_session):
    # Create
    p_in = ProductCreate(
        product_code="CRUD_P1",
        product_name="CRUD Test Product",
        product_category="Test",
        unit_cost=Decimal("150.00"),
        criticality_level="CRITICAL",
    )
    p = crud.create_product(db_session, p_in)
    assert p.id is not None
    assert p.product_code == "CRUD_P1"

    # Read
    p_read = crud.get_product(db_session, p.id)
    assert p_read is not None
    assert p_read.product_name == "CRUD Test Product"

    p_code = crud.get_product_by_code(db_session, "CRUD_P1")
    assert p_code.id == p.id

    all_prods = crud.get_products(db_session)
    assert len(all_prods) >= 1

    # Update
    p_up = ProductUpdate(unit_cost=Decimal("165.00"), product_name="Updated CRUD Product")
    p_updated = crud.update_product(db_session, p.id, p_up)
    assert p_updated.unit_cost == Decimal("165.00")
    assert p_updated.product_name == "Updated CRUD Product"

    # Delete
    del_ok = crud.delete_product(db_session, p.id)
    assert del_ok is True
    assert crud.get_product(db_session, p.id) is None


def test_supplier_crud(db_session):
    # Create
    s_in = SupplierCreate(
        supplier_code="CRUD_S1",
        supplier_name="CRUD Supplier",
        supplier_country="Taiwan",
        reliability_score=0.92,
        supplier_capacity_units=15000,
        average_lead_time_days=12.0,
        risk_level="MEDIUM",
    )
    s = crud.create_supplier(db_session, s_in)
    assert s.id is not None

    # Read
    s_read = crud.get_supplier(db_session, s.id)
    assert s_read.supplier_code == "CRUD_S1"

    s_code = crud.get_supplier_by_code(db_session, "CRUD_S1")
    assert s_code.id == s.id

    # Update
    s_up = SupplierUpdate(reliability_score=0.95, risk_level="LOW")
    s_updated = crud.update_supplier(db_session, s.id, s_up)
    assert s_updated.reliability_score == 0.95
    assert s_updated.risk_level == "LOW"

    # Delete
    assert crud.delete_supplier(db_session, s.id) is True
    assert crud.get_supplier(db_session, s.id) is None


def test_closed_loop_crud_and_traceability(db_session):
    # 1. Product & Supplier
    p = crud.create_product(
        db_session,
        ProductCreate(
            product_code="CL_P1",
            product_name="Microchip X1",
            product_category="Electronics",
            unit_cost=Decimal("125.50"),
            criticality_level="CRITICAL",
        ),
    )
    s = crud.create_supplier(
        db_session,
        SupplierCreate(
            supplier_code="CL_S1",
            supplier_name="Pacific Silicon",
            supplier_country="Taiwan",
            reliability_score=0.88,
            supplier_capacity_units=15000,
            average_lead_time_days=14.0,
            risk_level="HIGH",
        ),
    )

    # 2. Inventory
    inv = crud.create_inventory(
        db_session,
        InventoryCreate(
            product_id=p.id,
            warehouse_code="WH-NORTH-01",
            inventory_units=2500,
            reserved_units=500,
            available_units=2000,
            reorder_point=1200,
            safety_stock_units=600,
            inventory_date=date(2026, 10, 1),
        ),
    )
    assert inv.id is not None

    # 3. Supply Event
    event = crud.create_supply_event(
        db_session,
        SupplyEventCreate(
            event_id="EVT_CRUD_001",
            product_id=p.id,
            supplier_id=s.id,
            event_date=date(2026, 9, 15),
            demand_units=3000,
            supplier_capacity_units=15000,
            planned_lead_time_days=14.0,
            transport_cost=Decimal("4500.00"),
            unit_cost=Decimal("125.50"),
        ),
    )
    assert event.id is not None

    # 4. Prediction
    pred = crud.create_prediction(
        db_session,
        PredictionCreate(
            event_id=event.id,
            model_name="XGBoost Delay Classifier",
            model_version="v1.0",
            prediction_type="DELAY",
            delay_probability=0.87,
            predicted_delay_days=14.0,
            risk_score=0.82,
            risk_level="HIGH",
            prediction_timestamp=datetime.now(timezone.utc),
        ),
    )
    assert pred.id is not None

    # 5. Optimization Run
    opt = crud.create_optimization_run(
        db_session,
        OptimizationRunCreate(
            event_id=event.id,
            prediction_id=pred.id,
            optimization_run_id="OPT_CRUD_001",
            objective_type="BALANCED",
            optimization_status="COMPLETED",
            total_cost_baseline=Decimal("381000.00"),
            recommended_cost=Decimal("396000.00"),
            service_level_target=0.98,
            solver_name="PULP_CBC",
            solver_status="OPTIMAL",
        ),
    )
    assert opt.id is not None

    # 6. Recommendations
    rec_a = crud.create_recommendation(
        db_session,
        RecommendationCreate(
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
        ),
    )
    rec_b = crud.create_recommendation(
        db_session,
        RecommendationCreate(
            optimization_run_id=opt.id,
            recommendation_code="ALT_B",
            strategy_type="SECONDARY_SUPPLIER",
            strategy_description="Secondary supplier allocation",
            estimated_cost=Decimal("22000.00"),
            estimated_delay_days=4.0,
            estimated_risk=0.15,
            service_level=0.95,
            cost_difference=Decimal("22000.00"),
            risk_difference=-0.67,
            rank=2,
            is_recommended=False,
        ),
    )
    recs = crud.get_recommendations(db_session, opt.id)
    assert len(recs) == 2

    # 7. Decision
    dec = crud.create_decision(
        db_session,
        DecisionCreate(
            decision_id="DEC_CRUD_001",
            optimization_run_id=opt.id,
            recommendation_id=rec_a.id,
            decision_maker="VP Operations",
            decision_status="APPROVED",
            decision_reason="Air freight avoids assembly line shutdown.",
            execution_status="COMPLETED",
        ),
    )
    assert dec.id is not None

    # 8. Outcome
    out = crud.create_outcome(
        db_session,
        OutcomeCreate(
            decision_id=dec.id,
            actual_cost=Decimal("18000.00"),
            actual_delay_days=2.0,
            actual_service_level=0.99,
            actual_risk=0.05,
            actual_quantity_delivered=3000,
            cost_variance=Decimal("3000.00"),
            delay_variance=0.0,
            service_level_variance=0.0,
            outcome_status="SUCCESS",
            evaluation_date=date.today(),
            notes="Fuel surcharge created $3k cost variance.",
        ),
    )
    assert out.id is not None

    # 9. Feedback
    fb = crud.create_feedback(
        db_session,
        FeedbackCreate(
            outcome_id=out.id,
            optimization_run_id=opt.id,
            feedback_type="COST_VARIANCE",
            cost_error=Decimal("3000.00"),
            learning_signal="Adjust air freight fuel surcharge calibration parameter.",
        ),
    )
    assert fb.id is not None

    # 10. Traceability Trail Verification
    trail = crud.get_traceability_trail(db_session, dec.id)
    assert trail is not None
    assert trail["product"].product_code == "CL_P1"
    assert trail["supplier"].supplier_code == "CL_S1"
    assert trail["prediction"].delay_probability == 0.87
    assert trail["optimization_run"].optimization_run_id == "OPT_CRUD_001"
    assert len(trail["recommendations"]) == 2
    assert trail["selected_recommendation"].recommendation_code == "ALT_A"
    assert trail["decision"].decision_id == "DEC_CRUD_001"
    assert trail["outcome"].actual_cost == Decimal("18000.00")
    assert trail["feedback"][0].cost_error == Decimal("3000.00")
