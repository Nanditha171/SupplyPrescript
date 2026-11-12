"""End-to-End closed-loop workflow integration tests (Day 18)."""

from datetime import date
from decimal import Decimal
import pytest

from src.database.models import Product, Supplier, SupplyEvent, Inventory
from src.workflows.orchestration import WorkflowOrchestrator
from src.workflows.workflow_schemas import WorkflowStatus
from src.database.crud import get_traceability_trail


@pytest.fixture
def workflow_db(db_session):
    p = Product(product_code="P001", product_name="Microchip X1 High-Density MCU", product_category="Electronics", unit_cost=Decimal("125.50"), criticality_level="CRITICAL")
    s = Supplier(supplier_code="SUP-001", supplier_name="Pacific Silicon Ltd", supplier_country="Taiwan", reliability_score=0.88, supplier_capacity_units=15000, average_lead_time_days=14.0, risk_level="HIGH")
    db_session.add_all([p, s])
    db_session.commit()

    inv = Inventory(
        product_id=p.id, warehouse_code="WH-NORTH", inventory_units=50, reserved_units=10,
        available_units=40, reorder_point=200, safety_stock_units=100, inventory_date=date.today()
    )
    db_session.add(inv)
    db_session.commit()

    ev = SupplyEvent(
        event_id="EVT-001", product_id=p.id, supplier_id=s.id, event_date=date.today(),
        demand_units=1000, supplier_capacity_units=15000, planned_lead_time_days=14.0,
        unit_cost=Decimal("125.50"), transport_cost=Decimal("2000.00"), air_freight_cost=Decimal("15000.00"),
        service_risk_score=0.85
    )
    db_session.add(ev)
    db_session.commit()

    return ev


def test_complete_end_to_end_workflow(db_session, workflow_db):
    """Execute complete 8-stage closed-loop workflow:
    Predict -> Assess Risk -> Optimize -> Recommend -> Decide -> Execute -> Measure -> Learn.
    """
    orchestrator = WorkflowOrchestrator()

    res = orchestrator.execute_closed_loop_workflow(
        db=db_session,
        event_id="EVT-001",
        decision_maker="Global Operations VP",
        selected_strategy_code="ALT_A",  # Air freight
        actual_cost=18000.0,  # Demonstrating $18,000 actual vs $15,000 estimated
        actual_delay_days=2.0,
        actual_service_level=1.0,
    )

    assert res.overall_status == WorkflowStatus.COMPLETED
    assert len(res.steps) == 5
    assert res.recommended_strategy == "AIR_FREIGHT"
    assert res.decision_id is not None
    assert res.execution_status == "COMPLETED"
    assert res.outcome_summary is not None
    assert res.outcome_summary["cost_variance"] == 3000.0
    assert res.feedback_summary is not None
    assert "AIR_FREIGHT" in res.feedback_summary["learning_signal"]

    # Verify complete end-to-end audit trail in database
    from src.database.crud import get_decision_by_code
    dec_record = get_decision_by_code(db_session, res.decision_id)
    assert dec_record is not None

    trail = get_traceability_trail(db_session, dec_record.id)
    assert trail is not None
    assert trail["supply_event"].event_id == "EVT-001"
    assert trail["prediction"] is not None
    assert trail["optimization_run"] is not None
    assert len(trail["recommendations"]) >= 1
    assert trail["selected_recommendation"].strategy_type == "AIR_FREIGHT"
    assert trail["decision"].execution_status == "COMPLETED"
    assert trail["outcome"] is not None
    assert float(trail["outcome"].cost_variance) == 3000.0
    assert len(trail["feedback"]) >= 1


def test_workflow_nonexistent_event(db_session):
    orchestrator = WorkflowOrchestrator()
    res = orchestrator.execute_closed_loop_workflow(db=db_session, event_id="DOES_NOT_EXIST")
    assert res.overall_status == WorkflowStatus.FAILED
    assert "not found" in res.error_detail.lower()

