"""Tests for FastAPI endpoints across the closed-loop workflow (Day 12)."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.database.connection import get_db


@pytest.fixture
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_api_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["database"]["status"] == "connected"


def test_api_products_crud(client):
    # Create product
    payload = {
        "product_code": "API_P1",
        "product_name": "API Test Product",
        "product_category": "Semiconductors",
        "unit_cost": 120.00,
        "criticality_level": "CRITICAL",
    }
    create_res = client.post("/api/products", json=payload)
    assert create_res.status_code == 201
    prod = create_res.json()
    assert prod["product_code"] == "API_P1"

    # List products
    list_res = client.get("/api/products")
    assert list_res.status_code == 200
    assert len(list_res.json()) >= 1

    # Get single product
    get_res = client.get(f"/api/products/{prod['id']}")
    assert get_res.status_code == 200
    assert get_res.json()["product_name"] == "API Test Product"


def test_api_suppliers_crud(client):
    payload = {
        "supplier_code": "API_SUP1",
        "supplier_name": "API Test Supplier",
        "supplier_country": "Taiwan",
        "reliability_score": 0.95,
        "supplier_capacity_units": 10000,
        "average_lead_time_days": 14.0,
        "risk_level": "LOW",
    }
    create_res = client.post("/api/suppliers", json=payload)
    assert create_res.status_code == 201
    sup = create_res.json()
    assert sup["supplier_code"] == "API_SUP1"


def test_api_end_to_end_decision_cycle(client):
    """Test full cycle: Product -> Supplier -> Shipment -> Prediction -> Risk -> Optimization -> Recs -> Decision -> Execution -> Outcome -> Feedback."""
    # 1. Product & Supplier
    p_res = client.post("/api/products", json={
        "product_code": "P_E2E",
        "product_name": "E2E Microchip",
        "product_category": "Electronics",
        "unit_cost": 125.50,
        "criticality_level": "CRITICAL",
    }).json()

    s_res = client.post("/api/suppliers", json={
        "supplier_code": "S_E2E",
        "supplier_name": "E2E Supplier",
        "supplier_country": "Taiwan",
        "reliability_score": 0.88,
        "supplier_capacity_units": 15000,
        "average_lead_time_days": 14.0,
        "risk_level": "HIGH",
    }).json()

    # 2. Inventory
    inv_res = client.post("/api/inventory", json={
        "product_id": p_res["id"],
        "warehouse_code": "WH-E2E",
        "inventory_units": 200,
        "reserved_units": 50,
        "available_units": 150,
        "reorder_point": 500,
        "safety_stock_units": 200,
        "inventory_date": "2026-10-15",
    })
    assert inv_res.status_code == 201

    # 3. Supply Event
    ship_res = client.post("/api/shipments", json={
        "event_id": "EVT-E2E-001",
        "product_id": p_res["id"],
        "supplier_id": s_res["id"],
        "event_date": "2026-10-15",
        "demand_units": 1000,
        "supplier_capacity_units": 15000,
        "planned_lead_time_days": 14.0,
        "unit_cost": 125.50,
        "transport_cost": 2000.0,
        "secondary_supplier_premium": 12.55,
        "air_freight_cost": 15000.0,
        "service_risk_score": 0.85,
    })
    assert ship_res.status_code == 201

    # 4. Predict
    pred_res = client.post("/api/predictions/EVT-E2E-001")
    assert pred_res.status_code == 201
    pred = pred_res.json()
    assert pred["delay_probability"] >= 0.0

    # 5. Risk Assessment
    risk_res = client.post("/api/risks/EVT-E2E-001")
    assert risk_res.status_code == 201
    risk = risk_res.json()
    assert risk["composite_risk_score"] >= 0.0

    # 6. Optimization Run
    opt_res = client.post("/api/optimization/EVT-E2E-001?objective_type=MIN_COST")
    assert opt_res.status_code == 201
    opt_run = opt_res.json()
    assert opt_run["solver_status"] in ("OPTIMAL", "FEASIBLE")

    # 7. Recommendations
    rec_res = client.get(f"/api/recommendations/{opt_run['optimization_run_id']}")
    assert rec_res.status_code == 200
    rec_data = rec_res.json()
    assert len(rec_data["recommendations"]) >= 1
    top_rec = rec_data["recommendations"][0]

    # Find the recommendation DB id by querying the database or using top_rec
    # Query run details
    run_detail = client.get(f"/api/optimization/{opt_run['id']}").json()

    # Query all recommendations to get the DB id
    from src.database.models import Recommendation
    # In testclient we can post decision directly with top_rec info
    # 8. Record Decision
    from src.database.crud import get_recommendations
    dec_payload = {
        "optimization_run_id": opt_run["id"],
        "recommendation_id": 1,  # In SQLite, first rec id is 1
        "decision_maker": "Chief Supply Chain Officer",
        "decision_status": "APPROVED",
        "decision_reason": "Approved air charter to protect critical product launch.",
        "execution_status": "NOT_STARTED",
    }
    dec_res = client.post("/api/decisions", json=dec_payload)
    assert dec_res.status_code == 201
    dec = dec_res.json()

    # 9. Update Execution Status: NOT_STARTED -> IN_PROGRESS -> COMPLETED
    in_prog_res = client.put(f"/api/decisions/{dec['id']}", json={"execution_status": "IN_PROGRESS"})
    assert in_prog_res.status_code == 200
    assert in_prog_res.json()["execution_status"] == "IN_PROGRESS"

    exec_res = client.put(f"/api/decisions/{dec['id']}", json={"execution_status": "COMPLETED"})
    assert exec_res.status_code == 200
    assert exec_res.json()["execution_status"] == "COMPLETED"

    # 10. Record Outcome ($18,000 actual vs $15,000 estimated)
    out_payload = {
        "decision_id": dec["id"],
        "actual_cost": 18000.0,
        "actual_delay_days": 2.0,
        "actual_service_level": 1.0,
        "actual_risk": 0.08,
        "actual_quantity_delivered": 1000,
        "notes": "Delivered with $3k surcharge.",
    }
    out_res = client.post("/api/outcomes", json=out_payload)
    assert out_res.status_code == 201
    outcome = out_res.json()
    assert float(outcome["cost_variance"]) == 3000.0

    # 11. Trigger Closed-Loop Feedback
    fb_res = client.post(f"/api/feedback/{outcome['id']}")
    assert fb_res.status_code == 201
    feedback = fb_res.json()
    assert float(feedback["cost_error"]) == 3000.0
