"""
Unit and Integration Tests for API Interfaces & Integration Contract
"""

import pytest
import os
import sys

# Ensure root directory is on PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from api.interfaces import (
    predict_delay,
    optimize_actions,
    validate_recommendation,
    evaluate_decision,
    get_model_metrics,
)


def test_api_interfaces_contract():
    shipment_payload = {
        "supplier_name": "GlobalTech Micro",
        "product_category": "Semiconductors & Electronics",
        "shipping_mode": "Standard Ground",
        "lead_time_days": 10.0,
        "shipping_cost": 5000.0,
        "inventory_level": 300,
        "order_quantity": 200,
        "supplier_delay_rate": 0.35,
        "route_distance_km": 4000.0,
        "weather_risk_index": 0.60,
        "congestion_index": 0.50,
    }

    # Step 1: Predict
    pred = predict_delay(shipment_payload)
    assert "delay_probability" in pred
    assert "predicted_delay_days" in pred
    assert "risk_level" in pred

    # Step 2: Optimize
    opt = optimize_actions(shipment_payload, pred, {"max_budget": 20000.0, "max_delivery_days": 12.0})
    assert "options" in opt
    assert len(opt["options"]) == 3
    assert "recommended_option" in opt

    # Step 3: Validate
    chosen = opt["options"][0]
    val = validate_recommendation(chosen, {"max_budget": 20000.0, "max_delivery_days": 12.0})
    assert "status" in val
    assert "is_feasible" in val

    # Step 4: Evaluate
    actual = {"actual_cost": chosen["cost"] + 500.0, "actual_delivery_days": chosen["delivery_days"] + 0.5}
    ev = evaluate_decision(pred, chosen, actual)
    assert "cost_variance" in ev
    assert "delivery_variance_days" in ev
    assert "mitigation_roi_pct" in ev


def test_fastapi_endpoints():
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "OPERATIONAL"

    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    assert "classification" in metrics_resp.json()["data"]
