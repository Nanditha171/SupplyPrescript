"""
Unit and Integration Tests for Closed-Loop Evaluation & Feedback
"""

import pytest
import os
import sys

# Ensure root directory is on PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from evaluation.evaluator import evaluate_decision
from evaluation.feedback import record_closed_loop_event, get_feedback_summary, load_history


def test_evaluate_decision_metrics():
    prediction = {"predicted_delay_days": 8.0, "input_summary": {"lead_time_days": 10.0}}
    decision = {"id": "OPTION_A", "name": "Air Freight", "cost": 12000.0, "delivery_days": 3.0}
    actual_outcome = {"actual_cost": 13500.0, "actual_delivery_days": 3.5, "notes": "Slight customs delay"}

    result = evaluate_decision(prediction, decision, actual_outcome)
    assert result["predicted_cost"] == 12000.0
    assert result["actual_cost"] == 13500.0
    assert result["cost_variance"] == 1500.0
    assert result["delivery_variance_days"] == 0.5
    assert result["actual_days_saved"] > 0
    assert result["penalties_avoided"] > 0
    assert result["mitigation_roi_pct"] is not None


def test_record_closed_loop_event_persistence():
    shipment = {"shipment_id": "TEST-SHP-001", "supplier_name": "Test Supplier", "shipping_cost": 5000.0}
    prediction = {"is_delayed": 1, "predicted_delay_days": 6.0, "delay_probability": 0.85, "risk_level": "HIGH"}
    decision = {"id": "OPTION_B", "name": "Secondary Supplier", "cost": 7500.0, "delivery_days": 5.0}
    actual = {"actual_cost": 7400.0, "actual_delivery_days": 5.0, "notes": "On-time arrival"}
    eval_res = evaluate_decision(prediction, decision, actual)

    record = record_closed_loop_event(shipment, prediction, decision, actual, eval_res)
    assert record["event_id"].startswith("EVT-")
    assert record["shipment_id"] == "TEST-SHP-001"

    summary = get_feedback_summary()
    assert summary["total_decisions"] >= 1
