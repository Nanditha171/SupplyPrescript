"""
Unit and Integration Tests for Prescriptive Optimization & Constraints Audit
"""

import pytest
import os
import sys

# Ensure root directory is on PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from optimization.constraints import OptimizationConstraints
from optimization.validator import audit_option, validate_all_options
from optimization.optimizer import generate_candidate_options, optimize_actions


@pytest.fixture
def sample_shipment():
    return {
        "shipping_cost": 6000.0,
        "lead_time_days": 14.0,
        "order_quantity": 300,
        "route_distance_km": 8000.0,
    }


@pytest.fixture
def sample_prediction():
    return {
        "is_delayed": 1,
        "predicted_delay_days": 9.5,
        "delay_probability": 0.88,
        "risk_level": "CRITICAL",
    }


def test_candidate_options_generation(sample_shipment, sample_prediction):
    options = generate_candidate_options(sample_shipment, sample_prediction)
    assert len(options) == 3, "Optimizer must generate exactly 3 strategic alternatives."
    ids = [opt["id"] for opt in options]
    assert "OPTION_A" in ids
    assert "OPTION_B" in ids
    assert "OPTION_C" in ids

    # Check that costs and delivery times are physically realistic and distinct
    costs = [opt["cost"] for opt in options]
    days = [opt["delivery_days"] for opt in options]
    assert costs[0] > costs[1] > costs[2], "Option A should cost more than B, which costs more than C."
    assert days[0] < days[1] < days[2], "Option A should be faster than B, which is faster than C."


def test_audit_option_feasibility():
    constraints = OptimizationConstraints(max_budget=20000.0, max_delivery_days=8.0)
    
    feasible_opt = {"cost": 15000.0, "delivery_days": 4.0, "capacity": 500}
    audit_res = audit_option(feasible_opt, constraints, order_quantity=200)
    assert audit_res["is_feasible"] is True
    assert audit_res["status"] == "FEASIBLE"
    assert len(audit_res["violations"]) == 0

    # Over-budget violation
    overbudget_opt = {"cost": 25000.0, "delivery_days": 3.0, "capacity": 500}
    audit_overbudget = audit_option(overbudget_opt, constraints, order_quantity=200)
    assert audit_overbudget["is_feasible"] is False
    assert audit_overbudget["status"] == "INFEASIBLE"
    assert "Hard Budget" in audit_overbudget["audit_summary"]

    # Over-deadline violation
    overtime_opt = {"cost": 8000.0, "delivery_days": 12.0, "capacity": 500}
    audit_overtime = audit_option(overtime_opt, constraints, order_quantity=200)
    assert audit_overtime["is_feasible"] is False
    assert audit_overtime["status"] == "INFEASIBLE"
    assert "Delivery SLA" in audit_overtime["audit_summary"]


def test_optimize_actions_selects_optimal(sample_shipment, sample_prediction):
    constraints = OptimizationConstraints(max_budget=30000.0, max_delivery_days=15.0)
    result = optimize_actions(sample_shipment, sample_prediction, constraints)

    assert result["is_optimal"] is True
    assert result["recommended_option_id"] is not None
    assert len(result["options"]) == 3
    assert "why_explanation" in result
    assert result["why_explanation"]["recommended_action"] != "None"


def test_impossible_constraints_handling(sample_shipment, sample_prediction):
    # Impossible budget ($1,000 when base is $6,000) and impossible delivery (1 day)
    impossible_constraints = OptimizationConstraints(max_budget=1000.0, max_delivery_days=1.0)
    result = optimize_actions(sample_shipment, sample_prediction, impossible_constraints)

    assert result["solver_status"] == "INFEASIBLE"
    for opt in result["options"]:
        assert opt["is_feasible"] is False
        assert opt["status"] == "INFEASIBLE"
