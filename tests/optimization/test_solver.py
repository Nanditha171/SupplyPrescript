"""Tests for Prescriptive Optimization Solver (Day 10)."""

import pytest
from src.optimization.solution_parser import OptimizationInput
from src.optimization.solver import PrescriptiveOptimizationSolver
from src.optimization.feasibility import validate_operational_feasibility
from src.optimization.cost_calculator import CostCalculator


@pytest.fixture
def solver():
    return PrescriptiveOptimizationSolver()


def test_feasibility_validation_normal():
    res = validate_operational_feasibility(
        demand_units=1000,
        available_inventory=200,
        primary_capacity=800,
        secondary_capacity=500,
        air_freight_capacity=2000,
        service_level_target=0.95,
    )
    assert res.is_feasible is True
    assert res.net_demand_units == 800
    assert res.max_service_level_possible >= 0.95


def test_feasibility_validation_negative_input():
    res = validate_operational_feasibility(
        demand_units=-50,
        available_inventory=100,
        primary_capacity=500,
        secondary_capacity=500,
        air_freight_capacity=1000,
    )
    assert res.is_feasible is False
    assert any("negative" in v for v in res.violations)


def test_solver_feasible_scenario(solver):
    inp = OptimizationInput(
        event_id="TEST-EVT-001",
        product_code="P001",
        demand_units=500,
        available_inventory=100,
        unit_cost=100.0,
        base_transport_cost=1500.0,
        primary_capacity=500,
        secondary_capacity=1000,
        air_freight_capacity=2000,
        predicted_delay_days=14.0,
        delay_probability=0.85,
        composite_risk_score=0.75,
        objective_type="MIN_COST",
    )
    res = solver.solve(inp)
    assert res.solver_status in ("OPTIMAL", "FEASIBLE")
    assert res.estimated_total_cost > 0
    assert res.primary_supplier_quantity + res.secondary_supplier_quantity + res.air_freight_quantity + inp.available_inventory >= inp.demand_units


def test_solver_air_freight_alternative(solver):
    """Core demonstration scenario Alternative A: Air freight."""
    inp = OptimizationInput(
        event_id="EVT-001",
        product_code="P001",
        demand_units=1000,
        available_inventory=0,
        unit_cost=125.50,
        base_transport_cost=2000.0,
        primary_capacity=0,  # Disrupted primary supplier
        secondary_capacity=1000,
        air_freight_capacity=2000,
        predicted_delay_days=14.0,
        delay_probability=0.87,
        composite_risk_score=0.82,
        air_freight_base_cost=15000.0,
        forced_strategy="AIR_FREIGHT",
    )
    res = solver.solve(inp)
    assert res.solver_status in ("OPTIMAL", "FEASIBLE")
    assert "AIR_FREIGHT" in res.selected_strategies
    assert res.cost_breakdown.air_freight_cost == 15000.0
    assert res.estimated_delivery_days == 2.0


def test_solver_secondary_supplier_alternative(solver):
    """Core demonstration scenario Alternative B: Secondary Supplier with 10% premium."""
    inp = OptimizationInput(
        event_id="EVT-001",
        product_code="P001",
        demand_units=1000,
        available_inventory=0,
        unit_cost=125.50,
        base_transport_cost=2000.0,
        primary_capacity=0,
        secondary_capacity=1000,
        air_freight_capacity=2000,
        predicted_delay_days=14.0,
        delay_probability=0.87,
        composite_risk_score=0.82,
        secondary_premium_rate=0.10,
        forced_strategy="SECONDARY_SUPPLIER",
    )
    res = solver.solve(inp)
    assert res.solver_status in ("OPTIMAL", "FEASIBLE")
    assert "SECONDARY_SUPPLIER" in res.selected_strategies
    assert res.cost_breakdown.secondary_premium_cost > 0
    assert res.estimated_delivery_days == 7.0


def test_solver_launch_delay_alternative(solver):
    """Core demonstration scenario Alternative C: Delay product launch."""
    inp = OptimizationInput(
        event_id="EVT-001",
        product_code="P001",
        demand_units=1000,
        available_inventory=0,
        unit_cost=125.50,
        base_transport_cost=2000.0,
        primary_capacity=0,
        secondary_capacity=0,
        air_freight_capacity=0,
        predicted_delay_days=14.0,
        delay_probability=0.87,
        composite_risk_score=0.82,
        forced_strategy="DELAY_LAUNCH",
    )
    res = solver.solve(inp)
    assert res.solver_status in ("OPTIMAL", "FEASIBLE")
    assert res.is_launch_delayed is True
    assert res.cost_breakdown.launch_delay_cost > 0
    assert res.estimated_delivery_days == 14.0


def test_solver_zero_demand_edge_case(solver):
    inp = OptimizationInput(
        event_id="ZERO-DEMAND",
        product_code="P001",
        demand_units=0,
        available_inventory=50,
        unit_cost=100.0,
        base_transport_cost=0.0,
        primary_capacity=500,
        secondary_capacity=500,
        air_freight_capacity=500,
        predicted_delay_days=0.0,
        delay_probability=0.1,
        composite_risk_score=0.1,
    )
    res = solver.solve(inp)
    assert res.solver_status in ("OPTIMAL", "FEASIBLE")
    assert res.estimated_total_cost == 0.0
    assert res.service_level_achieved == 1.0


def test_solver_infeasible_strict_shortage(solver):
    """When shortage and delay are not allowed and capacity is 0, status must be INFEASIBLE."""
    inp = OptimizationInput(
        event_id="INFEASIBLE-EVT",
        product_code="P001",
        demand_units=1000,
        available_inventory=0,
        unit_cost=100.0,
        base_transport_cost=0.0,
        primary_capacity=0,
        secondary_capacity=0,
        air_freight_capacity=0,
        predicted_delay_days=10.0,
        delay_probability=0.5,
        composite_risk_score=0.5,
        allow_unmet_demand=False,
        allow_launch_delay=False,
    )
    res = solver.solve(inp)
    assert res.solver_status == "INFEASIBLE"
    assert res.objective_value == float("inf")

