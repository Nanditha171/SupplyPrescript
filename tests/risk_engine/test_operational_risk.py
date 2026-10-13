"""Unit tests for operational risk dimensions and leakage prevention."""

import numpy as np
import pandas as pd
import pytest

from src.risk_engine.operational_risk import (
    calculate_capacity_risk,
    calculate_inventory_risk,
    calculate_service_risk,
    calculate_supplier_risk,
    compute_past_supplier_delay_rates,
)


# ==========================================
# 1. Supplier Risk Tests
# ==========================================

def test_supplier_risk_calculation():
    """Verify supplier risk inversion and historical rate blending."""
    # Perfect reliability -> 0.0 risk
    assert calculate_supplier_risk(1.0) == 0.0
    # Inverted score: 1.0 - 0.70 = 0.30
    assert calculate_supplier_risk(0.70) == 0.30
    # Zero reliability -> 1.0 risk
    assert calculate_supplier_risk(0.0) == 1.0

    # Combined with strictly past historical delay rate:
    # 0.70 * (1 - 0.80) + 0.30 * 0.40 = 0.14 + 0.12 = 0.26
    combined = calculate_supplier_risk(reliability=0.80, historical_delay_rate=0.40)
    assert combined == 0.26


def test_supplier_risk_invalid_inputs():
    """Verify validation on supplier reliability."""
    with pytest.raises(ValueError, match="in \\[0.0, 1.0\\]"):
        calculate_supplier_risk(1.5)

    with pytest.raises(ValueError, match="in \\[0.0, 1.0\\]"):
        calculate_supplier_risk(-0.1)

    with pytest.raises(TypeError):
        calculate_supplier_risk("high")  # type: ignore


def test_historical_delay_leakage_prevention():
    """CRITICAL: Verify past-only delay calculation strictly avoids target leakage."""
    # Create sequence of 3 shipments for SUP001
    df = pd.DataFrame([
        {"date": "2025-01-01", "supplier_id": "SUP001", "delay_occurred": 1},
        {"date": "2025-01-08", "supplier_id": "SUP001", "delay_occurred": 0},
        {"date": "2025-01-15", "supplier_id": "SUP001", "delay_occurred": 1},
    ])

    past_rates = compute_past_supplier_delay_rates(df)

    # First shipment has no prior history -> must be 0.0
    assert past_rates.iloc[0] == 0.0

    # Second shipment only sees shipment 1 (1 delay out of 1 -> 1.0)
    # DOES NOT see its own outcome (0) or future outcome (1)
    assert past_rates.iloc[1] == 1.0

    # Third shipment sees shipments 1 and 2 (1 delay out of 2 -> 0.5)
    # DOES NOT leak its own delay outcome (1)
    assert past_rates.iloc[2] == 0.5


# ==========================================
# 2. Inventory Risk Tests
# ==========================================

def test_inventory_risk_calculation():
    """Verify continuous piece-wise inventory risk scores."""
    # Zero stock -> 1.0 max risk
    assert calculate_inventory_risk(coverage_ratio=0.0) == 1.00

    # Critical stock (< 0.50): coverage = 0.25 -> 0.80 + 0.20*(1 - 0.5) = 0.90
    assert calculate_inventory_risk(coverage_ratio=0.25) == 0.90

    # Boundary: coverage = 0.50 -> 0.80
    assert calculate_inventory_risk(coverage_ratio=0.50) == 0.80

    # Low coverage (0.50 <= coverage < 1.00): coverage = 0.75 -> 0.60
    assert calculate_inventory_risk(coverage_ratio=0.75) == 0.60

    # Adequate coverage: coverage = 1.00 -> 0.40
    assert calculate_inventory_risk(coverage_ratio=1.00) == 0.40

    # Abundant coverage: coverage = 1.50 -> 0.20
    assert calculate_inventory_risk(coverage_ratio=1.50) == 0.20

    # Ample coverage: coverage >= 2.00 -> 0.00
    assert calculate_inventory_risk(coverage_ratio=2.50) == 0.00


def test_inventory_risk_zero_demand_and_units():
    """Verify inventory risk when computed from raw units and zero-demand edge case."""
    # Zero demand -> no risk
    assert calculate_inventory_risk(inventory_units=500, demand_units=0) == 0.0

    # Derived coverage: 250 / 500 = 0.50
    assert calculate_inventory_risk(inventory_units=250, demand_units=500) == 0.80


def test_inventory_risk_invalid_inputs():
    """Verify rejection of negative or invalid inputs."""
    with pytest.raises(ValueError, match="negative"):
        calculate_inventory_risk(coverage_ratio=-0.5)

    with pytest.raises(ValueError, match="non-negative"):
        calculate_inventory_risk(inventory_units=-10, demand_units=100)


# ==========================================
# 3. Capacity Risk Tests
# ==========================================

def test_capacity_risk_calculation():
    """Verify capacity risk bands and pressure scaling."""
    # Low utilization: 500 / 1000 = 0.50 -> < 0.70
    score_low = calculate_capacity_risk(demand_units=500, capacity_units=1000)
    assert 0.0 <= score_low < 0.30

    # Moderate utilization: 750 / 1000 = 0.75 -> [0.30, 0.60)
    score_mod = calculate_capacity_risk(demand_units=750, capacity_units=1000)
    assert 0.30 <= score_mod < 0.60

    # High utilization: 900 / 1000 = 0.90 -> [0.60, 0.85)
    score_high = calculate_capacity_risk(demand_units=900, capacity_units=1000)
    assert 0.60 <= score_high < 0.85

    # Critical overload: 1100 / 1000 = 1.10 -> [0.85, 1.00]
    score_crit = calculate_capacity_risk(demand_units=1100, capacity_units=1000)
    assert 0.85 <= score_crit <= 1.00


def test_capacity_risk_zero_capacity_handling():
    """Verify zero-capacity handling without division by zero."""
    # Zero capacity with positive demand -> 1.0 (complete failure to supply)
    assert calculate_capacity_risk(demand_units=500, capacity_units=0) == 1.0

    # Zero capacity with zero demand -> 0.0
    assert calculate_capacity_risk(demand_units=0, capacity_units=0) == 0.0


def test_capacity_risk_invalid_inputs():
    """Verify rejection of negative capacity or demand."""
    with pytest.raises(ValueError, match="cannot be negative"):
        calculate_capacity_risk(demand_units=-100, capacity_units=500)

    with pytest.raises(ValueError, match="cannot be negative"):
        calculate_capacity_risk(demand_units=500, capacity_units=-500)


# ==========================================
# 4. Service Risk Tests
# ==========================================

def test_service_risk_calculation():
    """Verify product criticality mapping and shortage modulation."""
    # If product criticality is missing/None, dimension is unavailable (None)
    assert calculate_service_risk(product_criticality=None) is None
    assert calculate_service_risk(product_criticality="") is None
    assert calculate_service_risk(product_criticality="nan") is None

    # CRITICAL product under severe shortage (coverage < 0.50) -> full exposure (1.00)
    assert calculate_service_risk("CRITICAL", coverage_ratio=0.30) == 1.00

    # CRITICAL product with healthy inventory buffer (coverage >= 1.0) -> cushioned
    crit_cushioned = calculate_service_risk("CRITICAL", coverage_ratio=1.50)
    assert crit_cushioned == 0.40

    # LOW criticality product -> low base score
    low_crit = calculate_service_risk("LOW", coverage_ratio=0.30)
    assert low_crit == 0.25
