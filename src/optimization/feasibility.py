"""Feasibility validation module for Supply Prescript Prescriptive Optimization."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeasibilityCheckResult:
    """Summary of pre-solver operational feasibility assessment."""

    is_feasible: bool
    violations: List[str]
    warnings: List[str]
    available_capacity_units: int
    net_demand_units: int
    max_service_level_possible: float


def validate_operational_feasibility(
    demand_units: int,
    available_inventory: int,
    primary_capacity: int,
    secondary_capacity: int,
    air_freight_capacity: int,
    service_level_target: float = 0.95,
    allow_unmet_demand: bool = True,
    allow_launch_delay: bool = True,
) -> FeasibilityCheckResult:
    """Validate operational capacity and bounds prior to linear program initialization.

    Args:
        demand_units: Gross demand for the replenishment period.
        available_inventory: Non-reserved usable stock on hand.
        primary_capacity: Available capacity of the designated primary supplier.
        secondary_capacity: Available capacity of verified secondary/backup suppliers.
        air_freight_capacity: Maximum quantity that logistics can airlift.
        service_level_target: Minimum required proportion of demand satisfied.
        allow_unmet_demand: Whether shortage penalty is permitted if capacity fails.
        allow_launch_delay: Whether postponement can mitigate capacity shortfall.

    Returns:
        FeasibilityCheckResult with validation status and diagnostic details.
    """
    violations: List[str] = []
    warnings: List[str] = []

    # 1. Non-negativity checks
    if demand_units < 0:
        violations.append(f"Demand units cannot be negative: {demand_units}")
    if available_inventory < 0:
        violations.append(f"Available inventory cannot be negative: {available_inventory}")
    if primary_capacity < 0:
        violations.append(f"Primary supplier capacity cannot be negative: {primary_capacity}")
    if secondary_capacity < 0:
        violations.append(f"Secondary supplier capacity cannot be negative: {secondary_capacity}")
    if air_freight_capacity < 0:
        violations.append(f"Air freight capacity cannot be negative: {air_freight_capacity}")
    if not (0.0 <= service_level_target <= 1.0):
        violations.append(f"Service level target must be within [0, 1]: {service_level_target}")

    if violations:
        return FeasibilityCheckResult(
            is_feasible=False,
            violations=violations,
            warnings=warnings,
            available_capacity_units=0,
            net_demand_units=max(0, demand_units - max(0, available_inventory)),
            max_service_level_possible=0.0,
        )

    # 2. Net demand after deducting available stock
    # Note: available inventory is strictly deducted once and not double-counted
    net_demand = max(0, demand_units - available_inventory)

    # Total recovery supply potential across alternatives
    total_recovery_capacity = primary_capacity + secondary_capacity + air_freight_capacity
    total_available_supply = available_inventory + total_recovery_capacity

    # 3. Maximum reachable service level
    if demand_units == 0:
        max_service_level = 1.0
    else:
        max_service_level = min(1.0, total_available_supply / demand_units)

    # 4. Strict capacity verification
    required_units = int(demand_units * service_level_target)
    if total_available_supply < required_units and not allow_unmet_demand and not allow_launch_delay:
        violations.append(
            f"Insufficient total network supply ({total_available_supply} units) to meet required "
            f"service level target {service_level_target * 100:.1f}% ({required_units} units) "
            f"when shortages and postponement are disabled."
        )

    if total_available_supply < demand_units:
        warnings.append(
            f"Total available network capacity ({total_available_supply} units) is less than gross demand "
            f"({demand_units} units). Shortage penalties or postponement will be required."
        )

    if available_inventory >= demand_units:
        warnings.append(
            f"Available inventory ({available_inventory} units) already covers gross demand ({demand_units} units). "
            f"Replenishment may not be immediately urgent."
        )

    is_feasible = len(violations) == 0

    return FeasibilityCheckResult(
        is_feasible=is_feasible,
        violations=violations,
        warnings=warnings,
        available_capacity_units=total_recovery_capacity,
        net_demand_units=net_demand,
        max_service_level_possible=max_service_level,
    )

