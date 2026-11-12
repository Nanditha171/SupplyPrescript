"""Cost calculation and financial modeling for prescriptive optimization recovery strategies."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Any, Optional


@dataclass
class CostBreakdown:
    """Detailed financial breakdown of an operational supply chain recovery solution."""

    primary_procurement_cost: float
    primary_transport_cost: float
    secondary_procurement_cost: float
    secondary_premium_cost: float
    air_freight_cost: float
    shortage_penalty_cost: float
    launch_delay_cost: float
    total_cost: float
    learned_adjustments_applied: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_procurement_cost": round(self.primary_procurement_cost, 2),
            "primary_transport_cost": round(self.primary_transport_cost, 2),
            "secondary_procurement_cost": round(self.secondary_procurement_cost, 2),
            "secondary_premium_cost": round(self.secondary_premium_cost, 2),
            "air_freight_cost": round(self.air_freight_cost, 2),
            "shortage_penalty_cost": round(self.shortage_penalty_cost, 2),
            "launch_delay_cost": round(self.launch_delay_cost, 2),
            "total_cost": round(self.total_cost, 2),
            "learned_adjustments_applied": self.learned_adjustments_applied,
        }


class CostCalculator:
    """Computes exact cost components with learned parameter multipliers applied."""

    def __init__(self, config_cost_models: Optional[Dict[str, Any]] = None):
        self.config = config_cost_models or {}

        # Default cost assumptions if config not fully supplied
        self.air_freight_cfg = self.config.get("air_freight", {})
        self.secondary_cfg = self.config.get("secondary_supplier", {})
        self.launch_delay_cfg = self.config.get("launch_delay", {})
        self.shortage_cfg = self.config.get("shortage", {})

        self.default_secondary_premium = float(self.secondary_cfg.get("premium_percentage", 0.10))
        self.default_shortage_penalty = float(self.shortage_cfg.get("penalty_per_unit", 250.00))
        self.daily_delay_cost = float(self.launch_delay_cfg.get("daily_postponement_cost", 1200.00))
        self.fixed_delay_cost = float(self.launch_delay_cfg.get("fixed_impact_cost", 5000.00))

    def calculate_cost_breakdown(
        self,
        primary_qty: int,
        secondary_qty: int,
        air_freight_qty: int,
        unmet_demand_qty: int,
        is_launch_delayed: bool,
        delay_days: float,
        unit_cost: float,
        base_transport_cost: float,
        secondary_premium_rate: Optional[float] = None,
        air_freight_base_cost: Optional[float] = None,
        learned_multipliers: Optional[Dict[str, float]] = None,
    ) -> CostBreakdown:
        """Calculate complete financial breakdown with learned multipliers applied transparently.

        Args:
            primary_qty: Units fulfilled via primary supplier.
            secondary_qty: Units fulfilled via secondary supplier.
            air_freight_qty: Units fulfilled via expedited air freight.
            unmet_demand_qty: Unfulfilled units incurring shortage penalty.
            is_launch_delayed: Whether product launch was postponed.
            delay_days: Expected delay days for launch postponement.
            unit_cost: Base product manufacturing unit cost.
            base_transport_cost: Standard baseline transportation cost.
            secondary_premium_rate: Specific premium percentage for secondary sourcing (e.g. 0.10 for 10%).
            air_freight_base_cost: Base expedited freight cost.
            learned_multipliers: Multipliers learned from historical outcome variances (e.g., {'AIR_FREIGHT': 1.15}).

        Returns:
            CostBreakdown object with all sub-costs.
        """
        multipliers = learned_multipliers or {}
        air_multiplier = multipliers.get("AIR_FREIGHT", 1.0)
        secondary_multiplier = multipliers.get("SECONDARY_SUPPLIER", 1.0)

        # 1. Primary Supplier Costs
        c_prim_proc = primary_qty * unit_cost
        # Allocate proportional transport cost if applicable
        c_prim_trans = base_transport_cost if primary_qty > 0 else 0.0

        # 2. Secondary Supplier Costs
        premium_rate = secondary_premium_rate if secondary_premium_rate is not None else self.default_secondary_premium
        base_sec_proc = secondary_qty * unit_cost
        base_sec_prem = secondary_qty * unit_cost * premium_rate
        # Apply learned multiplier to secondary premium if calibrated
        c_sec_proc = base_sec_proc
        c_sec_prem = base_sec_prem * secondary_multiplier

        # 3. Air Freight Costs
        # If air_freight_base_cost provided (e.g. $15,000 demonstration assumption), use as baseline
        if air_freight_base_cost is not None and air_freight_base_cost > 0:
            c_air = (air_freight_base_cost if air_freight_qty > 0 else 0.0) * air_multiplier
        else:
            # Variable model
            fixed_air = float(self.air_freight_cfg.get("fixed_setup_cost", 2500.00)) if air_freight_qty > 0 else 0.0
            var_air = air_freight_qty * float(self.air_freight_cfg.get("variable_cost_per_unit", 12.50))
            c_air = (fixed_air + var_air) * air_multiplier

        # 4. Shortage Penalties
        c_shortage = unmet_demand_qty * self.default_shortage_penalty

        # 5. Launch Delay Costs
        if is_launch_delayed:
            c_delay = self.fixed_delay_cost + (self.daily_delay_cost * delay_days)
        else:
            c_delay = 0.0

        total_cost = (
            c_prim_proc
            + c_prim_trans
            + c_sec_proc
            + c_sec_prem
            + c_air
            + c_shortage
            + c_delay
        )

        return CostBreakdown(
            primary_procurement_cost=c_prim_proc,
            primary_transport_cost=c_prim_trans,
            secondary_procurement_cost=c_sec_proc,
            secondary_premium_cost=c_sec_prem,
            air_freight_cost=c_air,
            shortage_penalty_cost=c_shortage,
            launch_delay_cost=c_delay,
            total_cost=total_cost,
            learned_adjustments_applied={
                "AIR_FREIGHT": air_multiplier,
                "SECONDARY_SUPPLIER": secondary_multiplier,
            },
        )

