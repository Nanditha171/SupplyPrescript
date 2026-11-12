"""PuLP Mixed-Integer Linear Programming (MILP) Solver for Supply Chain Recovery Strategies."""

import time
import uuid
import logging
from typing import Dict, Any, List, Optional
import pulp

from src.optimization.feasibility import validate_operational_feasibility
from src.optimization.cost_calculator import CostCalculator, CostBreakdown
from src.optimization.solution_parser import OptimizationInput, OptimizationOutput

logger = logging.getLogger(__name__)


class PrescriptiveOptimizationSolver:
    """Solves the multi-alternative supply chain recovery optimization formulation."""

    def __init__(self, config_cost_models: Optional[Dict[str, Any]] = None):
        self.cost_calculator = CostCalculator(config_cost_models)
        self.config_cost_models = config_cost_models or {}

    def solve(self, opt_input: OptimizationInput) -> OptimizationOutput:
        """Formulate and solve the recovery optimization problem with PuLP.

        Args:
            opt_input: Complete optimization parameters and constraints.

        Returns:
            OptimizationOutput with solution quantities, costs, and solver diagnostics.
        """
        start_time = time.perf_counter()
        run_id = f"OPT-{uuid.uuid4().hex[:8].upper()}"

        # 1. Feasibility Pre-Check
        feasibility = validate_operational_feasibility(
            demand_units=opt_input.demand_units,
            available_inventory=opt_input.available_inventory,
            primary_capacity=opt_input.primary_capacity,
            secondary_capacity=opt_input.secondary_capacity,
            air_freight_capacity=opt_input.air_freight_capacity,
            service_level_target=opt_input.service_level_target,
            allow_unmet_demand=opt_input.allow_unmet_demand,
            allow_launch_delay=opt_input.allow_launch_delay,
        )

        if not feasibility.is_feasible:
            exec_time = time.perf_counter() - start_time
            zero_cost = self.cost_calculator.calculate_cost_breakdown(
                primary_qty=0,
                secondary_qty=0,
                air_freight_qty=0,
                unmet_demand_qty=opt_input.demand_units,
                is_launch_delayed=False,
                delay_days=opt_input.predicted_delay_days,
                unit_cost=opt_input.unit_cost,
                base_transport_cost=opt_input.base_transport_cost,
            )
            return OptimizationOutput(
                optimization_run_id=run_id,
                event_id=opt_input.event_id,
                solver_status="INFEASIBLE",
                objective_value=float("inf"),
                estimated_total_cost=0.0,
                primary_supplier_quantity=0,
                secondary_supplier_quantity=0,
                air_freight_quantity=0,
                unmet_demand=opt_input.demand_units,
                is_launch_delayed=False,
                selected_strategies=[],
                cost_breakdown=zero_cost,
                constraint_status={"feasibility_errors": feasibility.violations},
                solver_execution_time=exec_time,
                service_level_achieved=0.0,
                estimated_delivery_days=opt_input.predicted_delay_days,
                estimated_risk=1.0,
            )

        # 2. Initialize PuLP Model
        problem_name = f"Supply_Prescript_{opt_input.event_id}_{run_id}"
        prob = pulp.LpProblem(problem_name, pulp.LpMinimize)

        # 3. Decision Variables
        # Continuous / Integer Quantities
        net_demand = max(0, opt_input.demand_units - opt_input.available_inventory)
        upper_bound = max(opt_input.demand_units, 100000)

        x_prim = pulp.LpVariable("x_primary", lowBound=0, upBound=opt_input.primary_capacity, cat=pulp.LpInteger)
        x_sec = pulp.LpVariable("x_secondary", lowBound=0, upBound=opt_input.secondary_capacity, cat=pulp.LpInteger)
        x_air = pulp.LpVariable("x_air_freight", lowBound=0, upBound=opt_input.air_freight_capacity, cat=pulp.LpInteger)
        
        # Unmet demand
        if opt_input.allow_unmet_demand:
            s_unmet = pulp.LpVariable("s_unmet_demand", lowBound=0, upBound=upper_bound, cat=pulp.LpInteger)
        else:
            s_unmet = pulp.LpVariable("s_unmet_demand", lowBound=0, upBound=0, cat=pulp.LpInteger)

        # Strategy Activation Binary Variables
        y_air = pulp.LpVariable("y_air_freight", cat=pulp.LpBinary)
        y_sec = pulp.LpVariable("y_secondary", cat=pulp.LpBinary)
        y_delay = pulp.LpVariable("y_launch_delay", cat=pulp.LpBinary)

        # Strategy-specific constraints when forced_strategy is active
        if opt_input.forced_strategy == "AIR_FREIGHT":
            prob += y_air == 1, "force_air_freight"
            prob += y_sec == 0, "disable_secondary"
            prob += y_delay == 0, "disable_delay"
            prob += x_prim == 0, "primary_disrupted"
            prob += s_unmet == 0, "no_shortage_when_evaluating_air"
        elif opt_input.forced_strategy == "SECONDARY_SUPPLIER":
            prob += y_sec == 1, "force_secondary"
            prob += y_air == 0, "disable_air"
            prob += y_delay == 0, "disable_delay"
            prob += x_prim == 0, "primary_disrupted"
            prob += s_unmet == 0, "no_shortage_when_evaluating_secondary"
        elif opt_input.forced_strategy == "DELAY_LAUNCH":
            prob += y_delay == 1, "force_launch_delay"
            prob += y_air == 0, "disable_air"
            prob += y_sec == 0, "disable_secondary"
            prob += s_unmet == 0, "no_shortage_when_evaluating_delay"
        elif opt_input.forced_strategy == "PRIMARY_SUPPLIER":
            prob += y_air == 0, "disable_air"
            prob += y_sec == 0, "disable_secondary"
            prob += y_delay == 0, "disable_delay"
            prob += s_unmet == 0, "no_shortage_when_evaluating_primary"
        elif not opt_input.allow_launch_delay:
            prob += y_delay == 0, "launch_delay_disabled"

        # 4. Constraints
        # Demand Fulfillment: inventory + allocations + unmet + postponed = demand
        prob += (
            opt_input.available_inventory + x_prim + x_sec + x_air + s_unmet + opt_input.demand_units * y_delay >= opt_input.demand_units,
            "demand_fulfillment",
        )

        # Capacity linkages with binary activation
        prob += x_air <= opt_input.air_freight_capacity * y_air, "air_freight_capacity_activation"
        prob += x_sec <= opt_input.secondary_capacity * y_sec, "secondary_capacity_activation"

        # Minimum service level constraint (fulfilled / demand >= target, postponing absorbs shortfall)
        if opt_input.demand_units > 0:
            min_fulfilled = int(opt_input.demand_units * opt_input.service_level_target)
            prob += (
                opt_input.available_inventory + x_prim + x_sec + x_air + opt_input.demand_units * y_delay >= min_fulfilled,
                "minimum_service_level",
            )

        # 5. Objective Function Formulation
        # Unit and expedited costs
        c_prim = opt_input.unit_cost
        sec_multiplier = opt_input.learned_multipliers.get("SECONDARY_SUPPLIER", 1.0)
        c_sec_unit = opt_input.unit_cost * (1.0 + opt_input.secondary_premium_rate) * sec_multiplier

        air_multiplier = opt_input.learned_multipliers.get("AIR_FREIGHT", 1.0)
        # Check if benchmark fixed air freight cost applies (e.g. $15,000)
        if opt_input.air_freight_base_cost is not None and opt_input.air_freight_base_cost > 0:
            air_cost_term = opt_input.air_freight_base_cost * air_multiplier * y_air
        else:
            air_fixed = 2500.0 * air_multiplier
            air_var = 12.50 * air_multiplier
            air_cost_term = air_fixed * y_air + air_var * x_air

        # Shortage penalty
        shortage_penalty = self.cost_calculator.default_shortage_penalty
        # Launch delay cost
        launch_delay_cost = (
            self.cost_calculator.fixed_delay_cost
            + self.cost_calculator.daily_delay_cost * opt_input.predicted_delay_days
        )

        cost_expression = (
            c_prim * x_prim
            + c_sec_unit * x_sec
            + air_cost_term
            + shortage_penalty * s_unmet
            + launch_delay_cost * y_delay
        )

        # Objective weighting
        obj_type = opt_input.objective_type.upper()
        if obj_type == "MIN_COST":
            prob += cost_expression, "minimize_total_cost"
        elif obj_type == "MIN_RISK":
            # Risk weights: air freight = 0.08, secondary = 0.15, launch delay = 0.35, shortage = 0.95
            risk_term = 0.08 * y_air + 0.15 * y_sec + 0.35 * y_delay + 0.95 * (s_unmet / max(1, opt_input.demand_units))
            ref_cost = max(1.0, float(opt_input.demand_units * opt_input.unit_cost * 1.5))
            prob += 0.20 * (cost_expression / ref_cost) + 0.80 * risk_term, "minimize_risk_dominant"
        elif obj_type == "MIN_DELAY":
            # Delay term: air = 2 days, sec = 7 days, primary/delay = predicted_delay_days
            delay_term = (
                2.0 * y_air
                + 7.0 * y_sec
                + opt_input.predicted_delay_days * y_delay
                + opt_input.predicted_delay_days * (x_prim / max(1, opt_input.demand_units))
            )
            ref_cost = max(1.0, float(opt_input.demand_units * opt_input.unit_cost * 1.5))
            ref_delay = max(1.0, opt_input.predicted_delay_days + 5.0)
            prob += 0.20 * (cost_expression / ref_cost) + 0.80 * (delay_term / ref_delay), "minimize_delay_dominant"
        else:  # BALANCED
            ref_cost = max(1.0, float(opt_input.demand_units * opt_input.unit_cost * 1.5))
            risk_term = 0.08 * y_air + 0.15 * y_sec + 0.35 * y_delay + 0.95 * (s_unmet / max(1, opt_input.demand_units))
            delay_term = (
                2.0 * y_air
                + 7.0 * y_sec
                + opt_input.predicted_delay_days * y_delay
                + opt_input.predicted_delay_days * (x_prim / max(1, opt_input.demand_units))
            )
            ref_delay = max(1.0, opt_input.predicted_delay_days + 5.0)
            prob += (
                opt_input.cost_weight * (cost_expression / ref_cost)
                + opt_input.risk_weight * risk_term
                + opt_input.time_weight * (delay_term / ref_delay)
            ), "balanced_objective"

        # 6. Execute PuLP CBC Solver
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=30)
        prob.solve(solver)

        exec_time = time.perf_counter() - start_time

        # 7. Map Solver Status
        status_map = {
            pulp.LpStatusOptimal: "OPTIMAL",
            pulp.LpStatusNotSolved: "NOT_SOLVED",
            pulp.LpStatusInfeasible: "INFEASIBLE",
            pulp.LpStatusUnbounded: "UNBOUNDED",
            pulp.LpStatusUndefined: "FEASIBLE" if prob.status == 0 else "ERROR",
        }
        solver_status = status_map.get(prob.status, "ERROR")

        if solver_status not in ("OPTIMAL", "FEASIBLE"):
            zero_cost = self.cost_calculator.calculate_cost_breakdown(
                primary_qty=0,
                secondary_qty=0,
                air_freight_qty=0,
                unmet_demand_qty=opt_input.demand_units,
                is_launch_delayed=False,
                delay_days=opt_input.predicted_delay_days,
                unit_cost=opt_input.unit_cost,
                base_transport_cost=opt_input.base_transport_cost,
            )
            return OptimizationOutput(
                optimization_run_id=run_id,
                event_id=opt_input.event_id,
                solver_status=solver_status,
                objective_value=float("inf"),
                estimated_total_cost=0.0,
                primary_supplier_quantity=0,
                secondary_supplier_quantity=0,
                air_freight_quantity=0,
                unmet_demand=opt_input.demand_units,
                is_launch_delayed=False,
                selected_strategies=[],
                cost_breakdown=zero_cost,
                constraint_status={"solver_code": prob.status},
                solver_execution_time=exec_time,
                service_level_achieved=0.0,
                estimated_delivery_days=opt_input.predicted_delay_days,
                estimated_risk=1.0,
            )

        # Extract values safely
        sol_prim = int(round(pulp.value(x_prim) or 0))
        sol_sec = int(round(pulp.value(x_sec) or 0))
        sol_air = int(round(pulp.value(x_air) or 0))
        sol_unmet = int(round(pulp.value(s_unmet) or 0))
        sol_delay = bool(round(pulp.value(y_delay) or 0) == 1)

        # Selected strategies list
        strategies = []
        if sol_air > 0 or (pulp.value(y_air) and round(pulp.value(y_air)) == 1):
            strategies.append("AIR_FREIGHT")
        if sol_sec > 0 or (pulp.value(y_sec) and round(pulp.value(y_sec)) == 1):
            strategies.append("SECONDARY_SUPPLIER")
        if sol_delay:
            strategies.append("DELAY_LAUNCH")
        if sol_prim > 0:
            strategies.append("PRIMARY_SUPPLIER")
        if not strategies:
            strategies.append("PRIMARY_SUPPLIER")

        # Calculate exact cost breakdown
        cost_breakdown = self.cost_calculator.calculate_cost_breakdown(
            primary_qty=sol_prim,
            secondary_qty=sol_sec,
            air_freight_qty=sol_air,
            unmet_demand_qty=sol_unmet,
            is_launch_delayed=sol_delay,
            delay_days=opt_input.predicted_delay_days,
            unit_cost=opt_input.unit_cost,
            base_transport_cost=opt_input.base_transport_cost,
            secondary_premium_rate=opt_input.secondary_premium_rate,
            air_freight_base_cost=opt_input.air_freight_base_cost if "AIR_FREIGHT" in strategies else None,
            learned_multipliers=opt_input.learned_multipliers,
        )

        total_fulfilled = opt_input.available_inventory + sol_prim + sol_sec + sol_air
        achieved_service_level = (
            1.0 if opt_input.demand_units == 0 else min(1.0, total_fulfilled / opt_input.demand_units)
        )

        # Delivery days calculation
        if "AIR_FREIGHT" in strategies:
            est_delivery_days = 2.0
            est_risk = 0.08
        elif "SECONDARY_SUPPLIER" in strategies:
            est_delivery_days = 7.0
            est_risk = 0.15
        elif sol_delay:
            est_delivery_days = opt_input.predicted_delay_days
            est_risk = 0.35
        else:
            est_delivery_days = opt_input.predicted_delay_days
            est_risk = opt_input.composite_risk_score

        # Check constraint slack
        constraint_status = {
            "demand_fulfillment_slack": total_fulfilled + sol_unmet - opt_input.demand_units,
            "primary_capacity_remaining": opt_input.primary_capacity - sol_prim,
            "secondary_capacity_remaining": opt_input.secondary_capacity - sol_sec,
            "air_freight_capacity_remaining": opt_input.air_freight_capacity - sol_air,
            "service_level_target": opt_input.service_level_target,
            "service_level_achieved": achieved_service_level,
        }

        return OptimizationOutput(
            optimization_run_id=run_id,
            event_id=opt_input.event_id,
            solver_status=solver_status,
            objective_value=float(pulp.value(prob.objective) or cost_breakdown.total_cost),
            estimated_total_cost=cost_breakdown.total_cost,
            primary_supplier_quantity=sol_prim,
            secondary_supplier_quantity=sol_sec,
            air_freight_quantity=sol_air,
            unmet_demand=sol_unmet,
            is_launch_delayed=sol_delay,
            selected_strategies=strategies,
            cost_breakdown=cost_breakdown,
            constraint_status=constraint_status,
            solver_execution_time=exec_time,
            service_level_achieved=achieved_service_level,
            estimated_delivery_days=est_delivery_days,
            estimated_risk=est_risk,
        )
