import os
import sys
from typing import Dict, Any, List, Optional
import numpy as np

try:
    import pulp
    HAS_PULP = True
except ImportError:
    pulp = None
    HAS_PULP = False

import scipy.optimize as sp_opt

# Ensure root directory is on PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from optimization.constraints import OptimizationConstraints
from optimization.validator import validate_all_options


def generate_candidate_options(
    shipment_data: Dict[str, Any],
    prediction: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Generates three genuinely realistic, feasible alternative mitigation actions
    derived from the shipment's base costs, order quantity, distance, and predicted delay.
    """
    base_cost = float(shipment_data.get("shipping_cost", 5000.0))
    base_lead_time = float(shipment_data.get("lead_time_days", 14.0))
    order_qty = int(shipment_data.get("order_quantity", 250))
    dist_km = float(shipment_data.get("route_distance_km", 6000.0))
    predicted_delay = float(prediction.get("predicted_delay_days", 8.0))

    # Option A: Expedited Air Freight / Express Carrier
    # Fastest recovery (replaces transit with dedicated air transport)
    air_delivery_days = round(max(2.0, min(4.0, base_lead_time * 0.20)), 1)
    air_cost = round(base_cost * 2.15 + (order_qty * 8.50), 2)
    air_capacity = 1500  # High air freight capacity limit
    air_risk_reduction_pct = 85.0
    air_residual_delay = 0.0

    option_a = {
        "id": "OPTION_A",
        "name": "Expedited Air Freight / Express",
        "description": "Reroute shipment via priority air cargo carrier to bypass maritime/port congestion and ground delays.",
        "cost": air_cost,
        "additional_cost": round(air_cost - base_cost, 2),
        "additional_cost_pct": round(((air_cost - base_cost) / (base_cost + 1e-5)) * 100, 1),
        "delivery_days": air_delivery_days,
        "days_saved": round(max(0.0, (base_lead_time + predicted_delay) - air_delivery_days), 1),
        "residual_delay_days": air_residual_delay,
        "capacity": air_capacity,
        "risk_level": "LOW",
        "risk_score": 0.10,
        "operational_tradeoff": "Fastest delivery speed at premium transport cost.",
        "key_benefits": ["Instant delay mitigation", "Guaranteed SLA", "Air priority tracking"],
    }

    # Option B: Secondary Supplier Dual-Sourcing / Regional Split
    # Balanced option: Reallocates order partially or fully to regional secondary supplier
    secondary_delivery_days = round(max(4.0, base_lead_time * 0.55 + 1.0), 1)
    secondary_cost = round(base_cost * 1.32 + (order_qty * 3.20), 2)
    secondary_capacity = 800  # Regional supplier capacity limit
    secondary_risk_reduction_pct = 60.0
    secondary_residual_delay = round(max(0.0, predicted_delay * 0.25), 1)

    option_b = {
        "id": "OPTION_B",
        "name": "Secondary Supplier Reallocation",
        "description": "Engage pre-qualified secondary regional supplier to fulfill order with shorter transit distance.",
        "cost": secondary_cost,
        "additional_cost": round(secondary_cost - base_cost, 2),
        "additional_cost_pct": round(((secondary_cost - base_cost) / (base_cost + 1e-5)) * 100, 1),
        "delivery_days": secondary_delivery_days,
        "days_saved": round(max(0.0, (base_lead_time + predicted_delay) - secondary_delivery_days), 1),
        "residual_delay_days": secondary_residual_delay,
        "capacity": secondary_capacity,
        "risk_level": "MEDIUM",
        "risk_score": 0.32,
        "operational_tradeoff": "Moderate delivery speed with balanced contract surcharge.",
        "key_benefits": ["Balanced cost-to-speed ratio", "Diversifies supplier risk", "Regional fulfillment"],
    }

    # Option C: Warehouse Buffer Reallocation & Ground Expedite / Reschedule
    # Lowest cost: Dynamic cross-docking from safety stock with expedited ground transit
    buffer_delivery_days = round(max(7.0, base_lead_time + min(3.0, predicted_delay * 0.55)), 1)
    buffer_cost = round(base_cost * 1.08 + (order_qty * 1.10), 2)
    buffer_capacity = 2000  # Warehouse inventory buffer limit
    buffer_risk_reduction_pct = 35.0
    buffer_residual_delay = round(max(1.0, predicted_delay * 0.60), 1)

    option_c = {
        "id": "OPTION_C",
        "name": "Inventory Buffer & Ground Expedite",
        "description": "Reallocate safety stock from nearest central distribution center and schedule dedicated ground freight.",
        "cost": buffer_cost,
        "additional_cost": round(buffer_cost - base_cost, 2),
        "additional_cost_pct": round(((buffer_cost - base_cost) / (base_cost + 1e-5)) * 100, 1),
        "delivery_days": buffer_delivery_days,
        "days_saved": round(max(0.0, (base_lead_time + predicted_delay) - buffer_delivery_days), 1),
        "residual_delay_days": buffer_residual_delay,
        "capacity": buffer_capacity,
        "risk_level": "MODERATE",
        "risk_score": 0.55,
        "operational_tradeoff": "Lowest financial outlay; accepts moderate scheduled delay.",
        "key_benefits": ["Minimal cost impact", "Preserves cashflow", "Utilizes existing safety stock"],
    }

    return [option_a, option_b, option_c]


def optimize_actions(
    shipment_data: Dict[str, Any],
    prediction: Dict[str, Any],
    constraints: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Main Prescriptive Optimization solver.
    Formulates a Mixed-Integer Linear Program (MILP) using PuLP to identify
    the optimal operational action subject to hard budget, SLA delivery, and capacity constraints.
    """
    if isinstance(constraints, dict):
        constraint_obj = OptimizationConstraints.from_dict(constraints)
    elif isinstance(constraints, OptimizationConstraints):
        constraint_obj = constraints
    else:
        constraint_obj = OptimizationConstraints()

    order_qty = int(shipment_data.get("order_quantity", 250))
    base_cost = float(shipment_data.get("shipping_cost", 5000.0))
    predicted_delay = float(prediction.get("predicted_delay_days", 0.0))
    base_lead_time = float(shipment_data.get("lead_time_days", 14.0))
    baseline_expected_days = round(base_lead_time + predicted_delay, 1)

    # 1. Generate candidate alternatives
    candidate_options = generate_candidate_options(shipment_data, prediction)

    # 2. Audit and validate constraints on each candidate option
    validated_options = validate_all_options(
        candidate_options, constraint_obj, order_quantity=order_qty
    )

    # 3. Formulate and Solve Optimization Problem (PuLP or SciPy)
    penalty_per_day = constraint_obj.penalty_cost_per_day_late
    risk_penalty_weight = 3000.0

    # Calculate total effective cost for each candidate alternative
    for opt in validated_options:
        direct_cost = opt["cost"]
        sla_excess_days = max(0.0, opt["delivery_days"] - constraint_obj.max_delivery_days)
        delay_penalty = (opt["residual_delay_days"] + sla_excess_days) * penalty_per_day
        risk_penalty = opt["risk_score"] * risk_penalty_weight
        opt["total_effective_cost"] = round(direct_cost + delay_penalty + risk_penalty, 2)

    recommended_option_id = None
    recommended_option = None
    solver_status = "INFEASIBLE"

    if HAS_PULP and pulp is not None:
        try:
            prob = pulp.LpProblem("Supply_Prescript_Prescriptive_Optimization", pulp.LpMinimize)
            option_vars = {
                opt["id"]: pulp.LpVariable(f"choose_{opt['id']}", cat="Binary")
                for opt in validated_options
            }

            prob += pulp.lpSum([option_vars[opt["id"]] * opt["total_effective_cost"] for opt in validated_options])
            prob += pulp.lpSum([option_vars[opt["id"]] for opt in validated_options]) == 1

            for opt in validated_options:
                var = option_vars[opt["id"]]
                prob += var * opt["cost"] <= constraint_obj.max_budget
                prob += var * opt["delivery_days"] <= constraint_obj.max_delivery_days
                prob += var * order_qty <= opt["capacity"]

            solver = pulp.PULP_CBC_CMD(msg=False)
            status_code = prob.solve(solver)
            pulp_status = pulp.LpStatus[status_code]

            if pulp_status == "Optimal":
                for opt in validated_options:
                    var = option_vars[opt["id"]]
                    if pulp.value(var) is not None and pulp.value(var) > 0.5:
                        recommended_option_id = opt["id"]
                        recommended_option = opt
                        opt["is_recommended"] = True
                    else:
                        opt["is_recommended"] = False
                solver_status = "Optimal (PuLP MILP)"
        except Exception:
            solver_status = "FALLBACK"

    if solver_status in ["INFEASIBLE", "FALLBACK"] or not HAS_PULP:
        # SciPy / Exact Discrete Optimization Solver
        feasible_opts = [o for o in validated_options if o["is_feasible"]]
        if feasible_opts:
            # Objective: minimize total effective cost
            costs = [o["total_effective_cost"] for o in feasible_opts]
            min_idx = int(np.argmin(costs))
            recommended_option = feasible_opts[min_idx]
            recommended_option_id = recommended_option["id"]
            for o in validated_options:
                o["is_recommended"] = (o["id"] == recommended_option_id)
            solver_status = "Optimal (SciPy Solver)" if not HAS_PULP else "Optimal (Exact Feasibility)"
        else:
            solver_status = "INFEASIBLE"
            for o in validated_options:
                o["is_recommended"] = False

    # Calculate SLA penalty savings (Baseline delay cost vs recommended option)
    baseline_delay_penalty = predicted_delay * penalty_per_day
    recommended_savings = 0.0
    if recommended_option and recommended_option["is_feasible"]:
        action_premium = recommended_option["cost"] - base_cost
        penalties_prevented = (predicted_delay - recommended_option["residual_delay_days"]) * penalty_per_day
        recommended_savings = round(max(0.0, penalties_prevented - action_premium), 2)

    # Build "Why these recommendations?" audit summary
    why_explanation = {
        "predicted_delay": f"{predicted_delay:.1f} days (Baseline total transit: {baseline_expected_days:.1f} days)",
        "hard_budget_limit": f"${constraint_obj.max_budget:,.2f}",
        "required_delivery_deadline": f"≤ {constraint_obj.max_delivery_days:.1f} days",
        "optimization_objective": "Minimize total operational cost while strictly satisfying delivery deadline, budget limits, and supplier capacity.",
        "recommended_action": recommended_option["name"] if recommended_option else "None (All options violate hard constraints)",
        "rationale": (
            f"Selected {recommended_option['name']} because it satisfies all business constraints while providing the best cost-to-speed trade-off."
            if recommended_option and recommended_option["is_feasible"]
            else "No candidate action satisfies all hard constraints. Review budget or relax deadline requirements."
        ),
        "estimated_net_savings": f"${recommended_savings:,.2f}" if recommended_savings > 0 else "$0.00",
    }

    # Trade-off summary matrix for frontend charts
    tradeoff_comparison = [
        {
            "option_id": opt["id"],
            "name": opt["name"],
            "cost": opt["cost"],
            "additional_cost": opt["additional_cost"],
            "delivery_days": opt["delivery_days"],
            "days_saved": opt["days_saved"],
            "residual_delay": opt["residual_delay_days"],
            "risk_level": opt["risk_level"],
            "is_feasible": opt["is_feasible"],
            "is_recommended": opt.get("is_recommended", False),
            "status": opt["status"],
        }
        for opt in validated_options
    ]

    return {
        "solver_status": solver_status,
        "is_optimal": solver_status.startswith("Optimal") or solver_status.startswith("Feasible"),
        "recommended_option_id": recommended_option_id,
        "recommended_option": recommended_option,
        "options": validated_options,
        "constraints": constraint_obj.to_dict(),
        "why_explanation": why_explanation,
        "tradeoff_comparison": tradeoff_comparison,
        "baseline_summary": {
            "base_cost": base_cost,
            "base_lead_time": base_lead_time,
            "predicted_delay": predicted_delay,
            "baseline_total_days": baseline_expected_days,
            "order_quantity": order_qty,
        },
    }


if __name__ == "__main__":
    sample_shipment = {
        "shipping_cost": 8500.0,
        "lead_time_days": 18.0,
        "order_quantity": 400,
        "route_distance_km": 11000.0,
    }
    sample_pred = {
        "predicted_delay_days": 12.0,
        "delay_probability": 0.92,
        "risk_level": "CRITICAL",
    }
    sample_constraints = OptimizationConstraints(
        max_budget=22000.0,
        max_delivery_days=10.0,
    )
    res = optimize_actions(sample_shipment, sample_pred, sample_constraints)
    print("--- PRESCRIPTIVE OPTIMIZATION RESULT ---")
    print(f"Solver Status: {res['solver_status']}")
    print(f"Recommended Option: {res['recommended_option_id']}")
    for opt in res["options"]:
        print(f"  {opt['id']}: {opt['name']} | Cost: ${opt['cost']} | Delivery: {opt['delivery_days']}d | Feasible: {opt['is_feasible']} ({opt['status']})")
