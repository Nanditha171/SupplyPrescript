"""
Supply Prescript - Optimization Audit & Feasibility Validator
Validates candidate prescriptive actions against hard and soft business constraints.
Author: Prasanna (ML & Optimization Lead)
"""

from typing import Dict, Any, List, Tuple
from optimization.constraints import OptimizationConstraints


def audit_option(
    option: Dict[str, Any],
    constraints: OptimizationConstraints,
    order_quantity: int = 100,
) -> Dict[str, Any]:
    """
    Performs rigorous audit of a single prescriptive mitigation option.
    
    Checks:
    1. Budget Constraint: option_cost <= max_budget
    2. Delivery SLA Constraint: delivery_days <= max_delivery_days
    3. Capacity Constraint: order_quantity <= available_capacity
    
    Returns audit details with boolean checkmarks and violation explanations.
    """
    cost = float(option.get("cost", 0.0))
    delivery_days = float(option.get("delivery_days", 0.0))
    capacity = int(option.get("capacity", 999999))

    violations: List[str] = []
    
    # 1. Budget check
    budget_satisfied = cost <= constraints.max_budget
    if not budget_satisfied:
        violations.append(
            f"Violates Hard Budget: Action Cost (${cost:,.2f}) exceeds Maximum Budget (${constraints.max_budget:,.2f})"
        )

    # 2. Delivery SLA check
    delivery_satisfied = delivery_days <= constraints.max_delivery_days
    if not delivery_satisfied:
        violations.append(
            f"Violates Delivery SLA: Expected Delivery ({delivery_days:.1f} days) exceeds Required Deadline ({constraints.max_delivery_days:.1f} days)"
        )

    # 3. Capacity check
    capacity_satisfied = order_quantity <= capacity
    if not capacity_satisfied:
        violations.append(
            f"Violates Capacity: Order Quantity ({order_quantity} units) exceeds Carrier/Supplier Capacity ({capacity} units)"
        )

    is_feasible = budget_satisfied and delivery_satisfied and capacity_satisfied
    status = "FEASIBLE" if is_feasible else "INFEASIBLE"

    return {
        "status": status,
        "is_feasible": is_feasible,
        "constraint_checks": {
            "budget_satisfied": budget_satisfied,
            "delivery_satisfied": delivery_satisfied,
            "capacity_satisfied": capacity_satisfied,
        },
        "violations": violations,
        "audit_summary": "All constraints satisfied" if is_feasible else " | ".join(violations),
    }


def validate_all_options(
    options: List[Dict[str, Any]],
    constraints: OptimizationConstraints,
    order_quantity: int = 100,
) -> List[Dict[str, Any]]:
    """Audits all options in a candidate set."""
    validated = []
    for opt in options:
        opt_copy = dict(opt)
        audit = audit_option(opt_copy, constraints, order_quantity=order_quantity)
        opt_copy["status"] = audit["status"]
        opt_copy["is_feasible"] = audit["is_feasible"]
        opt_copy["constraint_checks"] = audit["constraint_checks"]
        opt_copy["violations"] = audit["violations"]
        opt_copy["audit_summary"] = audit["audit_summary"]
        validated.append(opt_copy)
    return validated
