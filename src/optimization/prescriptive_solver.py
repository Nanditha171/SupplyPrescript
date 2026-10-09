
from pathlib import Path
import json

import pandas as pd
from scipy.optimize import linprog


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "supply_prescript_processed_dataset.csv"
OUTPUT_PATH = ROOT / "reports" / "prescriptive_recommendations.json"


def recommend_actions(
    baseline_cost,
    air_freight_cost,
    secondary_supplier_cost,
    baseline_delay_days,
    air_freight_delay_days,
    secondary_supplier_delay_days,
    budget,
    max_acceptable_delay,
):
    """
    Select the lowest-cost feasible option.

    Decision variables:
    x0 = choose regular supply
    x1 = choose air freight
    x2 = choose secondary supplier

    Each option is represented by a binary-like decision variable.
    The constraints require exactly one option to be selected.
    """

    costs = [
        baseline_cost,
        air_freight_cost,
        secondary_supplier_cost,
    ]

    delays = [
        baseline_delay_days,
        air_freight_delay_days,
        secondary_supplier_delay_days,
    ]

    # One option must be selected.
    A_eq = [[1, 1, 1]]
    b_eq = [1]

    # Budget and maximum acceptable delay constraints.
    A_ub = [
        costs,
        delays,
    ]
    b_ub = [
        budget,
        max_acceptable_delay,
    ]

    result = linprog(
        c=costs,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=[(0, 1), (0, 1), (0, 1)],
        method="highs",
        integrality=[1, 1, 1],
    )

    options = ["Regular supply", "Air freight", "Secondary supplier"]

    if not result.success:
        return {
            "status": "no_feasible_solution",
            "message": result.message,
        }

    selected_index = int(round(result.x.argmax()))

    return {
        "status": "success",
        "recommended_action": options[selected_index],
        "estimated_cost": round(float(costs[selected_index]), 2),
        "estimated_delay_days": delays[selected_index],
        "budget": budget,
        "max_acceptable_delay_days": max_acceptable_delay,
    }


def main():
    print("Loading supply-chain data...")
    data = pd.read_csv(DATA_PATH)

    # Use the latest record as a demonstration scenario.
    row = data.iloc[-1]

    baseline_cost = float(row["estimated_primary_cost_usd"])
    secondary_cost = float(row["estimated_secondary_cost_usd"])
    air_cost = float(row["air_freight_cost_usd"])

    baseline_delay = max(float(row["planned_lead_time_days"]), 1)
    air_delay = max(1, baseline_delay * 0.25)
    secondary_delay = max(1, baseline_delay * 0.60)

    budget = max(air_cost, secondary_cost, baseline_cost) * 1.10
    max_delay = baseline_delay * 0.70

    recommendation = recommend_actions(
        baseline_cost=baseline_cost,
        air_freight_cost=air_cost,
        secondary_supplier_cost=secondary_cost,
        baseline_delay_days=baseline_delay,
        air_freight_delay_days=air_delay,
        secondary_supplier_delay_days=secondary_delay,
        budget=budget,
        max_acceptable_delay=max_delay,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(recommendation, indent=4),
        encoding="utf-8",
    )

    print("\n=== PRESCRIPTIVE RECOMMENDATION ===")
    print(json.dumps(recommendation, indent=4))
    print(f"\nSaved recommendation to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()