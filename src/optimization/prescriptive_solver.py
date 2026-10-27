from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "supply_prescript_processed_dataset.csv"
)

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
    Evaluate three supply-chain actions using budget
    and maximum acceptable delay constraints.
    """

    options = [
        {
            "action": "Regular supply",
            "cost": float(baseline_cost),
            "delay": float(baseline_delay_days),
        },
        {
            "action": "Air freight",
            "cost": float(air_freight_cost),
            "delay": float(air_freight_delay_days),
        },
        {
            "action": "Secondary supplier",
            "cost": float(secondary_supplier_cost),
            "delay": float(secondary_supplier_delay_days),
        },
    ]

    # Evaluate every action against business constraints.
    for option in options:
        option["feasible"] = (
            option["cost"] <= budget
            and option["delay"] <= max_acceptable_delay
        )

        option["cost_saving_vs_air_freight"] = round(
            float(air_freight_cost) - option["cost"],
            2,
        )

        option["days_saved_vs_regular"] = round(
            float(baseline_delay_days) - option["delay"],
            2,
        )

        option["cost"] = round(option["cost"], 2)
        option["delay"] = round(option["delay"], 2)
        option["delay_days"] = option["delay"]
        option["recommended"] = False

        if option["cost"] > budget:
            option["reason"] = "Exceeds the available budget."
        elif option["delay"] > max_acceptable_delay:
            option["reason"] = (
                "Exceeds the maximum acceptable delay."
            )
        else:
            option["reason"] = (
                "Meets the budget and delay constraints."
            )

    # Sort feasible options by cost and then delay.
    feasible_options = sorted(
        [
            option
            for option in options
            if option["feasible"]
        ],
        key=lambda option: (
            option["cost"],
            option["delay"],
        ),
    )

    # Return all three options even when none is feasible.
    if not feasible_options:
        return {
            "status": "no_feasible_solution",
            "message": (
                "No action satisfies both the budget and "
                "maximum acceptable delay constraints."
            ),
            "recommended_action": None,
            "budget": round(float(budget), 2),
            "max_acceptable_delay_days": round(
                float(max_acceptable_delay),
                2,
            ),
            "alternatives": options,
            "optimization_success": False,
        }

    # Recommend the lowest-cost feasible action.
    best = feasible_options[0]
    best["recommended"] = True

    return {
        "status": "success",
        "recommended_action": best["action"],
        "estimated_cost": best["cost"],
        "estimated_delay_days": best["delay"],
        "budget": round(float(budget), 2),
        "max_acceptable_delay_days": round(
            float(max_acceptable_delay),
            2,
        ),
        "alternatives": options,
        "optimization_success": True,
    }


def main():
    print("Loading supply-chain data...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)

    if data.empty:
        raise ValueError(
            "The processed supply-chain dataset is empty."
        )

    required_columns = [
        "estimated_primary_cost_usd",
        "estimated_secondary_cost_usd",
        "air_freight_cost_usd",
        "planned_lead_time_days",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required dataset columns: {missing_columns}"
        )

    row = data.iloc[-1]

    baseline_cost = float(
        row["estimated_primary_cost_usd"]
    )

    secondary_cost = float(
        row["estimated_secondary_cost_usd"]
    )

    air_cost = float(
        row["air_freight_cost_usd"]
    )

    baseline_delay = max(
        float(row["planned_lead_time_days"]),
        1,
    )

    air_delay = max(
        1,
        baseline_delay * 0.25,
    )

    secondary_delay = max(
        1,
        baseline_delay * 0.60,
    )

    # Example business constraints.
    budget = max(
        air_cost,
        secondary_cost,
        baseline_cost,
    ) * 1.10

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

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            recommendation,
            indent=4,
        ),
        encoding="utf-8",
    )

    print("\n=== PRESCRIPTIVE RECOMMENDATIONS ===")

    print(
        json.dumps(
            recommendation,
            indent=4,
        )
    )

    print(
        f"\nSaved recommendations to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()