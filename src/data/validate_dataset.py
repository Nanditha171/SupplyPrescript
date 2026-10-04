"""
Supply Prescript — Dataset Validation Suite (Day 2)
==================================================
Performs exhaustive structural, range, logical, and referential validation
on the generated raw supply-chain dataset before ingestion into downstream pipelines.
"""

import os
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = [
    "date",
    "supplier_id",
    "supplier_name",
    "supplier_country",
    "supplier_reliability",
    "product_id",
    "product_name",
    "category",
    "demand_units",
    "inventory_units",
    "supplier_capacity_units",
    "planned_lead_time_days",
    "actual_lead_time_days",
    "delay_occurred",
    "delay_days",
    "unit_cost_usd",
    "transport_cost_usd",
    "secondary_supplier_premium_pct",
    "inventory_demand_ratio",
    "estimated_primary_cost_usd",
    "estimated_secondary_cost_usd",
    "air_freight_cost_usd",
    "service_risk",
]

ALLOWED_RISK_TIERS = {"Low", "Medium", "High"}
ALLOWED_CATEGORIES = {"Electronics", "Energy", "Components"}


def validate_supply_chain_dataset(dataset_path: Path) -> bool:
    """
    Validates the integrity, schema, logical consistency, and statistical boundaries of the dataset.
    Returns True if valid, False otherwise.
    """
    if not dataset_path.exists():
        print(f"Error: Dataset file not found at {dataset_path}")
        return False

    df = pd.read_csv(dataset_path)
    total_records = len(df)
    total_columns = len(df.columns)

    # 1. Column presence check
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        print(f"Validation Failure: Missing columns: {missing_cols}")
        return False

    # 2. Missing values check
    null_counts = int(df[REQUIRED_COLUMNS].isnull().sum().sum())

    # 3. Duplicate observations check
    duplicate_count = int(df.duplicated().sum())

    # 4. Numeric Range Validations
    invalid_reliability = int((
        (df["supplier_reliability"] < 0.0) | (df["supplier_reliability"] > 1.0)
    ).sum())

    invalid_costs = int((
        (df["unit_cost_usd"] <= 0)
        | (df["transport_cost_usd"] <= 0)
        | (df["air_freight_cost_usd"] <= 0)
        | (df["estimated_primary_cost_usd"] <= 0)
        | (df["estimated_secondary_cost_usd"] <= 0)
        | (df["secondary_supplier_premium_pct"] < 0.0)
        | (df["secondary_supplier_premium_pct"] > 1.0)
    ).sum())

    invalid_lead_times = int((
        (df["planned_lead_time_days"] <= 0)
        | (df["actual_lead_time_days"] < df["planned_lead_time_days"])
        | (df["actual_lead_time_days"] != (df["planned_lead_time_days"] + df["delay_days"]))
    ).sum())

    invalid_delay_records = int((
        ((df["delay_occurred"] == 0) & (df["delay_days"] != 0))
        | ((df["delay_occurred"] == 1) & (df["delay_days"] <= 0))
        | (~df["delay_occurred"].isin([0, 1]))
    ).sum())

    invalid_demand_inventory = int((
        (df["demand_units"] <= 0)
        | (df["inventory_units"] < 0)
        | (df["supplier_capacity_units"] <= 0)
    ).sum())

    invalid_categories = int((~df["category"].isin(ALLOWED_CATEGORIES)).sum())
    invalid_risks = int((~df["service_risk"].isin(ALLOWED_RISK_TIERS)).sum())

    # Metrics for report
    delay_rate = (df["delay_occurred"].sum() / total_records) * 100 if total_records > 0 else 0
    risk_counts = df["service_risk"].value_counts().to_dict()
    high_risk_count = risk_counts.get("High", 0)
    med_risk_count = risk_counts.get("Medium", 0)
    low_risk_count = risk_counts.get("Low", 0)

    # Determine overall status
    is_valid = (
        total_records >= 500
        and null_counts == 0
        and duplicate_count == 0
        and invalid_reliability == 0
        and invalid_costs == 0
        and invalid_lead_times == 0
        and invalid_delay_records == 0
        and invalid_demand_inventory == 0
        and invalid_categories == 0
        and invalid_risks == 0
    )

    # Print Validation Report
    print("=" * 40)
    print("SUPPLY PRESCRIPT DATA VALIDATION")
    print("=" * 40)
    print(f"Records                  : {total_records}")
    print(f"Columns                  : {total_columns}")
    print(f"Missing Values           : {null_counts}")
    print(f"Duplicate Records        : {duplicate_count}")
    print(f"Invalid Reliability      : {invalid_reliability}")
    print(f"Invalid Costs            : {invalid_costs}")
    print(f"Invalid Lead Times       : {invalid_lead_times}")
    print(f"Invalid Delay Records    : {invalid_delay_records}")
    print(f"Invalid Demand/Inv       : {invalid_demand_inventory}")
    print(f"Invalid Categories/Risks : {invalid_categories + invalid_risks}")
    print()
    print(f"Delay Rate               : {delay_rate:.1f}%")
    print(f"High Risk Records        : {high_risk_count}")
    print(f"Medium Risk Records      : {med_risk_count}")
    print(f"Low Risk Records         : {low_risk_count}")
    print()
    print(f"STATUS: {'DATASET VALID' if is_valid else 'DATASET INVALID'}")
    print("=" * 40)

    return is_valid


def main():
    base_dir = Path(__file__).resolve().parent.parent.parent
    dataset_path = base_dir / "data" / "raw" / "supply_prescript_raw_dataset.csv"
    validate_supply_chain_dataset(dataset_path)


if __name__ == "__main__":
    main()
