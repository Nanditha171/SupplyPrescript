"""
Supply Prescript — Synthetic Supply Chain Dataset Generator (Day 2)
===================================================================
Generates a realistic, structured supply-chain operational dataset representing
weekly replenishment planning cycles across 8 products and 8 global suppliers
over an 18-month historical timeframe (78 weeks = 624 records).
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

# =====================================================================
# Configuration & Constants
# =====================================================================
RANDOM_SEED = 42
WEEKS_COUNT = 78  # 78 weekly planning cycles = ~18 months

# Risk classification thresholds (configurable)
HIGH_RISK_RATIO_THRESHOLD = 0.70
MED_RISK_RATIO_THRESHOLD = 1.00
MED_RISK_RELIABILITY_THRESHOLD = 0.93

# Supplier Definitions (Country-specific reliability and lead times)
SUPPLIERS = [
    {
        "supplier_id": "SUP001",
        "supplier_name": "Alpha Components",
        "supplier_country": "India",
        "supplier_reliability": 0.96,
        "lead_time_range": (2, 7),
        "capacity_range": (1500, 2400),
    },
    {
        "supplier_id": "SUP002",
        "supplier_name": "Beta Electronics",
        "supplier_country": "Vietnam",
        "supplier_reliability": 0.92,
        "lead_time_range": (6, 14),
        "capacity_range": (1200, 2500),
    },
    {
        "supplier_id": "SUP003",
        "supplier_name": "Gamma Semiconductors",
        "supplier_country": "Taiwan",
        "supplier_reliability": 0.89,
        "lead_time_range": (10, 16),
        "capacity_range": (1000, 2600),
    },
    {
        "supplier_id": "SUP004",
        "supplier_name": "Delta Manufacturing",
        "supplier_country": "China",
        "supplier_reliability": 0.94,
        "lead_time_range": (6, 14),
        "capacity_range": (1200, 2450),
    },
    {
        "supplier_id": "SUP005",
        "supplier_name": "Epsilon Tech",
        "supplier_country": "Malaysia",
        "supplier_reliability": 0.91,
        "lead_time_range": (4, 14),
        "capacity_range": (1000, 2450),
    },
    {
        "supplier_id": "SUP006",
        "supplier_name": "Zeta Industrial",
        "supplier_country": "South Korea",
        "supplier_reliability": 0.97,
        "lead_time_range": (6, 13),
        "capacity_range": (1100, 2400),
    },
    {
        "supplier_id": "SUP007",
        "supplier_name": "Eta Systems",
        "supplier_country": "India",
        "supplier_reliability": 0.95,
        "lead_time_range": (2, 7),
        "capacity_range": (1200, 2600),
    },
    {
        "supplier_id": "SUP008",
        "supplier_name": "Theta Components",
        "supplier_country": "Singapore",
        "supplier_reliability": 0.93,
        "lead_time_range": (2, 9),
        "capacity_range": (1100, 2600),
    },
]

# Product Definitions (8 core supply chain materials)
PRODUCTS = [
    {"product_id": "P001", "product_name": "Microchips", "category": "Electronics", "demand_range": (800, 1650)},
    {"product_id": "P002", "product_name": "Processors", "category": "Electronics", "demand_range": (550, 1200)},
    {"product_id": "P003", "product_name": "Memory Modules", "category": "Electronics", "demand_range": (800, 1400)},
    {"product_id": "P004", "product_name": "Display Panels", "category": "Electronics", "demand_range": (400, 950)},
    {"product_id": "P005", "product_name": "Battery Cells", "category": "Energy", "demand_range": (400, 1100)},
    {"product_id": "P006", "product_name": "Sensors", "category": "Electronics", "demand_range": (700, 1300)},
    {"product_id": "P007", "product_name": "Power Modules", "category": "Electronics", "demand_range": (500, 980)},
    {"product_id": "P008", "product_name": "Circuit Boards", "category": "Electronics", "demand_range": (800, 1850)},
]


def generate_supply_chain_dataset(
    num_weeks: int = WEEKS_COUNT,
    start_date: str = "2025-01-01",
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generates a realistic, weekly observation dataset containing 624 observations (78 weeks x 8 products).
    """
    np.random.seed(seed)

    weekly_dates = pd.date_range(start=start_date, periods=num_weeks, freq="7D")
    records = []

    for week_idx, current_dt in enumerate(weekly_dates):
        date_str = current_dt.strftime("%Y-%m-%d")

        # Shuffle supplier assignment per product each week for realistic multi-sourcing
        supplier_indices = np.random.permutation(len(SUPPLIERS))

        for prod_idx, product in enumerate(PRODUCTS):
            supplier = SUPPLIERS[supplier_indices[prod_idx]]

            # Demand and Capacity
            demand_units = int(np.random.randint(product["demand_range"][0], product["demand_range"][1] + 1))
            supplier_capacity_units = int(
                np.random.randint(
                    max(supplier["capacity_range"][0], demand_units - 100),
                    max(supplier["capacity_range"][1], demand_units + 800) + 1,
                )
            )

            # Inventory distribution with occasional crunches
            inv_dist_type = np.random.choice(["crunch", "moderate", "healthy"], p=[0.14, 0.32, 0.54])
            if inv_dist_type == "crunch":
                inv_factor = np.random.uniform(0.32, 0.69)
            elif inv_dist_type == "moderate":
                inv_factor = np.random.uniform(0.70, 0.99)
            else:
                inv_factor = np.random.uniform(1.00, 2.75)

            inventory_units = int(max(50, round(demand_units * inv_factor)))
            inventory_demand_ratio = round(inventory_units / demand_units, 3)

            # Planned lead time
            lead_min, lead_max = supplier["lead_time_range"]
            planned_lead_time_days = int(np.random.randint(lead_min, lead_max + 1))

            # Calibrated delay probability (~8-10% delay rate)
            rel_gap = 1.0 - supplier["supplier_reliability"]
            cap_util = demand_units / supplier_capacity_units
            lead_norm = planned_lead_time_days / 15.0

            logit = -4.10 + (13.5 * rel_gap) + (1.20 * cap_util) + (0.60 * lead_norm) + (0.35 if inventory_demand_ratio < 0.70 else 0.0)
            delay_prob = 1.0 / (1.0 + np.exp(-logit))
            delay_prob = float(np.clip(delay_prob, 0.015, 0.35))

            delay_occurred = int(np.random.rand() < delay_prob)
            if delay_occurred == 1:
                delay_days = int(np.random.randint(3, 16))
            else:
                delay_days = 0

            actual_lead_time_days = planned_lead_time_days + delay_days

            # Unit and Transportation costs
            unit_cost_usd = round(float(np.random.uniform(8.0, 150.0)), 2)
            transport_cost_usd = round(float(np.random.uniform(500.0, 5000.0)), 2)
            secondary_supplier_premium_pct = round(float(np.random.uniform(0.05, 0.20)), 3)

            # Estimated costs
            estimated_primary_cost_usd = round((demand_units * unit_cost_usd) + transport_cost_usd, 2)
            secondary_procurement = demand_units * unit_cost_usd * (1.0 + secondary_supplier_premium_pct)
            secondary_transport = transport_cost_usd * (1.0 + (0.5 * secondary_supplier_premium_pct))
            estimated_secondary_cost_usd = round(secondary_procurement + secondary_transport, 2)

            # Air Freight cost (expedited recovery)
            air_freight_cost_usd = round(
                (transport_cost_usd * float(np.random.uniform(1.8, 4.2))) + (demand_units * float(np.random.uniform(1.0, 2.5))),
                2,
            )

            # Service Risk classification
            if delay_occurred == 1 or inventory_demand_ratio < HIGH_RISK_RATIO_THRESHOLD:
                service_risk = "High"
            elif (
                inventory_demand_ratio < MED_RISK_RATIO_THRESHOLD
                or supplier["supplier_reliability"] < MED_RISK_RELIABILITY_THRESHOLD
            ):
                service_risk = "Medium"
            else:
                service_risk = "Low"

            records.append(
                {
                    "date": date_str,
                    "supplier_id": supplier["supplier_id"],
                    "supplier_name": supplier["supplier_name"],
                    "supplier_country": supplier["supplier_country"],
                    "supplier_reliability": supplier["supplier_reliability"],
                    "product_id": product["product_id"],
                    "product_name": product["product_name"],
                    "category": product["category"],
                    "demand_units": demand_units,
                    "inventory_units": inventory_units,
                    "supplier_capacity_units": supplier_capacity_units,
                    "planned_lead_time_days": planned_lead_time_days,
                    "actual_lead_time_days": actual_lead_time_days,
                    "delay_occurred": delay_occurred,
                    "delay_days": delay_days,
                    "unit_cost_usd": unit_cost_usd,
                    "transport_cost_usd": transport_cost_usd,
                    "secondary_supplier_premium_pct": secondary_supplier_premium_pct,
                    "inventory_demand_ratio": inventory_demand_ratio,
                    "estimated_primary_cost_usd": estimated_primary_cost_usd,
                    "estimated_secondary_cost_usd": estimated_secondary_cost_usd,
                    "air_freight_cost_usd": air_freight_cost_usd,
                    "service_risk": service_risk,
                }
            )

    return pd.DataFrame(records)


def create_data_dictionary() -> pd.DataFrame:
    """
    Creates the standardized data dictionary DataFrame for Supply Prescript.
    """
    dictionary_entries = [
        {"column": "date", "description": "Observation date", "data_type": "String", "role": "Input"},
        {"column": "supplier_id", "description": "Unique supplier ID", "data_type": "String", "role": "Input"},
        {"column": "supplier_name", "description": "Supplier name", "data_type": "String", "role": "Input"},
        {"column": "supplier_country", "description": "Supplier country", "data_type": "String", "role": "Input"},
        {"column": "supplier_reliability", "description": "Historical supplier reliability", "data_type": "Float", "role": "Input"},
        {"column": "product_id", "description": "Unique product ID", "data_type": "String", "role": "Input"},
        {"column": "product_name", "description": "Product/material name", "data_type": "String", "role": "Input"},
        {"column": "category", "description": "Product category", "data_type": "String", "role": "Input"},
        {"column": "demand_units", "description": "Expected demand quantity", "data_type": "Integer", "role": "Input"},
        {"column": "inventory_units", "description": "Available inventory quantity", "data_type": "Integer", "role": "Input"},
        {"column": "supplier_capacity_units", "description": "Supplier capacity", "data_type": "Integer", "role": "Input"},
        {"column": "planned_lead_time_days", "description": "Expected lead time", "data_type": "Integer", "role": "Input"},
        {"column": "actual_lead_time_days", "description": "Observed lead time", "data_type": "Integer", "role": "Outcome"},
        {"column": "delay_occurred", "description": "Delay flag (0/1)", "data_type": "Integer", "role": "Target"},
        {"column": "delay_days", "description": "Number of delayed days", "data_type": "Integer", "role": "Target"},
        {"column": "unit_cost_usd", "description": "Procurement cost per unit", "data_type": "Float", "role": "Input"},
        {"column": "transport_cost_usd", "description": "Standard transportation cost", "data_type": "Float", "role": "Input"},
        {"column": "secondary_supplier_premium_pct", "description": "Secondary supplier premium", "data_type": "Float", "role": "Input"},
        {"column": "inventory_demand_ratio", "description": "Inventory divided by demand", "data_type": "Float", "role": "Derived"},
        {"column": "estimated_primary_cost_usd", "description": "Estimated primary sourcing cost", "data_type": "Float", "role": "Derived"},
        {"column": "estimated_secondary_cost_usd", "description": "Estimated secondary sourcing cost", "data_type": "Float", "role": "Derived"},
        {"column": "air_freight_cost_usd", "description": "Estimated expedited air-freight cost", "data_type": "Float", "role": "Derived"},
        {"column": "service_risk", "description": "Operational risk classification", "data_type": "String", "role": "Derived"},
    ]
    return pd.DataFrame(dictionary_entries)


def print_dataset_summary(df: pd.DataFrame) -> None:
    """
    Prints a rich operational summary of the generated dataset.
    """
    total_records = len(df)
    delayed_records = (df["delay_occurred"] == 1).sum()
    delay_pct = (delayed_records / total_records) * 100
    avg_delay_days = df[df["delay_occurred"] == 1]["delay_days"].mean()
    risk_counts = df["service_risk"].value_counts().to_dict()

    print("\n" + "=" * 60)
    print("       SUPPLY PRESCRIPT -- DATASET GENERATION SUMMARY")
    print("=" * 60)
    print(f"Number of records             : {total_records}")
    print(f"Number of suppliers           : {df['supplier_id'].nunique()} ({', '.join(df['supplier_country'].unique())})")
    print(f"Number of products            : {df['product_id'].nunique()} across {df['category'].nunique()} categories")
    print(f"Date range                    : {df['date'].min()} to {df['date'].max()}")
    print(f"Number of delayed shipments   : {delayed_records}")
    print(f"Delay percentage              : {delay_pct:.2f}%")
    print(f"Average demand (units)        : {df['demand_units'].mean():.1f}")
    print(f"Average inventory (units)     : {df['inventory_units'].mean():.1f}")
    print(f"Average lead time (days)      : {df['actual_lead_time_days'].mean():.2f} (Planned: {df['planned_lead_time_days'].mean():.2f})")
    print(f"Average delay duration (days) : {avg_delay_days:.2f} days (when delayed)")
    print(f"Average procurement cost      : ${df['unit_cost_usd'].mean():.2f} / unit")
    print(f"Average transportation cost   : ${df['transport_cost_usd'].mean():.2f} / shipment")
    print(f"Average air freight cost      : ${df['air_freight_cost_usd'].mean():.2f} / shipment")
    print("-" * 60)
    print("Risk Tier Distribution:")
    for risk_tier in ["High", "Medium", "Low"]:
        count = risk_counts.get(risk_tier, 0)
        pct = (count / total_records) * 100
        print(f"  * {risk_tier:<8}: {count:>4} records ({pct:>5.1f}%)")
    print("=" * 60 + "\n")


def main():
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = base_dir / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = raw_dir / "supply_prescript_raw_dataset.csv"
    dict_path = data_dir / "supply_prescript_data_dictionary.csv"

    print("Generating synthetic supply chain dataset (Day 2)...")
    df = generate_supply_chain_dataset(num_weeks=WEEKS_COUNT)

    print(f"Saving raw dataset to: {dataset_path}")
    df.to_csv(dataset_path, index=False)

    print("Generating data dictionary...")
    dict_df = create_data_dictionary()
    print(f"Saving data dictionary to: {dict_path}")
    dict_df.to_csv(dict_path, index=False)

    print_dataset_summary(df)


if __name__ == "__main__":
    main()
