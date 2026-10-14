from pathlib import Path

import joblib
import pandas as pd


# Locate the project directory
ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "supply_delay_model.joblib"

# Load the trained model pipeline
model = joblib.load(MODEL_PATH)


# Features used when training the model
NUMERIC_FEATURES = [
    "demand_units",
    "inventory_units",
    "supplier_capacity_units",
    "supplier_reliability",
    "planned_lead_time_days",
    "unit_cost_usd",
    "transport_cost_usd",
    "secondary_supplier_premium_pct",
    "inventory_coverage_ratio",
    "inventory_gap_units",
    "capacity_utilization",
    "cost_per_demand_unit",
    "secondary_cost_difference",
    "air_freight_premium",
    "supplier_risk_score",
    "demand_pressure",
    "shortage_flag",
    "previous_delay_days",
    "previous_delay_flag",
    "rolling_average_delay",
    "rolling_average_demand",
    "rolling_average_inventory",
    "demand_outlier_flag",
    "inventory_outlier_flag",
    "cost_outlier_flag",
    "lead_time_outlier_flag",
]

CATEGORICAL_FEATURES = [
    "supplier_id",
    "supplier_country",
    "product_id",
    "category",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def predict_delay(shipment_data: dict) -> dict:
    """Predict whether a shipment will be delayed."""

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in shipment_data
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    # Select only the model's expected input features
    df = pd.DataFrame([
        {feature: shipment_data[feature] for feature in FEATURES}
    ])

    # Generate prediction and probability
    probability = float(model.predict_proba(df)[0][1])
    prediction = int(model.predict(df)[0])

    return {
        "prediction": "DELAY" if prediction == 1 else "NO DELAY",
        "delay_probability": round(probability, 4),
        "delay_probability_percent": round(probability * 100, 2),
    }