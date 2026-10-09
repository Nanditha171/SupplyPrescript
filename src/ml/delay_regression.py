
from pathlib import Path

import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.compose import make_column_selector
from sklearn import set_config
from sklearn.pipeline import make_pipeline
import numpy as np
import joblib
import json

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "supply_prescript_processed_dataset.csv"
MODEL_PATH = ROOT / "models" / "delay_duration_model.joblib"
REPORT_PATH = ROOT / "reports" / "delay_regression_metrics.json"

NUMERIC_FEATURES = [
    "demand_units", "inventory_units", "supplier_capacity_units",
    "supplier_reliability", "planned_lead_time_days", "unit_cost_usd",
    "transport_cost_usd", "secondary_supplier_premium_pct",
    "inventory_coverage_ratio", "inventory_gap_units", "capacity_utilization",
    "cost_per_demand_unit", "secondary_cost_difference", "air_freight_premium",
    "supplier_risk_score", "demand_pressure", "shortage_flag",
    "previous_delay_days", "previous_delay_flag", "rolling_average_delay",
    "rolling_average_demand", "rolling_average_inventory",
    "demand_outlier_flag", "inventory_outlier_flag", "cost_outlier_flag",
    "lead_time_outlier_flag"
]

CATEGORICAL_FEATURES = [
    "supplier_id", "supplier_country", "product_id", "category"
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "delay_days"


def main():
    print("Loading processed supply-chain data...")
    data = pd.read_csv(DATA_PATH)

    # Duration prediction is trained on delayed shipments only.
    delayed = data[data[TARGET] > 0].copy()
    X = delayed[FEATURES]
    y = delayed[TARGET].astype(float)

    # Chronological split: earlier shipments train, later shipments test.
    split_index = int(len(delayed) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        ))
    ])

    print(f"Delayed shipments: {len(delayed)}")
    print(f"Training rows: {len(X_train)} | Testing rows: {len(X_test)}")
    print("Training XGBoost regression model...")

    model.fit(X_train, y_train)
    predictions = np.maximum(model.predict(X_test), 0)

    metrics = {
        "mae_days": round(float(mean_absolute_error(y_test, predictions)), 4),
        "rmse_days": round(float(np.sqrt(mean_squared_error(y_test, predictions))), 4),
        "r2_score": round(float(r2_score(y_test, predictions)), 4),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test))
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(metrics, indent=4), encoding="utf-8")

    print("\n=== DELAY DURATION RESULTS ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Metrics saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()