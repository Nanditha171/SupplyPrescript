from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "supply_prescript_processed_dataset.csv"
MODEL = ROOT / "models" / "supply_delay_model.joblib"
REPORT = ROOT / "reports" / "baseline_metrics.json"

NUMERIC = [
    "demand_units", "inventory_units", "supplier_capacity_units",
    "supplier_reliability", "planned_lead_time_days", "unit_cost_usd",
    "transport_cost_usd", "secondary_supplier_premium_pct",
    "inventory_coverage_ratio", "inventory_gap_units",
    "capacity_utilization", "cost_per_demand_unit",
    "secondary_cost_difference", "air_freight_premium",
    "supplier_risk_score", "demand_pressure", "shortage_flag",
    "previous_delay_days", "previous_delay_flag", "rolling_average_delay",
    "rolling_average_demand", "rolling_average_inventory",
    "demand_outlier_flag", "inventory_outlier_flag",
    "cost_outlier_flag", "lead_time_outlier_flag",
]
CATEGORICAL = ["supplier_id", "supplier_country", "product_id", "category"]
TARGET = "delay_occurred"


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA)

    features = NUMERIC + CATEGORICAL
    missing = [c for c in features + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).sort_values("date").reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(int)

    X = df[features]
    y = df[TARGET]
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(
            "Chronological split lacks both classes. Dataset split needs review."
        )

    print(f"Rows: {len(df)} | Train: {len(X_train)} | Test: {len(X_test)}")

    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
    ])

    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())

    model = Pipeline([
        ("preprocessor", prep),
        ("classifier", XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=negatives / positives if positives else 1,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    print("Training XGBoost model...")
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "rows": len(df),
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "features": features,
        "target": TARGET,
    }

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL)
    REPORT.write_text(json.dumps(metrics, indent=4), encoding="utf-8")

    print("\n=== BASELINE MODEL RESULTS ===")
    for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        print(f"{key}: {metrics[key]:.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_test, pred, labels=[0, 1],
        target_names=["On time", "Delayed"], zero_division=0
    ))
    print(f"Model saved: {MODEL}")
    print(f"Metrics saved: {REPORT}")


if __name__ == "__main__":
    main()
