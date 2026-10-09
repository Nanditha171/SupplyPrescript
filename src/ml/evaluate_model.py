from pathlib import Path
import pandas as pd
from joblib import load
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
data_path = ROOT / "data" / "processed" / "supply_prescript_processed_dataset.csv"
model_path = ROOT / "models" / "supply_delay_model.joblib"
output_path = ROOT / "reports" / "threshold_evaluation.csv"

numeric_features = [
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
categorical_features = ["supplier_id", "supplier_country", "product_id", "category"]
features = numeric_features + categorical_features

def main():
    print("Loading dataset and model...")
    data = pd.read_csv(data_path)
    model = load(model_path)

    X = data[features]
    y = data["delay_occurred"].astype(int)

    split_index = int(len(data) * 0.8)
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    probabilities = model.predict_proba(X_test)[:, 1]
    rows = []

    for threshold in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).ravel()

        rows.append({
            "threshold": threshold,
            "precision": round(precision_score(y_test, predictions, zero_division=0), 4),
            "recall": round(recall_score(y_test, predictions, zero_division=0), 4),
            "f1_score": round(f1_score(y_test, predictions, zero_division=0), 4),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        })

    results = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print("\n=== THRESHOLD EVALUATION RESULTS ===")
    print(results.to_string(index=False))
    print(f"\nSaved results to: {output_path}")

if __name__ == "__main__":
    main()
