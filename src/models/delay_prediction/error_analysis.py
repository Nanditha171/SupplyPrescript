"""Error and subgroup stability analysis for delay prediction.

Generates:
- Record-level error audit (reports/model_evaluation/error_analysis.csv)
- Operational subgroup performance breakdowns (by supplier, country, category, risk factors)
- Feature error visualization (reports/model_evaluation/feature_error_analysis.png)
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    sns = None

from src.models.delay_prediction.config import DelayPredictionConfig, default_config
from src.models.delay_prediction.features import split_data_chronologically

logger = logging.getLogger(__name__)


def build_error_analysis_dataset(
    df_full: pd.DataFrame,
    test_indices: pd.Index,
    y_prob: np.ndarray,
    threshold: float = 0.50,
) -> pd.DataFrame:
    """Construct record-level audit dataset for error analysis on the held-out test set.

    Strictly separates pre-event predictors from post-event outcomes.

    Args:
        df_full: Complete source DataFrame.
        test_indices: Chronological test set index positions.
        y_prob: Predicted delay probabilities for test records.
        threshold: Decision threshold for delay classification.

    Returns:
        DataFrame containing identifiers, actuals, predictions, error types, and key features.
    """
    test_df = df_full.loc[test_indices].copy()

    # Synthesize event_id if not present
    if "event_id" not in test_df.columns:
        if "date" in test_df.columns and "supplier_id" in test_df.columns and "product_id" in test_df.columns:
            event_ids = (
                "EVT_"
                + test_df["date"].astype(str)
                + "_"
                + test_df["supplier_id"].astype(str)
                + "_"
                + test_df["product_id"].astype(str)
            )
        else:
            event_ids = [f"EVT_{i+1:06d}" for i in range(len(test_df))]
    else:
        event_ids = test_df["event_id"]

    y_true = test_df["delay_occurred"].astype(int).values
    y_prob_arr = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob_arr >= threshold).astype(int)

    error_types = []
    for actual, pred in zip(y_true, y_pred):
        if actual == 1 and pred == 1:
            error_types.append("TP")  # Correctly flagged delay
        elif actual == 0 and pred == 0:
            error_types.append("TN")  # Correctly unflagged on-time
        elif actual == 0 and pred == 1:
            error_types.append("FP")  # False alarm (unnecessary cost)
        elif actual == 1 and pred == 0:
            error_types.append("FN")  # Missed delay (stockout / disruption)
        else:
            error_types.append("UNKNOWN")

    audit_df = pd.DataFrame({
        "event_id": event_ids,
        "date": test_df.get("date", pd.Series(index=test_df.index, dtype=str)),
        "actual_delay": y_true,
        "predicted_delay": y_pred,
        "delay_probability": np.round(y_prob_arr, 4),
        "threshold": float(threshold),
        "is_correct": (y_true == y_pred),
        "error_type": error_types,
        "supplier_id": test_df.get("supplier_id", "N/A"),
        "supplier_name": test_df.get("supplier_name", "N/A"),
        "supplier_country": test_df.get("supplier_country", "N/A"),
        "product_id": test_df.get("product_id", "N/A"),
        "product_name": test_df.get("product_name", "N/A"),
        "category": test_df.get("category", "N/A"),
        # Key pre-event features for root-cause diagnosis
        "supplier_reliability": test_df.get("supplier_reliability", np.nan),
        "planned_lead_time_days": test_df.get("planned_lead_time_days", np.nan),
        "shortage_flag": test_df.get("shortage_flag", np.nan),
        "demand_pressure": test_df.get("demand_pressure", np.nan),
        "capacity_utilization": test_df.get("capacity_utilization", np.nan),
        "inventory_coverage_ratio": test_df.get("inventory_coverage_ratio", np.nan),
        "unit_cost_usd": test_df.get("unit_cost_usd", np.nan),
        "transport_cost_usd": test_df.get("transport_cost_usd", np.nan),
    }, index=test_df.index)

    # Note: actual post-event outcome reference for business severity investigation
    if "delay_days" in test_df.columns:
        audit_df["outcome_delay_days_ref"] = test_df["delay_days"].values

    return audit_df


def analyze_subgroups(
    error_df: pd.DataFrame,
    group_col: str,
    min_sample_size: int = 5,
) -> pd.DataFrame:
    """Compute subgroup performance metrics across operational segments.

    Args:
        error_df: Record-level error audit DataFrame.
        group_col: Column name to segment by.
        min_sample_size: Minimum group size required for robust reporting.

    Returns:
        DataFrame with subgroup metrics, counts, and small-sample flags.
    """
    if group_col not in error_df.columns:
        return pd.DataFrame()

    records = []
    for group_val, grp in error_df.groupby(group_col):
        n_total = len(grp)
        n_pos = int((grp["actual_delay"] == 1).sum())
        n_tp = int((grp["error_type"] == "TP").sum())
        n_fp = int((grp["error_type"] == "FP").sum())
        n_fn = int((grp["error_type"] == "FN").sum())
        n_tn = int((grp["error_type"] == "TN").sum())

        precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0.0
        recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (n_tp + n_tn) / n_total if n_total > 0 else 0.0
        prevalence = n_pos / n_total if n_total > 0 else 0.0

        records.append({
            "group": str(group_val),
            "sample_size": n_total,
            "actual_delays": n_pos,
            "delay_prevalence": round(prevalence, 4),
            "true_positives": n_tp,
            "false_positives": n_fp,
            "false_negatives": n_fn,
            "true_negatives": n_tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "adequate_sample": (n_total >= min_sample_size),
        })

    sub_df = pd.DataFrame(records).sort_values("sample_size", ascending=False)
    return sub_df


def plot_feature_error_analysis(
    error_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create multi-panel diagnostic chart of error patterns across key operational drivers.

    Args:
        error_df: Record-level error audit DataFrame.
        output_path: Path to save PNG image.
    """
    if not HAS_MATPLOTLIB or plt is None:
        logger.warning("Matplotlib not available. Skipping feature error analysis plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=300)

    # 1. Error Types by Product Category
    ax1 = axes[0, 0]
    if "category" in error_df.columns:
        cat_counts = (
            pd.crosstab(error_df["category"], error_df["error_type"], normalize="index")
            * 100
        )
        for col in ["TN", "TP", "FP", "FN"]:
            if col not in cat_counts.columns:
                cat_counts[col] = 0.0
        cat_counts[["TN", "TP", "FP", "FN"]].plot(
            kind="bar",
            stacked=True,
            color=["#059669", "#0284C7", "#F59E0B", "#E11D48"],
            ax=ax1,
            edgecolor="#334155",
            lw=0.5,
        )
        ax1.set_title("Classification Outcomes by Product Category (%)", fontsize=11, weight="bold")
        ax1.set_ylabel("Percentage (%)", fontsize=10)
        ax1.set_xlabel("")
        ax1.tick_params(axis="x", rotation=25)
        ax1.legend(loc="upper right", frameon=True, fontsize=8)
        ax1.grid(True, linestyle=":", alpha=0.5)

    # 2. Probability Distribution by Error Type
    ax2 = axes[0, 1]
    palette_dict = {"TP": "#0284C7", "TN": "#059669", "FP": "#F59E0B", "FN": "#E11D48"}
    error_order = [e for e in ["TP", "TN", "FP", "FN"] if e in error_df["error_type"].values]
    if sns is not None:
        sns.boxplot(
            data=error_df,
            x="error_type",
            y="delay_probability",
            hue="error_type",
            order=error_order,
            palette=palette_dict,
            legend=False,
            ax=ax2,
        )
    else:
        # Fallback without seaborn
        data_to_plot = [error_df[error_df["error_type"] == e]["delay_probability"] for e in error_order]
        ax2.boxplot(data_to_plot, tick_labels=error_order)

    ax2.axhline(0.50, color="#64748B", linestyle="--", lw=1.2, label="Threshold (0.50)")
    ax2.set_title("Predicted Delay Probability by Classification Outcome", fontsize=11, weight="bold")
    ax2.set_xlabel("Outcome / Error Category", fontsize=10)
    ax2.set_ylabel("Predicted Delay Probability", fontsize=10)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, linestyle=":", alpha=0.5)

    # 3. Error Breakdown by Supplier Reliability Tier
    ax3 = axes[1, 0]
    if "supplier_reliability" in error_df.columns:
        rel = error_df["supplier_reliability"]
        rel_tier = pd.cut(
            rel,
            bins=[0.0, 0.85, 0.95, 1.01],
            labels=["Low (<0.85)", "Moderate (0.85-0.95)", "High (>=0.95)"],
            include_lowest=True,
        )
        tier_counts = pd.crosstab(rel_tier, error_df["error_type"], normalize="index") * 100
        for col in ["TN", "TP", "FP", "FN"]:
            if col not in tier_counts.columns:
                tier_counts[col] = 0.0
        tier_counts[["TN", "TP", "FP", "FN"]].plot(
            kind="bar",
            stacked=True,
            color=["#059669", "#0284C7", "#F59E0B", "#E11D48"],
            ax=ax3,
            edgecolor="#334155",
            lw=0.5,
        )
        ax3.set_title("Outcome Breakdown by Supplier Reliability Tier (%)", fontsize=11, weight="bold")
        ax3.set_ylabel("Percentage (%)", fontsize=10)
        ax3.set_xlabel("")
        ax3.tick_params(axis="x", rotation=15)
        ax3.legend(loc="upper right", frameon=True, fontsize=8)
        ax3.grid(True, linestyle=":", alpha=0.5)

    # 4. Error Breakdown by Shortage Flag
    ax4 = axes[1, 1]
    if "shortage_flag" in error_df.columns:
        shortage_labels = error_df["shortage_flag"].map({1: "Inventory Shortage", 0: "Adequate Inventory"}).fillna("Unknown")
        shortage_counts = pd.crosstab(shortage_labels, error_df["error_type"], normalize="index") * 100
        for col in ["TN", "TP", "FP", "FN"]:
            if col not in shortage_counts.columns:
                shortage_counts[col] = 0.0
        shortage_counts[["TN", "TP", "FP", "FN"]].plot(
            kind="bar",
            stacked=True,
            color=["#059669", "#0284C7", "#F59E0B", "#E11D48"],
            ax=ax4,
            edgecolor="#334155",
            lw=0.5,
        )
        ax4.set_title("Outcome Breakdown by Inventory Shortage Status (%)", fontsize=11, weight="bold")
        ax4.set_ylabel("Percentage (%)", fontsize=10)
        ax4.set_xlabel("")
        ax4.tick_params(axis="x", rotation=15)
        ax4.legend(loc="upper right", frameon=True, fontsize=8)
        ax4.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    logger.info(f"Saved feature error analysis plot to {output_path}")
    plt.close(fig)


def run_error_analysis(
    dataset_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    threshold: Optional[float] = None,
    config: DelayPredictionConfig = default_config,
) -> Dict[str, Any]:
    """Execute complete error analysis pipeline on held-out test set."""
    effective_data_path = dataset_path or config.dataset_path
    effective_model_path = model_path or config.model_path
    effective_out_dir = output_dir or config.report_dir
    effective_th = threshold if threshold is not None else config.classification_threshold

    effective_out_dir.mkdir(parents=True, exist_ok=True)

    if not effective_data_path.exists():
        raise FileNotFoundError(f"Dataset not found at '{effective_data_path}'.")

    if not effective_model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at '{effective_model_path}'.")

    logger.info(f"Loading model from {effective_model_path}...")
    pipeline = joblib.load(effective_model_path)

    logger.info(f"Loading data from {effective_data_path}...")
    df = pd.read_csv(effective_data_path)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data_chronologically(df, config)

    y_prob_test = pipeline.predict_proba(X_test)[:, 1]

    # Build audit dataset
    error_df = build_error_analysis_dataset(
        df_full=df,
        test_indices=X_test.index,
        y_prob=y_prob_test,
        threshold=effective_th,
    )

    # Export error analysis CSV
    csv_out_path = effective_out_dir / "error_analysis.csv"
    error_df.to_csv(csv_out_path, index=False)
    logger.info(f"Saved record-level error analysis CSV to {csv_out_path}")

    # Generate diagnostic plot
    png_out_path = effective_out_dir / "feature_error_analysis.png"
    plot_feature_error_analysis(error_df, png_out_path)

    # Subgroup breakdowns
    subgroups = {
        "category": analyze_subgroups(error_df, "category", min_sample_size=5),
        "supplier_country": analyze_subgroups(error_df, "supplier_country", min_sample_size=5),
        "supplier_name": analyze_subgroups(error_df, "supplier_name", min_sample_size=5),
        "shortage_flag": analyze_subgroups(error_df, "shortage_flag", min_sample_size=5),
    }

    return {
        "error_df": error_df,
        "csv_path": csv_out_path,
        "png_path": png_out_path,
        "subgroups": subgroups,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run record-level error and subgroup stability analysis.")
    parser.add_argument("--data", type=str, default=None, help="Dataset CSV path")
    parser.add_argument("--model", type=str, default=None, help="Trained model path")
    parser.add_argument("--output-dir", type=str, default=None, help="Reports output directory")
    parser.add_argument("--threshold", type=float, default=None, help="Classification decision threshold")
    args = parser.parse_args()

    data_p = Path(args.data) if args.data else None
    mod_p = Path(args.model) if args.model else None
    out_p = Path(args.output_dir) if args.output_dir else None

    run_error_analysis(dataset_path=data_p, model_path=mod_p, output_dir=out_p, threshold=args.threshold)
