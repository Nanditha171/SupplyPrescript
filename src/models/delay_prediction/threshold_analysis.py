"""Threshold analysis and business-cost optimization for delay prediction.

Evaluates trade-offs across decision thresholds:
- Precision vs Recall vs F1
- False Positives vs False Negatives
- Cost-optimal threshold selection using validation data only
- Generates threshold_analysis.csv and threshold_tradeoff.png
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
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

from src.models.delay_prediction.config import DelayPredictionConfig, default_config
from src.models.delay_prediction.features import split_data_chronologically
from src.models.delay_prediction.validation import (
    calculate_cost_metric,
    compute_classification_metrics,
)

logger = logging.getLogger(__name__)

# Configurable evaluation thresholds (0.10 to 0.90 in increments of 0.05)
DEFAULT_THRESHOLDS: List[float] = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]

# Business cost parameters (illustrative defaults)
DEFAULT_C_FN: float = 5000.0  # Cost of missed delay (stockout, expedited recovery, SLA penalty)
DEFAULT_C_FP: float = 1000.0  # Cost of false alarm (precautionary buffer, inspection)


def compute_threshold_grid(
    y_true: Union[pd.Series, np.ndarray, List[int]],
    y_prob: Union[pd.Series, np.ndarray, List[float]],
    thresholds: Optional[List[float]] = None,
    c_fn: float = DEFAULT_C_FN,
    c_fp: float = DEFAULT_C_FP,
) -> pd.DataFrame:
    """Evaluate classification and operational cost metrics across a grid of thresholds.

    Args:
        y_true: Binary ground truth (0 = on-time, 1 = delay).
        y_prob: Predicted delay probabilities.
        thresholds: List of thresholds to evaluate.
        c_fn: Cost per False Negative.
        c_fp: Cost per False Positive.

    Returns:
        DataFrame containing metrics and counts for each threshold.
    """
    threshold_list = thresholds or DEFAULT_THRESHOLDS
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    rows: List[Dict[str, Any]] = []

    for th in threshold_list:
        th_val = round(float(th), 4)
        metrics = compute_classification_metrics(y_true_arr, y_prob_arr, threshold=th_val)
        cm = metrics["confusion_matrix"]
        costs = calculate_cost_metric(y_true_arr, (y_prob_arr >= th_val).astype(int), c_fn=c_fn, c_fp=c_fp)

        tp = cm["tp"]
        fp = cm["fp"]
        fn = cm["fn"]
        tn = cm["tn"]
        pred_positives = tp + fp

        rows.append({
            "threshold": th_val,
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1_score": round(metrics["f1"], 4),
            "specificity": round(metrics["specificity"], 4),
            "false_positive_count": int(fp),
            "false_negative_count": int(fn),
            "true_positive_count": int(tp),
            "true_negative_count": int(tn),
            "predicted_positive_count": int(pred_positives),
            "total_error_cost": round(costs["total_error_cost"], 2),
            "average_cost_per_shipment": round(costs["average_cost_per_shipment"], 2),
        })

    return pd.DataFrame(rows)


def select_optimal_threshold(
    y_val: Union[pd.Series, np.ndarray, List[int]],
    y_prob_val: Union[pd.Series, np.ndarray, List[float]],
    criterion: str = "cost",
    thresholds: Optional[List[float]] = None,
    c_fn: float = DEFAULT_C_FN,
    c_fp: float = DEFAULT_C_FP,
) -> Tuple[float, Dict[str, Any]]:
    """Select the optimal decision threshold using validation data only.

    CRITICAL RULE:
    The final test set must NEVER be used to choose or tune the operational threshold.
    Selection is conducted strictly on validation data to prevent optimistic bias.

    Args:
        y_val: Validation ground truth.
        y_prob_val: Validation predicted probabilities.
        criterion: 'cost' to minimize total error cost, or 'f1' to maximize F1-score.
        thresholds: Threshold range to consider.
        c_fn: Cost per false negative.
        c_fp: Cost per false positive.

    Returns:
        Tuple of (selected_threshold, metrics_dict_at_threshold).
    """
    grid_df = compute_threshold_grid(y_val, y_prob_val, thresholds=thresholds, c_fn=c_fn, c_fp=c_fp)

    if criterion == "cost":
        # Minimize total error cost; break ties with higher F1
        best_row = grid_df.sort_values(
            by=["total_error_cost", "f1_score", "recall"],
            ascending=[True, False, False],
        ).iloc[0]
    elif criterion == "f1":
        # Maximize F1; break ties with lower cost
        best_row = grid_df.sort_values(
            by=["f1_score", "total_error_cost"],
            ascending=[False, True],
        ).iloc[0]
    else:
        raise ValueError(f"Unsupported selection criterion '{criterion}'. Choose 'cost' or 'f1'.")

    opt_th = float(best_row["threshold"])
    logger.info(
        f"Optimal threshold on validation data: {opt_th:.2f} (criterion='{criterion}', "
        f"cost=${best_row['total_error_cost']:,.0f}, F1={best_row['f1_score']:.4f}, "
        f"Recall={best_row['recall']:.4f}, Precision={best_row['precision']:.4f})"
    )
    return opt_th, best_row.to_dict()


def plot_threshold_tradeoff(
    df_thresholds: pd.DataFrame,
    recommended_threshold: float,
    baseline_threshold: float = 0.50,
    output_path: Optional[Path] = None,
) -> None:
    """Plot precision, recall, F1, and cost curves across decision thresholds.

    Args:
        df_thresholds: Output DataFrame from compute_threshold_grid.
        recommended_threshold: Cost-optimal threshold selected on validation.
        baseline_threshold: Default Day 6 baseline threshold (0.50).
        output_path: Destination path for the saved PNG figure.
    """
    if not HAS_MATPLOTLIB or plt is None:
        logger.warning("Matplotlib not available. Skipping threshold tradeoff plot.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 9), dpi=300, sharex=True)

    th = df_thresholds["threshold"]
    prec = df_thresholds["precision"]
    rec = df_thresholds["recall"]
    f1 = df_thresholds["f1_score"]
    cost = df_thresholds["total_error_cost"]

    # --- Subplot 1: Classification Trade-Off ---
    ax1.plot(th, prec, label="Precision", color="#0284C7", lw=2.5, marker="o", markersize=4)
    ax1.plot(th, rec, label="Recall", color="#E11D48", lw=2.5, marker="s", markersize=4)
    ax1.plot(th, f1, label="F1-Score", color="#10B981", lw=2.5, marker="^", markersize=4)

    ax1.axvline(
        x=baseline_threshold,
        color="#64748B",
        linestyle="--",
        lw=1.5,
        label=f"Baseline Threshold ({baseline_threshold:.2f})",
    )
    ax1.axvline(
        x=recommended_threshold,
        color="#7C3AED",
        linestyle="-.",
        lw=2.0,
        label=f"Recommended ({recommended_threshold:.2f})",
    )

    ax1.set_ylabel("Metric Value", fontsize=11, weight="semibold")
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_title("Classification Metric Trade-Off Across Thresholds", fontsize=12, weight="bold")
    ax1.legend(loc="best", frameon=True, fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # --- Subplot 2: Operational Error Cost ---
    ax2.plot(th, cost, label="Total Error Cost ($)", color="#D97706", lw=2.5, marker="d", markersize=4)
    ax2.axvline(
        x=baseline_threshold,
        color="#64748B",
        linestyle="--",
        lw=1.5,
        label=f"Baseline ({baseline_threshold:.2f})",
    )
    ax2.axvline(
        x=recommended_threshold,
        color="#7C3AED",
        linestyle="-.",
        lw=2.0,
        label=f"Recommended ({recommended_threshold:.2f})",
    )

    # Annotate minimum cost point
    min_cost = df_thresholds["total_error_cost"].min()
    min_cost_th = df_thresholds.loc[df_thresholds["total_error_cost"] == min_cost, "threshold"].iloc[0]
    ax2.scatter([min_cost_th], [min_cost], color="#DC2626", s=80, zorder=5)
    ax2.annotate(
        f"Min Cost: ${min_cost:,.0f}\n(th={min_cost_th:.2f})",
        (min_cost_th, min_cost),
        textcoords="offset points",
        xytext=(15, 10),
        fontsize=9,
        weight="bold",
        color="#7F1D1D",
        arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.2),
    )

    ax2.set_xlabel("Decision Threshold (Probability of Delay)", fontsize=11, weight="semibold")
    ax2.set_ylabel("Total Error Cost ($)", fontsize=11, weight="semibold")
    ax2.set_title("Operational Error Cost by Decision Threshold", fontsize=12, weight="bold")
    ax2.legend(loc="upper right", frameon=True, fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
        logger.info(f"Saved threshold tradeoff plot to {output_path}")
    plt.close(fig)


def run_threshold_analysis(
    dataset_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    c_fn: float = DEFAULT_C_FN,
    c_fp: float = DEFAULT_C_FP,
    config: DelayPredictionConfig = default_config,
) -> Dict[str, Any]:
    """Execute threshold evaluation pipeline.

    Workflow:
    1. Load trained Day 6 pipeline.
    2. Split dataset chronologically into Train, Validation, and Test sets.
    3. Generate validation probabilities and select cost-optimal threshold.
    4. Generate held-out test set probabilities and compute test threshold grid.
    5. Save reports/model_evaluation/threshold_analysis.csv and threshold_tradeoff.png.

    Args:
        dataset_path: Path to dataset CSV.
        model_path: Path to delay_classifier.joblib.
        output_dir: Destination reports directory.
        c_fn: False Negative cost assumption.
        c_fp: False Positive cost assumption.
        config: Central configuration.

    Returns:
        Dictionary with recommended threshold, test grid, and paths.
    """
    effective_data_path = dataset_path or config.dataset_path
    effective_model_path = model_path or config.model_path
    effective_out_dir = output_dir or config.report_dir

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

    # 1. Validation predictions for threshold selection
    y_prob_val = pipeline.predict_proba(X_val)[:, 1]
    rec_th, val_best_metrics = select_optimal_threshold(
        y_val, y_prob_val, criterion="cost", c_fn=c_fn, c_fp=c_fp
    )

    # 2. Test predictions for held-out evaluation across threshold grid
    y_prob_test = pipeline.predict_proba(X_test)[:, 1]
    test_grid_df = compute_threshold_grid(y_test, y_prob_test, c_fn=c_fn, c_fp=c_fp)

    # 3. Export CSV
    csv_out_path = effective_out_dir / "threshold_analysis.csv"
    test_grid_df.to_csv(csv_out_path, index=False)
    logger.info(f"Saved threshold analysis CSV to {csv_out_path}")

    # 4. Generate Plot
    png_out_path = effective_out_dir / "threshold_tradeoff.png"
    plot_threshold_tradeoff(
        test_grid_df,
        recommended_threshold=rec_th,
        baseline_threshold=config.classification_threshold,
        output_path=png_out_path,
    )

    return {
        "recommended_threshold": rec_th,
        "baseline_threshold": config.classification_threshold,
        "validation_best": val_best_metrics,
        "threshold_grid_df": test_grid_df,
        "csv_path": csv_out_path,
        "png_path": png_out_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze classification thresholds and operational trade-offs.")
    parser.add_argument("--data", type=str, default=None, help="Dataset CSV path")
    parser.add_argument("--model", type=str, default=None, help="Trained model path")
    parser.add_argument("--output-dir", type=str, default=None, help="Reports output directory")
    parser.add_argument("--c-fn", type=float, default=DEFAULT_C_FN, help="False negative cost ($)")
    parser.add_argument("--c-fp", type=float, default=DEFAULT_C_FP, help="False positive cost ($)")
    args = parser.parse_args()

    data_p = Path(args.data) if args.data else None
    mod_p = Path(args.model) if args.model else None
    out_p = Path(args.output_dir) if args.output_dir else None

    run_threshold_analysis(dataset_path=data_p, model_path=mod_p, output_dir=out_p, c_fn=args.c_fn, c_fp=args.c_fp)
