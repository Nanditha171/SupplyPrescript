"""Validation module for supply chain delay prediction model.

Handles:
- Dataset and test-set integrity validation
- Model artifact and metadata compatibility checks
- Mathematically robust classification metrics calculation
- Cost-based evaluation with configurable operational cost parameters
- Probability calibration analysis
- Model candidate comparison from Day 6 metadata
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.models.delay_prediction.config import DelayPredictionConfig, default_config

logger = logging.getLogger(__name__)


def validate_dataset_and_split(
    df: pd.DataFrame,
    config: DelayPredictionConfig = default_config,
) -> Dict[str, Any]:
    """Validate dataset structure, target column, pre-event features, and chronological boundaries.

    Args:
        df: Input processed dataset DataFrame.
        config: Delay prediction configuration.

    Returns:
        Dictionary with validation status, checks performed, and record summary.

    Raises:
        ValueError: If target column is missing, contains non-binary values, or no valid features exist.
    """
    if df is None or df.empty:
        raise ValueError("Provided dataset is empty or None.")

    issues: List[str] = []
    warnings: List[str] = []

    # 1. Target column validation
    target_col = config.target_column
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    target_series = df[target_col]
    if target_series.isnull().any():
        null_count = int(target_series.isnull().sum())
        issues.append(f"Target column contains {null_count} missing/null values.")

    unique_vals = set(target_series.dropna().unique())
    # Verify binary 0/1
    if not unique_vals.issubset({0, 1, 0.0, 1.0, True, False}):
        raise ValueError(
            f"Target column '{target_col}' contains invalid non-binary values: {unique_vals}. "
            "Must be binary with positive class = 1 (delay_occurred)."
        )

    # 2. Check for chronological date column
    has_date = (
        config.date_column in df.columns
        and pd.to_datetime(df[config.date_column], errors="coerce").notnull().all()
    )
    if not has_date:
        warnings.append(
            f"Date column '{config.date_column}' is missing or contains invalid timestamps."
        )

    # 3. Check for pre-event features vs leakage columns
    leakage_in_cols = [c for c in config.leakage_columns if c in df.columns and c != target_col]
    if leakage_in_cols:
        warnings.append(
            f"Dataset contains post-event leakage columns ({leakage_in_cols}); "
            "ensure these are strictly excluded from pre-event model features."
        )

    # 4. Check for shipment / event identifiers
    id_cols_present = [c for c in config.identifier_columns if c in df.columns]

    val_summary = {
        "is_valid": len(issues) == 0,
        "total_records": len(df),
        "target_column": target_col,
        "target_distribution": target_series.value_counts().to_dict(),
        "has_date_column": has_date,
        "leakage_columns_detected": leakage_in_cols,
        "identifier_columns_present": id_cols_present,
        "issues": issues,
        "warnings": warnings,
    }

    if issues:
        raise ValueError(f"Dataset validation failed: {'; '.join(issues)}")

    logger.info("Dataset and split validation passed successfully.")
    return val_summary


def validate_model_artifacts(
    model_path: Path,
    metadata_path: Path,
) -> Dict[str, Any]:
    """Verify that Day 6 model artifact and metadata file exist and are compatible.

    Args:
        model_path: Path to delay_classifier.joblib.
        metadata_path: Path to model_metadata.json.

    Returns:
        Loaded metadata dictionary.

    Raises:
        FileNotFoundError: If required artifacts do not exist.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved Day 6 model not found at '{model_path}'. "
            "Please ensure Day 6 model training has been completed."
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Day 6 model metadata not found at '{metadata_path}'. "
            "Please ensure model metadata JSON is available."
        )

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    logger.info(
        f"Verified model artifacts for '{metadata.get('model_name', 'delay_classifier')}' "
        f"v{metadata.get('model_version', 'unknown')} ({metadata.get('selected_algorithm')})."
    )
    return metadata


def compute_classification_metrics(
    y_true: Union[pd.Series, np.ndarray, List[int]],
    y_prob: Union[pd.Series, np.ndarray, List[float]],
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Calculate mathematically robust classification metrics for binary delay prediction.

    Explicitly specifies positive class as delay_occurred = 1.
    Gracefully handles edge cases where only one class is present in y_true.

    Args:
        y_true: Ground truth binary targets (0 or 1).
        y_prob: Predicted probability of delay (positive class 1).
        threshold: Decision threshold for positive prediction.

    Returns:
        Dictionary of metrics including accuracy, precision, recall, f1, specificity,
        roc_auc, average_precision, brier_score, confusion_matrix, support.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    if len(y_true_arr) != len(y_prob_arr):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true_arr)} items, y_prob has {len(y_prob_arr)} items."
        )

    if len(y_true_arr) == 0:
        raise ValueError("Input arrays must not be empty.")

    # Validate probability range
    if np.any(y_prob_arr < 0.0) or np.any(y_prob_arr > 1.0):
        # Clip slight floating point anomalies
        y_prob_arr = np.clip(y_prob_arr, 0.0, 1.0)

    # Positive class definition: delay_occurred = 1
    y_pred_arr = (y_prob_arr >= threshold).astype(int)

    unique_classes = np.unique(y_true_arr)
    has_both_classes = len(unique_classes) > 1

    # Confusion matrix
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Counts and support
    support_0 = int(np.sum(y_true_arr == 0))
    support_1 = int(np.sum(y_true_arr == 1))
    total_samples = len(y_true_arr)
    prevalence = float(support_1 / total_samples) if total_samples > 0 else 0.0

    # Basic metrics
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0))

    # Specificity = TN / (TN + FP)
    actual_negatives = tn + fp
    specificity = float(tn / actual_negatives) if actual_negatives > 0 else 0.0

    # Probability-based metrics (require both classes to be valid)
    if has_both_classes:
        roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
        pr_auc = float(average_precision_score(y_true_arr, y_prob_arr, pos_label=1))
        brier = float(brier_score_loss(y_true_arr, y_prob_arr))
    else:
        # Document limitation: ROC-AUC and PR-AUC mathematically undefined for single class
        roc_auc = None
        pr_auc = None
        brier = float(brier_score_loss(y_true_arr, y_prob_arr))

    clf_rep_str = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=[0, 1],
        target_names=["On-Time (0)", "Delayed (1)"],
        digits=4,
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "specificity": specificity,
        "roc_auc": roc_auc,
        "average_precision": pr_auc,
        "brier_score": brier,
        "positive_class_label": 1,
        "threshold": float(threshold),
        "prevalence": prevalence,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "support": {
            "class_0": support_0,
            "class_1": support_1,
            "total": total_samples,
        },
        "has_both_classes": bool(has_both_classes),
        "classification_report_str": clf_rep_str,
    }


def calculate_cost_metric(
    y_true: Union[pd.Series, np.ndarray, List[int]],
    y_pred: Union[pd.Series, np.ndarray, List[int]],
    c_fn: float = 5000.0,
    c_fp: float = 1000.0,
    c_tp: float = 0.0,
    c_tn: float = 0.0,
) -> Dict[str, float]:
    """Calculate operational business costs of classification outcomes.

    Supply Prescript Cost Model:
    Total Error Cost = (False Negatives * C_FN) + (False Positives * C_FP)
                       + (True Positives * C_TP) + (True Negatives * C_TN)

    Args:
        y_true: Actual delay outcomes (0 or 1).
        y_pred: Predicted delay outcomes (0 or 1).
        c_fn: Estimated cost of missing a shipment delay (e.g., stockout / SLA breach).
        c_fp: Estimated cost of false alarm intervention (e.g., unnecessary air-freight/expedite).
        c_tp: Cost of intervention on actual delay.
        c_tn: Cost of leaving on-time shipment unflagged.

    Returns:
        Dictionary with cost breakdown and per-shipment average.
    """
    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_pred).astype(int)

    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fn_cost = float(fn * c_fn)
    fp_cost = float(fp * c_fp)
    tp_cost = float(tp * c_tp)
    tn_cost = float(tn * c_tn)
    total_cost = fn_cost + fp_cost + tp_cost + tn_cost
    n_samples = len(y_t)
    avg_cost = total_cost / max(1, n_samples)

    return {
        "c_fn": float(c_fn),
        "c_fp": float(c_fp),
        "c_tp": float(c_tp),
        "c_tn": float(c_tn),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "fn_cost": fn_cost,
        "fp_cost": fp_cost,
        "total_error_cost": fn_cost + fp_cost,
        "total_cost": total_cost,
        "average_cost_per_shipment": avg_cost,
    }


def evaluate_calibration(
    y_true: Union[pd.Series, np.ndarray, List[int]],
    y_prob: Union[pd.Series, np.ndarray, List[float]],
    n_bins: int = 5,
    strategy: str = "uniform",
) -> Dict[str, Any]:
    """Evaluate empirical probability calibration using binned reliability curve.

    Args:
        y_true: Binary ground truth labels.
        y_prob: Predicted delay probabilities.
        n_bins: Number of probability bins.
        strategy: 'uniform' or 'quantile'.

    Returns:
        Dictionary containing prob_true, prob_pred, brier_score, and binned details.
    """
    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_prob, dtype=float)

    if len(np.unique(y_t)) < 2:
        return {
            "is_calibrated": False,
            "brier_score": float(brier_score_loss(y_t, y_p)),
            "prob_true": [],
            "prob_pred": [],
            "bin_counts": [],
            "note": "Only one class present in data; calibration curve unavailable.",
        }

    prob_true, prob_pred = calibration_curve(y_t, y_p, n_bins=n_bins, strategy=strategy)
    brier = float(brier_score_loss(y_t, y_p))

    # Calculate count of samples per bin
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_p, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    bin_counts = [int(np.sum(bin_indices == i)) for i in range(n_bins)]

    return {
        "is_calibrated": True,
        "brier_score": brier,
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
        "bin_counts": bin_counts,
        "strategy": strategy,
        "n_bins": n_bins,
    }


def get_candidate_models_summary(metadata: Dict[str, Any]) -> pd.DataFrame:
    """Format Day 6 candidate model validation comparisons into a clean DataFrame."""
    comp = metadata.get("model_comparison", {})
    if not comp:
        return pd.DataFrame()

    records = []
    for model_name, metrics in comp.items():
        records.append({
            "model_name": model_name,
            "accuracy": metrics.get("accuracy", np.nan),
            "precision": metrics.get("precision", np.nan),
            "recall": metrics.get("recall", np.nan),
            "f1": metrics.get("f1", np.nan),
            "roc_auc": metrics.get("roc_auc", np.nan),
            "average_precision": metrics.get("average_precision", np.nan),
            "brier_score": metrics.get("brier_score", np.nan),
        })

    return pd.DataFrame(records).sort_values("average_precision", ascending=False)
