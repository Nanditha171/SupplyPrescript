"""Unit and integration tests for Model Evaluation and Validation (Day 7).

Tests cover:
1. Correct positive-class definition (delay_occurred = 1).
2. Confusion matrix calculation (TN, FP, FN, TP).
3. Precision, recall, and F1-score calculation.
4. Undefined metrics when a class is absent.
5. ROC-AUC handling when only one class exists.
6. Threshold analysis across the configured range.
7. Cost-based error calculation.
8. Probability range validation.
9. Missing target handling.
10. Test-set identifier preservation.
11. Prevention of threshold tuning on the final test set.
12. Output report generation.
13. Reproducibility of evaluation results.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models.delay_prediction.config import default_config
from src.models.delay_prediction.error_analysis import (
    analyze_subgroups,
    build_error_analysis_dataset,
)
from src.models.delay_prediction.evaluate import run_evaluation
from src.models.delay_prediction.features import split_data_chronologically
from src.models.delay_prediction.threshold_analysis import (
    DEFAULT_THRESHOLDS,
    compute_threshold_grid,
    select_optimal_threshold,
)
from src.models.delay_prediction.validation import (
    calculate_cost_metric,
    compute_classification_metrics,
    evaluate_calibration,
    validate_dataset_and_split,
)


# -------------------------------------------------------------------------
# Test 1: Positive-Class Definition
# -------------------------------------------------------------------------

def test_positive_class_definition():
    """Verify positive class is strictly pinned to delay_occurred = 1."""
    # 2 positives, 2 negatives
    # Model flags 1 positive (TP=1) and 1 negative (FP=1), misses 1 positive (FN=1)
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.4, 0.8, 0.1])
    threshold = 0.50

    metrics = compute_classification_metrics(y_true, y_prob, threshold=threshold)

    assert metrics["positive_class_label"] == 1
    # TP=1, FP=1, FN=1, TN=1
    assert metrics["confusion_matrix"]["tp"] == 1
    assert metrics["confusion_matrix"]["fp"] == 1
    assert metrics["confusion_matrix"]["fn"] == 1
    assert metrics["confusion_matrix"]["tn"] == 1
    # Precision for class 1 = 1 / (1 + 1) = 0.5
    assert metrics["precision"] == pytest.approx(0.5)
    # Recall for class 1 = 1 / (1 + 1) = 0.5
    assert metrics["recall"] == pytest.approx(0.5)


# -------------------------------------------------------------------------
# Test 2: Confusion Matrix Calculation
# -------------------------------------------------------------------------

def test_confusion_matrix_calculation():
    """Verify confusion matrix counts (TN, FP, FN, TP) against known ground truth."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1])
    # Predictions: 3 TN, 1 FP, 1 FN, 2 TP
    y_prob = np.array([0.1, 0.2, 0.3, 0.8, 0.2, 0.7, 0.9])
    threshold = 0.50

    metrics = compute_classification_metrics(y_true, y_prob, threshold=threshold)
    cm = metrics["confusion_matrix"]

    assert cm["tn"] == 3
    assert cm["fp"] == 1
    assert cm["fn"] == 1
    assert cm["tp"] == 2
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == len(y_true)


# -------------------------------------------------------------------------
# Test 3: Precision, Recall, and F1-Score Calculation
# -------------------------------------------------------------------------

def test_precision_recall_f1_calculation():
    """Verify accuracy, precision, recall, F1, and specificity calculations."""
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.3])
    # threshold 0.5:
    # y_pred = [1, 1, 0, 1, 0, 0]
    # TP=2, FN=1, FP=1, TN=2
    metrics = compute_classification_metrics(y_true, y_prob, threshold=0.50)

    expected_acc = 4 / 6
    expected_prec = 2 / 3
    expected_rec = 2 / 3
    expected_f1 = 2 / 3
    expected_spec = 2 / 3

    assert metrics["accuracy"] == pytest.approx(expected_acc, rel=1e-3)
    assert metrics["precision"] == pytest.approx(expected_prec, rel=1e-3)
    assert metrics["recall"] == pytest.approx(expected_rec, rel=1e-3)
    assert metrics["f1"] == pytest.approx(expected_f1, rel=1e-3)
    assert metrics["specificity"] == pytest.approx(expected_spec, rel=1e-3)


# -------------------------------------------------------------------------
# Test 4: Undefined Metrics When a Class is Absent
# -------------------------------------------------------------------------

def test_undefined_metrics_when_class_absent():
    """Verify metrics calculation when positive class is entirely absent."""
    y_true = np.array([0, 0, 0, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.45])

    # No positives exist; at th=0.50 all pred=0, so TP=0, FP=0, FN=0, TN=5
    metrics = compute_classification_metrics(y_true, y_prob, threshold=0.50)

    assert metrics["accuracy"] == 1.0
    assert metrics["specificity"] == 1.0
    # Precision & recall division by zero handled safely with 0.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["support"]["class_1"] == 0
    assert metrics["support"]["class_0"] == 5


# -------------------------------------------------------------------------
# Test 5: ROC-AUC Handling When Only One Class Exists
# -------------------------------------------------------------------------

def test_roc_auc_handling_when_one_class_exists():
    """Verify ROC-AUC returns None without crashing when only one class is present."""
    y_true_all_zeros = np.array([0, 0, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4])

    metrics = compute_classification_metrics(y_true_all_zeros, y_prob)
    assert metrics["has_both_classes"] is False
    assert metrics["roc_auc"] is None
    assert metrics["average_precision"] is None
    assert isinstance(metrics["brier_score"], float)

    calib = evaluate_calibration(y_true_all_zeros, y_prob)
    assert calib["is_calibrated"] is False


# -------------------------------------------------------------------------
# Test 6: Threshold Analysis Across the Configured Range
# -------------------------------------------------------------------------

def test_threshold_analysis_range():
    """Verify threshold analysis evaluates across the full range with correct monotonicity."""
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0, 1, 0])
    y_prob = np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.82, 0.88, 0.95])

    grid_df = compute_threshold_grid(y_true, y_prob, thresholds=DEFAULT_THRESHOLDS)

    assert len(grid_df) == len(DEFAULT_THRESHOLDS)
    expected_cols = [
        "threshold",
        "precision",
        "recall",
        "f1_score",
        "specificity",
        "false_positive_count",
        "false_negative_count",
        "true_positive_count",
        "true_negative_count",
        "predicted_positive_count",
        "total_error_cost",
    ]
    for col in expected_cols:
        assert col in grid_df.columns

    # As threshold increases, recall should be non-increasing
    recalls = grid_df["recall"].values
    assert np.all(np.diff(recalls) <= 1e-4)

    # As threshold increases, specificity should be non-decreasing
    specificities = grid_df["specificity"].values
    assert np.all(np.diff(specificities) >= -1e-4)


# -------------------------------------------------------------------------
# Test 7: Cost-Based Error Calculation
# -------------------------------------------------------------------------

def test_cost_based_error_calculation():
    """Verify business cost formula: Total Cost = (FN * C_FN) + (FP * C_FP)."""
    # 2 FN, 3 FP
    y_true = np.array([1, 1, 0, 0, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 1])
    # actual 1s: indices 0, 1, 5. pred: [0, 0, 1] -> FN=2, TP=1
    # actual 0s: indices 2, 3, 4. pred: [1, 1, 1] -> FP=3, TN=0

    c_fn = 5000.0
    c_fp = 1000.0
    costs = calculate_cost_metric(y_true, y_pred, c_fn=c_fn, c_fp=c_fp)

    expected_fn_cost = 2 * 5000.0
    expected_fp_cost = 3 * 1000.0
    expected_total = expected_fn_cost + expected_fp_cost

    assert costs["false_negatives"] == 2
    assert costs["false_positives"] == 3
    assert costs["fn_cost"] == expected_fn_cost
    assert costs["fp_cost"] == expected_fp_cost
    assert costs["total_error_cost"] == expected_total
    assert costs["average_cost_per_shipment"] == expected_total / len(y_true)


# -------------------------------------------------------------------------
# Test 8: Probability Range Validation
# -------------------------------------------------------------------------

def test_probability_range_validation():
    """Verify probability range clipping and validity."""
    y_true = np.array([0, 1])
    # Extreme/out-of-bounds probabilities
    y_prob_extreme = np.array([-0.05, 1.05])

    metrics = compute_classification_metrics(y_true, y_prob_extreme, threshold=0.50)
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert metrics["confusion_matrix"]["tn"] == 1
    assert metrics["confusion_matrix"]["tp"] == 1


# -------------------------------------------------------------------------
# Test 9: Missing Target Handling
# -------------------------------------------------------------------------

def test_missing_target_handling():
    """Verify validation raises ValueError when target column is missing or all null."""
    df_missing_target = pd.DataFrame({
        "supplier_id": ["SUP1", "SUP2"],
        "demand_units": [100, 200],
    })
    with pytest.raises(ValueError, match="Target column 'delay_occurred' not found"):
        validate_dataset_and_split(df_missing_target, default_config)

    df_null_target = pd.DataFrame({
        "delay_occurred": [np.nan, np.nan],
        "supplier_id": ["SUP1", "SUP2"],
    })
    with pytest.raises(ValueError, match="missing/null"):
        validate_dataset_and_split(df_null_target, default_config)


# -------------------------------------------------------------------------
# Test 10: Test-Set Identifier Preservation
# -------------------------------------------------------------------------

def test_test_set_identifier_preservation():
    """Verify that record-level error analysis preserves event_id, date, and supplier/product keys."""
    df_sample = pd.DataFrame({
        "date": ["2026-04-01", "2026-04-02"],
        "supplier_id": ["SUP001", "SUP002"],
        "supplier_name": ["Alpha Logistics", "Beta Components"],
        "product_id": ["PRD01", "PRD02"],
        "product_name": ["Sensor", "Gear"],
        "category": ["Electronics", "Mechanical"],
        "supplier_country": ["Germany", "Japan"],
        "delay_occurred": [1, 0],
        "supplier_reliability": [0.92, 0.88],
    })
    y_prob = np.array([0.75, 0.20])

    audit_df = build_error_analysis_dataset(df_sample, df_sample.index, y_prob, threshold=0.50)

    assert len(audit_df) == 2
    assert "event_id" in audit_df.columns
    assert "date" in audit_df.columns
    assert "supplier_name" in audit_df.columns
    assert "product_name" in audit_df.columns
    assert audit_df.loc[0, "error_type"] == "TP"
    assert audit_df.loc[1, "error_type"] == "TN"
    assert bool(audit_df.loc[0, "is_correct"]) is True


# -------------------------------------------------------------------------
# Test 11: Prevention of Threshold Tuning on the Final Test Set
# -------------------------------------------------------------------------

def test_prevention_of_threshold_tuning_on_test_set():
    """Verify threshold selection is performed on validation data, maintaining test set isolation."""
    # Validation data
    y_val = np.array([0, 0, 1, 1, 0, 1])
    y_prob_val = np.array([0.2, 0.3, 0.6, 0.8, 0.4, 0.7])

    # Select threshold using validation only
    rec_th, val_metrics = select_optimal_threshold(
        y_val, y_prob_val, criterion="cost", c_fn=5000.0, c_fp=1000.0
    )

    assert 0.10 <= rec_th <= 0.90
    assert "total_error_cost" in val_metrics

    # Test data evaluated separately with fixed rec_th
    y_test = np.array([0, 1, 0, 0])
    y_prob_test = np.array([0.1, 0.8, 0.3, 0.2])
    test_metrics = compute_classification_metrics(y_test, y_prob_test, threshold=rec_th)

    assert test_metrics["threshold"] == rec_th
    assert test_metrics["confusion_matrix"]["tp"] == 1


# -------------------------------------------------------------------------
# Test 12: Output Report Generation
# -------------------------------------------------------------------------

def test_output_report_generation(tmp_path):
    """End-to-end evaluation test: verifies all required report files and plots are generated."""
    res = run_evaluation(report_dir=tmp_path)

    from src.models.delay_prediction.evaluate import HAS_MATPLOTLIB

    expected_files = [
        "metrics.json",
        "classification_report.txt",
        "evaluation_summary.md",
        "threshold_analysis.csv",
        "error_analysis.csv",
    ]
    if HAS_MATPLOTLIB:
        expected_files.extend([
            "confusion_matrix.png",
            "roc_curve.png",
            "precision_recall_curve.png",
            "calibration_curve.png",
            "probability_distribution.png",
            "threshold_tradeoff.png",
            "feature_error_analysis.png",
        ])

    for fname in expected_files:
        fpath = tmp_path / fname
        assert fpath.exists(), f"Required file '{fname}' was not generated in {tmp_path}."
        assert fpath.stat().st_size > 0, f"Generated file '{fname}' is empty."

    # Validate metrics.json content
    with open(tmp_path / "metrics.json", "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
    assert "test_metrics" in metrics_data
    assert "accuracy" in metrics_data["test_metrics"]
    assert metrics_data["test_metrics"]["accuracy"] == pytest.approx(0.6855, rel=1e-2)


# -------------------------------------------------------------------------
# Test 13: Reproducibility of Evaluation Results
# -------------------------------------------------------------------------

def test_reproducibility_of_evaluation_results():
    """Verify that evaluating the model multiple times produces deterministic, identical results."""
    df = pd.read_csv(default_config.dataset_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_chronologically(df, default_config)

    import joblib
    pipeline = joblib.load(default_config.model_path)

    # Run prediction 1
    y_prob_1 = pipeline.predict_proba(X_test)[:, 1]
    m1 = compute_classification_metrics(y_test, y_prob_1, threshold=0.50)

    # Run prediction 2
    y_prob_2 = pipeline.predict_proba(X_test)[:, 1]
    m2 = compute_classification_metrics(y_test, y_prob_2, threshold=0.50)

    np.testing.assert_array_equal(y_prob_1, y_prob_2)
    assert m1["accuracy"] == m2["accuracy"]
    assert m1["f1"] == m2["f1"]
    assert m1["roc_auc"] == m2["roc_auc"]
    assert m1["average_precision"] == m2["average_precision"]
    assert m1["confusion_matrix"] == m2["confusion_matrix"]
