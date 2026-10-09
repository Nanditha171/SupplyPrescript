"""Comprehensive unit tests for Day 6 Delay Prediction ML module.

Tests cover:
1. Target column validation & error on missing target
2. Binary target normalization & non-binary rejection
3. Feature selection and column resolution
4. Strict post-event leakage column exclusion
5. Missing numerical value handling via pipeline imputation
6. Unseen categorical value handling via OneHotEncoder(handle_unknown='ignore')
7. End-to-end model training on sample dataset
8. Probability boundaries [0.0, 1.0]
9. Operational risk level threshold mappings (LOW, MEDIUM, HIGH, CRITICAL)
10. Required prediction fields verification
11. Model serialization & Joblib reload roundtrip
12. Zero-retraining verification during inference
13. Strict train-only fitting for preprocessing
14. Clear error handling on invalid inputs
15. Safe database persistence error handling
"""

import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.models.delay_prediction.config import DelayPredictionConfig
from src.models.delay_prediction.features import (
    build_preprocessing_pipeline,
    select_pre_event_features,
    split_data_chronologically,
    validate_and_normalize_target,
)
from src.models.delay_prediction.predict import (
    DelayPredictor,
    map_probability_to_risk_level,
    persist_predictions_to_database,
    predict_delay,
)
from src.models.delay_prediction.train import (
    save_model_and_metadata,
    train_and_compare_models,
)


@pytest.fixture
def sample_synthetic_dataset() -> pd.DataFrame:
    """Create a minimal valid synthetic dataset with pre-event and leakage columns."""
    dates = pd.date_range("2025-01-01", periods=60, freq="W")
    np.random.seed(42)

    df = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "supplier_id": np.random.choice(["SUP001", "SUP002", "SUP003"], size=60),
            "supplier_country": np.random.choice(["India", "Vietnam", "Germany"], size=60),
            "category": np.random.choice(["Raw Material", "Packaging", "Electronics"], size=60),
            "demand_units": np.random.randint(200, 2000, size=60),
            "inventory_units": np.random.randint(100, 1500, size=60),
            "supplier_capacity_units": np.random.randint(500, 3000, size=60),
            "supplier_reliability": np.random.uniform(0.70, 0.99, size=60),
            "planned_lead_time_days": np.random.randint(5, 30, size=60),
            "unit_cost_usd": np.random.uniform(10.0, 500.0, size=60),
            "transport_cost_usd": np.random.uniform(100.0, 5000.0, size=60),
            "secondary_supplier_premium_pct": np.random.uniform(0.05, 0.35, size=60),
            # Derived / rolling features
            "inventory_coverage_ratio": np.random.uniform(0.2, 2.0, size=60),
            "inventory_gap_units": np.random.randint(-500, 500, size=60),
            "shortage_flag": np.random.choice([0, 1], size=60),
            "capacity_utilization": np.random.uniform(0.4, 1.2, size=60),
            "demand_pressure": np.random.uniform(0.5, 1.5, size=60),
            "supplier_risk_score": np.random.uniform(0.05, 0.85, size=60),
            "cost_per_demand_unit": np.random.uniform(1.0, 50.0, size=60),
            "secondary_cost_difference": np.random.uniform(50.0, 1000.0, size=60),
            "air_freight_premium": np.random.uniform(200.0, 2500.0, size=60),
            "previous_delay_days": np.random.randint(0, 10, size=60),
            "previous_delay_flag": np.random.choice([0, 1], size=60),
            "rolling_average_delay": np.random.uniform(0.0, 5.0, size=60),
            "rolling_average_demand": np.random.uniform(500.0, 1500.0, size=60),
            "rolling_average_inventory": np.random.uniform(400.0, 1200.0, size=60),
            "demand_outlier_flag": np.zeros(60, dtype=int),
            "inventory_outlier_flag": np.zeros(60, dtype=int),
            "cost_outlier_flag": np.zeros(60, dtype=int),
            "lead_time_outlier_flag": np.zeros(60, dtype=int),
            # Post-event leakage columns
            "actual_lead_time_days": np.random.randint(5, 40, size=60),
            "delay_days": np.random.randint(0, 15, size=60),
            "lead_time_variance_days": np.random.randint(-5, 15, size=60),
            "service_risk": np.random.choice(["LOW", "MEDIUM", "HIGH"], size=60),
            # Binary target
            "delay_occurred": np.random.choice([0, 1], size=60, p=[0.75, 0.25]),
        }
    )
    return df


# -------------------------------------------------------------------
# 1 & 2. Target validation & binary normalization
# -------------------------------------------------------------------
def test_target_validation_missing_column(sample_synthetic_dataset):
    df_missing = sample_synthetic_dataset.drop(columns=["delay_occurred"])
    with pytest.raises(ValueError, match="Required target column 'delay_occurred' is missing"):
        validate_and_normalize_target(df_missing, "delay_occurred")


def test_target_validation_non_binary(sample_synthetic_dataset):
    df_invalid = sample_synthetic_dataset.copy()
    df_invalid["delay_occurred"] = [0, 1, 2] * 20  # contains 2
    with pytest.raises(ValueError, match="non-binary values"):
        validate_and_normalize_target(df_invalid, "delay_occurred")


def test_target_normalization_boolean_and_strings():
    df_bool = pd.DataFrame({"delay_occurred": [True, False, True, False]})
    res = validate_and_normalize_target(df_bool)
    assert list(res) == [1, 0, 1, 0]

    df_str = pd.DataFrame({"delay_occurred": ["delayed", "on_time", "1", "0"]})
    res_str = validate_and_normalize_target(df_str)
    assert list(res_str) == [1, 0, 1, 0]


# -------------------------------------------------------------------
# 3 & 4. Feature selection and strict leakage exclusion
# -------------------------------------------------------------------
def test_feature_selection_excludes_leakage(sample_synthetic_dataset):
    config = DelayPredictionConfig()
    X, num_cols, cat_cols = select_pre_event_features(sample_synthetic_dataset, config)

    forbidden_cols = {
        "actual_lead_time_days",
        "delay_days",
        "lead_time_variance_days",
        "service_risk",
        "lead_time_risk",
        "delay_occurred",
    }
    for col in X.columns:
        assert col not in forbidden_cols, f"Leakage column '{col}' leaked into feature matrix!"

    assert "demand_units" in num_cols
    assert "supplier_country" in cat_cols


# -------------------------------------------------------------------
# 5 & 6. Imputation and unseen categorical values
# -------------------------------------------------------------------
def test_preprocessing_handles_missing_numerical_and_unseen_categories():
    num_features = ["demand_units", "inventory_units"]
    cat_features = ["supplier_country", "category"]

    preprocessor = build_preprocessing_pipeline(num_features, cat_features)

    # Training data
    train_df = pd.DataFrame(
        {
            "demand_units": [100.0, 200.0, np.nan, 400.0],
            "inventory_units": [50.0, np.nan, 150.0, 200.0],
            "supplier_country": ["India", "Germany", "India", "Vietnam"],
            "category": ["Electronics", "Raw Material", "Packaging", "Electronics"],
        }
    )
    preprocessor.fit(train_df)

    # Test data with unseen country & category + NaNs
    test_df = pd.DataFrame(
        {
            "demand_units": [np.nan, 500.0],
            "inventory_units": [300.0, np.nan],
            "supplier_country": ["Brazil", "India"],  # "Brazil" is unseen
            "category": ["Automotive", "Electronics"],  # "Automotive" is unseen
        }
    )

    # Must transform without error
    transformed = preprocessor.transform(test_df)
    assert not np.isnan(transformed).any(), "Transformed output contains NaNs!"
    assert transformed.shape[0] == 2


# -------------------------------------------------------------------
# 7 & 8. Model training and probability bounds
# -------------------------------------------------------------------
def test_model_training_and_probability_bounds(sample_synthetic_dataset, tmp_path):
    config = DelayPredictionConfig()
    config.model_dir = tmp_path / "models"
    config.report_dir = tmp_path / "reports"

    pipeline, best_model_name, summary, eval_artifacts = train_and_compare_models(
        sample_synthetic_dataset, config
    )

    assert isinstance(pipeline, Pipeline)
    assert best_model_name in ["LogisticRegression", "RandomForest", "XGBoost"]
    assert "validation_metrics" in summary
    assert "test_metrics" in summary

    # Check probabilities on test set
    X_test = eval_artifacts["X_test"]
    probs = pipeline.predict_proba(X_test)[:, 1]
    assert len(probs) == len(X_test)
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Probabilities outside [0, 1]!"


# -------------------------------------------------------------------
# 9. Operational risk level threshold mappings
# -------------------------------------------------------------------
def test_risk_level_threshold_mapping():
    assert map_probability_to_risk_level(0.15) == "LOW"
    assert map_probability_to_risk_level(0.299) == "LOW"
    assert map_probability_to_risk_level(0.30) == "MEDIUM"
    assert map_probability_to_risk_level(0.599) == "MEDIUM"
    assert map_probability_to_risk_level(0.60) == "HIGH"
    assert map_probability_to_risk_level(0.799) == "HIGH"
    assert map_probability_to_risk_level(0.80) == "CRITICAL"
    assert map_probability_to_risk_level(0.99) == "CRITICAL"
    assert map_probability_to_risk_level(1.0) == "CRITICAL"


# -------------------------------------------------------------------
# 10, 11 & 12. Prediction fields, serialization & zero retraining
# -------------------------------------------------------------------
def test_model_saving_reloading_and_zero_retraining(sample_synthetic_dataset, tmp_path):
    config = DelayPredictionConfig()
    config.model_dir = tmp_path / "models"
    config.report_dir = tmp_path / "reports"

    pipeline, best_model_name, summary, eval_artifacts = train_and_compare_models(
        sample_synthetic_dataset, config
    )
    saved_paths = save_model_and_metadata(
        pipeline, best_model_name, summary, eval_artifacts, config
    )

    assert Path(saved_paths["model_path"]).exists()
    assert Path(saved_paths["metadata_path"]).exists()

    # Reload predictor from disk
    predictor = DelayPredictor(
        model_path=saved_paths["model_path"],
        metadata_path=saved_paths["metadata_path"],
        config=config,
    )

    # Single prediction
    single_record = sample_synthetic_dataset.iloc[0].to_dict()
    res_single = predictor.predict_single(single_record)

    required_keys = [
        "delay_probability",
        "predicted_delay",
        "risk_score",
        "risk_level",
        "model_name",
        "model_version",
        "prediction_timestamp",
    ]
    for key in required_keys:
        assert key in res_single, f"Missing required prediction key: {key}"

    # Batch prediction
    res_batch = predictor.predict_batch(sample_synthetic_dataset.head(5))
    assert len(res_batch) == 5
    for key in required_keys:
        assert key in res_batch.columns

    # Verify zero retraining (pipeline instance remains unchanged)
    initial_weights = id(predictor._pipeline)
    predictor.predict_batch(sample_synthetic_dataset.head(5))
    assert id(predictor._pipeline) == initial_weights


# -------------------------------------------------------------------
# 13. Preprocessing fitted strictly on train only
# -------------------------------------------------------------------
def test_preprocessor_not_fitted_on_test_data(sample_synthetic_dataset):
    config = DelayPredictionConfig()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_chronologically(
        sample_synthetic_dataset, config
    )

    _, num_cols, cat_cols = select_pre_event_features(sample_synthetic_dataset, config)
    preprocessor = build_preprocessing_pipeline(num_cols, cat_cols)

    # Fit on X_train only
    preprocessor.fit(X_train)

    train_mean = preprocessor.named_transformers_["num"].named_steps["scaler"].mean_
    assert train_mean is not None
    # Verify that transforming X_test does not alter scaler mean
    mean_before = train_mean.copy()
    preprocessor.transform(X_test)
    mean_after = preprocessor.named_transformers_["num"].named_steps["scaler"].mean_
    np.testing.assert_array_equal(mean_before, mean_after)


# -------------------------------------------------------------------
# 14. Clear error handling on invalid inputs
# -------------------------------------------------------------------
def test_predict_invalid_inputs(sample_synthetic_dataset, tmp_path):
    config = DelayPredictionConfig()
    config.model_dir = tmp_path / "models"
    config.report_dir = tmp_path / "reports"

    pipeline, best_model_name, summary, eval_artifacts = train_and_compare_models(
        sample_synthetic_dataset, config
    )
    saved_paths = save_model_and_metadata(
        pipeline, best_model_name, summary, eval_artifacts, config
    )

    predictor = DelayPredictor(model_path=saved_paths["model_path"], config=config)

    # Empty DataFrame
    with pytest.raises(ValueError, match="empty or None"):
        predictor.predict_batch(pd.DataFrame())

    # Unsupported type
    with pytest.raises(TypeError):
        predict_delay(12345, model_path=saved_paths["model_path"], config=config)


# -------------------------------------------------------------------
# 15. Safe database persistence handling
# -------------------------------------------------------------------
def test_database_persistence_safe_handling():
    # DataFrame with non-existent events
    dummy_preds = pd.DataFrame(
        [
            {
                "event_id": "NON_EXISTENT_EVENT_999",
                "model_name": "supply_delay_classifier",
                "model_version": "1.0.0",
                "delay_probability": 0.85,
                "risk_score": 0.85,
                "risk_level": "CRITICAL",
            }
        ]
    )
    # Must complete safely without raising unhandled exceptions
    count, err = persist_predictions_to_database(dummy_preds)
    assert isinstance(count, int)
