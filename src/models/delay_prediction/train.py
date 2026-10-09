"""Model training, validation comparison, and pipeline serialization for Delay Prediction.

Implements:
- Model A: Logistic Regression (Interpretable Baseline)
- Model B: Random Forest Classifier (Non-linear ensemble)
- Model C: XGBoost Classifier (Gradient boosted trees)
- Rigorous validation-based model selection
- Complete pipeline serialization and metadata versioning
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.models.delay_prediction.config import DelayPredictionConfig, default_config
from src.models.delay_prediction.features import (
    build_preprocessing_pipeline,
    get_feature_names_from_preprocessor,
    select_pre_event_features,
    split_data_chronologically,
    validate_and_normalize_target,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def get_candidate_models(
    scale_pos_weight: float = 1.0, random_seed: int = 42
) -> Dict[str, BaseEstimator]:
    """Instantiate candidate classification models.

    Args:
        scale_pos_weight: Ratio of negative to positive classes for gradient boosting.
        random_seed: Reproducibility seed.

    Returns:
        Dictionary mapping model names to model instances.
    """
    models: Dict[str, BaseEstimator] = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            random_state=random_seed,
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1,
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=random_seed,
            eval_metric="logloss",
            n_jobs=-1,
        )
        logger.info("XGBoost candidate model initialized.")
    except (ImportError, Exception) as exc:
        logger.warning(f"XGBoost is not available ({exc}). Proceeding with Scikit-learn models.")

    return models


def evaluate_candidate_model(
    model: BaseEstimator,
    X_val_proc: np.ndarray,
    y_val: pd.Series,
    threshold: float = 0.50,
) -> Dict[str, float]:
    """Compute validation metrics for a candidate model.

    Args:
        model: Fitted classifier model.
        X_val_proc: Preprocessed validation feature matrix.
        y_val: Validation ground truth series.
        threshold: Decision threshold for positive prediction.

    Returns:
        Dictionary of validation metrics.
    """
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_val_proc)[:, 1]
    elif hasattr(model, "decision_function"):
        # Map decision function via sigmoid
        df_vals = model.decision_function(X_val_proc)
        y_prob = 1.0 / (1.0 + np.exp(-df_vals))
    else:
        y_prob = model.predict(X_val_proc).astype(float)

    y_pred = (y_prob >= threshold).astype(int)

    has_both_classes = len(np.unique(y_val)) > 1

    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "recall": float(recall_score(y_val, y_pred, zero_division=0)),
        "f1": float(f1_score(y_val, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_val, y_prob)) if has_both_classes else 0.0,
        "average_precision": (
            float(average_precision_score(y_val, y_prob)) if has_both_classes else 0.0
        ),
        "brier_score": float(brier_score_loss(y_val, y_prob)) if has_both_classes else 0.0,
    }
    return metrics


def train_and_compare_models(
    df: pd.DataFrame,
    config: DelayPredictionConfig = default_config,
) -> Tuple[Pipeline, str, Dict[str, Any], Dict[str, Any]]:
    """Execute complete model training and validation comparison workflow.

    Workflow:
    1. Validate target and split data chronologically.
    2. Fit ColumnTransformer on training data only.
    3. Train each candidate classifier on preprocessed training data.
    4. Evaluate candidate classifiers on validation set.
    5. Select best model based on configured validation metric (PR-AUC / average precision).
    6. Construct end-to-end full Pipeline (Preprocessor + Best Estimator).
    7. Evaluate selected model on held-out test set.

    Args:
        df: Processed historical dataset.
        config: Configuration parameters.

    Returns:
        Tuple of (Full Fitted Pipeline, Best Model Name, Comparison Results Dict, Evaluation Artifacts Dict).
    """
    logger.info("=" * 60)
    logger.info("STEP 1: Validating Target and Splitting Data Chronologically")
    logger.info("=" * 60)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data_chronologically(df, config)

    # Class imbalance analysis
    n_train_pos = int((y_train == 1).sum())
    n_train_neg = int((y_train == 0).sum())
    pos_ratio = n_train_pos / len(y_train) if len(y_train) > 0 else 0.0
    scale_pos_weight = (n_train_neg / max(1, n_train_pos))

    logger.info(
        f"Training set class distribution: 0={n_train_neg} ({100*(1-pos_ratio):.1f}%), "
        f"1={n_train_pos} ({100*pos_ratio:.1f}%). Imbalance scale_pos_weight: {scale_pos_weight:.2f}"
    )

    # Pre-event feature names
    _, num_features, cat_features = select_pre_event_features(df, config)

    logger.info("=" * 60)
    logger.info("STEP 2: Fitting Preprocessor on Training Data Only")
    logger.info("=" * 60)

    preprocessor = build_preprocessing_pipeline(num_features, cat_features)
    preprocessor.fit(X_train)

    X_train_proc = preprocessor.transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    transformed_feature_names = get_feature_names_from_preprocessor(
        preprocessor, num_features, cat_features
    )
    logger.info(f"Preprocessor produced {len(transformed_feature_names)} transformed features.")

    logger.info("=" * 60)
    logger.info("STEP 3: Training and Comparing Candidate Classifiers on Validation Set")
    logger.info("=" * 60)

    candidates = get_candidate_models(scale_pos_weight=scale_pos_weight, random_seed=config.random_seed)
    model_results: Dict[str, Dict[str, float]] = {}
    fitted_candidates: Dict[str, BaseEstimator] = {}

    for name, model in candidates.items():
        logger.info(f"Training candidate model: {name}...")
        model.fit(X_train_proc, y_train)
        fitted_candidates[name] = model

        val_metrics = evaluate_candidate_model(
            model, X_val_proc, y_val, threshold=config.classification_threshold
        )
        model_results[name] = val_metrics
        logger.info(
            f"Validation Results for {name}: "
            f"PR-AUC (Avg Precision)={val_metrics['average_precision']:.4f}, "
            f"F1={val_metrics['f1']:.4f}, "
            f"ROC-AUC={val_metrics['roc_auc']:.4f}, "
            f"Precision={val_metrics['precision']:.4f}, "
            f"Recall={val_metrics['recall']:.4f}, "
            f"Brier={val_metrics['brier_score']:.4f}"
        )

    # Select best model based on model_selection_metric (e.g., average_precision or f1)
    selection_metric = config.model_selection_metric
    best_model_name = max(
        model_results.keys(),
        key=lambda k: (model_results[k].get(selection_metric, 0.0), model_results[k].get("f1", 0.0)),
    )
    best_estimator = fitted_candidates[best_model_name]
    logger.info(
        f"Selected Best Model: '{best_model_name}' based on validation {selection_metric} "
        f"({model_results[best_model_name].get(selection_metric):.4f})"
    )

    # Build end-to-end composite pipeline
    full_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", best_estimator),
        ]
    )

    logger.info("=" * 60)
    logger.info("STEP 4: Evaluating Selected Pipeline on Untouched Test Set")
    logger.info("=" * 60)

    test_metrics = evaluate_candidate_model(
        best_estimator, X_test_proc, y_test, threshold=config.classification_threshold
    )
    logger.info(
        f"Held-Out Test Set Results for {best_model_name}: "
        f"PR-AUC={test_metrics['average_precision']:.4f}, "
        f"F1={test_metrics['f1']:.4f}, "
        f"ROC-AUC={test_metrics['roc_auc']:.4f}, "
        f"Precision={test_metrics['precision']:.4f}, "
        f"Recall={test_metrics['recall']:.4f}, "
        f"Accuracy={test_metrics['accuracy']:.4f}, "
        f"Brier={test_metrics['brier_score']:.4f}"
    )

    comparison_summary = {
        "candidate_models": model_results,
        "selected_model": best_model_name,
        "selection_metric": selection_metric,
        "validation_metrics": model_results[best_model_name],
        "test_metrics": test_metrics,
        "class_imbalance": {
            "train_positive_count": n_train_pos,
            "train_negative_count": n_train_neg,
            "train_delay_percentage": round(float(pos_ratio * 100), 2),
            "scale_pos_weight": round(float(scale_pos_weight), 2),
        },
        "dataset_split": {
            "total_records": len(df),
            "train_records": len(X_train),
            "val_records": len(X_val),
            "test_records": len(X_test),
        },
    }

    eval_artifacts = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "X_test_proc": X_test_proc,
        "preprocessor": preprocessor,
        "best_estimator": best_estimator,
        "transformed_feature_names": transformed_feature_names,
        "numerical_features": num_features,
        "categorical_features": cat_features,
    }

    return full_pipeline, best_model_name, comparison_summary, eval_artifacts


def save_model_and_metadata(
    pipeline: Pipeline,
    best_model_name: str,
    summary: Dict[str, Any],
    eval_artifacts: Dict[str, Any],
    config: DelayPredictionConfig = default_config,
) -> Dict[str, Path]:
    """Serialize model artifacts and write metadata JSON.

    Args:
        pipeline: Fitted full Scikit-learn Pipeline.
        best_model_name: Name of selected model architecture.
        summary: Summary comparison dictionary.
        eval_artifacts: Artifacts dictionary with features and transformers.
        config: Configuration with destination paths.

    Returns:
        Dictionary mapping artifact keys to their saved paths.
    """
    config.model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save full pipeline
    model_path = config.model_path
    joblib.dump(pipeline, model_path)
    logger.info(f"Saved full pipeline to {model_path}")

    # 2. Save standalone preprocessor
    preprocessor_path = config.preprocessor_path
    joblib.dump(pipeline.named_steps["preprocessor"], preprocessor_path)
    logger.info(f"Saved preprocessor pipeline to {preprocessor_path}")

    # 3. Compile comprehensive metadata
    import sklearn
    import xgboost

    metadata = {
        "model_name": config.model_name,
        "model_version": config.model_version,
        "selected_algorithm": best_model_name,
        "model_class": pipeline.named_steps["classifier"].__class__.__name__,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "training_dataset_path": str(config.dataset_path),
        "target_column": config.target_column,
        "date_column": config.date_column,
        "selected_features": eval_artifacts["numerical_features"] + eval_artifacts["categorical_features"],
        "numerical_features": eval_artifacts["numerical_features"],
        "categorical_features": eval_artifacts["categorical_features"],
        "transformed_feature_count": len(eval_artifacts["transformed_feature_names"]),
        "train_record_count": summary["dataset_split"]["train_records"],
        "val_record_count": summary["dataset_split"]["val_records"],
        "test_record_count": summary["dataset_split"]["test_records"],
        "target_class_distribution": summary["class_imbalance"],
        "model_comparison": summary["candidate_models"],
        "validation_metrics": summary["validation_metrics"],
        "test_metrics": summary["test_metrics"],
        "classification_threshold": config.classification_threshold,
        "risk_thresholds": {k: list(v) for k, v in config.risk_thresholds.items()},
        "random_seed": config.random_seed,
        "library_versions": {
            "scikit-learn": sklearn.__version__,
            "xgboost": getattr(xgboost, "__version__", "not_installed"),
            "joblib": joblib.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }

    metadata_path = config.metadata_path
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved model metadata to {metadata_path}")

    return {
        "model_path": model_path,
        "preprocessor_path": preprocessor_path,
        "metadata_path": metadata_path,
    }


def run_training_pipeline(
    dataset_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    config: DelayPredictionConfig = default_config,
) -> Tuple[Pipeline, Dict[str, Any], Dict[str, Any]]:
    """Entry point to run dataset loading, training, comparison, and serialization."""
    effective_data_path = dataset_path or config.dataset_path
    if not effective_data_path.exists():
        raise FileNotFoundError(
            f"Processed training dataset not found at '{effective_data_path}'. "
            "Please ensure Day 3 dataset generation has completed."
        )

    logger.info(f"Loading processed dataset from {effective_data_path}...")
    df = pd.read_csv(effective_data_path)
    logger.info(f"Loaded dataset with shape {df.shape}.")

    if output_dir:
        config.model_dir = output_dir

    pipeline, best_model_name, summary, eval_artifacts = train_and_compare_models(df, config)
    save_model_and_metadata(pipeline, best_model_name, summary, eval_artifacts, config)

    logger.info("Training pipeline completed successfully.")
    return pipeline, summary, eval_artifacts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and compare delay prediction models for Supply Prescript.")
    parser.add_argument("--data", type=str, default=None, help="Path to processed dataset CSV")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save trained model artifacts")
    args = parser.parse_args()

    data_p = Path(args.data) if args.data else None
    out_p = Path(args.output_dir) if args.output_dir else None

    run_training_pipeline(dataset_path=data_p, output_dir=out_p)
