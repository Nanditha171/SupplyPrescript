"""
Supply Prescript - Model Training Pipeline
Trains XGBoost Classifier (delay probability) and XGBoost Regressor (delay duration).
Author: Prasanna (ML & Optimization Lead)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import sys
from typing import Dict, Any, List, Optional

# Ensure root directory is on PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from ml.data_preparation import prepare_datasets, ALL_MODEL_FEATURES

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def train_models(save_artifacts: bool = True) -> Dict[str, Any]:
    """
    Trains predictive models (XGBoost or GradientBoosting), computes comprehensive metrics,
    and saves serialized artifacts for runtime inference.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(">>> Loading and preparing dataset...")
    data = prepare_datasets()

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train_cls = data["y_train_cls"]
    y_test_cls = data["y_test_cls"]
    y_train_reg = data["y_train_reg"]
    y_test_reg = data["y_test_reg"]
    preprocessor = data["preprocessor"]
    feature_names = data["feature_names"]

    # 1. Train Classification Model
    if HAS_XGB:
        print(">>> Training XGBoost Classifier (is_delayed)...")
        clf = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            eval_metric="logloss",
        )
    else:
        print(">>> Training GradientBoosting Classifier (is_delayed)...")
        clf = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
        )
    clf.fit(X_train, y_train_cls)

    y_pred_cls = clf.predict(X_test)
    y_prob_cls = clf.predict_proba(X_test)[:, 1]

    clf_metrics = {
        "accuracy": round(float(accuracy_score(y_test_cls, y_pred_cls)), 4),
        "precision": round(float(precision_score(y_test_cls, y_pred_cls, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test_cls, y_pred_cls, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test_cls, y_pred_cls, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test_cls, y_prob_cls)), 4),
        "confusion_matrix": confusion_matrix(y_test_cls, y_pred_cls).tolist(),
    }

    # 2. Train Regression Model
    if HAS_XGB:
        print(">>> Training XGBoost Regressor (delay_days)...")
        reg = XGBRegressor(
            n_estimators=180,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
        )
    else:
        print(">>> Training GradientBoosting Regressor (delay_days)...")
        reg = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.06,
            random_state=42,
        )
    reg.fit(X_train, y_train_reg)

    y_pred_reg = reg.predict(X_test)
    # Clip negative delay predictions
    y_pred_reg = np.clip(y_pred_reg, 0, None)

    reg_metrics = {
        "mae": round(float(mean_absolute_error(y_test_reg, y_pred_reg)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))), 4),
        "r2_score": round(float(r2_score(y_test_reg, y_pred_reg)), 4),
    }

    # 3. Extract Feature Importances
    importances = clf.feature_importances_
    feat_imp = [
        {"feature": name, "importance": round(float(imp), 4)}
        for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    ]

    all_metrics = {
        "classification": clf_metrics,
        "regression": reg_metrics,
        "feature_importances": feat_imp,
        "training_samples": int(X_train.shape[0]),
        "testing_samples": int(X_test.shape[0]),
        "feature_names": feature_names,
    }

    print("\n==========================================")
    print("MODEL EVALUATION RESULTS")
    print("==========================================")
    print(f"Classification Accuracy : {clf_metrics['accuracy'] * 100:.2f}%")
    print(f"Classification Precision: {clf_metrics['precision'] * 100:.2f}%")
    print(f"Classification Recall   : {clf_metrics['recall'] * 100:.2f}%")
    print(f"Classification F1 Score : {clf_metrics['f1_score'] * 100:.2f}%")
    print(f"Classification ROC-AUC  : {clf_metrics['roc_auc']:.4f}")
    print("------------------------------------------")
    print(f"Regression MAE          : {reg_metrics['mae']:.2f} days")
    print(f"Regression RMSE         : {reg_metrics['rmse']:.2f} days")
    print(f"Regression R² Score     : {reg_metrics['r2_score']:.4f}")
    print("==========================================\n")

    if save_artifacts:
        clf_path = os.path.join(MODELS_DIR, "delay_classifier.joblib")
        reg_path = os.path.join(MODELS_DIR, "delay_regressor.joblib")
        prep_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
        metrics_path = os.path.join(MODELS_DIR, "metrics.json")

        joblib.dump(clf, clf_path)
        joblib.dump(reg, reg_path)
        joblib.dump(preprocessor, prep_path)

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2)

        print(f"Saved model artifacts to {MODELS_DIR}")

    return {
        "classifier": clf,
        "regressor": reg,
        "preprocessor": preprocessor,
        "metrics": all_metrics,
    }


if __name__ == "__main__":
    train_models()
