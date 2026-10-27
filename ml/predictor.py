"""
Supply Prescript - Real-Time Predictor Module
Provides inference, delay probabilities, duration estimations, and risk explanations.
Author: Prasanna (ML & Optimization Lead)
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

# Ensure root directory is on PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ml.data_preparation import (
    ALL_MODEL_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    engineer_features,
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Global cached model instances
_CLASSIFIER = None
_REGRESSOR = None
_PREPROCESSOR = None
_METRICS = None


def load_artifacts():
    """Loads and caches trained ML artifacts."""
    global _CLASSIFIER, _REGRESSOR, _PREPROCESSOR, _METRICS
    if _CLASSIFIER is None or _REGRESSOR is None or _PREPROCESSOR is None:
        clf_path = os.path.join(MODELS_DIR, "delay_classifier.joblib")
        reg_path = os.path.join(MODELS_DIR, "delay_regressor.joblib")
        prep_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
        metrics_path = os.path.join(MODELS_DIR, "metrics.json")

        if not os.path.exists(clf_path) or not os.path.exists(reg_path) or not os.path.exists(prep_path):
            from ml.train_model import train_models
            artifacts = train_models(save_artifacts=True)
            _CLASSIFIER = artifacts["classifier"]
            _REGRESSOR = artifacts["regressor"]
            _PREPROCESSOR = artifacts["preprocessor"]
            _METRICS = artifacts["metrics"]
        else:
            try:
                _CLASSIFIER = joblib.load(clf_path)
                _REGRESSOR = joblib.load(reg_path)
                _PREPROCESSOR = joblib.load(prep_path)
                if os.path.exists(metrics_path):
                    with open(metrics_path, "r", encoding="utf-8") as f:
                        _METRICS = json.load(f)
            except Exception as e:
                # If model unpickling fails due to environment mismatch, auto-train compatible artifacts
                from ml.train_model import train_models
                artifacts = train_models(save_artifacts=True)
                _CLASSIFIER = artifacts["classifier"]
                _REGRESSOR = artifacts["regressor"]
                _PREPROCESSOR = artifacts["preprocessor"]
                _METRICS = artifacts["metrics"]


def get_risk_level(prob: float) -> str:
    """Classifies delay probability into standard operational risk levels."""
    if prob >= 0.75:
        return "CRITICAL"
    elif prob >= 0.50:
        return "HIGH"
    elif prob >= 0.25:
        return "MEDIUM"
    else:
        return "LOW"


def predict_delay(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main prediction interface for Supply Prescript.
    
    Args:
        input_data: Dictionary containing shipment attributes:
            - supplier_name (str)
            - product_category (str)
            - shipping_mode (str)
            - lead_time_days (float)
            - shipping_cost (float)
            - inventory_level (int)
            - order_quantity (int)
            - supplier_delay_rate (float, 0-1)
            - route_distance_km (float)
            - weather_risk_index (float, 0-1)
            - congestion_index (float, 0-1)
            
    Returns:
        Dictionary containing:
            - is_delayed (int: 0 or 1)
            - delay_probability (float: 0.0 to 1.0)
            - predicted_delay_days (float: estimated extra days)
            - expected_total_days (float: lead_time + predicted_delay)
            - risk_level (str: "LOW", "MEDIUM", "HIGH", "CRITICAL")
            - confidence_score (float: model confidence)
            - top_risk_drivers (list of dicts with driver name, value, and impact)
            - input_summary (dict of key input parameters)
    """
    load_artifacts()

    # Create a DataFrame from input dictionary
    df = pd.DataFrame([input_data])
    df = engineer_features(df)

    # Ensure all required features are present with sensible defaults
    for num_col in NUMERICAL_FEATURES:
        if num_col not in df.columns:
            df[num_col] = 0.0
    for cat_col in CATEGORICAL_FEATURES:
        if cat_col not in df.columns:
            df[cat_col] = "Standard Ground" if cat_col == "shipping_mode" else "Tier-2"

    X_raw = df[ALL_MODEL_FEATURES]
    X_trans = _PREPROCESSOR.transform(X_raw)

    # Classification inference
    prob = float(_CLASSIFIER.predict_proba(X_trans)[0, 1])
    is_delayed = int(prob >= 0.50)

    # Regression inference (predicted delay duration in days)
    reg_pred = float(_REGRESSOR.predict(X_trans)[0])
    if not is_delayed and prob < 0.35:
        predicted_delay_days = 0.0
    else:
        # Scale regression prediction smoothly with delay probability
        predicted_delay_days = round(max(0.0, reg_pred * min(1.0, prob / 0.5)), 1)
        if is_delayed and predicted_delay_days < 1.0:
            predicted_delay_days = 1.5

    lead_time = float(input_data.get("lead_time_days", 5.0))
    expected_total_days = round(lead_time + predicted_delay_days, 1)
    risk_level = get_risk_level(prob)

    # Model confidence metric (distance from decision boundary 0.5 + AUC factor)
    boundary_dist = abs(prob - 0.5) * 2.0  # 0.0 to 1.0
    auc_score = _METRICS.get("classification", {}).get("roc_auc", 0.95) if _METRICS else 0.95
    confidence_score = round(float(0.70 + 0.30 * boundary_dist * auc_score), 3)

    # Compute key risk drivers (explainability)
    drivers = []
    sup_rate = float(input_data.get("supplier_delay_rate", 0.1))
    if sup_rate > 0.20:
        drivers.append({
            "factor": "Historical Supplier Delay Rate",
            "value": f"{sup_rate * 100:.1f}%",
            "impact": "HIGH",
            "direction": "Increases Delay Risk",
        })

    weather = float(input_data.get("weather_risk_index", 0.1))
    if weather > 0.40:
        drivers.append({
            "factor": "Severe Weather Disruption Index",
            "value": f"{weather:.2f}",
            "impact": "HIGH" if weather > 0.65 else "MODERATE",
            "direction": "Increases Delay Risk",
        })

    congestion = float(input_data.get("congestion_index", 0.1))
    if congestion > 0.40:
        drivers.append({
            "factor": "Port / Corridor Congestion Index",
            "value": f"{congestion:.2f}",
            "impact": "HIGH" if congestion > 0.65 else "MODERATE",
            "direction": "Increases Delay Risk",
        })

    inv_level = float(input_data.get("inventory_level", 500))
    order_qty = float(input_data.get("order_quantity", 250))
    if inv_level < order_qty * 0.5:
        drivers.append({
            "factor": "Low Safety Stock Buffer",
            "value": f"{inv_level:.0f} units (Order: {order_qty:.0f})",
            "impact": "MODERATE",
            "direction": "Low Cushion for Delays",
        })

    if not drivers:
        drivers.append({
            "factor": "Normal Operating Conditions",
            "value": "All risk metrics within safe thresholds",
            "impact": "LOW",
            "direction": "Favorable Delivery Probability",
        })

    return {
        "is_delayed": is_delayed,
        "delay_probability": round(prob, 4),
        "delay_probability_pct": round(prob * 100, 1),
        "predicted_delay_days": predicted_delay_days,
        "expected_total_days": expected_total_days,
        "risk_level": risk_level,
        "confidence_score": confidence_score,
        "confidence_pct": round(confidence_score * 100, 1),
        "top_risk_drivers": drivers,
        "input_summary": {
            "supplier_name": input_data.get("supplier_name", "Unknown Supplier"),
            "product_category": input_data.get("product_category", "Standard Goods"),
            "shipping_mode": input_data.get("shipping_mode", "Standard Ground"),
            "lead_time_days": lead_time,
            "shipping_cost": float(input_data.get("shipping_cost", 1000.0)),
            "order_quantity": int(order_qty),
            "inventory_level": int(inv_level),
            "supplier_delay_rate": sup_rate,
            "route_distance_km": float(input_data.get("route_distance_km", 1500.0)),
            "weather_risk_index": weather,
            "congestion_index": congestion,
        },
    }


def get_model_metrics() -> Dict[str, Any]:
    """Returns the latest training and evaluation metrics."""
    load_artifacts()
    return _METRICS or {}


if __name__ == "__main__":
    sample_shipment = {
        "supplier_name": "Shenzhen Silicon Ltd",
        "supplier_tier": "Tier-2",
        "product_category": "Semiconductors & Electronics",
        "shipping_mode": "Ocean Intermodal",
        "lead_time_days": 24.0,
        "shipping_cost": 8500.0,
        "inventory_level": 80,
        "order_quantity": 500,
        "supplier_delay_rate": 0.45,
        "route_distance_km": 11500.0,
        "weather_risk_index": 0.72,
        "congestion_index": 0.68,
    }
    result = predict_delay(sample_shipment)
    print("--- SAMPLE PREDICTION ---")
    print(json.dumps(result, indent=2))
