"""
Unit and Integration Tests for ML Module
"""

import pytest
import os
import sys
import pandas as pd

# Ensure root directory is on PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ml.data_preparation import load_data, engineer_features, prepare_datasets, ALL_MODEL_FEATURES
from ml.predictor import predict_delay, get_model_metrics


def test_data_loading_and_features():
    df = load_data()
    assert len(df) > 500, "Dataset should contain at least 500 records."
    assert "is_delayed" in df.columns
    assert "delay_days" in df.columns

    df_eng = engineer_features(df)
    assert "lead_time_to_distance_ratio" in df_eng.columns
    assert "inventory_coverage_ratio" in df_eng.columns
    assert "risk_multiplier" in df_eng.columns


def test_model_metrics_available():
    metrics = get_model_metrics()
    assert "classification" in metrics
    assert "regression" in metrics
    assert metrics["classification"]["accuracy"] > 0.70
    assert metrics["classification"]["roc_auc"] > 0.80
    assert metrics["regression"]["mae"] >= 0.0


def test_predict_delay_high_risk():
    high_risk_input = {
        "supplier_name": "Shenzhen Silicon Ltd",
        "supplier_tier": "Tier-2",
        "product_category": "Semiconductors & Electronics",
        "shipping_mode": "Ocean Intermodal",
        "lead_time_days": 28.0,
        "shipping_cost": 12000.0,
        "inventory_level": 50,
        "order_quantity": 800,
        "supplier_delay_rate": 0.55,
        "route_distance_km": 12500.0,
        "weather_risk_index": 0.85,
        "congestion_index": 0.78,
    }
    result = predict_delay(high_risk_input)
    assert result["is_delayed"] == 1
    assert result["delay_probability"] >= 0.60
    assert result["predicted_delay_days"] > 2.0
    assert result["risk_level"] in ["HIGH", "CRITICAL"]
    assert len(result["top_risk_drivers"]) > 0


def test_predict_delay_low_risk():
    low_risk_input = {
        "supplier_name": "Bavaria Logistics Parts",
        "supplier_tier": "Tier-1",
        "product_category": "Automotive Assemblies",
        "shipping_mode": "Standard Ground",
        "lead_time_days": 5.0,
        "shipping_cost": 1500.0,
        "inventory_level": 1200,
        "order_quantity": 100,
        "supplier_delay_rate": 0.04,
        "route_distance_km": 800.0,
        "weather_risk_index": 0.05,
        "congestion_index": 0.08,
    }
    result = predict_delay(low_risk_input)
    assert result["delay_probability"] < 0.45
    assert result["risk_level"] in ["LOW", "MEDIUM"]
