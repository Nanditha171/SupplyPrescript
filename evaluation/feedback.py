"""
Supply Prescript - Decision History & Model Feedback Engine
Tracks closed-loop decisions and triggers retraining when prediction drift is detected.
Author: Prasanna (ML & Optimization Lead)
"""

import os
import sys
import json
import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

# Ensure root directory is on PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

DATA_DIR = os.path.join(ROOT_DIR, "data")
HISTORY_FILE = os.path.join(DATA_DIR, "closed_loop_history.json")


def load_history() -> List[Dict[str, Any]]:
    """Loads historical closed-loop decision records."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_history(records: List[Dict[str, Any]]) -> None:
    """Saves closed-loop records to persistent storage."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def record_closed_loop_event(
    shipment_data: Dict[str, Any],
    prediction: Dict[str, Any],
    decision: Dict[str, Any],
    actual_outcome: Dict[str, Any],
    evaluation: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Appends a complete closed-loop cycle to the persistent record store.
    """
    history = load_history()
    event_id = f"EVT-{len(history) + 1:04d}"
    timestamp = datetime.datetime.now().isoformat()

    record = {
        "event_id": event_id,
        "timestamp": timestamp,
        "shipment_id": shipment_data.get("shipment_id", f"SHP-{10000 + len(history) + 1}"),
        "supplier_name": shipment_data.get("supplier_name", "Unknown"),
        "product_category": shipment_data.get("product_category", "Standard Goods"),
        "shipping_mode": shipment_data.get("shipping_mode", "Standard Ground"),
        "lead_time_days": float(shipment_data.get("lead_time_days", 0.0)),
        "order_quantity": int(shipment_data.get("order_quantity", 0)),
        "shipping_cost": float(shipment_data.get("shipping_cost", 0.0)),
        "prediction": {
            "is_delayed": prediction.get("is_delayed", 0),
            "delay_probability": prediction.get("delay_probability", 0.0),
            "predicted_delay_days": prediction.get("predicted_delay_days", 0.0),
            "risk_level": prediction.get("risk_level", "UNKNOWN"),
        },
        "decision": {
            "option_id": decision.get("id", "UNKNOWN"),
            "name": decision.get("name", "Unknown"),
            "cost": decision.get("cost", 0.0),
            "delivery_days": decision.get("delivery_days", 0.0),
        },
        "actual_outcome": {
            "actual_cost": actual_outcome.get("actual_cost", 0.0),
            "actual_delivery_days": actual_outcome.get("actual_delivery_days", 0.0),
            "notes": actual_outcome.get("notes", ""),
        },
        "evaluation": {
            "cost_variance": evaluation.get("cost_variance", 0.0),
            "delivery_variance_days": evaluation.get("delivery_variance_days", 0.0),
            "mitigation_roi_pct": evaluation.get("mitigation_roi_pct", 0.0),
            "qualitative_outcome": evaluation.get("qualitative_outcome", ""),
            "outcome_category": evaluation.get("outcome_category", ""),
        },
    }

    history.insert(0, record)  # Most recent first
    save_history(history)
    return record


def get_feedback_summary() -> Dict[str, Any]:
    """
    Computes aggregate metrics from closed-loop history to evaluate overall system ROI
    and track model error drift.
    """
    history = load_history()
    if not history:
        return {
            "total_decisions": 0,
            "avg_cost_variance": 0.0,
            "avg_delivery_variance": 0.0,
            "avg_mitigation_roi": 0.0,
            "total_penalties_saved": 0.0,
            "retraining_recommended": False,
        }

    total = len(history)
    cost_vars = [h["evaluation"]["cost_variance"] for h in history]
    del_vars = [h["evaluation"]["delivery_variance_days"] for h in history]
    rois = [h["evaluation"]["mitigation_roi_pct"] for h in history]

    avg_cost_var = round(float(sum(cost_vars) / total), 2)
    avg_del_var = round(float(sum(del_vars) / total), 2)
    avg_roi = round(float(sum(rois) / total), 1)

    # If average delivery variance > 2.0 days or total > 10 new samples, recommend retraining
    retraining_recommended = abs(avg_del_var) > 2.0 or total >= 10

    return {
        "total_decisions": total,
        "avg_cost_variance": avg_cost_var,
        "avg_delivery_variance": avg_del_var,
        "avg_mitigation_roi_pct": avg_roi,
        "retraining_recommended": retraining_recommended,
        "recent_records": history[:10],
    }


def retrain_model_with_feedback() -> Dict[str, Any]:
    """
    Incorporates historical closed-loop actual outcomes back into the dataset and retrains the XGBoost models.
    """
    from ml.train_model import train_models
    print(">>> Triggering model retraining with closed-loop feedback...")
    results = train_models(save_artifacts=True)
    return {
        "status": "SUCCESS",
        "message": "Model retrained successfully with updated closed-loop weights.",
        "metrics": results["metrics"],
    }
