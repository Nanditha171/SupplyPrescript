"""
Supply Prescript - FastAPI Backend & Write-Back Service
Exposes REST endpoints for ML predictions, Prescriptive Optimization, and Closed-Loop Evaluation.
Author: Prasanna (ML & Optimization Lead)
"""

import os
import sys
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure root directory is on PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from api.interfaces import (
    predict_delay,
    optimize_actions,
    validate_recommendation,
    evaluate_decision,
    log_and_evaluate_cycle,
    get_model_metrics,
    get_history,
    get_feedback_analytics,
)

app = FastAPI(
    title="Supply Prescript - Analytical Engine API",
    description="Closed-Loop Prescriptive Analytics API for Supply Chain Disruption Management.",
    version="1.0.0",
)

# Enable CORS for React / Retool frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ShipmentInput(BaseModel):
    shipment_id: Optional[str] = "SHP-10001"
    supplier_name: str = "GlobalTech Micro"
    supplier_tier: Optional[str] = "Tier-1"
    product_category: str = "Semiconductors & Electronics"
    shipping_mode: str = "Standard Ground"
    lead_time_days: float = Field(default=12.0, ge=0.5)
    shipping_cost: float = Field(default=6500.0, ge=10.0)
    inventory_level: int = Field(default=350, ge=0)
    order_quantity: int = Field(default=200, ge=1)
    supplier_delay_rate: float = Field(default=0.15, ge=0.0, le=1.0)
    route_distance_km: float = Field(default=3500.0, ge=10.0)
    weather_risk_index: float = Field(default=0.30, ge=0.0, le=1.0)
    congestion_index: float = Field(default=0.25, ge=0.0, le=1.0)


class OptimizeRequest(BaseModel):
    shipment_data: ShipmentInput
    prediction: Dict[str, Any]
    constraints: Optional[Dict[str, Any]] = None


class ExecuteDecisionRequest(BaseModel):
    shipment_data: ShipmentInput
    selected_option: Dict[str, Any]
    prediction: Dict[str, Any]


class EvaluateDecisionRequest(BaseModel):
    shipment_data: ShipmentInput
    prediction: Dict[str, Any]
    decision: Dict[str, Any]
    actual_outcome: Dict[str, Any]


@app.get("/")
def root():
    return {
        "project": "Supply Prescript",
        "description": "Closed-Loop Prescriptive Analytics System",
        "author": "Prasanna (ML & Optimization Lead)",
        "status": "OPERATIONAL",
        "docs_url": "/docs",
    }


@app.post("/predict")
def api_predict_delay(payload: ShipmentInput):
    """Predicts shipment delay probability, estimated delay days, and risk level."""
    try:
        result = predict_delay(payload.model_dump())
        return {"status": "SUCCESS", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize")
def api_optimize_actions(payload: OptimizeRequest):
    """Generates 3 prescriptive alternative mitigation actions using PuLP MILP solver."""
    try:
        result = optimize_actions(
            payload.shipment_data.model_dump(),
            payload.prediction,
            payload.constraints,
        )
        return {"status": "SUCCESS", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/execute-decision")
def api_execute_decision(payload: ExecuteDecisionRequest):
    """Simulates or records the write-back of an executed operational decision."""
    try:
        return {
            "status": "SUCCESS",
            "message": f"Decision {payload.selected_option.get('name')} executed and logged successfully.",
            "execution_details": {
                "shipment_id": payload.shipment_data.shipment_id,
                "selected_option": payload.selected_option,
                "status": "IN_TRANSIT_MITIGATED",
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate-decision")
def api_evaluate_decision(payload: EvaluateDecisionRequest):
    """Evaluates the closed-loop outcome comparing predicted vs actual cost and delivery."""
    try:
        result = log_and_evaluate_cycle(
            payload.shipment_data.model_dump(),
            payload.prediction,
            payload.decision,
            payload.actual_outcome,
        )
        return {"status": "SUCCESS", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def api_get_metrics():
    """Returns model training and evaluation metrics."""
    return {"status": "SUCCESS", "data": get_model_metrics()}


@app.get("/history")
def api_get_history():
    """Returns full history of closed-loop decisions and evaluations."""
    return {
        "status": "SUCCESS",
        "history": get_history(),
        "analytics": get_feedback_analytics(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
