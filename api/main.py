from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --------------------------------------------------
# IMPORT MACHINE LEARNING AND DATABASE COMPONENTS
# --------------------------------------------------

from src.ml.predict import predict_delay
from src.database.connection import SessionLocal
from src.database.models import ShipmentPrediction


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="SupplyPrescript API",
    description=(
        "Supply-chain delay prediction, "
        "PostgreSQL storage, and prediction history"
    ),
    version="1.0.0",
)


# --------------------------------------------------
# SHIPMENT INPUT MODEL
# --------------------------------------------------

class ShipmentInput(BaseModel):
    demand_units: float = Field(ge=0)
    inventory_units: float = Field(ge=0)
    supplier_capacity_units: float = Field(ge=0)
    supplier_reliability: float = Field(ge=0, le=1)
    planned_lead_time_days: float = Field(ge=0)
    unit_cost_usd: float = Field(ge=0)
    transport_cost_usd: float = Field(ge=0)
    secondary_supplier_premium_pct: float = Field(ge=0)
    inventory_coverage_ratio: float = Field(ge=0)
    inventory_gap_units: float
    capacity_utilization: float = Field(ge=0)
    cost_per_demand_unit: float = Field(ge=0)
    secondary_cost_difference: float
    air_freight_premium: float = Field(ge=0)
    supplier_risk_score: float
    demand_pressure: float
    shortage_flag: int = Field(ge=0, le=1)
    previous_delay_days: float = Field(ge=0)
    previous_delay_flag: int = Field(ge=0, le=1)
    rolling_average_delay: float = Field(ge=0)
    rolling_average_demand: float = Field(ge=0)
    rolling_average_inventory: float = Field(ge=0)
    demand_outlier_flag: int = Field(ge=0, le=1)
    inventory_outlier_flag: int = Field(ge=0, le=1)
    cost_outlier_flag: int = Field(ge=0, le=1)
    lead_time_outlier_flag: int = Field(ge=0, le=1)

    supplier_id: str
    supplier_country: str
    product_id: str
    category: str


# --------------------------------------------------
# HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "project": "SupplyPrescript",
        "status": "running",
        "message": "SupplyPrescript API is ready",
    }


# --------------------------------------------------
# HEALTH CHECK ENDPOINT
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
    }


# --------------------------------------------------
# PREDICT SHIPMENT DELAY AND SAVE TO POSTGRESQL
# --------------------------------------------------

@app.post("/predict")
def predict_shipment(shipment: ShipmentInput):
    db = SessionLocal()

    try:
        # Convert the request to a dictionary
        shipment_data = shipment.model_dump()

        # Generate the machine-learning prediction
        result = predict_delay(shipment_data)

        # Create a database record
        record = ShipmentPrediction(
            supplier_id=shipment.supplier_id,
            product_id=shipment.product_id,
            category=shipment.category,
            demand_units=shipment.demand_units,
            inventory_units=shipment.inventory_units,
            planned_lead_time_days=shipment.planned_lead_time_days,
            prediction=result["prediction"],
            delay_probability=result["delay_probability"],
        )

        # Save the record to PostgreSQL
        db.add(record)
        db.commit()
        db.refresh(record)

        # Return the prediction and saved record ID
        return {
            **result,
            "saved_to_database": True,
            "prediction_id": record.id,
        }

    except ValueError as exc:
        db.rollback()

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        db.rollback()

        print(f"Prediction/database error: {exc}")

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction or database save failed. "
                "Check the server logs."
            ),
        ) from exc

    finally:
        db.close()


# --------------------------------------------------
# RETRIEVE PREDICTION HISTORY
# --------------------------------------------------

@app.get("/predictions/history")
def get_prediction_history():
    db = SessionLocal()

    try:
        records = (
            db.query(ShipmentPrediction)
            .order_by(ShipmentPrediction.id.desc())
            .limit(100)
            .all()
        )

        predictions = []

        for record in records:
            predictions.append(
                {
                    "id": record.id,
                    "supplier_id": record.supplier_id,
                    "product_id": record.product_id,
                    "category": record.category,
                    "demand_units": record.demand_units,
                    "inventory_units": record.inventory_units,
                    "planned_lead_time_days": (
                        record.planned_lead_time_days
                    ),
                    "prediction": record.prediction,
                    "delay_probability": record.delay_probability,
                    "created_at": (
                        record.created_at.isoformat()
                        if record.created_at
                        else None
                    ),
                }
            )

        return {
            "total_records": len(predictions),
            "predictions": predictions,
        }

    except Exception as exc:
        print(f"Prediction history error: {exc}")

        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve prediction history.",
        ) from exc

    finally:
        db.close()