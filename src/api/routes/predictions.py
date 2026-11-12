"""Machine learning delay prediction endpoints."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.database.schemas import PredictionResponse
from src.api.services.prediction_service import PredictionService

router = APIRouter(prefix="/predictions", tags=["Predictions"])
service = PredictionService()


@router.get("", response_model=List[PredictionResponse])
def get_predictions(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """List historical machine learning inference predictions."""
    return service.list_predictions(db, skip=skip, limit=limit)


@router.get("/{event_id}", response_model=PredictionResponse)
def get_prediction_for_event(event_id: str, db: Session = Depends(get_db)):
    """Retrieve existing delay prediction for a specific event_id."""
    pred = service.get_prediction_by_event(db, event_id)
    if not pred:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No prediction found for event '{event_id}'.")
    return pred


@router.post("/{event_id}", response_model=PredictionResponse, status_code=status.HTTP_201_CREATED)
def run_prediction_for_event(event_id: str, db: Session = Depends(get_db)):
    """Trigger ML inference for a shipment and return delay probability and risk level."""
    try:
        return service.predict_for_shipment(db, event_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction failed: {str(e)}")

