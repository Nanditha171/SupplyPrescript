"""Operational Risk Assessment endpoints."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.database.schemas import RiskAssessmentResponse
from src.api.services.risk_service import RiskService

router = APIRouter(prefix="/risks", tags=["Risk Assessment"])
service = RiskService()


@router.get("", response_model=List[RiskAssessmentResponse])
def get_risk_assessments(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """List operational risk assessments."""
    return service.list_risk_assessments(db, skip=skip, limit=limit)


@router.get("/{event_id}", response_model=RiskAssessmentResponse)
def get_risk_for_event(event_id: str, db: Session = Depends(get_db)):
    """Retrieve operational risk assessment for a specific shipment."""
    risk = service.get_risk_assessment_by_event(db, event_id)
    if not risk:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No risk assessment found for event '{event_id}'.")
    return risk


@router.post("/{event_id}", response_model=RiskAssessmentResponse, status_code=status.HTTP_201_CREATED)
def run_risk_assessment(event_id: str, db: Session = Depends(get_db)):
    """Compute multidimensional operational risk (delay, supplier, inventory, capacity, service)."""
    try:
        return service.assess_risk_for_shipment(db, event_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Risk assessment failed: {str(e)}")

