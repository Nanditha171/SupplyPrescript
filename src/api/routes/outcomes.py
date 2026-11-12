"""Outcome Evaluation Engine endpoints (Day 14)."""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.database.schemas import OutcomeResponse
from src.outcome_engine.schemas import OutcomeRecordRequest
from src.outcome_engine.outcome_service import OutcomeService, OutcomeEvaluationError

router = APIRouter(prefix="/outcomes", tags=["Outcomes"])
service = OutcomeService()


@router.get("", response_model=List[OutcomeResponse])
def get_outcomes(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """List recorded post-execution outcomes."""
    return service.list_outcomes(db, skip=skip, limit=limit)


@router.get("/summary")
def get_outcomes_summary(db: Session = Depends(get_db)):
    """Get aggregated operational KPI summary of executed recovery strategies."""
    return service.get_aggregate_summary(db)


@router.get("/{decision_id}", response_model=OutcomeResponse)
def get_outcome_for_decision(decision_id: str, db: Session = Depends(get_db)):
    """Retrieve outcome evaluation by decision ID."""
    if not decision_id.isdigit():
        from src.database.crud import get_decision_by_code
        dec = get_decision_by_code(db, decision_id)
        if not dec:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Decision '{decision_id}' not found.")
        d_id = dec.id
    else:
        d_id = int(decision_id)

    outcome = service.get_outcome_by_decision_id(db, d_id)
    if not outcome:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No outcome recorded for decision '{decision_id}'.")
    return outcome


@router.post("", response_model=OutcomeResponse, status_code=status.HTTP_201_CREATED)
def record_outcome(payload: OutcomeRecordRequest, db: Session = Depends(get_db)):
    """Record post-execution actuals and compute operational variances."""
    try:
        return service.record_outcome(db, payload)
    except OutcomeEvaluationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to record outcome: {str(e)}")

