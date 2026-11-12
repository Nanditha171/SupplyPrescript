"""Closed-Loop Feedback and Learning Engine endpoints (Day 15)."""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.database.schemas import FeedbackResponse
from src.feedback_engine.feedback_service import FeedbackService, FeedbackSafeguardError
from src.feedback_engine.performance_tracker import get_strategy_performance_metrics

router = APIRouter(prefix="/feedback", tags=["Feedback & Learning"])
service = FeedbackService()


@router.get("", response_model=List[FeedbackResponse])
def get_feedback_records(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """List historical feedback signals and calibrated parameter adjustments."""
    return service.list_feedbacks(db, skip=skip, limit=limit)


@router.get("/metrics")
def get_feedback_performance_metrics(db: Session = Depends(get_db)):
    """Retrieve strategy-wise cost accuracy and observation counts."""
    return get_strategy_performance_metrics(db)


@router.post("/{outcome_id}", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
def trigger_feedback_for_outcome(
    outcome_id: int,
    alpha: Optional[float] = Query(None, ge=0.01, le=1.0, description="Optional custom learning rate"),
    min_observations: Optional[int] = Query(None, ge=1, le=50, description="Optional custom minimum sample threshold"),
    db: Session = Depends(get_db),
):
    """Trigger closed-loop feedback learning, calculate EWMA multiplier, and apply safeguards."""
    try:
        return service.process_outcome_feedback(
            db=db,
            outcome_id=outcome_id,
            alpha=alpha,
            min_observations=min_observations,
        )
    except FeedbackSafeguardError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Feedback processing failed: {str(e)}")

