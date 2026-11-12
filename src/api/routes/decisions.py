"""Decision Tracking and Management endpoints (Day 13)."""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.database.schemas import DecisionResponse
from src.decision_management.schemas import DecisionCreateRequest, DecisionUpdateRequest
from src.decision_management.decision_service import DecisionService
from src.decision_management.decision_validation import DecisionValidationError

router = APIRouter(prefix="/decisions", tags=["Decisions"])
service = DecisionService()


@router.get("", response_model=List[DecisionResponse])
def get_decisions(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """List recorded managerial decisions and execution statuses."""
    return service.list_decisions(db, skip=skip, limit=limit)


@router.get("/{decision_id}", response_model=DecisionResponse)
def get_decision(decision_id: str, db: Session = Depends(get_db)):
    """Retrieve decision by unique decision_id string or database integer ID."""
    if decision_id.isdigit():
        dec = service.get_decision(db, int(decision_id))
    else:
        dec = service.get_decision_by_code(db, decision_id)

    if not dec:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Decision '{decision_id}' not found.")
    return dec


@router.post("", response_model=DecisionResponse, status_code=status.HTTP_201_CREATED)
def record_decision(payload: DecisionCreateRequest, db: Session = Depends(get_db)):
    """Record manager's selection, approval, or override with justification."""
    try:
        return service.record_decision(db, payload)
    except DecisionValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to record decision: {str(e)}")


@router.put("/{decision_id}", response_model=DecisionResponse)
def update_decision(decision_id: str, payload: DecisionUpdateRequest, db: Session = Depends(get_db)):
    """Update decision state or track execution lifecycle (NOT_STARTED -> IN_PROGRESS -> COMPLETED)."""
    # Resolve integer ID
    db_id = int(decision_id) if decision_id.isdigit() else None
    if db_id is None:
        dec = service.get_decision_by_code(db, decision_id)
        if not dec:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Decision '{decision_id}' not found.")
        db_id = dec.id

    try:
        return service.update_decision(db, db_id, payload)
    except DecisionValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to update decision: {str(e)}")

