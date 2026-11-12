"""Prescriptive Optimization Solver endpoints (Day 10)."""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import select

from src.api.dependencies import get_db
from src.database.models import SupplyEvent, Prediction, RiskAssessment, OptimizationRun
from src.database.schemas import OptimizationRunResponse
from src.optimization.optimization_service import OptimizationService
from src.recommendation_engine.recommendation_service import RecommendationService

router = APIRouter(prefix="/optimization", tags=["Optimization"])
opt_service = OptimizationService()
rec_service = RecommendationService(opt_service)


@router.get("", response_model=List[OptimizationRunResponse])
def get_optimization_runs(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """List historical prescriptive optimization runs."""
    stmt = select(OptimizationRun).order_by(OptimizationRun.created_at.desc()).offset(skip).limit(limit)
    return list(db.execute(stmt).scalars().all())


@router.get("/{run_id}", response_model=OptimizationRunResponse)
def get_optimization_run(run_id: str, db: Session = Depends(get_db)):
    """Retrieve optimization run by run_id string or integer ID."""
    if run_id.isdigit():
        run = db.get(OptimizationRun, int(run_id))
    else:
        stmt = select(OptimizationRun).where(OptimizationRun.optimization_run_id == run_id)
        run = db.execute(stmt).scalars().first()

    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Optimization run '{run_id}' not found.")
    return run


@router.post("/{event_id}", response_model=OptimizationRunResponse, status_code=status.HTTP_201_CREATED)
def run_optimization(
    event_id: str,
    objective_type: str = Query("MIN_COST", pattern="^(MIN_COST|MIN_RISK|MIN_DELAY|BALANCED)$"),
    air_freight_base_cost: Optional[float] = None,
    secondary_premium_rate: Optional[float] = None,
    db: Session = Depends(get_db),
):
    """Execute PuLP MILP solver to formulate and solve optimal recovery strategy."""
    stmt = select(SupplyEvent).where(SupplyEvent.event_id == event_id)
    event = db.execute(stmt).scalars().first()
    if not event:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Shipment '{event_id}' not found.")

    # Retrieve or generate prediction
    pred_stmt = select(Prediction).where(Prediction.event_id == event.id).order_by(Prediction.created_at.desc())
    pred = db.execute(pred_stmt).scalars().first()
    if not pred:
        from src.api.services.prediction_service import PredictionService
        pred = PredictionService().predict_for_shipment(db, event_id)

    # Retrieve risk assessment if present
    risk_stmt = select(RiskAssessment).where(RiskAssessment.event_id == event.id).order_by(RiskAssessment.created_at.desc())
    risk = db.execute(risk_stmt).scalars().first()

    # Formulate optimization input
    opt_inp = opt_service.build_optimization_input_from_event(
        db=db,
        event=event,
        prediction=pred,
        risk=risk,
        objective_type=objective_type,
        air_freight_base_cost=air_freight_base_cost,
        secondary_premium_rate=secondary_premium_rate,
    )

    db_run, output = opt_service.execute_and_persist(db, opt_inp, event.id, pred.id)

    # Automatically generate and persist recommendations for this run
    try:
        rec_service.generate_and_persist_recommendations(
            db=db,
            optimization_run=db_run,
            ranking_mode=objective_type,
            air_freight_base_cost=air_freight_base_cost,
            secondary_premium_rate=secondary_premium_rate,
        )
    except Exception as e:
        # Non-fatal error logging
        pass

    return db_run

