"""Recommendation Engine endpoints (Day 11)."""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import select

from src.api.dependencies import get_db
from src.database.models import OptimizationRun, Recommendation
from src.recommendation_engine.recommendation_service import RecommendationService

router = APIRouter(prefix="/recommendations", tags=["Recommendations"])
service = RecommendationService()


@router.get("/{run_id}")
def get_recommendations_for_run(
    run_id: str,
    ranking_mode: Optional[str] = Query(None, pattern="^(MIN_COST|MIN_RISK|MIN_DELAY|BALANCED)$"),
    db: Session = Depends(get_db),
):
    """Retrieve ranked alternatives, business explanations, and trade-off comparison matrix."""
    if run_id.isdigit():
        run = db.get(OptimizationRun, int(run_id))
    else:
        stmt = select(OptimizationRun).where(OptimizationRun.optimization_run_id == run_id)
        run = db.execute(stmt).scalars().first()

    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Optimization run '{run_id}' not found.")

    comparison = service.generate_and_persist_recommendations(
        db=db,
        optimization_run=run,
        ranking_mode=ranking_mode,
    )
    return comparison.to_dict()
