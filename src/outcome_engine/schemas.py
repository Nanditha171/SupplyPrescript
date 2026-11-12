"""Schemas for Outcome Evaluation Engine (Day 14)."""

from datetime import date, datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class OutcomeRecordRequest(BaseModel):
    """Payload to record post-execution actual outcome metrics."""

    decision_id: int
    actual_cost: float = Field(..., ge=0.0)
    actual_delay_days: float = Field(..., ge=0.0)
    actual_service_level: float = Field(..., ge=0.0, le=1.0)
    actual_risk: float = Field(..., ge=0.0, le=1.0)
    actual_quantity_delivered: int = Field(..., ge=0)
    evaluation_date: Optional[date] = None
    notes: Optional[str] = None


class OutcomeVarianceSummary(BaseModel):
    """Detailed expected vs actual comparison metrics."""

    decision_id: int
    recommendation_id: int
    strategy_type: str
    estimated_cost: float
    actual_cost: float
    cost_variance: float
    cost_variance_pct: float
    is_cost_favorable: bool

    estimated_delay_days: float
    actual_delay_days: float
    delay_variance: float
    is_delay_favorable: bool

    estimated_service_level: float
    actual_service_level: float
    service_level_variance: float
    is_service_favorable: bool

    outcome_status: str
    evaluation_notes: Optional[str] = None

