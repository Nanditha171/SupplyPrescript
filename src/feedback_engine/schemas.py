"""Schemas for Closed-Loop Feedback & Learning Engine (Day 15)."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class FeedbackCreationRequest(BaseModel):
    """Payload to trigger closed-loop feedback learning from an outcome."""

    outcome_id: int
    alpha: Optional[float] = Field(0.20, ge=0.01, le=1.0)
    min_observations: Optional[int] = Field(3, ge=1, le=50)


class LearnedMultiplierState(BaseModel):
    """State of learned multiplier for a specific operational recovery strategy."""

    strategy_type: str
    baseline_multiplier: float = 1.0
    current_multiplier: float
    observation_count: int
    min_observations_required: int
    is_active: bool
    last_updated: datetime
    historical_ratios: List[float]


class FeedbackSummary(BaseModel):
    """Full feedback learning response with updated parameters and audit trails."""

    feedback_id: int
    outcome_id: int
    optimization_run_id: int
    strategy_type: str
    cost_error: float
    delay_error: float
    risk_error: float
    current_actual_to_estimated_ratio: float
    learned_multiplier: float
    observation_count: int
    safeguard_status: str
    learning_signal: str

