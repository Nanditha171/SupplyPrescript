"""Schemas for Decision Management & Auditing (Day 13)."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class DecisionCreateRequest(BaseModel):
    """Request payload to select, approve, or override a recommendation."""

    optimization_run_id: int
    recommendation_id: int
    decision_maker: str = Field(..., min_length=2, max_length=150)
    decision_status: str = Field("APPROVED", pattern="^(APPROVED|REJECTED|OVERRIDDEN|CANCELLED)$")
    decision_reason: Optional[str] = Field(None, max_length=1000)
    execution_status: str = Field("NOT_STARTED", pattern="^(NOT_STARTED|IN_PROGRESS|COMPLETED|FAILED)$")


class DecisionUpdateRequest(BaseModel):
    """Request payload to update decision status or execution tracking."""

    decision_status: Optional[str] = Field(None, pattern="^(APPROVED|REJECTED|OVERRIDDEN|CANCELLED)$")
    execution_status: Optional[str] = Field(None, pattern="^(NOT_STARTED|IN_PROGRESS|COMPLETED|FAILED)$")
    decision_reason: Optional[str] = None
    executed_at: Optional[datetime] = None


class DecisionAuditEntry(BaseModel):
    """Audit record capturing the full state of a decision at an event point."""

    decision_id: str
    optimization_run_id: int
    recommendation_id: int
    strategy_type: str
    decision_maker: str
    decision_status: str
    execution_status: str
    decision_reason: Optional[str]
    selected_at: datetime
    executed_at: Optional[datetime]
    created_at: datetime

