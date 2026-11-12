"""Schemas and data models for End-to-End Workflow Orchestration (Day 18)."""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class WorkflowStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class WorkflowStepResult(BaseModel):
    """Result status of an individual workflow stage."""

    step_name: str
    step_number: int
    status: str
    started_at: datetime
    completed_at: datetime
    execution_time_ms: float
    output_summary: Dict[str, Any]
    error_message: Optional[str] = None


class WorkflowExecutionResult(BaseModel):
    """Complete end-to-end execution audit report."""

    workflow_id: str
    event_id: str
    overall_status: WorkflowStatus
    started_at: datetime
    completed_at: Optional[datetime]
    total_execution_time_ms: float
    steps: List[WorkflowStepResult]
    prediction_summary: Optional[Dict[str, Any]] = None
    risk_summary: Optional[Dict[str, Any]] = None
    optimization_run_id: Optional[str] = None
    recommended_strategy: Optional[str] = None
    decision_id: Optional[str] = None
    execution_status: Optional[str] = None
    outcome_summary: Optional[Dict[str, Any]] = None
    feedback_summary: Optional[Dict[str, Any]] = None
    error_detail: Optional[str] = None

