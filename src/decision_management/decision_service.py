"""Core Decision Management Service for recording and executing manager choices (Day 13)."""

import uuid
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import select

from src.database.models import Decision, Recommendation, OptimizationRun
from src.database.crud import (
    create_decision as db_create_decision,
    get_decision as db_get_decision,
    get_decision_by_code,
)
from src.database.schemas import DecisionCreate, DecisionStatusUpdate
from src.decision_management.schemas import DecisionCreateRequest, DecisionUpdateRequest, DecisionAuditEntry
from src.decision_management.decision_validation import (
    validate_decision_creation,
    validate_status_transition,
    DecisionValidationError,
)
from src.decision_management.audit_service import DecisionAuditService

logger = logging.getLogger(__name__)


class DecisionService:
    """Coordinates managerial reviews, selections, approvals, overrides, and execution tracking."""

    def __init__(self, audit_service: Optional[DecisionAuditService] = None):
        self.audit_service = audit_service or DecisionAuditService()

    def record_decision(self, db: Session, req: DecisionCreateRequest) -> Decision:
        """Record and validate a manager's decision choice in the database.

        Args:
            db: Database session.
            req: Decision creation payload.

        Returns:
            Newly created Decision database entity.
        """
        run = db.get(OptimizationRun, req.optimization_run_id)
        rec = db.get(Recommendation, req.recommendation_id)

        # Validate existence & rules
        validate_decision_creation(
            recommendation=rec,
            optimization_run=run,
            decision_maker=req.decision_maker,
            decision_status=req.decision_status,
            decision_reason=req.decision_reason,
        )

        # Determine if this was an override of system top recommendation
        effective_status = req.decision_status
        if not rec.is_recommended and effective_status == "APPROVED":
            effective_status = "OVERRIDDEN"

        # Unique decision identifier
        decision_code = f"DEC-{uuid.uuid4().hex[:8].upper()}"

        create_schema = DecisionCreate(
            decision_id=decision_code,
            optimization_run_id=req.optimization_run_id,
            recommendation_id=req.recommendation_id,
            decision_maker=req.decision_maker,
            decision_status=effective_status,
            decision_reason=req.decision_reason,
            execution_status=req.execution_status,
        )

        db_decision = db_create_decision(db, create_schema)
        logger.info(f"Recorded decision {decision_code}: status={effective_status}, maker={req.decision_maker}")
        return db_decision

    def update_decision(self, db: Session, decision_id: int, req: DecisionUpdateRequest) -> Decision:
        """Update the operational status or execution lifecycle of an existing decision."""
        decision = db.get(Decision, decision_id)
        if not decision:
            raise DecisionValidationError(f"Decision with ID {decision_id} not found.")

        # Validate transition legality
        validate_status_transition(
            current_decision_status=decision.decision_status,
            new_decision_status=req.decision_status,
            current_execution_status=decision.execution_status,
            new_execution_status=req.execution_status,
        )

        if req.decision_status:
            decision.decision_status = req.decision_status
        if req.execution_status:
            decision.execution_status = req.execution_status
            if req.execution_status == "COMPLETED" and not decision.executed_at:
                decision.executed_at = req.executed_at or datetime.now(timezone.utc)
        if req.decision_reason is not None:
            decision.decision_reason = req.decision_reason

        db.commit()
        db.refresh(decision)
        logger.info(f"Updated decision {decision.decision_id}: exec_status={decision.execution_status}")
        return decision

    def get_decision(self, db: Session, decision_id: int) -> Optional[Decision]:
        return db.get(Decision, decision_id)

    def get_decision_by_code(self, db: Session, code: str) -> Optional[Decision]:
        return get_decision_by_code(db, code)

    def list_decisions(self, db: Session, skip: int = 0, limit: int = 100) -> List[Decision]:
        stmt = select(Decision).order_by(Decision.created_at.desc()).offset(skip).limit(limit)
        return list(db.execute(stmt).scalars().all())

