"""Audit logging and historical traceability service for decisions (Day 13)."""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import select

from src.database.models import Decision, Recommendation, OptimizationRun
from src.decision_management.schemas import DecisionAuditEntry

logger = logging.getLogger(__name__)


class DecisionAuditService:
    """Manages audit trails and historical change records for decision events."""

    def format_audit_entry(self, decision: Decision) -> DecisionAuditEntry:
        """Construct structured audit entry from Decision ORM model."""
        strategy = decision.recommendation.strategy_type if decision.recommendation else "UNKNOWN"
        return DecisionAuditEntry(
            decision_id=decision.decision_id,
            optimization_run_id=decision.optimization_run_id,
            recommendation_id=decision.recommendation_id,
            strategy_type=strategy,
            decision_maker=decision.decision_maker,
            decision_status=decision.decision_status,
            execution_status=decision.execution_status,
            decision_reason=decision.decision_reason,
            selected_at=decision.selected_at,
            executed_at=decision.executed_at,
            created_at=decision.created_at,
        )

    def get_run_decisions_history(self, db: Session, optimization_run_id: int) -> List[DecisionAuditEntry]:
        """Retrieve all decisions associated with an optimization run in chronological order."""
        stmt = (
            select(Decision)
            .where(Decision.optimization_run_id == optimization_run_id)
            .order_by(Decision.created_at.asc())
        )
        decisions = list(db.execute(stmt).scalars().all())
        return [self.format_audit_entry(d) for d in decisions]

