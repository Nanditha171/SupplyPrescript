"""Outcome Evaluation Service for measuring and persisting post-execution performance (Day 14)."""

import logging
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select, func

from src.database.models import Decision, Recommendation, Outcome, OptimizationRun
from src.database.crud import create_outcome as db_create_outcome, get_outcome_by_decision
from src.database.schemas import OutcomeCreate
from src.outcome_engine.schemas import OutcomeRecordRequest, OutcomeVarianceSummary
from src.outcome_engine.variance_calculator import calculate_variances
from src.outcome_engine.performance_metrics import classify_outcome_status

logger = logging.getLogger(__name__)


class OutcomeEvaluationError(Exception):
    """Raised when outcome evaluation violates prerequisites or business rules."""
    pass


class OutcomeService:
    """Evaluates post-execution actual operational metrics against optimization estimates."""

    def record_outcome(self, db: Session, req: OutcomeRecordRequest) -> Outcome:
        """Record post-execution operational results and calculate variances.

        Args:
            db: Database session.
            req: Outcome metrics payload.

        Returns:
            Newly created Outcome database model.
        """
        decision = db.get(Decision, req.decision_id)
        if not decision:
            raise OutcomeEvaluationError(f"Decision with ID {req.decision_id} not found.")

        # Prerequisite: Decision must be executed before outcome can be recorded
        if decision.execution_status != "COMPLETED":
            raise OutcomeEvaluationError(
                f"Cannot record outcome for decision {decision.decision_id}: "
                f"Execution status is '{decision.execution_status}'. Must be 'COMPLETED'."
            )

        # Check for duplicate outcome
        existing_outcome = get_outcome_by_decision(db, decision.id)
        if existing_outcome:
            raise OutcomeEvaluationError(
                f"An outcome has already been recorded for decision {decision.decision_id} (Outcome ID: {existing_outcome.id})."
            )

        recommendation = decision.recommendation
        if not recommendation:
            raise OutcomeEvaluationError(f"Decision {decision.decision_id} is missing linked recommendation.")

        est_cost = float(recommendation.estimated_cost)
        est_delay = float(recommendation.estimated_delay_days)
        est_sla = float(recommendation.service_level)

        # Calculate Variances
        var_data = calculate_variances(
            actual_cost=req.actual_cost,
            estimated_cost=est_cost,
            actual_delay_days=req.actual_delay_days,
            estimated_delay_days=est_delay,
            actual_service_level=req.actual_service_level,
            estimated_service_level=est_sla,
        )

        outcome_status = classify_outcome_status(
            actual_service_level=req.actual_service_level,
            delay_variance=var_data["delay_variance"],
            cost_variance=var_data["cost_variance"],
            estimated_cost=est_cost,
            is_cancelled=(decision.decision_status == "CANCELLED"),
        )

        eval_date = req.evaluation_date or date.today()

        outcome_schema = OutcomeCreate(
            decision_id=decision.id,
            actual_cost=Decimal(str(round(req.actual_cost, 2))),
            actual_delay_days=req.actual_delay_days,
            actual_service_level=req.actual_service_level,
            actual_risk=req.actual_risk,
            actual_quantity_delivered=req.actual_quantity_delivered,
            cost_variance=Decimal(str(round(var_data["cost_variance"], 2))),
            delay_variance=var_data["delay_variance"],
            service_level_variance=var_data["service_level_variance"],
            outcome_status=outcome_status,
            evaluation_date=eval_date,
            notes=req.notes,
        )

        db_outcome = db_create_outcome(db, outcome_schema)
        logger.info(
            f"Recorded outcome for decision {decision.decision_id}: status={outcome_status}, "
            f"cost_var={var_data['cost_variance']}, delay_var={var_data['delay_variance']}"
        )
        return db_outcome

    def get_outcome(self, db: Session, outcome_id: int) -> Optional[Outcome]:
        return db.get(Outcome, outcome_id)

    def get_outcome_by_decision_id(self, db: Session, decision_id: int) -> Optional[Outcome]:
        return get_outcome_by_decision(db, decision_id)

    def list_outcomes(self, db: Session, skip: int = 0, limit: int = 100) -> List[Outcome]:
        stmt = select(Outcome).order_by(Outcome.created_at.desc()).offset(skip).limit(limit)
        return list(db.execute(stmt).scalars().all())

    def get_aggregate_summary(self, db: Session) -> Dict[str, Any]:
        """Generate high-level aggregated operational KPI metrics across all evaluated outcomes."""
        outcomes = self.list_outcomes(db, limit=1000)
        total_executed = db.query(Decision).filter(Decision.execution_status == "COMPLETED").count()
        total_outcomes = len(outcomes)

        if total_outcomes == 0:
            return {
                "total_executed_decisions": total_executed,
                "total_evaluated_outcomes": 0,
                "successful_decisions": 0,
                "partial_success_decisions": 0,
                "failed_decisions": 0,
                "average_cost_variance": 0.0,
                "average_delay_variance": 0.0,
                "average_service_level": 1.0,
                "strategy_breakdown": {},
            }

        successes = sum(1 for o in outcomes if o.outcome_status == "SUCCESS")
        partials = sum(1 for o in outcomes if o.outcome_status == "PARTIAL_SUCCESS")
        fails = sum(1 for o in outcomes if o.outcome_status == "FAILED")

        avg_cost_var = sum(float(o.cost_variance) for o in outcomes) / total_outcomes
        avg_delay_var = sum(o.delay_variance for o in outcomes) / total_outcomes
        avg_sla = sum(o.actual_service_level for o in outcomes) / total_outcomes

        strategy_counts: Dict[str, Dict[str, Any]] = {}
        for o in outcomes:
            st = o.decision.recommendation.strategy_type if o.decision and o.decision.recommendation else "UNKNOWN"
            if st not in strategy_counts:
                strategy_counts[st] = {"count": 0, "total_cost_var": 0.0, "successes": 0}
            strategy_counts[st]["count"] += 1
            strategy_counts[st]["total_cost_var"] += float(o.cost_variance)
            if o.outcome_status == "SUCCESS":
                strategy_counts[st]["successes"] += 1

        for st, data in strategy_counts.items():
            data["avg_cost_variance"] = round(data["total_cost_var"] / data["count"], 2)
            data["success_rate_pct"] = round((data["successes"] / data["count"]) * 100.0, 1)

        return {
            "total_executed_decisions": total_executed,
            "total_evaluated_outcomes": total_outcomes,
            "successful_decisions": successes,
            "partial_success_decisions": partials,
            "failed_decisions": fails,
            "average_cost_variance": round(avg_cost_var, 2),
            "average_delay_variance": round(avg_delay_var, 2),
            "average_service_level": round(avg_sla, 4),
            "strategy_breakdown": strategy_counts,
        }

