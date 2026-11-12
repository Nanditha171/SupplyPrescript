"""Modular stage execution functions for the Supply Prescript closed-loop workflow."""

import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select

from src.database.models import SupplyEvent, Prediction, RiskAssessment, OptimizationRun, Decision, Outcome, Feedback
from src.api.services.prediction_service import PredictionService
from src.api.services.risk_service import RiskService
from src.optimization.optimization_service import OptimizationService
from src.recommendation_engine.recommendation_service import RecommendationService
from src.decision_management.decision_service import DecisionService
from src.decision_management.schemas import DecisionCreateRequest, DecisionUpdateRequest
from src.outcome_engine.outcome_service import OutcomeService
from src.outcome_engine.schemas import OutcomeRecordRequest
from src.feedback_engine.feedback_service import FeedbackService
from src.workflows.workflow_schemas import WorkflowStepResult

logger = logging.getLogger(__name__)


class SupplyChainWorkflowRunner:
    """Executes discrete stages of the closed-loop prescriptive cycle."""

    def __init__(self):
        self.pred_service = PredictionService()
        self.risk_service = RiskService()
        self.opt_service = OptimizationService()
        self.rec_service = RecommendationService(self.opt_service)
        self.dec_service = DecisionService()
        self.out_service = OutcomeService()
        self.fb_service = FeedbackService()

    def run_step_predict(self, db: Session, event_id: str, step_num: int = 1) -> Tuple[Prediction, WorkflowStepResult]:
        t0 = time.perf_counter()
        start_dt = datetime.now(timezone.utc)
        pred = self.pred_service.predict_for_shipment(db, event_id)
        exec_ms = (time.perf_counter() - t0) * 1000.0

        step_res = WorkflowStepResult(
            step_name="Delay Prediction",
            step_number=step_num,
            status="COMPLETED",
            started_at=start_dt,
            completed_at=datetime.now(timezone.utc),
            execution_time_ms=round(exec_ms, 2),
            output_summary={
                "prediction_id": pred.id,
                "delay_probability": pred.delay_probability,
                "predicted_delay_days": pred.predicted_delay_days,
                "risk_level": pred.risk_level,
            },
        )
        return pred, step_res

    def run_step_risk(self, db: Session, event_id: str, step_num: int = 2) -> Tuple[RiskAssessment, WorkflowStepResult]:
        t0 = time.perf_counter()
        start_dt = datetime.now(timezone.utc)
        risk = self.risk_service.assess_risk_for_shipment(db, event_id, prediction_service=self.pred_service)
        exec_ms = (time.perf_counter() - t0) * 1000.0

        step_res = WorkflowStepResult(
            step_name="Operational Risk Assessment",
            step_number=step_num,
            status="COMPLETED",
            started_at=start_dt,
            completed_at=datetime.now(timezone.utc),
            execution_time_ms=round(exec_ms, 2),
            output_summary={
                "risk_assessment_id": risk.id,
                "composite_risk_score": risk.composite_risk_score,
                "risk_level": risk.risk_level,
                "delay_risk": risk.delay_risk_score,
                "supplier_risk": risk.supplier_risk_score,
            },
        )
        return risk, step_res

    def run_step_optimize_and_recommend(
        self,
        db: Session,
        event: SupplyEvent,
        pred: Prediction,
        risk: RiskAssessment,
        objective_type: str = "MIN_COST",
        air_freight_base_cost: Optional[float] = None,
        secondary_premium_rate: Optional[float] = None,
        step_num: int = 3,
    ) -> Tuple[OptimizationRun, Any, WorkflowStepResult]:
        t0 = time.perf_counter()
        start_dt = datetime.now(timezone.utc)

        opt_inp = self.opt_service.build_optimization_input_from_event(
            db=db,
            event=event,
            prediction=pred,
            risk=risk,
            objective_type=objective_type,
            air_freight_base_cost=air_freight_base_cost,
            secondary_premium_rate=secondary_premium_rate,
        )

        db_run, out = self.opt_service.execute_and_persist(db, opt_inp, event.id, pred.id)
        comparison = self.rec_service.generate_and_persist_recommendations(
            db=db,
            optimization_run=db_run,
            ranking_mode=objective_type,
            air_freight_base_cost=air_freight_base_cost,
            secondary_premium_rate=secondary_premium_rate,
        )
        exec_ms = (time.perf_counter() - t0) * 1000.0

        step_res = WorkflowStepResult(
            step_name="Optimization & Recommendation",
            step_number=step_num,
            status="COMPLETED",
            started_at=start_dt,
            completed_at=datetime.now(timezone.utc),
            execution_time_ms=round(exec_ms, 2),
            output_summary={
                "optimization_run_id": db_run.optimization_run_id,
                "solver_status": db_run.solver_status,
                "recommended_cost": float(db_run.recommended_cost) if db_run.recommended_cost else 0.0,
                "recommended_code": comparison.recommended_strategy_code,
                "alternatives_count": len(comparison.recommendations),
            },
        )
        return db_run, comparison, step_res

    def run_step_decide_and_execute(
        self,
        db: Session,
        run_id: int,
        recommendation_id: int,
        decision_maker: str = "Manager",
        reason: Optional[str] = None,
        step_num: int = 4,
    ) -> Tuple[Decision, WorkflowStepResult]:
        t0 = time.perf_counter()
        start_dt = datetime.now(timezone.utc)

        # 1. Record decision
        req = DecisionCreateRequest(
            optimization_run_id=run_id,
            recommendation_id=recommendation_id,
            decision_maker=decision_maker,
            decision_status="APPROVED",
            decision_reason=reason,
            execution_status="NOT_STARTED",
        )
        decision = self.dec_service.record_decision(db, req)

        # 2. Track execution lifecycle
        self.dec_service.update_decision(db, decision.id, DecisionUpdateRequest(execution_status="IN_PROGRESS"))
        decision = self.dec_service.update_decision(db, decision.id, DecisionUpdateRequest(execution_status="COMPLETED"))

        exec_ms = (time.perf_counter() - t0) * 1000.0

        step_res = WorkflowStepResult(
            step_name="Decision & Execution",
            step_number=step_num,
            status="COMPLETED",
            started_at=start_dt,
            completed_at=datetime.now(timezone.utc),
            execution_time_ms=round(exec_ms, 2),
            output_summary={
                "decision_id": decision.decision_id,
                "decision_status": decision.decision_status,
                "execution_status": decision.execution_status,
                "executed_at": decision.executed_at.isoformat() if decision.executed_at else None,
            },
        )
        return decision, step_res

    def run_step_measure_and_learn(
        self,
        db: Session,
        decision_id: int,
        actual_cost: float,
        actual_delay_days: float,
        actual_service_level: float = 1.0,
        actual_risk: float = 0.08,
        actual_quantity: int = 1000,
        notes: Optional[str] = None,
        step_num: int = 5,
    ) -> Tuple[Outcome, Feedback, WorkflowStepResult]:
        t0 = time.perf_counter()
        start_dt = datetime.now(timezone.utc)

        # 1. Record outcome
        out_req = OutcomeRecordRequest(
            decision_id=decision_id,
            actual_cost=actual_cost,
            actual_delay_days=actual_delay_days,
            actual_service_level=actual_service_level,
            actual_risk=actual_risk,
            actual_quantity_delivered=actual_quantity,
            notes=notes,
        )
        outcome = self.out_service.record_outcome(db, out_req)

        # 2. Close feedback loop
        feedback = self.fb_service.process_outcome_feedback(db, outcome.id)

        exec_ms = (time.perf_counter() - t0) * 1000.0

        step_res = WorkflowStepResult(
            step_name="Measurement & Prescriptive Learning",
            step_number=step_num,
            status="COMPLETED",
            started_at=start_dt,
            completed_at=datetime.now(timezone.utc),
            execution_time_ms=round(exec_ms, 2),
            output_summary={
                "outcome_id": outcome.id,
                "cost_variance": float(outcome.cost_variance),
                "delay_variance": outcome.delay_variance,
                "outcome_status": outcome.outcome_status,
                "feedback_id": feedback.id,
                "learning_signal": feedback.learning_signal,
            },
        )
        return outcome, feedback, step_res

