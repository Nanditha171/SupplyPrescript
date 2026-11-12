"""Feedback Service closing the prescriptive loop from outcomes to future assumptions (Day 15)."""

import os
import yaml
import logging
from decimal import Decimal
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select

from src.database.models import Outcome, Decision, Recommendation, OptimizationRun, Feedback
from src.database.crud import create_feedback as db_create_feedback, get_feedback_by_outcome
from src.database.schemas import FeedbackCreate
from src.feedback_engine.schemas import FeedbackCreationRequest, FeedbackSummary, LearnedMultiplierState
from src.feedback_engine.learning_rules import validate_learning_candidate, clamp_multiplier, FeedbackSafeguardError
from src.feedback_engine.cost_calibration import compute_ewma_multiplier
from src.feedback_engine.performance_tracker import get_strategy_performance_metrics

logger = logging.getLogger(__name__)


def load_learning_config(config_path: str = "config/optimization_config.yaml") -> Dict[str, Any]:
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("learning", {})


class FeedbackService:
    """Calculates error signals, enforces learning safeguards, and updates optimization parameters."""

    def __init__(self, config_path: str = "config/optimization_config.yaml"):
        self.learning_cfg = load_learning_config(config_path)
        self.default_alpha = float(self.learning_cfg.get("default_alpha", 0.20))
        self.default_min_obs = int(self.learning_cfg.get("min_observations_required", 3))
        self.lower_bound = float(self.learning_cfg.get("lower_bound_multiplier", 0.70))
        self.upper_bound = float(self.learning_cfg.get("upper_bound_multiplier", 1.30))

    def process_outcome_feedback(
        self,
        db: Session,
        outcome_id: int,
        alpha: Optional[float] = None,
        min_observations: Optional[int] = None,
    ) -> Feedback:
        """Process an outcome into closed-loop feedback and calculate calibrated multiplier."""
        outcome = db.get(Outcome, outcome_id)
        if not outcome:
            raise FeedbackSafeguardError(f"Outcome with ID {outcome_id} not found.")

        # Protection against duplicate processing
        existing = get_feedback_by_outcome(db, outcome.id)
        if existing:
            logger.info(f"Feedback already exists for Outcome {outcome_id}. Returning existing.")
            return existing[0]

        decision = outcome.decision
        if not decision:
            raise FeedbackSafeguardError("Linked decision missing for outcome.")

        recommendation = decision.recommendation
        if not recommendation:
            raise FeedbackSafeguardError("Linked recommendation missing for decision.")

        opt_run = decision.optimization_run
        if not opt_run:
            raise FeedbackSafeguardError("Linked optimization run missing.")

        est_cost = float(recommendation.estimated_cost)
        act_cost = float(outcome.actual_cost)
        est_delay = float(recommendation.estimated_delay_days)
        act_delay = float(outcome.actual_delay_days)
        est_risk = float(recommendation.estimated_risk)
        act_risk = float(outcome.actual_risk)

        # 1. Safeguard sanity check
        is_valid, reason = validate_learning_candidate(act_cost, est_cost, outcome.outcome_status)
        if not is_valid:
            logger.warning(f"Outcome {outcome_id} rejected by safeguard: {reason}")
            # Persist feedback record marking safeguard rejection without adjusting weights
            fb_create = FeedbackCreate(
                outcome_id=outcome.id,
                optimization_run_id=opt_run.id,
                feedback_type="COST_VARIANCE",
                cost_error=Decimal(str(round(act_cost - est_cost, 2))),
                delay_error=round(act_delay - est_delay, 2),
                risk_error=round(act_risk - est_risk, 4),
                prediction_error=0.0,
                weight_adjustment=0.0,
                learning_signal=f"REJECTED_BY_SAFEGUARD: {reason}",
            )
            return db_create_feedback(db, fb_create)

        # 2. Historical Observations for this Strategy
        st_type = recommendation.strategy_type
        # Query all historical completed outcomes for this strategy
        stmt = (
            select(Outcome, Recommendation)
            .join(Decision, Outcome.decision_id == Decision.id)
            .join(Recommendation, Decision.recommendation_id == Recommendation.id)
            .where(Recommendation.strategy_type == st_type)
            .order_by(Outcome.evaluation_date.asc(), Outcome.id.asc())
        )
        history = db.execute(stmt).all()

        ratios: List[float] = []
        for past_out, past_rec in history:
            p_est = float(past_rec.estimated_cost)
            p_act = float(past_out.actual_cost)
            if p_est > 0:
                v_ok, _ = validate_learning_candidate(p_act, p_est, past_out.outcome_status)
                if v_ok:
                    ratios.append(p_act / p_est)

        obs_count = len(ratios)
        active_min_obs = min_observations or self.default_min_obs
        active_alpha = alpha or self.default_alpha

        cost_error_val = act_cost - est_cost
        delay_error_val = act_delay - est_delay
        risk_error_val = act_risk - est_risk

        if obs_count < active_min_obs:
            # Evidence insufficient: Retain baseline 1.0, do not manufacture unvalidated updates
            calibrated_mult = 1.0
            signal = (
                f"{st_type}: 1.00; Observation {obs_count}/{active_min_obs} recorded. "
                f"Retaining baseline estimate (1.00) until minimum sample threshold ({active_min_obs}) is satisfied."
            )
            weight_adj = 0.0
        else:
            # Sufficient observations: compute EWMA multiplier bounded in [0.70, 1.30]
            calibrated_mult, was_clamped = compute_ewma_multiplier(
                historical_ratios=ratios,
                alpha=active_alpha,
                initial_multiplier=1.0,
                lower_bound=self.lower_bound,
                upper_bound=self.upper_bound,
            )
            clamp_note = " [CLAMPED]" if was_clamped else ""
            signal = (
                f"{st_type}: {calibrated_mult:.2f}; Calibrated cost multiplier updated to {calibrated_mult:.2f}{clamp_note} "
                f"based on {obs_count} validated observations (EWMA alpha={active_alpha:.2f})."
            )
            weight_adj = round(calibrated_mult - 1.0, 4)

        fb_create = FeedbackCreate(
            outcome_id=outcome.id,
            optimization_run_id=opt_run.id,
            feedback_type="COST_VARIANCE",
            cost_error=Decimal(str(round(cost_error_val, 2))),
            delay_error=round(delay_error_val, 2),
            risk_error=round(risk_error_val, 4),
            prediction_error=0.0,
            weight_adjustment=weight_adj,
            learning_signal=signal,
        )

        db_fb = db_create_feedback(db, fb_create)
        logger.info(f"Feedback {db_fb.id} recorded for Outcome {outcome_id}: {signal}")
        return db_fb

    def get_feedback_for_outcome(self, db: Session, outcome_id: int) -> List[Feedback]:
        return get_feedback_by_outcome(db, outcome_id)

    def list_feedbacks(self, db: Session, skip: int = 0, limit: int = 100) -> List[Feedback]:
        stmt = select(Feedback).order_by(Feedback.created_at.desc()).offset(skip).limit(limit)
        return list(db.execute(stmt).scalars().all())

