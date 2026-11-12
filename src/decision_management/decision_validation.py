"""Validation rules and state transition checks for Decision Management (Day 13)."""

from typing import Tuple, List, Optional
from src.database.models import Decision, Recommendation, OptimizationRun


class DecisionValidationError(Exception):
    """Raised when a decision operation violates business or integrity rules."""
    pass


VALID_DECISION_STATUSES = {"PENDING", "APPROVED", "REJECTED", "OVERRIDDEN", "CANCELLED"}
VALID_EXECUTION_STATUSES = {"NOT_STARTED", "IN_PROGRESS", "COMPLETED", "FAILED"}

VALID_DECISION_TRANSITIONS = {
    "PENDING": {"APPROVED", "REJECTED", "OVERRIDDEN", "CANCELLED"},
    "APPROVED": {"OVERRIDDEN", "CANCELLED"},
    "OVERRIDDEN": {"APPROVED", "CANCELLED"},
    "REJECTED": {"CANCELLED"},
    "CANCELLED": set(),
}

VALID_EXECUTION_TRANSITIONS = {
    "NOT_STARTED": {"IN_PROGRESS", "FAILED", "CANCELLED"},
    "IN_PROGRESS": {"COMPLETED", "FAILED"},
    "COMPLETED": set(),  # Final state
    "FAILED": {"IN_PROGRESS"},  # Allow retry
}


def validate_decision_creation(
    recommendation: Optional[Recommendation],
    optimization_run: Optional[OptimizationRun],
    decision_maker: str,
    decision_status: str,
    decision_reason: Optional[str],
) -> None:
    """Validate decision parameters prior to database creation."""
    if not optimization_run:
        raise DecisionValidationError("Invalid optimization_run_id: Optimization run does not exist.")

    if not recommendation:
        raise DecisionValidationError("Invalid recommendation_id: Recommendation does not exist.")

    if recommendation.optimization_run_id != optimization_run.id:
        raise DecisionValidationError(
            f"Recommendation (id={recommendation.id}) does not belong to Optimization Run (id={optimization_run.id})."
        )

    if not decision_maker or not decision_maker.strip():
        raise DecisionValidationError("Decision maker name is required.")

    status_upper = decision_status.upper()
    if status_upper not in VALID_DECISION_STATUSES:
        raise DecisionValidationError(f"Invalid decision status: {decision_status}. Allowed: {VALID_DECISION_STATUSES}")

    # Override requires documented business reason
    is_top_rec = recommendation.is_recommended
    if not is_top_rec and status_upper != "REJECTED":
        if status_upper != "OVERRIDDEN" and not decision_reason:
            raise DecisionValidationError(
                "Selecting a non-recommended alternative constitutes an override and requires a documented reason."
            )
        if status_upper == "OVERRIDDEN" and (not decision_reason or len(decision_reason.strip()) < 5):
            raise DecisionValidationError("An explicit override reason of at least 5 characters is mandatory.")


def validate_status_transition(
    current_decision_status: str,
    new_decision_status: Optional[str],
    current_execution_status: str,
    new_execution_status: Optional[str],
) -> None:
    """Validate lifecycle state transitions for decision and execution."""
    if new_decision_status and new_decision_status != current_decision_status:
        allowed = VALID_DECISION_TRANSITIONS.get(current_decision_status, set())
        if new_decision_status not in allowed:
            raise DecisionValidationError(
                f"Illegal decision transition: Cannot transition from {current_decision_status} to {new_decision_status}."
            )

    if new_execution_status and new_execution_status != current_execution_status:
        # Check if decision was approved/overridden before execution starts
        if new_execution_status in ("IN_PROGRESS", "COMPLETED") and current_decision_status not in ("APPROVED", "OVERRIDDEN"):
            raise DecisionValidationError(
                f"Cannot execute a decision that is not APPROVED or OVERRIDDEN (current status: {current_decision_status})."
            )

        allowed_exec = VALID_EXECUTION_TRANSITIONS.get(current_execution_status, set())
        if new_execution_status not in allowed_exec:
            raise DecisionValidationError(
                f"Illegal execution transition: Cannot transition from {current_execution_status} to {new_execution_status}."
            )

