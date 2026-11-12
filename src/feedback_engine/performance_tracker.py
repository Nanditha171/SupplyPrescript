"""Historical performance and cost prediction tracking across strategies (Day 15)."""

from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import select

from src.database.models import Feedback, Outcome, Decision, Recommendation


def get_strategy_performance_metrics(db: Session) -> Dict[str, Any]:
    """Calculate historical cost error and learning statistics grouped by recovery strategy."""
    stmt = (
        select(Feedback, Outcome, Decision, Recommendation)
        .join(Outcome, Feedback.outcome_id == Outcome.id)
        .join(Decision, Outcome.decision_id == Decision.id)
        .join(Recommendation, Decision.recommendation_id == Recommendation.id)
        .order_by(Feedback.created_at.asc())
    )
    records = db.execute(stmt).all()

    strategy_stats: Dict[str, Dict[str, Any]] = {}

    for fb, out, dec, rec in records:
        st = rec.strategy_type
        if st not in strategy_stats:
            strategy_stats[st] = {
                "count": 0,
                "total_cost_error": 0.0,
                "total_delay_error": 0.0,
                "ratios": [],
                "learned_multiplier": 1.0,
            }

        cost_err = float(fb.cost_error)
        delay_err = float(fb.delay_error)
        est_cost = float(rec.estimated_cost)
        act_cost = float(out.actual_cost)

        strategy_stats[st]["count"] += 1
        strategy_stats[st]["total_cost_error"] += cost_err
        strategy_stats[st]["total_delay_error"] += delay_err
        if est_cost > 0:
            strategy_stats[st]["ratios"].append(act_cost / est_cost)

    summary: Dict[str, Any] = {}
    for st, data in strategy_stats.items():
        n = data["count"]
        avg_cost_err = round(data["total_cost_error"] / n, 2)
        avg_delay_err = round(data["total_delay_error"] / n, 2)
        summary[st] = {
            "observations_count": n,
            "average_cost_error": avg_cost_err,
            "average_delay_error": avg_delay_err,
            "latest_ratio": round(data["ratios"][-1], 3) if data["ratios"] else 1.0,
        }

    return summary

