"""Multi-criteria ranking engine for supply chain recovery recommendations (Day 11)."""

from typing import List, Dict, Any
from src.recommendation_engine.schemas import RecommendationOption


def rank_recommendations(
    options: List[RecommendationOption],
    ranking_mode: str = "MIN_COST",
    weights: Dict[str, float] = None,
) -> List[RecommendationOption]:
    """Rank recommendation alternatives based on the selected objective mode.

    Modes supported:
        - MIN_COST: Prioritizes lowest financial expenditure subject to feasibility.
        - MIN_RISK: Prioritizes lowest disruption exposure.
        - MIN_DELAY: Prioritizes quickest replenishment delivery turnaround.
        - BALANCED: Evaluates normalized multi-objective composite score.

    Args:
        options: List of generated feasible recommendation options.
        ranking_mode: Objective ranking strategy.
        weights: Configured weights for multi-objective balanced mode.

    Returns:
        Ranked list of RecommendationOption with rank and is_recommended set.
    """
    if not options:
        return []

    mode = ranking_mode.upper()
    active_weights = weights or {"cost_weight": 0.34, "risk_weight": 0.33, "time_weight": 0.33}

    # Helper scoring for BALANCED mode with strict normalization
    max_cost = max([o.estimated_cost for o in options] + [1.0])
    max_risk = max([o.estimated_risk for o in options] + [0.01])
    max_delay = max([o.estimated_delay_days for o in options] + [1.0])

    def balanced_score(opt: RecommendationOption) -> float:
        norm_cost = opt.estimated_cost / max_cost
        norm_risk = opt.estimated_risk / max_risk
        norm_delay = opt.estimated_delay_days / max_delay
        return (
            active_weights.get("cost_weight", 0.34) * norm_cost
            + active_weights.get("risk_weight", 0.33) * norm_risk
            + active_weights.get("time_weight", 0.33) * norm_delay
        )

    # Sort options according to mode
    if mode == "MIN_COST":
        sorted_opts = sorted(
            options,
            key=lambda o: (o.estimated_cost, o.estimated_delay_days, o.estimated_risk),
        )
    elif mode == "MIN_RISK":
        sorted_opts = sorted(
            options,
            key=lambda o: (o.estimated_risk, o.estimated_cost, o.estimated_delay_days),
        )
    elif mode == "MIN_DELAY":
        sorted_opts = sorted(
            options,
            key=lambda o: (o.estimated_delay_days, o.estimated_cost, o.estimated_risk),
        )
    elif mode == "BALANCED":
        sorted_opts = sorted(
            options,
            key=lambda o: (balanced_score(o), o.estimated_cost),
        )
    else:
        # Default to MIN_COST
        sorted_opts = sorted(
            options,
            key=lambda o: (o.estimated_cost, o.estimated_delay_days, o.estimated_risk),
        )

    # Assign ranks and recommended flag
    baseline_cost = sorted_opts[0].estimated_cost
    baseline_risk = sorted_opts[0].estimated_risk

    for idx, opt in enumerate(sorted_opts, start=1):
        opt.rank = idx
        opt.is_recommended = (idx == 1)
        opt.cost_difference = opt.estimated_cost - baseline_cost
        opt.risk_difference = opt.estimated_risk - baseline_risk

    return sorted_opts

