"""Recommendation service for generating, ranking, and persisting business recommendations (Day 11)."""

import logging
from decimal import Decimal
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select

from src.database.models import OptimizationRun, Recommendation, SupplyEvent, Prediction, RiskAssessment
from src.database.crud import create_recommendation, get_recommendations
from src.database.schemas import RecommendationCreate
from src.optimization.solver import PrescriptiveOptimizationSolver
from src.optimization.solution_parser import OptimizationInput
from src.optimization.optimization_service import OptimizationService
from src.recommendation_engine.schemas import RecommendationOption, RecommendationComparison
from src.recommendation_engine.ranking import rank_recommendations
from src.recommendation_engine.explanation import generate_strategy_explanation

logger = logging.getLogger(__name__)


class RecommendationService:
    """Coordinates generation, multi-criteria ranking, and persistence of recommendations."""

    def __init__(self, optimization_service: Optional[OptimizationService] = None):
        self.opt_service = optimization_service or OptimizationService()
        self.solver = self.opt_service.solver

    def generate_alternatives_for_event(
        self,
        db: Session,
        event: SupplyEvent,
        prediction: Prediction,
        risk: Optional[RiskAssessment] = None,
        ranking_mode: str = "MIN_COST",
        air_freight_base_cost: Optional[float] = None,
        secondary_premium_rate: Optional[float] = None,
    ) -> List[RecommendationOption]:
        """Generate and evaluate distinct operational recovery alternatives for an event."""
        product_name = event.product.product_name if event.product else "critical component"

        # Strategies to evaluate
        candidate_strategies = ["AIR_FREIGHT", "SECONDARY_SUPPLIER", "DELAY_LAUNCH", "PRIMARY_SUPPLIER"]
        options: List[RecommendationOption] = []

        code_map = {
            "AIR_FREIGHT": "ALT_A",
            "SECONDARY_SUPPLIER": "ALT_B",
            "DELAY_LAUNCH": "ALT_C",
            "PRIMARY_SUPPLIER": "ALT_D",
            "PARTIAL_ALLOCATION": "ALT_E",
        }

        for st in candidate_strategies:
            opt_inp = self.opt_service.build_optimization_input_from_event(
                db=db,
                event=event,
                prediction=prediction,
                risk=risk,
                objective_type=ranking_mode,
                air_freight_base_cost=air_freight_base_cost,
                secondary_premium_rate=secondary_premium_rate,
                forced_strategy=st,
            )

            res = self.solver.solve(opt_inp)

            # Only include feasible solutions
            if res.solver_status in ("OPTIMAL", "FEASIBLE"):
                desc, adv, disadv, reason = generate_strategy_explanation(
                    strategy_type=st,
                    estimated_cost=res.estimated_total_cost,
                    estimated_delay_days=res.estimated_delivery_days,
                    estimated_risk=res.estimated_risk,
                    service_level=res.service_level_achieved,
                    product_name=product_name,
                )

                allocations = {
                    "primary": res.primary_supplier_quantity,
                    "secondary": res.secondary_supplier_quantity,
                    "air_freight": res.air_freight_quantity,
                    "unmet": res.unmet_demand,
                }

                option = RecommendationOption(
                    recommendation_code=code_map.get(st, f"ALT_{st}"),
                    strategy_type=st,
                    strategy_description=desc,
                    estimated_cost=res.estimated_total_cost,
                    estimated_delay_days=res.estimated_delivery_days,
                    estimated_risk=res.estimated_risk,
                    service_level=res.service_level_achieved,
                    rank=0,
                    is_recommended=False,
                    advantages=adv,
                    disadvantages=disadv,
                    reasoning=reason,
                    feasibility_status="FEASIBLE",
                    allocated_quantities=allocations,
                )
                options.append(option)

        # Apply multi-criteria ranking
        ranked = rank_recommendations(options, ranking_mode=ranking_mode)
        return ranked

    def generate_and_persist_recommendations(
        self,
        db: Session,
        optimization_run: OptimizationRun,
        ranking_mode: Optional[str] = None,
        air_freight_base_cost: Optional[float] = None,
        secondary_premium_rate: Optional[float] = None,
    ) -> RecommendationComparison:
        """Generate all alternatives, rank them, persist in DB, and format comparative report."""
        mode = ranking_mode or optimization_run.objective_type
        event = optimization_run.event
        prediction = optimization_run.prediction
        risk = db.execute(
            select(RiskAssessment).where(RiskAssessment.event_id == event.id).order_by(RiskAssessment.id.desc())
        ).scalars().first()

        # Check if recommendations already persisted for this run
        existing_recs = get_recommendations(db, optimization_run.id)
        if existing_recs:
            # Reconstruct RecommendationOption list
            opts = []
            for r in existing_recs:
                desc, adv, disadv, reason = generate_strategy_explanation(
                    strategy_type=r.strategy_type,
                    estimated_cost=float(r.estimated_cost),
                    estimated_delay_days=r.estimated_delay_days,
                    estimated_risk=r.estimated_risk,
                    service_level=r.service_level,
                    product_name=event.product.product_name if event.product else "item",
                )
                opts.append(
                    RecommendationOption(
                        recommendation_code=r.recommendation_code,
                        strategy_type=r.strategy_type,
                        strategy_description=r.strategy_description,
                        estimated_cost=float(r.estimated_cost),
                        estimated_delay_days=r.estimated_delay_days,
                        estimated_risk=r.estimated_risk,
                        service_level=r.service_level,
                        rank=r.rank,
                        is_recommended=r.is_recommended,
                        advantages=adv,
                        disadvantages=disadv,
                        reasoning=reason,
                        cost_difference=float(r.cost_difference),
                        risk_difference=r.risk_difference,
                    )
                )
            ranked_opts = rank_recommendations(opts, ranking_mode=mode)
        else:
            ranked_opts = self.generate_alternatives_for_event(
                db=db,
                event=event,
                prediction=prediction,
                risk=risk,
                ranking_mode=mode,
                air_freight_base_cost=air_freight_base_cost,
                secondary_premium_rate=secondary_premium_rate,
            )

            # Persist into DB
            for opt in ranked_opts:
                rec_create = RecommendationCreate(
                    optimization_run_id=optimization_run.id,
                    recommendation_code=opt.recommendation_code,
                    strategy_type=opt.strategy_type,
                    strategy_description=opt.strategy_description,
                    estimated_cost=Decimal(str(round(opt.estimated_cost, 2))),
                    estimated_delay_days=opt.estimated_delay_days,
                    estimated_risk=opt.estimated_risk,
                    service_level=opt.service_level,
                    cost_difference=Decimal(str(round(opt.cost_difference, 2))),
                    risk_difference=opt.risk_difference,
                    rank=opt.rank,
                    is_recommended=opt.is_recommended,
                )
                create_recommendation(db, rec_create)

        # Build tabular matrix
        comp_table = []
        rec_code = ranked_opts[0].recommendation_code if ranked_opts else "ALT_A"
        for opt in ranked_opts:
            comp_table.append(
                {
                    "recommendation_code": opt.recommendation_code,
                    "strategy": opt.strategy_type,
                    "estimated_cost": f"${opt.estimated_cost:,.2f}",
                    "delivery_time": f"{opt.estimated_delay_days:.1f} days",
                    "risk": f"{opt.estimated_risk:.2f}",
                    "service_level": f"{opt.service_level * 100:.1f}%",
                    "rank": opt.rank,
                    "is_recommended": opt.is_recommended,
                    "feasibility": opt.feasibility_status,
                }
            )

        return RecommendationComparison(
            optimization_run_id=optimization_run.optimization_run_id,
            event_id=event.event_id,
            ranking_mode=mode,
            recommendations=ranked_opts,
            comparison_table=comp_table,
            recommended_strategy_code=rec_code,
        )

