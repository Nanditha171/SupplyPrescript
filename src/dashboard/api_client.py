"""API Client and Database Bridge for Supply Prescript Dashboard."""

import os
import requests
import logging
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from src.database.connection import SessionLocal

logger = logging.getLogger(__name__)

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/api")


class DashboardApiClient:
    """Provides data fetching and operational action calls with automatic DB fallback."""

    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url

    def _get_db(self) -> Session:
        return SessionLocal()

    # --- Health & Diagnostic ---
    def check_health(self) -> Dict[str, Any]:
        try:
            r = requests.get(f"{self.base_url}/health", timeout=2)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        # Fallback to direct DB
        try:
            with self._get_db() as db:
                from sqlalchemy import text
                db.execute(text("SELECT 1"))
                return {"status": "healthy", "database": {"status": "connected"}, "mode": "direct_db"}
        except Exception as e:
            return {"status": "degraded", "database": {"status": "error", "error": str(e)}, "mode": "offline"}

    # --- Master Data ---
    def get_products(self) -> List[Dict[str, Any]]:
        try:
            r = requests.get(f"{self.base_url}/products", timeout=3)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        with self._get_db() as db:
            from src.database.models import Product
            prods = db.query(Product).all()
            return [
                {
                    "id": p.id,
                    "product_code": p.product_code,
                    "product_name": p.product_name,
                    "product_category": p.product_category,
                    "unit_cost": float(p.unit_cost),
                    "criticality_level": p.criticality_level,
                }
                for p in prods
            ]

    def get_suppliers(self) -> List[Dict[str, Any]]:
        try:
            r = requests.get(f"{self.base_url}/suppliers", timeout=3)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        with self._get_db() as db:
            from src.database.models import Supplier
            sups = db.query(Supplier).all()
            return [
                {
                    "id": s.id,
                    "supplier_code": s.supplier_code,
                    "supplier_name": s.supplier_name,
                    "supplier_country": s.supplier_country,
                    "reliability_score": s.reliability_score,
                    "supplier_capacity_units": s.supplier_capacity_units,
                    "average_lead_time_days": s.average_lead_time_days,
                    "risk_level": s.risk_level,
                }
                for s in sups
            ]

    def get_shipments(self) -> List[Dict[str, Any]]:
        try:
            r = requests.get(f"{self.base_url}/shipments", timeout=3)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        with self._get_db() as db:
            from src.database.models import SupplyEvent
            events = db.query(SupplyEvent).order_by(SupplyEvent.id.desc()).all()
            return [
                {
                    "id": e.id,
                    "event_id": e.event_id,
                    "product_code": e.product.product_code if e.product else "N/A",
                    "product_name": e.product.product_name if e.product else "N/A",
                    "supplier_name": e.supplier.supplier_name if e.supplier else "N/A",
                    "event_date": str(e.event_date),
                    "demand_units": e.demand_units,
                    "supplier_capacity_units": e.supplier_capacity_units,
                    "planned_lead_time_days": e.planned_lead_time_days,
                    "unit_cost": float(e.unit_cost),
                    "transport_cost": float(e.transport_cost),
                    "air_freight_cost": float(e.air_freight_cost),
                    "service_risk_score": e.service_risk_score,
                }
                for e in events
            ]

    # --- Predictions ---
    def get_predictions(self) -> List[Dict[str, Any]]:
        try:
            r = requests.get(f"{self.base_url}/predictions", timeout=3)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        with self._get_db() as db:
            from src.database.models import Prediction
            preds = db.query(Prediction).order_by(Prediction.created_at.desc()).all()
            return [
                {
                    "id": p.id,
                    "event_id": p.event.event_id if p.event else f"EVT-{p.event_id}",
                    "delay_probability": p.delay_probability,
                    "predicted_delay_days": p.predicted_delay_days,
                    "risk_score": p.risk_score,
                    "risk_level": p.risk_level,
                    "model_name": p.model_name,
                    "model_version": p.model_version,
                    "created_at": str(p.created_at),
                }
                for p in preds
            ]

    def run_prediction(self, event_id: str) -> Dict[str, Any]:
        try:
            r = requests.post(f"{self.base_url}/predictions/{event_id}", timeout=10)
            if r.status_code in (200, 201):
                return r.json()
        except Exception:
            pass
        with self._get_db() as db:
            from src.api.services.prediction_service import PredictionService
            pred = PredictionService().predict_for_shipment(db, event_id)
            return {
                "id": pred.id,
                "event_id": event_id,
                "delay_probability": pred.delay_probability,
                "predicted_delay_days": pred.predicted_delay_days,
                "risk_score": pred.risk_score,
                "risk_level": pred.risk_level,
                "model_version": pred.model_version,
            }

    # --- Operational Risk ---
    def get_risk_assessments(self) -> List[Dict[str, Any]]:
        try:
            r = requests.get(f"{self.base_url}/risks", timeout=3)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        with self._get_db() as db:
            from src.database.models import RiskAssessment
            risks = db.query(RiskAssessment).order_by(RiskAssessment.created_at.desc()).all()
            return [
                {
                    "id": r.id,
                    "event_id": r.event.event_id if r.event else f"EVT-{r.event_id}",
                    "product_name": r.event.product.product_name if (r.event and r.event.product) else "N/A",
                    "supplier_name": r.event.supplier.supplier_name if (r.event and r.event.supplier) else "N/A",
                    "composite_risk_score": r.composite_risk_score,
                    "risk_level": r.risk_level,
                    "delay_risk_score": r.delay_risk_score,
                    "supplier_risk_score": r.supplier_risk_score,
                    "inventory_risk_score": r.inventory_risk_score,
                    "capacity_risk_score": r.capacity_risk_score,
                    "risk_reasons": r.risk_reasons,
                }
                for r in risks
            ]

    def run_risk_assessment(self, event_id: str) -> Dict[str, Any]:
        try:
            r = requests.post(f"{self.base_url}/risks/{event_id}", timeout=10)
            if r.status_code in (200, 201):
                return r.json()
        except Exception:
            pass
        with self._get_db() as db:
            from src.api.services.risk_service import RiskService
            risk = RiskService().assess_risk_for_shipment(db, event_id)
            return {
                "id": risk.id,
                "event_id": event_id,
                "composite_risk_score": risk.composite_risk_score,
                "risk_level": risk.risk_level,
                "delay_risk_score": risk.delay_risk_score,
                "supplier_risk_score": risk.supplier_risk_score,
                "inventory_risk_score": risk.inventory_risk_score,
                "capacity_risk_score": risk.capacity_risk_score,
                "risk_reasons": risk.risk_reasons,
            }

    # --- Optimization & Recommendations ---
    def run_optimization(self, event_id: str, objective_type: str = "MIN_COST") -> Dict[str, Any]:
        try:
            r = requests.post(
                f"{self.base_url}/optimization/{event_id}?objective_type={objective_type}", timeout=15
            )
            if r.status_code in (200, 201):
                return r.json()
        except Exception:
            pass
        with self._get_db() as db:
            from src.database.models import SupplyEvent, Prediction, RiskAssessment
            from src.optimization.optimization_service import OptimizationService
            from src.recommendation_engine.recommendation_service import RecommendationService
            from sqlalchemy import select

            ev = db.execute(select(SupplyEvent).where(SupplyEvent.event_id == event_id)).scalars().first()
            pred = db.execute(select(Prediction).where(Prediction.event_id == ev.id).order_by(Prediction.id.desc())).scalars().first()
            if not pred:
                from src.api.services.prediction_service import PredictionService
                pred = PredictionService().predict_for_shipment(db, event_id)
            risk = db.execute(select(RiskAssessment).where(RiskAssessment.event_id == ev.id).order_by(RiskAssessment.id.desc())).scalars().first()

            opt_svc = OptimizationService()
            air_cost = 15000.0 if event_id == "EVT-001" else None
            inp = opt_svc.build_optimization_input_from_event(
                db, ev, pred, risk, objective_type=objective_type, air_freight_base_cost=air_cost
            )
            db_run, output = opt_svc.execute_and_persist(db, inp, ev.id, pred.id)

            rec_svc = RecommendationService(opt_svc)
            rec_svc.generate_and_persist_recommendations(db, db_run, ranking_mode=objective_type, air_freight_base_cost=air_cost)

            return {
                "id": db_run.id,
                "optimization_run_id": db_run.optimization_run_id,
                "solver_status": db_run.solver_status,
                "objective_type": db_run.objective_type,
                "recommended_cost": float(db_run.recommended_cost) if db_run.recommended_cost else 0.0,
            }

    def get_recommendations(self, run_id: str) -> Dict[str, Any]:
        try:
            r = requests.get(f"{self.base_url}/recommendations/{run_id}", timeout=10)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        with self._get_db() as db:
            from src.database.models import OptimizationRun
            from src.recommendation_engine.recommendation_service import RecommendationService
            from sqlalchemy import select

            if run_id.isdigit():
                run = db.get(OptimizationRun, int(run_id))
            else:
                run = db.execute(select(OptimizationRun).where(OptimizationRun.optimization_run_id == run_id)).scalars().first()

            if not run:
                return {"recommendations": [], "comparison_table": []}

            rec_svc = RecommendationService()
            comp = rec_svc.generate_and_persist_recommendations(db, run)
            return comp.to_dict()

    # --- Decisions ---
    def get_decisions(self) -> List[Dict[str, Any]]:
        try:
            r = requests.get(f"{self.base_url}/decisions", timeout=3)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        with self._get_db() as db:
            from src.database.models import Decision
            decs = db.query(Decision).order_by(Decision.created_at.desc()).all()
            return [
                {
                    "id": d.id,
                    "decision_id": d.decision_id,
                    "strategy": d.recommendation.strategy_type if d.recommendation else "UNKNOWN",
                    "recommendation_code": d.recommendation.recommendation_code if d.recommendation else "N/A",
                    "decision_maker": d.decision_maker,
                    "decision_status": d.decision_status,
                    "execution_status": d.execution_status,
                    "decision_reason": d.decision_reason,
                    "selected_at": str(d.selected_at),
                    "executed_at": str(d.executed_at) if d.executed_at else "Not Executed",
                }
                for d in decs
            ]

    def record_decision(self, run_id: int, rec_id: int, maker: str, status: str, reason: Optional[str] = None) -> Dict[str, Any]:
        payload = {
            "optimization_run_id": run_id,
            "recommendation_id": rec_id,
            "decision_maker": maker,
            "decision_status": status,
            "decision_reason": reason,
            "execution_status": "NOT_STARTED",
        }
        try:
            r = requests.post(f"{self.base_url}/decisions", json=payload, timeout=5)
            if r.status_code in (200, 201):
                return r.json()
        except Exception:
            pass
        with self._get_db() as db:
            from src.decision_management.decision_service import DecisionService
            from src.decision_management.schemas import DecisionCreateRequest
            req = DecisionCreateRequest(**payload)
            d = DecisionService().record_decision(db, req)
            return {"id": d.id, "decision_id": d.decision_id, "status": d.decision_status, "execution_status": d.execution_status}

    def update_decision_execution(self, decision_id: int, new_execution_status: str) -> Dict[str, Any]:
        payload = {"execution_status": new_execution_status}
        try:
            r = requests.put(f"{self.base_url}/decisions/{decision_id}", json=payload, timeout=5)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        with self._get_db() as db:
            from src.decision_management.decision_service import DecisionService
            from src.decision_management.schemas import DecisionUpdateRequest
            req = DecisionUpdateRequest(**payload)
            d = DecisionService().update_decision(db, decision_id, req)
            return {"id": d.id, "decision_id": d.decision_id, "execution_status": d.execution_status}

    # --- Outcomes ---
    def get_outcomes(self) -> List[Dict[str, Any]]:
        try:
            r = requests.get(f"{self.base_url}/outcomes", timeout=3)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        with self._get_db() as db:
            from src.database.models import Outcome
            outs = db.query(Outcome).order_by(Outcome.created_at.desc()).all()
            return [
                {
                    "id": o.id,
                    "decision_code": o.decision.decision_id if o.decision else f"DEC-{o.decision_id}",
                    "strategy": o.decision.recommendation.strategy_type if (o.decision and o.decision.recommendation) else "UNKNOWN",
                    "actual_cost": float(o.actual_cost),
                    "estimated_cost": float(o.decision.recommendation.estimated_cost) if (o.decision and o.decision.recommendation) else 0.0,
                    "cost_variance": float(o.cost_variance),
                    "actual_delay_days": o.actual_delay_days,
                    "delay_variance": o.delay_variance,
                    "actual_service_level": o.actual_service_level,
                    "outcome_status": o.outcome_status,
                    "evaluation_date": str(o.evaluation_date),
                }
                for o in outs
            ]

    def record_outcome(self, decision_id: int, actual_cost: float, actual_delay_days: float, actual_sla: float = 1.0, notes: str = "") -> Dict[str, Any]:
        payload = {
            "decision_id": decision_id,
            "actual_cost": actual_cost,
            "actual_delay_days": actual_delay_days,
            "actual_service_level": actual_sla,
            "actual_risk": 0.08,
            "actual_quantity_delivered": 1000,
            "notes": notes,
        }
        try:
            r = requests.post(f"{self.base_url}/outcomes", json=payload, timeout=5)
            if r.status_code in (200, 201):
                return r.json()
        except Exception:
            pass
        with self._get_db() as db:
            from src.outcome_engine.outcome_service import OutcomeService
            from src.outcome_engine.schemas import OutcomeRecordRequest
            out = OutcomeService().record_outcome(db, OutcomeRecordRequest(**payload))
            return {
                "id": out.id,
                "cost_variance": float(out.cost_variance),
                "delay_variance": out.delay_variance,
                "outcome_status": out.outcome_status,
            }

    # --- Feedback ---
    def get_feedback(self) -> List[Dict[str, Any]]:
        try:
            r = requests.get(f"{self.base_url}/feedback", timeout=3)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        with self._get_db() as db:
            from src.database.models import Feedback
            fbs = db.query(Feedback).order_by(Feedback.created_at.desc()).all()
            return [
                {
                    "id": fb.id,
                    "outcome_id": fb.outcome_id,
                    "strategy": fb.outcome.decision.recommendation.strategy_type if (fb.outcome and fb.outcome.decision and fb.outcome.decision.recommendation) else "UNKNOWN",
                    "cost_error": float(fb.cost_error),
                    "delay_error": fb.delay_error,
                    "weight_adjustment": fb.weight_adjustment,
                    "learning_signal": fb.learning_signal,
                    "created_at": str(fb.created_at),
                }
                for fb in fbs
            ]

    def trigger_feedback(self, outcome_id: int) -> Dict[str, Any]:
        try:
            r = requests.post(f"{self.base_url}/feedback/{outcome_id}", timeout=5)
            if r.status_code in (200, 201):
                return r.json()
        except Exception:
            pass
        with self._get_db() as db:
            from src.feedback_engine.feedback_service import FeedbackService
            fb = FeedbackService().process_outcome_feedback(db, outcome_id)
            return {
                "id": fb.id,
                "cost_error": float(fb.cost_error),
                "learning_signal": fb.learning_signal,
            }

