"""Risk Service evaluating multidimensional operational risk and persisting assessments."""

import logging
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select

from src.database.models import SupplyEvent, Prediction, RiskAssessment, Inventory
from src.database.crud import create_risk_assessment as db_create_risk_assessment
from src.database.schemas import RiskAssessmentCreate
from src.risk_engine.config import load_risk_config
from src.risk_engine.schemas import RiskAssessmentInput
from src.risk_engine.risk_scoring import assess_single_shipment

logger = logging.getLogger(__name__)


class RiskService:
    """Service layer managing operational risk assessments and persistence."""

    def __init__(self, config_path: str = "config/risk_config.yaml"):
        self.config = load_risk_config(config_path)

    def assess_risk_for_shipment(
        self,
        db: Session,
        event_id: str,
        prediction_service=None,
    ) -> RiskAssessment:
        """Evaluate operational risk dimensions for a shipment event and persist result."""
        stmt = select(SupplyEvent).where(SupplyEvent.event_id == event_id)
        event = db.execute(stmt).scalars().first()
        if not event:
            raise ValueError(f"Supply event '{event_id}' not found.")

        # Check existing assessment
        existing_stmt = select(RiskAssessment).where(RiskAssessment.event_id == event.id).order_by(RiskAssessment.created_at.desc())
        existing = db.execute(existing_stmt).scalars().first()
        if existing:
            return existing

        # Ensure prediction exists
        pred_stmt = select(Prediction).where(Prediction.event_id == event.id).order_by(Prediction.created_at.desc())
        pred = db.execute(pred_stmt).scalars().first()
        if not pred:
            if prediction_service:
                pred = prediction_service.predict_for_shipment(db, event_id)
            else:
                from src.api.services.prediction_service import PredictionService
                pred = PredictionService().predict_for_shipment(db, event_id)

        # Query on-hand inventory
        inv_stmt = select(Inventory).where(Inventory.product_id == event.product_id).order_by(Inventory.inventory_date.desc())
        inv = db.execute(inv_stmt).scalars().first()
        avail_units = float(inv.available_units) if inv else 0.0

        risk_input = RiskAssessmentInput(
            event_id=event.event_id,
            delay_probability=pred.delay_probability,
            supplier_reliability=event.supplier.reliability_score if event.supplier else 0.85,
            historical_delay_rate=0.15,
            demand_units=float(event.demand_units),
            inventory_units=avail_units,
            supplier_capacity_units=float(event.supplier_capacity_units),
            product_criticality=event.product.criticality_level if event.product else "HIGH",
        )

        res = assess_single_shipment(risk_input, self.config, model_version=pred.model_version)

        reasons_str = "; ".join(res.risk_reason_descriptions) if res.risk_reason_descriptions else "Normal operational variance"

        rec_create = RiskAssessmentCreate(
            event_id=event.id,
            prediction_id=pred.id,
            delay_risk_score=round(res.delay_risk_score, 4),
            supplier_risk_score=round(res.supplier_risk_score, 4),
            inventory_risk_score=round(res.inventory_risk_score, 4),
            capacity_risk_score=round(res.capacity_risk_score, 4),
            service_risk_score=round(res.service_risk_score, 4) if res.service_risk_score else None,
            composite_risk_score=round(res.composite_risk_score, 4),
            risk_level=res.risk_level,
            risk_reasons=reasons_str,
            configuration_version=self.config.version,
            assessment_timestamp=datetime.now(timezone.utc),
        )

        db_risk = db_create_risk_assessment(db, rec_create)
        return db_risk

    def get_risk_assessment_by_event(self, db: Session, event_id: str) -> Optional[RiskAssessment]:
        stmt = (
            select(RiskAssessment)
            .join(SupplyEvent, RiskAssessment.event_id == SupplyEvent.id)
            .where(SupplyEvent.event_id == event_id)
            .order_by(RiskAssessment.created_at.desc())
        )
        return db.execute(stmt).scalars().first()

    def list_risk_assessments(self, db: Session, skip: int = 0, limit: int = 100) -> List[RiskAssessment]:
        stmt = select(RiskAssessment).order_by(RiskAssessment.created_at.desc()).offset(skip).limit(limit)
        return list(db.execute(stmt).scalars().all())
