"""Database Persistence Adapter for Supply Prescript Risk Engine.

Persists historical risk evaluations and links them to:
- SupplyEvent (via foreign key event_id)
- Prediction (via foreign key prediction_id, when present)

Features:
- Preserves historical records without overwriting past evaluations.
- Strict transaction management with guaranteed rollback on error.
- Graceful degradation: handles unavailable database cleanly, reporting
  failure without crashing file exports or falsely claiming persistence.
"""

from datetime import datetime, timezone
import logging
from typing import List, Optional, Tuple

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import select

from src.risk_engine.schemas import RiskAssessmentResult

logger = logging.getLogger(__name__)


def persist_risk_assessments(
    assessments: List[RiskAssessmentResult],
    session=None,
) -> Tuple[int, Optional[str]]:
    """Persist a batch of RiskAssessmentResult records to the database.

    Args:
        assessments: List of RiskAssessmentResult models.
        session: Optional active SQLAlchemy session (e.g. for testing). If None,
                 creates a session using SessionLocal.

    Returns:
        Tuple of (persisted_records_count, error_message_if_any).
    """
    if not assessments:
        return 0, None

    # Safe lazy import of database modules
    try:
        from src.database.connection import SessionLocal
        from src.database.models import RiskAssessment, SupplyEvent, Prediction
    except ImportError as exc:
        msg = f"Database modules could not be imported: {exc}"
        logger.warning(msg)
        return 0, msg

    created_own_session = False
    db = session
    if db is None:
        try:
            db = SessionLocal()
            created_own_session = True
            # Test connectivity
            db.execute(select(1))
        except Exception as conn_err:
            msg = f"Database connection unavailable: {conn_err}"
            logger.warning(msg)
            if db:
                try:
                    db.close()
                except Exception:
                    pass
            return 0, msg

    persisted_count = 0
    try:
        for assessment in assessments:
            event_code = assessment.event_id

            # Find matching SupplyEvent in DB
            stmt = select(SupplyEvent).where(SupplyEvent.event_id == event_code)
            event_obj = db.execute(stmt).scalars().first()

            if not event_obj:
                logger.debug(
                    f"SupplyEvent with code '{event_code}' not found in database; skipping DB record."
                )
                continue

            # Find latest matching prediction for this event if exists
            pred_stmt = (
                select(Prediction)
                .where(Prediction.event_id == event_obj.id)
                .order_by(Prediction.prediction_timestamp.desc())
            )
            prediction_obj = db.execute(pred_stmt).scalars().first()
            prediction_db_id = prediction_obj.id if prediction_obj else None

            # Parse timestamp safely
            try:
                ts = datetime.fromisoformat(assessment.assessment_timestamp)
            except Exception:
                ts = datetime.now(timezone.utc)

            risk_record = RiskAssessment(
                event_id=event_obj.id,
                prediction_id=prediction_db_id,
                delay_risk_score=float(assessment.delay_risk_score),
                supplier_risk_score=float(assessment.supplier_risk_score),
                inventory_risk_score=float(assessment.inventory_risk_score),
                capacity_risk_score=float(assessment.capacity_risk_score),
                service_risk_score=(
                    float(assessment.service_risk_score)
                    if assessment.service_risk_score is not None
                    else None
                ),
                composite_risk_score=float(assessment.composite_risk_score),
                risk_level=assessment.risk_level.upper(),
                risk_reasons=",".join(assessment.risk_reasons),
                configuration_version=assessment.configuration_version,
                assessment_timestamp=ts,
            )
            db.add(risk_record)
            persisted_count += 1

        db.commit()
        logger.info(
            f"Successfully persisted {persisted_count} historical risk assessments to database."
        )
        return persisted_count, None

    except SQLAlchemyError as err:
        db.rollback()
        err_msg = f"Database persistence transaction failed: {err}"
        logger.error(err_msg)
        return 0, err_msg
    except Exception as err:
        db.rollback()
        err_msg = f"Unexpected error during persistence: {err}"
        logger.error(err_msg)
        return 0, err_msg
    finally:
        if created_own_session and db:
            db.close()
