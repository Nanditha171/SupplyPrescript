"""CRUD (Create, Read, Update, Delete) Operations for Supply Prescript.

Provides modular database access functions cleanly decoupled from API endpoints.
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session, selectinload
from sqlalchemy import select, update, delete

from src.database.models import (
    Product,
    Supplier,
    Inventory,
    SupplyEvent,
    Prediction,
    OptimizationRun,
    Recommendation,
    Decision,
    Outcome,
    Feedback,
)
from src.database.schemas import (
    ProductCreate,
    ProductUpdate,
    SupplierCreate,
    SupplierUpdate,
    InventoryCreate,
    InventoryUpdate,
    SupplyEventCreate,
    PredictionCreate,
    OptimizationRunCreate,
    RecommendationCreate,
    DecisionCreate,
    DecisionStatusUpdate,
    OutcomeCreate,
    FeedbackCreate,
)


# ==========================================
# 1. Product CRUD
# ==========================================

def create_product(db: Session, product: ProductCreate) -> Product:
    db_obj = Product(**product.model_dump())
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj


def get_product(db: Session, product_id: int) -> Optional[Product]:
    return db.get(Product, product_id)


def get_product_by_code(db: Session, product_code: str) -> Optional[Product]:
    stmt = select(Product).where(Product.product_code == product_code)
    return db.execute(stmt).scalars().first()


def get_products(db: Session, skip: int = 0, limit: int = 100) -> List[Product]:
    stmt = select(Product).offset(skip).limit(limit)
    return list(db.execute(stmt).scalars().all())


def update_product(db: Session, product_id: int, product_update: ProductUpdate) -> Optional[Product]:
    db_obj = db.get(Product, product_id)
    if not db_obj:
        return None
    update_data = product_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_obj, key, value)
    db.commit()
    db.refresh(db_obj)
    return db_obj


def delete_product(db: Session, product_id: int) -> bool:
    db_obj = db.get(Product, product_id)
    if not db_obj:
        return False
    db.delete(db_obj)
    db.commit()
    return True


# ==========================================
# 2. Supplier CRUD
# ==========================================

def create_supplier(db: Session, supplier: SupplierCreate) -> Supplier:
    db_obj = Supplier(**supplier.model_dump())
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj


def get_supplier(db: Session, supplier_id: int) -> Optional[Supplier]:
    return db.get(Supplier, supplier_id)


def get_supplier_by_code(db: Session, supplier_code: str) -> Optional[Supplier]:
    stmt = select(Supplier).where(Supplier.supplier_code == supplier_code)
    return db.execute(stmt).scalars().first()


def get_suppliers(db: Session, skip: int = 0, limit: int = 100) -> List[Supplier]:
    stmt = select(Supplier).offset(skip).limit(limit)
    return list(db.execute(stmt).scalars().all())


def update_supplier(db: Session, supplier_id: int, supplier_update: SupplierUpdate) -> Optional[Supplier]:
    db_obj = db.get(Supplier, supplier_id)
    if not db_obj:
        return None
    update_data = supplier_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_obj, key, value)
    db.commit()
    db.refresh(db_obj)
    return db_obj


def delete_supplier(db: Session, supplier_id: int) -> bool:
    db_obj = db.get(Supplier, supplier_id)
    if not db_obj:
        return False
    db.delete(db_obj)
    db.commit()
    return True


# ==========================================
# 3. Inventory CRUD
# ==========================================

def create_inventory(db: Session, inventory: InventoryCreate) -> Inventory:
    db_obj = Inventory(**inventory.model_dump())
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj


def get_inventory(db: Session, inventory_id: int) -> Optional[Inventory]:
    return db.get(Inventory, inventory_id)


def get_product_inventory(db: Session, product_id: int) -> List[Inventory]:
    stmt = select(Inventory).where(Inventory.product_id == product_id).order_by(Inventory.inventory_date.desc())
    return list(db.execute(stmt).scalars().all())


def update_inventory(db: Session, inventory_id: int, inventory_update: InventoryUpdate) -> Optional[Inventory]:
    db_obj = db.get(Inventory, inventory_id)
    if not db_obj:
        return None
    update_data = inventory_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_obj, key, value)
    db.commit()
    db.refresh(db_obj)
    return db_obj


# ==========================================
# 4. Supply Events CRUD
# ==========================================

def create_supply_event(db: Session, event: SupplyEventCreate) -> SupplyEvent:
    db_obj = SupplyEvent(**event.model_dump())
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj


def get_supply_event(db: Session, event_id: int) -> Optional[SupplyEvent]:
    return db.get(SupplyEvent, event_id)


def get_supply_event_by_code(db: Session, event_code: str) -> Optional[SupplyEvent]:
    stmt = select(SupplyEvent).where(SupplyEvent.event_id == event_code)
    return db.execute(stmt).scalars().first()


def get_supply_events(db: Session, skip: int = 0, limit: int = 100) -> List[SupplyEvent]:
    stmt = select(SupplyEvent).order_by(SupplyEvent.event_date.desc()).offset(skip).limit(limit)
    return list(db.execute(stmt).scalars().all())


# ==========================================
# 5. Predictions CRUD
# ==========================================

def create_prediction(db: Session, prediction: PredictionCreate) -> Prediction:
    db_obj = Prediction(**prediction.model_dump())
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj


def get_prediction(db: Session, prediction_id: int) -> Optional[Prediction]:
    return db.get(Prediction, prediction_id)


def get_predictions_for_event(db: Session, event_id: int) -> List[Prediction]:
    stmt = (
        select(Prediction)
        .where(Prediction.event_id == event_id)
        .order_by(Prediction.prediction_timestamp.desc())
    )
    return list(db.execute(stmt).scalars().all())


# ==========================================
# 6. Optimization Runs & Recommendations CRUD
# ==========================================

def create_optimization_run(db: Session, run: OptimizationRunCreate) -> OptimizationRun:
    db_obj = OptimizationRun(**run.model_dump())
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj


def get_optimization_run(db: Session, run_id: int) -> Optional[OptimizationRun]:
    return db.get(OptimizationRun, run_id)


def get_optimization_run_by_code(db: Session, optimization_run_id: str) -> Optional[OptimizationRun]:
    stmt = (
        select(OptimizationRun)
        .options(selectinload(OptimizationRun.recommendations))
        .where(OptimizationRun.optimization_run_id == optimization_run_id)
    )
    return db.execute(stmt).scalars().first()


def create_recommendation(db: Session, recommendation: RecommendationCreate) -> Recommendation:
    db_obj = Recommendation(**recommendation.model_dump())
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj


def get_recommendations(db: Session, optimization_run_id: int) -> List[Recommendation]:
    stmt = (
        select(Recommendation)
        .where(Recommendation.optimization_run_id == optimization_run_id)
        .order_by(Recommendation.rank.asc())
    )
    return list(db.execute(stmt).scalars().all())


# ==========================================
# 7. Decisions CRUD
# ==========================================

def create_decision(db: Session, decision: DecisionCreate) -> Decision:
    db_obj = Decision(**decision.model_dump())
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj


def get_decision(db: Session, decision_id: int) -> Optional[Decision]:
    return db.get(Decision, decision_id)


def get_decision_by_code(db: Session, decision_code: str) -> Optional[Decision]:
    stmt = select(Decision).where(Decision.decision_id == decision_code)
    return db.execute(stmt).scalars().first()


def update_decision_status(
    db: Session, decision_id: int, status_update: DecisionStatusUpdate
) -> Optional[Decision]:
    db_obj = db.get(Decision, decision_id)
    if not db_obj:
        return None
    update_data = status_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_obj, key, value)
    db.commit()
    db.refresh(db_obj)
    return db_obj


# ==========================================
# 8. Outcomes CRUD
# ==========================================

def create_outcome(db: Session, outcome: OutcomeCreate) -> Outcome:
    db_obj = Outcome(**outcome.model_dump())
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj


def get_outcome(db: Session, outcome_id: int) -> Optional[Outcome]:
    return db.get(Outcome, outcome_id)


def get_outcome_by_decision(db: Session, decision_id: int) -> Optional[Outcome]:
    stmt = select(Outcome).where(Outcome.decision_id == decision_id)
    return db.execute(stmt).scalars().first()


# ==========================================
# 9. Feedback CRUD
# ==========================================

def create_feedback(db: Session, feedback: FeedbackCreate) -> Feedback:
    db_obj = Feedback(**feedback.model_dump())
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj


def get_feedback(db: Session, feedback_id: int) -> Optional[Feedback]:
    return db.get(Feedback, feedback_id)


def get_feedback_by_outcome(db: Session, outcome_id: int) -> List[Feedback]:
    stmt = select(Feedback).where(Feedback.outcome_id == outcome_id).order_by(Feedback.created_at.desc())
    return list(db.execute(stmt).scalars().all())


def get_feedback_by_run(db: Session, optimization_run_id: int) -> List[Feedback]:
    stmt = (
        select(Feedback)
        .where(Feedback.optimization_run_id == optimization_run_id)
        .order_by(Feedback.created_at.desc())
    )
    return list(db.execute(stmt).scalars().all())


# ==========================================
# 10. End-to-End Traceability Audit Trail
# ==========================================

def get_traceability_trail(db: Session, decision_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve the complete closed-loop audit trail for an operational decision.

    Traceability chain:
    Supply Event -> Prediction -> Optimization Run -> Recommendations -> Decision -> Outcome -> Feedback
    """
    decision = db.get(Decision, decision_id)
    if not decision:
        return None

    run = db.get(OptimizationRun, decision.optimization_run_id)
    if not run:
        return None

    prediction = db.get(Prediction, run.prediction_id)
    event = db.get(SupplyEvent, run.event_id)
    product = db.get(Product, event.product_id) if event else None
    supplier = db.get(Supplier, event.supplier_id) if event else None
    recommendations = get_recommendations(db, run.id)
    selected_rec = db.get(Recommendation, decision.recommendation_id)
    outcome = get_outcome_by_decision(db, decision.id)
    feedback = get_feedback_by_outcome(db, outcome.id) if outcome else []

    return {
        "supply_event": event,
        "product": product,
        "supplier": supplier,
        "prediction": prediction,
        "optimization_run": run,
        "recommendations": recommendations,
        "selected_recommendation": selected_rec,
        "decision": decision,
        "outcome": outcome,
        "feedback": feedback,
    }
