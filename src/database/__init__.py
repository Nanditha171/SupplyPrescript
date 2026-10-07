"""Supply Prescript Database Layer.

Provides SQLAlchemy models, Pydantic schemas, database connections, and CRUD utilities
for closed-loop prescriptive analytics.
"""

from src.database.base import Base
from src.database.connection import engine, SessionLocal, get_db
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

__all__ = [
    "Base",
    "engine",
    "SessionLocal",
    "get_db",
    "Product",
    "Supplier",
    "Inventory",
    "SupplyEvent",
    "Prediction",
    "OptimizationRun",
    "Recommendation",
    "Decision",
    "Outcome",
    "Feedback",
]
