from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String

from src.database.connection import Base


class ShipmentPrediction(Base):
    __tablename__ = "shipment_predictions"

    id = Column(Integer, primary_key=True, index=True)

    supplier_id = Column(String(100), nullable=False)
    product_id = Column(String(100), nullable=False)
    category = Column(String(100), nullable=False)

    demand_units = Column(Float, nullable=False)
    inventory_units = Column(Float, nullable=False)
    planned_lead_time_days = Column(Float, nullable=False)

    prediction = Column(String(20), nullable=False)
    delay_probability = Column(Float, nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )