"""Supply events and shipments endpoints."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.database.schemas import SupplyEventCreate, SupplyEventResponse
from src.api.services.supply_chain_service import SupplyChainService

router = APIRouter(prefix="/shipments", tags=["Shipments"])
service = SupplyChainService()


@router.get("", response_model=List[SupplyEventResponse])
def get_shipments(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """List operational supply events and shipments."""
    return service.list_shipments(db, skip=skip, limit=limit)


@router.get("/{event_id}", response_model=SupplyEventResponse)
def get_shipment(event_id: str, db: Session = Depends(get_db)):
    """Retrieve supply event by unique event_id (e.g., 'EVT-001') or database ID."""
    if event_id.isdigit():
        shipment = service.get_shipment_by_id(db, int(event_id))
    else:
        shipment = service.get_shipment_by_code(db, event_id)

    if not shipment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Shipment '{event_id}' not found.")
    return shipment


@router.post("", response_model=SupplyEventResponse, status_code=status.HTTP_201_CREATED)
def create_shipment(payload: SupplyEventCreate, db: Session = Depends(get_db)):
    """Create a new supply chain shipment event."""
    return service.create_shipment(db, payload)

