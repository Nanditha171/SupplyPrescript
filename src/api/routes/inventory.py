"""Inventory and warehouse stock endpoints."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.database.schemas import InventoryCreate, InventoryUpdate, InventoryResponse
from src.api.services.supply_chain_service import SupplyChainService

router = APIRouter(prefix="/inventory", tags=["Inventory"])
service = SupplyChainService()


@router.get("", response_model=List[InventoryResponse])
def get_inventory(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """List inventory snapshots across warehouses."""
    return service.list_inventory(db, skip=skip, limit=limit)


@router.get("/{product_id}", response_model=List[InventoryResponse])
def get_inventory_by_product(product_id: int, db: Session = Depends(get_db)):
    """Retrieve warehouse stock levels for a specific product ID."""
    return service.get_inventory_by_product(db, product_id)


@router.post("", response_model=InventoryResponse, status_code=status.HTTP_201_CREATED)
def create_inventory(payload: InventoryCreate, db: Session = Depends(get_db)):
    """Record an inventory stock snapshot."""
    return service.create_inventory(db, payload)


@router.put("/{inventory_id}", response_model=InventoryResponse)
def update_inventory(inventory_id: int, payload: InventoryUpdate, db: Session = Depends(get_db)):
    """Update stock units or reservations."""
    updated = service.update_inventory(db, inventory_id, payload)
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Inventory record {inventory_id} not found.")
    return updated

