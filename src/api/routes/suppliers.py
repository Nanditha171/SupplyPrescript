"""Suppliers directory endpoints."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.database.schemas import SupplierCreate, SupplierUpdate, SupplierResponse
from src.api.services.supply_chain_service import SupplyChainService

router = APIRouter(prefix="/suppliers", tags=["Suppliers"])
service = SupplyChainService()


@router.get("", response_model=List[SupplierResponse])
def get_suppliers(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """List approved suppliers."""
    return service.list_suppliers(db, skip=skip, limit=limit)


@router.get("/{supplier_id}", response_model=SupplierResponse)
def get_supplier(supplier_id: int, db: Session = Depends(get_db)):
    """Retrieve supplier by ID."""
    sup = service.get_supplier(db, supplier_id)
    if not sup:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Supplier {supplier_id} not found.")
    return sup


@router.post("", response_model=SupplierResponse, status_code=status.HTTP_201_CREATED)
def create_supplier(payload: SupplierCreate, db: Session = Depends(get_db)):
    """Register a new supplier."""
    return service.create_supplier(db, payload)


@router.put("/{supplier_id}", response_model=SupplierResponse)
def update_supplier(supplier_id: int, payload: SupplierUpdate, db: Session = Depends(get_db)):
    """Update supplier parameters."""
    updated = service.update_supplier(db, supplier_id, payload)
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Supplier {supplier_id} not found.")
    return updated

