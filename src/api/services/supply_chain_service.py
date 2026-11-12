"""Supply chain data service for Products, Suppliers, Inventory, and Supply Events."""

from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select

from src.database.models import Product, Supplier, Inventory, SupplyEvent
from src.database import crud
from src.database.schemas import (
    ProductCreate,
    ProductUpdate,
    SupplierCreate,
    SupplierUpdate,
    InventoryCreate,
    InventoryUpdate,
    SupplyEventCreate,
)


class SupplyChainService:
    """Service layer handling core supply-chain master data and operational events."""

    # Product operations
    def list_products(self, db: Session, skip: int = 0, limit: int = 100) -> List[Product]:
        return crud.get_products(db, skip=skip, limit=limit)

    def get_product(self, db: Session, product_id: int) -> Optional[Product]:
        return crud.get_product(db, product_id)

    def create_product(self, db: Session, payload: ProductCreate) -> Product:
        return crud.create_product(db, payload)

    def update_product(self, db: Session, product_id: int, payload: ProductUpdate) -> Optional[Product]:
        return crud.update_product(db, product_id, payload)

    # Supplier operations
    def list_suppliers(self, db: Session, skip: int = 0, limit: int = 100) -> List[Supplier]:
        return crud.get_suppliers(db, skip=skip, limit=limit)

    def get_supplier(self, db: Session, supplier_id: int) -> Optional[Supplier]:
        return crud.get_supplier(db, supplier_id)

    def create_supplier(self, db: Session, payload: SupplierCreate) -> Supplier:
        return crud.create_supplier(db, payload)

    def update_supplier(self, db: Session, supplier_id: int, payload: SupplierUpdate) -> Optional[Supplier]:
        return crud.update_supplier(db, supplier_id, payload)

    # Inventory operations
    def list_inventory(self, db: Session, skip: int = 0, limit: int = 100) -> List[Inventory]:
        return crud.get_inventories(db, skip=skip, limit=limit)

    def get_inventory_by_product(self, db: Session, product_id: int) -> List[Inventory]:
        return crud.get_inventory_by_product(db, product_id)

    def create_inventory(self, db: Session, payload: InventoryCreate) -> Inventory:
        return crud.create_inventory(db, payload)

    def update_inventory(self, db: Session, inventory_id: int, payload: InventoryUpdate) -> Optional[Inventory]:
        return crud.update_inventory(db, inventory_id, payload)

    # Supply Events / Shipments
    def list_shipments(self, db: Session, skip: int = 0, limit: int = 100) -> List[SupplyEvent]:
        return crud.get_supply_events(db, skip=skip, limit=limit)

    def get_shipment_by_code(self, db: Session, event_id: str) -> Optional[SupplyEvent]:
        return crud.get_supply_event_by_code(db, event_id)

    def get_shipment_by_id(self, db: Session, shipment_id: int) -> Optional[SupplyEvent]:
        return crud.get_supply_event(db, shipment_id)

    def create_shipment(self, db: Session, payload: SupplyEventCreate) -> SupplyEvent:
        return crud.create_supply_event(db, payload)

