"""Test database check constraints and uniqueness validations."""

from datetime import date
from decimal import Decimal
import pytest
from sqlalchemy.exc import IntegrityError

from src.database.models import (
    Product,
    Supplier,
    Inventory,
    Prediction,
)


def test_product_unit_cost_positive_constraint(db_session):
    bad_prod = Product(
        product_code="BAD_PROD",
        product_name="Negative Cost",
        product_category="Test",
        unit_cost=Decimal("-10.00"),  # Violates unit_cost > 0
        criticality_level="LOW",
    )
    db_session.add(bad_prod)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_supplier_reliability_constraint(db_session):
    bad_sup = Supplier(
        supplier_code="BAD_SUP",
        supplier_name="Invalid Reliability",
        supplier_country="Nowhere",
        reliability_score=1.5,  # Violates <= 1.0
        supplier_capacity_units=1000,
        average_lead_time_days=5.0,
        risk_level="LOW",
    )
    db_session.add(bad_sup)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_inventory_math_and_non_negative_constraint(db_session):
    prod = Product(
        product_code="P_INV_TEST",
        product_name="Inventory Test SKU",
        product_category="Test",
        unit_cost=Decimal("20.00"),
        criticality_level="MEDIUM",
    )
    db_session.add(prod)
    db_session.commit()

    # Negative inventory units
    bad_inv1 = Inventory(
        product_id=prod.id,
        warehouse_code="WH-1",
        inventory_units=-50,
        reserved_units=0,
        available_units=-50,
        reorder_point=10,
        safety_stock_units=5,
        inventory_date=date.today(),
    )
    db_session.add(bad_inv1)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()

    # Reserved > Total
    bad_inv2 = Inventory(
        product_id=prod.id,
        warehouse_code="WH-1",
        inventory_units=100,
        reserved_units=150,
        available_units=-50,
        reorder_point=10,
        safety_stock_units=5,
        inventory_date=date.today(),
    )
    db_session.add(bad_inv2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_unique_product_code_constraint(db_session):
    p1 = Product(
        product_code="DUP_01",
        product_name="Product 1",
        product_category="Electronics",
        unit_cost=Decimal("50.00"),
        criticality_level="HIGH",
    )
    p2 = Product(
        product_code="DUP_01",  # Duplicate code
        product_name="Product 2",
        product_category="Electronics",
        unit_cost=Decimal("60.00"),
        criticality_level="HIGH",
    )
    db_session.add(p1)
    db_session.commit()

    db_session.add(p2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()
