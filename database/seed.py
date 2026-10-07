"""Database Seeder Script for Supply Prescript.

Populates the database with realistic supply-chain operational data, including the
Microchip X1 closed-loop demonstration scenario.
"""

import sys
import os
from decimal import Decimal
from datetime import date, datetime, timezone
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.database.connection import engine as primary_engine, SessionLocal as primary_session_factory
from src.database.base import Base
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


def get_active_seeding_engine_and_session():
    """Attempt connection to configured engine; fallback to local SQLite if host unavailable."""
    try:
        with primary_engine.connect() as conn:
            conn.execute(Base.metadata.tables["products"].select().limit(1))
        return primary_engine, primary_session_factory
    except Exception:
        seed_engine = create_engine("sqlite:///supply_prescript_seed.db")

        @event.listens_for(seed_engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        SeedSession = sessionmaker(autocommit=False, autoflush=False, bind=seed_engine)
        return seed_engine, SeedSession


def seed_database():
    print("============================================================")
    print("      SUPPLY PRESCRIPT - DATABASE SEEDER                    ")
    print("============================================================")

    engine, session_factory = get_active_seeding_engine_and_session()
    Base.metadata.create_all(bind=engine)
    db = session_factory()

    try:
        # Check if already seeded
        existing_products = db.query(Product).count()
        if existing_products >= 8:
            print(f"[INFO] Database already contains {existing_products} products. Skipping duplicate seed.")
            return

        print("Seeding Products (8 products)...")
        products_data = [
            Product(product_code="P001", product_name="Microchip X1 High-Density MCU", product_category="Electronics", unit_cost=Decimal("125.50"), criticality_level="CRITICAL"),
            Product(product_code="P002", product_name="Automotive Lithium Cell Module", product_category="Energy Storage", unit_cost=Decimal("85.00"), criticality_level="CRITICAL"),
            Product(product_code="P003", product_name="Precision Aluminum Enclosure", product_category="Mechanical Hardware", unit_cost=Decimal("34.20"), criticality_level="HIGH"),
            Product(product_code="P004", product_name="Industrial Optic Sensor Module", product_category="Sensors", unit_cost=Decimal("48.75"), criticality_level="HIGH"),
            Product(product_code="P005", product_name="High-Temp Copper Wiring Harness", product_category="Electrical Cables", unit_cost=Decimal("18.50"), criticality_level="MEDIUM"),
            Product(product_code="P006", product_name="Silicone Thermal Interface Pad", product_category="Thermal Materials", unit_cost=Decimal("4.80"), criticality_level="LOW"),
            Product(product_code="P007", product_name="Titanium Fastener Fast-Pack 100", product_category="Fasteners", unit_cost=Decimal("12.30"), criticality_level="LOW"),
            Product(product_code="P008", product_name="Heavy-Duty Actuator Motor 24V", product_category="Electromechanical", unit_cost=Decimal("210.00"), criticality_level="HIGH"),
        ]
        db.add_all(products_data)
        db.commit()

        print("Seeding Suppliers (8 suppliers)...")
        suppliers_data = [
            Supplier(supplier_code="SUP-001", supplier_name="Pacific Silicon Semiconductor Ltd", supplier_country="Taiwan", reliability_score=0.88, supplier_capacity_units=15000, average_lead_time_days=14.0, risk_level="HIGH"),
            Supplier(supplier_code="SUP-002", supplier_name="Shenzhen Powercell Battery Corp", supplier_country="China", reliability_score=0.91, supplier_capacity_units=25000, average_lead_time_days=18.0, risk_level="MEDIUM"),
            Supplier(supplier_code="SUP-003", supplier_name="Kyoto Precision Dynamics", supplier_country="Japan", reliability_score=0.98, supplier_capacity_units=8000, average_lead_time_days=7.0, risk_level="LOW"),
            Supplier(supplier_code="SUP-004", supplier_name="EuroSensors AG", supplier_country="Germany", reliability_score=0.95, supplier_capacity_units=12000, average_lead_time_days=10.0, risk_level="LOW"),
            Supplier(supplier_code="SUP-005", supplier_name="Austin Advanced Components LLC", supplier_country="USA", reliability_score=0.94, supplier_capacity_units=20000, average_lead_time_days=5.0, risk_level="LOW"),
            Supplier(supplier_code="SUP-006", supplier_name="Vanguard Thermal Solutions", supplier_country="South Korea", reliability_score=0.85, supplier_capacity_units=30000, average_lead_time_days=15.0, risk_level="MEDIUM"),
            Supplier(supplier_code="SUP-007", supplier_name="Alpine Fastener & Tooling SA", supplier_country="Switzerland", reliability_score=0.97, supplier_capacity_units=50000, average_lead_time_days=6.0, risk_level="LOW"),
            Supplier(supplier_code="SUP-008", supplier_name="Veritas Motion Technologies", supplier_country="Canada", reliability_score=0.79, supplier_capacity_units=6000, average_lead_time_days=21.0, risk_level="CRITICAL"),
        ]
        db.add_all(suppliers_data)
        db.commit()

        prod_map = {p.product_code: p.id for p in db.query(Product).all()}
        sup_map = {s.supplier_code: s.id for s in db.query(Supplier).all()}

        print("Seeding Inventory Snapshots...")
        inventory_data = [
            Inventory(product_id=prod_map["P001"], warehouse_code="WH-NORTH-01", inventory_units=2500, reserved_units=500, available_units=2000, reorder_point=1200, safety_stock_units=600, inventory_date=date(2026, 10, 1)),
            Inventory(product_id=prod_map["P001"], warehouse_code="WH-SOUTH-02", inventory_units=1800, reserved_units=300, available_units=1500, reorder_point=1000, safety_stock_units=500, inventory_date=date(2026, 10, 1)),
            Inventory(product_id=prod_map["P002"], warehouse_code="WH-NORTH-01", inventory_units=4000, reserved_units=1000, available_units=3000, reorder_point=2000, safety_stock_units=1000, inventory_date=date(2026, 10, 1)),
            Inventory(product_id=prod_map["P003"], warehouse_code="WH-CENTRAL-01", inventory_units=6000, reserved_units=1200, available_units=4800, reorder_point=2500, safety_stock_units=1200, inventory_date=date(2026, 10, 1)),
            Inventory(product_id=prod_map["P004"], warehouse_code="WH-NORTH-01", inventory_units=3200, reserved_units=600, available_units=2600, reorder_point=1500, safety_stock_units=700, inventory_date=date(2026, 10, 1)),
            Inventory(product_id=prod_map["P005"], warehouse_code="WH-CENTRAL-01", inventory_units=8500, reserved_units=1500, available_units=7000, reorder_point=3000, safety_stock_units=1500, inventory_date=date(2026, 10, 1)),
            Inventory(product_id=prod_map["P006"], warehouse_code="WH-SOUTH-02", inventory_units=15000, reserved_units=2000, available_units=13000, reorder_point=5000, safety_stock_units=2500, inventory_date=date(2026, 10, 1)),
            Inventory(product_id=prod_map["P007"], warehouse_code="WH-CENTRAL-01", inventory_units=22000, reserved_units=3000, available_units=19000, reorder_point=8000, safety_stock_units=4000, inventory_date=date(2026, 10, 1)),
            Inventory(product_id=prod_map["P008"], warehouse_code="WH-NORTH-01", inventory_units=1200, reserved_units=400, available_units=800, reorder_point=700, safety_stock_units=350, inventory_date=date(2026, 10, 1)),
        ]
        db.add_all(inventory_data)
        db.commit()

        print("Seeding Supply Events (20 events)...")
        events_spec = [
            ("EVT-2026-001", "P001", "SUP-001", date(2026, 9, 15), 3000, 15000, 14.0, 16.0, True, 2.0, Decimal("4500.00"), Decimal("125.50"), Decimal("12550.00"), Decimal("15000.00"), 0.82),
            ("EVT-2026-002", "P002", "SUP-002", date(2026, 9, 16), 5000, 25000, 18.0, 18.0, False, 0.0, Decimal("6200.00"), Decimal("85.00"), Decimal("4250.00"), Decimal("21000.00"), 0.35),
            ("EVT-2026-003", "P003", "SUP-003", date(2026, 9, 18), 2000, 8000, 7.0, 7.0, False, 0.0, Decimal("1800.00"), Decimal("34.20"), Decimal("2000.00"), Decimal("8500.00"), 0.12),
            ("EVT-2026-004", "P004", "SUP-004", date(2026, 9, 20), 1500, 12000, 10.0, 11.0, True, 1.0, Decimal("2100.00"), Decimal("48.75"), Decimal("3600.00"), Decimal("9200.00"), 0.28),
            ("EVT-2026-005", "P005", "SUP-005", date(2026, 9, 22), 4000, 20000, 5.0, 5.0, False, 0.0, Decimal("1400.00"), Decimal("18.50"), Decimal("1800.00"), Decimal("6000.00"), 0.15),
            ("EVT-2026-006", "P006", "SUP-006", date(2026, 9, 25), 10000, 30000, 15.0, 20.0, True, 5.0, Decimal("3200.00"), Decimal("4.80"), Decimal("2400.00"), Decimal("11000.00"), 0.65),
            ("EVT-2026-007", "P007", "SUP-007", date(2026, 9, 26), 15000, 50000, 6.0, 6.0, False, 0.0, Decimal("2200.00"), Decimal("12.30"), Decimal("3000.00"), Decimal("7500.00"), 0.08),
            ("EVT-2026-008", "P008", "SUP-008", date(2026, 9, 28), 800, 6000, 21.0, 33.0, True, 12.0, Decimal("5800.00"), Decimal("210.00"), Decimal("16800.00"), Decimal("28000.00"), 0.92),
            ("EVT-2026-009", "P001", "SUP-001", date(2026, 9, 29), 2500, 15000, 14.0, 15.0, True, 1.0, Decimal("4200.00"), Decimal("125.50"), Decimal("10000.00"), Decimal("14000.00"), 0.74),
            ("EVT-2026-010", "P002", "SUP-002", date(2026, 10, 1), 4500, 25000, 18.0, 18.0, False, 0.0, Decimal("5900.00"), Decimal("85.00"), Decimal("3800.00"), Decimal("19500.00"), 0.25),
            ("EVT-2026-011", "P003", "SUP-005", date(2026, 10, 2), 3000, 20000, 5.0, 5.0, False, 0.0, Decimal("1900.00"), Decimal("34.20"), Decimal("2500.00"), Decimal("9000.00"), 0.10),
            ("EVT-2026-012", "P004", "SUP-004", date(2026, 10, 3), 2200, 12000, 10.0, 10.0, False, 0.0, Decimal("2600.00"), Decimal("48.75"), Decimal("4200.00"), Decimal("10500.00"), 0.18),
            ("EVT-2026-013", "P005", "SUP-005", date(2026, 10, 4), 6000, 20000, 5.0, 6.0, True, 1.0, Decimal("1700.00"), Decimal("18.50"), Decimal("2200.00"), Decimal("7200.00"), 0.32),
            ("EVT-2026-014", "P006", "SUP-006", date(2026, 10, 5), 8000, 30000, 15.0, 16.0, True, 1.0, Decimal("2900.00"), Decimal("4.80"), Decimal("1900.00"), Decimal("9500.00"), 0.44),
            ("EVT-2026-015", "P007", "SUP-007", date(2026, 10, 5), 12000, 50000, 6.0, 6.0, False, 0.0, Decimal("1950.00"), Decimal("12.30"), Decimal("2400.00"), Decimal("6800.00"), 0.06),
            ("EVT-2026-016", "P008", "SUP-003", date(2026, 10, 6), 600, 8000, 7.0, 7.0, False, 0.0, Decimal("3100.00"), Decimal("210.00"), Decimal("6300.00"), Decimal("18000.00"), 0.14),
            ("EVT-2026-017", "P001", "SUP-001", date(2026, 10, 6), 3500, 15000, 14.0, None, False, 0.0, Decimal("4800.00"), Decimal("125.50"), Decimal("14000.00"), Decimal("16500.00"), 0.85),
            ("EVT-2026-018", "P002", "SUP-002", date(2026, 10, 7), 5500, 25000, 18.0, None, False, 0.0, Decimal("6700.00"), Decimal("85.00"), Decimal("4700.00"), Decimal("23000.00"), 0.40),
            ("EVT-2026-019", "P003", "SUP-003", date(2026, 10, 7), 2500, 8000, 7.0, None, False, 0.0, Decimal("2050.00"), Decimal("34.20"), Decimal("2200.00"), Decimal("9200.00"), 0.11),
            ("EVT-2026-020", "P008", "SUP-008", date(2026, 10, 7), 950, 6000, 21.0, None, False, 0.0, Decimal("6400.00"), Decimal("210.00"), Decimal("19950.00"), Decimal("32000.00"), 0.89),
        ]

        supply_events_data = []
        for code, pcode, scode, ev_date, dem, cap, plt, alt, d_occ, d_days, tcost, ucost, sprem, acost, srisk in events_spec:
            event = SupplyEvent(
                event_id=code,
                product_id=prod_map[pcode],
                supplier_id=sup_map[scode],
                event_date=ev_date,
                demand_units=dem,
                supplier_capacity_units=cap,
                planned_lead_time_days=plt,
                actual_lead_time_days=alt,
                delay_occurred=d_occ,
                delay_days=d_days,
                transport_cost=tcost,
                unit_cost=ucost,
                secondary_supplier_premium=sprem,
                air_freight_cost=acost,
                service_risk_score=srisk,
            )
            supply_events_data.append(event)
        
        db.add_all(supply_events_data)
        db.commit()

        evt_map = {e.event_id: e.id for e in db.query(SupplyEvent).all()}

        print("Seeding Predictions...")
        pred1 = Prediction(
            event_id=evt_map["EVT-2026-001"],
            model_name="XGBoost Delay Classifier",
            model_version="v1.0",
            prediction_type="DELAY",
            delay_probability=0.87,
            predicted_delay_days=14.0,
            risk_score=0.82,
            risk_level="HIGH",
            prediction_timestamp=datetime(2026, 9, 15, 8, 30, tzinfo=timezone.utc),
        )
        pred2 = Prediction(
            event_id=evt_map["EVT-2026-008"],
            model_name="XGBoost Delay Classifier",
            model_version="v1.0",
            prediction_type="DELAY",
            delay_probability=0.93,
            predicted_delay_days=12.0,
            risk_score=0.91,
            risk_level="CRITICAL",
            prediction_timestamp=datetime(2026, 9, 28, 10, 15, tzinfo=timezone.utc),
        )
        db.add_all([pred1, pred2])
        db.commit()
        db.refresh(pred1)
        db.refresh(pred2)

        print("Seeding Optimization Runs...")
        opt1 = OptimizationRun(
            event_id=evt_map["EVT-2026-001"],
            prediction_id=pred1.id,
            optimization_run_id="OPT-RUN-2026-001",
            objective_type="BALANCED",
            optimization_status="COMPLETED",
            total_cost_baseline=Decimal("381000.00"),
            recommended_cost=Decimal("396000.00"),
            service_level_target=0.98,
            risk_weight=0.40,
            cost_weight=0.40,
            time_weight=0.20,
            solver_name="PULP_CBC",
            solver_status="OPTIMAL",
            execution_time_ms=142.3,
        )
        opt2 = OptimizationRun(
            event_id=evt_map["EVT-2026-008"],
            prediction_id=pred2.id,
            optimization_run_id="OPT-RUN-2026-008",
            objective_type="MIN_RISK",
            optimization_status="COMPLETED",
            total_cost_baseline=Decimal("173800.00"),
            recommended_cost=Decimal("201800.00"),
            service_level_target=0.99,
            risk_weight=0.60,
            cost_weight=0.20,
            time_weight=0.20,
            solver_name="PULP_CBC",
            solver_status="OPTIMAL",
            execution_time_ms=115.8,
        )
        db.add_all([opt1, opt2])
        db.commit()
        db.refresh(opt1)
        db.refresh(opt2)

        print("Seeding Prescriptive Recommendations...")
        recs = [
            Recommendation(optimization_run_id=opt1.id, recommendation_code="ALT_A", strategy_type="AIR_FREIGHT", strategy_description="Expedite primary supplier shipment via priority air freight charter", estimated_cost=Decimal("15000.00"), estimated_delay_days=2.0, estimated_risk=0.08, service_level=0.99, cost_difference=Decimal("15000.00"), risk_difference=-0.74, rank=1, is_recommended=True),
            Recommendation(optimization_run_id=opt1.id, recommendation_code="ALT_B", strategy_type="SECONDARY_SUPPLIER", strategy_description="Split procurement 60/40 with secondary qualified domestic supplier at 10% premium", estimated_cost=Decimal("22000.00"), estimated_delay_days=4.0, estimated_risk=0.15, service_level=0.95, cost_difference=Decimal("22000.00"), risk_difference=-0.67, rank=2, is_recommended=False),
            Recommendation(optimization_run_id=opt1.id, recommendation_code="ALT_C", strategy_type="DELAY_LAUNCH", strategy_description="Reschedule assembly launch timeline by 14 days without logistics expediting spend", estimated_cost=Decimal("0.00"), estimated_delay_days=14.0, estimated_risk=0.85, service_level=0.70, cost_difference=Decimal("0.00"), risk_difference=0.03, rank=3, is_recommended=False),
            Recommendation(optimization_run_id=opt2.id, recommendation_code="ALT_A", strategy_type="SECONDARY_SUPPLIER", strategy_description="Emergency redirect of 800 motor units to Japanese precision supplier", estimated_cost=Decimal("28000.00"), estimated_delay_days=3.0, estimated_risk=0.10, service_level=0.98, cost_difference=Decimal("28000.00"), risk_difference=-0.81, rank=1, is_recommended=True),
            Recommendation(optimization_run_id=opt2.id, recommendation_code="ALT_B", strategy_type="AIR_FREIGHT", strategy_description="Air freight remaining stock from Canadian central distribution center", estimated_cost=Decimal("32000.00"), estimated_delay_days=2.0, estimated_risk=0.12, service_level=0.97, cost_difference=Decimal("32000.00"), risk_difference=-0.79, rank=2, is_recommended=False),
        ]
        db.add_all(recs)
        db.commit()

        rec1 = db.query(Recommendation).filter_by(optimization_run_id=opt1.id, recommendation_code="ALT_A").first()
        rec2_a = db.query(Recommendation).filter_by(optimization_run_id=opt2.id, recommendation_code="ALT_A").first()

        print("Seeding Human Decisions...")
        dec1 = Decision(
            decision_id="DEC-2026-001",
            optimization_run_id=opt1.id,
            recommendation_id=rec1.id,
            decision_maker="Sarah Chen (VP Global Supply Chain)",
            decision_status="APPROVED",
            decision_reason="Selected Air Freight (ALT_A) to prevent factory line shutdown on Tier-1 automotive customer order.",
            selected_at=datetime(2026, 9, 15, 10, 30, tzinfo=timezone.utc),
            execution_status="COMPLETED",
            executed_at=datetime(2026, 9, 15, 11, 0, tzinfo=timezone.utc),
        )
        dec2 = Decision(
            decision_id="DEC-2026-008",
            optimization_run_id=opt2.id,
            recommendation_id=rec2_a.id,
            decision_maker="Marcus Vance (Procurement Lead)",
            decision_status="APPROVED",
            decision_reason="Approved secondary supplier redirect to avoid 12-day actuator bottleneck.",
            selected_at=datetime(2026, 9, 28, 13, 0, tzinfo=timezone.utc),
            execution_status="COMPLETED",
            executed_at=datetime(2026, 9, 28, 13, 30, tzinfo=timezone.utc),
        )
        db.add_all([dec1, dec2])
        db.commit()
        db.refresh(dec1)
        db.refresh(dec2)

        print("Seeding Outcomes (Demonstrating variance vs recommendation)...")
        out1 = Outcome(
            decision_id=dec1.id,
            actual_cost=Decimal("18000.00"),
            actual_delay_days=2.0,
            actual_service_level=0.99,
            actual_risk=0.05,
            actual_quantity_delivered=3000,
            cost_variance=Decimal("3000.00"),
            delay_variance=0.0,
            service_level_variance=0.0,
            outcome_status="SUCCESS",
            evaluation_date=date(2026, 9, 18),
            notes="Air freight shipment landed at regional airport on schedule; unexpected peak jet fuel surcharge added $3,000 to freight baseline.",
        )
        out2 = Outcome(
            decision_id=dec2.id,
            actual_cost=Decimal("29500.00"),
            actual_delay_days=3.5,
            actual_service_level=0.97,
            actual_risk=0.08,
            actual_quantity_delivered=800,
            cost_variance=Decimal("1500.00"),
            delay_variance=0.5,
            service_level_variance=-0.01,
            outcome_status="SUCCESS",
            evaluation_date=date(2026, 10, 2),
            notes="Secondary supplier met delivery requirements with minor handling charge increase.",
        )
        db.add_all([out1, out2])
        db.commit()
        db.refresh(out1)
        db.refresh(out2)

        print("Seeding Feedback (Closing the learning loop)...")
        fb1 = Feedback(
            outcome_id=out1.id,
            optimization_run_id=opt1.id,
            feedback_type="COST_VARIANCE",
            cost_error=Decimal("3000.00"),
            delay_error=0.0,
            risk_error=-0.03,
            prediction_error=0.0,
            weight_adjustment=0.05,
            learning_signal="Incorporate +20% jet fuel seasonal rate multiplier into future Air Freight cost parameter models for APAC corridors.",
        )
        fb2 = Feedback(
            outcome_id=out2.id,
            optimization_run_id=opt2.id,
            feedback_type="COST_VARIANCE",
            cost_error=Decimal("1500.00"),
            delay_error=0.5,
            risk_error=-0.02,
            prediction_error=0.0,
            weight_adjustment=0.02,
            learning_signal="Update handling fee calibration for Kyoto precision parts expedited customs clearance.",
        )
        db.add_all([fb1, fb2])
        db.commit()

        print("\n[SUCCESS] Seed data populated successfully!")
        print("  * Products: 8")
        print("  * Suppliers: 8")
        print("  * Inventory records: 9")
        print("  * Supply Events: 20")
        print("  * Predictions: 2")
        print("  * Optimization Runs: 2")
        print("  * Recommendations: 5")
        print("  * Decisions: 2")
        print("  * Outcomes: 2")
        print("  * Feedback records: 2")

    except Exception as exc:
        db.rollback()
        print(f"[FAIL] Seeding failed: {exc}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_database()
