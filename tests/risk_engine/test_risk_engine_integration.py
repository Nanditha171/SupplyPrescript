"""Integration tests for Risk Engine batch processing, database persistence, and reproducibility."""

from datetime import date, datetime, timezone
from pathlib import Path
import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.database.base import Base
from src.database.models import Product, Supplier, SupplyEvent, Prediction, RiskAssessment
from src.risk_engine.config import load_risk_config
from src.risk_engine.persistence import persist_risk_assessments
from src.risk_engine.run_risk_engine import run_risk_engine
from src.risk_engine.schemas import RiskAssessmentResult


@pytest.fixture
def test_db_session():
    """Create in-memory SQLite database session with complete schema."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSession()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


def test_end_to_end_batch_risk_engine(tmp_path):
    """Execute complete end-to-end batch risk workflow on actual processed dataset."""
    config_path = Path("config/risk_config.yaml")
    dataset_path = Path("data/processed/supply_prescript_processed_dataset.csv")

    assert config_path.exists(), "risk_config.yaml must exist."
    assert dataset_path.exists(), "Processed dataset must exist."

    # Run engine pointing outputs to temporary directory
    summary = run_risk_engine(
        config_path=config_path,
        dataset_path=dataset_path,
        output_dir=tmp_path,
        persist_db=False,
    )

    # 1. Verify summary counts
    assert summary.total_shipments_assessed == 624
    assert summary.processed_count == 624
    assert summary.failed_count == 0

    # 2. Verify CSV exports
    assessments_csv = tmp_path / "risk_assessments.csv"
    high_risk_csv = tmp_path / "high_risk_shipments.csv"
    assert assessments_csv.exists()
    assert high_risk_csv.exists()

    df_assess = pd.read_csv(assessments_csv)
    assert len(df_assess) == 624

    required_contract_cols = [
        "event_id",
        "delay_probability",
        "delay_risk_score",
        "supplier_risk_score",
        "inventory_risk_score",
        "capacity_risk_score",
        "composite_risk_score",
        "risk_level",
        "risk_reasons",
        "model_version",
        "assessment_timestamp",
    ]
    for col in required_contract_cols:
        assert col in df_assess.columns, f"Required optimization contract column '{col}' missing."

    df_hr = pd.read_csv(high_risk_csv)
    assert len(df_hr) > 0
    assert set(df_hr["risk_level"].unique()).issubset({"HIGH", "CRITICAL"})

    # 3. Verify charts
    assert (tmp_path / "risk_distribution.png").exists()
    assert (tmp_path / "supplier_risk_comparison.png").exists()
    assert (tmp_path / "product_risk_distribution.png").exists()

    # 4. Verify markdown report
    md_report = tmp_path / "risk_assessment_summary.md"
    assert md_report.exists()
    with open(md_report, "r", encoding="utf-8") as f:
        md_text = f.read()
    assert "Executive Summary" in md_text
    assert "Optimization Engine" in md_text


def test_batch_assessment_reproducibility(tmp_path):
    """Verify that multiple consecutive executions on identical data produce identical scores."""
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"

    run_risk_engine(output_dir=out1, persist_db=False)
    run_risk_engine(output_dir=out2, persist_db=False)

    df1 = pd.read_csv(out1 / "risk_assessments.csv")
    df2 = pd.read_csv(out2 / "risk_assessments.csv")

    pd.testing.assert_series_equal(df1["composite_risk_score"], df2["composite_risk_score"])
    pd.testing.assert_series_equal(df1["risk_level"], df2["risk_level"])
    pd.testing.assert_series_equal(df1["delay_risk_score"], df2["delay_risk_score"])


def test_database_persistence_integration(test_db_session):
    """Verify persistence of RiskAssessment linked to SupplyEvent and Prediction."""
    # Seed minimal domain entities in test database
    prod = Product(
        product_code="P_TEST",
        product_name="Test Product",
        product_category="Electronics",
        unit_cost=50.0,
        criticality_level="HIGH",
    )
    sup = Supplier(
        supplier_code="SUP_TEST",
        supplier_name="Test Supplier",
        supplier_country="Taiwan",
        reliability_score=0.90,
        supplier_capacity_units=5000,
        average_lead_time_days=10.0,
        risk_level="LOW",
    )
    test_db_session.add_all([prod, sup])
    test_db_session.commit()

    event = SupplyEvent(
        event_id="EVT_PERSIST_TEST",
        product_id=prod.id,
        supplier_id=sup.id,
        event_date=date(2025, 1, 1),
        demand_units=500,
        supplier_capacity_units=1000,
        planned_lead_time_days=7.0,
        delay_occurred=False,
        delay_days=0.0,
        unit_cost=50.0,
        service_risk_score=0.1,
    )
    test_db_session.add(event)
    test_db_session.commit()

    pred = Prediction(
        event_id=event.id,
        model_name="supply_delay_classifier",
        model_version="1.0.0",
        prediction_type="DELAY",
        delay_probability=0.75,
        predicted_delay_days=0.0,
        risk_score=0.75,
        risk_level="HIGH",
    )
    test_db_session.add(pred)
    test_db_session.commit()

    assessment = RiskAssessmentResult(
        event_id="EVT_PERSIST_TEST",
        delay_probability=0.75,
        predicted_delay=True,
        delay_risk_score=0.75,
        supplier_risk_score=0.10,
        inventory_risk_score=0.40,
        capacity_risk_score=0.50,
        service_risk_score=None,
        composite_risk_score=0.58,
        risk_level="MEDIUM",
        risk_reasons=["HIGH_DELAY_PROBABILITY"],
        model_version="1.0.0",
        configuration_version="1.0.0",
        assessment_timestamp="2026-10-13T10:00:00+00:00",
    )

    count, err = persist_risk_assessments([assessment], session=test_db_session)
    assert count == 1
    assert err is None

    # Query DB to verify
    stmt = select(RiskAssessment).where(RiskAssessment.event_id == event.id)
    saved = test_db_session.execute(stmt).scalars().first()
    assert saved is not None
    assert saved.prediction_id == pred.id
    assert saved.composite_risk_score == 0.58
    assert saved.risk_level == "MEDIUM"


def test_database_persistence_unreachable_host():
    """Verify graceful handling when database connection fails without crashing."""
    from unittest.mock import patch

    # Mock SessionLocal to simulate database network failure
    with patch("src.database.connection.SessionLocal", side_effect=Exception("Connection refused")):
        count, err = persist_risk_assessments([
            RiskAssessmentResult(
                event_id="EVT_FAIL",
                delay_probability=0.5,
                predicted_delay=False,
                delay_risk_score=0.5,
                supplier_risk_score=0.5,
                inventory_risk_score=0.5,
                capacity_risk_score=0.5,
                service_risk_score=None,
                composite_risk_score=0.5,
                risk_level="MEDIUM",
                model_version="1.0.0",
                configuration_version="1.0.0",
            )
        ])
        assert count == 0
        assert err is not None
        assert "unavailable" in err.lower() or "connection refused" in err.lower()
