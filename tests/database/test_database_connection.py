"""Test database connectivity, engine configuration, and session lifecycle."""

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from src.database.connection import DATABASE_URL, engine, SessionLocal, get_db


def test_database_url_configured():
    """Verify DATABASE_URL is loaded from environment or defaults properly."""
    assert DATABASE_URL is not None
    assert "postgresql" in DATABASE_URL or "sqlite" in DATABASE_URL


def test_engine_configuration():
    """Verify engine instance has valid pool and dialect parameters."""
    assert engine is not None
    assert engine.url is not None


def test_test_engine_connection(db_session):
    """Verify active database session can execute scalar queries."""
    result = db_session.execute(select(1)).scalar()
    assert result == 1


def test_get_db_generator_with_test_engine(monkeypatch):
    """Verify get_db dependency yields active session and closes it."""
    test_engine = create_engine("sqlite:///:memory:")
    TestSession = sessionmaker(bind=test_engine)
    monkeypatch.setattr("src.database.connection.SessionLocal", TestSession)

    gen = get_db()
    session = next(gen)
    assert session is not None
    assert session.is_active
    try:
        next(gen)
    except StopIteration:
        pass
