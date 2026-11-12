"""Database connection and session management for Supply Prescript."""

import os
from typing import Generator
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
import logging

logger = logging.getLogger(__name__)

"""Database connection and session management for Supply Prescript."""

import os
from pathlib import Path
from typing import Generator
from dotenv import load_dotenv
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
import logging

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def _resolve_database_url() -> str:
    """Resolve database URL from Streamlit secrets, environment, or SQLite fallback."""
    # 1. Streamlit Community Cloud secrets check
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "DATABASE_URL" in st.secrets:
            return str(st.secrets["DATABASE_URL"])
    except Exception:
        pass

    # 2. Environment variable check
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        return env_url

    # 3. Default to SQLite
    project_root = Path(__file__).resolve().parents[2]
    return f"sqlite:///{project_root / 'supply_prescript.db'}"

DATABASE_URL = _resolve_database_url()

def _create_configured_engine(db_url: str):
    """Instantiate SQLAlchemy engine with graceful connectivity verification."""
    global DATABASE_URL
    is_sqlite = db_url.startswith("sqlite")

    if not is_sqlite:
        engine_kwargs = {
            "echo": os.getenv("SQL_ECHO", "False").lower() in ("true", "1", "yes"),
            "future": True,
            "pool_size": int(os.getenv("DB_POOL_SIZE", "10")),
            "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "20")),
            "pool_pre_ping": True,
            "connect_args": {"connect_timeout": int(os.getenv("DB_CONNECT_TIMEOUT", "2"))},
        }
        try:
            pg_engine = create_engine(db_url, **engine_kwargs)
            # Test ping to ensure database is truly reachable
            with pg_engine.connect() as conn:
                pass
            return pg_engine
        except Exception as e:
            logger.warning(
                f"Configured PostgreSQL host is unreachable ({e}). "
                "Switching to embedded SQLite for autonomous execution."
            )
            project_root = Path(__file__).resolve().parents[2]
            DATABASE_URL = f"sqlite:///{project_root / 'supply_prescript.db'}"
            db_url = DATABASE_URL
            is_sqlite = True

    sqlite_engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False},
        echo=os.getenv("SQL_ECHO", "False").lower() in ("true", "1", "yes"),
        future=True,
    )

    @event.listens_for(sqlite_engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return sqlite_engine

engine = _create_configured_engine(DATABASE_URL)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    future=True,
)

def ensure_database_ready() -> None:
    """Ensure database schema and baseline data are seeded on first launch."""
    try:
        from src.database.base import Base
        from src.database.models import Product
        from database.seed import seed_database

        Base.metadata.create_all(bind=engine)
        with SessionLocal() as session:
            count = session.query(Product).count()
            if count == 0:
                logger.info("Initializing baseline operational supply chain seed data...")
                seed_database(target_engine=engine, session_factory=SessionLocal)
    except Exception as exc:
        logger.warning(f"Database readiness verification notice: {exc}")

# Verify schema readiness
ensure_database_ready()


def get_db() -> Generator[Session, None, None]:
    """Provide a transactional database session dependency.

    Yields:
        Session: An active SQLAlchemy database session.
    """
    db = SessionLocal()
    try:
        yield db
    except SQLAlchemyError as exc:
        logger.error(f"Database session error: {exc}")
        db.rollback()
        raise
    finally:
        db.close()
