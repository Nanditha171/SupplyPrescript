"""Database connection and session management for Supply Prescript."""

import os
from typing import Generator
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
import logging

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Read DATABASE_URL from environment
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5432/supply_prescript"
)

# Engine configuration arguments
engine_kwargs = {
    "echo": os.getenv("SQL_ECHO", "False").lower() in ("true", "1", "yes"),
    "future": True,
}

if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    engine_kwargs["pool_size"] = int(os.getenv("DB_POOL_SIZE", "10"))
    engine_kwargs["max_overflow"] = int(os.getenv("DB_MAX_OVERFLOW", "20"))
    engine_kwargs["pool_pre_ping"] = True
    # Fast connect timeout so unavailable host fails fast instead of hanging
    engine_kwargs["connect_args"] = {"connect_timeout": int(os.getenv("DB_CONNECT_TIMEOUT", "3"))}

engine = create_engine(DATABASE_URL, **engine_kwargs)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    future=True,
)


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
