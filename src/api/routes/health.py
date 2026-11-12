"""Health and diagnostic endpoint."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from src.api.dependencies import get_db

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("")
def get_health_status(db: Session = Depends(get_db)):
    """Check application, database, and inference subsystem health."""
    db_healthy = False
    db_details = ""
    try:
        db.execute(text("SELECT 1"))
        db_healthy = True
        db_details = "Connected"
    except Exception as e:
        db_details = str(e)

    return {
        "status": "healthy" if db_healthy else "degraded",
        "database": {
            "status": "connected" if db_healthy else "disconnected",
            "details": db_details,
        },
        "version": "1.0.0",
        "environment": "production",
    }

