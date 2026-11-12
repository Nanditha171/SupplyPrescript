"""FastAPI Application Main Entry Point for Supply Prescript (Day 12)."""

import os
import logging
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import (
    health,
    products,
    suppliers,
    inventory,
    shipments,
    predictions,
    risks,
    optimization,
    recommendations,
    decisions,
    outcomes,
    feedback,
)

logger = logging.getLogger("supply_prescript_api")

app = FastAPI(
    title="Supply Prescript API",
    description=(
        "Intelligent supply-chain decision-support platform transforming predictive analytics "
        "into optimized operational decisions through the closed-loop workflow: "
        "Predict -> Assess Risk -> Optimize -> Recommend -> Decide -> Execute -> Measure -> Learn."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/api/v1/openapi.json",
)

# CORS configuration
allowed_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:8501,http://127.0.0.1:8501,http://localhost:3000,http://127.0.0.1:3000,*",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include All Routers under /api
api_prefix = "/api"
app.include_router(health.router, prefix=api_prefix)
app.include_router(products.router, prefix=api_prefix)
app.include_router(suppliers.router, prefix=api_prefix)
app.include_router(inventory.router, prefix=api_prefix)
app.include_router(shipments.router, prefix=api_prefix)
app.include_router(predictions.router, prefix=api_prefix)
app.include_router(risks.router, prefix=api_prefix)
app.include_router(optimization.router, prefix=api_prefix)
app.include_router(recommendations.router, prefix=api_prefix)
app.include_router(decisions.router, prefix=api_prefix)
app.include_router(outcomes.router, prefix=api_prefix)
app.include_router(feedback.router, prefix=api_prefix)


@app.get("/")
def root():
    """Welcome endpoint pointing to interactive documentation."""
    return {
        "name": "Supply Prescript API",
        "version": "1.0.0",
        "documentation": "/docs",
        "redoc": "/redoc",
        "workflow": "Predict -> Assess Risk -> Optimize -> Recommend -> Decide -> Execute -> Measure -> Learn",
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Structured JSON error responses for unhandled server exceptions."""
    logger.error(f"Unhandled error on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred during operation processing.",
            "detail": str(exc),
            "path": request.url.path,
        },
    )

