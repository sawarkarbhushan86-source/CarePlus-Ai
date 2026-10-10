"""
CARE-PULSE AI — FastAPI Application Entrypoint
Explainable Healthcare Intelligence, Patient Readmission Risk Prediction,
and Hospital Operations Analytics Platform.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api import api_router
from app.services.data_loader import data_loader_service
from app.services.training_service import training_service

# Logging setup
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
)
logger = logging.getLogger("care_pulse_ai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context for startup verification and warm-up."""
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.APP_VERSION} [{settings.ENVIRONMENT}]")

    # Check dataset availability
    data_summary = data_loader_service.get_dataset_summary()
    if data_summary.get("status") == "available":
        logger.info(f"Healthcare dataset detected. SHA256: {data_summary.get('fingerprint')[:16]}...")
    else:
        logger.warning("Dataset not found. Run 'python scripts/download_dataset.py' to acquire.")

    # Check model artifacts
    benchmark = training_service.get_benchmark_summary()
    if benchmark:
        logger.info(f"Champion model ready: {benchmark.best_model_name} (ROC-AUC: {benchmark.best_roc_auc:.4f})")
    else:
        logger.info("No prior benchmark cache found. Models can be trained via /api/v1/models/train or CLI.")

    yield
    logger.info(f"Shutting down {settings.PROJECT_NAME}...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "Explainable Healthcare Intelligence, Patient Readmission Risk Prediction, "
        "and Hospital Analytics Platform based on historical 130-US Hospitals cohort data (1999-2008).\n\n"
        "**Clinical Disclaimer:** Research & portfolio demonstration platform only. Not a cleared medical device."
    ),
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API v1 Router
app.include_router(api_router)


@app.get("/", summary="Root Welcome and Platform Metadata", tags=["Root"])
def root_endpoint():
    """Welcome endpoint with project overview and API documentation links."""
    return {
        "project": settings.PROJECT_NAME,
        "description": "Explainable Healthcare Intelligence & 30-Day Readmission Risk Platform",
        "version": settings.APP_VERSION,
        "documentation": "/docs",
        "health_check": "/api/v1/health",
        "benchmark_summary": "/api/v1/models/benchmark",
        "disclaimer": "Research and portfolio demonstration only. Not for autonomous clinical decisions.",
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler returning structured JSON error details."""
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc),
            "path": request.url.path,
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)
