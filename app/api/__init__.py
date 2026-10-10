"""CARE-PULSE AI API Routers."""

from fastapi import APIRouter
from app.api.health import router as health_router
from app.api.dataset_routes import router as dataset_router
from app.api.analytics_routes import router as analytics_router
from app.api.model_routes import router as model_router
from app.api.explanation_routes import router as explanation_router
from app.api.report_routes import router as report_router

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(health_router)
api_router.include_router(dataset_router)
api_router.include_router(analytics_router)
api_router.include_router(model_router)
api_router.include_router(explanation_router)
api_router.include_router(report_router)

__all__ = ["api_router"]
