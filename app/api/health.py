"""
CARE-PULSE AI — Health and System Status API Routes
"""

import sys
import platform
from datetime import datetime
from fastapi import APIRouter
from app.config import settings
from app.services.data_loader import data_loader_service
from app.services.training_service import training_service

router = APIRouter(prefix="/health", tags=["Health & System"])


@router.get("", summary="System Health and Readiness Check")
def get_health_status():
    """Verify application uptime, dataset availability, and trained model status."""
    data_summary = data_loader_service.get_dataset_summary()
    has_dataset = data_summary.get("status") == "available"
    has_models = len(list(settings.MODELS_DIR.glob("*.joblib"))) > 0

    return {
        "status": "healthy" if (has_dataset and has_models) else "ready_for_setup",
        "app_name": settings.PROJECT_NAME,
        "version": settings.APP_VERSION,
        "timestamp": datetime.utcnow().isoformat(),
        "environment": settings.ENVIRONMENT,
        "dataset_available": has_dataset,
        "models_trained": has_models,
        "champion_model": "Calibrated XGBoost" if has_models else "None",
        "clinical_disclaimer": "Research and Portfolio Platform. Not for autonomous clinical decisions.",
    }


@router.get("/system", summary="Hardware and Environment Diagnostics")
def get_system_diagnostics():
    """Diagnostic system metrics including Python environment, platform, and directory paths."""
    return {
        "platform": platform.platform(),
        "python_version": sys.version,
        "architecture": platform.machine(),
        "base_directory": str(settings.BASE_DIR),
        "models_directory": str(settings.MODELS_DIR),
        "data_directory": str(settings.DATA_DIR),
        "database_path": str(settings.DB_PATH),
    }
